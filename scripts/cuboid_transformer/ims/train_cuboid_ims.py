import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, DeviceStatsMonitor

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR
from torch.nn import functional as F
import torchmetrics

from earthformer.datasets.ims.ims_datamodule import IMSLightningDataModule
from earthformer.visualization.ims.ims_visualize import IMSVisualize
from earthformer.config import cfg
from earthformer.utils.optim import SequentialLR, warmup_lambda
from earthformer.utils.utils import get_parameter_names
from earthformer.utils.apex_ddp import ApexDDPStrategy
from earthformer.utils.ims.vgg import Vgg16
from earthformer.utils.ims.load_model import load_model
from earthformer.utils.ims.fss_loss import FSSLoss

import logging
import wandb
import warnings
from shutil import copyfile
import numpy as np
from datetime import datetime, timedelta
from omegaconf import OmegaConf
import os, sys
import argparse
import json
from pysteps.verification.spatialscores import fss_init, fss_accum, fss_compute

FIRST_VERSION_NUM = 44 # TODO: change this
pretrained_checkpoints_dir = cfg.pretrained_checkpoints_dir


class CuboidIMSModule(pl.LightningModule):
    def __init__(self,
                 args: dict = None,
                 logging_dir: str = None):
        super(CuboidIMSModule, self).__init__()

        self.args = args

        if args is None or args["cfg"] is None:
            self.cfg_file_path = os.path.join(os.path.dirname(__file__), "cfg_ims.yaml")
        else:
            self.cfg_file_path = args["cfg"]

        # save hyperparams
        train_cfg = OmegaConf.load(open(self.cfg_file_path, "r"))
        self.save_hyperparameters(train_cfg)

        # data module
        self.dm = self._get_dm()

        # load cuboid attention model
        self.cuboid_attention_model = load_model(model_cfg=self.hparams.model)
        self.vgg_model = Vgg16()
        self.fss_loss = FSSLoss(threshold=self.hparams.optim.fss.threshold,
                                scale=self.hparams.optim.fss.scale,
                                hwc=self.hparams.model.hwc,
                                seq_len=self.hparams.model.out_len,
                                pixel_scale=self.hparams.dataset.preprocess.scale,
                                device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        self.validation_loss = torchmetrics.MeanSquaredError()  # TODO: why they are different?

        # total_num_steps = (number of epochs) * (number of batches in the train data)
        self.total_num_steps = int(self.hparams.optim.max_epochs *
                                   len(self.dm.ims_train) /
                                   self.hparams.optim.total_batch_size)

        # create logging directories and set up logging
        self._init_logging(logging_dir)

    def perceptual_loss(self, y, y_hat):
        """
        the shape of y, y_hat is NTHWC.
        the calculated loss is the average of all the samples in that batch.
        """
        # TODO: the following code requires final testing
        # if grayscale, duplicate all channels 3 times
        if self.hparams.model.hwc[-1] == 1:
            y = torch.repeat_interleave(y, 3, dim=-1)
            y_hat = torch.repeat_interleave(y_hat, 3, dim=-1)

        # flatten the batch into one long sequence,
        # which can be interpreted as a batch of images
        y = y.flatten(end_dim=1)
        y_hat = y_hat.flatten(end_dim=1)

        # change dimensions to be TCHW
        y = y.permute(0, 3, 1, 2)
        y_hat = y_hat.permute(0, 3, 1, 2)

        y = getattr(self.vgg_model(y), self.hparams.optim.vgg.layer)
        y_hat = getattr(self.vgg_model(y_hat), self.hparams.optim.vgg.layer)

        loss = F.mse_loss(y, y_hat)  # computes mean over all pixels
        return loss

    def _init_logging(self, logging_dir: str = None):
        # creates logging directories and adds their path as data members
        if logging_dir is None:
            logging_dir = os.path.join(os.path.dirname(__file__), "logging")
        self.logging_dir = logging_dir
        os.makedirs(self.logging_dir, exist_ok=True)

        self.our_logs_dir = os.path.join(self.logging_dir, "our_logs")
        os.makedirs(self.our_logs_dir, exist_ok=True)

        # add a new directory for the curr version 
        max_version_num = FIRST_VERSION_NUM
        for d in os.listdir(self.our_logs_dir):
            if os.path.isdir(os.path.join(self.our_logs_dir, d)):
                if (d.split("_")[-1]).isnumeric():
                    max_version_num = max(max_version_num, int(d.split("_")[-1]))

        self.curr_version_num = max_version_num + 1
        self.curr_version_dir = os.path.join(self.our_logs_dir, f"version_{self.curr_version_num}")
        os.makedirs(self.curr_version_dir, exist_ok=True)

        self.scores_dir = os.path.join(self.curr_version_dir, "scores")
        self.examples_dir = os.path.join(self.curr_version_dir, "examples")
        self.checkpoints_dir = os.path.join(self.curr_version_dir, "checkpoints")
        os.makedirs(self.scores_dir, exist_ok=True)
        os.makedirs(self.examples_dir, exist_ok=True)
        os.makedirs(self.checkpoints_dir, exist_ok=True)

        # get visualization config
        self.visualize = IMSVisualize(save_dir=self.examples_dir,
                                      scale=self.hparams.dataset.preprocess.scale,
                                      fs=self.hparams.logging.visualize.fs,
                                      figsize=self.hparams.logging.visualize.figsize,
                                      plot_stride=self.hparams.logging.visualize.plot_stride)

        # save a copy of the current config inside the logging dir
        cfg_file_target_path = os.path.join(self.curr_version_dir, "cfg.yaml")
        if (not os.path.exists(cfg_file_target_path)) or \
                (not os.path.samefile(self.cfg_file_path, cfg_file_target_path)):
            copyfile(self.cfg_file_path, cfg_file_target_path)

        # save the command line args into json
        args_file_target_path = os.path.join(self.curr_version_dir, "args.json")
        with open(args_file_target_path, 'w') as f:
            json.dump(self.args, f, indent=2)

    def _get_x_y_from_batch(self, batch):
        # batch.shape is (times, sample) where times shape is S (list of integer timestamps)
        # and sample shape is (T, H, W, C)
        start_time, sample = batch

        return start_time, sample[:, :self.hparams.model.in_len, :, :, :], \
               sample[:, self.hparams.model.in_len:(self.hparams.model.in_len + self.hparams.model.out_len), :, :, :]

    def forward(self, x):
        return self.cuboid_attention_model(x)

    def training_step(self, batch, batch_idx):
        start_time, x, y = self._get_x_y_from_batch(batch)

        y_hat = self(x)

        mse_loss = F.mse_loss(y, y_hat)
        perceptual_loss = self.perceptual_loss(y, y_hat)
        fss_loss = self.fss_loss(target=y, output=y_hat)
        loss = self.hparams.optim.loss_coefficients.mse * mse_loss + \
               self.hparams.optim.loss_coefficients.perceptual * perceptual_loss + \
               self.hparams.optim.loss_coefficients.fss * fss_loss

        data_idx = int(batch_idx * self.hparams.optim.micro_batch_size)

        # save our visualization
        self.save_visualization(seq_start_time=start_time[0],
                                in_seq=x[0],
                                target_seq=y[0],
                                pred_seq_list=y_hat[0],
                                data_idx=data_idx,
                                mode="train")

        self.log('train_loss_step', loss, prog_bar=True, on_step=True, on_epoch=False)
        self.log('train_mse_loss_step', mse_loss, on_step=True, on_epoch=False)
        self.log('train_perceptual_loss_step', perceptual_loss, on_step=True, on_epoch=False)
        self.log('train_fss_loss_step', fss_loss, on_step=True, on_epoch=False)
        return loss

    def predict_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        # the code is taken from scripts/cuboid_transformer/sevir/train_cuboid_sevir.py
        # disable the weight decay for layer norm weights and all bias terms.
        # https://arxiv.org/pdf/2106.15739.pdf
        decay_parameters = [name for name in get_parameter_names(self.cuboid_attention_model, [torch.nn.LayerNorm]) \
                            if "bias" not in name]
        optimizer_grouped_parameters = [{
            'params': [p for n, p in self.cuboid_attention_model.named_parameters()
                       if n in decay_parameters],
            'weight_decay': self.hparams.optim.wd
        }, {
            'params': [p for n, p in self.cuboid_attention_model.named_parameters()
                       if n not in decay_parameters],
            'weight_decay': 0.0
        }]

        if self.hparams.optim.method == 'adamw':
            optimizer = AdamW(params=optimizer_grouped_parameters,
                              lr=self.hparams.optim.lr,
                              weight_decay=self.hparams.optim.wd)
        else:
            raise NotImplementedError

        warmup_iter = int(np.round(self.hparams.optim.warmup_percentage * self.total_num_steps))

        if self.hparams.optim.lr_scheduler_mode == 'cosine':
            warmup_scheduler = LambdaLR(optimizer,
                                        lr_lambda=warmup_lambda(warmup_steps=warmup_iter,
                                                                min_lr_ratio=self.hparams.optim.warmup_min_lr_ratio))
            cosine_scheduler = CosineAnnealingLR(optimizer,
                                                 T_max=(self.total_num_steps - warmup_iter),
                                                 eta_min=self.hparams.optim.min_lr_ratio * self.hparams.optim.lr)
            lr_scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler],
                                        milestones=[warmup_iter])
            lr_scheduler_config = {
                'scheduler': lr_scheduler,
                'interval': 'step',
                'frequency': 1,
            }

        else:
            raise NotImplementedError

        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler_config}

    def get_trainer_kwargs(self, gpus):
        """
        Default kwargs used when initializing pl.Trainer
        """
        # TODO: early stopping not implemented currently
        # because it depends on SEVIRSkillScore objects
        # if self.hparams.optim.early_stop:
        #     callbacks += [EarlyStopping(monitor="valid_loss_epoch",
        #                                 min_delta=0.0,
        #                                 patience=self.oc.optim.early_stop_patience,
        #                                 verbose=False,
        #                                 mode=self.oc.optim.early_stop_mode), ]

        # ModelCheckpoint allows fine-grained control over checkpointing
        checkpoint_callback = ModelCheckpoint(
            monitor="val_loss_epoch",
            filename="model-{epoch:03d}",
            save_top_k=self.hparams.optim.save_top_k,
            save_last=True,
            mode="min",
            dirpath=self.checkpoints_dir
        )

        callbacks = []
        callbacks += [checkpoint_callback, ]
        if self.hparams.logging.monitor_lr:
            callbacks += [LearningRateMonitor(logging_interval='step'), ]
        if self.hparams.logging.monitor_device:
            callbacks += [DeviceStatsMonitor(), ]

        # logging
        loggers = []
        """ 
        TensorBoardLogger - to see the metrics run the following commands on timon:
        cd leah/cloud-forecasting-transformer/
        tensorboard --logdir scripts/cuboid_transformer/ims/logging/lightning_logs --bind_all
        
        Then, go to chrome and connect http://http://192.168.0.177/:6006/.
        """
        if self.hparams.logging.use_tensorbaord:
            loggers.append(pl_loggers.TensorBoardLogger(save_dir=self.logging_dir,
                                                        version=self.curr_version_num))

        """
        CSVLogger
        """
        if self.hparams.logging.use_csv:
            loggers.append(pl_loggers.CSVLogger(save_dir=self.logging_dir,
                                                version=self.curr_version_num))

        """
        WandbLogger
        """
        if self.hparams.logging.use_wandb:
            loggers.append(pl_loggers.WandbLogger(project="cloud-forecasting-transformer",
                                                  save_dir=self.logging_dir,
                                                  name=f"version_{self.curr_version_num}"))

        trainer_kwargs = dict(
            devices=gpus,
            accumulate_grad_batches=self.hparams.optim.total_batch_size // (self.hparams.optim.micro_batch_size * gpus),
            callbacks=callbacks,
            # log
            logger=loggers,
            log_every_n_steps=max(1, int(self.hparams.trainer.log_step_ratio * self.total_num_steps)),
            track_grad_norm=self.hparams.logging.track_grad_norm,
            # save
            default_root_dir=self.logging_dir,
            # ddp
            accelerator=self.hparams.trainer.accelerator,
            strategy=ApexDDPStrategy(find_unused_parameters=False, delay_allreduce=True),
            # optimization
            max_epochs=self.hparams.optim.max_epochs,
            check_val_every_n_epoch=self.hparams.trainer.check_val_every_n_epoch,
            gradient_clip_val=self.hparams.optim.gradient_clip_val,
            # NVIDIA amp
            precision=self.hparams.trainer.precision,
        )
        return trainer_kwargs

    def validation_step(self, batch, batch_idx):
        start_time, x, y = self._get_x_y_from_batch(batch)
        y_hat = self(x)

        # TODO: verify we know what it means, seems like the first in any microbatch
        data_idx = int(
            batch_idx * self.hparams.optim.micro_batch_size)

        # save our visualization
        self.save_visualization(seq_start_time=start_time[0],
                                in_seq=x[0],
                                target_seq=y[0],
                                pred_seq_list=y_hat[0],
                                data_idx=data_idx,
                                mode="val")

        loss = self.validation_loss(y_hat, y)
        self.log('val_loss_step', loss, prog_bar=True, on_step=True, on_epoch=False)

        fss_batch = self._calc_fss_batch(y, y_hat)
        self.log('val_fss_step', fss_batch, on_step=True, on_epoch=False)

    def validation_epoch_end(self, outputs):
        epoch_loss = self.validation_loss.compute()
        self.log("val_loss_epoch", epoch_loss, sync_dist=True, on_epoch=True)
        self.validation_loss.reset()

    def _torch_to_numpy(self, e):
        return e.detach().float().cpu().numpy()

    def _calc_fss_batch(self, y, y_hat):
        """
        y and y_hat are from layout NTHWC.
        calculates accumulated fss for the whole batch.
        compares between every pair of ground truth frame and predicted frame for every sequence in the batch.
        if there are more than one channel in the frame, every one of them is compared separately.
        """
        # TODO: instead of a loop apply in vectors
        pixel_scale = 255 if self.hparams.dataset.preprocess.scale else 1
        fss = fss_init(self.hparams.trainer.fss.threshold, self.hparams.trainer.fss.scale)
        for i in range(self.hparams.optim.micro_batch_size):
            y_sample, y_hat_sample = self._torch_to_numpy(y[i]), self._torch_to_numpy(y_hat[i])
            for j in range(self.hparams.model.out_len):
                for c in range(self.hparams.model.hwc[-1]):
                    fss_accum(fss, y_sample[j, :, :, c] * pixel_scale, y_hat_sample[j, :, :, c] * pixel_scale)

        return fss_compute(fss)

    def save_visualization(
            self,
            data_idx: int,
            seq_start_time: torch.tensor,  # timestamp
            in_seq: torch.Tensor,
            target_seq: torch.Tensor,
            pred_seq_list: torch.Tensor,
            mode: str = "train"):

        # determine which examples are candidates to vizualize
        if mode == "train":
            example_data_idx_list = self.hparams.logging.visualize.train_example_data_idx_list
        elif mode == "val":
            example_data_idx_list = self.hparams.logging.visualize.val_example_data_idx_list
        elif mode == "test":
            example_data_idx_list = self.hparams.logging.visualize.test_example_data_idx_list
        else:
            raise ValueError(f"Wrong mode {mode}! Must be in ['train', 'val', 'test'].")

        in_seq = self._torch_to_numpy(in_seq)
        target_seq = self._torch_to_numpy(target_seq)
        pred_seq_list = self._torch_to_numpy(pred_seq_list)
        start_time = datetime.fromtimestamp(seq_start_time.item())
        time_delta = timedelta(minutes=self.hparams.dataset.time_delta)

        if data_idx in example_data_idx_list:
            # TODO: add times to our visualization?
            self.visualize.save_example(save_prefix=f'{mode}_epoch_{self.current_epoch}_data_{data_idx}',
                                        in_seq=in_seq,
                                        target_seq=target_seq,
                                        pred_seq_list=[pred_seq_list],
                                        time_delta=self.hparams.dataset.time_delta,
                                        )

            if self.hparams.logging.use_wandb:
                x_images = [wandb.Image(image, caption=start_time + t * time_delta) for t, image in enumerate(in_seq)]
                y_images = [wandb.Image(image, caption=start_time + (t + self.hparams.model.in_len) * time_delta) for
                            t, image in enumerate(target_seq)]
                y_hat_images = [wandb.Image(image, caption=start_time + (t + self.hparams.model.in_len) * time_delta)
                                for
                                t, image in enumerate(pred_seq_list)]

                wandb.log({f"{mode}": {"x": x_images, "y": y_images, "y_hat": y_hat_images}})

    def _get_dm(self):
        dm = IMSLightningDataModule(start_date=datetime(*self.hparams.dataset.start_date),
                                    # TODO: get date filter for each one instead of a fixed date
                                    train_val_split_date=datetime(*self.hparams.dataset.train_val_split_date),
                                    train_test_split_date=datetime(*self.hparams.dataset.train_test_split_date),
                                    end_date=datetime(*self.hparams.dataset.end_date),
                                    batch_size=self.hparams.optim.micro_batch_size,
                                    batch_layout=self.hparams.dataset.batch_layout,
                                    num_workers=self.hparams.optim.num_workers,
                                    img_type=self.hparams.dataset.img_type,
                                    seq_len=self.hparams.dataset.seq_len,
                                    raw_seq_len=self.hparams.dataset.raw_seq_len,
                                    stride=self.hparams.dataset.stride,
                                    time_delta=self.hparams.dataset.time_delta,
                                    raw_time_delta=self.hparams.dataset.raw_time_delta,
                                    layout=self.hparams.dataset.layout,
                                    raw_img_shape=self.hparams.dataset.raw_img_shape,
                                    ims_catalog=self.hparams.dataset.ims_catalog,
                                    ims_data_dir=self.hparams.dataset.ims_data_dir,
                                    grayscale=self.hparams.dataset.preprocess.grayscale,
                                    left=self.hparams.dataset.preprocess.crop.left,
                                    top=self.hparams.dataset.preprocess.crop.top,
                                    width=self.hparams.dataset.preprocess.crop.width,
                                    height=self.hparams.dataset.preprocess.crop.height,
                                    scale=self.hparams.dataset.preprocess.scale,
                                    )
        dm.prepare_data()
        dm.setup()
        return dm

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--logging-dir', default=None, type=str)
    parser.add_argument('--gpus', default=1, type=int)
    parser.add_argument('--cfg', default=None, type=str, help="config file path.")
    parser.add_argument('--ckpt-path', default=None, type=str,
                        help="when set the model will start from that pretrained checkpoint.")
    parser.add_argument('--state-dict-file-name', default=None, type=str,
                        help="when set the model will start from that state dict.")
    parser.add_argument('--pretrained', default=False, type=bool,
                        help="when set to True the model will only be tested."
                             "only one of --state-dict-file-name or --ckpt-path must be set.")
    return parser


def main():
    logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)  # suppress WARN massages in console

    parser = get_parser()
    args = parser.parse_args()

    # model
    l_module = CuboidIMSModule(logging_dir=args.logging_dir,
                               args=args.__dict__)
    # data
    dm = l_module.dm

    # seed
    seed_everything(seed=l_module.hparams.optim.seed, workers=True)

    # set trainer
    trainer_kwargs = l_module.get_trainer_kwargs(args.gpus)
    trainer = pl.Trainer(**trainer_kwargs)

    if args.state_dict_file_name is not None and args.ckpt_path is not None:
        sys.exit("both state-dict-file-name and ckpt-path are set!")

    if args.state_dict_file_name is not None:
        state_dict_path = os.path.join(pretrained_checkpoints_dir, args.state_dict_file_name)
        if not os.path.exists(state_dict_path):
            warnings.warn(f"state dict {state_dict_path} not exists!")
        else:
            state_dict = torch.load(state_dict_path)
            l_module.cuboid_attention_model.load_state_dict(state_dict=state_dict)
            print(f"Using state dict {state_dict_path}")

    if args.ckpt_path is not None:
        if not os.path.exists(args.ckpt_path):
            warnings.warn(f"checkpoint {args.ckpt_path} not exists!")
        else:
            print(f"Using checkpoint {args.ckpt_path}")

    if args.pretrained:
        trainer.test(model=l_module,
                     datamodule=dm)
    else:
        trainer.fit(model=l_module,
                    datamodule=dm,
                    ckpt_path=args.ckpt_path)


if __name__ == "__main__":
    main()
