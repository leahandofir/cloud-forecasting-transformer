from pytorch_lightning import seed_everything
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, DeviceStatsMonitor, EarlyStopping

import torch

from earthformer.datasets.ims.ims_datamodule import IMSLightningDataModule
from earthformer.visualization.ims.ims_visualize import IMSVisualize
from earthformer.config import cfg
from earthformer.utils.apex_ddp import ApexDDPStrategy
from earthformer.utils.ims.load_ims import load_dataset_params

import wandb
from shutil import copyfile
from datetime import datetime, timedelta
from omegaconf import OmegaConf
import os, sys
import json
import logging
import warnings
import argparse

FIRST_VERSION_NUM = 0

class IMSModule(pl.LightningModule):
    def __init__(self,
                 args: dict = None,
                 logging_dir: str = None,
                 current_dir: str = None):
        super(IMSModule, self).__init__()

        if current_dir is None:
            current_dir = os.path.dirname(__file__)
        self.current_dir = current_dir

        self.args = args

        if args is None or args["cfg"] is None:
            self.cfg_file_path = os.path.join(self.current_dir, "cfg_ims.yaml")
        else:
            self.cfg_file_path = args["cfg"]

        # save hyperparams
        train_cfg = OmegaConf.load(open(self.cfg_file_path, "r"))
        self.save_hyperparameters(train_cfg)

        # data module
        self.dm = self._get_dm()

        # total_num_steps = (number of epochs) * (number of batches in the train data)
        self.total_num_steps = int(self.hparams.optim.max_epochs *
                                   len(self.dm.ims_train) /
                                   self.hparams.optim.total_batch_size)

        # create logging directories and set up logging
        self._init_logging(logging_dir)

    def _init_logging(self, logging_dir: str = None):
        # creates logging directories and adds their path as data members
        if logging_dir is None:
            logging_dir = os.path.join(self.current_dir, "logging")
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

        self.examples_dir = os.path.join(self.curr_version_dir, "examples")
        self.checkpoints_dir = os.path.join(self.curr_version_dir, "checkpoints")
        os.makedirs(self.examples_dir, exist_ok=True)
        os.makedirs(self.checkpoints_dir, exist_ok=True)

        # get visualization config
        self.visualize = IMSVisualize(save_dir=self.examples_dir,
                                      scale=self.hparams.dataset.preprocess.scale,
                                      fs=self.hparams.logging.visualize.fs,
                                      figsize=self.hparams.logging.visualize.figsize,
                                      plot_stride=self.hparams.logging.visualize.plot_stride,
                                      cmap=self.hparams.logging.visualize.cmap)

        # save a copy of the current config inside the logging dir
        cfg_file_target_path = os.path.join(self.curr_version_dir, "cfg.yaml")
        if (not os.path.exists(cfg_file_target_path)) or \
                (not os.path.samefile(self.cfg_file_path, cfg_file_target_path)):
            copyfile(self.cfg_file_path, cfg_file_target_path)

        # save the command line args into json
        args_file_target_path = os.path.join(self.curr_version_dir, "args.json")
        with open(args_file_target_path, 'w') as f:
            json.dump(self.args, f, indent=2)

    def get_trainer_kwargs(self, gpus):
        """
        Default kwargs used when initializing pl.Trainer
        """
        callbacks = []

        # ModelCheckpoint allows fine-grained control over checkpointing
        checkpoint_callback = ModelCheckpoint(
            monitor="val_loss_epoch",
            filename="model-{epoch:03d}",
            save_top_k=self.hparams.optim.save_top_k,
            save_last=True,
            mode="min",
            dirpath=self.checkpoints_dir
        )

        callbacks += [checkpoint_callback, ]

        if self.hparams.optim.early_stop:
            early_callback = EarlyStopping(monitor="val_loss_epoch",
                                           patience=self.hparams.optim.early_stop_patience,
                                           verbose=False,
                                           mode=self.hparams.optim.early_stop_mode)
            callbacks += [early_callback, ]

        if self.hparams.logging.monitor_lr:
            callbacks += [LearningRateMonitor(logging_interval='step'), ]
        if self.hparams.logging.monitor_device:
            callbacks += [DeviceStatsMonitor(), ]

        # logging
        loggers = []
        """ 
        TensorBoardLogger - to see the metrics run the following commands on timon:
        cd leah/cloud-forecasting-transformer/
        tensorboard --logdir <our_logging_dir> --bind_all
        
        Then, go to chrome and connect http://192.168.0.177:6006/.
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
            loggers.append(pl_loggers.WandbLogger(project=self.hparams.logging.wandb.project,
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

    def _torch_to_numpy(self, e):
        return e.detach().float().cpu().numpy()
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
        dm = IMSLightningDataModule(train_val_split_date=datetime(*self.hparams.dataset.train_val_split_date),
                                    train_test_split_date=datetime(*self.hparams.dataset.train_test_split_date),
                                    batch_size=self.hparams.optim.micro_batch_size,
                                    batch_layout=self.hparams.dataset.batch_layout,
                                    num_workers=self.hparams.optim.num_workers,
                                    **load_dataset_params(self.hparams.dataset)
                                    )
        dm.prepare_data()
        dm.setup()
        return dm

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--logging-dir', default=None, type=str)
    parser.add_argument('--gpus', default=1, type=int)
    parser.add_argument('--cfg', default=None, type=str, help="config file path.")
    parser.add_argument('--seed', default=0, type=int, help="training seed.")
    parser.add_argument('--ckpt-path', default=None, type=str,
                        help="when set the model will start from that pretrained checkpoint.")
    parser.add_argument('--state-dict-file-name', default=None, type=str,
                        help="when set the model will start from that state dict.")
    parser.add_argument('--pretrained', default=False, type=bool,
                        help="when set to True the model will only be tested."
                             "only one of --state-dict-file-name or --ckpt-path must be set.")
    return parser


def main(ims_module):
    logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)  # suppress WARN massages in console
    pretrained_checkpoints_dir = cfg.pretrained_checkpoints_dir

    parser = get_parser()
    args = parser.parse_args()

    # seed
    seed_everything(seed=args.seed, workers=True)

    # model
    l_module = ims_module(logging_dir=args.logging_dir,
                          args=args.__dict__)
    # data
    dm = l_module.dm

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
