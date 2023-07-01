import pytorch_lightning as pl
from pytorch_lightning import seed_everything

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torchmetrics

from earthformer.baselines.rainformer.rainformer import Net
from earthformer.baselines.rainformer.loss import BMAEloss
from earthformer.utils.ims.load import get_x_y_from_batch
from earthformer.config import cfg

import logging
import warnings
import os, sys
import argparse

pretrained_checkpoints_dir = cfg.pretrained_checkpoints_dir

class RainformerIMSModule(IMSModule):
    def __init__(self,
                 args: dict = None,
                 logging_dir: str = None):
        super(RainformerIMSModule, self).__init__(args=args,
                                                  logging_dir=logging_dir,
                                                  current_dir=os.path.dirname(__file__))

        # load rainformer model
        self.rainformer_model = Net(
            input_channel=self.hparams.model.input_channel,
            hidden_dim=self.hparams.model.hidden_dim,
            downscaling_factors=self.hparams.model.downscaling_factors,
            layers=self.hparams.model.layers,
            heads=self.hparams.model.heads,
            head_dim=self.hparams.model.head_dim,
            window_size=self.hparams.model.window_size,
            relative_pos_embedding=self.hparams.model.relative_pos_embedding)

        self.training_loss = BMAEloss()
        self.validation_loss = torchmetrics.MeanSquaredError()

    def forward(self, x):
        return self.rainformer_model(x)

    def training_step(self, batch, batch_idx):
        start_time, x, y = get_x_y_from_batch(batch, self.hparams.model.in_len, self.hparams.model.out_len)

        y_hat = self(x)

        loss = self.training_loss(y_hat, y)

        data_idx = int(batch_idx * self.hparams.optim.micro_batch_size)

        # save our visualization
        self.save_visualization(seq_start_time=start_time[0],
                                in_seq=x[0],
                                target_seq=y[0],
                                pred_seq_list=y_hat[0],
                                data_idx=data_idx,
                                mode="train")

        self.log('train_loss_step', loss, prog_bar=True, on_step=True, on_epoch=False)
        return loss

    def configure_optimizers(self):
        if self.hparams.optim.method == 'adamw':
            optimizer = AdamW(params=self.rainformer_model.parameters(),
                              lr=self.hparams.optim.lr)
        else:
            raise NotImplementedError

        lr_scheduler = ReduceLROnPlateau(optimizer,
                                         mode=self.hparams.optim.scheduler.mode,
                                         factor=self.hparams.optim.scheduler.factor,
                                         patience=self.hparams.optim.scheduler.patience,
                                         verbose=True)

        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler}

    def validation_step(self, batch, batch_idx):
        start_time, x, y = self._get_x_y_from_batch(batch)
        y_hat = self(x)

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

    def validation_epoch_end(self, outputs):
        epoch_loss = self.validation_loss.compute()
        self.log("val_loss_epoch", epoch_loss, sync_dist=True, on_epoch=True)
        self.validation_loss.reset()

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
    l_module = RainformerIMSModule(logging_dir=args.logging_dir,
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
