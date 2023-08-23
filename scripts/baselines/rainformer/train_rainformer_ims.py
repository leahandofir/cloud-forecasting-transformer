from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch

from earthformer.baselines.rainformer.rainformer import Net
from earthformer.baselines.rainformer.loss import BMAEloss
from earthformer.utils.ims.load_ims import get_x_y_from_batch
from earthformer.config import cfg
from earthformer.train.train_ims import IMSModule, main

import os

pretrained_checkpoints_dir = cfg.pretrained_checkpoints_dir

class RainformerIMSModule(IMSModule):
    def __init__(self,
                 args: dict = None,
                 logging_dir: str = None):
        super(RainformerIMSModule, self).__init__(args=args,
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

    def forward(self, x):
        return self.rainformer_model(x)

    def training_step(self, batch, batch_idx):
        # rainformer gets input of shape NTHW
        start_time, x, y = get_x_y_from_batch(batch, self.hparams.model.in_len, self.hparams.model.out_len)
        y_hat = self.rainformer_model(torch.squeeze(x, dim=-1))
        y_hat = torch.unsqueeze(y_hat, dim=-1)

        loss = self.training_loss(y_hat, y)

        data_idx = int(batch_idx * self.hparams.optim.micro_batch_size)

        # save our visualization
        self.save_visualization(seq_start_time=start_time[0],
                                in_seq=x[0],
                                target_seq=y[0],
                                pred_seq_list=y_hat[0],
                                data_idx=data_idx,
                                mode="train",
                                scale=self.hparams.dataset.preprocess.scale)

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

        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler, 'monitor': 'val_loss_epoch'}

    def _get_y_hat_from_batch(self, batch):
        # rainformer gets input of shape NTHW
        start_time, x, y = get_x_y_from_batch(batch, self.hparams.model.in_len, self.hparams.model.out_len)
        y_hat = self.rainformer_model(torch.squeeze(x, dim=-1))
        y_hat = torch.unsqueeze(y_hat, dim=-1)
        return start_time, x, y, y_hat

    def validation_step(self, batch, batch_idx):
        start_time, x, y, y_hat = self._get_y_hat_from_batch(batch)
        self.compute_validation_loss(batch_idx, start_time, x, y, y_hat)

    def test_step(self, batch, batch_idx):
        start_time, x, y, y_hat = self._get_y_hat_from_batch(batch)
        self.compute_test_loss(batch_idx, start_time, x, y, y_hat)

if __name__ == "__main__":
    main(RainformerIMSModule)
