from torch.optim import AdamW
import torch

from earthformer.baselines.predrnn.predrnn import RNN
from earthformer.baselines.predrnn.preprocess_patches import reshape_patch, reshape_patch_back
from earthformer.baselines.predrnn.scheduling import reserve_schedule_sampling_exp, schedule_sampling
from earthformer.utils.ims.load_ims import get_x_y_from_batch
from earthformer.config import cfg
from earthformer.train.train_ims import IMSModule, main

import os

pretrained_checkpoints_dir = cfg.pretrained_checkpoints_dir


class PredRNNIMSModule(IMSModule):
    def __init__(self,
                 args: dict = None,
                 logging_dir: str = None):
        super(PredRNNIMSModule, self).__init__(args=args,
                                               current_dir=os.path.dirname(__file__))

        # load predrnn model
        self.eta = self.hparams.model.sampling_start_value
        self.predrnn_model = RNN(num_layers=len(self.hparams.model.num_hidden),
                                 num_hidden=self.hparams.model.num_hidden,
                                 args=self.hparams.model)

    def forward(self, x):
        return self.rainformer_model(x)

    def training_step(self, batch, batch_idx):
        start_time, seq = batch
        seq = reshape_patch(seq, self.hparams.model.patch_size)
        itr = (self.current_epoch * (len(self.dm.ims_train) // self.hparams.optim.micro_batch_size)) + batch_idx + 1

        if self.hparams.model.reverse_scheduled_sampling == 1:
            real_input_flag = reserve_schedule_sampling_exp(itr=itr,
                                                            batch_size=self.hparams.optim.micro_batch_size,
                                                            args=self.hparams.model)
        else:
            self.eta, real_input_flag = schedule_sampling(eta=self.eta,
                                                          itr=itr,
                                                          batch_size=self.hparams.optim.micro_batch_size,
                                                          args=self.hparams.model)

        mask = torch.tensor(real_input_flag,
                            dtype=torch.float,
                            device=self.device)

        next_frames, loss = self.predrnn_model(seq, mask, device=self.device)

        if self.hparams.model.reverse_input:
            seq_rev = torch.flip(seq, [1])
            next_frames_rev, loss_rev = self.predrnn_model(seq_rev, mask, device=self.device)
            loss += loss_rev
            loss = loss / 2

        start_time, x, y = get_x_y_from_batch(batch, self.hparams.model.input_length,
                                              self.hparams.model.total_length - self.hparams.model.input_length)
        y_hat = reshape_patch_back(next_frames, self.hparams.model.patch_size)[:, -(self.hparams.model.total_length - self.hparams.model.input_length):]
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
            optimizer = AdamW(params=self.predrnn_model.parameters(),
                              lr=self.hparams.optim.lr)
        else:
            raise NotImplementedError

        return {'optimizer': optimizer}

    def _get_y_hat_from_batch(self, batch):
        start_time, seq = batch
        seq = reshape_patch(seq, self.hparams.model.patch_size)

        # reverse schedule sampling
        if self.hparams.model.reverse_scheduled_sampling == 1:
            mask_input = 1
        else:
            mask_input = self.hparams.model.input_length

        real_input_flag = torch.zeros(
            (self.hparams.optim.micro_batch_size,
             self.hparams.model.total_length - mask_input - 1,
             self.hparams.model.img_width // self.hparams.model.patch_size,
             self.hparams.model.img_width // self.hparams.model.patch_size,
             self.hparams.model.patch_size ** 2 * self.hparams.model.img_channel), device=self.device)

        if self.hparams.model.reverse_scheduled_sampling == 1:
            real_input_flag[:, :self.hparams.model.input_length - 1, :, :] = 1.0

        next_frames, loss = self.predrnn_model(seq, real_input_flag, device=self.device)

        start_time, x, y = get_x_y_from_batch(batch, self.hparams.model.input_length,
                                              self.hparams.model.total_length - self.hparams.model.input_length)
        y_hat = reshape_patch_back(next_frames, self.hparams.model.patch_size)[:, -(self.hparams.model.total_length - self.hparams.model.input_length):]

        return start_time, x, y, y_hat

    def validation_step(self, batch, batch_idx):
        start_time, x, y, y_hat = self._get_y_hat_from_batch(batch)
        self.compute_validation_loss(batch_idx, start_time, x, y, y_hat)

    def test_step(self, batch, batch_idx):
        start_time, x, y, y_hat = self._get_y_hat_from_batch(batch)
        self.compute_test_loss(batch_idx, start_time, x, y, y_hat)


if __name__ == "__main__":
    main(PredRNNIMSModule)
