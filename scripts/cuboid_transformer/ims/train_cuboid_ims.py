import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR
from torch.nn import functional as F
import lpips

from earthformer.utils.optim import SequentialLR, warmup_lambda
from earthformer.utils.utils import get_parameter_names
from earthformer.utils.ims.vgg_ims import Vgg16
from earthformer.utils.ims.load_ims import load_model, get_x_y_from_batch
from earthformer.utils.ims.fss_ims import FSSLoss
from earthformer.utils.ims.lpips_ims import preprocess as lpips_preprocess
from earthformer.train.train_ims import IMSModule, main

import numpy as np
import os
from pysteps.verification.spatialscores import fss_init, fss_accum, fss_compute


class CuboidIMSModule(IMSModule):
    def __init__(self,
                 args: dict = None,
                 logging_dir: str = None):
        super(CuboidIMSModule, self).__init__(args=args,
                                              current_dir=os.path.dirname(__file__))

        # load cuboid attention model
        self.cuboid_attention_model = load_model(model_cfg=self.hparams.model)
        self.vgg_model = Vgg16() if \
            not args["test"] and (self.hparams.optim.vgg.enabled or self.hparams.optim.loss.coefficients.vgg > 0) else None
        self.lpips_loss = lpips.LPIPS(net=self.hparams.optim.lpips.net) if \
            not args["test"] and (self.hparams.optim.lpips.enabled or self.hparams.optim.loss.coefficients.lpips > 0) else None
        self.fss_loss = None if args["test"] else \
                        FSSLoss(threshold=self.hparams.optim.fss.threshold,
                                scale=self.hparams.optim.fss.scale,
                                smooth_factor=self.hparams.optim.fss.smooth_factor,
                                hwc=self.hparams.model.hwc,
                                seq_len=self.hparams.model.out_len,
                                pixel_scale=self.hparams.dataset.preprocess.scale,
                                strategy=self.hparams.optim.fss.strategy,
                                minimize=True,
                                device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    def forward(self, x):
        return self.cuboid_attention_model(x)

    def training_step(self, batch, batch_idx):
        start_time, x, y = get_x_y_from_batch(batch, self.hparams.model.in_len, self.hparams.model.out_len)

        y_hat = self(x)

        mse_loss = F.mse_loss(y, y_hat)

        if self.hparams.optim.vgg.enabled:
            vgg_loss = self.vgg_model.loss(y, y_hat, layer=self.hparams.optim.vgg.layer)
        else:
            vgg_loss = 0

        if self.hparams.optim.lpips.enabled:
            # https://github.com/richzhang/PerceptualSimilarity/blob/master/test_network.py
            lpips_loss = self.lpips_loss(lpips_preprocess(y), lpips_preprocess(y_hat)).mean()
        else:
            lpips_loss = 0

        fss_loss = self.fss_loss(target=y, output=y_hat)

        if self.current_epoch < self.hparams.optim.loss_warmup.epochs:
            loss = self.hparams.optim.loss_warmup.coefficients.mse * mse_loss + \
                   self.hparams.optim.loss_warmup.coefficients.vgg * vgg_loss + \
                   self.hparams.optim.loss_warmup.coefficients.fss * fss_loss + \
                   self.hparams.optim.loss_warmup.coefficients.lpips * lpips_loss
        else:
            loss = self.hparams.optim.loss.coefficients.mse * mse_loss + \
                   self.hparams.optim.loss.coefficients.vgg * vgg_loss + \
                   self.hparams.optim.loss.coefficients.fss * fss_loss + \
                   self.hparams.optim.loss.coefficients.lpips * lpips_loss

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
        self.log('train_fss_loss_step', fss_loss, on_step=True, on_epoch=False)
        if self.hparams.optim.vgg.enabled:
            self.log('train_vgg_loss_step', vgg_loss, on_step=True, on_epoch=False)
        if self.hparams.optim.lpips.enabled:
            self.log('train_lpips_loss_step', lpips_loss, on_step=True, on_epoch=False)

        # for sanity check
        if self.hparams.trainer.fss.enabled:
            fss_batch = self._calc_fss_batch(y, y_hat)
            self.log('val_fss_step', fss_batch, on_step=True, on_epoch=False)

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

    def validation_step(self, batch, batch_idx):
        start_time, x, y = get_x_y_from_batch(batch, self.hparams.model.in_len, self.hparams.model.out_len)
        y_hat = self(x)

        self.compute_validation_loss(batch_idx, start_time, x, y, y_hat)

    def test_step(self, batch, batch_idx):
        start_time, x, y = get_x_y_from_batch(batch, self.hparams.model.in_len, self.hparams.model.out_len)
        y_hat = self(x)

        self.compute_test_loss(batch_idx, start_time, x, y, y_hat)

    def _calc_fss_batch(self, y, y_hat):
        """
        y and y_hat are of shape NTHWC.
        Calculates accumulated fss for the whole batch.
        Compares between every pair of ground truth frame and predicted frame for every sequence in the batch.
        If there is more than one channel in the frame, every one of the frames is compared separately.
        """
        pixel_scale = 255 if self.hparams.dataset.preprocess.scale else 1
        fss = fss_init(self.hparams.trainer.fss.threshold, self.hparams.trainer.fss.scale)
        for i in range(self.hparams.optim.micro_batch_size):
            y_sample, y_hat_sample = self._torch_to_numpy(y[i]), self._torch_to_numpy(y_hat[i])
            for j in range(self.hparams.model.out_len):
                for c in range(self.hparams.model.hwc[-1]):
                    fss_accum(fss, y_sample[j, :, :, c] * pixel_scale, y_hat_sample[j, :, :, c] * pixel_scale)

        return fss_compute(fss)


if __name__ == "__main__":
    main(CuboidIMSModule)
