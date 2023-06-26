import torch
import torch.nn.functional as F


class FSSLoss(torch.nn.Module):
    # TODO: write what every input is suppose to be
    def __init__(self,
                 threshold: int,
                 scale: int,
                 hwc: tuple,
                 seq_len: int,
                 minimize: bool = True,
                 smooth_factor: int = 20,
                 pixel_scale: bool = True,
                 device: str = 'cuda:0'):
        super(FSSLoss, self).__init__()
        self.threshold = threshold
        self.scale = scale
        self.hwc = hwc
        self.seq_len = seq_len
        self.minimize = -1 if minimize else 0
        self.smooth_factor = smooth_factor
        self.pixel_scale = 255 if pixel_scale else 1
        self.neighborhood_filter_dim = (self.seq_len * self.hwc[-1], self.seq_len * self.hwc[-1], self.scale, self.scale) # TODO: check
        self.neighborhood_filter_mat = torch.full(self.neighborhood_filter_dim, 1 / self.scale ** 2, device=device)

    # warning - heavily assumes layout NTWHC!

    def _preprocess(self, batch):
        # the input is of shape NTHWC
        # rescale pixels back to 0-255
        batch = batch * self.pixel_scale

        # flatten sequence to 3-dim by increasing number of channels, the resulted shape is NCWH
        # (used negative indices in order to handle both batch-input and single-input)
        batch = torch.flatten(batch.moveaxis(-1, -3), start_dim=-4, end_dim=-3)

        # 'binarize' data
        batch = F.hardtanh(self.smooth_factor * (batch - self.threshold), min_val=0, max_val=1)

        return batch

    def forward(self, output, target):
        # preprocess
        output = self._preprocess(output)
        target = self._preprocess(target)

        # compute each neighborhood's average value by applying convolution filter and convert shape back to NHWC
        if self.scale > 1:
            F_n = F.conv2d(output, self.neighborhood_filter_mat).moveaxis(-3, -1)
            O_n = F.conv2d(target, self.neighborhood_filter_mat).moveaxis(-3, -1)
        else:
            F_n = output
            O_n = target

        numerator = ((F_n - O_n) ** 2).sum(dim=-2).sum(dim=-2)
        denominator = (F_n ** 2).sum(dim=-2).sum(dim=-2) + (O_n ** 2).sum(dim=-2).sum(dim=-2)

        # compute the mean loss for each sequence (loss is computed frame by frame)
        fss_per_batch = (1 - numerator / denominator).mean(dim=-1)

        # return the average loss over all batches, multiply by -1 if we want to minimize
        return self.minimize * fss_per_batch.mean()
