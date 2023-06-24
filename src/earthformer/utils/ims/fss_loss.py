import torch
import torch.nn.functional as F


class FSSLoss(torch.nn.Module):
    # TODO: write what every input is suppose to be
    def __init__(self,
                 threshold: int,
                 scale: int,
                 hwc: tuple,
                 minimize: bool = True,
                 smooth_factor: int = 20,
                 pixel_scale: int = 255):
        super(FSSLoss, self).__init__()
        self.threshold = threshold
        self.scale = scale
        self.hwc = hwc
        self.minimize = -1 if minimize else 0
        self.smooth_factor = smooth_factor
        self.pixel_scale = pixel_scale

    # warning - heavily assumes layout NTWHC!
    def forward(self, output, target):
        # rescale pixels back to 0-255
        output = output * self.pixel_scale
        target = target * self.pixel_scale

        # flatten sequence to 3-dim by increasing number of channels
        # (used negative indices in order to handle both batch-input and single-input)
        output = torch.flatten(output.moveaxis(-1, -3), start_dim=-4, end_dim=-3).moveaxis(-3, -1)
        target = torch.flatten(target.moveaxis(-1, -3), start_dim=-4, end_dim=-3).moveaxis(-3, -1)

        # 'binarize' data
        output = F.hardtanh(self.smooth_factor * (output - self.threshold), min_val=0, max_val=1)
        target = F.hardtanh(self.smooth_factor * (target - self.threshold), min_val=0, max_val=1)

        # compute each neighborhood's average value by applying convolution filter
        if self.scale > 1:
            neighborhood_filter_dim = (output.shape[-1], output.shape[-1], self.scale, self.scale)
            neighborhood_filter_mat = torch.full(neighborhood_filter_dim, 1 / self.scale ** 2)

            F_n = F.conv2d(output.moveaxis(-1, -3), neighborhood_filter_mat).moveaxis(-3, -1)
            O_n = F.conv2d(target.moveaxis(-1, -3), neighborhood_filter_mat).moveaxis(-3, -1)
        else:
            F_n = output
            O_n = target

        numerator = ((F_n - O_n) ** 2).sum(dim=-2).sum(dim=-2)
        denominator = (F_n ** 2).sum(dim=-2).sum(dim=-2) + (O_n ** 2).sum(dim=-2).sum(dim=-2)

        # compute the mean loss for each sequence (loss is computed frame by frame)
        # and multiply by -1 if we want to minimize
        return self.minimize * (1 - numerator / denominator).mean(dim=-1)
