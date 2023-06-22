import torch
import torch.nn.functional as F

class FSSLoss(nn.Module):
    def __init__(self, threshold, scale, hwc, pixel_scale):
        super(FSSLoss, self).__init__()
        self.threshold = threshold
        self.scale = scale
        self.hwc = hwc
        self.pixel_scale = pixel_scale

    # WORK IN PROGRESS: right now FSS is computed for each couple of frames separately,
    #                   and is NOT aggregated over the entire sequence
    # warning - heavily assumes layout NTWHC !
    def forward(self, output, target):
        
        # rescale pixels back to 0-255
        output = output * self.pixel_scale
        target = target * self.pixel_scale

        # # if grayscale, 'force' to be 3-channel
        # if self.hwc[-1] == 1:
        #     output = torch.repeat_interleave(output, 3, dim=-1)
        #     target = torch.repeat_interleave(target, 3, dim=-1)

        # flatten sequence to 3-dim by increasing number of channels
        # (used negative indices in order to handle both batch-input and single-input)
        output = torch.flatten(output.moveaxis(-1, -3), start_dim=-4, end_dim=-3).moveaxis(-3, -1)
        target = torch.flatten(target.moveaxis(-1, -3), start_dim=-4, end_dim=-3).moveaxis(-3, -1) 
        
        # 'binarize' data
        output = F.hardtanh(20 * (output - self.threshold), min_val=0, max_val=1)
        target = F.hardtanh(20 * (target - self.threshold), min_val=0, max_val=1)

        # compute each neighborhood's average value by applying convolution filter
        if self.scale > 1:
            neighborhood_filter_dim = (output.shape[-1], output.shape[-1], self.scale, self.scale)
            neighborhood_filter_mat = torch.full(neighborhood_filter_dim, 1 / self.scale ** 2)

            F_n = F.conv2d(output.moveaxis(-1, -3), neighborhood_filter_mat).moveaxis(-3, -1) 
            O_n = F.conv2d(target.moveaxis(-1, -3), neighborhood_filter_mat).moveaxis(-3, -1) 
        else:
            F_n = output
            O_n = target

        numerator   = ((F_n - O_n) ** 2).sum(dim=-2).sum(dim=-2)                             
        denominator = (F_n ** 2).sum(dim=-2).sum(dim=-2) + (O_n ** 2).sum(dim=-2).sum(dim=-2)

        return 1 - numerator / denominator