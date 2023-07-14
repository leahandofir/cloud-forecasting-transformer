# following the instructions:
# https://github.com/richzhang/PerceptualSimilarity#b-backpropping-through-the-metric

import torch

def preprocess(x):
    # x is a torch sequence of shape NxTxHxWxC.
    # the goal of this method is to convert this to (N*T)x3xHxW

    # scale the values to [-1,1]
    x = torch.clamp(x, min=-1, max=1)

    # reshape to desired shape
    x = torch.flatten(x, start_dim=0, end_dim=1)

    # if grayscale, duplicate all channels 3 times
    if x.shape[-1] == 1:
        x = torch.repeat_interleave(x, 3, dim=-1)

    # switch dimensions
    x = x.moveaxis(-1, -3)

    return x
