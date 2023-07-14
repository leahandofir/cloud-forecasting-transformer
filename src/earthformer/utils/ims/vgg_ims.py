# source: https://github.com/pytorch/examples/blob/d91adc972cef0083231d22bcc75b7aaa30961863/fast_neural_style/neural_style/vgg.py
# relevant articles: https://cs.stanford.edu/people/jcjohns/papers/eccv16/JohnsonECCV16.pdf, https://arxiv.org/pdf/1409.1556.pdf

from collections import namedtuple
import torch
from torchvision import models
from torch.nn import functional as F

class Vgg16(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg16, self).__init__()
        vgg_pretrained_features = models.vgg16(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        vgg_outputs = namedtuple("VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3'])
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3)
        return out

    def loss(self, y, y_hat, layer):
        """
        our original code.
        the shape of y, y_hat is NTHWC.
        the calculated loss is the average of all the samples in that batch.
        """
        # if grayscale, duplicate all channels 3 times
        if y.shape[-1] == 1:
            y = torch.repeat_interleave(y, 3, dim=-1)
            y_hat = torch.repeat_interleave(y_hat, 3, dim=-1)

        # flatten the batch into one long sequence,
        # which can be interpreted as a batch of images
        y = y.flatten(end_dim=1)
        y_hat = y_hat.flatten(end_dim=1)

        # change dimensions to be TCHW
        y = y.permute(0, 3, 1, 2)
        y_hat = y_hat.permute(0, 3, 1, 2)

        y = getattr(self(y), layer)
        y_hat = getattr(self(y_hat), layer)

        loss = F.mse_loss(y, y_hat)  # computes mean over all pixels
        return loss
