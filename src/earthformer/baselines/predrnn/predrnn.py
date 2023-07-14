# this code is taken from https://github.com/thuml/predrnn-pytorch/blob/master/core/models/predrnn.py

__author__ = 'yunbo'

import torch
import torch.nn as nn
from earthformer.baselines.predrnn.spatio_temporal_lstm_cell import SpatioTemporalLSTMCell


class RNN(nn.Module):
    def __init__(self,
                 num_layers,
                 num_hidden,
                 args):
        super(RNN, self).__init__()

        self.frame_channel = args.patch_size * args.patch_size * args.img_channel
        self.num_layers = num_layers
        self.num_hidden = num_hidden
        self.total_length = args.total_length
        self.reverse_scheduled_sampling = args.reverse_scheduled_sampling
        self.input_length = args.input_length
        cell_list = []

        width = args.img_width // args.patch_size
        self.MSE_criterion = nn.MSELoss()

        for i in range(num_layers):
            in_channel = self.frame_channel if i == 0 else num_hidden[i - 1]
            cell_list.append(
                SpatioTemporalLSTMCell(in_channel, num_hidden[i], width, args.filter_size,
                                       args.stride, args.layer_norm)
            )
        self.cell_list = nn.ModuleList(cell_list)
        self.conv_last = nn.Conv2d(num_hidden[num_layers - 1], self.frame_channel,
                                   kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, frames_tensor, mask_true, device="cuda:0"):
        # [batch, length, height, width, channel] -> [batch, length, channel, height, width]
        frames = frames_tensor.permute(0, 1, 4, 2, 3).contiguous()
        mask_true = mask_true.permute(0, 1, 4, 2, 3).contiguous()

        batch = frames.shape[0]
        height = frames.shape[3]
        width = frames.shape[4]

        next_frames = []
        h_t = []
        c_t = []

        for i in range(self.num_layers):
            zeros = torch.zeros([batch, self.num_hidden[i], height, width], device=device)
            h_t.append(zeros)
            c_t.append(zeros)

        memory = torch.zeros([batch, self.num_hidden[0], height, width], device=device)

        for t in range(self.total_length - 1):
            # reverse schedule sampling
            if self.reverse_scheduled_sampling == 1:
                if t == 0:
                    net = frames[:, t]
                else:
                    net = mask_true[:, t - 1] * frames[:, t] + (1 - mask_true[:, t - 1]) * x_gen
            else:
                if t < self.input_length:
                    net = frames[:, t]
                else:
                    net = mask_true[:, t - self.input_length] * frames[:, t] + \
                          (1 - mask_true[:, t - self.input_length]) * x_gen

            h_t[0], c_t[0], memory = self.cell_list[0](net, h_t[0], c_t[0], memory)

            for i in range(1, self.num_layers):
                h_t[i], c_t[i], memory = self.cell_list[i](h_t[i - 1], h_t[i], c_t[i], memory)

            x_gen = self.conv_last(h_t[self.num_layers - 1])
            next_frames.append(x_gen)

        # [length, batch, channel, height, width] -> [batch, length, height, width, channel]
        next_frames = torch.stack(next_frames, dim=0).permute(1, 0, 3, 4, 2).contiguous()
        loss = self.MSE_criterion(next_frames, frames_tensor[:, 1:])
        return next_frames, loss
