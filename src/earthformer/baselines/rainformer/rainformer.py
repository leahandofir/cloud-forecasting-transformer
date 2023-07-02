# this code is taken from: https://github.com/Zjut-MultimediaPlus/Rainformer/tree/main

import torch
from torch import nn
from earthformer.baselines.rainformer.swintransformer import StageModule, StageModule_up, StageModule_up_final

class Net(nn.Module):
    def __init__(self, input_channel, hidden_dim, downscaling_factors, layers, heads, head_dim, window_size, relative_pos_embedding):
        super(Net, self).__init__()
        self.stage1 = StageModule(in_channels=input_channel, hidden_dimension=hidden_dim, layers=layers[0],
                                  downscaling_factor=downscaling_factors[0], num_heads=heads[0], head_dim=head_dim,
                                  window_size=window_size, relative_pos_embedding=relative_pos_embedding, h_w=[96, 96])

        self.stage2 = StageModule(in_channels=hidden_dim, hidden_dimension=hidden_dim * 2, layers=layers[1],
                                  downscaling_factor=downscaling_factors[1], num_heads=heads[1], head_dim=head_dim,
                                  window_size=window_size, relative_pos_embedding=relative_pos_embedding, h_w=[48, 48])

        self.stage3 = StageModule(in_channels=hidden_dim * 2, hidden_dimension=hidden_dim * 4, layers=layers[2],
                                  downscaling_factor=downscaling_factors[2], num_heads=heads[2], head_dim=head_dim,
                                  window_size=window_size, relative_pos_embedding=relative_pos_embedding, h_w=[24, 24])

        self.stage4 = StageModule(in_channels=hidden_dim * 4, hidden_dimension=hidden_dim * 8, layers=layers[3],
                                  downscaling_factor=downscaling_factors[3], num_heads=heads[3], head_dim=head_dim,
                                  window_size=window_size, relative_pos_embedding=relative_pos_embedding, h_w=[12, 12])

        # ----------------------------------------------------------------------------------------

        self.stage5 = StageModule_up(in_channels=hidden_dim * 8, hidden_dimension=hidden_dim * 4,
                                     layers=layers[3], upscaling_factor=downscaling_factors[3], num_heads=heads[3], head_dim=head_dim,
                                     window_size=window_size, relative_pos_embedding=relative_pos_embedding, h_w=[24, 24])

        self.stage6 = StageModule_up(in_channels=hidden_dim * 8, hidden_dimension=hidden_dim * 2,
                                     layers=layers[2], upscaling_factor=downscaling_factors[2], num_heads=heads[2], head_dim=head_dim,
                                     window_size=window_size, relative_pos_embedding=relative_pos_embedding, h_w=[48, 48])

        self.stage7 = StageModule_up(in_channels=hidden_dim * 4, hidden_dimension=hidden_dim * 1,
                                     layers=layers[1], upscaling_factor=downscaling_factors[1], num_heads=heads[1], head_dim=head_dim,
                                     window_size=window_size, relative_pos_embedding=relative_pos_embedding, h_w=[96, 96])

        self.stage8 = StageModule_up_final(in_channels=hidden_dim * 2, hidden_dimension=input_channel,
                                     layers=layers[0], upscaling_factor=downscaling_factors[0], num_heads=heads[0],
                                     head_dim=head_dim,
                                     window_size=window_size, relative_pos_embedding=relative_pos_embedding, h_w=[384, 384])

    def forward(self, x):
        x1 = self.stage1(x)     # (4, 96, 96, 96)
        x2 = self.stage2(x1)    # (4, 192, 48, 48)
        x3 = self.stage3(x2)    # (4, 384, 24, 24)
        x4 = self.stage4(x3)    # (4, 768, 12, 12)

        x5 = self.stage5(x4, x3)    # (4, 768, 24, 24)
        x6 = self.stage6(x5, x2)    # (4, 384, 48, 48)
        x7 = self.stage7(x6, x1)    # (4, 192, 96, 96)
        x8 = self.stage8(x7)        # (4, 9, 384, 384)

        return x8