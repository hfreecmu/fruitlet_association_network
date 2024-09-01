import torch
import torch.nn as nn
import numpy as np
import math

from data.dataset import CLOUD_STDS

class Fixed3DPositionalEncoder(nn.Module):
    def __init__(self,
                 d_model,
                 temperature,
                 orig_resolution,
                 ):
        super().__init__()
        
        self.temperature = temperature
        self.scale = 2 * math.pi
        self.bin_size = orig_resolution

        # has to be even number
        self.num_pos_feats = d_model // 6
        pad = d_model - self.num_pos_feats * 5
        self.zero_pad = nn.ZeroPad2d((pad, 0, 0, 0))  # left padding
        
    def forward(self, x): 
        # x = x // self._bin_size
        x = x / self.bin_size

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x_min = x[:, :, 0, None] * self.scale / dim_t
        pos_x_max = x[:, :, 1, None] * self.scale / dim_t
        pos_y_min = x[:, :, 2, None] * self.scale / dim_t
        pos_y_max = x[:, :, 3, None] * self.scale / dim_t
        pos_z_med = x[:, :, 4, None] * self.scale / dim_t

        pos_x_min = torch.stack((pos_x_min[:, :, 0::2].sin(), pos_x_min[:, :, 1::2].cos()), dim=3).flatten(2)
        pos_x_max = torch.stack((pos_x_max[:, :, 0::2].sin(), pos_x_max[:, :, 1::2].cos()), dim=3).flatten(2)
        pos_y_min = torch.stack((pos_y_min[:, :, 0::2].sin(), pos_y_min[:, :, 1::2].cos()), dim=3).flatten(2)
        pos_y_max = torch.stack((pos_y_max[:, :, 0::2].sin(), pos_y_max[:, :, 1::2].cos()), dim=3).flatten(2)
        pos_z_med = torch.stack((pos_z_med[:, :, 0::2].sin(), pos_z_med[:, :, 1::2].cos()), dim=3).flatten(2)

        pos = torch.cat((pos_x_min, pos_x_max, pos_y_min, pos_y_max, pos_z_med), dim=2)

        enc = self.zero_pad(pos)

        return enc