import torch
import torch.nn as nn
import numpy as np
import math

class PhenoFixed3DPositionalEncoder(nn.Module):
    def __init__(self,
                 d_model,
                 temperature,
                 orig_resolution,
                 **kwargs,
                 ):
        super().__init__()
        
        self.temperature = temperature
        self.scale = 2 * math.pi
        self.bin_size = orig_resolution

        # has to be even number
        self.num_pos_feats = d_model // 3
        if self.num_pos_feats % 2 == 1:
            self.num_pos_feats -= 1

        pad = d_model - self.num_pos_feats * 3
        self.zero_pad = nn.ZeroPad2d((pad, 0, 0, 0))  # left padding
        
    def encode(self, x):
        # x = x // self._bin_size
        x = x / self.bin_size

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x[:, :, 0, None] * self.scale / dim_t
        pos_y = x[:, :, 1, None] * self.scale / dim_t
        pos_z = x[:, :, 2, None] * self.scale / dim_t

        pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
        pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)
        pos_z = torch.stack((pos_z[:, :, 0::2].sin(), pos_z[:, :, 1::2].cos()), dim=3).flatten(2)

        pos = torch.cat((pos_x, pos_x, pos_y), dim=2)

        enc = self.zero_pad(pos)

        return enc

    def forward(self, x): 
        enc = self.encode(x)
        return enc