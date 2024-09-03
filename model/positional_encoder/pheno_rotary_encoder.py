import torch
import torch.nn as nn
import numpy as np
import math

class PhenoRotary3DPositionalEncoder(nn.Module):
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
        self.d_model = d_model

        # has to be even number
        self.num_pos_feats = d_model // 6
        if self.num_pos_feats % 2 == 1:
            self.num_pos_feats -= 1

        pad = d_model - self.num_pos_feats * 6

        if pad % 2 == 1:
            raise RuntimeError('did not expect this')
        
        # divide by 2 now because one for sin and cos
        self.zero_pad = nn.ZeroPad2d((pad // 2, 0, 0, 0))  # left padding
                
    def encode(self, x):
        # x = x // self._bin_size
        x = x / self.bin_size

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x_min = x[:, :, 0, None] * self.scale / dim_t
        pos_x_max = x[:, :, 1, None] * self.scale / dim_t
        pos_y_min = x[:, :, 2, None] * self.scale / dim_t
        pos_y_max = x[:, :, 3, None] * self.scale / dim_t
        pos_z_min = x[:, :, 4, None] * self.scale / dim_t
        pos_z_max = x[:, :, 5, None] * self.scale / dim_t

        to_sin = torch.concatenate((
            pos_x_min[:, :, 0::2], 
            pos_x_max[:, :, 0::2],
            pos_y_min[:, :, 0::2],
            pos_y_max[:, :, 0::2],
            pos_z_min[:, :, 0::2],
            pos_z_max[:, :, 0::2],
            ), dim=-1)
        
        to_cos = torch.concatenate((
            pos_x_min[:, :, 1::2], 
            pos_x_max[:, :, 1::2],
            pos_y_min[:, :, 1::2],
            pos_y_max[:, :, 1::2],
            pos_z_min[:, :, 1::2],
            pos_z_max[:, :, 1::2],
            ), dim=-1)
        
        to_sin = self.zero_pad(to_sin)
        to_cos = self.zero_pad(to_cos)

        sin_vals = to_sin.sin()
        cos_vals = to_cos.cos()

        rotary_mat = torch.zeros(x.shape[0], x.shape[1], self.d_model, self.d_model, dtype=x.dtype, device=x.device)

        for dim_ind in range(self.d_model // 2):
            rotary_mat[:, :, 2*dim_ind, 2*dim_ind] = cos_vals[:, :, dim_ind]
            rotary_mat[:, :, 2*dim_ind, 2*dim_ind + 1] = -sin_vals[:, :, dim_ind]
            rotary_mat[:, :, 2*dim_ind + 1, 2*dim_ind] = sin_vals[:, :, dim_ind]
            rotary_mat[:, :, 2*dim_ind + 1, 2*dim_ind + 1] = cos_vals[:, :, dim_ind]

        one_enc = torch.ones(x.shape[0], x.shape[1], 1, self.d_model, dtype=x.dtype, device=x.device)
        enc = torch.matmul(one_enc, rotary_mat).squeeze(-2)

        return enc

    def forward(self, x): 
        enc = self.encode(x)
        return enc