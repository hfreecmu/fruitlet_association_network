# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Various positional encodings for the transformer.
"""
import math
import torch
from torch import nn

class RotaryEmbedding(nn.Module):
    def __init__(self, d_model, temperature=10000):
        super().__init__()

        N_steps = d_model // 4
        self.num_pos_feats = N_steps
        self.temperature = temperature
        self.scale = 2 * math.pi
        self.d_model = d_model

    def forward(self, tensor):
        x = tensor

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x0 = x[:, :, 0, None] * self.scale / dim_t
        pos_y0 = x[:, :, 1, None] * self.scale / dim_t
        pos_x1 = x[:, :, 2, None] * self.scale / dim_t
        pos_y1 = x[:, :, 3, None] * self.scale / dim_t

        to_sin = torch.concatenate((
            pos_x0[:, :, 0::2], 
            pos_y0[:, :, 0::2],
            pos_x1[:, :, 0::2],
            pos_y1[:, :, 0::2],
            ), dim=-1)
        
        to_cos = torch.concatenate((
            pos_x0[:, :, 1::2], 
            pos_y0[:, :, 1::2],
            pos_x1[:, :, 1::2],
            pos_y1[:, :, 1::2],
            ), dim=-1)
        
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

class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """
    def __init__(self, d_model, temperature=10000):
        super().__init__()

        N_steps = d_model // 4
        self.num_pos_feats = N_steps
        self.temperature = temperature
        self.scale = 2 * math.pi

    def forward(self, tensor):
        x = tensor

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x0 = x[:, :, 0, None] * self.scale / dim_t
        pos_y0 = x[:, :, 1, None] * self.scale / dim_t
        pos_x1 = x[:, :, 2, None] * self.scale / dim_t
        pos_y1 = x[:, :, 3, None] * self.scale / dim_t

        pos_x0 = torch.stack((pos_x0[:, :, 0::2].sin(), pos_x0[:, :, 1::2].cos()), dim=3).flatten(2)
        pos_y0 = torch.stack((pos_y0[:, :, 0::2].sin(), pos_y0[:, :, 1::2].cos()), dim=3).flatten(2)
        pos_x1 = torch.stack((pos_x1[:, :, 0::2].sin(), pos_x1[:, :, 1::2].cos()), dim=3).flatten(2)
        pos_y1 = torch.stack((pos_y1[:, :, 0::2].sin(), pos_y1[:, :, 1::2].cos()), dim=3).flatten(2)

        pos = torch.cat((pos_x0, pos_y0, pos_x1, pos_y1), dim=2)
        return pos

def build_position_encoding(hidden_dim, use_rot):
    if use_rot:
        position_embedding = PositionEmbeddingSine(hidden_dim)
    else:
        position_embedding = RotaryEmbedding(hidden_dim)
    return position_embedding