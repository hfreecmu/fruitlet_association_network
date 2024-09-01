# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Various positional encodings for the transformer.
"""
import math
import torch
from torch import nn

class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """
    def __init__(self, num_pos_feats=64, temperature=10000):
        super().__init__()
        self.num_pos_feats = num_pos_feats
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

def build_position_encoding(hidden_dim):
    N_steps = hidden_dim // 4
    position_embedding = PositionEmbeddingSine(N_steps)

    return position_embedding