import torch
import torch.nn as nn
import numpy as np

from model.transformer.blocks_v2 import TransformerEncoder, TransformerEncoderLayer
from data.dataset import CLOUD_STDS

class FixedPositionalEncoder(nn.Module):
    def __init__(self,
                 d_model,
                 temperature,
                 orig_resolution,
                 ):
        super().__init__()
        
        self.temperature = temperature
        self.d_model = d_model

        bin_size = torch.from_numpy(orig_resolution / np.array(CLOUD_STDS, dtype=np.float32))
        self.register_buffer('_bin_size', bin_size)

        # 3 for xyz
        # adding 8 for box
        self.enc_dim = self.d_model // 3 // 2 // 8
        pad = self.d_model - self.enc_dim * 3 * 2 * 8
        self.zero_pad = nn.ZeroPad2d((pad, 0, 0, 0))  # left padding
        

    def forward(self, x): 
        # x = x // self._bin_size
        x = x / self._bin_size

        dim_t = torch.arange(self.enc_dim, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (dim_t / self.enc_dim)

        pos = x[:, :, :, :, None] / dim_t
        pos = torch.stack((pos.sin(), pos.cos()), dim=5).flatten(2)

        enc = self.zero_pad(pos)

        return enc