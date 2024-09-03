import torch
import torch.nn as nn

class ZeroEncoder(nn.Module):
    def __init__(self, 
                 output_dim, 
                 **kwargs):
        super().__init__()

        self.output_dim = output_dim

    def forward(self, x):
        return torch.zeros(x.shape[0], x.shape[1], self.output_dim, dtype=x.dtype, device=x.device)
