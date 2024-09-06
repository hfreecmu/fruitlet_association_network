import torch
import torch.nn as nn

class PhenoEncoder(nn.Module):
    def __init__(self, 
                 output_dim,
                 **kwargs):
        super().__init__()

        self.input_proj = nn.Linear(8, output_dim)

    def forward(self, x):
        x = self.input_proj(x)
        return x
