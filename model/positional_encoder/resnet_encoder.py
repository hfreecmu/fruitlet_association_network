import torch.nn as nn
import torchvision
from torchvision.ops import FrozenBatchNorm2d

class ResNetEncoder(nn.Module):
    def __init__(self, 
                 output_dim, 
                 image_size, 
                 encoder_type,
                 pretrained=True,
                 **kwargs):
        super().__init__()

        if pretrained:
            weights = 'DEFAULT'
        else:
            weights = None

        if encoder_type == 'resnet-18':
            model = torchvision.models.resnet18
        elif encoder_type == 'resnet-34':
            model = torchvision.models.resnet34
        else:
            raise RuntimeError('Invalid encoder_type: ' + encoder_type)

        self.model = nn.Sequential(
            *list(model(weights=weights, norm_layer=FrozenBatchNorm2d).children())[:-2]
        )

        self.model[0] = nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        output_size = 512*(image_size//32)**2
        self.output_proj = nn.Linear(output_size, output_dim)

    def forward(self, x):
        x = self.model(x)
        x = x.reshape(x.shape[0], -1)
        x = self.output_proj(x)
        return x