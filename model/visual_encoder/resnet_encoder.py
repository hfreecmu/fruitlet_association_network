import torch.nn as nn
import torchvision
from torchvision.ops import FrozenBatchNorm2d

from model.transformer.blocks_double import MLP

class ResNetEncoder(nn.Module):
    def __init__(self, 
                 image_size, 
                 output_dim, 
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
        #self.output_proj = nn.Linear(output_size, output_dim)

        self.output_proj = MLP(output_size, output_dim, output_dim, 2)
        self.use_out_proj = MLP(output_size, output_dim, 1, 2)


    def forward(self, x):
        x = self.model(x)
        x = x.reshape(x.shape[0], -1)
        #x = self.output_proj(x)

        use_proj = nn.functional.sigmoid(self.use_out_proj(x))
        x = self.output_proj(x) * use_proj

        return x