import torch.nn as nn
import torchvision

MODEL_DICT = {'resnet-18': (torchvision.models.resnet18, 
                            torchvision.models.ResNet18_Weights.IMAGENET1K_V1,
                            512),
              'resnet-50': (torchvision.models.resnet50,
                            torchvision.models.ResNet50_Weights.IMAGENET1K_V2,
                            2048)}

class ResNetEncoder(nn.Module):
    def __init__(self, model_type, output_dim, pretrained=False, **kwargs):
        super().__init__()

        if not model_type in MODEL_DICT:
            raise RuntimeError('Invalid resnet model type: ' + model_type)

        model_func, weights, final_layer_input_dim = MODEL_DICT[model_type]

        if pretrained:
            raise RuntimeError('Pretrained currently not supported. \
                               Remember to fix mean / std.')
        
            model = model_func(weights=weights)
        else:
            model = model_func()

        model.fc = nn.Linear(final_layer_input_dim, output_dim)

        self.model = model

    def forward(self, x):
        x = self.model(x)
        return x

