import torch.nn as nn
import timm

class ViTEncoder(nn.Module):
    def __init__(self, 
                 image_size, 
                 output_dim, 
                 use_cls, 
                 pretrained=True,
                 **kwargs):
        super().__init__()

        self.model = timm.create_model(
                model_name='vit_base_patch16_clip_224.openai',
                pretrained=pretrained,
                global_pool='', # '' means no pooling
                num_classes=0,            # remove classification layer,
                img_size=image_size,
        )

        self.use_cls = use_cls

        if self.use_cls:
            self.output_proj = nn.Linear(768, output_dim)
        else:
            output_size = ((image_size//16)**2 + 1)*768
            self.output_proj = nn.Linear(output_size, output_dim)

    def forward(self, x):
        x = self.model(x)
        if self.use_cls:
            x = x[:, 0]
        else:
            x = x.reshape(x.shape[0], -1)
        x = self.output_proj(x)
        return x