import torch
import torch.nn as nn

from model.visual_encoder.resnet_encoder import ResNetEncoder
from model.transformer.transformer_encoder import TransformerEncoder

class FruitletAssociator(nn.Module):
    def __init__(self,
                 d_model,
                 offset_scaling,
                 vis_encoder_args,
                 transformer_encoder_args
                 ):
        super().__init__()

        self.d_model = d_model

        self.visual_encoder = ResNetEncoder(output_dim=d_model, **vis_encoder_args)
        self.transformer_encoder = TransformerEncoder(d_model=d_model,
                                                      offset_scaling=offset_scaling,
                                                      **transformer_encoder_args)

    # assumes ellipsoids already aligned
    # ellipses could mean ellipsoids depends how many dims
    def forward(self, data_0, data_1):
        fruitlet_ims_0, fruitlet_ellipses_0, is_pad_0 = data_0
        fruitlet_ims_1, fruitlet_ellipses_1, is_pad_1 = data_1

        num_batches, fruitlets_per_batch, im_dim, im_height, im_width = fruitlet_ims_0.shape
        full_batch_size = num_batches*fruitlets_per_batch

        fi_0 = fruitlet_ims_0.view(full_batch_size, im_dim, im_height, im_width)
        fi_1 = fruitlet_ims_1.view(full_batch_size, im_dim, im_height, im_width)
        fi = torch.vstack((fi_0, fi_1))
        
        vis_enc = self.visual_encoder(fi)

        vis_enc_0, vis_enc_1 = vis_enc.split(full_batch_size)
        vis_enc_0 = vis_enc_0.view(num_batches, fruitlets_per_batch, self.d_model)
        vis_enc_1 = vis_enc_1.view(num_batches, fruitlets_per_batch, self.d_model)

        res = self.transformer_encoder(
            vis_enc_0, fruitlet_ellipses_0, 
            vis_enc_1, fruitlet_ellipses_1,
            None, None,
            is_pad_0, is_pad_1,
        )

        return res
