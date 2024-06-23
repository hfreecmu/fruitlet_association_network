import torch
import torch.nn as nn

from model.transformer.blocks_double import TransformerEncoderLayer, TransformerEncoder
from model.positional_encoder.cloud_encoder import FixedPositionalEncoder

# TODO linear layer at end?
class FruitletAssociator(nn.Module):
    def __init__(self,
                 d_model,
                 #image_size,
                 #vis_encoder_args,
                 pos_encoder_args,
                 trans_encoder_args,
                 **kwargs,
                 ):
        super().__init__()

        #vis_encoder_args['output_dim'] = d_model
        #vis_encoder_args['image_size'] = image_size
        #self.vis_encoder = get_vis_encoder(**vis_encoder_args)

        pos_encoder_args['d_model'] = d_model
        self.pos_encoder = FixedPositionalEncoder(**pos_encoder_args)

        self.d_model = d_model
        self.scale = d_model**0.5

        trans_encoder_args['d_model'] = d_model
        encoder_layer = TransformerEncoderLayer(**trans_encoder_args)
        encoder_norm = nn.LayerNorm(d_model) if encoder_layer.normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, 
                                           trans_encoder_args['num_layers'], 
                                           encoder_norm)
        
    def forward(self, data_0, data_1):
        _, cloud_0, is_pad_0 = data_0
        _, cloud_1, is_pad_1 = data_1

        # num_batches, fruitlets_per_batch, _, im_height, im_width = ims_0.shape
        # full_batch_size = num_batches*fruitlets_per_batch

        # comb_ims = torch.vstack([
        #     ims_0.view(full_batch_size, 3, im_height, im_width),
        #     ims_1.view(full_batch_size, 3, im_height, im_width)
        # ])

        # comb_pos = torch.vstack([
        #     pos_0.view(full_batch_size, 4, im_height, im_width),
        #     pos_1.view(full_batch_size, 4, im_height, im_width)
        # ])

        # vis_enc = self.vis_encoder(comb_ims)
        # pos_enc = self.pos_encoder(comb_pos)

        # vis_enc_0, vis_enc_1 = vis_enc.split(full_batch_size)
        # vis_enc_0 = vis_enc_0.view(num_batches, fruitlets_per_batch, self.d_model)
        # vis_enc_1 = vis_enc_1.view(num_batches, fruitlets_per_batch, self.d_model)

        # pos_enc_0, pos_enc_1 = pos_enc.split(full_batch_size)
        # pos_enc_0 = pos_enc_0.view(num_batches, fruitlets_per_batch, self.d_model)
        # pos_enc_1 = pos_enc_1.view(num_batches, fruitlets_per_batch, self.d_model)

        pos_enc_0 = self.pos_encoder(cloud_0)
        pos_enc_1 = self.pos_encoder(cloud_1)
        
        # vis_enc_0 = torch.zeros_like(pos_enc_0)
        # vis_enc_1 = torch.zeros_like(pos_enc_1)
        enc_0, enc_1 = self.encoder(pos_enc_0, pos_enc_1,
                                    src_key_padding_mask_0=is_pad_0,
                                    src_key_padding_mask_1=is_pad_1,
                                    #pos_0=pos_enc_0,
                                    #pos_1=pos_enc_1,
                                    )
        
        enc_0 = enc_0 / self.scale
        enc_1 = enc_1 / self.scale

        return enc_0, enc_1
        
