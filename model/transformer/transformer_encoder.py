import torch
import torch.nn as nn

import model.transformer.blocks as blocks
from model.positional_encoder.ellipsoid_encoder import EllipsoidEncoder, ELLIPSE_DICT

class TransformerEncoder(nn.Module):
    def __init__(self,
                 d_model,
                 offset_scaling,
                 num_layers,
                 pos_encoder_args,
                 transformer_encoder_layer_args,
                 ):
        super().__init__()

        self.num_layers = num_layers
        self.d_model = d_model
        self.offset_scaling = offset_scaling

        self.encoder_norm = nn.LayerNorm(d_model)

        self.transformer_encoder_layers = nn.ModuleList()
        #TODO bonn uses dropout = 0 and relu
        for _ in range(self.num_layers):
            self.transformer_encoder_layers.append(
                TransformerEncoderLayer(d_model=d_model,
                                        **transformer_encoder_layer_args,
                                        )
            )

            # TODO input proj for visual or descriptor encoder?
            # bonn has one

        # TODO positional encoder for each layer? I say no
        self.positional_encoder = EllipsoidEncoder(output_dim=d_model, 
                                                   hidden_layer_dim=d_model,
                                                   **pos_encoder_args,
                                                   )

        self.offset_head = blocks.MLP(
            d_model, d_model, output_dim=ELLIPSE_DICT[pos_encoder_args['ellipse_type']], 
            num_layers=3, tanh=False
        )

    def forward(self, feats_0, pos_0,
                feats_1, pos_1,
                attn_mask_0, attn_mask_1, 
                padding_mask_0, padding_mask_1):
        num_batches, fruitlets_per_batch, ellipse_dim = pos_0.shape
        full_batch_size = num_batches*fruitlets_per_batch
        
        #TODO forward prediction before any layer like bonn?
        # TODO like this or pass vis feats each layer?
        
        # TODO unsure if I should add any before layers
        # not sure why bonn does it
        all_feats = []
        all_offsets = []

        for ind in range(self.num_layers):
            # encode pos
            fp_0 = pos_0.view(full_batch_size, ellipse_dim)
            fp_1 = pos_1.view(full_batch_size, ellipse_dim)
            fp = torch.vstack([fp_0, fp_1])
            feats_pos = self.positional_encoder(fp)
            feats_pos_0, feats_pos_1 = feats_pos.split(full_batch_size)
            feats_pos_0 = feats_pos_0.view(num_batches, fruitlets_per_batch, self.d_model)
            feats_pos_1 = feats_pos_1.view(num_batches, fruitlets_per_batch, self.d_model)

            # pass through transformer layer
            feats_0, feats_1 = self.transformer_encoder_layers[ind](
                feats_0, feats_pos_0, feats_1, feats_pos_1,
                attn_mask_0, attn_mask_1, 
                padding_mask_0, padding_mask_1
            )

            # update ellipse
            feats_comb = torch.vstack([
                feats_0.view(full_batch_size, self.d_model),
                feats_1.view(full_batch_size, self.d_model)
            ])
            encoder_output = self.encoder_norm(feats_comb)
            offset = self.offset_head(encoder_output)
            offset_0, offset_1 = offset.split(full_batch_size)
            offset_0 = offset_0.view(num_batches, fruitlets_per_batch, -1)
            offset_1 = offset_0.view(num_batches, fruitlets_per_batch, -1)

            # TODO do I want to sigmoid this?
            # I will say yes because coordinates are normalized?
            # I should make all within unit sphere
            # also I'd want t
            # offset_scaled_0 = torch.sigmoid(offset_0) * self.offset_scaling
            # offset_scaled_1 = torch.sigmoid(offset_1) * self.offset_scaling
            offset_scaled_0 = torch.tanh(offset_0) * self.offset_scaling
            offset_scaled_1 = torch.tanh(offset_1) * self.offset_scaling

            #pos_0 = pos_0 + offset_scaled_0
            #pos_1 = pos_1 + offset_scaled_1

            all_feats.append([feats_0, feats_1])
            all_offsets.append([offset_0, offset_1])

        return feats_0, pos_0, feats_1, pos_1, all_feats, all_offsets

class TransformerEncoderLayer(nn.Module):
    def __init__(self, 
                 d_model,
                 nhead,
                 dropout,
                 activation,
                 dim_feedforward,
                 ):
        super().__init__()

        self.self_attn = blocks.SelfAttentionLayer(d_model=d_model,
                                                   nhead=nhead,
                                                   dropout=dropout,
                                                   activation=activation)

        self.cross_atten = blocks.CrossAttentionLayer(d_model=d_model,
                                                      nhead=nhead,
                                                      dropout=dropout,
                                                      activation=activation)


        self.feed_forward = blocks.FFNLayer(d_model=d_model,
                                            dim_feedforward=dim_feedforward,
                                            dropout=dropout,
                                            activation=activation)

    def forward(self, feats_0, feats_pos_0, 
                feats_1, feats_pos_1,
                attn_mask_0, attn_mask_1, 
                padding_mask_0, padding_mask_1):
        
        feats_0, feats_1 = self.self_attn(feats_0, feats_1,
                                          attn_mask_0, attn_mask_1, 
                                          padding_mask_0, padding_mask_1,
                                          feats_pos_0, feats_pos_1)
        
        feats_0, feats_1 = self.cross_atten(feats_0, feats_1,
                                            attn_mask_0, attn_mask_1, 
                                            padding_mask_0, padding_mask_1,
                                            feats_pos_0, feats_pos_1)
        
        feats_0 = self.feed_forward(feats_0)
        feats_1 = self.feed_forward(feats_1)

        return feats_0, feats_1
        
