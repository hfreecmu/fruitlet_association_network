import torch
import torch.nn as nn
import numpy as np

from model.transformer.blocks_v2 import TransformerEncoder, TransformerEncoderLayer
from data.dataset import CLOUD_STDS

class FixedPositionalEncoder(nn.Module):
    def __init__(self,
                 output_dim,
                 temperature=10000,
                 orig_resolution=0.002,
                 ):
        super().__init__()
        
        self.temperature = temperature
        self.output_dim = output_dim

        bin_size = torch.from_numpy(orig_resolution / np.array(CLOUD_STDS, dtype=np.float32))
        self.register_buffer('_bin_size', bin_size)

        # 3 for xyz
        self.enc_dim = self.output_dim // 3 // 2
        pad = self.output_dim - self.enc_dim * 3 * 2
        self.zero_pad = nn.ZeroPad2d((pad, 0, 0, 0))  # left padding
        

    def forward(self, x): 
        x = x // self._bin_size

        dim_t = torch.arange(self.enc_dim, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (dim_t / self.enc_dim)


        pos = x[:, :, :, None] / dim_t
        pos = torch.stack((pos.sin(), pos.cos()), dim=4).flatten(2)

        enc = self.zero_pad(pos)
        
        return enc
        
class CloudEncoder(nn.Module):
    def __init__(self, 
                 d_model,
                 trans_encoder_args,
                 nearest_k=50
                 ):
        super().__init__()

        trans_encoder_args['d_model'] = d_model
        self.nearest_k = nearest_k
        self.nhead = trans_encoder_args['nhead']

        self.pos_encoder = FixedPositionalEncoder(d_model)

        encoder_layer = TransformerEncoderLayer(**trans_encoder_args)
        encoder_norm = nn.LayerNorm(d_model) if encoder_layer.normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, 
                                          trans_encoder_args['num_layers'], 
                                          encoder_norm)
        
        self.query_embed = nn.Embedding(1, d_model)
        
    def forward(self, x, is_pad):
        _, n_fruitlets, max_cloud_size, _ = x.shape

        encs = []
        for batch_ind in range(x.shape[0]):
            x_batch = x[batch_ind]
            is_pad_batch = is_pad[batch_ind]

            dists = torch.cdist(x_batch, x_batch)
            _, inds = torch.topk(dists, k=self.nearest_k, largest=False, dim=2, sorted=False)

            # adding 1 for embed
            # attn_mask true not attend
            attn_mask = torch.ones(n_fruitlets, max_cloud_size + 1, max_cloud_size + 1, 
                                    dtype=torch.bool, device=x_batch.device)

            # attn_mask[np.arange(n_fruitlets)[:, None, None],
            #           np.arange(max_cloud_size)[None, :, None],
            #           inds] = False
            for fruitlet_ind in range(n_fruitlets):
                for cloud_ind in range(max_cloud_size):
                    attn_mask[fruitlet_ind, cloud_ind, inds[fruitlet_ind, cloud_ind]] = False

            # encode cloud
            pos_embed = self.pos_encoder(x_batch)
            query_embed = self.query_embed.weight.unsqueeze(0).repeat(n_fruitlets, 1, 1)
        
            comb_embeds = torch.concatenate([pos_embed, query_embed], dim=1)
            comb_is_pad = torch.concatenate([is_pad_batch, torch.zeros(n_fruitlets, 1, dtype=bool, 
                                                                       device=comb_embeds.device)], dim=-1)
            # all attends to query and query attends to all
            attn_mask[:, -1, :] = False
            attn_mask[:, :, -1] = False
            #attn_mask[:, -1, -1] = False

            attn_interleave = torch.repeat_interleave(attn_mask, self.nhead, dim=0)
            #attn_interleave = attn_mask.repeat(self.nhead, 1, 1)

            #src = torch.zeros_like(comb_embeds)


            # pass through encoder
            enc = self.encoder(
                src=comb_embeds, 
                mask=attn_interleave,
                src_key_padding_mask=comb_is_pad, 
                #pos=comb_embeds
            )

            enc = enc[:, -1]

            encs.append(enc)

        encs = torch.stack(encs)

        return encs