import torch
import torch.nn as nn
import torch.nn.functional as F

from model.transformer.blocks_double import TransformerEncoderLayer, TransformerEncoder, MLP
from model.positional_encoder.cloud_encoder import Fixed3DPositionalEncoder
from model.positional_encoder.pheno_encoder import PhenoFixed3DPositionalEncoder
from model.positional_encoder.position_encoding import build_position_encoding
from model.positional_encoder.zero_encoder import ZeroEncoder
from model.model_util import get_vis_encoder

class FruitletAssociator(nn.Module):
    def __init__(self,
                 d_model,
                 image_size,
                 vis_encoder_args,
                 pos_encoder_args,
                 trans_encoder_args,
                 loss_params,
                 include_bce,
                 is_pheno=False,
                 **kwargs,
                 ):
        super().__init__()

        vis_encoder_args['output_dim'] = d_model
        vis_encoder_args['image_size'] = image_size
        self.vis_encoder = get_vis_encoder(**vis_encoder_args)
        self.encoder_type = vis_encoder_args['encoder_type']

        pos_encoder_args['d_model'] = d_model

        if 'zero' in pos_encoder_args['pos_encoder_type']:
            self.pos_encoder_2d = ZeroEncoder(d_model)
        else:
            self.pos_encoder_2d = build_position_encoding(d_model)

        if not is_pheno:
            self.pos_encoder_3d = Fixed3DPositionalEncoder(**pos_encoder_args)
        else:
            self.pos_encoder_3d = PhenoFixed3DPositionalEncoder(**pos_encoder_args)

        self.d_model = d_model
        self.scale = d_model**0.5

        trans_encoder_args['d_model'] = d_model
        encoder_layer = TransformerEncoderLayer(**trans_encoder_args)
        encoder_norm = nn.LayerNorm(d_model) if encoder_layer.normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, 
                                           trans_encoder_args['num_layers'], 
                                           encoder_norm)
        
        if loss_params['loss_type'] == 'matching':
            self.final_proj = nn.Linear(d_model, d_model)
            self.matchability = nn.Linear(d_model, 1)
        
        self.loss_params = loss_params

        if include_bce:
            self.confidence_pred = MLP(2*d_model, d_model, 1, 3)

        self.include_bce = include_bce
        
    def forward(self, data_0, data_1, matches_gt):
        ims_0, cloud_0, is_pad_0, pos_2ds_0 = data_0
        ims_1, cloud_1, is_pad_1, pos_2ds_1 = data_1

        num_batches, fruitlets_per_batch = ims_0.shape[0:2]
        full_batch_size = num_batches*fruitlets_per_batch

        if not 'pheno' in self.encoder_type:
            _, _, _, im_height, im_width = ims_0.shape
            

            comb_ims = torch.vstack([
                ims_0.view(full_batch_size, 4, im_height, im_width),
                ims_1.view(full_batch_size, 4, im_height, im_width)
            ])
        else:
            comb_ims = torch.vstack([
                ims_0.view(full_batch_size, -1),
                ims_1.view(full_batch_size, -1)
            ])

        vis_enc = self.vis_encoder(comb_ims)

        vis_enc_0, vis_enc_1 = vis_enc.split(full_batch_size)
        vis_enc_0 = vis_enc_0.view(num_batches, fruitlets_per_batch, self.d_model)
        vis_enc_1 = vis_enc_1.view(num_batches, fruitlets_per_batch, self.d_model)

        pos_enc_0_3d = self.pos_encoder_3d(cloud_0)
        pos_enc_1_3d = self.pos_encoder_3d(cloud_1)

        pos_enc_0_2d = self.pos_encoder_2d(pos_2ds_0)
        pos_enc_1_2d = self.pos_encoder_2d(pos_2ds_1)

        pos_enc_0 = pos_enc_0_3d + pos_enc_0_2d
        pos_enc_1 = pos_enc_1_3d + pos_enc_1_2d

        if 'zero' in self.encoder_type:
            vis_enc_0, vis_enc_1 = pos_enc_0, pos_enc_1
            pos_enc_0, pos_enc_1 = None, None
        
        enc_0, enc_1 = self.encoder(vis_enc_0, vis_enc_1,
                                    src_key_padding_mask_0=is_pad_0,
                                    src_key_padding_mask_1=is_pad_1,
                                    pos_0=pos_enc_0,
                                    pos_1=pos_enc_1,
                                    )
        
        # enc_0 and enc_1 are used for contrastive loss
        enc_0 = enc_0 / self.scale
        enc_1 = enc_1 / self.scale

        if self.loss_params['loss_type'] == 'matching':
            # sim, z0, z1 are used for matching loss
            mdesc0, mdesc1 = self.final_proj(enc_0), self.final_proj(enc_1)
            if self.loss_params['use_dist']:
                sim = -torch.cdist(mdesc0, mdesc1)
            else:
                sim = torch.einsum("bmd,bnd->bmn", mdesc0, mdesc1)
                
            z0 = self.matchability(enc_0)
            z1 = self.matchability(enc_1)
        else:
            sim = None
            z0 = None
            z1 = None

        if not self.include_bce:
             return enc_0, enc_1, sim, z0, z1, [], []

        # thse are used for bce_loss. behaviour depends if contrastive or matching
        if self.loss_params['loss_type'] == 'matching':
            with torch.no_grad():
                certainties = F.logsigmoid(z0) + F.logsigmoid(z1).transpose(1, 2)

        confidence_tests = []
        gt_confidences = []
        for ind in range(enc_0.shape[0]):
            b_enc_0 = enc_0[ind, ~is_pad_0[ind]]
            b_enc_1 = enc_1[ind, ~is_pad_1[ind]]
            b_match = matches_gt[ind, ~is_pad_0[ind]][:, ~is_pad_1[ind]]
            
            with torch.no_grad():
                if self.loss_params['loss_type'] == 'matching':
                    b_sim = sim[ind:ind+1, ~is_pad_0[ind]][:, :, ~is_pad_1[ind]]
                    b_cert = certainties[ind:ind+1, ~is_pad_0[ind]][:, :, ~is_pad_1[ind]]

                    scores0 = F.log_softmax(b_sim, 2)
                    scores1 = F.log_softmax(b_sim.transpose(-1, -2).contiguous(), 2).transpose(-1, -2)

                    scores = scores0 + scores1 + b_cert
                    scores = scores[0]

                    match_inds = torch.argwhere(torch.exp(scores) > self.loss_params['match_thresh'])
                elif 'contrastive' in self.loss_params['loss_type']:
                    
                    features_1 = F.normalize(b_enc_0, p=2, dim=-1)[None]
                    features_2 = F.normalize(b_enc_1, p=2, dim=-1)[None]

                    if self.loss_params['dist_type'] == 'l2':
                        distances = torch.cdist(features_1, features_2)[0]
                        match_inds = torch.argwhere(distances < self.loss_params['match_thresh'])
                    elif self.loss_params['dist_type'] == 'cos':
                        cosines = torch.einsum("bmd,bnd->bmn", features_1, features_2)[0]
                        match_inds = torch.argwhere(cosines > self.loss_params['match_thresh'])
                    else:
                        raise RuntimeError('Invalid dist type: ' + self.loss_params['dist_type'])
                else:
                    raise RuntimeError('Invalid loss type: ' + self.loss_params['loss_type'])


            if match_inds.shape[0] == 0:
                continue

            enc_0s_to_cat = b_enc_0[match_inds[:, 0]]
            enc_1s_to_cat = b_enc_1[match_inds[:, 1]]
            is_match = b_match[match_inds[:, 0], match_inds[:, 1]]
            comb_encs = torch.concatenate([enc_0s_to_cat, enc_1s_to_cat], dim=-1)
            confidence_tests.append(comb_encs)
            gt_confidences.append(is_match)

        if len(confidence_tests) > 0:
            confidence_tests = torch.concatenate(confidence_tests)
            gt_confidences = torch.concatenate(gt_confidences)

            pred_confidences = self.confidence_pred(confidence_tests).squeeze(-1)
        else:
            pred_confidences = []
            gt_confidences = []

        return enc_0, enc_1, sim, z0, z1, pred_confidences, gt_confidences
        
