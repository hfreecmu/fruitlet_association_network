import torch
import torch.nn as nn

from model.visual_encoder.resnet_encoder import ResNetEncoder
from model.positional_encoder.ellipsoid_encoder import EllipsoidEncoder
from model.associator import FruitletAssociator
from model.transformer.transformer_encoder import TransformerEncoderLayer

device ='cuda'
d_model = 256
offset_scaling = 2.5

vis_encoder_args = {
    'model_type': 'resnet-18',
}

pos_encoder_args = {
    'ellipse_type': 'ellipse_2d',
    'num_hidden_layers': 2,
    'activation': 'leaky_relu',
    'norm': 'batch_no_track',
    'dropout': 0.3,
}

transformer_encoder_layer_args = {
    'nhead': 8,
    'dropout': 0.3,
    'activation': 'relu',
    'dim_feedforward': 1024,
}

transformer_encoder_args = {
    'num_layers': 6,
    'pos_encoder_args': pos_encoder_args,
    'transformer_encoder_layer_args': transformer_encoder_layer_args,
}

epsilon = 1e-8
def infonce_loss(feats_0, feats_1, is_pad_0=None, is_pad_1=None):
    # negative l2
    dists = -torch.cdist(feats_0, feats_1)

    dists_exp = torch.exp(dists)

    total_loss = 0
    for ind in range(feats_0.shape[1]):
        pos = dists_exp[:, ind, ind]

        neg_0 = dists_exp[:, ind, :].sum(-1)
        neg_1 = dists_exp[:, :, ind].sum(-1)

        loss_0 = -torch.log((epsilon + pos) / (epsilon + neg_0))
        loss_1 = -torch.log((epsilon + pos) / (epsilon + neg_1))

        total_loss = total_loss + loss_0.sum() + loss_1.sum()

    return total_loss

l2_loss = nn.MSELoss(reduction='none')
smooth_l1_loss = torch.nn.SmoothL1Loss(reduction='none')
def ellipse_2d_loss(pos_orig_0, offset_0, 
                    pos_orig_1, offset_1, 
                    offset_scaling,
                    is_pad_0=None, is_pad_1=None):
    
    pos_0 = pos_orig_0 + torch.tanh(offset_0)*offset_scaling
    pos_1 = pos_orig_1 + torch.tanh(offset_1)*offset_scaling

    positions_0 = pos_0[:, :, 0:3]
    positions_1 = pos_1[:, :, 0:3]
    scales_0 = pos_0[:, :, 3:5]
    scales_1 = pos_1[:, :, 3:5]
    angle_0 = pos_0[:, :, 5]
    angle_1 = pos_1[:, :, 5]

    pos_loss = l2_loss(positions_0, positions_1)
    scale_loss = l2_loss(scales_0, scales_1)

    phi = (torch.pi / 2) * (angle_0 - angle_1)
    cos_phi = torch.cos(phi)
    sin_phi = torch.sin(phi)
    rho = torch.where(phi >= 0, torch.atan2(sin_phi, cos_phi), torch.atan2(-sin_phi, -cos_phi))
    angle_loss = smooth_l1_loss(rho, torch.zeros_like(rho))

    return pos_loss.sum(), scale_loss.sum(), angle_loss.sum()

fruitlet_associator = FruitletAssociator(d_model=d_model,
                                         offset_scaling=offset_scaling,
                                         vis_encoder_args=vis_encoder_args,
                                         transformer_encoder_args=transformer_encoder_args
                                         ).to(device)

fruitlet_ims_0 = torch.randn(2, 8, 3, 64, 64, dtype=torch.float32, device=device)
fruitlet_ims_1 = torch.randn_like(fruitlet_ims_0)

fruitlet_ellipses_0 = torch.randn(2, 8, 6, dtype=torch.float32, device=device)
fruitlet_ellipses_1 = torch.randn_like(fruitlet_ellipses_0)

is_pad_0 = torch.zeros(2, 8, dtype=bool, device=device)
is_pad_1 = torch.zeros_like(is_pad_0)

data_0 = (fruitlet_ims_0, fruitlet_ellipses_0, is_pad_0)
data_1 = (fruitlet_ims_1, fruitlet_ellipses_1, is_pad_1)

feats_0, pos_0, feats_1, pos_1, all_feats, all_offsets = fruitlet_associator(data_0, data_1)

nce_loss = infonce_loss(feats_0, feats_1)

offset_test_0, offset_test_1 = all_offsets[0]
pos_loss, scale_loss, angle_loss = ellipse_2d_loss(fruitlet_ellipses_0, offset_test_0, 
                fruitlet_ellipses_1, offset_test_1,
                offset_scaling)

breakpoint()
#image = cv2.imread()