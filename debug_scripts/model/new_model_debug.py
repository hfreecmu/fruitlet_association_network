import torch
from model.associator import FruitletAssociator

device = 'cuda'

d_model = 256
image_size = 64

vis_encoder_args = {
    'pretrained': True,
    'use_cls': False,
    'encoder_type': 'resnet',
}

pos_encoder_args = {
    'pretrained': True,
    'use_cls': False,
    'encoder_type': 'resnet',
}

trans_encoder_args = {
    'nhead': 8,
    'dim_feedforward': 1024,
    'dropout': 0.0, # change if overfit
    'num_layers': 6,
}

model = FruitletAssociator(d_model, image_size, 
                           vis_encoder_args,
                           pos_encoder_args,
                           trans_encoder_args).to(device)

ims_0 = torch.randn(2, 8, 3, 64, 64).to(device)
ims_1 = torch.randn(2, 8, 3, 64, 64).to(device)

pos_0 = torch.randn(2, 8, 4, 64, 64).to(device)
pos_1 = torch.randn(2, 8, 4, 64, 64).to(device)

is_pad_0 = torch.round(torch.rand(2, 8)).to(bool).to(device)
is_pad_1 = torch.round(torch.rand(2, 8)).to(bool).to(device)

data_0 = (ims_0, pos_0, is_pad_0)
data_1 = (ims_1, pos_1, is_pad_1)

enc_0, enc_1 = model(data_0, data_1)
breakpoint()
