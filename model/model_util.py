from model.visual_encoder.vit_encoder import ViTEncoder as VisViTEncoder
from model.visual_encoder.resnet_encoder import ResNetEncoder as VisResNetEncoder

from model.positional_encoder.vit_encoder import ViTEncoder as PosViTEncoder
from model.positional_encoder.resnet_encoder import ResNetEncoder as PosResNetEncoder

def get_vis_encoder(encoder_type, **kwargs):
    if encoder_type == 'vit':
        return VisViTEncoder(**kwargs)
    elif 'resnet' in encoder_type:
        return VisResNetEncoder(encoder_type=encoder_type, **kwargs)
    else:
        raise RuntimeError('Invalid vis encoder type: ' + encoder_type)
    
def get_pos_encoder(encoder_type, **kwargs):
    if encoder_type == 'vit':
        return PosViTEncoder(**kwargs)
    elif 'resnet' in encoder_type:
        return PosResNetEncoder(encoder_type=encoder_type, **kwargs)
    else:
        raise RuntimeError('Invalid pos encoder type: ' + encoder_type)
    
