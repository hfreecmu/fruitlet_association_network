from model.visual_encoder.vit_encoder import ViTEncoder as VisViTEncoder
from model.visual_encoder.resnet_encoder import ResNetEncoder as VisResNetEncoder

def get_vis_encoder(encoder_type, **kwargs):
    if encoder_type == 'vit':
        return VisViTEncoder(**kwargs)
    elif 'resnet' in encoder_type:
        return VisResNetEncoder(encoder_type=encoder_type, **kwargs)
    else:
        raise RuntimeError('Invalid vis encoder type: ' + encoder_type)
    
