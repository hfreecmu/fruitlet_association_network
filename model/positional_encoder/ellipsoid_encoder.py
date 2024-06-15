import torch.nn as nn

ACTIVATION_DICT = {
    'relu': nn.ReLU,
    'leaky_relu': nn.LeakyReLU,
    'tanh': nn.Tanh,
}

NORM_DICT = {
    'none': (nn.Identity, {}),
    'batch': (nn.BatchNorm1d, {}),
    'batch_no_track': (nn.BatchNorm1d, {'track_running_stats': False}),
    'frozen_batch': (nn.BatchNorm1d, {'track_running_stats': False, 
                                      'affine': False}),
    'layer': (nn.LayerNorm, {}),
}

# TODO or should I do rotation angle thing from cv class?
ELLIPSE_DICT = {
    'ellipse_2d': 4,#7, # 3 for pos, 2 for scale, 1 for angle #TODO not sure, 1 for has ellipse
    'ellipsoid_3d': 11, # 3 for pos, 3 for scale, and 4 for qua #TODO not sure, 1 for has ellipse
}

class EllipsoidEncoder(nn.Module):
    def __init__(self, ellipse_type, hidden_layer_dim, num_hidden_layers, 
                 output_dim, activation, norm, dropout):
        super().__init__()


        if not ellipse_type in ELLIPSE_DICT:
            raise RuntimeError('Invalid ellipse_type: ' + ellipse_type)

        if not activation in ACTIVATION_DICT:
            raise RuntimeError('Invalid activation: ' + activation)
        
        if not norm in NORM_DICT:
            raise RuntimeError('Invalid norm: ' + norm)
        
        input_dim = ELLIPSE_DICT[ellipse_type]
        activation_func = ACTIVATION_DICT[activation]
        norm_func, norm_kwargs = NORM_DICT[norm]

        layers = []
        prev_dim = input_dim
        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(prev_dim, hidden_layer_dim))
            layers.append(norm_func(hidden_layer_dim, **norm_kwargs))
            layers.append(activation_func())

            if dropout is not None:
                layers.append(nn.Dropout(dropout))

            prev_dim = hidden_layer_dim

        layers.append(nn.Linear(prev_dim, output_dim))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.layers(x)
        return x

