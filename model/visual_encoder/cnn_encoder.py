import torch
import torch.nn as nn

#TODO leaky or non leaky?
def conv(in_channels, out_channels, relu, norm, dropout, kernel_size, stride, padding=0):
    layers = []

    conv_layer = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=True)
    layers.append(conv_layer)

    if norm == "instance":
        layers.append(nn.InstanceNorm2d(out_channels))
    elif norm is None:
        pass
    else:
        raise RuntimeError('Illagel norm passed: ' + norm)
    
    if relu:
        layers.append(nn.ReLU())

    if dropout:
        layers.append(nn.Dropout(0.5))

    return nn.Sequential(*layers)

#TODO add or cat?
class ResnetBlock(nn.Module):
    def __init__(self, in_channels, conv_out_channels, relu, norm, dropout, mode="add"):
        super(ResnetBlock, self).__init__()
        self.conv_layer = conv(in_channels=in_channels, out_channels=conv_out_channels, relu=relu, norm=norm, dropout=dropout, kernel_size=3, stride=1, padding=1)
        self.mode = mode

    def forward(self, x):
        if self.mode == "cat":
            out = torch.cat([x, self.conv_layer(x)], 1)
        elif self.mode == "add":
            out = x + self.conv_layer(x)
        else:
            raise RuntimeError("Illegal mode: " + self.mode)
        return out

class DescriptorEncoder(nn.Module):
    def __init__(self, output_dim, num_res=3, num_res_dropout=1, norm='instance', **kwargs):
        super(DescriptorEncoder, self).__init__()

        self.conv_in = conv(4, output_dim, relu=True, norm=norm, dropout=True, kernel_size=7, stride=3)

        resnets = []
        for i in range(num_res):
            dropout = (i < num_res_dropout)
            resnets.append(ResnetBlock(output_dim, output_dim, relu=True, norm=norm, dropout=dropout))
        self.resnets = nn.Sequential(*resnets)

        self.conv_out = conv(output_dim, output_dim, relu=False, norm=None, dropout=False, kernel_size=7, stride=3)

        self.proj_out = nn.Linear(6400, output_dim)

    def forward(self, x):
        x = self.conv_in(x)
        x = self.resnets(x)

        x = self.conv_out(x)
        
        x = x.flatten(1)
        x = self.proj_out(x)

        return x