import os

import torch
from torch import nn

#import engine.common
from .stylegan import EqualConv2d
from .stylegan_op import FusedLeakyReLU
#from models.styleganv2.modules import EqualConv2d
#from models.styleganv2.op import FusedLeakyReLU
#from models.common.unet import Unetv2

class EqualDecoder(nn.Module):

    def __init__(self, in_dim, out_dim, layer_num=4, normalization='instance'):
        super().__init__()

        norm_layer = nn.InstanceNorm2d if normalization == 'instance' else nn.BatchNorm2d

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        
        n_out = out_dim

        decoder_layers = []
        decoder_layers.append(EqualConv2d(in_dim, n_out, 3, 1, 1, bias=False))
        decoder_layers.append(norm_layer(n_out, affine=True))
        decoder_layers.append(FusedLeakyReLU(n_out))

        for i in range(layer_num - 1):
          decoder_layers.append(EqualConv2d(n_out, n_out, 3, 1, 1, bias=False))
          decoder_layers.append(norm_layer(n_out, affine=True))
          decoder_layers.append(FusedLeakyReLU(n_out))

        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, input):
        return self.decoder(input)
    

class SmoothNet(nn.Module):
    '''
    A few convolutional layers to smooth the geometric feature tensor
    '''
    def __init__(self, input_nc=16, hidden_nc=16, output_nc=16, use_relu=False):
        super().__init__()
        self.use_relu = use_relu

        self.conv1 = nn.Conv2d(input_nc, hidden_nc, kernel_size=5, stride=1, padding=2, bias=False)
        self.conv2 = nn.Conv2d(hidden_nc, hidden_nc, kernel_size=5, stride=1, padding=2, bias=False)
        self.conv3 = nn.Conv2d(hidden_nc, output_nc, kernel_size=5, stride=1, padding=2, bias=False)
        if use_relu:
            self.relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        if self.use_relu:
            x = self.relu(x)
        x = self.conv2(x)
        if self.use_relu:
            x = self.relu(x)
        x = self.conv3(x)

        return x
