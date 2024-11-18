import torch
import torch.nn as nn
import functools
import numpy as np
import torch.nn.functional as F
from torch import autograd

from torch.nn.modules.utils import _ntuple
TORCH_VERSION = tuple(int(x) for x in torch.__version__.split(".")[:2])

import math

#import util.util as util

###############################################################################
# Functions
###############################################################################
def weights_init(m):
    classname = m.__class__.__name__
    # if classname.find('Conv') != -1:
    if classname.find("conv2d") != -1:
        nn.init.kaiming_normal_(
            m.weight.data,
            a = 0,
            model="fan_out"
        )
        #if classname.find('StyledConv') == -1 and classname.find('FashionDiscConv2d') == -1:
        #if classname not in ["CoordConv2d", "StyledConv", "FashionDiscConv2d", "CoordConvLayer"]:
        #if m.has_key("weight"):
        #m.weight.data.normal_(0.0, 0.02)
    elif classname.find('Linear') != -1 or classname.find('Conv1d') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def define_supreso_lightweight(input_nc, output_nc, factor = 4, norm='instance', out_list = ["rgb"], up_layer = 0):    
    norm_layer = get_norm_layer(norm_type=norm)

    netG = SupResoNet(input_nc, output_nc, factor, norm_layer, out_list = out_list, up_layer = up_layer)
            
    netG.apply(weights_init)
    return netG

class SupResoNet(nn.Module):

    def __init__(self, input_nc, output_nc, factor = 4, norm_layer=nn.InstanceNorm2d, padding_type='reflect', out_list = ["rgb"], up_layer = 0):
                        
        super(SupResoNet, self).__init__()
        activation = nn.ReLU(True)
        self.max_pool = torch.nn.MaxPool2d((2,1),(1,1))

        self.out_list = out_list
        self.output_net_list = []

        if factor == 0:
            for out in self.out_list:                
                out_net = [nn.Conv2d(input_nc, 3, kernel_size=3, stride=1, padding=1), nn.Tanh()]
                setattr(self, f'{out}_net', nn.Sequential(*out_net))
                self.output_net_list.append(f'{out}_net')
            return

        mid_channel = 256

        combine_net = [nn.Conv2d(input_nc, mid_channel, kernel_size=3, stride=1, padding=1), norm_layer(mid_channel), activation]

        if factor == 4: 
            nres = 2
            ndec = 2
        elif factor == 8: 
            nres = 2
            ndec = 2
            factor = 4
        elif factor == 2: 
            nres = 1
            ndec = 1

        self.factor = factor

        ### resnet blocks
        for i in range(nres):
            combine_net += [ResnetBlock(mid_channel, padding_type=padding_type, activation=activation, norm_layer=norm_layer)]
            
        #decoder = [nn.Conv2d(mid_channel, mid_channel, kernel_size=3, stride=1, padding=1), norm_layer(mid_channel), activation]

        decoder = []

        for i in range(1, ndec + 1):
            
            inc = mid_channel // i
            outc = mid_channel // (i * 2)

            decoder += [nn.ConvTranspose2d(inc, outc, kernel_size=3, stride=2, padding=1, output_padding=1), norm_layer(outc), activation]
        
        out_channels = {
            "rgb": 3,
            "mask": 1,
            "normal": 3,
            "uv": 2
        }

        if up_layer:
            outc = mid_channel // factor
            for i in range(up_layer):
                decoder += [nn.Conv2d(outc, outc, kernel_size=3, stride=1, padding=1), norm_layer(outc), activation]



        for out in self.out_list:
            out_net = [nn.Conv2d(mid_channel // factor, mid_channel // factor, kernel_size=3, stride=1, padding=1),activation]
            
            out_net += [nn.ReflectionPad2d(3), nn.Conv2d(mid_channel // factor, out_channels[out], kernel_size=7, padding=0), nn.Tanh()]
            setattr(self, f'{out}_net', nn.Sequential(*out_net))
            self.output_net_list.append(f'{out}_net')


        #rgb_net = [nn.Conv2d(mid_channel // factor, mid_channel // factor, kernel_size=3, stride=1, padding=1),activation]
        #rgb_net += [nn.ReflectionPad2d(3), nn.Conv2d(mid_channel // factor, output_nc, kernel_size=7, padding=0), nn.Tanh()]
        
        self.combine_net = nn.Sequential(*combine_net)
        self.decoder_net = nn.Sequential(*decoder)
        #self.rgb_net = nn.Sequential(*rgb_net)

    def get_last_layer(self):
        def get_layers(model):
            layers = []
            for name, module in model.named_children():
                if isinstance(module, nn.Sequential):
                    layers += get_layers(module)
                elif isinstance(module, nn.ModuleList):
                    for m in module:
                        layers += get_layers(m)
                else:
                    layers.append(module)
            return layers
        layers = get_layers(self.rgb_net)
        return layers[-2]

    def forward(self, input):

        net = input
        if hasattr(self, "combine_net"):
            net = self.combine_net(net)

        if hasattr(self, "decoder_net"):
            net = self.decoder_net(net)

        output_list = []
        for netname in self.output_net_list:
            output_list.append(getattr(self, netname)(net))

        return torch.cat(output_list, 1)


# Exponential moving average for generator weights
def accumulate(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        if k.startswith("criterion") or k.startswith("netD"): continue
        par1[k].data.mul_(decay).add_(par2[k].data, alpha=1 - decay)

def net_to_GPU(net, gpu_ids=[]):
    if False and len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.cuda(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)
    net.apply(weights_init)
    return net

##############################################################################
# Losses
##############################################################################

##############################################################################
# Generator
##############################################################################
class LocalEnhancer(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=32, n_downsample_global=3, n_blocks_global=9, 
                 n_local_enhancers=1, n_blocks_local=3, norm_layer=nn.BatchNorm2d, padding_type='reflect'):        
        super(LocalEnhancer, self).__init__()
        self.n_local_enhancers = n_local_enhancers
        
        ###### global generator model #####           
        ngf_global = ngf * (2**n_local_enhancers)
        model_global = GlobalGenerator(input_nc, output_nc, ngf_global, n_downsample_global, n_blocks_global, norm_layer).model        
        model_global = [model_global[i] for i in range(len(model_global)-3)] # get rid of final convolution layers        
        self.model = nn.Sequential(*model_global)                

        ###### local enhancer layers #####
        for n in range(1, n_local_enhancers+1):
            ### downsample            
            ngf_global = ngf * (2**(n_local_enhancers-n))
            model_downsample = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf_global, kernel_size=7, padding=0), 
                                norm_layer(ngf_global), nn.ReLU(True),
                                nn.Conv2d(ngf_global, ngf_global * 2, kernel_size=3, stride=2, padding=1), 
                                norm_layer(ngf_global * 2), nn.ReLU(True)]
            ### residual blocks
            model_upsample = []
            for i in range(n_blocks_local):
                model_upsample += [ResnetBlock(ngf_global * 2, padding_type=padding_type, norm_layer=norm_layer)]

            ### upsample
            model_upsample += [nn.ConvTranspose2d(ngf_global * 2, ngf_global, kernel_size=3, stride=2, padding=1, output_padding=1), 
                               norm_layer(ngf_global), nn.ReLU(True)]      

            ### final convolution
            if n == n_local_enhancers:                
                model_upsample += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh()]                       
            
            setattr(self, 'model'+str(n)+'_1', nn.Sequential(*model_downsample))
            setattr(self, 'model'+str(n)+'_2', nn.Sequential(*model_upsample))                  
        
        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

    def forward(self, input): 
        ### create input pyramid
        input_downsampled = [input]
        for i in range(self.n_local_enhancers):
            input_downsampled.append(self.downsample(input_downsampled[-1]))

        ### output at coarest level
        output_prev = self.model(input_downsampled[-1])        
        ### build up one layer at a time
        for n_local_enhancers in range(1, self.n_local_enhancers+1):
            model_downsample = getattr(self, 'model'+str(n_local_enhancers)+'_1')
            model_upsample = getattr(self, 'model'+str(n_local_enhancers)+'_2')            
            input_i = input_downsampled[self.n_local_enhancers-n_local_enhancers]            
            output_prev = model_upsample(model_downsample(input_i) + output_prev)
        return output_prev

    
class _NewEmptyTensorOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, new_shape):
        ctx.shape = x.shape
        return x.new_empty(new_shape)

    @staticmethod
    def backward(ctx, grad):
        shape = ctx.shape
        return _NewEmptyTensorOp.apply(grad, shape), None

def requires_grad(model, flag=True):
    for name, p in model.named_parameters():
        p.requires_grad = flag

def tensor_mean(input):
    if torch.is_tensor(input):
        return input.mean()
    elif isinstance(input, list):
        mean_list = [tensor_mean(l) for l in input]
        return sum(mean_list) / len(mean_list)
    else:
        return NotImplemented()

def d_r1_loss(real_pred, real_img):
    grad_real, = autograd.grad(outputs = tensor_mean(real_pred),
                               inputs=real_img,
                               create_graph=True)#, allow_unused=True

    grad_penalty = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()
    return grad_penalty

def d_r1_loss_org(real_pred, real_img):
    grad_real, = autograd.grad(outputs=real_pred.sum(),
                               inputs=real_img,
                               create_graph=True)#, allow_unused=True

    grad_penalty = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()
    return grad_penalty

def d_r1_loss_eg3d(real_pred, real_img_list):
    r1_grads = autograd.grad(outputs = [real_pred.sum()],
                               inputs = real_img_list,
                               create_graph=True)#, allow_unused=True
    r1_0 = r1_grads[0]    
    r1_penalty = r1_0.square().sum([1,2,3]) 
    
    if len(real_img_list) > 1:
        r1_1 = r1_grads[1]
        r1_penalty += r1_1.square().sum([1,2,3])

    #r1_penalty = grad_real.square().sum([1,2,3])
    #grad_penalty = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()
    return r1_penalty

def d_logistic_loss(real_pred, fake_pred, tmp = 1.0):

    real_pred = tensor_mean(real_pred)
    fake_pred = tensor_mean(fake_pred)

    real_loss = F.softplus(-real_pred * tmp)
    fake_loss = F.softplus(fake_pred)
    return real_loss.mean(), fake_loss.mean()

def interpolate(input, size=None, scale_factor=None, mode="nearest", align_corners=None):
    """
    A wrapper around :func:`torch.nn.functional.interpolate` to support zero-size tensor.
    """
    if TORCH_VERSION > (1, 4) or input.numel() > 0:
        return torch.nn.functional.interpolate(
            input, size, scale_factor, mode, align_corners=align_corners
        )

    def _check_size_scale_factor(dim):
        if size is None and scale_factor is None:
            raise ValueError("either size or scale_factor should be defined")
        if size is not None and scale_factor is not None:
            raise ValueError("only one of size or scale_factor should be defined")
        if (
            scale_factor is not None
            and isinstance(scale_factor, tuple)
            and len(scale_factor) != dim
        ):
            raise ValueError(
                "scale_factor shape must match input shape. "
                "Input is {}D, scale_factor size is {}".format(dim, len(scale_factor))
            )

    def _output_size(dim):
        _check_size_scale_factor(dim)
        if size is not None:
            return size
        scale_factors = _ntuple(dim)(scale_factor)
        # math.floor might return float in py2.7
        return [int(math.floor(input.size(i + 2) * scale_factors[i])) for i in range(dim)]

    output_shape = tuple(_output_size(2))
    output_shape = input.shape[:-2] + output_shape
    return _NewEmptyTensorOp.apply(input, output_shape)

def return_style_conv_upsample(in_channel, out_channel, style_dim = 0):
    blur_kernel = [1, 3, 3, 1]
    convs = []
    
    from .stylegan import StyledConv

    convs.append(
        StyledConv(
            in_channel,
            out_channel,
            3,
            style_dim,
            upsample=True,
            blur_kernel=blur_kernel,
        )
    )

    convs.append(
        StyledConv(
            out_channel, out_channel, 3, style_dim, blur_kernel=blur_kernel
        )
    )
    return convs


class ResnetBlock_Connect(nn.Module):
    
    def __init__(self, in_dim, out_dim, padding_type, norm_layer, activation=nn.ReLU(True), use_dropout=False):
        super(ResnetBlock_Connect, self).__init__()
        self.conv_block = self.build_conv_block(in_dim, out_dim, padding_type, norm_layer, activation, use_dropout)

    def build_conv_block(self, in_dim, out_dim, padding_type, norm_layer, activation, use_dropout):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=p),
                       norm_layer(out_dim),
                       activation]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=p),
                       norm_layer(out_dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
                
        out = self.conv_block(x)
                
        
        return out  
        
# Define a resnet block
class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, activation=nn.ReLU(True), use_dropout=False):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, activation, use_dropout)

    def build_conv_block(self, dim, padding_type, norm_layer, activation, use_dropout):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim),
                       activation]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
                
        out = x + self.conv_block(x)
                
        return out

class Encoder(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=32, n_downsampling=4, norm_layer=nn.BatchNorm2d):
        super(Encoder, self).__init__()        
        self.output_nc = output_nc        

        model = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0), 
                 norm_layer(ngf), nn.ReLU(True)]             
        ### downsample
        for i in range(n_downsampling):
            mult = 2**i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1),
                      norm_layer(ngf * mult * 2), nn.ReLU(True)]

        ### upsample         
        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1, output_padding=1),
                       norm_layer(int(ngf * mult / 2)), nn.ReLU(True)]        

        model += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh()]
        self.model = nn.Sequential(*model) 

    def forward(self, input, inst):
        outputs = self.model(input)

        # instance-wise average pooling
        outputs_mean = outputs.clone()
        inst_list = np.unique(inst.cpu().numpy().astype(int))        
        for i in inst_list:
            for b in range(input.size()[0]):
                indices = (inst[b:b+1] == int(i)).nonzero() # n x 4            
                for j in range(self.output_nc):
                    output_ins = outputs[indices[:,0] + b, indices[:,1] + j, indices[:,2], indices[:,3]]                    
                    mean_feat = torch.mean(output_ins).expand_as(output_ins)                                        
                    outputs_mean[indices[:,0] + b, indices[:,1] + j, indices[:,2], indices[:,3]] = mean_feat                       
        return outputs_mean

class MultiscaleDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, 
                 use_sigmoid=False, num_D=3, getIntermFeat=False):
        super(MultiscaleDiscriminator, self).__init__()
        self.num_D = num_D
        self.n_layers = n_layers
        self.getIntermFeat = getIntermFeat
     
        for i in range(num_D):
            netD = NLayerDiscriminator(input_nc, ndf, n_layers, norm_layer, use_sigmoid, getIntermFeat)
            if getIntermFeat:                                
                for j in range(n_layers+2):
                    setattr(self, 'scale'+str(i)+'_layer'+str(j), getattr(netD, 'model'+str(j)))                                   
            else:
                setattr(self, 'layer'+str(i), netD.model)

        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

    def singleD_forward(self, model, input):
        if self.getIntermFeat:
            result = [input]
            for i in range(len(model)):
                result.append(model[i](result[-1]))
            return result[1:]
        else:
            return [model(input)]

    def forward(self, input):        
        num_D = self.num_D
        result = []
        input_downsampled = input
        for i in range(num_D):
            if self.getIntermFeat:
                model = [getattr(self, 'scale'+str(num_D-1-i)+'_layer'+str(j)) for j in range(self.n_layers+2)]
            else:
                model = getattr(self, 'layer'+str(num_D-1-i))
            result.append(self.singleD_forward(model, input_downsampled))
            if i != (num_D-1):
                input_downsampled = self.downsample(input_downsampled)
        return result
        
# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False, getIntermFeat=False):
        super(NLayerDiscriminator, self).__init__()
        self.getIntermFeat = getIntermFeat
        self.n_layers = n_layers

        kw = 4
        padw = int(np.ceil((kw-1.0)/2))
        sequence = [[nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]]

        nf = ndf
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            sequence += [[
                nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw),
                norm_layer(nf), nn.LeakyReLU(0.2, True)
            ]]

        nf_prev = nf
        nf = min(nf * 2, 512)
        sequence += [[
            nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw),
            norm_layer(nf),
            nn.LeakyReLU(0.2, True)
        ]]

        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]

        if use_sigmoid:
            sequence += [[nn.Sigmoid()]]

        if getIntermFeat:
            for n in range(len(sequence)):
                setattr(self, 'model'+str(n), nn.Sequential(*sequence[n]))
        else:
            sequence_stream = []
            for n in range(len(sequence)):
                sequence_stream += sequence[n]
            self.model = nn.Sequential(*sequence_stream)

    def forward(self, input):
        if self.getIntermFeat:
            res = [input]
            for n in range(self.n_layers+2):
                model = getattr(self, 'model'+str(n))
                res.append(model(res[-1]))
            return res[1:]
        else:
            return self.model(input)        

from torchvision import models
class Vgg19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)        
        h_relu3 = self.slice3(h_relu2)        
        h_relu4 = self.slice4(h_relu3)        
        h_relu5 = self.slice5(h_relu4)                
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out
