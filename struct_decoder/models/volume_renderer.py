import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import numpy as np

import sys
sys.path.append("..")
from Engine3D.th_utils.io.prints import *

from torch_utils.ops import bias_act

# Basic SIREN fully connected layer
class LinearLayer(nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, bias_init=0, std_init=1, freq_init=False, is_first=False):
        super().__init__()
        if is_first:
            self.weight = nn.Parameter(torch.empty(out_dim, in_dim).uniform_(-1 / in_dim, 1 / in_dim))
        elif freq_init:
            self.weight = nn.Parameter(torch.empty(out_dim, in_dim).uniform_(-np.sqrt(6 / in_dim) / 25, np.sqrt(6 / in_dim) / 25))
        else:
            self.weight = nn.Parameter(0.25 * nn.init.kaiming_normal_(torch.randn(out_dim, in_dim), a=0.2, mode='fan_in', nonlinearity='leaky_relu'))

        self.bias = nn.Parameter(nn.init.uniform_(torch.empty(out_dim), a=-np.sqrt(1/in_dim), b=np.sqrt(1/in_dim)))

        self.bias_init = bias_init
        self.std_init = std_init

    def forward(self, input):
        out = self.std_init * F.linear(input, self.weight, bias=self.bias) + self.bias_init

        return out

# Siren layer with frequency modulation and offset
class FiLMSiren(nn.Module):
    def __init__(self, in_channel, out_channel, style_dim, is_first=False):
        super().__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel

        if is_first:
            self.weight = nn.Parameter(torch.empty(out_channel, in_channel).uniform_(-1 / 3, 1 / 3))
        else:
            self.weight = nn.Parameter(torch.empty(out_channel, in_channel).uniform_(-np.sqrt(6 / in_channel) / 25, np.sqrt(6 / in_channel) / 25))

        self.bias = nn.Parameter(nn.Parameter(nn.init.uniform_(torch.empty(out_channel), a=-np.sqrt(1/in_channel), b=np.sqrt(1/in_channel))))
        self.activation = torch.sin

        self.gamma = LinearLayer(style_dim, out_channel, bias_init=30, std_init=15)
        self.beta = LinearLayer(style_dim, out_channel, bias_init=0, std_init=0.25)

    def forward_with_gamma_beta(self, input, gamma, beta):
        out = F.linear(input, self.weight, bias=self.bias)
        out = self.activation(gamma * out + beta)

        return out

    def forward(self, input, style):

        batch = style.shape[0]
        
        out = F.linear(input, self.weight, bias=self.bias)

        gamma = self.gamma(style).view(batch, -1, out.shape[-1])
        beta = self.beta(style).view(batch, -1, out.shape[-1])

        out = self.activation(gamma * out + beta)

        return out


# Siren Generator Model
class SirenGenerator(nn.Module):
    def __init__(self, D=8, W=256, style_dim=256, input_ch=3, input_ch_views=3, output_ch=4,
                 output_features=True):
        super(SirenGenerator, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.style_dim = style_dim
        self.output_features = output_features

        self.pts_linears = nn.ModuleList(
            [FiLMSiren(3, W, style_dim=style_dim, is_first=True)] + \
            [FiLMSiren(W, W, style_dim=style_dim) for i in range(D-1)])

        self.views_linears = FiLMSiren(input_ch_views + W, W,
                                       style_dim=style_dim)
        self.rgb_linear = LinearLayer(W, output_ch - 1, freq_init=True)
        self.sigma_linear = LinearLayer(W, 1, freq_init=True)

    def mapping_network(self, style):
        batch, _ = style.shape
        gamma_list = [
            self.pts_linears[i].gamma(style).view(batch, 1, -1)
            for i in range(len(self.pts_linears))
        ] + [self.views_linears.gamma(style).view(batch, 1, -1),]
        beta_list = [
            self.pts_linears[i].beta(style).view(batch, 1, -1)
            for i in range(len(self.pts_linears))
        ] + [self.views_linears.beta(style).view(batch, 1, -1),]
        return torch.stack(gamma_list, 0), torch.stack(beta_list, 0)

    def forward_with_gamma_beta(self, x, gamma_list, beta_list):
        input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)
        mlp_out = input_pts.contiguous()
        for i in range(len(self.pts_linears)):
            mlp_out = self.pts_linears[i].forward_with_gamma_beta(mlp_out, gamma_list[i], beta_list[i])

        sdf = self.sigma_linear(mlp_out)

        mlp_out = torch.cat([mlp_out, input_views], -1)
        out_features = self.views_linears.forward_with_gamma_beta(mlp_out, gamma_list[-1], beta_list[-1])
        rgb = self.rgb_linear(out_features)

        outputs = torch.cat([rgb, sdf], -1)
        if self.output_features:
            outputs = torch.cat([outputs, out_features], -1)

        return outputs

    def forward(self, x, styles):

        input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)

        mlp_out = input_pts.contiguous()
        for i in range(len(self.pts_linears)):
            mlp_out = self.pts_linears[i](mlp_out, styles)


        sdf = self.sigma_linear(mlp_out)

        #printg(x.shape, styles.shape, mlp_out.shape, input_views.shape)
        mlp_out = torch.cat([mlp_out, input_views], -1)
        out_features = self.views_linears(mlp_out, styles)
        rgb = self.rgb_linear(out_features)

        outputs = torch.cat([rgb, sdf], -1)
        if self.output_features:
            outputs = torch.cat([outputs, out_features], -1)

        return outputs

# Siren Generator Model
class SirenGeneratorVoxel(nn.Module):
    def __init__(self, D=8, W=256, style_dim=256, input_ch=3, input_ch_views=3, output_ch=4,
                 output_features=True, is_style = True, use_pose_encoding = True, use_net = True, camera_cond = False):
        super(SirenGeneratorVoxel, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.style_dim = style_dim
        self.output_features = output_features

        self.is_style = is_style
        self.use_pose_encoding = use_pose_encoding
        self.use_net = use_net
        if use_pose_encoding:
            if is_style:
                self.pts_linears = nn.ModuleList(
                    [FiLMSiren(3, W, style_dim=style_dim, is_first=True)] + \
                    [FiLMSiren(W, W, style_dim=style_dim) for i in range(D-1)])
            else:
                
                self.pts_linears = nn.ModuleList(\
                    [LinearLayer(3 + style_dim, W, freq_init=True)] + \
                    [LinearLayer(W, W) for i in range(D-1)]
                )

            self.camera_cond = camera_cond
            if camera_cond:
                self.views_linears = FiLMSiren(input_ch_views + W, W, style_dim=style_dim)
            #cat style with pos feature.
            self.rgb_linear = LinearLayer(W, output_ch - 1, freq_init=True)
            self.sigma_linear = LinearLayer(W, 1, freq_init=True)
        elif not use_net:
            pass
        else:
            self.rgb_linear = LinearLayer(style_dim, output_ch - 1, freq_init=True)
            self.sigma_linear = LinearLayer(style_dim, 1, freq_init=True)

    def forward(self, x, styles):

        input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)

        if self.use_pose_encoding:
            mlp_out = input_pts.contiguous()
            if self.is_style:
                for i in range(len(self.pts_linears)):
                    mlp_out = self.pts_linears[i](mlp_out, styles)
            else:
                mlp_out = torch.cat([mlp_out, styles], -1)#.permute(0,2,1)
                for i in range(len(self.pts_linears)):
                    mlp_out = self.pts_linears[i](mlp_out)


        elif not self.use_net:
            a = nn.Tanh()
            sdf = a(styles[..., [-1]])
            rgb = styles[..., :-1]
            return torch.cat([rgb, sdf], -1)
        else:
            mlp_out = styles

        sdf = self.sigma_linear(mlp_out)
        rgb = self.rgb_linear(mlp_out)

        outputs = torch.cat([rgb, sdf], -1)
        return outputs

# Siren Generator Model
class SirenGeneratorNoView(nn.Module):
    def __init__(self, D=8, W=256, style_dim=256, input_ch=3, input_ch_views=3, output_ch=4,
                 output_features=True):
        super(SirenGenerator, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.style_dim = style_dim
        self.output_features = output_features

        self.pts_linears = nn.ModuleList(
            [FiLMSiren(3, W, style_dim=style_dim, is_first=True)] + \
            [FiLMSiren(W, W, style_dim=style_dim) for i in range(D-1)])

        self.rgb_linear = LinearLayer(W, output_ch - 1, freq_init=True)
        self.sigma_linear = LinearLayer(W, 1, freq_init=True)

    def mapping_network(self, style):
        batch, _ = style.shape
        gamma_list = [
            self.pts_linears[i].gamma(style).view(batch, 1, -1)
            for i in range(len(self.pts_linears))
        ] + [self.views_linears.gamma(style).view(batch, 1, -1),]
        beta_list = [
            self.pts_linears[i].beta(style).view(batch, 1, -1)
            for i in range(len(self.pts_linears))
        ] + [self.views_linears.beta(style).view(batch, 1, -1),]
        return torch.stack(gamma_list, 0), torch.stack(beta_list, 0)

    def forward_with_gamma_beta(self, x, gamma_list, beta_list):
        input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)
        mlp_out = input_pts.contiguous()
        for i in range(len(self.pts_linears)):
            mlp_out = self.pts_linears[i].forward_with_gamma_beta(mlp_out, gamma_list[i], beta_list[i])

        sdf = self.sigma_linear(mlp_out)

        mlp_out = torch.cat([mlp_out, input_views], -1)
        out_features = self.views_linears.forward_with_gamma_beta(mlp_out, gamma_list[-1], beta_list[-1])
        rgb = self.rgb_linear(out_features)

        outputs = torch.cat([rgb, sdf], -1)
        if self.output_features:
            outputs = torch.cat([outputs, out_features], -1)

        return outputs

    def forward(self, x, styles):

        input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)

        mlp_out = input_pts.contiguous()
        for i in range(len(self.pts_linears)):
            mlp_out = self.pts_linears[i](mlp_out, styles)


        sdf = self.sigma_linear(mlp_out)
        rgb = self.rgb_linear(mlp_out)

        outputs = torch.cat([rgb, sdf], -1)
        return outputs




class FullyConnectedLayer(torch.nn.Module):
    def __init__(self,
        in_features,                # Number of input features.
        out_features,               # Number of output features.
        bias            = True,     # Apply additive bias before the activation function?
        activation      = 'linear', # Activation function: 'relu', 'lrelu', etc.
        lr_multiplier   = 1,        # Learning rate multiplier.
        bias_init       = 0,        # Initial value for the additive bias.
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.activation = activation
        self.weight = torch.nn.Parameter(torch.randn([out_features, in_features]) / lr_multiplier)
        self.bias = torch.nn.Parameter(torch.full([out_features], np.float32(bias_init))) if bias else None
        self.weight_gain = lr_multiplier / np.sqrt(in_features)
        self.bias_gain = lr_multiplier

    def forward(self, x):
        w = self.weight.to(x.dtype) * self.weight_gain
        b = self.bias
        if b is not None:
            b = b.to(x.dtype)
            if self.bias_gain != 1:
                b = b * self.bias_gain

        if self.activation == 'linear' and b is not None:
            x = torch.addmm(b.unsqueeze(0), x, w.t())
        else:
            x = x.matmul(w.t())
            x = bias_act.bias_act(x, b, act=self.activation)
        return x

    def extra_repr(self):
        return f'in_features={self.in_features:d}, out_features={self.out_features:d}, activation={self.activation:s}'


class OSGDecoder(torch.nn.Module):
    def __init__(self, n_features, options):
        super().__init__()
        self.hidden_dim = 64

        self.net = torch.nn.Sequential(
            FullyConnectedLayer(n_features, self.hidden_dim, lr_multiplier=options['decoder_lr_mul']),
            torch.nn.Softplus(),
            FullyConnectedLayer(self.hidden_dim, 1 + options['decoder_output_dim'], lr_multiplier=options['decoder_lr_mul'])
        )
        
    def forward(self, sampled_features, ray_directions):
        # Aggregate features
        sampled_features = sampled_features.mean(1)
        x = sampled_features

        N, M, C = x.shape
        x = x.view(N*M, C)

        x = self.net(x)
        x = x.view(N, M, -1)
        rgb = torch.sigmoid(x[..., 1:])*(1 + 2*0.001) - 0.001 # Uses sigmoid clamping from MipNeRF
        sigma = x[..., 0:1]
        return {'rgb': rgb, 'sigma': sigma}
    

        planes = planes.view(len(planes), 3, 32, planes.shape[-2], planes.shape[-1])

def project_onto_planes(planes, coordinates):
    """
    Does a projection of a 3D point onto a batch of 2D planes,
    returning 2D plane coordinates.

    Takes plane axes of shape n_planes, 3, 3
    # Takes coordinates of shape N, M, 3
    # returns projections of shape N*n_planes, M, 2
    """
    N, M, C = coordinates.shape
    n_planes, _, _ = planes.shape
    coordinates = coordinates.unsqueeze(1).expand(-1, n_planes, -1, -1).reshape(N*n_planes, M, 3)
    inv_planes = torch.linalg.inv(planes).unsqueeze(0).expand(N, -1, -1, -1).reshape(N*n_planes, 3, 3)
    projections = torch.bmm(coordinates, inv_planes)
    return projections[..., :2]

def sample_from_planes(plane_axes, plane_features, coordinates, mode='bilinear', padding_mode='zeros', box_warp=None):
    assert padding_mode == 'zeros'
    N, n_planes, C, H, W = plane_features.shape
    _, M, _ = coordinates.shape
    plane_features = plane_features.view(N*n_planes, C, H, W)

    coordinates = (2/box_warp) * coordinates # TODO: add specific box bounds

    projected_coordinates = project_onto_planes(plane_axes, coordinates).unsqueeze(1)
    output_features = torch.nn.functional.grid_sample(plane_features, projected_coordinates.float(), mode=mode, padding_mode=padding_mode, align_corners=False).permute(0, 3, 2, 1).reshape(N, n_planes, M, C)
    return output_features

# Siren Generator Model
class VoxelFeatureNet(nn.Module):
    def __init__(self, input_ch = 0, output_ch=4, hidden = 64, D = 1,
                 output_features=True):
        
        super(VoxelFeatureNet, self).__init__()
        
        self.input_ch = input_ch
        self.output_features = output_features

        self.feature_linear = [FullyConnectedLayer(input_ch, hidden), torch.nn.Softplus()]
        for i in range(D - 1):
            self.feature_linear += [FullyConnectedLayer(hidden, hidden), torch.nn.Softplus()]
        self.feature_linear = torch.nn.Sequential(*self.feature_linear) 
        
        self.rgb_linear = LinearLayer(input_ch, output_ch - 1, freq_init=True)
        self.sigma_linear = LinearLayer(input_ch, 1, freq_init=True)

    def forward(self, x, styles):
        if x.shape[-1] > 3:
            input_pts, input_views = torch.split(x, [3, x.shape[-1] - 3], dim=-1)
        else:
            input_pts = x

        xy = project_onto_planes(styles, input_pts)
        feat = sample_from_planes()

        sampled_features = sampled_features.mean(1)
        out = self.feature_linear(feat)

        sdf = self.sigma_linear(out)
        rgb = self.rgb_linear(out)

        outputs = torch.cat([rgb, sdf], -1)
        return outputs



# concatentate input with style
class SDFDecoder(nn.Module):
    def __init__(self, D=8, W=256, style_dim=256, input_ch=3, input_ch_views=3, output_ch=4,
                 output_features=True):
        super(SDFDecoder, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.style_dim = style_dim
        self.output_features = output_features

        self.actvn = nn.LeakyReLU()
        self.pts_linears = [nn.Conv1d(3 + style_dim, W, 1), self.actvn] 
        for i in range(D-1):
            self.pts_linears += [nn.Conv1d(W, W, 1), self.actvn]
        self.pts_linears = torch.nn.Sequential(*self.pts_linears)

        # self.views_linears = FiLMSiren(input_ch_views + W, W, style_dim=style_dim)
        self.views_linears = torch.nn.Sequential(*[nn.Conv1d(input_ch_views + W + style_dim, W, 1), self.actvn])

        self.rgb_linear = LinearLayer(W, 3, freq_init=True)
        self.sigma_linear = LinearLayer(W, 1, freq_init=True)

    def mapping_network(self, style):
        batch, _ = style.shape
        gamma_list = [
            self.pts_linears[i].gamma(style).view(batch, 1, -1)
            for i in range(len(self.pts_linears))
        ] + [self.views_linears.gamma(style).view(batch, 1, -1),]
        beta_list = [
            self.pts_linears[i].beta(style).view(batch, 1, -1)
            for i in range(len(self.pts_linears))
        ] + [self.views_linears.beta(style).view(batch, 1, -1),]
        return torch.stack(gamma_list, 0), torch.stack(beta_list, 0)

    def forward_with_gamma_beta(self, x, gamma_list, beta_list):
        input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)
        mlp_out = input_pts.contiguous()
        for i in range(len(self.pts_linears)):
            mlp_out = self.pts_linears[i].forward_with_gamma_beta(mlp_out, gamma_list[i], beta_list[i])

        sdf = self.sigma_linear(mlp_out)

        mlp_out = torch.cat([mlp_out, input_views], -1)
        out_features = self.views_linears.forward_with_gamma_beta(mlp_out, gamma_list[-1], beta_list[-1])
        rgb = self.rgb_linear(out_features)

        outputs = torch.cat([rgb, sdf], -1)
        if self.output_features:
            outputs = torch.cat([outputs, out_features], -1)

        return outputs

    def forward(self, x, styles):
        #printy("*** ", x.shape, styles.shape)

        input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)


        input_pts = torch.cat((input_pts, styles), -1)
        mlp_out = input_pts.contiguous().permute(0, 2, 1) #B, P, C to B,C,P
        for i in range(len(self.pts_linears)):
            mlp_out = self.pts_linears[i](mlp_out)

        mlp_out = mlp_out.permute(0,2,1) #to B, P, C
        sdf = self.sigma_linear(mlp_out)

        #print(x.shape, styles.shape, mlp_out.shape, input_views.shape)
        mlp_out = torch.cat([mlp_out, input_views, styles], -1).permute(0,2,1)
        out_features = self.views_linears(mlp_out).permute(0,2,1)
        rgb = self.rgb_linear(out_features)

        outputs = torch.cat([rgb, sdf], -1)
        if self.output_features:
            outputs = torch.cat([outputs, out_features], -1)

        return outputs


# Siren layer with frequency modulation and offset per pts
class FiLMSirenPP_notuse(nn.Module):
    def __init__(self, in_channel, out_channel, style_dim, is_first=False):
        super().__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel

        if is_first:
            self.weight = nn.Parameter(torch.empty(out_channel, in_channel).uniform_(-1 / 3, 1 / 3))
        else:
            self.weight = nn.Parameter(torch.empty(out_channel, in_channel).uniform_(-np.sqrt(6 / in_channel) / 25, np.sqrt(6 / in_channel) / 25))

        self.bias = nn.Parameter(nn.Parameter(nn.init.uniform_(torch.empty(out_channel), a=-np.sqrt(1/in_channel), b=np.sqrt(1/in_channel))))
        self.activation = torch.sin

        self.gamma = LinearLayer(style_dim, out_channel, bias_init=30, std_init=15)
        self.beta = LinearLayer(style_dim, out_channel, bias_init=0, std_init=0.25)

    def forward_with_gamma_beta(self, input, gamma, beta):
        out = F.linear(input, self.weight, bias=self.bias)
        out = self.activation(gamma * out + beta)

        return out

    def forward(self, input, style):

        batch, features = style.shape
        
        out = F.linear(input, self.weight, bias=self.bias)
        gamma = self.gamma(style).view(batch, 1, -1)
        beta = self.beta(style).view(batch, 1, -1)

        out = self.activation(gamma * out + beta)

        return out

# concatentate input with style
class SDFDecoderStyle_notuse(nn.Module):
    def __init__(self, D=8, W=256, style_dim=256, input_ch=3, input_ch_views=3, output_ch=4,
                 output_features=True):
        super(SDFDecoderStyle, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.style_dim = style_dim
        self.output_features = output_features

        self.pts_linears = nn.ModuleList(
            [FiLMSirenPP(3, W, style_dim=style_dim, is_first=True)] + \
            [FiLMSirenPP(W, W, style_dim=style_dim) for i in range(D-1)])

        self.views_linears = FiLMSirenPP(input_ch_views + W, W,
                                       style_dim=style_dim)
        
        self.rgb_linear = LinearLayer(W, 3, freq_init=True)
        self.sigma_linear = LinearLayer(W, 1, freq_init=True)

    def forward(self, x, styles):
        printy("*** ", x.shape, styles.shape)

        input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)
        mlp_out = input_pts.contiguous()
        for i in range(len(self.pts_linears)):
            mlp_out = self.pts_linears[i](mlp_out, styles)


        sdf = self.sigma_linear(mlp_out)

        #print(x.shape, styles.shape, mlp_out.shape, input_views.shape)
        mlp_out = torch.cat([mlp_out, input_views], -1)
        out_features = self.views_linears(mlp_out, styles)
        rgb = self.rgb_linear(out_features)

        outputs = torch.cat([rgb, sdf], -1)
        if self.output_features:
            outputs = torch.cat([outputs, out_features], -1)

        return outputs



# Full volume renderer
class VolumeFeatureRenderer(nn.Module):
    def __init__(self, opt, style_dim=256, out_im_res=64, mode='train'):
        super().__init__()
        self.test = mode != 'train'
        self.perturb = opt.perturb
        self.offset_sampling = not opt.no_offset_sampling # Stratified sampling used otherwise
        self.N_samples = opt.N_samples
        self.raw_noise_std = opt.raw_noise_std
        self.return_xyz = opt.return_xyz
        self.return_sdf = opt.return_sdf
        self.static_viewdirs = opt.static_viewdirs
        self.z_normalize = not opt.no_z_normalize
        self.out_im_res = out_im_res
        self.force_background = opt.force_background
        self.with_sdf = not opt.no_sdf
        if 'no_features_output' in opt.keys():
            self.output_features = False
        else:
            self.output_features = True

        if self.with_sdf:
            self.sigmoid_beta = nn.Parameter(0.1 * torch.ones(1))

        # create meshgrid to generate rays
        i, j = torch.meshgrid(torch.linspace(0.5, self.out_im_res - 0.5, self.out_im_res),
                              torch.linspace(0.5, self.out_im_res - 0.5, self.out_im_res))

        self.register_buffer('i', i.t().unsqueeze(0), persistent=False)
        self.register_buffer('j', j.t().unsqueeze(0), persistent=False)

        # create integration values
        if self.offset_sampling:
            t_vals = torch.linspace(0., 1.-1/self.N_samples, steps=self.N_samples).view(1,1,1,-1)
        else: # Original NeRF Stratified sampling
            t_vals = torch.linspace(0., 1., steps=self.N_samples).view(1,1,1,-1)

        self.register_buffer('t_vals', t_vals, persistent=False)
        self.register_buffer('inf', torch.Tensor([1e10]), persistent=False)
        self.register_buffer('zero_idx', torch.LongTensor([0]), persistent=False)

        if self.test:
            self.perturb = False
            self.raw_noise_std = 0.

        self.channel_dim = -1
        self.samples_dim = 3
        self.input_ch = 3
        self.input_ch_views = 3
        self.feature_out_size = opt.width




        # set Siren Generator model
        self.network = SirenGenerator(D=opt.depth, W=opt.width, style_dim=style_dim, input_ch=self.input_ch,
                                      output_ch=4, input_ch_views=self.input_ch_views,
                                      output_features=self.output_features)

    def get_rays(self, focal, c2w):
        dirs = torch.stack([(self.i - self.out_im_res * .5) / focal,
                            -(self.j - self.out_im_res * .5) / focal,
                            -torch.ones_like(self.i).expand(focal.shape[0],self.out_im_res, self.out_im_res)], -1)

        # Rotate ray directions from camera frame to the world frame
        rays_d = torch.sum(dirs[..., None, :] * c2w[:,None,None,:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]

        # Translate camera frame's origin to the world frame. It is the origin of all rays.
        rays_o = c2w[:,None,None,:3,-1].expand(rays_d.shape)
        if self.static_viewdirs:
            viewdirs = dirs
        else:
            viewdirs = rays_d

        return rays_o, rays_d, viewdirs

    def get_eikonal_term(self, pts, sdf):
        eikonal_term = autograd.grad(outputs=sdf, inputs=pts,
                                     grad_outputs=torch.ones_like(sdf),
                                     create_graph=True)[0]

        return eikonal_term

    def sdf_activation(self, input):
        sigma = torch.sigmoid(input / self.sigmoid_beta) / self.sigmoid_beta

        return sigma

    def volume_integration(self, raw, z_vals, rays_d, pts, return_eikonal=False):
        dists = z_vals[...,1:] - z_vals[...,:-1]
        rays_d_norm = torch.norm(rays_d.unsqueeze(self.samples_dim), dim=self.channel_dim)
        # dists still has 4 dimensions here instead of 5, hence, in this case samples dim is actually the channel dim
        dists = torch.cat([dists, self.inf.expand(rays_d_norm.shape)], self.channel_dim)  # [N_rays, N_samples]
        dists = dists * rays_d_norm

        # If sdf modeling is off, the sdf variable stores the
        # pre-integration raw sigma MLP outputs.
        if self.output_features:
            rgb, sdf, features = torch.split(raw, [3, 1, self.feature_out_size], dim=self.channel_dim)
        else:
            rgb, sdf = torch.split(raw, [3, 1], dim=self.channel_dim)

        noise = 0.
        if self.raw_noise_std > 0.:
            noise = torch.randn_like(sdf) * self.raw_noise_std

        if self.with_sdf:
            sigma = self.sdf_activation(-sdf)

            if return_eikonal:
                eikonal_term = self.get_eikonal_term(pts, sdf)
            else:
                eikonal_term = None

            sigma = 1 - torch.exp(-sigma * dists.unsqueeze(self.channel_dim))
        else:
            sigma = sdf
            eikonal_term = None

            sigma = 1 - torch.exp(-F.softplus(sigma + noise) * dists.unsqueeze(self.channel_dim))

        visibility = torch.cumprod(torch.cat([torch.ones_like(torch.index_select(sigma, self.samples_dim, self.zero_idx)),
                                              1.-sigma + 1e-10], self.samples_dim), self.samples_dim)
        visibility = visibility[...,:-1,:]
        weights = sigma * visibility


        if self.return_sdf:
            sdf_out = sdf
        else:
            sdf_out = None

        if self.force_background:
            weights[...,-1,:] = 1 - weights[...,:-1,:].sum(self.samples_dim)

        rgb_map = -1 + 2 * torch.sum(weights * torch.sigmoid(rgb), self.samples_dim)  # switch to [-1,1] value range

        if self.output_features:
            feature_map = torch.sum(weights * features, self.samples_dim)
        else:
            feature_map = None

        # Return surface point cloud in world coordinates.
        # This is used to generate the depth maps visualizations.
        # We use world coordinates to avoid transformation errors between
        # surface renderings from different viewpoints.
        if self.return_xyz:
            xyz = torch.sum(weights * pts, self.samples_dim)
            mask = weights[...,-1,:] # background probability map
        else:
            xyz = None
            mask = None

        return rgb_map, feature_map, sdf_out, mask, xyz, eikonal_term

    def run_network(self, inputs, viewdirs, styles=None):
        input_dirs = viewdirs.unsqueeze(self.samples_dim).expand(inputs.shape)
        net_inputs = torch.cat([inputs, input_dirs], self.channel_dim)
        outputs = self.network(net_inputs, styles=styles)

        return outputs

    def render_rays(self, ray_batch, styles=None, return_eikonal=False):
        batch, h, w, _ = ray_batch.shape
        split_pattern = [3, 3, 2]
        if ray_batch.shape[-1] > 8:
            split_pattern += [3]
            rays_o, rays_d, bounds, viewdirs = torch.split(ray_batch, split_pattern, dim=self.channel_dim)
        else:
            rays_o, rays_d, bounds = torch.split(ray_batch, split_pattern, dim=self.channel_dim)
            viewdirs = None

        near, far = torch.split(bounds, [1, 1], dim=self.channel_dim)
        z_vals = near * (1.-self.t_vals) + far * (self.t_vals)

        if self.perturb > 0.:
            if self.offset_sampling:
                # random offset samples
                upper = torch.cat([z_vals[...,1:], far], -1)
                lower = z_vals.detach()
                t_rand = torch.rand(batch, h, w).unsqueeze(self.channel_dim).to(z_vals.device)
            else:
                # get intervals between samples
                mids = .5 * (z_vals[...,1:] + z_vals[...,:-1])
                upper = torch.cat([mids, z_vals[...,-1:]], -1)
                lower = torch.cat([z_vals[...,:1], mids], -1)
                # stratified samples in those intervals
                t_rand = torch.rand(z_vals.shape).to(z_vals.device)

            z_vals = lower + (upper - lower) * t_rand

        pts = rays_o.unsqueeze(self.samples_dim) + rays_d.unsqueeze(self.samples_dim) * z_vals.unsqueeze(self.channel_dim)

        if return_eikonal:
            pts.requires_grad = True

        if self.z_normalize:
            normalized_pts = pts * 2 / ((far - near).unsqueeze(self.samples_dim))
        else:
            normalized_pts = pts

        raw = self.run_network(normalized_pts, viewdirs, styles=styles)
        rgb_map, features, sdf, mask, xyz, eikonal_term = self.volume_integration(raw, z_vals, rays_d, pts, return_eikonal=return_eikonal)

        return rgb_map, features, sdf, mask, xyz, eikonal_term

    def render(self, focal, c2w, near, far, styles, c2w_staticcam=None, return_eikonal=False):
        rays_o, rays_d, viewdirs = self.get_rays(focal, c2w)
        viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)

        # Create ray batch
        near = near.unsqueeze(-1) * torch.ones_like(rays_d[...,:1])
        far = far.unsqueeze(-1) * torch.ones_like(rays_d[...,:1])
        rays = torch.cat([rays_o, rays_d, near, far], -1)
        rays = torch.cat([rays, viewdirs], -1)
        rays = rays.float()
        rgb, features, sdf, mask, xyz, eikonal_term = self.render_rays(rays, styles=styles, return_eikonal=return_eikonal)

        return rgb, features, sdf, mask, xyz, eikonal_term

    def mlp_init_pass(self, cam_poses, focal, near, far, styles=None):
        rays_o, rays_d, viewdirs = self.get_rays(focal, cam_poses)
        viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)

        near = near.unsqueeze(-1) * torch.ones_like(rays_d[...,:1])
        far = far.unsqueeze(-1) * torch.ones_like(rays_d[...,:1])
        z_vals = near * (1.-self.t_vals) + far * (self.t_vals)

        # get intervals between samples
        mids = .5 * (z_vals[...,1:] + z_vals[...,:-1])
        upper = torch.cat([mids, z_vals[...,-1:]], -1)
        lower = torch.cat([z_vals[...,:1], mids], -1)
        # stratified samples in those intervals
        t_rand = torch.rand(z_vals.shape).to(z_vals.device)

        z_vals = lower + (upper - lower) * t_rand
        pts = rays_o.unsqueeze(self.samples_dim) + rays_d.unsqueeze(self.samples_dim) * z_vals.unsqueeze(self.channel_dim)
        if self.z_normalize:
            normalized_pts = pts * 2 / ((far - near).unsqueeze(self.samples_dim))
        else:
            normalized_pts = pts

        raw = self.run_network(normalized_pts, viewdirs, styles=styles)
        _, sdf = torch.split(raw, [3, 1], dim=self.channel_dim)
        sdf = sdf.squeeze(self.channel_dim)
        target_values = pts.detach().norm(dim=-1) - ((far - near) / 4)

        return sdf, target_values

    def forward(self, cam_poses, focal, near, far, styles=None, return_eikonal=False):
        rgb, features, sdf, mask, xyz, eikonal_term = self.render(focal, c2w=cam_poses, near=near, far=far, styles=styles, return_eikonal=return_eikonal)

        rgb = rgb.permute(0,3,1,2).contiguous()
        if self.output_features:
            features = features.permute(0,3,1,2).contiguous()

        if xyz != None:
            xyz = xyz.permute(0,3,1,2).contiguous()
            mask = mask.permute(0,3,1,2).contiguous()

        return rgb, features, sdf, mask, xyz, eikonal_term
