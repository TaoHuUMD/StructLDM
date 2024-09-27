import torch
import torch.nn as nn
import functools
from torch.autograd import Variable
import numpy as np
from .facenet import Sphere20a, senet50
import torch.nn.functional as F
from torch import autograd

from torch.nn.modules.utils import _ntuple
TORCH_VERSION = tuple(int(x) for x in torch.__version__.split(".")[:2])

#from .vision_transformer import SwinUnet

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

    elif classname.find('Linear') != -1 or classname.find('Conv1d') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def check_trainable_params(net_class):
    for key, value in net_class.named_parameters():
        print(key, value.requires_grad)
        continue
        if not value.requires_grad:
            continue
        print(key, value.requires_grad)

def check_params_device(net_class):
    for key, value in net_class.named_parameters():
        print(key, value.device)
        continue
        if not value.requires_grad:
            continue
        print(key, value.requires_grad)

def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer

def define_uvlatent_smooth_net(input_nc, output_nc, ngf = 64, netG = "global", n_downsample_global=1, n_blocks_global=0, n_local_enhancers=1, 
             n_blocks_local=3, norm='instance', gpu_ids=[]):
    
    norm_layer = get_norm_layer(norm_type=norm)    
    netG = GlobalGenerator(input_nc, output_nc, ngf, n_downsample_global, n_blocks_global, norm_layer)           
    #print(netG)
    if False and len(gpu_ids) > 0:
        assert(torch.cuda.is_available())   
        netG.cuda(gpu_ids[0])
        netG = torch.nn.DataParallel(netG, gpu_ids)
    netG.apply(weights_init)
    return netG

def g_nonsaturating_loss(fake_pred):
    fake_pred = tensor_mean(fake_pred)
    loss = F.softplus(-fake_pred).mean()

    return loss

def upsample_net(input_nc, output_nc, norm='instance', gpu_ids=[]):    
    norm_layer = get_norm_layer(norm_type=norm)  
    
    netG = UpSample_Net(input_nc, output_nc, norm_layer)
    
    if False and len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        netG.cuda(gpu_ids[0])
        netG = torch.nn.DataParallel(netG, gpu_ids)
    netG.apply(weights_init)
    return netG

def define_PoseNet(input_nc, output_nc, ngf, netG, n_downsample_global=3, n_blocks_global=9, n_local_enhancers=1, 
             n_blocks_local=3, norm='instance', gpu_ids=[], shape_mem = 0, opt = None):
    
    norm_layer = get_norm_layer(norm_type=norm)
    
    netG = PoseNetGenerator(input_nc, output_nc, ngf, n_downsample_global, n_blocks_global, norm_layer, shape_mem)       
    netG.apply(weights_init)
    return netG

def define_Tex2Tex(input_nc, output_nc, ngf, netG, n_downsample_global=3, n_blocks_global=9, n_local_enhancers=1, 
             n_blocks_local=3, norm='instance', gpu_ids=[], shape_mem = 0, opt = None):
    
    norm_layer = get_norm_layer(norm_type=norm)
    
    netG = NetImgTranslation(input_nc, output_nc, ngf, n_downsample_global, n_blocks_global, norm_layer, shape_mem)       
    netG.apply(weights_init)
    return netG


def define_G(input_nc, output_nc, ngf, netG, n_downsample_global=3, n_blocks_global=9, n_local_enhancers=1, 
             n_blocks_local=3, norm='instance', gpu_ids=[], shape_mem = 0, opt = None):    
    norm_layer = get_norm_layer(norm_type=norm)

    if netG == 'shape_mem':    
        netG = GlobalGenerator_Memory(input_nc, output_nc, ngf, n_downsample_global, n_blocks_global, norm_layer, opt)

    elif netG == 'global':    
        netG = GlobalGenerator(input_nc, output_nc, ngf, n_downsample_global, n_blocks_global, norm_layer, shape_mem)       
    elif netG == 'local':        
        netG = LocalEnhancer(input_nc, output_nc, ngf, n_downsample_global, n_blocks_global, 
                                  n_local_enhancers, n_blocks_local, norm_layer)
    elif netG == 'encoder':
        netG = Encoder(input_nc, output_nc, ngf, n_downsample_global, norm_layer)
    else:
        raise('generator not implemented!')
   
    netG.apply(weights_init)
    return netG


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

def define_D(input_nc, ndf, n_layers_D, norm='instance', use_sigmoid=False, num_D=1, getIntermFeat=False, gpu_ids=[]):        
    norm_layer = get_norm_layer(norm_type=norm)   
    netD = MultiscaleDiscriminator(input_nc, ndf, n_layers_D, norm_layer, use_sigmoid, num_D, getIntermFeat)   
    #print(netD)
    if False and len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        netD.cuda(gpu_ids[0])
        netD = torch.nn.DataParallel(netD, gpu_ids)
    netD.apply(weights_init)
    return netD

def print_network(net):
    if isinstance(net, list):
        net = net[0]
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)

##############################################################################
# Losses
##############################################################################
class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                self.real_label_var = Variable(real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = Variable(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):
        if isinstance(input[0], list):
            loss = 0
            for input_i in input:
                pred = input_i[-1]
                
                target_tensor = self.get_target_tensor(pred, target_is_real)

                loss += self.loss(pred, target_tensor)
            return loss
        else:            
            target_tensor = self.get_target_tensor(input[-1], target_is_real)
            return self.loss(input[-1], target_tensor)

class VGGLoss(nn.Module):
    def __init__(self, gpu_ids):
        super(VGGLoss, self).__init__()        
        self.vgg = Vgg19().cuda()
        self.criterion = nn.L1Loss()
        self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]        

    def forward(self, x, y):              
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())        
        return loss

###########################CUSTOM LOSSES################################
class FaceLoss(nn.Module):
    def __init__(self, pretrained_path='asset/spretrains/sphere20a_20171020.pth'):
        super(FaceLoss, self).__init__()
        if 'senet' in pretrained_path:
            self.net = senet50(include_top=False)
            self.load_senet_model(pretrained_path)
            self.height, self.width = 224, 224
        else:
            self.net = Sphere20a()
            self.load_sphere_model(pretrained_path)
            self.height, self.width = 112, 96

        self.net.eval()
        self.criterion = nn.L1Loss()
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]

        # from uvm_lib.engine.demo_visualizer import MotionImitationVisualizer
        # self._visualizer = MotionImitationVisualizer('debug', ip='http://10.10.10.100', port=31102)

    def forward(self, imgs1, imgs2, kps1=None, kps2=None, bbox1=None, bbox2=None):
        """
        Args:
            imgs1:
            imgs2:
            kps1:
            kps2:
            bbox1:
            bbox2:

        Returns:

        """
        #filtering images
        imgs_n1, imgs_n2, bbox_n1, bbox_n2 = [], [], [], []
        for i in range(bbox1.shape[0]):
            if not torch.all(torch.eq(bbox1[i], torch.zeros(4))):
                imgs_n1.append(imgs1[i])
                imgs_n2.append((imgs2[i]))
                bbox_n1.append(bbox1[i])
                bbox_n2.append(bbox2[i])
        if len(imgs_n1) == 0:
            return 0.0

        [imgs_n1, imgs_n2, bbox_n1, bbox_n2] = map(torch.stack, [imgs_n1, imgs_n2, bbox_n1, bbox_n2])



        if kps1 is not None:
            head_imgs1 = self.crop_head_kps(imgs1, kps1)
        elif bbox1 is not None:
            head_imgs1 = self.crop_head_bbox(imgs_n1, bbox_n1)
        elif self.check_need_resize(imgs1):
            head_imgs1 = F.interpolate(imgs1, size=(self.height, self.width), mode='bilinear', align_corners=True)
        else:
            head_imgs1 = imgs1

        if kps2 is not None:
            head_imgs2 = self.crop_head_kps(imgs2, kps2)
        elif bbox2 is not None:
            head_imgs2 = self.crop_head_bbox(imgs_n2, bbox_n2)
        elif self.check_need_resize(imgs2):
            head_imgs2 = F.interpolate(imgs1, size=(self.height, self.width), mode='bilinear', align_corners=True)
        else:
            head_imgs2 = imgs2

        loss = self.compute_loss(head_imgs1, head_imgs2)

        # import cv2
        # facereal = util.tensor2im(head_imgs2[1])
        # cv2.imwrite('/HPS/impl_deep_volume3/static00/exp_data/tex_debug/face/facereal.jpg', facereal)



        return loss

    def compute_loss(self, img1, img2):
        """
        :param img1: (n, 3, 112, 96), [-1, 1]
        :param img2: (n, 3, 112, 96), [-1, 1], if it is used in training,
                     img2 is reference image (GT), use detach() to stop backpropagation.
        :return:
        """
        f1, f2 = self.net(img1), self.net(img2)

        loss = 0.0
        for i in range(len(f1)):
            loss += self.criterion(f1[i], f2[i].detach())

        return loss

    def check_need_resize(self, img):
        return img.shape[2] != self.height or img.shape[3] != self.width

    def crop_head_bbox(self, imgs, bboxs):
        """
        Args:
            bboxs: (N, 4), 4 = [lt_x, lt_y, rt_x, rt_y]
            #wrong you mf!, code says xx yy, and documents says xy xy, you slimy f

        Returns:
            resize_image:
        """
        bs, _, ori_h, ori_w = imgs.shape

        head_imgs = []

        for i in range(bs):
            min_x, max_x, min_y, max_y = bboxs[i]
            head = imgs[i:i+1, :, min_y:max_y, min_x:max_x]  # (1, c, h', w')
            head = F.interpolate(head, size=(self.height, self.width), mode='bilinear', align_corners=True)
            head_imgs.append(head)

        head_imgs = torch.cat(head_imgs, dim=0)

        return head_imgs

    def crop_head_kps(self, imgs, kps):
        """
        :param imgs: (N, C, H, W)
        :param kps: (N, 19, 2)
        :return:
        """
        bs, _, ori_h, ori_w = imgs.shape

        rects = self.find_head_rect(kps, ori_h, ori_w)
        head_imgs = []

        for i in range(bs):
            min_x, max_x, min_y, max_y = rects[i]
            head = imgs[i:i+1, :, min_y:max_y, min_x:max_x]  # (1, c, h', w')
            head = F.interpolate(head, size=(self.height, self.width), mode='bilinear', align_corners=True)
            head_imgs.append(head)

        head_imgs = torch.cat(head_imgs, dim=0)

        return head_imgs

    @staticmethod
    @torch.no_grad()
    def find_head_rect(kps, height, width):
        NECK_IDS = 12

        kps = (kps + 1) / 2.0

        necks = kps[:, NECK_IDS, 0]
        zeros = torch.zeros_like(necks)
        ones = torch.ones_like(necks)

        # min_x = int(max(0.0, np.min(kps[HEAD_IDS:, 0]) - 0.1) * image_size)
        min_x, _ = torch.min(kps[:, NECK_IDS:, 0] - 0.05, dim=1)
        min_x = torch.max(min_x, zeros)

        max_x, _ = torch.max(kps[:, NECK_IDS:, 0] + 0.05, dim=1)
        max_x = torch.min(max_x, ones)

        # min_x = int(max(0.0, np.min(kps[HEAD_IDS:, 0]) - 0.1) * image_size)
        min_y, _ = torch.min(kps[:, NECK_IDS:, 1] - 0.05, dim=1)
        min_y = torch.max(min_y, zeros)

        max_y, _ = torch.max(kps[:, NECK_IDS:, 1], dim=1)
        max_y = torch.min(max_y, ones)

        min_x = (min_x * width).long()      # (T, 1)
        max_x = (max_x * width).long()      # (T, 1)
        min_y = (min_y * height).long()     # (T, 1)
        max_y = (max_y * height).long()     # (T, 1)

        # print(min_x.shape, max_x.shape, min_y.shape, max_y.shape)
        rects = torch.stack((min_x, max_x, min_y, max_y), dim=1)

        # import ipdb
        # ipdb.set_trace()

        return rects

    def load_senet_model(self, pretrain_model):
        # saved_data = torch.load(pretrain_model, encoding='latin1')
        from uvm_lib.engine.thutil import load_pickle_file
        saved_data = load_pickle_file(pretrain_model)
        save_weights_dict = dict()

        for key, val in saved_data.items():
            if key.startswith('fc'):
                continue
            save_weights_dict[key] = torch.from_numpy(val)

        self.net.load_state_dict(save_weights_dict)

        print('load face model from {}'.format(pretrain_model))

    def load_sphere_model(self, pretrain_model):
        saved_data = torch.load(pretrain_model)
        save_weights_dict = dict()

        for key, val in saved_data.items():
            if key.startswith('fc6'):
                continue
            save_weights_dict[key] = val

        self.net.load_state_dict(save_weights_dict)

        print('load face model from {}'.format(pretrain_model))


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

class UpSample_Net(nn.Module):
    def __init__(self, input_nc, output_nc, norm_layer=nn.BatchNorm2d, padding_type='reflect'):        
        super(UpSample_Net, self).__init__()        
        activation = nn.ReLU(True)

        model = []
        
        ### downsample        
        decoder_n_upsampling = int(np.log2(output_nc/input_nc) + 0.9)

        #
        unified_channel = int(2**(int(np.log2(input_nc))))

        model += [nn.Conv2d(input_nc, unified_channel * 2, kernel_size=3, stride=1, padding=1),
                    norm_layer(unified_channel * 2), activation]
        
        for i in range(1, decoder_n_upsampling):
            mult = 2**i * unified_channel
            model += [nn.Conv2d(mult, mult * 2, kernel_size=3, stride=1, padding=1),
                      norm_layer(mult * 2), activation]
        self.model = nn.Sequential(*model)
    def forward(self, input):
        return self.model(input)

class GlobalGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, n_downsampling=3, n_blocks=9, norm_layer=nn.BatchNorm2d, shape_mem = 0, padding_type='reflect'):
        assert(n_blocks >= 0)
        super(GlobalGenerator, self).__init__()        
        activation = nn.ReLU(True)        

        model = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0), norm_layer(ngf), activation]
        ### downsample
        for i in range(n_downsampling):
            mult = 2**i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1),
                      norm_layer(ngf * mult * 2), activation]

        ### resnet blocks
        mult = 2**n_downsampling
        
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, activation=activation, norm_layer=norm_layer)]

        ### upsample         
        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1, output_padding=1), norm_layer(int(ngf * mult / 2)), activation]

        inc = ngf
        while inc < output_nc:
            outc = inc * 2
            if outc >= output_nc:
                outc = output_nc

            model += [nn.Conv2d(inc, outc, kernel_size=3, stride=1, padding=1),activation]
            inc *= 2

        model += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh()]
        self.model = nn.Sequential(*model)
            
    def forward(self, input, shape_mem = None):
        return self.model(input)


class PoseNetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, n_downsampling=3, n_blocks=9, norm_layer=nn.BatchNorm2d, shape_mem = 0, padding_type='reflect'):
        assert(n_blocks >= 0)
        super(PoseNetGenerator, self).__init__()        
        activation = nn.ReLU(True)        

        model = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0), norm_layer(ngf), activation]
        ### downsample
        for i in range(n_downsampling):
            mult = 2**i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1),
                      norm_layer(ngf * mult * 2), activation]

        ### resnet blocks
        mult = 2**n_downsampling
        
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, activation=activation, norm_layer=norm_layer)]

        ### upsample         
        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1, output_padding=1), norm_layer(int(ngf * mult / 2)), activation]

        inc = ngf
        while inc < output_nc:
            outc = inc * 2
            if outc >= output_nc:
                outc = output_nc
                model += [nn.Conv2d(inc, outc, kernel_size=3, stride=1, padding=1),activation]
                inc = outc
                break
            model += [nn.Conv2d(inc, outc, kernel_size=3, stride=1, padding=1),activation]
            inc *= 2

        model += [nn.ReflectionPad2d(3), nn.Conv2d(inc, output_nc, kernel_size=7, padding=0), nn.Tanh()]
        self.model = nn.Sequential(*model)
            
    def forward(self, input, shape_mem = None):

        return self.model(input)



class NetImgTranslation(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, n_downsampling=3, n_blocks=9, norm_layer=nn.BatchNorm2d, shape_mem = 0, padding_type='reflect'):
        assert(n_blocks >= 0)
        super(NetImgTranslation, self).__init__()        
        activation = nn.ReLU(True)        

        model = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0), norm_layer(ngf), activation]
        ### downsample
        for i in range(n_downsampling):
            mult = 2**i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1),
                      norm_layer(ngf * mult * 2), activation]

        ### resnet blocks
        mult = 2**n_downsampling
        
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, activation=activation, norm_layer=norm_layer)]

        ### upsample         
        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1, output_padding=1), norm_layer(int(ngf * mult / 2)), activation]

        inc = ngf
        while inc < output_nc:
            outc = inc * 2
            if outc >= output_nc:
                outc = output_nc
                model += [nn.Conv2d(inc, outc, kernel_size=3, stride=1, padding=1),activation]
                inc = outc
                break
            model += [nn.Conv2d(inc, outc, kernel_size=3, stride=1, padding=1),activation]
            inc *= 2

        model += [nn.ReflectionPad2d(3), nn.Conv2d(inc, output_nc, kernel_size=7, padding=0), nn.Tanh()]
        self.model = nn.Sequential(*model)
            
    def forward(self, input, shape_mem = None):

        return self.model(input)


def define_ResizeNet(input_nc, output_nc, org_size, tgt_size, gpu_ids = []):
    
    if tgt_size[0] < org_size[0]:        
        resize_net = DownsampleNet(input_nc, output_nc, org_size, tgt_size)
    else:
        resize_net = UpsampleNet(input_nc, output_nc, org_size, tgt_size)
    
    
    if False and len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        resize_net.cuda(gpu_ids[0])
        resize_net = torch.nn.DataParallel(resize_net, gpu_ids)
    resize_net.apply(weights_init)
    return resize_net




    
class DownsampleNet(nn.Module):
    def __init__(self, input_nc, output_nc, org_size, tgt_size):
        
        super(DownsampleNet, self).__init__()
        factor_h = int(org_size[0] / tgt_size[0])
        factor_w = int(org_size[1] / tgt_size[1])
        assert factor_h == factor_w

        self.tgt_size = tgt_size

        self.scale_factor = 2    
        self.factor = factor_h
        if self.factor == 2 or self.factor == 4:
            resize_net_0 = [nn.Conv2d(input_nc, output_nc, kernel_size=3, stride=2, padding=1), nn.LeakyReLU()] 
            self.resize_net_0 = nn.Sequential(*resize_net_0)
        elif self.factor == 8:
            resize_net_0 = [nn.Conv2d(input_nc, output_nc, kernel_size=3, stride=2, padding=1), nn.LeakyReLU()]
            resize_net_1 = [nn.Conv2d(input_nc, output_nc, kernel_size=3, stride=2, padding=1), nn.LeakyReLU()]
            
            self.resize_net_0 = nn.Sequential(*resize_net_0)
            self.resize_net_1 = nn.Sequential(*resize_net_1)
            
    def interp2d(self, input):
        return torch.nn.functional.interpolate(input, size = self.tgt_size, mode='bilinear', align_corners=False)#, antialias=True

        
    def forward(self, input):
        if self.factor ==2 :
            return self.resize_net_0(input)
        elif self.factor == 4:
            net = self.resize_net_0(input)
            return self.interp2d(net)
        elif self.factor == 8:
            net = self.resize_net_0(input)
            net = self.interp2d(net)
            return self.resize_net_1(net)


    
class UpsampleNet(nn.Module):
    def __init__(self, input_nc, output_nc, org_size, tgt_size):
        
        super(UpsampleNet, self).__init__()
        factor_h = int(tgt_size[0] / org_size[0])
        factor_w = int(tgt_size[1] / org_size[1])
        assert factor_h == factor_w

        kernel_size = 4
        self.scale_factor = 2    
        self.factor = factor_h
        if self.factor == 2 or self.factor == 4:
            resize_net_0 = [nn.ConvTranspose2d(
                        input_nc, output_nc, kernel_size, stride=2, padding=int(kernel_size / 2 - 1)
            )]
            self.resize_net_0 = nn.Sequential(*resize_net_0)
        elif self.factor == 8:
            resize_net_0 = [nn.ConvTranspose2d(input_nc, output_nc, kernel_size, stride=2, padding=int(kernel_size / 2 - 1))]
            resize_net_1 = [nn.ConvTranspose2d(output_nc, output_nc, kernel_size, stride=2, padding=int(kernel_size / 2 - 1))]
            
            self.resize_net_0 = nn.Sequential(*resize_net_0)
            self.resize_net_1 = nn.Sequential(*resize_net_1)
            
    def interp2d(self, input):
        return interpolate(
            input, scale_factor=self.scale_factor, mode="bilinear", align_corners=False
        )
        
    def forward(self, input):
        #print('resize factor ', self.factor)
        if self.factor ==2 :
            return self.resize_net_0(input)
        elif self.factor == 4:
            net = self.resize_net_0(input)
            return self.interp2d(net)
        elif self.factor == 8:
            net = self.resize_net_0(input)
            net = self.interp2d(net)
            return self.resize_net_1(net)

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
