import torch
import torch.nn as nn
import functools
from torch.autograd import Variable
import numpy as np
from .facenet import Sphere20a, senet50
import torch.nn.functional as F

#import util.util as util


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

    def get_head_img(self, imgs1, imgs2, kps1=None, kps2=None, bbox1=None, bbox2=None):
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
            head_imgs2 = F.interpolate(imgs2, size=(self.height, self.width), mode='bilinear', align_corners=True)
        else:
            head_imgs2 = imgs2

        return head_imgs1, head_imgs2
        if head_imgs1 is None or head_imgs2 is None:
            return None, None
        

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
                
        # filtering images
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
            head_imgs2 = F.interpolate(imgs2, size=(self.height, self.width), mode='bilinear', align_corners=True)
        else:
            head_imgs2 = imgs2

        if head_imgs1 is None or head_imgs2 is None:
            return 0

        loss = self.compute_loss(head_imgs1, head_imgs2)
     
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
            min_x, min_y, max_x, max_y = bboxs[i]
                        
            head = imgs[i:i + 1, :, min_y:max_y, min_x:max_x]  # (1, c, h', w')                    
            
            if head.shape[2] < 5 or head.shape[3] < 5:
                return None    

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
            head = imgs[i:i + 1, :, min_y:max_y, min_x:max_x]  # (1, c, h', w')
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

        min_x = (min_x * width).long()  # (T, 1)
        max_x = (max_x * width).long()  # (T, 1)
        min_y = (min_y * height).long()  # (T, 1)
        max_y = (max_y * height).long()  # (T, 1)

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


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss
    Takes embeddings of two samples and a target label == 1 if samples are from the same class and label == 0 otherwise
    """

    def __init__(self, margin = 0.1):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.eps = 1e-9

    #by default it treats as if it is from the same identity
    def forward(self, output1, output2, target = 1, size_average=True):
        distances = (output2 - output1).pow(2).sum(1)  # squared distances
        losses = 0.5 * (target * distances +
                        (1 + -1 * target) * F.relu(self.margin - (distances + self.eps).sqrt()).pow(2))
        return losses.mean() if size_average else losses.sum()


class TripletLoss(nn.Module):

    """
    Triplet loss
    Takes embeddings of an anchor sample, a positive sample and a negative sample
    """

    def __init__(self, margin):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative, size_average=True):
        distance_positive = (anchor - positive).pow(2).sum(1)  # .pow(.5)
        distance_negative = (anchor - negative).pow(2).sum(1)  # .pow(.5)
        losses = F.relu(distance_positive - distance_negative + self.margin)
        return losses.mean() if size_average else losses.sum()

class TotalVariation(nn.Module):
    r"""Computes the Total Variation according to [1].

    Shape:
        - Input: :math:`(N, C, H, W)` or :math:`(C, H, W)`.
        - Output: :math:`(N,)` or scalar.

    Examples:
        >>> tv = TotalVariation()
        >>> output = tv(torch.ones((2, 3, 4, 4), requires_grad=True))
        >>> output.data
        tensor([0., 0.])
        >>> output.sum().backward()  # grad can be implicitly created only for scalar outputs

    Reference:
        [1] https://en.wikipedia.org/wiki/Total_variation
    """
    
    def __init__(self, direction="hw"):
        super(TotalVariation, self).__init__()
        self.direction = direction
        print(self.direction)        

    def forward(self, img) -> torch.Tensor:
        if self.direction == "hw":
            return self.total_variation_hw(img)
        elif self.direction == "h":
            return self.total_variation_h(img)
        elif self.direction == "w":
            return self.total_variation_w(img)
    
    def total_variation_h(self, img: torch.Tensor) -> torch.Tensor:
        r"""Function that computes Total Variation according to [1].

        Args:
            img: the input image with shape :math:`(N, C, H, W)` or :math:`(C, H, W)`.

        Return:
            a scalar with the computer loss.

        Examples:
            >>> total_variation(torch.ones(3, 4, 4))
            tensor(0.)

        Reference:
            [1] https://en.wikipedia.org/wiki/Total_variation
        """
        if not isinstance(img, torch.Tensor):
            raise TypeError(f"Input type is not a torch.Tensor. Got {type(img)}")

        if len(img.shape) < 3 or len(img.shape) > 4:
            raise ValueError(f"Expected input tensor to be of ndim 3 or 4, but got {len(img.shape)}.")

        pixel_dif1 = img[..., 1:, :] - img[..., :-1, :]

        reduce_axes = (-3, -2, -1)
        res1 = pixel_dif1.abs().sum(dim=reduce_axes)

        return res1
    
    def total_variation_w(self, img: torch.Tensor) -> torch.Tensor:
        r"""Function that computes Total Variation according to [1].

        Args:
            img: the input image with shape :math:`(N, C, H, W)` or :math:`(C, H, W)`.

        Return:
            a scalar with the computer loss.

        Examples:
            >>> total_variation(torch.ones(3, 4, 4))
            tensor(0.)

        Reference:
            [1] https://en.wikipedia.org/wiki/Total_variation
        """
        if not isinstance(img, torch.Tensor):
            raise TypeError(f"Input type is not a torch.Tensor. Got {type(img)}")

        if len(img.shape) < 3 or len(img.shape) > 4:
            raise ValueError(f"Expected input tensor to be of ndim 3 or 4, but got {len(img.shape)}.")

        pixel_dif2 = img[..., :, 1:] - img[..., :, :-1]

        reduce_axes = (-3, -2, -1)
        res2 = pixel_dif2.abs().sum(dim=reduce_axes)

        return res2
    
    def total_variation_hw(self, img: torch.Tensor) -> torch.Tensor:
        r"""Function that computes Total Variation according to [1].

        Args:
            img: the input image with shape :math:`(N, C, H, W)` or :math:`(C, H, W)`.

        Return:
            a scalar with the computer loss.

        Examples:
            >>> total_variation(torch.ones(3, 4, 4))
            tensor(0.)

        Reference:
            [1] https://en.wikipedia.org/wiki/Total_variation
        """
        if not isinstance(img, torch.Tensor):
            raise TypeError(f"Input type is not a torch.Tensor. Got {type(img)}")

        if len(img.shape) < 3 or len(img.shape) > 4:
            raise ValueError(f"Expected input tensor to be of ndim 3 or 4, but got {len(img.shape)}.")

        pixel_dif1 = img[..., 1:, :] - img[..., :-1, :]
        pixel_dif2 = img[..., :, 1:] - img[..., :, :-1]

        reduce_axes = (-3, -2, -1)
        res1 = pixel_dif1.abs().sum(dim=reduce_axes)
        res2 = pixel_dif2.abs().sum(dim=reduce_axes)

        return res1 + res2

def Total_variation_loss(y):
    tv = (
        torch.sum(torch.abs(y[:, :, :, :-1] - y[:, :, :, 1:])) + 
        torch.sum(torch.abs(y[:, :, :-1, :] - y[:, :, 1:, :]))
    )
    
    tv /= (2*y.nelement())
    
    return tv

class VGGLoss(nn.Module):
    def __init__(self, gpu_ids=0):
        super(VGGLoss, self).__init__()        
        self.vgg = Vgg19().cuda(gpu_ids)
        self.criterion = nn.L1Loss()
        self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]        

    def forward(self, x, y):              
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())        
        return loss

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
