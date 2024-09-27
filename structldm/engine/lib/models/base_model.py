import os
import torch
import sys
from collections import OrderedDict
from ..util import util
from torchsummary import summary

import numpy as np
from uvm_lib.engine.thutil.num import vis_high_dimension_img, plot_scat
import cv2
import io


class BaseModel(torch.nn.Module):
    def name(self):
        return 'BaseModel'

    def initialize(self, opt):
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.isTrain = opt.isTrain
        self.Tensor = torch.cuda.FloatTensor if self.gpu_ids else torch.Tensor
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)
        self.visual_names = []
        self.model_names = []
        self.loss_names = []

    def set_input(self, input):
        self.input = input

    def evaluate(self, data):
        self.forward_org(data)

    def forward(self, data):
        pass

    # used in test time, no backprop
    def test(self):
        pass

    def get_image_paths(self):
        pass

    def optimize_parameters(self):
        pass

    def get_current_visuals(self):
        visual_ret = OrderedDict()
        if "visOutput" in self.visual_names:
            visual_ret["visOutput"] = getattr(self, "visOutput")
            return visual_ret

        for name in self.visual_names:
            if isinstance(name, str):
                visual_ret[name] = getattr(self, name)
        return visual_ret

    def tensor_to_viz(self, visuals):
        viz_mod = OrderedDict()
        for name in visuals:
            viz_mod[name] = util.tensor2im(visuals[name][0])
        return viz_mod


    def get_current_losses(self):
        """Return traning losses / errors. train.py will print out these errors on console, and save them to a file"""
        errors_ret = OrderedDict()
        for name in self.loss_names:
            if isinstance(name, str):
                errors_ret[name] = float(getattr(self, 'loss_' + name,0))  # float(...) works for both scalar tensor and float number
        return errors_ret

    def get_current_errors(self):
        return {}

    def save(self, label):
        pass

    def plot_latents_vts(self, uv_lat, vts_uv): #for nb
        uv_mask1 = self.render_posmap.get_uv_mask().cpu().numpy()            
        uv_mask = uv_mask1[...,None].repeat(3, axis=2)

        uv_seg = cv2.imread("../asset/uv_seg_10.png", cv2.IMREAD_UNCHANGED).astype(np.float32)
        h, w = uv_mask.shape[0], uv_mask.shape[1]
        uv_seg = cv2.resize(uv_seg, (h, w), interpolation=cv2.INTER_NEAREST)
        
        uv_seg = (uv_seg - 50)/20
        uv_seg[ uv_seg < 0 ] = 0
                
        uv_seg = torch.from_numpy(uv_seg).float().cuda()[None,...].permute(0,3,1,2)
        vts_seg = self.render_posmap.index_posmap_by_vts(uv_seg, vts_uv).permute(0,2,1)[0]
        
        uv_lat_2 = vis_high_dimension_img(uv_lat, 2)
        self.uv_lat_2d = plot_scat(uv_lat_2, vts_seg.cpu().numpy())
        
        uv_lat_3 = vis_high_dimension_img(uv_lat, 3)
        vts_uv_np = vts_uv.permute(0,2,1).cpu().numpy()[0] 
        
        hs, ws = 64, 64
        self.uv_lat = np.zeros((hs, ws, 3))
        
        for i in range(uv_lat_3.shape[0]):
            print(uv_lat_3.shape[0])
            x, y = vts_uv_np[i][0] * hs, vts_uv_np[i][1] * ws
            x, y = int(np.floor(x)), int(np.floor(y))
            self.uv_lat[x][y] = uv_lat_3[i]
        
        self.uv_lat *= 255
    
    def plot_latents(self):        
        ind = 0
        
        uv_mask1 = self.render_posmap.get_uv_mask().cpu().numpy()            
        uv_mask = uv_mask1[...,None].repeat(3, axis=2)
        
        uv_mask_bool = np.where(uv_mask1>0, True, False)
        
        uv_seg = cv2.imread("../asset/uv_seg_10.png", cv2.IMREAD_UNCHANGED).astype(np.float32)
        h, w = uv_mask.shape[0], uv_mask.shape[1]
        uv_seg = cv2.resize(uv_seg, (h, w), interpolation=cv2.INTER_NEAREST)
        
        uv_seg = (uv_seg - 50)/20
        uv_seg[uv_seg<0] = 0
        #uv_seg = uv_seg.astype(np.uint)
            
        self.uv_lat = self.uv_lat[ind].permute(1,2,0).cpu().numpy()
        masked_uv = self.uv_lat * uv_mask[...,[0]].repeat(self.uv_lat.shape[-1], axis = 2)
        
        #printb(self.pose_lat.shape)
        
        self.pose_lat = self.pose_lat[0].permute(1,2,0).cpu().numpy()
        masked_pose_lat = self.pose_lat * uv_mask[...,[0]].repeat(self.uv_lat.shape[-1], axis = 2)
        
        # uvmap_t = np.zeros((128,128,2))
        # for i in range(uvmap_t.shape[0]):
        #     for j in range(uvmap_t.shape[1]):
        #         uvmap_t[i][j] = np.array([i/128, j/128])
        # uvmap = np.concatenate((uvmap_t, uvmap_t), 2)
        # self.lat_2d = plot_scat(uvmap_t, uv_seg)
        uv_lat_2 = vis_high_dimension_img(masked_uv, 2)
        self.uv_lat_2d = plot_scat(uv_lat_2, uv_seg)
        
        pose_lat_2d = vis_high_dimension_img(masked_pose_lat, 2)
        self.pose_lat_2d = plot_scat(pose_lat_2d, uv_seg)
        
        pose_lat = vis_high_dimension_img(masked_pose_lat, 3)
        self.pose_lat = pose_lat * uv_mask * 255
        
        uv_lat_3 = vis_high_dimension_img(masked_uv, 3)
                    
        self.uv_lat = uv_lat_3 * uv_mask * 255
        
        is_posed_uv = False
        for vis in self.visual_names:
            if "posed_uv_lat" == vis:
                is_posed_uv = True
                break
            continue
        if is_posed_uv:                
            self.posed_uv_lat = self.render_posmap.index_posmap_by_uvmap(torch.from_numpy(self.uv_lat).float()[None,...].permute(0,3,1,2).cuda(), self.nr_uv.cuda()).detach().permute(0,2,3,1).cpu().numpy()[0]

    def save_networks(self, epoch):
        """Save all the networks to the disk.

        Parameters:
            epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
        """
        for name in self.model_names:
            if isinstance(name, str):
                save_filename = '%s_net_%s.pth' % (epoch, name)
                save_path = os.path.join(self.save_dir, save_filename)
                net = getattr(self, 'net' + name)
                torch.save(net.cpu().state_dict(), save_path)
                if len(self.gpu_ids) and torch.cuda.is_available():
                    net.cuda()


    # helper saving function that can be used by subclasses
    def save_network(self, network, network_label, epoch_label, gpu_ids):
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(self.save_dir, save_filename)
        torch.save(network.cpu().state_dict(), save_path)
        if len(gpu_ids) and torch.cuda.is_available():
            network.cuda()

    # helper loading function that can be used by subclasses
    def load_network(self, network, network_label, epoch_label, save_dir=''):        
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        if not save_dir:
            save_dir = self.save_dir
        save_path = os.path.join(save_dir, save_filename)        
        if not os.path.isfile(save_path):
            print('%s not exists yet!' % save_path)
            if network_label == 'G':
                raise('Generator must exist!')
        else:
            #network.load_state_dict(torch.load(save_path))
            try:
                network.load_state_dict(torch.load(save_path))
            except:   
                pretrained_dict = torch.load(save_path)                
                model_dict = network.state_dict()
                try:
                    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}                    
                    network.load_state_dict(pretrained_dict)
                    if self.opt.verbose:
                        print('Pretrained network %s has excessive layers; Only loading layers that are used' % network_label)
                except:
                    print('Pretrained network %s has fewer layers; The following are not initialized:' % network_label)
                    for k, v in pretrained_dict.items():                      
                        if v.size() == model_dict[k].size():
                            model_dict[k] = v

                    if sys.version_info >= (3,0):
                        not_initialized = set()
                    else:
                        from sets import Set
                        not_initialized = Set()                    

                    for k, v in model_dict.items():
                        if k not in pretrained_dict or v.size() != pretrained_dict[k].size():
                            not_initialized.add(k.split('.')[0])
                    
                    print(sorted(not_initialized))
                    network.load_state_dict(model_dict)                  

    def load_networks(self, epoch):
        """Load all the networks from the disk.

        Parameters:
            epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
        """
        for name in self.model_names:
            if isinstance(name, str):
                load_filename = '%s_net_%s.pth' % (epoch, name)
                load_path = os.path.join(self.save_dir, load_filename)
                net = getattr(self, 'net' + name)
                #if isinstance(net, torch.nn.DataParallel):
                #    net = net.module
                print('loading the model from %s' % load_path)
                # if you are using PyTorch newer than 0.4 (e.g., built from
                # GitHub source), you can remove str() on self.device
                state_dict = torch.load(load_path)
                if hasattr(state_dict, '_metadata'):
                    del state_dict._metadata

                # patch InstanceNorm checkpoints prior to 0.4
                # for key in list(state_dict.keys()):  # need to copy keys here because we mutate in loop
                #     self.__patch_instance_norm_state_dict(state_dict, net, key.split('.'))
                net.load_state_dict(state_dict)

    def update_learning_rate():
        pass
