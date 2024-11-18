from matplotlib.pyplot import axes
import numpy as np
import torch
import os
import torch.nn as nn

import sys
sys.path.append("..")
from Engine.th_utils.networks import networks
from Engine.th_utils.io.prints import *

from Engine.th_utils.animation.uv_generator import Index_UV_Generator

class LatentModule(nn.Module):
    def name(self):
        return 'Model'
    
    def requires_grad(self, model, flag):
        for name, p in model.named_parameters():
            p.requires_grad = flag

    def model_weight(self, model):
        for param in model.parameters():
            printg("net ", torch.sum(param.data))
            
    def __init__(self, opt):
        super(LatentModule, self).__init__()

        self.opt = opt

        self.render_posmap = Index_UV_Generator(self.opt.uv_reso, self.opt.uv_reso, uv_type=self.opt.uv_type, data_dir="../asset/data/uv_sampler", device="cpu")
        self.vts_uv = self.render_posmap.get_vts_uv().permute(0,2,1) #.cuda()
        self.vts_uv.requires_grad = False
    
        self.struct_template_uv_mask = None

        self.uv_dim = self.opt.uv_dim

        if hasattr(self.opt, "load_processing_net") and self.opt.load_processing_net:
            from .model import GeomConvLayersNeW
            
            self.netLatSmooth = GeomConvLayersNeW(input_nc = 32, hidden_nc = 32, output_nc = 32, use_relu=self.opt.use_relu, kernel_size=self.opt.smooth_kernel_size)
            
            self.requires_grad(self.netLatSmooth, False)
            print("load process")
            self.netLatSmooth.eval()

        
        template_uv_mask = self.render_posmap.get_uv_mask()[None,...].detach().to("cpu")
        self.template_uv_mask = template_uv_mask.repeat(self.opt.uv_dim, 1, 1) 

        self.scale = 1
        if self.opt.data_name == "deepfashion":
            self.scale = self.opt.scale_fc #1 / 7

        self.full_pipeline = False

        self.build_embedding()

        self.is_test = False

        if hasattr(self.opt, "is_test") and self.opt.is_test:
            self.is_test = True
            return

    def __len__(self):
        return len(self.uv_embedding_list)

    def build_embedding(self, G_params=None):
        self.uv_embedding_list = nn.ModuleList()

        self.uv_reso = self.opt.uv_reso
        dataset_size = self.opt.lat_num 

        self.dataset_size = dataset_size

        for i in range(dataset_size):
          emd = nn.Embedding(self.uv_reso * self.uv_reso, self.opt.uv_dim)
          self.uv_embedding_list.append(emd.cpu())

        networks.requires_grad(self.uv_embedding_list, False)

       
    def store_normalized_latent(self):
        self.normalized_latent = []


        if self.is_test and (self.opt.normalization == "none" or self.opt.normalization == "scale"):
            return

        self.min_lat = 1<<26
        self.max_lat = -(1<<26) 

        for idx in range(len(self.uv_embedding_list)):
            lat = self.get_latent(idx)
            if self.opt.normalization == "none":
                lat_ = (lat * self.dilation_mask_expand)
                
                self.min_lat = min(self.min_lat, torch.min(lat_).item())
                self.max_lat = max(self.max_lat, torch.max(lat_).item())

            elif self.opt.normalization == "scale":
                lat_ = (lat * self.dilation_mask_expand) * self.scale
                self.min_lat = min(self.min_lat, torch.min(lat_).item())
                self.max_lat = max(self.max_lat, torch.max(lat_).item())
                #self.normalized_latent.append(lat_) #.cuda()
            else:
                assert lat.shape ==self.mean.shape and lat.shape ==self.std.shape
                lat_ = ((lat - self.mean) / self.std * self.dilation_mask_expand)
                self.min_lat = min(self.min_lat, torch.min(lat_).item())
                self.max_lat = max(self.max_lat, torch.max(lat_).item())


            self.normalized_latent.append(lat_) #.cuda()

        printg("after scale ", self.min_lat, self.max_lat)


    def get_stas(self):
        return (self.min_lat, self.max_lat)

    def print_std(self, name, t):
        printg(name, t.mean(), t.std(), t.min(), t.max())

    def calc_latent(self):

        self.uv_embedding_list = self.uv_embedding_list[:self.opt.lat_valid_num]
        printg("valid latent ", len(self.uv_embedding_list))

        lat_all = torch.cat([self.get_latent(i) for i in range(len(self.uv_embedding_list))], 0)

        c = lat_all.shape[0]
        lat_all = self.dilation_mask * lat_all
        assert lat_all.shape[0] == c

        def point_norm(lat_):
            mean, std = torch.mean(lat_, 0, keepdim = True), torch.std(lat_, 0, keepdim = True).float()
            std = torch.where(std < 1e-7, torch.tensor(1, dtype=std.dtype), std)
            return mean.expand(self.uv_dim, -1, -1), std.expand(self.uv_dim, -1, -1)

        def global_norm(lat_):
            mean, std = torch.mean(lat_), torch.std(lat_)
            return torch.ones_like(self.uv_seg)[0] * mean, torch.ones_like(self.uv_seg)[0] * std

        if self.opt.normalization == "point":
            self.mean, self.std = point_norm(lat_all)
            
        elif self.opt.normalization == "global":
            self.mean, self.std = global_norm(lat_all)
            
        elif self.opt.normalization == "none":
            pass
             
        else:
            print("which normalization ?")
            raise NotImplementedError()
     
        if hasattr(self, "mean"):
            print(self.mean.mean(), self.std.mean())

        printy("std mean ", lat_all.mean(), lat_all.std(), lat_all.min(), lat_all.max())

        if hasattr(self, "normalized_latent"):
            self.min_lat = 1<<26
            self.max_lat = -(1<<26)
            self.min_lat = min(self.min_lat, torch.min(self.normalized_latent).item())
            self.max_lat = max(self.max_lat, torch.max(self.normalized_latent).item()) 
            printg(" normalized std mean ", self.normalized_latent.mean(), self.normalized_latent.std(), self.normalized_latent.min(), self.normalized_latent.max())


        self.min_lat = lat_all.min()
        self.max_lat = lat_all.max()

        if not hasattr(self, "normalized_latent") and not self.is_test: #and self.opt.normalization != "none"
            self.store_normalized_latent()
   
    def normalize_input(self, sample):
        if self.opt.normalization == "none": return sample
        if self.opt.normalization == "scale":
            return sample / self.scale
        
        if self.opt.normalization == "sqrt":
            return torch.pow(sample / self.sqrt_scale, 2)
        
        batch_size = sample.shape[0]
        std = self.std[None,...].expand(batch_size, -1, -1, -1)
        mean = self.mean[None,...].expand(batch_size, -1, -1, -1)
        return (sample - mean.to(sample.device)) / std.to(sample.device) 


    def denorm(self, sample):
        if self.opt.normalization == "none": return sample
        if self.opt.normalization == "scale":
            return sample / self.scale
        
        if self.opt.normalization == "sqrt":
            return torch.pow(sample / self.sqrt_scale, 2)
        

        batch_size = sample.shape[0]
        std = self.std[None,...].expand(batch_size, -1, -1, -1)
        mean = self.mean[None,...].expand(batch_size, -1, -1, -1)
        return sample * std.to(sample.device) + mean.to(sample.device)

    def get_org_latent(self):
        lat_all = torch.cat([self.get_latent(i) for i in range(len(self.uv_embedding_list))], 0)
        return lat_all


    def get_normalized_latent(self, idx):

        cond = {}
 
        if hasattr(self.opt, "remove_invalid_id") and self.opt.remove_invalid_id and idx in self.invalid_list_id:
            data_size = len(self.normalized_latent) if isinstance(self.normalized_latent, list) else self.normalized_latent.shape[0]
            return self.get_normalized_latent((idx + 1) % data_size)#, cond
            
        if hasattr(self.opt, "use_partial_latent") and self.opt.use_partial_latent:
          cond.update({
              "use_partial_latent" : True,
              "latent_msk": self.valid_latent_mask_list[[idx]]
          })

        return self.normalized_latent[idx], cond
    
    def get_latent(self, idx, size = None):
        if size is None:
            #str_reso = self.opt.structured_reso
            reso = self.uv_reso
        embed = self.uv_embedding_list[idx](torch.arange(0, reso*reso).cpu()).view(reso, reso, -1).permute(2,0,1) 

        embed_ = embed
        if hasattr(self, "netLatSmooth"):
            with torch.no_grad():
                embed_ = self.netLatSmooth(embed[None, ...])[0]

        if idx ==0:
            printy("lat sum 0 ", torch.sum(embed_))

        return embed_
