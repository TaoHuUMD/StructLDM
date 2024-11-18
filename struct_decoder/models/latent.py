import numpy as np
import torch
import torch.nn as nn

import sys
sys.path.append("..")
from Engine.th_utils.networks import networks
from Engine.th_utils.io.prints import *


class LatentModule(nn.Module):
    def name(self):
        return 'Model'
    
    def __init__(self, opt, render_posemap = None, template_uv_mask = None, structure_template_uv_mask = None, vts_uv = None):
        
        super(LatentModule, self).__init__()

        self.opt = opt
        self.render_posmap = render_posemap          
        self.vts_uv = vts_uv
        self.template_uv_mask = template_uv_mask 
        self.struct_template_uv_mask = structure_template_uv_mask

        self.full_pipeline = False
        self.gpu_ids = opt.gpu_ids[0]


        self.global_1d_sample_z_eval = [self.make_1d_noise(self.opt.val_n_sample, self.opt.df.style_dim_1d, device=self.gpu_ids).repeat_interleave(self.opt.val_n_sample, dim=0).requires_grad_(False)] 


        self.global_1d_sample_z_eval_diff = [self.make_1d_noise(self.opt.val_n_sample, self.opt.df.style_dim_1d, device=self.gpu_ids).requires_grad_(False)] 


        self.global_1d_sample_z_eval_1000 = [self.make_1d_noise(1000, self.opt.df.style_dim_1d, device=self.gpu_ids).requires_grad_(False)]



        self.global_2d_sample_z_eval_1000 = [self.make_2d_noise(1000, self.opt.df.structured_dim, self.opt.df.structured_reso, device=self.gpu_ids).requires_grad_(False)] 


        self.global_2d_sample_z_eval = [self.make_2d_noise(self.opt.val_n_sample, self.opt.df.structured_dim, self.opt.df.structured_reso, device=self.gpu_ids).repeat_interleave(self.opt.val_n_sample, dim=0).requires_grad_(False)] 


        self.global_2d_sample_z_eval_diff = [self.make_2d_noise(self.opt.val_n_sample, self.opt.df.structured_dim, self.opt.df.structured_reso, device=self.gpu_ids).requires_grad_(False)] 

        self.build_embedding()

    def build_embedding(self, G_params=None):
        self.uv_embedding_list = nn.ModuleList()

        self.uv_reso = self.opt.df.uv_reso
        self.gpu_ids = self.opt.gpu_ids

        dataset_size = self.opt.dataset.dataset_size 
        self.dataset_size = dataset_size

        printg("lat size", dataset_size)

        #auto decoder by structured latent
        if self.opt.stage_1_fitting:
            if self.opt.df.uvh_trip_direct:
                for i in range(dataset_size):
                    emd = nn.Embedding(self.uv_reso * self.uv_reso, self.opt.texdecoder_outdim)
                    self.uv_embedding_list.append(networks.net_to_GPU(emd, self.gpu_ids))            
            else:
                raise NotImplementedError()
       
    def make_1d_noise(self, batch, dim, device):
        return torch.randn(batch, dim, device=device)

    def make_2d_noise(self, batch, dim, reso, device):
        return torch.randn(batch, dim, reso, reso, device=device)
    
    def get_latent_diff(self, idx, size = None):

        if size is None:
            #str_reso = self.opt.structured_reso
            reso = self.uv_reso
        embed = self.uv_embedding_list[idx](torch.arange(0, reso*reso).cuda()).view(reso, reso, -1).permute(2,0,1) 

        if self.opt.df.stage1_ad_2d or self.opt.df.upsample_topo:

            uvlatent = embed[None]

            uvmask = self.uv_mask.to(embed.device)


            if self.opt.df.stage1_ad_2d:
                uvlatent = self.netLatSmooth(uvlatent)
            elif self.opt.df.upsample_topo:
                
                latent_3d = self.render_posmap.index_posmap_by_vts(uvlatent, self.vts_uv, to_rotate = False).permute(0,2,1)
                uv_latent_ = self.render_posmap.render_feature_map(latent_3d, self.uv_reso).permute(0,3,1,2) 

                r_rot_uv = uv_latent_[0]
                if r_rot_uv.shape[0] > r_rot_uv.shape[-1]:
                    r_rot_uv = r_rot_uv.permute(1,2,0)
                r_rot_uv = torch.rot90(r_rot_uv, 1, (1,2))
                uvlatent = torch.flip(r_rot_uv, [0, 1])[None,...]
                
            

            if self.opt.df.mask_latent and not self.opt.tex_trip:                
                uvlatent = uvmask * uvlatent
            
            uvlatent = torch.nn.functional.interpolate(uvlatent, size=(256, 256), mode='bilinear')


            embed = uvmask * embed[None]
            embed = torch.nn.functional.interpolate(embed, size=(256, 256), mode='bilinear')[0]

            return (embed, uvlatent[0])
        else:
            if self.opt.df.mask_latent and not self.opt.tex_trip:                
                uvmask = self.uv_mask.to(embed.device)
                embed = uvmask * embed[None]
                embed = embed[0]
            
            embed = torch.nn.functional.interpolate(embed[None], size=(256, 256), mode='bilinear')[0]
            return embed

    
    def extract_latent_stage_1(self, batch):

        batch_size = batch["beta"].shape[0]
        device = batch["beta"].device

        frame_id = batch['index'] #list
        if len(frame_id) == 1:
            frame_id = frame_id

        if self.opt.isInfer and self.opt.phase == "evaluate":
            frame_id = batch['eva_index']

        if self.opt.isTrain:

            if self.opt.df.stage1_cross_tex:
                frame_id = [frame_id] +  [batch['cross_index']]

            if self.opt.df.aug_compos_tex:
                frame_id += [batch['compositional_index']]

        str_reso = self.opt.structured_reso

        mean_uv_latent = None
        if self.opt.isInfer:
            def get_mean_embedding():
                num = min(self.dataset_size, 1000)
                frame_id_idx = torch.from_numpy(np.round(np.linspace(0, self.dataset_size - 1, num)).astype(int)).to(device)            
                ebd_idx = torch.arange(0, str_reso*str_reso).to(device)
                return torch.cat([self.uv_embedding_list[i](ebd_idx).view(1, str_reso, str_reso, -1) for i in frame_id_idx], 0).mean(0, True)

            if frame_id[0] == -1: #calc mean embedding
                if self.opt.df.stage1_ad_1d and self.opt.df.stage1_1d_1d:
                    #todo need to udate further
                    frame_id[0] = 0
                else:
                    mean_uv_latent = get_mean_embedding().expand(batch_size, -1, -1, -1).permute(0,3,1,2)
        
        if self.opt.df.uvh_trip_direct:
            if "diff_samples" in batch.keys():
                uvlatent = batch["diff_samples"]
                if uvlatent.ndimension() == 3:
                    uvlatent = uvlatent[None,...]

            elif mean_uv_latent is not None:
                uvlatent = mean_uv_latent
            else:    
                uvlatent = torch.cat([self.uv_embedding_list[i](torch.arange(0, self.uv_reso * self.uv_reso).to(device)).view(1, self.uv_reso, self.uv_reso, -1).permute(0,3,1,2) for i in frame_id], 0) #0 for batch.  #! 1 for cross
                    
            batch["uvlatent"] = uvlatent
       
        return uvlatent

    def forward(self, batch):
        opt = self.opt
        if opt.df.stage_1_fitting:
            return self.extract_latent_stage_1(batch) #texture triplane
        
