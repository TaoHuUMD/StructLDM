import numpy as np
import torch

#from pdb import set_trace as st

import os


import sys
sys.path.append("..")
from Engine.th_utils.networks import networks

from Engine.th_utils.io.prints import *

import torch.nn as nn

from Engine.th_utils.io.prints import *


from Engine.th_utils.networks.net_utils import load_model_full, save_model_full

from Engine.th_utils.animation.uv_generator import Index_UV_Generator

import torchvision.transforms 
import torchvision.transforms as T


from .latent import LatentModule
from .base_model import BaseModel

from .model import VoxelHumanGenerator as EVAGenerator

class Model(BaseModel):
    def name(self):
        return 'Model'
    
    def project_setup(self, opt):

        self.opt.use_posmap = False

        if self.opt.gen_ratio <= 2:
            assert opt.df.use_face or opt.face_D

        if not self.opt.isTrain:
            self.opt.df.stage1_cross_tex = False

        self.opt.output_ch = self.opt.df.df_nerf_dim

    def initialize(self, opt):
                
        BaseModel.initialize(self, opt)
        self.project_setup(opt)

        self.isTrain = opt.isTrain
                                  
        self.uv_reso = self.opt.df.uv_reso
        self.gpu_ids = self.opt.gpu_ids[0]
        self.is_evaluate = False
        self.is_vis_lat = False

        self.render_posmap = Index_UV_Generator(self.uv_reso, self.uv_reso, uv_type=self.opt.df.uv_type, data_dir= os.path.join(self.opt.data_dir, "asset/uv_sampler"))
        self.vts_uv = self.render_posmap.get_vts_uv().cuda().permute(0,2,1)
        self.vts_uv.requires_grad = False

        size = min(self.opt.model.renderer_spatial_output_dim)
        if size >= 512: 
            size /= 2
            size = int(size)

        self.render_posmap_b = Index_UV_Generator(size, size, uv_type="BF", data_dir= os.path.join(self.opt.data_dir, "asset/uv_sampler"))
        self.vts_uv_b = self.render_posmap_b.get_vts_uv().cuda().permute(0,2,1)
        self.vts_uv_b.requires_grad = False    

        self.visual_names = []                                
        self.visual_names += ['real_image', 'fake_image', 'fake_uv', 'real_uv']

        if not self.opt.df.no_visualize_pose:
            self.visual_names += ["pose_uv"]
                    
        if self.opt.df.visualize_depth:
            self.visual_names += ['depth_pred']

        if self.opt.df.visualize_normal:
            self.visual_names += ['normal_pred']

        if self.opt.df.df_ab_Dtex:            
            self.visual_names += ['fake_uv']
        

        self.visual_names += ['real_image', 'fake_image']
            
        if self.opt.df.df_ab_tex_rec:
            self.visual_names += ['tex_rec']


        if self.opt.df.df_ab_nerf_rec:
            self.visual_names += ['nerf_rec']

        if self.opt.normal_recon:
            self.visual_names += ['normal_pred']
        

        if self.opt.df.stage1_cross_tex:
            self.visual_names += ['vis_cross_img']
            self.visual_names += ['vis_cross_uv']
            
                            
        self.visual_names += ['visOutput']

        #1, h, w
        self.template_uv_mask = self.render_posmap.get_uv_mask()[None, ...].detach().unsqueeze(1).to(self.gpu_ids)

        if self.opt.df.structured_2d:
            self.struct_template_uv_mask = torch.nn.functional.interpolate(self.template_uv_mask, size=(self.opt.structured_reso, self.opt.structured_reso), mode='nearest')
        else:
            self.struct_template_uv_mask = None

        self.build_nr_net()

    def get_last_layer(self):
        if not hasattr(self, "netSup"):
            return None

        last_layer = self.netSup.get_last_layer()
        return last_layer.weight


    def build_nr_net(self):
         
        opt = self.opt

        self.img_gen_h, self.img_gen_w = opt.model.renderer_spatial_output_dim
        self.img_nerf_h, self.img_nerf_w = self.img_gen_h // (self.opt.nerf_ratio / self.opt.gen_ratio), self.img_gen_w // (self.opt.nerf_ratio / self.opt.gen_ratio)
        self.img_nerf_h, self.img_nerf_w  = int(self.img_nerf_h), int(self.img_nerf_w)

        G_params, self.G_model_names = [], []     


        self.nerfRender = EVAGenerator(model_opt=opt.model, renderer_opt=opt.rendering, full_pipeline=False, voxhuman_name=opt.model.voxhuman_name, opt=self.opt)
        self.G_model_names.append('nerfRender')
        if opt.use_nerf: G_params += [{"params": self.nerfRender.parameters()}]

        self.net_Latent = LatentModule(opt=self.opt, render_posemap=self.render_posmap, template_uv_mask=self.template_uv_mask, structure_template_uv_mask = self.struct_template_uv_mask, vts_uv = self.vts_uv)
        emd_lr = self.opt.training.glr if self.opt.training.lr_embed_same else self.opt.training.lr_embed
        
        self.emd_lr = emd_lr


        self.net_Latent_embedding = self.net_Latent.uv_embedding_list
        if self.opt.training.fix_embed:
            emd_lr = 0
            networks.requires_grad(self.net_Latent, False)
            networks.requires_grad(self.net_Latent.uv_embedding_list, False)
        else:
            self.G_model_names.append('net_Latent_embedding')
            G_params += [{"params": self.net_Latent.uv_embedding_list.parameters(), 'lr': emd_lr}]
        
            latent_net_list = []
            for k in dict(self.net_Latent.named_parameters()).keys():
                if k.startswith("uv_embedding_list"): continue
                latent_net_list.append(k.split(".")[0])
            
            for uni_net in set(latent_net_list):
                model_name = f'net_Latent_{uni_net}'
                setattr(self, model_name, getattr(self.net_Latent, uni_net))
                self.G_model_names.append(model_name)
                G_params += [{"params": getattr(self, model_name).parameters()}]

        input_dim = self.opt.df.df_nerf_dim
        if self.opt.df.df_ab_sup_2d_style:
            if self.opt.stage1_ad_struct and self.opt.structured_2d:
                input_dim += self.opt.df.structured_dim
            elif self.opt.stage1_ad_1d:
                input_dim += self.opt.texdecoder_outdim
                
        out_list = ["rgb"]
        output_nc = 3
        if self.opt.nerf_ratio > self.opt.gen_ratio:
            self.netSup = networks.define_supreso_lightweight(input_nc = input_dim, output_nc = output_nc, factor = int(self.opt.nerf_ratio // self.opt.gen_ratio), out_list=out_list, up_layer = self.opt.up_layer)        

            self.netSup = networks.net_to_GPU(self.netSup, self.gpu_ids)
            self.G_model_names.append('netSup')        
            G_params += [{"params": getattr(self, "netSup").parameters()}]
        
        
        #optimizer        
        self.g_lr = self.opt.training.glr
        self.d_lr = self.opt.training.dlr

        self.optimizer_G = torch.optim.Adam(G_params, lr = self.g_lr, betas=(0, 0.9))
                
        
        #if self.opt.verbose:
        print('---------- Networks initialized -------------')

        self.global_sample_z_eval = [torch.randn(self.opt.val_n_sample, self.opt.style_dim, device="cuda").repeat_interleave(8,dim=0).requires_grad_(False)]
        

    def requires_grad_G(self, status):        
        for netname in self.G_model_names:        
            networks.requires_grad(getattr(self, netname), status)


    def requires_grad_G_normal(self, status = True):        
        networks.requires_grad(getattr(self, "nerfRender"), status)


    def input_tocuda(self, batch):
        for k in batch:
            if torch.is_tensor(batch[k]): 
                batch[k] = batch[k].cuda()
                continue
            if (not isinstance(batch[k], np.int)) and (not isinstance(batch[k], list)) and (not isinstance(batch[k][0], str)):
                if self.opt.rank == 0: printg(k, type(batch[k]))
                batch[k] = batch[k].cuda()
    
    def get_transforms(self, infer):
        is_aug_this_epoch = False
        if (not infer) and self.opt.aug_nr and np.random.rand() > 0.8:
            if self.opt.small_rot: scale = 10
            else: scale = 180

            r = np.random.rand() * 2 - 1.0
            transforms = T.Compose([
                torchvision.transforms.RandomRotation(degrees=(r*scale, r*scale), fill = 0)
                ]
            )
            transforms_img = T.Compose([
                torchvision.transforms.RandomRotation(degrees=(r*scale, r*scale), fill = 1 if self.opt.rendering.white_bg else -1)
                ]
            )
            is_aug_this_epoch = True
        else: 
            transforms = T.Compose([])
            transforms_img = T.Compose([])
        
        return is_aug_this_epoch, transforms, transforms_img

    
    def crop_img(self, img_list, bbox):
        #b,c,h,w
        b, c, h, w = img_list[0].shape
        x0, x1, y0, y1 = bbox #batch["crop_bbox"]

        th, tw = self.img_gen_h, self.img_gen_w
        
        h_ = h - y0 - y1
        w_ = w - x0  - x1
        if th > h_: y0 -= (th - h_)
        else: y0 += (h_ - th)

        if tw > w_: x0 -= (tw - w_)
        else: x0 += (w_ - tw)

        self.crop_offset_y = y0
        self.crop_offset_x = x0

        for img in img_list:
            img = img[:, :, y0 : h - y1, x0 : w - x1]

    def forward_G(self, batch, infer=False):
        
        self.input_tocuda(batch)
        
        batch["img_gen"] = batch["img_gen"].permute(0,3,1,2)

        if batch["msk_gen"].ndimension() == 3:
            batch["msk_gen"] = batch["msk_gen"][..., None]

        batch["msk_gen"] = batch["msk_gen"].permute(0,3,1,2)
        if batch["msk_gen"].shape[1] > 1: batch["msk_gen"] = batch["msk_gen"][:, [0], ...]


        if infer:
            self.batch_index = batch['eva_index'][0] #batch['index'][0]
            if isinstance(self.batch_index, list):
                self.batch_index = self.batch_index[0]

        self.loss_G_L1 = 0

        self.dataset_id = 0      
        self.frame_index = batch["image_name"]

        is_aug_this_epoch, transforms, transforms_img = self.get_transforms(infer)

        if True: #2d, posed uv           
            smpl_vts_loc = batch["smpl_vertices"][...,:3].clone()  
            smpl_vts_wrd = batch["smpl_vertices"][...,3:6].clone()  
            smpl_vts_wrd_notrans = batch["smpl_vertices"][...,6:].clone()  

            if smpl_vts_wrd.ndimension() == 4:
                smpl_vts_wrd = smpl_vts_wrd[0]
            elif smpl_vts_wrd.ndimension() == 2:
                smpl_vts_wrd = smpl_vts_wrd[None,...]

            assert smpl_vts_wrd.ndimension() == 3
            is_mask = is_normal = is_depth = False

            if self.opt.df.no_visualize_pose:
                posed_uv = None
            

        if self.opt.is_crop:          
            if True:
                b, c, h, w = batch["img_gen"].shape

                x0, x1, y0, y1 = batch["crop_bbox"]
            
                th, tw = self.img_gen_h, self.img_gen_w
                
                h_ = h - y0 - y1
                w_ = w - x0  - x1
                if th > h_: y0 -= (th - h_)
                else: y0 += (h_ - th)

                if tw > w_: x0 -= (tw - w_)
                else: x0 += (w_ - tw)
                #printg(x0, y0, x1, y1)

                self.crop_offset_y = y0
                self.crop_offset_x = x0

                if posed_uv is not None:
                    posed_uv = posed_uv[:, :, y0 : h - y1, x0 : w - x1]
                batch["img_gen"] = batch["img_gen"][:, :, y0 : h - y1, x0 : w - x1]
                batch["msk_gen"] = batch["msk_gen"][:, :, y0 : h - y1, x0 : w - x1]

        
        if posed_uv is not None:
            self.pose_uv = posed_uv.detach()
        else:
            self.pose_uv = None

        real_image = transforms_img(batch["img_gen"])

        if posed_uv is not None:
            posed_uv = transforms(posed_uv)

        assert self.opt.rendering.use_sdf_render

        uvlatent = self.net_Latent(batch)
        nerf_input_latent = uvlatent

        return_sdf = self.opt.min_surf_lambda > 0 and not infer
        return_eikonal = self.opt.eikonal_lambda > 0 and not infer
        if self.opt.df.visualize_normal:
            return_eikonal = True

        out = self.nerfRender(nerf_input_latent, 
                batch["extrinsics"], 
                batch["focal"],
                batch["beta"],
                batch["theta"],
                batch["trans"],
                return_sdf = return_sdf,
                return_normal = True,
                return_eikonal = return_eikonal,
                return_sdf_xyz = False,
                smpl_vts_wrd = smpl_vts_wrd_notrans,
                smpl_vts_can = None,
                batch = batch,
                scale = batch["scale"].item() if "scale" in batch.keys() else 1

            )
                
        nerf_img = out[1].permute(0,3,1,2) 
        nerf_latent = nerf_img
        nerf_normal = out[-1]
        

        if nerf_img is None: return -1


        nerf_latent = transforms_img(nerf_latent)
        #
        if self.opt.is_crop:
            b, c, h, w = real_image.shape                    
            _, _, hn, wn = nerf_latent.shape

            factor = self.img_gen_h // self.img_nerf_h

            x0, x1, y0, y1 = batch["crop_bbox"]
            x0 = x0 / factor
            x1 = x1 / factor
            y0 = y0 / factor
            y1 = y1 / factor

            x0 = int(x0)
            x1 = int(x1)
            y0 = int(y0)
            y1 = int(y1)
            
            th, tw = self.img_nerf_h, self.img_nerf_w
            
            h_ = hn - y0 - y1
            w_ = wn - x0  - x1
            if th > h_: y0 -= (th - h_)
            else: y0 += (h_ - th)

            if tw > w_: x0 -= (tw - w_)
            else: x0 += (w_ - tw)
            
            nerf_latent = nerf_latent[:,:, y0 : hn - y1, x0 : wn - x1]
            if nerf_normal is not None: 
                nerf_normal = nerf_normal[:, y0 : hn - y1, x0 : wn - x1, :]


        nerf_latent_ = nerf_latent
        
        if hasattr(self, "netSup"):
            fake_image = self.netSup(nerf_latent_)
        else:
            fake_image = nerf_latent_
    
        self.fake_image = fake_image.detach()
        self.real_image = real_image.detach()

        if self.opt.df.visualize_depth or self.opt.df.visualize_normal: #
            normal_pred = (nerf_normal).detach()
            depth_pred = nerf_latent[:, -1, ...]

            try:
                nerf_normal_bg = torch.all(torch.abs(normal_pred) < 1e-3, dim = -1)
                nerf_normal_msk = ~nerf_normal_bg

                normal_pred_org = normal_pred.clone()
                normal_black_org, normal_gray_org = normal_pred.clone(), normal_pred.clone()

                normal_pred[nerf_normal_msk] = normal_pred[nerf_normal_msk] / torch.norm(normal_pred[nerf_normal_msk], dim=-1, keepdim=True) #* 0.8

                normal_black, normal_gray, normal_white = normal_pred.clone(), normal_pred.clone(), normal_pred.clone()
                normal_black[nerf_normal_bg] = -1
                normal_black_org[nerf_normal_bg] = -1 
                normal_white[nerf_normal_bg] = 1
                                
                d = -0.3
                normal_gray[nerf_normal_bg] = d
                normal_gray_org[nerf_normal_bg] = d
                
                depth_pred[nerf_normal_bg] = 0

                nodepth_norm = depth_pred.clone()

                def renorm(d, v):
                    m_ = (d > v).bool()
                    min_, max_ = d[m_].min(), d[m_].max()
                    d_ = (d - min_) / (max_ - min_)
                    d_[d_<0] = 0
                    return d_
                                
                depth_pred6 = renorm(depth_pred, depth_pred.max() * 0.6)
                depth_pred7 = renorm(depth_pred, depth_pred.max() * 0.7)
                depth_pred8 = renorm(depth_pred, depth_pred.max() * 0.8)


                depth_pred = self.normalize_tensor_msk(depth_pred, nerf_normal_msk)
                #renormalize

                depth_black, depth_gray, depth_white = depth_pred.clone(), depth_pred.clone(), depth_pred.clone()
                depth_black[nerf_normal_bg] = 0

                depth_pred6[nerf_normal_bg] = 0
                depth_pred7[nerf_normal_bg] = 0
                depth_pred8[nerf_normal_bg] = 0

                depth_gray = depth_gray * 0.6 + (d + 1) / 2 * 1.2
                depth_gray[nerf_normal_bg] = (d + 1) / 2
                depth_white[nerf_normal_bg] = 1

                normal_black_raw = normal_black.clone()
                normal_black *= depth_black[..., None]

                normal_gray *= depth_black[..., None]

                normal_pred = torch.cat([normal_black_raw, normal_black], -2)
                depth_pred = torch.cat([depth_black, depth_pred6,depth_pred7,depth_pred8], -1) #  depth_black
                self.depth_pred = depth_pred.detach()
                self.normal_pred = normal_pred.detach().permute(0, 3, 1, 2)
            
            except Exception as e:
                printg(e)
                st()

        if self.opt.phase == "test":
            return 

        if infer: return 1
        if fake_image is None: return -1


    def init_setup(self):
        pass

    def tensor_img(self, t):
        def add_channel(img):
            if img.shape[0] >= 3: return img
            im0 = torch.zeros((3 - img.shape[0], img.shape[1], img.shape[2])).to(img.device)
            return torch.cat((im0, img), 0)

        if not torch.is_tensor(t): return t
        if t.ndim == 4:
            return (t.permute(0, 2, 3, 1).cpu().numpy() + 1)/2 * 255

        return (t.permute(1,2,0).cpu().numpy() + 1)/2 * 255

    def compute_visuals(self, phase = "train"):

        ind=0                           

        def fc_pad_img(img, tgt_h, tgt_w):
            im0 = np.ones((tgt_h, tgt_w, 3)) * 255
            if img is not None:
                h, w, _ = img.shape
                im0[:h, :w, :] = img
            return im0
        
        def tensor_img(t):
            def add_channel(img):
                if img.shape[0] >= 3: return img
                im0 = torch.zeros((3 - img.shape[0], img.shape[1], img.shape[2])).to(img.device)
                return torch.cat((im0, img), 0)

            if not torch.is_tensor(t): return t
            if t.ndim == 4:
                return (t.permute(0, 2, 3, 1).cpu().numpy() + 1)/2 * 255

            return (t.permute(1,2,0).cpu().numpy() + 1)/2 * 255
                          
        row1 = []
        r1list = ["fake_image", "real_image", "vis_cross_img"] 
        
        for item in r1list:
            if item in self.visual_names and hasattr(self, item) and getattr(self, item) is not None:
                setattr(self, item, tensor_img(getattr(self, item)[0]))
                row1.append(getattr(self, item)[..., :3])

        if "pose_uv" in self.visual_names and hasattr(self, "pose_uv") and getattr(self, "pose_uv") is not None:
            self.pose_uv = self.pose_uv[ind].permute(1,2,0).cpu().numpy()
            im0 = np.zeros((self.pose_uv.shape[0], self.pose_uv.shape[1], 1))
            self.pose_uv = np.concatenate((im0, self.pose_uv), 2) * 255
            row1.append(getattr(self, "pose_uv"))
        
        row2 = []
        r2list = ["pred_normal_uv", "vis_cross_uv", "tex_rec", "nerf_rec", "normal_pred"] #
        
        for item in r2list:
            if item in self.visual_names and hasattr(self, item) and getattr(self, item) is not None:
                setattr(self, item, tensor_img(getattr(self, item)[0]))
                row2.append(getattr(self, item))

        if "depth_pred" in self.visual_names: 
            self.depth_pred = self.depth_pred[ind][None].permute(1,2,0).cpu().numpy()
            self.depth_pred = np.concatenate([self.depth_pred, self.depth_pred, self.depth_pred], -1) 
            self.depth_pred = self.depth_pred * 255
            row2.append(self.depth_pred)

        valid_r2 = []
        r2_h, r2_w = 0, 0
        for i in row2:
            if i is None: continue
            r2_h = max(r2_h, i.shape[0])
            r2_w = max(r2_w, i.shape[1])
        
        valid_r2 = []
        for i in row2:
            if i is None: continue
            valid_r2.append(fc_pad_img(i, r2_h, i.shape[1]))
    
        row1 = np.concatenate(row1, 1)
        if len(valid_r2) == 0:
            self.visOutput = row1
            return self.fake_image
        
        row2 = np.concatenate(valid_r2, 1)
        r1_h, r1_w, _  = row1.shape
        r2_h, r2_w, _ = row2.shape
        if r1_w > r2_w:
            row2 = np.concatenate((row2, fc_pad_img(None, r2_h, r1_w - r2_w)), 1)
        elif r1_w < r2_w:
            row1 = np.concatenate((row1, fc_pad_img(None, r1_h, r2_w - r1_w)), 1)
        
        self.visOutput = np.concatenate((row1, row2), 0)

        self.is_evaluate = False
        return self.fake_image

    def export_latent(self):
        self.net_Latent.export_latent()    

    def normalize_tensor_msk(self, input, msk):
        max_d = input[msk].max()
        min_d = input[msk].min()        
        return (input - min_d) / (max_d - min_d)

    def normalize_tensor(self, input):
        max_d = input.max()
        min_d = input.min()        
        return (input - min_d) / (max_d - min_d)
         
    def load_all(self, epoch = "", resume = True):

        if self.opt.phase == "test":
            self.optimizer_G = None 
            self.optimizer_D = None

        out = load_model_full(self.save_dir, self,
                    self.optimizer_G, self.optimizer_D if hasattr(self, "optimizer_D") else None,
                    scheduler = None, recorder =  None,
                    resume = resume, epoch = epoch, device = self.gpu_ids, strict=False)
            
        if isinstance(out, int) and out == -1:
            begin_epoch, epoch_iter= -1, 0

            if self.opt.load_pretrained_model:
                printg("load pretrained model %s" % (self.opt.load_model_name))
                pretrained_dir = self.opt.training.checkpoints_dir.replace(self.opt.name, self.opt.load_model_name)
                out = load_model_full(pretrained_dir, self,
                       self.optimizer_G, self.optimizer_D if hasattr(self, "optimizer_D") else None,
                       scheduler = None, recorder =  None,
                       resume = resume, epoch = epoch, strict = self.opt.load_strict, device = self.gpu_ids)
                if isinstance(out, int) and out == -1:
                    printg("error, %s not trained" % self.opt.load_model_name, pretrained_dir)
                    exit()
                
        else:
            begin_epoch, epoch_iter, g_lr, d_lr = out

            if self.opt.load_pretrained_model:
                printg("%s model exists, still load pretrained model %s ?" % (self.opt.name, self.opt.load_model_name))
                if self.opt.confirm_pretrained_model:
                    pretrained_dir = self.opt.training.checkpoints_dir.replace(self.opt.name, self.opt.load_model_name)
                    out = load_model_full(pretrained_dir, self,
                        self.optimizer_G, self.optimizer_D if hasattr(self, "optimizer_D") else None,
                        scheduler = None, recorder =  None,
                        resume = resume, epoch = epoch, strict = self.opt.load_strict, device = self.gpu_ids)
                    if isinstance(out, int) and out == -1:
                        printg("error, %s not trained" % self.opt.load_model_name, pretrained_dir)
                        exit()

            self.g_lr = g_lr
            self.d_lr = d_lr

        return begin_epoch, epoch_iter
     
    def save_all(self, label = "", epoch = -1, iter = 0):
        save_model_full(self.save_dir, self, self.optimizer_G, self.optimizer_D if hasattr(self, "optimizer_D") else None, label = label, epoch = epoch, iter = iter, g_lr = self.g_lr, d_lr = self.d_lr)        
        return 
    
    def inference(self, batch, infer=False):
        return self.forward_G(batch, True)
        
class InferenceModel(Model):
    def forward(self, inp):
        label, inst = inp
        return self.inference(label, inst)
