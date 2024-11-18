import numpy as np
import os
import cv2

import sys
sys.path.append("..")
from pdb import set_trace as st

import numpy as np
import torch

from torch.utils.data.sampler import WeightedRandomSampler

from Engine.th_utils.io.prints import *
from Engine.th_utils.load_smpl_tmp import get_smpl_vertices, load_smpl
from Engine.th_utils.distributed.sampler import DistributedSamplerWrapper, data_sampler
#from Engine.lib.data.base_dataset import BaseDataset
from .base_dataset import BaseDataset
from .renderpeople import RenderPeopleSViewDataset

class Dataset(BaseDataset):

    def initialize(self, opt, phase, data_dict):
        self.opt = opt

        self.phase = phase
        self.data_root = opt.dataset.dataset_path

        printg("create %s dataset" % phase)
        self.smpl_model = load_smpl(gender="neutral", device="cpu")

        self.dataset_rp = RenderPeopleSViewDataset(
            opt.dataset.dataset_path,
            None,
            os.path.join(opt.dataset.dataset_path, "{people_id}/outputs_re_fitting/refit_smpl_2nd.npz"),
            os.path.join(opt.dataset.dataset_path, "{people_id}/img/{camera}/{frame:04d}.jpg"),
            os.path.join(opt.dataset.dataset_path, "{people_id}/mask/{camera}/{frame:04d}.png"),
            os.path.join(opt.dataset.dataset_path, "{people_id}/mask/{camera}/{frame:04d}.png"),
            os.path.join(opt.dataset.dataset_path, "{people_id}/cameras.json"),
            transform=None, opt = opt, phase = phase
        )
        
        self.dataset_size = self.dataset_rp.__len__()

        self.start_idx = 0
        if self.opt.df.load_test_samples:
            samples_dir = self.opt.df.samples_dir

            samples = []
            self.diff_npz_file_list = []
           
            diff_pth = os.path.join(samples_dir, "0.npz")                
            s = np.load(diff_pth)                
            arr = s["arr_0"]
                            
            if arr.ndim == 3: arr = arr[None, ...]
            samples.append(arr)                        
                            
            samples = np.concatenate(samples, 0)
            self.diff_samples = samples
            self.dataset_size = self.diff_samples.shape[0]
            
    def get_train_sampler(self, world_size, rank):
        if self.opt.gaussian_weighted_sampler:                
            sampler = WeightedRandomSampler(self.dataset_rp.weights, len(self.dataset_rp.weights))
            if self.opt.training.distributed:
                self.train_sampler = DistributedSamplerWrapper(sampler, num_replicas=world_size, rank=rank)
            else:
                self.train_sampler = sampler
        else:
            self.train_sampler = data_sampler(self, shuffle=True, distributed = self.opt.training.distributed)

        return self.train_sampler
           
    def prepare_input(self, ret):
        # read xyz, normal, color from the ply file

        betas = ret["beta"]#.astype(np.float32).copy().reshape(10,)                
        poses_ = ret["theta"]#.astype(np.float32).copy()
        trans_ = ret["trans"] #smpl trans

        device= "cpu"
        xyz,_ = get_smpl_vertices(self.smpl_model, gender="neutral", pose_ = poses_, trans_=trans_, shape_= betas, device=device)
        
        nxyz = (xyz).copy()

        if self.opt.df.use_global_posemap:
            Rh = poses_.reshape(72,1)[:3].cpu().numpy() 
            R = np.eye(3)
            Th = trans_*0
            poses = poses_.clone().reshape(24,3)
            poses[:1] *= 0
        else:
            if not torch.is_tensor(poses_):
                poses_ = torch.from_numpy(poses_).float()

            if not torch.is_tensor(trans_):
                trans_ = torch.from_numpy(trans_).float()
            if not torch.is_tensor(betas):
                betas = torch.from_numpy(betas).float()

            Rh = poses_.reshape(72,1)[:3].cpu().numpy()            
            R = cv2.Rodrigues(Rh)[0].astype(np.float32)
            Th = trans_.cpu().numpy()
            
            poses = poses_.clone().reshape(24,3)
            poses[:1] *= 0
            smpl_to_world_R = np.linalg.inv(R)

            with torch.no_grad():
                out = self.smpl_model(betas.reshape(1,10).float())
                joint_0_pos = out.joints[0][0][None,...].detach().cpu().numpy()
        
            j0 = joint_0_pos
            Th = Th - np.dot(j0, smpl_to_world_R) + j0                    
            xyz = np.dot(xyz - Th, R)
            
        
        min_xyz = np.min(xyz, axis=0)
        max_xyz = np.max(xyz, axis=0)
        min_xyz[1] = min_xyz[1] - 0.1
        max_xyz[1] = max_xyz[1] + 0.1
        can_bounds = np.stack([min_xyz, max_xyz], axis=0)

        # transformation augmentation
        center = np.array([0, 0, 0]).astype(np.float32)
        rot = np.array([[np.cos(0), -np.sin(0)], [np.sin(0), np.cos(0)]])
        rot = rot.astype(np.float32)
        trans = np.array([0, 0, 0]).astype(np.float32)

        # obtain the bounds for coord construction
        min_xyz = np.min(xyz, axis=0)
        max_xyz = np.max(xyz, axis=0)

        min_xyz[1] = min_xyz[1] - 0.1
        max_xyz[1] = max_xyz[1] + 0.1
        bounds = np.stack([min_xyz, max_xyz], axis=0)

        cxyz = xyz.astype(np.float32)
        nxyz = nxyz.astype(np.float32)
        
        feature = np.concatenate([cxyz, nxyz, nxyz - trans_.cpu().numpy()], axis=1).astype(np.float32)

        # construct the coordinate
        dhw = xyz[:, [2, 1, 0]]
        min_dhw = min_xyz[[2, 1, 0]]
        max_dhw = max_xyz[[2, 1, 0]]
        voxel_size = np.array(self.opt.voxel_size)
        coord = np.round((dhw - min_dhw) / voxel_size).astype(np.int32)

        # construct the output shape
        out_sh = np.ceil((max_dhw - min_dhw) / voxel_size).astype(np.int32)
        x = 32
        out_sh = (out_sh | (x - 1)) + 1

        return feature, coord, out_sh, can_bounds, bounds, Rh, Th, center, rot, trans, betas, poses, poses_, trans_

    def __getitem__(self, index):        
        return self.get_item_packed(self.start_idx + index)

    def __len__(self):
        return self.dataset_size
    
    def get_item_packed(self, index):
        
        i = index
        ret_ = self.dataset_rp.__getitem__(i)
        
        H, W = ret_["org_img_size"]
        Cam_intri = ret_["intrinsics"]
        extrinsics = ret_["extrinsics"]
        if extrinsics.shape[0] == 3:
            extrinsics = extrinsics[None, ...]

        Cam_K = Cam_intri
        Cam_R = extrinsics[:, :3, :3]
        Cam_T = extrinsics[:, :3, 3]
        
        Cam_K_gen = np.copy(Cam_K)
        Cam_K_gen[:2] = Cam_K_gen[:2] * ret_["ratio"]

        Cam_K_nerf = Cam_K.copy().astype(np.float32)
        Cam_K_nerf[:2] = Cam_K_nerf[:2] / self.opt.nerf_ratio
                
        H_nerf, W_nerf = int(H / self.opt.nerf_ratio), int(W / self.opt.nerf_ratio)

        msk = ret_["msk"].astype(np.uint8)
        img_nerf = cv2.resize(ret_["img"], (W_nerf, H_nerf), interpolation = cv2.INTER_LANCZOS4)#, 
        msk_nerf = cv2.resize(msk, (W_nerf, H_nerf), interpolation=cv2.INTER_NEAREST)

        ret={}
        
        ret.update({

            'beta': ret_["beta"], 
            'theta': ret_["theta"],
            'trans': ret_["trans"],
            'img_gen': ret_["img"], 
            #"is_fliped": ret_["is_fliped"],
            "image_name": ret_["image_name"],
            "eva_index": ret_["eva_index"],

            'org_img_size': (H, W), #ret_['org_img_size']

            "msk_gen": msk, 
            'img_nerf': img_nerf,
            'mask_nerf': msk_nerf,
            'Cam_K_nerf': Cam_K_nerf,

            "extrinsics": extrinsics[0],
            "focal": 0,
            'Cam_K_gen': Cam_K_gen,            
            "phase": self.phase,

            'Cam_R': Cam_R.reshape(3,3),
            'Cam_T': Cam_T.reshape(3,1),
            "cam_ind": ret_["cam_ind"],
            "frame_index": ret_["image_name"],
            'index': ret_["index"],
            "frame_name": ret_["frame_name"]

        })
          
        smpl_vertices, coord, out_sh, can_bounds, bounds, Rh, Th, center, rot, trans, betas, poses, org_poses, org_trans = self.prepare_input(ret_)
     
        ret.update({
                "can_bounds": can_bounds,
                'smpl_vertices': smpl_vertices,
                'coord': coord,
                'out_sh': out_sh,
                'betas': betas, 
                'poses': poses,
                'bounds': bounds,
                'R': cv2.Rodrigues(Rh)[0].astype(np.float32) if Rh is not None else 0,        
                'Th': Th,
                'center': center,
                'smpl_trans': Th,
                'ratio': ret_["ratio"],
                'crop_bbox': ret_["crop_bbox"]                
            })
        
        if 'swap_id' in ret_.keys():
            ret.update({"swap_id": ret_["swap_id"]})
        
        if self.opt.df.load_test_samples:
            inter_index = (index - self.start_idx)
            pack_idx = int(inter_index / 64)            
            npz_idx, npz_name = 0, 0
            
            lat_inter_id = inter_index % 64
            org_img_name = ret_["frame_name"]
            new_img_name = org_img_name.split("_")[0]
            new_img_name += f"_{npz_idx}_{lat_inter_id}_{npz_name}"
            new_img_name += "_%s" % (org_img_name[org_img_name.find("_") + 1:])

            ret.update({"frame_name": new_img_name})
            try:    
                ret.update({"diff_samples": self.diff_samples[inter_index]})
            except Exception as e:
                printg(e)
                st()

        return ret
