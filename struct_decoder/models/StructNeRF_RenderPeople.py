import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import numpy as np
from functools import partial
from pdb import set_trace as st
from smplx.lbs import transform_mat, blend_shapes
from pytorch3d.ops.knn import knn_gather, knn_points

from .volume_renderer import SirenGenerator, SDFDecoder, SirenGeneratorNoView, SirenGeneratorVoxel
from .smpl_utils import init_smpl, get_J, get_shape_pose, batch_rodrigues


import sys
sys.path.append("..")
from Engine.th_utils.io.prints import *
from struct_decoder.util.smpl_renderer import SMPL_Renderer
from Engine.th_utils.animation.uv_generator import Index_UV_Generator

from Engine.th_utils import grid_sample_fix


class VoxelSDFRenderer(nn.Module):
    def __init__(self, opt, xyz_min, xyz_max, style_dim=256, mode='train', full_opt = None):
        super().__init__()
        self.xyz_min = torch.from_numpy(xyz_min).float().cuda()
        self.xyz_max = torch.from_numpy(xyz_max).float().cuda()
        self.test = mode != 'train'
        self.perturb = opt.perturb
        self.offset_sampling = not opt.no_offset_sampling # Stratified sampling used otherwise
        self.N_samples = opt.N_samples
        self.raw_noise_std = opt.raw_noise_std
        self.return_xyz = opt.return_xyz
        self.return_sdf = opt.return_sdf
        self.static_viewdirs = opt.static_viewdirs
        self.z_normalize = not opt.no_z_normalize
        self.force_background = opt.force_background
        self.with_sdf = not opt.no_sdf
        if 'no_features_output' in opt.keys():
            self.output_features = False
        else:
            self.output_features = True

        if self.with_sdf:
            self.sigmoid_beta = nn.Parameter(0.1 * torch.ones(1))

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
        self.input_ch_views = opt.input_ch_views
        self.feature_out_size = opt.width

        upsample_factor = full_opt.nerf_ratio // full_opt.gen_ratio
        upf = {
            1: 1,
            2: 1.5,
            4: 3
        }
        ch_ = {
            1: 128,
            2: 96,
            4: 64
        }

        self.output_ch = opt.output_ch            
        if opt.voxel_net:

            depth = int(opt.depth // upf[upsample_factor])
            width = ch_[upsample_factor] #int(opt.width // upf[upsample_factor])
            if depth <= 0: depth += 1

            depth = int(opt.depth // 4)
            if depth == 0: depth = 1
            width = full_opt.width

            self.network = SirenGeneratorVoxel(D=depth, W=width, style_dim=style_dim, input_ch=self.input_ch, output_ch = self.output_ch, input_ch_views=self.input_ch_views,
                output_features=self.output_features, is_style = not opt.cat_style, use_pose_encoding = not opt.no_pose_encoding, use_net = not opt.not_use_net, camera_cond = False) #
    
        else:
            depth = opt.depth #int(opt.depth // 2)

            if full_opt.dataset.dataset_name=="thuman2" and full_opt.df.noise_1d_eva:
                opt.width = 64 #full_opt.df.style_dim_1d
                style_dim = full_opt.df.style_dim_1d

                if depth == 8: depth -= 3
                elif depth == 6: depth -= 2
                elif depth == 4: depth -= 1

            width = opt.width
            self.network = SirenGenerator(D=depth, W=width, style_dim=style_dim, input_ch=self.input_ch,
                                        output_ch = self.output_ch, input_ch_views=self.input_ch_views,
                                        output_features=self.output_features)
    

      
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

    def render(self, rays, styles, return_eikonal=False):
        rgb, features, sdf, mask, xyz, eikonal_term = self.render_rays(rays, styles=styles, return_eikonal=return_eikonal)

        return rgb, features, sdf, mask, xyz, eikonal_term

    def mlp_init_pass(self, styles=None):
        rays_o, rays_d, viewdirs, near, far = None #TODO: hard coding the rays
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

    def forward(self, rays, styles=None, return_eikonal=False):
        rgb, features, sdf, mask, xyz, eikonal_term = self.render(rays, styles=styles, return_eikonal=return_eikonal)

        rgb = rgb.permute(0,3,1,2).contiguous()
        if self.output_features:
            features = features.permute(0,3,1,2).contiguous()

        if xyz != None:
            xyz = xyz.permute(0,3,1,2).contiguous()
            mask = mask.permute(0,3,1,2).contiguous()

        return rgb, features, sdf, mask, xyz, eikonal_term

        
class VoxelHuman(nn.Module):
    def __init__(self, opt, smpl_cfgs, style_dim, out_im_res=(128, 64), mode='train'):
        super(VoxelHuman, self).__init__()
        self.smpl_cfgs = smpl_cfgs
        self.style_dim = style_dim
                
        self.full_opt = opt.copy()
        self.df = opt.df.copy()
        self.gpu_ids = opt.gpu_ids[0]
        self.model_opt = opt.model.copy()

        self.img_size = self.model_opt.renderer_spatial_output_dim
        
        self.opt = opt.rendering.copy()
        opt = opt.rendering
        self.output_ch = opt.output_ch


        self.out_im_res = out_im_res
        self.is_train = (mode == 'train')
        self.stepsize = opt.stepsize
        self.with_sdf = not opt.no_sdf
        self.raw_noise_std = opt.raw_noise_std

        self.sigmoid_beta = nn.Parameter(0.1 * torch.ones(1))
        self.register_buffer('inf', torch.Tensor([1e10]), persistent=False)
        self.register_buffer('zero_idx', torch.LongTensor([0]), persistent=False)

        # create meshgrid to generate rays
        i, j = torch.meshgrid(torch.linspace(0.5, self.out_im_res[1] - 0.5, self.out_im_res[1]),
                              torch.linspace(0.5, self.out_im_res[0] - 0.5, self.out_im_res[0]))

        self.register_buffer('i', i.t().unsqueeze(0), persistent=False)
        self.register_buffer('j', j.t().unsqueeze(0), persistent=False)

        self.N_samples = opt.N_samples
        self.t_vals = torch.linspace(0., 1.-1/self.N_samples, steps=self.N_samples).view(-1).cuda()
        self.perturb = opt.perturb

        with open('./data/asset/smpl_template_sdf.npy', 'rb') as f:
            sdf_voxels = np.load(f)
        self.sdf_voxels = torch.from_numpy(sdf_voxels).reshape(1, 1, 128, 128, 128).cuda()
        self.sdf_voxels = self.sdf_voxels.permute(0, 1, 4, 3, 2)


        self.render_posmap = Index_UV_Generator(self.df.uv_reso, self.df.uv_reso, uv_type=self.df.uv_type, data_dir="./data/asset/uv_sampler")
        self.vts_uv = self.render_posmap.get_vts_uv().cuda().permute(0,2,1)
        self.vts_uv.requires_grad = False
        
        self.smpl_render = SMPL_Renderer(gender = "neutral", verts_uvs = self.render_posmap.verts_uvs, faces_uvs=self.render_posmap.faces_uvs, smpl_uv_vts = self.vts_uv)

        if self.df.tex_2dp:
            self.style_dim = style_dim = self.model_opt.texdecoder_outdim  
        elif self.df.tex_trip: 
            if self.df.tex_trip_c == self.model_opt.texdecoder_outdim // 3:
                self.style_dim = style_dim = self.df.tex_trip_c
            else:
                assert self.df.tex_trip_c  * 4 == self.model_opt.texdecoder_outdim  
                self.style_dim = style_dim = self.df.tex_trip_c * 2

        elif self.df.ablation_1d_direct:
            self.style_dim = style_dim = self.df.style_dim_1d

        if self.df.style_cond:
            self.style_dim += self.df.structured_dim
            style_dim += self.df.structured_dim

        # create human model
        zero_betas = torch.zeros(10)
        self.smpl_model = init_smpl(
            model_folder = smpl_cfgs['model_folder'],
            model_type = smpl_cfgs['model_type'],
            gender = smpl_cfgs['gender'],
            num_betas = smpl_cfgs['num_betas']
        )
        self.zero_init_J = get_J(zero_betas, self.smpl_model)
        parents = self.smpl_model.parents.cpu().numpy()
        self.parents = parents
        self.num_joints = num_joints = parents.shape[0]
        self.skeleton_children = np.zeros_like(parents) - 1
        for j in range(num_joints):
            pj = parents[j]
            if pj >= 0:
                self.skeleton_children[pj] = j

        self.vox_list = nn.ModuleList()
        self.vox_index = []
        self.smpl_index = []
        self.actual_vox_bbox = []

        self.smpl_seg = self.smpl_model.lbs_weights.argmax(-1)
        self.voxind2voxlist = {}

        for j in range(num_joints):
            pj = parents[j]
            j_coor = self.zero_init_J[j]
            pj_coor = self.zero_init_J[pj]
            mid = (j_coor + pj_coor) / 2.0

            xyz_min, xyz_max, cur_index = self.predefined_bbox(j)
            if xyz_min is None:
                continue
            xyz_min -= np.array([0.035, 0.035, 0.035])
            xyz_max += np.array([0.035, 0.035, 0.035])

            if j == 15:
                new_opt = opt.copy()
                if opt.is_aist:
                    new_opt.depth = 6
                else:
                    new_opt.depth = 8 
                new_opt.width = 128
                cur_vox = VoxelSDFRenderer(
                    new_opt, xyz_min, xyz_max, style_dim, mode=mode, full_opt = self.full_opt
                )
            elif j in [22, 23, 10, 11]:
                new_opt = opt.copy()
                new_opt.depth = 2
                new_opt.width = 128
                cur_vox = VoxelSDFRenderer(
                    new_opt, xyz_min, xyz_max, style_dim, mode=mode, full_opt = self.full_opt
                )
            elif j in [7, 8, 20, 21, 18, 19, 4, 5]:
                new_opt = opt.copy()
                new_opt.depth = 3
                new_opt.width = 128
                cur_vox = VoxelSDFRenderer(
                    new_opt, xyz_min, xyz_max, style_dim, mode=mode, full_opt = self.full_opt
                )
            else:
                new_opt = opt.copy()
                if opt.is_aist:
                    new_opt.depth = 3
                else:
                    new_opt.depth = 4
                new_opt.width = 128
                cur_vox = VoxelSDFRenderer(
                    new_opt, xyz_min, xyz_max, style_dim, mode=mode, full_opt = self.full_opt
                )
            
            self.vox_list.append(cur_vox)
            self.vox_index.append(j)
            self.smpl_index.append(cur_index)
            self.voxind2voxlist[j] = len(self.vox_list) - 1
            if j == 12:
                self.voxind2voxlist[16] = len(self.vox_list) - 1
                self.voxind2voxlist[17] = len(self.vox_list) - 1
                self.voxind2voxlist[13] = len(self.vox_list) - 1
                self.voxind2voxlist[14] = len(self.vox_list) - 1
            if j == 3:
                self.voxind2voxlist[1] = len(self.vox_list) - 1
                self.voxind2voxlist[2] = len(self.vox_list) - 1

    def compute_actual_bbox(self, beta, scale = 1):
        actual_vox_bbox = []
        init_J = get_J(beta.reshape(1, 10), self.smpl_model) * scale
        for j in range(self.num_joints):
            pj = self.parents[j]
            j_coor = init_J[j]
            pj_coor = init_J[pj]
            mid = (j_coor + pj_coor) / 2.0

            # spine direction
            if j in [15, 12, 6, 3]:
                h = np.abs(j_coor[1] - pj_coor[1])
                w = 0.3
                delta = np.array([w, 0.8 * h, w])

            elif j in [4, 5, 7, 8]:
                h = np.abs(j_coor[1] - pj_coor[1])
                w = 0.15
                delta = np.array([w, 0.6 * h, w])

            # arms direction
            elif j in [22, 20, 18, 23, 21, 19]:
                h = np.abs(j_coor[0] - pj_coor[0])
                w = 0.12
                delta = np.array([0.8 * h, w, w])

            # foot direction
            elif j in [10, 11]:
                h = np.abs(j_coor[2] - pj_coor[2])
                w = 0.08
                delta = np.array([w, w, 0.6 * h])

            else:
                continue

            xyz_min = mid - delta
            xyz_max = mid + delta

            if j == 15:
                xyz_max += np.array([0, 0.25, 0])
            elif j == 22:
                xyz_max += np.array([0.25, 0, 0])
            elif j == 23:
                xyz_min -= np.array([0.25, 0, 0])
            elif j == 3:
                xyz_min -= np.array([0, 0.1, 0])
            elif j == 12:
                xyz_min -= np.array([0, 0.25, 0])

            actual_vox_bbox.append((torch.from_numpy(xyz_min).float().cuda(), \
                                    torch.from_numpy(xyz_max).float().cuda()))

        return actual_vox_bbox

    def smpl_index_by_joint(self, joint_list):
        start_index = self.smpl_seg == joint_list[0]
        if len(joint_list) > 1:
            for i in range(1, len(joint_list)):
                start_index += self.smpl_seg == joint_list[i]
            return start_index > 0
        else:
            return start_index

    def predefined_bbox(self, j, only_cur_index=False):
        if j == 15:
            xyz_min = np.array([-0.0901, 0.2876, -0.0891])
            xyz_max = np.array([0.0916, 0.5555+0.04, 0.1390])
            xyz_min -= np.array([0.05, 0.05, 0.05])
            xyz_max += np.array([0.05, 0.05, 0.05])
            cur_index = self.smpl_index_by_joint([15])
        elif j == 12:
            xyz_min = np.array([-0.1752, 0.0208, -0.1198]) # combine 12 and 9
            xyz_max = np.array([0.1724, 0.2876, 0.1391])
            cur_index = self.smpl_index_by_joint([9, 13, 14, 6, 16, 17, 12, 15])
        elif j == 9 and only_cur_index:
            xyz_min = None
            xyz_max = None
            cur_index = self.smpl_index_by_joint([9, 13, 14, 6, 16, 17, 3])
        elif j == 6:
            xyz_min = np.array([-0.1569, -0.1144, -0.1095])
            xyz_max = np.array([0.1531, 0.0208, 0.1674])
            cur_index = self.smpl_index_by_joint([3, 6, 0, 9])
        elif j == 3:
            xyz_min = np.array([-0.1888, -0.3147, -0.1224])
            xyz_max = np.array([0.1852, -0.1144, 0.1679])
            cur_index = self.smpl_index_by_joint([3, 0, 1, 2, 6])
        elif j == 18:
            xyz_min = np.array([0.1724, 0.1450, -0.0750])
            xyz_max = np.array([0.4321, 0.2758, 0.0406])
            cur_index = self.smpl_index_by_joint([13, 18, 16])
        elif j == 20:
            xyz_min = np.array([0.4321, 0.1721, -0.0753])
            xyz_max = np.array([0.6813, 0.2668, 0.0064])
            cur_index = self.smpl_index_by_joint([16, 20, 18])
        elif j == 22:
            xyz_min = np.array([0.6813, 0.1882, -0.1180])
            xyz_max = np.array([0.8731, 0.2445, 0.0461])
            cur_index = self.smpl_index_by_joint([22, 20, 18])
        elif j == 19:
            xyz_min = np.array([-0.4289, 0.1426, -0.0785])
            xyz_max = np.array([-0.1752, 0.2754, 0.0460])
            cur_index = self.smpl_index_by_joint([14, 17, 19])
        elif j == 21:
            xyz_min = np.array([-0.6842, 0.1705, -0.0780])
            xyz_max = np.array([-0.4289, 0.2659, 0.0059])
            cur_index = self.smpl_index_by_joint([17, 19, 21])
        elif j == 23:
            xyz_min = np.array([-0.8720, 0.1839, -0.1195])
            xyz_max = np.array([-0.6842, 0.2420, 0.0465])
            cur_index = self.smpl_index_by_joint([23, 21, 19])
        elif j == 4:
            xyz_min = np.array([0, -0.6899, -0.0849])
            xyz_max = np.array([0.1893, -0.3147, 0.1335])
            cur_index = self.smpl_index_by_joint([0, 1, 4])
        elif j == 7:
            xyz_min = np.array([0.0268, -1.0879, -0.0891])
            xyz_max = np.array([0.1570, -0.6899, 0.0691])
            cur_index = self.smpl_index_by_joint([4, 1, 7])
        elif j == 10:
            xyz_min = np.array([0.0625, -1.1591-0.04, -0.0876])
            xyz_max = np.array([0.1600, -1.0879+0.02, 0.1669])
            cur_index = self.smpl_index_by_joint([7, 10, 4])
        elif j == 5:
            xyz_min = np.array([-0.1935, -0.6964, -0.0883])
            xyz_max = np.array([0, -0.3147, 0.1299])
            cur_index = self.smpl_index_by_joint([0, 2, 5])
        elif j == 8:
            xyz_min = np.array([-0.1611, -1.0948, -0.0911])
            xyz_max = np.array([-0.0301, -0.6964, 0.0649])
            cur_index = self.smpl_index_by_joint([2, 5, 8])
        elif j == 11:
            xyz_min = np.array([-0.1614, -1.1618-0.04, -0.0882])
            xyz_max = np.array([-0.0632, -1.0948+0.02, 0.1680])
            cur_index = self.smpl_index_by_joint([8, 11, 5])
        else:
            xyz_min = xyz_max = cur_index = None

        if only_cur_index:
            return cur_index

        return xyz_min, xyz_max, cur_index

    def batch_rigid_transform(self, rot_mats, init_J):
        joints = torch.from_numpy(init_J.reshape(1, -1, 3, 1)).cuda()
        parents = self.parents

        rel_joints = joints.clone()
        rel_joints[:, 1:] -= joints[:, parents[1:]]

        transforms_mat = transform_mat(
            rot_mats.reshape(-1, 3, 3),
            rel_joints.reshape(-1, 3, 1)).reshape(-1, joints.shape[1], 4, 4)

        transform_chain = [transforms_mat[:, 0]]
        for i in range(1, parents.shape[0]):
            curr_res = torch.matmul(transform_chain[parents[i]],
                                    transforms_mat[:, i])
            transform_chain.append(curr_res)

        transforms = torch.stack(transform_chain, dim=1)

        posed_joints = transforms[:, :, :3, 3]

        joints_homogen = F.pad(joints, [0, 0, 0, 1])

        rel_transforms = transforms - F.pad(
            torch.matmul(transforms, joints_homogen), [3, 0, 0, 0, 0, 0, 0, 0])

        return posed_joints, rel_transforms

    def transform_to_vox_local(self, rays_o, rays_d, transforms_mat, trans, scale = 1):
        inv_transforms_mat = torch.inverse(transforms_mat)[:3, :3].unsqueeze(0)
        rays_o_local = (rays_o - trans) / scale
        rays_o_local = torch.matmul(inv_transforms_mat, (rays_o_local - transforms_mat[:3, -1]).unsqueeze(-1))[..., 0]
        rays_d_local = torch.matmul(inv_transforms_mat, rays_d.unsqueeze(-1))[..., 0]

        return rays_o_local, rays_d_local

    def forward_transform_bbox(self, i, transforms_mat, xyz_min, xyz_max):
        # xyz_min = self.vox_list[i].xyz_min
        # xyz_max = self.vox_list[i].xyz_max
        new_xyz_min = torch.matmul(transforms_mat[:3, :3], xyz_min.reshape(3, 1))[..., 0] + transforms_mat[:3, -1]
        new_xyz_max = torch.matmul(transforms_mat[:3, :3], xyz_max.reshape(3, 1))[..., 0] + transforms_mat[:3, -1]
        return new_xyz_min.detach().cpu().numpy(), new_xyz_max.detach().cpu().numpy()

    def sample_ray_bbox_intersect(self, rays_o, rays_d, xyz_min, xyz_max):
        _rays_d = rays_d / rays_d.norm(dim=-1, keepdim=True)
        vec = torch.where(_rays_d==0, torch.full_like(_rays_d, 1e-6), _rays_d)
        rate_a = (xyz_max - rays_o) / vec
        rate_b = (xyz_min - rays_o) / vec
        t_min = torch.minimum(rate_a, rate_b).amax(-1)
        t_max = torch.maximum(rate_a, rate_b).amin(-1)
        mask_outbbox = (t_max <= t_min)
        t_min[mask_outbbox] = 10086
        t_max[mask_outbbox] = -10086
        return t_min.view(-1), t_max.view(-1), mask_outbbox.view(-1)

    def get_nearest_pts_uvh(self, smpl_v, sampled_pts_smpl_space):
        
        near_uv, h = self.smpl_render.get_nearest_pts_in_mesh(smpl_v, sampled_pts_smpl_space, k=1, num_samples = self.full_opt.uvVol_smpl_pts)

        return torch.cat((near_uv, h), 2)


    def sample_pts(self, batch, is_high = False):
        
        split = batch["phase"]
               
        img_label="img_nerf"
        msk_label="mask_nerf"
        cam_k_label="Cam_K_nerf"
        
        if is_high:
            img_label = "h_img_nerf"
            msk_label = "h_mask_nerf"
            cam_k_label = "h_Cam_K_nerf"
            
        batch_size = batch[img_label].shape[0]
        device = batch[img_label].device
        
        rgb_list = [] 
        ray_o_list = [] 
        ray_d_list = [] 
        near_list = [] 
        far_list = [] 
        coord_list = []
        mask_at_box_list = []
        for i in range(batch_size):

            if self.opt.dataset.dataset_name != "h36":
                rgb_i, ray_o_i, ray_d_i, near_i, far_i, coord_i, mask_at_box_i = if_nerf_dutils.sample_ray(batch[img_label][i].cpu().numpy(), batch[msk_label][i].cpu().numpy(), batch[cam_k_label][i].cpu().numpy(), batch["Cam_R"][i].cpu().numpy(), batch["Cam_T"][i].cpu().numpy(), batch["can_bounds"][i].cpu().numpy(), self.opt.rendering.nrays, split[i], self.opt, is_high)
            
            rgb_list.append(rgb_i[None,...])
            ray_o_list.append(ray_o_i[None,...])
            ray_d_list.append(ray_d_i[None,...])
            near_list.append(near_i[None,...])
            far_list.append(far_i[None,...])
            coord_list.append(coord_i[None,...])
            mask_at_box_list.append(mask_at_box_i[None,...])            

        rgb = torch.from_numpy(np.concatenate(rgb_list, 0)).float().to(device)
        ray_o  = torch.from_numpy(np.concatenate(ray_o_list, 0)).float().to(device)
        ray_d = torch.from_numpy(np.concatenate(ray_d_list, 0)).float().to(device)
        near = torch.from_numpy(np.concatenate(near_list, 0)).float().to(device)
        far = torch.from_numpy(np.concatenate(far_list, 0)).float().to(device)
        coord_ = torch.from_numpy(np.concatenate(coord_list, 0)).long().to(device)
        mask_at_box = torch.from_numpy(np.concatenate(mask_at_box_list, 0)).bool().to(device)

        batch["ray_coord"] = coord_.long()
        batch["rgb"] = rgb
        batch["mask_at_box"] = mask_at_box
        
        batch["ray_o"] = ray_o
        batch["ray_d"] = ray_d
        batch["near"] = near
        batch["far"] = far

    def get_rays_camera(self, H, W, K, R, T):
        #def get_rays(H, W, K, R, T):
        # calculate the camera origin

        rays_o = -np.dot(R.T, T).ravel()

        #print(T, rays_o, R, K)
        #exit()

        # calculate the world coodinates of pixels
        i, j = np.meshgrid(np.arange(W, dtype=np.float32),
                        np.arange(H, dtype=np.float32),
                        indexing='xy')
        xy1 = np.stack([i, j, np.ones_like(i)], axis=2)
        pixel_camera = np.dot(xy1, np.linalg.inv(K).T)
        pixel_world = np.dot(pixel_camera - T.ravel(), R)
        # calculate the ray direction

        rays_d = pixel_world - rays_o[None, None]
        rays_o = np.broadcast_to(rays_o, rays_d.shape)
        return rays_o, rays_d
        
    def sample_ray(self, beta, theta, trans, rays_o, rays_d, K=3, smpl_vts_wrd = None, scale = 1):
        _theta = theta.reshape(1, 24, 3, 3)
        #global smpl
        if smpl_vts_wrd is not None:
            smpl_v = smpl_vts_wrd.reshape(-1, 3)
        else:
            so = self.smpl_model(betas = beta.reshape(1, 10), body_pose = _theta[:, 1:], global_orient = _theta[:, 0].view(1, 1, 3, 3))
            smpl_v = so['vertices'].clone().reshape(-1, 3) * scale
            del so

        init_J = get_J(beta.reshape(1, 10), self.smpl_model) * scale 

        # first sample ray pts for each voxel
        rays_pts_local_list = []
        rays_d_pts_local_list = []
        mask_outbbox_list = []
        forward_skinning_transformation_list = []

        _, rel_transforms = self.batch_rigid_transform(theta, init_J)
        # inv_rel_transforms = torch.inverse(rel_transforms)

        shape_blend_shapes = blend_shapes(beta.reshape(1, 10), self.smpl_model.shapedirs).view(-1, 3)
        shape_blend_shapes = ((scale - 1) * self.smpl_model.v_template).view(-1, 3) + scale * shape_blend_shapes


        flat_rays_pts_global_list = []

        
        ident = torch.eye(3).cuda()
        pose_feature = (theta[:, 1:].view(1, -1, 3, 3) - ident).view(1, -1)
        pose_blend_shapes = torch.matmul(pose_feature, self.smpl_model.posedirs).view(-1, 3)
        all_blend_shapes = pose_blend_shapes + shape_blend_shapes
        inv_shape_transforms = torch.zeros([smpl_v.shape[0], 4, 4]).cuda()
        inv_shape_transforms[:, ...] = torch.eye(4)
        inv_shape_transforms[:, :3, -1] = -all_blend_shapes
        actual_vox_bbox = self.compute_actual_bbox(beta, scale)

        t_min_list = []
        t_max_list = []
        mask_outbbox_list = []
        bbox_transformation_list = []

        for i, cur_vox in enumerate(self.vox_list):
            vox_i = self.vox_index[i]
            if vox_i == 15:
                cur_transforms_mat = rel_transforms[0, vox_i]
            elif vox_i == 12:
                cur_transforms_mat = rel_transforms[0, 6]
            else:
                cur_transforms_mat = rel_transforms[0, self.parents[vox_i]]
            
            rays_o_local, rays_d_local = self.transform_to_vox_local(rays_o, rays_d, cur_transforms_mat, trans)
            bbox_transformation_list.append(cur_transforms_mat)

            cur_xyz_min, cur_xyz_max = actual_vox_bbox[i]
            cur_t_min, cur_t_max, cur_mask_outbbox = self.sample_ray_bbox_intersect(
                rays_o_local, rays_d_local, cur_xyz_min, cur_xyz_max
            )
            t_min_list.append(cur_t_min)
            t_max_list.append(cur_t_max)
            mask_outbbox_list.append(cur_mask_outbbox)

        ### cumulate t_min, t_max, mask_oubbox for all vox
        all_t_min = torch.stack(t_min_list, -1)
        all_t_max = torch.stack(t_max_list, -1)
        all_mask_outbbox = torch.stack(mask_outbbox_list, -1)
        t_min = torch.min(all_t_min, -1)[0]
        t_max = torch.max(all_t_max, -1)[0]
        # st()
        mask_outbbox = torch.all(all_mask_outbbox, -1)
        assert torch.all((t_min == 10086) == (t_max == -10086))
        assert torch.all((t_min == 10086) == mask_outbbox)
        
        valid_t_min = t_min[~mask_outbbox].view(-1, 1)
        valid_t_max = t_max[~mask_outbbox].view(-1, 1)
        valid_mask_outbbox_list = [
            m[~mask_outbbox] for m in mask_outbbox_list
        ]
        z_vals = valid_t_min * (1. - self.t_vals.view(1, -1)) + valid_t_max * self.t_vals.view(1, -1)
        if self.perturb > 0:
            upper = torch.cat([z_vals[...,1:], valid_t_max], -1)
            lower = z_vals.detach()
            t_rand = torch.rand(*z_vals.shape).to(z_vals.device)
            z_vals = lower + (upper - lower) * t_rand
        _rays_d = rays_d / rays_d.norm(dim=-1, keepdim=True)
        rays_pts_global = rays_o[~mask_outbbox].unsqueeze(1) + _rays_d[~mask_outbbox].unsqueeze(1) * z_vals.view(-1, self.N_samples, 1)
        rays_d_per_pts = _rays_d[~mask_outbbox].unsqueeze(1).repeat(1, self.N_samples, 1)
        assert rays_d_per_pts.shape == rays_pts_global.shape

        K = 1

        smpl_v_inv = torch.matmul(self.smpl_model.lbs_weights.reshape(-1, self.num_joints), rel_transforms.reshape(1, self.num_joints, 16)).reshape(-1, 4, 4)
        smpl_v_inv = torch.inverse(smpl_v_inv)

        uvh_list = []

        debug_uvh = True

        for i, cur_vox in enumerate(self.vox_list):
            vox_i = self.vox_index[i]
            cur_transforms_mat = rel_transforms[0, self.parents[vox_i]]

            _rays_pts_local = torch.zeros_like(rays_pts_global)
            _rays_d_pts_local = torch.zeros_like(rays_pts_global)

            ### extract current related smpl vertices
            cur_smpl_v = smpl_v[self.smpl_index[i], ...] + trans
            cur_blend_weights = self.smpl_model.lbs_weights[self.smpl_index[i], ...].reshape(-1, self.num_joints)

            debug = False
            if debug:
                printy(i, self.smpl_index[i].shape, cur_smpl_v.shape)

            flat_rays_pts_global = rays_pts_global[~valid_mask_outbbox_list[i]].reshape(1, -1, 3)
            flat_rays_d_pts_global = rays_d_per_pts[~valid_mask_outbbox_list[i]].reshape(1, -1, 3)
            # flat_rays_pts_global = torch.cat([cur_smpl_v.reshape(1, -1, 3), flat_rays_pts_global], 1)
            nn = knn_points(flat_rays_pts_global, cur_smpl_v.reshape(1, -1, 3), K=K)
        
            ### calculate interpolation weights #TODO: pamir has a better version
            interp_weights = 1 / nn.dists.reshape(1, -1, K, 1, 1)
            interp_weights[torch.where(torch.isinf(interp_weights))] = 100086
            interp_weights = interp_weights / interp_weights.sum(-3, keepdim=True)


            ### pamir inverse transformation
            ## BUG!!!!!
            # per_point_inv_transformation = torch.matmul(cur_blend_weights, inv_rel_transforms.reshape(1, self.num_joints, 16)).reshape(-1, 4, 4)
            # cur_inv_shape_transforms = inv_shape_transforms[self.smpl_index[i], ...].reshape(-1, 4, 4)
            # per_point_inv_transformation = torch.matmul(cur_inv_shape_transforms, per_point_inv_transformation)

            ### new-way of calculating per-point inv transformation
            # per_point_transformation = torch.matmul(cur_blend_weights, rel_transforms.reshape(1, self.num_joints, 16)).reshape(-1, 4, 4)
            per_point_inv_transformation = smpl_v_inv[self.smpl_index[i], ...].reshape(-1, 4, 4)

            cur_inv_shape_transforms = inv_shape_transforms[self.smpl_index[i], ...].reshape(-1, 4, 4)
            per_point_inv_transformation = torch.matmul(cur_inv_shape_transforms, per_point_inv_transformation)
            gather_inv_T = torch.gather(per_point_inv_transformation.reshape(1, -1, 1, 4, 4).repeat(1, 1, K, 1, 1), 1, nn.idx.reshape(1, -1, K, 1, 1).repeat(1, 1, 1, 4, 4))
            inv_T = (gather_inv_T * interp_weights).sum(-3).reshape(1, -1, 4, 4)

            homogen_coord = torch.ones([1, flat_rays_pts_global.shape[1], 1], dtype=rays_pts_global.dtype, device=rays_pts_global.device)
            #flat_rays_pts_global = flat_rays_pts_global - trans
            flat_rays_pts_global_homo = torch.cat([(flat_rays_pts_global - trans), homogen_coord], dim=2)
            rays_pts_local = torch.matmul(inv_T, torch.unsqueeze(flat_rays_pts_global_homo, dim=-1))[:, :, :3, 0]
            rays_d_pts_local = torch.matmul(inv_T[:, :, :3, :3], flat_rays_d_pts_global.unsqueeze(-1))[:, :, :3, 0]

            _rays_pts_local[~valid_mask_outbbox_list[i]] = rays_pts_local.view(-1, self.N_samples, 3)
            _rays_d_pts_local[~valid_mask_outbbox_list[i]] = rays_d_pts_local.view(-1, self.N_samples, 3)
            _rays_pts_local = _rays_pts_local.view(*rays_pts_global.shape)
            _rays_d_pts_local = _rays_d_pts_local.view(*rays_pts_global.shape)
            rays_pts_local_list.append(_rays_pts_local)
            rays_d_pts_local_list.append(_rays_d_pts_local)

            ### calculcate forward skinning transformation
            per_point_forward_transformation = torch.matmul(cur_blend_weights, rel_transforms.reshape(1, self.num_joints, 16)).reshape(-1, 4, 4)
            gather_forward_T = torch.gather(per_point_forward_transformation.reshape(1, -1, 1, 4, 4).repeat(1, 1, K, 1, 1), 1, nn.idx.reshape(1, -1, K, 1, 1).repeat(1, 1, 1, 4, 4))
            forward_T = (gather_forward_T * interp_weights).sum(-3).reshape(-1, 4, 4)
            # _rays_pts_local_forward_T[~mask_outbbox, :, :] = forward_T
            forward_skinning_transformation_list.append(forward_T)


            if True:
                if self.df.tex_2dp or self.df.tex_trip: #! trans

                    if flat_rays_pts_global.shape[1] ==0:
                        uvh_list.append(None)
                    else:
                        
                        _cur_xyz = rays_pts_local.view(-1, 3).detach() #smpl space
                        cur_xyz_min = self.vox_list[i].xyz_min.detach()
                        cur_xyz_max = self.vox_list[i].xyz_max.detach()

                        cur_new_mask = (_cur_xyz <= cur_xyz_max).sum(-1) + (_cur_xyz >= cur_xyz_min).sum(-1)
                        cur_new_mask = (cur_new_mask == 6)
                        
                        rays_pts_global_tmp = flat_rays_pts_global.view(-1, 3)[cur_new_mask].view(1, -1, 3)

                        if rays_pts_global_tmp.shape[1] == 0:            
                            uvh_list.append(None)
                        else:
                            use_torch = not self.df.org_nn
                            if use_torch:
                                if self.df.knn_part:
                                    part_uv = self.vts_uv[0].permute(1,0)
                                    part_uv = part_uv[self.smpl_index[i], ...][None, ...]#.permute(0,2,1)
                                    near_uv = self.smpl_render.get_nearest_pts_in_mesh_torch_part_2d(cur_smpl_v, rays_pts_global_tmp, k=1, num_samples = 1, uv = part_uv)
                                    h = torch.zeros(near_uv.shape)[..., :1].to(near_uv).float()
                                else:                            
                                    near_uv, h = self.smpl_render.get_nearest_pts_in_mesh_torch(smpl_v + trans, rays_pts_global_tmp, k=1, num_samples = self.full_opt.uvVol_smpl_pts)
                            else:
                                near_uv, h = self.smpl_render.get_nearest_pts_in_mesh(smpl_v + trans, rays_pts_global_tmp, k=1, num_samples = self.full_opt.uvVol_smpl_pts)

                            uvh = torch.cat((near_uv, h), 2) #.view(-1, 3)[cur_new_mask].view(1, -1, 3)
                            uvh_list.append(uvh)



        return rays_pts_local_list, rays_pts_global, mask_outbbox_list, valid_mask_outbbox_list, \
        mask_outbbox, forward_skinning_transformation_list, z_vals, rays_d_pts_local_list, bbox_transformation_list, smpl_v, actual_vox_bbox, uvh_list #flat_rays_pts_global_list, (smpl_v + trans).detach() #(False) #flat_rays_pts_global_list

    def get_rays(self, focal, c2w):
        dirs = torch.stack([(self.i - self.out_im_res[1] * .5) / focal[0],
                            (self.j - self.out_im_res[0] * .5) / focal[0],
                            torch.ones_like(self.i).expand(1,self.out_im_res[0], self.out_im_res[1])], -1).repeat(focal.shape[0], 1, 1, 1)

        # Rotate ray directions from camera frame to the world frame
        rays_d = torch.sum(dirs[..., None, :] * c2w[:,None,None,:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]

        # Translate camera frame's origin to the world frame. It is the origin of all rays.
        rays_o = c2w[:,None,None,:3,-1].expand(rays_d.shape)
        viewdirs = rays_d

        return rays_o, rays_d, viewdirs

    def sdf_activation(self, input):
        self.sigmoid_beta.data.copy_(max(torch.zeros_like(self.sigmoid_beta.data) + 2e-3, self.sigmoid_beta.data))
        sigma = torch.sigmoid(input / self.sigmoid_beta) / self.sigmoid_beta
        return sigma

    def marching_cube(self, styles, resolution=128, size=1):
        assert self.opt.input_ch_views == 3
        x, y, z = torch.meshgrid(torch.linspace(-size, size, resolution),
                                 torch.linspace(-size, size, resolution),
                                 torch.linspace(-size, size, resolution))
        pts = torch.stack([x, y, z], -1).view(-1, 3) # r x r x r x 3 => r^3 x 3
        sdf = torch.zeros_like(pts[..., 0])
        counter = torch.zeros_like(pts[..., 0])

        chunk = 300**3
        split_pts = torch.split(pts, chunk)
        split_sdf = torch.split(sdf, chunk)
        split_counter = torch.split(counter, chunk)

        counter_list = []
        sdf_list = []
        for i_chunk in range(len(split_pts)):
            cur_pts = split_pts[i_chunk].cuda()
            cur_sdf = split_sdf[i_chunk].cuda()
            cur_counter = split_counter[i_chunk].cuda()
            print("Marching Cube: {}/{}".format(i_chunk, len(split_pts)))

            for i, cur_vox in enumerate(self.vox_list):
                with torch.no_grad():
                    cur_xyz_min, cur_xyz_max = cur_vox.xyz_min, cur_vox.xyz_max
                    cur_xyz_min = cur_xyz_min.view(1, 3)
                    cur_xyz_max = cur_xyz_max.view(1, 3)

                    cur_pts_ind = (cur_pts <= cur_xyz_max).sum(-1) + (cur_pts >= cur_xyz_min).sum(-1)
                    cur_pts_ind = (cur_pts_ind == 6)
                    mask_cur_pts = cur_pts[cur_pts_ind, ...].view(1, -1, 3)

                    query_pts = mask_cur_pts.reshape(1, 1, 1, -1, 3) / 1.3
                    template_sdf = torch.nn.functional.grid_sample(
                        self.sdf_voxels, query_pts,
                        padding_mode = 'border', align_corners = True
                    ).reshape(-1)

                    window_alpha = 4; window_beta = 8
                    mask_cur_pts = (mask_cur_pts - (cur_vox.xyz_min + cur_vox.xyz_max) / 2.0) / (cur_vox.xyz_max - cur_vox.xyz_min)
                    weights = torch.exp(-window_alpha * ((mask_cur_pts * 2) ** window_beta).sum(-1))
                    fake_d = torch.zeros_like(mask_cur_pts)
                    fake_d[..., 0] = 1
                    cur_input = torch.cat([
                        mask_cur_pts, fake_d
                    ], -1)
                    cur_sdf[cur_pts_ind] += (cur_vox.network(
                        cur_input, styles=styles
                    ).view(-1, 4)[..., -1] + template_sdf) * weights.view(-1)
                    cur_counter[cur_pts_ind] += weights.view(-1)

            counter_list.append(cur_counter.detach().cpu())
            sdf_list.append(cur_sdf.detach().cpu())

        sdf = torch.cat(sdf_list, 0)
        counter = torch.cat(counter_list, 0)

        mask = counter == 0
        sdf[~mask] /= counter[~mask]
        if self.with_sdf:
            sdf[mask] = 1
        return sdf.view(resolution, resolution, resolution)

    def marching_cube_posed(self, styles, beta, theta, resolution=128, size=1, w_space=False, gamma_list=None, beta_list=None):
        # assert self.opt.input_ch_views == 3
        x, y, z = torch.meshgrid(torch.linspace(-size, size, resolution),
                                 torch.linspace(-size, size, resolution),
                                 torch.linspace(-size, size, resolution))
        pts = torch.stack([x, y, z], -1).view(-1, 3) # r x r x r x 3 => r^3 x 3
        sdf = torch.zeros_like(pts[..., 0])
        counter = torch.zeros_like(pts[..., 0])

        chunk = 300**3
        split_pts = torch.split(pts, chunk)
        split_sdf = torch.split(sdf, chunk)
        split_counter = torch.split(counter, chunk)

        counter_list = []
        sdf_list = []

        theta = batch_rodrigues(theta.reshape(-1, 3)).reshape(1, 24, 3, 3)
        _theta = theta.reshape(1, 24, 3, 3)
        so = self.smpl_model(betas = beta.reshape(1, 10), body_pose = _theta[:, 1:], global_orient = _theta[:, 0].view(1, 1, 3, 3))
        smpl_v = so['vertices'].clone().reshape(-1, 3)
        del so
        init_J = get_J(beta.reshape(1, 10), self.smpl_model)
        _, rel_transforms = self.batch_rigid_transform(theta, init_J)
        actual_vox_bbox = self.compute_actual_bbox(beta)
        shape_blend_shapes = blend_shapes(beta.reshape(1, 10), self.smpl_model.shapedirs).view(-1, 3)
        ident = torch.eye(3).cuda()
        pose_feature = (theta[:, 1:].view(1, -1, 3, 3) - ident).view(1, -1)
        pose_blend_shapes = torch.matmul(pose_feature, self.smpl_model.posedirs).view(-1, 3)
        all_blend_shapes = pose_blend_shapes + shape_blend_shapes
        inv_shape_transforms = torch.zeros([smpl_v.shape[0], 4, 4]).cuda()
        inv_shape_transforms[:, ...] = torch.eye(4)
        inv_shape_transforms[:, :3, -1] = -all_blend_shapes
        smpl_v_inv = torch.matmul(self.smpl_model.lbs_weights.reshape(-1, self.num_joints), rel_transforms.reshape(1, self.num_joints, 16)).reshape(-1, 4, 4)
        smpl_v_inv = torch.inverse(smpl_v_inv)
        K = 8
        
        styles_input = styles

        for i_chunk in range(len(split_pts)):
            cur_pts = split_pts[i_chunk].cuda()
            cur_sdf = split_sdf[i_chunk].cuda()
            cur_counter = split_counter[i_chunk].cuda()
            print("Marching Cube: {}/{}".format(i_chunk, len(split_pts)))

            for i, cur_vox in enumerate(self.vox_list):
                with torch.no_grad():
                    vox_i = self.vox_index[i]
                    if vox_i == 15:
                        cur_transforms_mat = rel_transforms[0, vox_i]
                    elif vox_i == 12:
                        cur_transforms_mat = rel_transforms[0, 6]
                    else:
                        cur_transforms_mat = rel_transforms[0, self.parents[vox_i]]
                    inv_transforms_mat = torch.inverse(cur_transforms_mat)[:3, :3].unsqueeze(0)
                    trans_cur_pts = torch.matmul(inv_transforms_mat, (cur_pts - cur_transforms_mat[:3, -1]).unsqueeze(-1))[..., 0]

                    cur_xyz_min, cur_xyz_max = actual_vox_bbox[i]
                    cur_xyz_min = cur_xyz_min.view(1, 3)
                    cur_xyz_max = cur_xyz_max.view(1, 3)

                    cur_pts_ind = (trans_cur_pts <= cur_xyz_max).sum(-1) + (trans_cur_pts >= cur_xyz_min).sum(-1)
                    cur_pts_ind = (cur_pts_ind == 6)
                    mask_cur_pts = trans_cur_pts[cur_pts_ind, ...].view(1, -1, 3)
                    if mask_cur_pts.shape[1] == 0:
                        continue

                    mask_cur_pts = torch.matmul(cur_transforms_mat[:3, :3].unsqueeze(0), mask_cur_pts.squeeze(0).unsqueeze(-1))[..., 0] + cur_transforms_mat[:3, -1]
                    mask_cur_pts = mask_cur_pts.unsqueeze(0)
                    cur_smpl_v = smpl_v[self.smpl_index[i], ...]
                    cur_blend_weights = self.smpl_model.lbs_weights[self.smpl_index[i], ...].reshape(-1, self.num_joints)
                    nn = knn_points(mask_cur_pts, cur_smpl_v.reshape(1, -1, 3), K=K)
                    interp_weights = 1 / nn.dists.reshape(1, -1, K, 1, 1)


                   
                    interp_weights[torch.where(torch.isinf(interp_weights))] = 100086
                    interp_weights = interp_weights / interp_weights.sum(-3, keepdim=True)
                    per_point_inv_transformation = smpl_v_inv[self.smpl_index[i], ...].reshape(-1, 4, 4)
                    cur_inv_shape_transforms = inv_shape_transforms[self.smpl_index[i], ...].reshape(-1, 4, 4)
                    per_point_inv_transformation = torch.matmul(cur_inv_shape_transforms, per_point_inv_transformation)
                    gather_inv_T = torch.gather(per_point_inv_transformation.reshape(1, -1, 1, 4, 4).repeat(1, 1, K, 1, 1), 1, nn.idx.reshape(1, -1, K, 1, 1).repeat(1, 1, 1, 4, 4))
                    inv_T = (gather_inv_T * interp_weights).sum(-3).reshape(1, -1, 4, 4)
                    homogen_coord = torch.ones([1, mask_cur_pts.shape[1], 1], dtype=mask_cur_pts.dtype, device=mask_cur_pts.device)
                    mask_cur_pts_homo = torch.cat([mask_cur_pts, homogen_coord], dim=2)
                    rays_pts_local = torch.matmul(inv_T, torch.unsqueeze(mask_cur_pts_homo, dim=-1))[:, :, :3, 0]

                    cur_xyz_min = self.vox_list[i].xyz_min
                    cur_xyz_max = self.vox_list[i].xyz_max
                    cur_new_mask = (rays_pts_local <= cur_xyz_max).sum(-1) + (rays_pts_local >= cur_xyz_min).sum(-1)
                    cur_new_mask = (cur_new_mask == 6)
                    new_mask_rays_pts_local = rays_pts_local[cur_new_mask]

                    uvh_list = []
                    if self.df.tex_2dp or self.df.tex_trip: #! trans

                        if mask_cur_pts.shape[1] ==0:
                            uvh = None
                        else:
                            
                            rays_pts_global_tmp = mask_cur_pts.view(-1, 3)[cur_new_mask.view(-1)].view(1, -1, 3)

                            if rays_pts_global_tmp.shape[1] == 0:            
                                uvh = None
                            else:
                                use_torch = not self.df.org_nn
                                if use_torch:                            
                                    near_uv, h = self.smpl_render.get_nearest_pts_in_mesh_torch_grad(smpl_v, rays_pts_global_tmp, k=1, num_samples = self.full_opt.uvVol_smpl_pts)
                                else:
                                    near_uv, h = self.smpl_render.get_nearest_pts_in_mesh(smpl_v, rays_pts_global_tmp, k=1, num_samples = self.full_opt.uvVol_smpl_pts)

                                uvh = torch.cat((near_uv, h), 2) #.view(-1, 3)[cur_new_mask].view(1, -1, 3)

                                styles_input = self.extract_uv_style_per_pts(uvh, styles).permute(0, 2, 1)

                        if uvh is None: continue

                    query_pts = new_mask_rays_pts_local.reshape(1, 1, 1, -1, 3) / 1.3
                    template_sdf = torch.nn.functional.grid_sample(
                        self.sdf_voxels, query_pts,
                        padding_mode = 'border', align_corners = True
                    ).reshape(-1)

                    window_alpha = 4; window_beta = 8
                    new_mask_rays_pts_local = (new_mask_rays_pts_local - (cur_vox.xyz_min + cur_vox.xyz_max) / 2.0) / (cur_vox.xyz_max - cur_vox.xyz_min)
                    weights = torch.exp(-window_alpha * ((new_mask_rays_pts_local * 2) ** window_beta).sum(-1))
                    fake_d = torch.zeros_like(new_mask_rays_pts_local)
                    fake_d[..., 0] = 1
                    if self.opt.input_ch_views == 3:
                        cur_input = torch.cat([
                            new_mask_rays_pts_local, fake_d
                        ], -1).unsqueeze(0)
                    else:
                        cur_input = new_mask_rays_pts_local.unsqueeze(0)
                    if w_space:
                        assert gamma_list is not None
                        assert beta_list is not None
                        new_mask_results = (cur_vox.network.forward_with_gamma_beta(
                            cur_input, gamma_list[i], beta_list[i]
                        ).view(-1, 4)[..., -1] + template_sdf) * weights.view(-1)
                    else:
                        new_mask_results = (cur_vox.network(
                            cur_input, styles=styles_input
                        ).view(cur_input.shape[1], self.output_ch)[..., -1] + template_sdf) * weights.view(-1)
                        #.view(-1, 4)
                    tmp_raw = torch.zeros_like(cur_sdf[cur_pts_ind])
                    tmp_raw[cur_new_mask.squeeze(0)] = new_mask_results
                    tmp_counter = torch.zeros_like(cur_counter[cur_pts_ind])
                    tmp_counter[cur_new_mask.squeeze(0)] = weights.view(-1)
                    cur_sdf[cur_pts_ind] += tmp_raw
                    cur_counter[cur_pts_ind] += tmp_counter

            counter_list.append(cur_counter.detach().cpu())
            sdf_list.append(cur_sdf.detach().cpu())

        sdf = torch.cat(sdf_list, 0)
        counter = torch.cat(counter_list, 0)

        mask = counter == 0
        sdf[~mask] /= counter[~mask]
        if self.with_sdf:
            sdf[mask] = 1
        return sdf.view(resolution, resolution, resolution)


    def extract_uv_style_per_pts(self, uvh, uv_latent):

        df = self.df

        if df.tex_2dp:    
            near_uv = uvh[..., :2]
            return self.render_posmap.index_posmap_by_vts(uv_latent, near_uv)

        elif self.df.tex_trip:
                        
            tc = self.df.tex_trip_c
            tex_uvh_plane = uv_latent
            tuh_dim, thv_dim, tuv_dim = tc, tc, self.model_opt.texdecoder_outdim - tc - tc
            uv_plane = tex_uvh_plane[:, :tuv_dim, ...]
            uh_plane = tex_uvh_plane[:, tuv_dim: tuv_dim + tuh_dim, ...]
            hv_plane = tex_uvh_plane[:, tuv_dim + tuh_dim: tuv_dim + tuh_dim + thv_dim, ...]

            near_uv = uvh[..., :2]
            h_pred = uvh[..., [2]]

            self.depth_bound = [-0.1, 0.1]
            h_pred *= 1/self.depth_bound[1]#-1,1
            h_pred = (h_pred + 1)/2 #[0, 1]
            
            uh = torch.cat((near_uv[..., [0]], h_pred), -1)
            hv = torch.cat((h_pred, near_uv[..., [1]]), -1)

            uv_feature = self.render_posmap.index_posmap_by_vts(uv_plane, near_uv)
            uh_feature = self.render_posmap.index_posmap_by_vts(uh_plane, uh)
            hv_feature = self.render_posmap.index_posmap_by_vts(hv_plane, hv)
            
            if tuh_dim == tuv_dim:
                fused_feat = torch.cat([uv_feature.unsqueeze(1), uh_feature.unsqueeze(1), hv_feature.unsqueeze(1)], 1).mean(1)
            else:
                uh_hv = torch.cat([uh_feature, hv_feature], 1)
                fused_feat = torch.cat([uv_feature.unsqueeze(1), uh_hv.unsqueeze(1)], 1).mean(1)                

            if self.df.style_cond:
                uv_style_2d = tex_uvh_plane[:, self.model_opt.texdecoder_outdim:, ...]
                assert uv_style_2d.shape[1] == self.df.structured_dim
                uv_style = self.render_posmap.index_posmap_by_vts(uv_style_2d, near_uv)
                return torch.cat([ torch.cat([f[None, ...], uv[None, ...]], 1)  for f, uv in zip (fused_feat, uv_style) ], 0)

            return fused_feat
        
        else:
            raise NotImplementedError()

    def extract_structured_triplanes(self, planes, id):
        
        id_rel = [
            1,    
            4,    
            5,    
            2,     
            6,    
            7,    
            8,    
            9,    
            3,    
            0,    
            10,    
            11,    
            12,    
            13,    
            14,    
            15
        ]


        position = [
            (0, 0, 64),
            (0, 64 + 64, 48),
            (48, 64 + 64, 48),
            (48 + 48, 64 + 64, 48),

            (0, 64, 32),
            (0, 64 + 32, 32),
            (32, 64, 32),
            (32, 64 + 32, 32),

            (64, 0, 32),
            (64, 32, 32),
            (64 + 32, 0, 32),
            (64 + 32, 32, 32),

            (64, 0 + 64, 32),
            (64, 32 + 64, 32),
            (64 + 32, 0 + 64, 32),
            (64 + 32, 32 + 64, 32)
        ]

        rid = id_rel[id]
        x, y, w = position[rid]
        return planes[:, :, x: x+ w, y:y+w]

    def generate_planes(self):
        """
        Defines planes by the three vectors that form the "axes" of the
        plane. Should work with arbitrary number of planes and planes of
        arbitrary orientation.
        """
        return torch.tensor([[[1, 0, 0],
                                [0, 1, 0],
                                [0, 0, 1]],
                                [[1, 0, 0],
                                [0, 0, 1],
                                [0, 1, 0]],
                                [[0, 0, 1],
                                [1, 0, 0],
                                [0, 1, 0]]], dtype=torch.float32)

    def project_onto_planes(self, planes, coordinates):
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

    def sample_from_planes(self, plane_axes, plane_features, coordinates, mode='bilinear', padding_mode='zeros', box_warp=None):
        assert padding_mode == 'zeros'
        N, n_planes, C, H, W = plane_features.shape
        _, M, _ = coordinates.shape
        plane_features = plane_features.view(N*n_planes, C, H, W)

        if box_warp is not None:
            coordinates = (2/box_warp) * coordinates # TODO: add specific box bounds

        projected_coordinates = self.project_onto_planes(plane_axes, coordinates).unsqueeze(1)
        output_features = grid_sample_fix.grid_sample(plane_features, projected_coordinates.float(), mode=mode, padding_mode=padding_mode, align_corners=False).permute(0, 3, 2, 1).reshape(N, n_planes, M, C)

        #grid_sample_gradfix.grid_sample

        return output_features



    def forward(self, cam_poses, focals, beta, theta, trans, styles = None, return_eikonal=False, no_white_bg=False, fix_viewdir=False, w_space=False, gamma_list=None, beta_list=None, smpl_vts_wrd = None, smpl_vts_can = None, batch = None, scale = 1):

        if styles is not None:
            real_batch_size = styles.shape[0]
            if focals.shape[0] < real_batch_size:
                focals = torch.cat([focals for i in range(real_batch_size)], 0)
                cam_poses = torch.cat([cam_poses for i in range(real_batch_size)], 0)

        batch_size = cam_poses.shape[0]
        beta = beta[:1]
        theta = theta[:1]
        trans = trans[:1]
        
        ray_o_list = [] 
        ray_d_list = []
        if batch is not None: 
            #printg(scale, "here")
            for i in range(batch_size):
                H, W = batch["img_nerf"].shape[1:3]
                ray_o_i, ray_d_i = self.get_rays_camera(H, W, batch["Cam_K_nerf"][i].cpu().numpy(), batch["Cam_R"][i].cpu().numpy(), batch["Cam_T"][i].cpu().numpy())
                
                ray_o_list.append(ray_o_i[None,...])
                ray_d_list.append(ray_d_i[None,...])

            rays_o  = torch.from_numpy(np.concatenate(ray_o_list, 0)).float().to(theta).reshape(batch_size, -1, 3)
            rays_d = torch.from_numpy(np.concatenate(ray_d_list, 0)).float().to(theta).reshape(batch_size, -1, 3)
            viewdirs = rays_d.reshape(-1, 3)
            rays_num = H * W
            self.out_im_res = (H, W)
        else:
            #print(self.out_im_res[0], self.out_im_res[1], focals, cam_poses)
            rays_num = self.out_im_res[0] * self.out_im_res[1]
            rays_o, rays_d, viewdirs = self.get_rays(focals, cam_poses)
            rays_o = rays_o.reshape(batch_size, -1, 3)
            rays_d = rays_d.reshape(batch_size, -1, 3)
            viewdirs = viewdirs.reshape(-1, 3)

        theta_rodrigues = batch_rodrigues(theta.reshape(-1, 3)).reshape(1, 24, 3, 3)
        rays_pts_local_list, rays_pts_global, mask_outbbox_list, valid_mask_outbbox_list, \
        mask_outbbox, forward_skinning_T_list, z_vals, rays_d_pts_local_list, \
        bbox_transformation_list, smpl_v, actual_vox_bbox, uvh_list = \
            self.sample_ray(
                beta.float(), theta_rodrigues.float(), trans.float(), rays_o[0].float(), rays_d[0].float(), smpl_vts_wrd = smpl_vts_wrd.float() if smpl_vts_wrd is not None else None, scale = scale
            )

        rays_o = rays_o.reshape(-1, 3)
        rays_d = rays_d.reshape(-1, 3)

        valid_rays_num = rays_pts_global.shape[0]

        #uvh_list = []
        rays_pts_global_list_foruv = []

        if len(uvh_list): #will not use
            uvh_list_expand = []
            for data in uvh_list:
                if data is None:
                    uvh_list_expand.append(None)
                else:
                    uvh_list_expand.append(data.repeat(batch_size, 1, 1))
            uvh_list = uvh_list_expand

        rays_pts_local_list = [
            data.repeat(batch_size, 1, 1) for data in rays_pts_local_list
        ]
        rays_d_pts_local_list = [
            data.repeat(batch_size, 1, 1) for data in rays_d_pts_local_list
        ]
        rays_pts_global = rays_pts_global.repeat(batch_size, 1, 1)
        mask_outbbox_list = [
            data.repeat(batch_size) for data in mask_outbbox_list
        ]
        valid_mask_outbbox_list = [
            data.repeat(batch_size) for data in valid_mask_outbbox_list
        ]
        mask_outbbox = mask_outbbox.repeat(batch_size)
        forward_skinning_T_list = [
            data.repeat(batch_size, 1, 1, 1) for data in forward_skinning_T_list
        ]
        z_vals = z_vals.repeat(batch_size, 1)
        
        eikonal_term_list = []
        actual_sdf_list = []
        
        dists = (z_vals[..., 1:] - z_vals[..., :-1]).reshape(batch_size, valid_rays_num, -1)
        rays_d_norm = torch.norm(rays_d, dim=-1).unsqueeze(-1).reshape(batch_size * rays_num, 1)
        valid_rays_d_norm = rays_d_norm[~mask_outbbox].reshape(batch_size, valid_rays_num, 1)
        # dists = torch.cat([dists, self.inf.expand(valid_rays_d_norm.shape)], -1)
        dists = torch.cat([dists, dists[..., -1:]], -1)

        
        raw = torch.zeros_like(rays_pts_global[..., 0]).unsqueeze(-1).repeat(1, 1, self.output_ch)
        raw_template = torch.zeros_like(rays_pts_global[..., 0]).unsqueeze(-1).repeat(1, 1, self.output_ch)
        normal = torch.zeros_like(rays_pts_global[..., 0]).unsqueeze(-1).repeat(1, 1, 3)
        counter = torch.zeros_like(rays_pts_global[..., 0]).unsqueeze(-1)

        styles_input = styles



        for i in range(len(self.vox_list)):

            cur_mask = ~valid_mask_outbbox_list[i]
            _cur_xyz = rays_pts_local_list[i][cur_mask].view(-1, 3) #smpl space
            _cur_rays_d = rays_d_pts_local_list[i][cur_mask].view(-1, 3)

            ### mask out points outside the original bbox ###
            cur_xyz_min = self.vox_list[i].xyz_min
            cur_xyz_max = self.vox_list[i].xyz_max
            cur_new_mask = (_cur_xyz <= cur_xyz_max).sum(-1) + (_cur_xyz >= cur_xyz_min).sum(-1)
            cur_new_mask = (cur_new_mask == 6)
            cur_xyz = _cur_xyz[cur_new_mask]


            cur_rays_d = _cur_rays_d[cur_new_mask]
            if fix_viewdir:
                cur_rays_d = torch.zeros_like(cur_rays_d)
                cur_rays_d[..., -1] = -1

            #printg("", self.full_opt.output_ch, self.output_ch)
            tmp_raw = torch.zeros_like(raw[cur_mask]).view(-1, self.output_ch)
            tmp_raw_template = torch.zeros_like(raw[cur_mask]).view(-1, self.output_ch)
            tmp_counter = torch.zeros_like(raw[cur_mask][..., 0]).view(-1)
            #################################################

            template_sdf = torch.nn.functional.grid_sample(
                self.sdf_voxels, cur_xyz.reshape(1, 1, 1, -1, 3) / 1.3,
                padding_mode = 'border', align_corners = True
            ).reshape(-1, 1)

            window_alpha = 4; window_beta = 8

            #! cur_xyz in Can?
            #can smpl and cur xyz.
            is_vis = True

            if len(uvh_list):
                if uvh_list[i] is None:
                    continue
                    styles_input = torch.ones([cur_xyz.shape[0], styles.shape[1]]).view(batch_size, -1, styles.shape[1]).to(cur_xyz.device)
                else:
                    cur_uvh = uvh_list[i]#[i].repeat(batch_size, 1, 1)
                    #cur_uvh = uvh_list[i].view(-1, 3)[cur_new_mask].view(batch_size, -1, 3)
                    if False and return_eikonal and self.df.uvh_trip_direct:
                        cur_uvh.requires_grad = True
                    styles_input = self.extract_uv_style_per_pts(cur_uvh, styles).permute(0, 2, 1)
                    #printg(styles_input.shape, styles.shape)
                    #printg("after ", cur_uvh.shape, cur_xyz.shape)


            cur_xyz = (cur_xyz - (self.vox_list[i].xyz_min + self.vox_list[i].xyz_max) / 2.) / (self.vox_list[i].xyz_max - self.vox_list[i].xyz_min)
            weights = torch.exp(-window_alpha * ((cur_xyz * 2) ** window_beta).sum(-1))
            if return_eikonal:
                cur_xyz.requires_grad = True
            if self.opt.input_ch_views == 3:
                cur_input = torch.cat([
                    cur_xyz.view(batch_size, -1, 3),
                    cur_rays_d.view(batch_size, -1, 3)
                ], -1)
            elif self.opt.input_ch_views == 0:
                # raise NotImplementedError
                cur_input = cur_xyz.view(batch_size, -1, 3)
            else:
                raise NotImplementedError
            
            if w_space:
                assert gamma_list is not None
                assert beta_list is not None
                tmp_output = self.vox_list[i].network.forward_with_gamma_beta(
                    cur_input, gamma_list[i], beta_list[i]
                ).view(-1, 4)
            else:
                #input: B, N, C
                if False and len(uvh_list) and uvh_list[i] is None and self.opt.cat_style:
                    tmp_output = torch.zeros([cur_xyz.shape[0], 4]).to(cur_xyz.device).requires_grad_(True)
                else: #OOM problem
                    if self.df.noise_1d_eva:
                        chunk = 4096 * 4
                        out_list = []
                        assert cur_input.shape[0] == 1
                        if cur_input.shape[1] == 0: continue
                        for ib in range(0, cur_input.shape[1], chunk):
                            out_list.append(self.vox_list[i].network(
                                cur_input[:, ib : ib + chunk, ...], styles=styles_input
                                ).view(-1, 4))
                            
                        tmp_output = torch.cat(out_list, 0)
                    else:
                        if self.df.struct_triplane:
                            
                            planes = self.extract_structured_triplanes(styles, i)#
                            h, w = planes.shape[-2], planes.shape[-1]
                            planes = planes.reshape(batch_size, 3, -1, h, w)

                            plane_axes = self.generate_planes().to(styles.device)
                            styles_input = self.sample_from_planes(plane_axes, planes, cur_xyz[None, ...], padding_mode='zeros', box_warp = None)
                            styles_input = styles_input.mean(1)

                        tmp_output = self.vox_list[i].network(
                            cur_input, styles=styles_input
                        ).view(cur_input.shape[1], self.output_ch)
            # N X 3, STYLE: B, 128

            actual_sdf_list.append(tmp_output[:, -1:].view(-1).clone())
            tmp_output[:, -1:] = tmp_output[:, -1:] + template_sdf

            tmp_raw[cur_new_mask] = tmp_output * weights.view(-1, 1)
            tmp_raw_template[cur_new_mask] = template_sdf * weights.view(-1, 1)
            tmp_counter[cur_new_mask] += weights.view(-1)

            raw[cur_mask] += tmp_raw.view(-1, self.N_samples, self.output_ch)
            raw_template[cur_mask] += tmp_raw_template.view(-1, self.N_samples, self.output_ch)
            counter[cur_mask] += tmp_counter.view(-1, self.N_samples, 1)


            if return_eikonal:
                eikonal_term = self.vox_list[i].get_eikonal_term(
                    cur_xyz, tmp_output[..., -1]
                )
                eikonal_term_list.append(eikonal_term)
                normal_tmp = torch.zeros_like(normal[cur_mask]).view(-1, 3)
                normal_tmp[cur_new_mask] = eikonal_term * weights.view(-1, 1)
                try:
                    if torch.any(cur_mask):
                        normal_tmp = torch.matmul(forward_skinning_T_list[i][..., :3, :3].reshape(-1, 3, 3), normal_tmp.reshape(-1, 3).unsqueeze(-1))[..., 0].view(-1, self.N_samples, 3)
                        normal[cur_mask] += normal_tmp.view(-1, self.N_samples, 3)
                except Exception as e:
                    print(e)
                    st()

        sdf = raw[..., -1]
        sdf[counter.squeeze(-1) == 0] = 1
        raw[..., -1] = sdf
        
        sdf_template = raw_template[..., -1]
        sdf_template[counter.squeeze(-1) == 0] = 1
        raw_template[..., -1] = sdf_template

        counter[torch.isclose(counter, torch.zeros_like(counter).cuda())] = 1
        
        raw = raw / counter

        raw_template = raw_template / counter

        normal = normal / counter

        # Render All
        if self.with_sdf:
            sigma = self.sdf_activation(-raw[..., -1].view(batch_size, valid_rays_num, -1))
            sigma = 1 - torch.exp(-sigma * dists)
            sigma_template = self.sdf_activation(-raw_template[..., -1].view(batch_size, valid_rays_num, -1))
            sigma_template = 1 - torch.exp(-sigma_template * dists)
        else:
            raise NotImplementedError

        visibility = torch.cumprod(torch.cat([torch.ones_like(torch.index_select(sigma, 2, self.zero_idx)), 1.-sigma + 1e-10], 2), 2)
        visibility = visibility[...,:-1]
        _weights = sigma * visibility

        visibility_template = torch.cumprod(torch.cat([torch.ones_like(torch.index_select(sigma_template, 2, self.zero_idx)), 1.-sigma_template + 1e-10], 2), 2)
        visibility_template = visibility_template[...,:-1]
        _weights_template = sigma_template * visibility_template

        rgb_ch = self.output_ch - 1

        _rgb_map = torch.sum(_weights.unsqueeze(-1) * torch.sigmoid(raw[..., :rgb_ch].view(batch_size, valid_rays_num, -1, rgb_ch)), 2)
        _xyz_map = torch.sum(_weights.unsqueeze(-1) * rays_pts_global.view(batch_size, valid_rays_num, -1, 3), 2)
        _depth_map = torch.sum(_weights * z_vals.view(batch_size, valid_rays_num, -1), -1)
        _depth_map_template = torch.sum(_weights_template * z_vals.view(batch_size, valid_rays_num, -1), -1)
        _weights_map = _weights.sum(dim=-1, keepdim=True)
        _weights_map_template = _weights_template.sum(dim=-1, keepdim=True)
            
        if return_eikonal:
            normal = torch.sum(_weights.unsqueeze(-1) * normal.view(batch_size, valid_rays_num, -1, 3), 2)

        if return_eikonal:
            cat_eikonal_term = torch.cat(eikonal_term_list, -2)
        else:
            cat_eikonal_term = None

        if self.opt.white_bg and not no_white_bg:
            _rgb_map = _rgb_map + 1 - _weights_map.view(batch_size, valid_rays_num, 1) 


        H, W = self.out_im_res

        rgb_map = torch.zeros(batch_size * H * W, rgb_ch).to(_rgb_map.device)
        if self.opt.white_bg and not no_white_bg:
            rgb_map += 1
        rgb_map[~mask_outbbox] = _rgb_map.view(-1, rgb_ch)
        rgb_map = rgb_map.view(batch_size, H, W, rgb_ch)

        xyz_map = torch.zeros(batch_size * H * W, 3).to(_xyz_map.device)
        xyz_map[~mask_outbbox] = _xyz_map.view(-1, 3)
        xyz = xyz_map.view(batch_size, H, W, 3)

        depth_map = torch.zeros(batch_size * H * W, 1).to(_depth_map.device) + _depth_map.max()
        depth_map[~mask_outbbox] = _depth_map.view(-1, 1)
        depth = depth_map.view(batch_size, H, W)
        
        depth_map_template = torch.zeros(batch_size * H * W, 1).to(_depth_map_template.device) + _depth_map_template.max()
        depth_map_template[~mask_outbbox] = _depth_map_template.view(-1, 1)
        depth_template = depth_map_template.view(batch_size, H, W)

        weights_map = torch.zeros(batch_size * H * W, 1).to(_weights_map.device)
        weights_map[~mask_outbbox] = _weights_map.view(-1, 1)
        mask = weights_map.view(batch_size, H, W)

        weights_map_template = torch.zeros(batch_size * H * W, 1).to(_weights_map_template.device)
        weights_map_template[~mask_outbbox] = _weights_map_template.view(-1, 1)
        mask_template = weights_map_template.view(batch_size, H, W)

        if return_eikonal:
            normal_map = torch.zeros(batch_size * H * W, 3).to(normal.device)
            normal_map[~mask_outbbox] = normal.view(-1, 3)
            feature_map = normal_map.view(batch_size, H, W, 3)
        else:
            feature_map = None
        rgb_map = -1 + 2 * rgb_map
        sdf = torch.cat(actual_sdf_list, 0)

        rgb_map = torch.cat([rgb_map, depth[..., None]], -1)
        #todo should be eikonal term not depth

        return rgb_map, feature_map, [sdf], mask, [xyz, depth, depth_template, mask_template], cat_eikonal_term


