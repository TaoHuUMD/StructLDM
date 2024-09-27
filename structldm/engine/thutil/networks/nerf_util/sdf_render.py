#borrowed from EVA3D [Hong et al.]
import os
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import numpy as np
from functools import partial
from pdb import set_trace as st
from smplx.lbs import transform_mat, blend_shapes

import torch.nn.functional as F
import torch

import trimesh
from skimage.measure import marching_cubes

import engine.thutil.grid_sample_fix as grid_sample_fix


#from ..io.prints import *

#from pytorch3d.ops.knn import knn_gather, knn_points
#from volume_renderer import SirenGenerator
#from smpl_utils import init_smpl, get_J, get_shape_pose, batch_rodrigues

class SDFRenderer:
    def __init__(self, opt, xyz_min=None, xyz_max=None, style_dim=256, mode='train'):
        
        #self.opt = opt.deepfashion

        # self.xyz_min = torch.from_numpy(xyz_min).float().cuda()
        # self.xyz_max = torch.from_numpy(xyz_max).float().cuda()
        # self.test = mode != 'train'
        # self.perturb = opt.perturb
        # self.offset_sampling = not opt.no_offset_sampling # Stratified sampling used otherwise
        # self.N_samples = opt.N_samples
        # self.raw_noise_std = opt.raw_noise_std
        # self.return_xyz = opt.return_xyz
        # self.return_sdf = opt.return_sdf
        # self.static_viewdirs = opt.static_viewdirs
        # self.z_normalize = not opt.no_z_normalize
        self.force_background = False

        self.opt = opt

        #self.with_sdf = not opt.no_sdf
        self.with_sdf = True
        self.raw_noise_std = 0

        if self.with_sdf:
            self.sigmoid_beta = nn.Parameter(0.1 * torch.ones(1))

        with open('assets/smpl_template_sdf.npy', 'rb') as f:
            sdf_voxels = np.load(f)
        self.sdf_voxels = torch.from_numpy(sdf_voxels).reshape(1, 1, 128, 128, 128).cuda()
        self.sdf_template = self.sdf_voxels.permute(0, 1, 4, 3, 2)

        self.transfer_to_orig = self.opt.rendering.transfer_to_mean

        #self.register_buffer('t_vals', t_vals, persistent=False)
        #self.register_buffer('inf', torch.Tensor([1e10]), persistent=False)
        #self.register_buffer('zero_idx', torch.LongTensor([0]), persistent=False)

        self.channel_dim = -1
        self.samples_dim = 3
        self.input_ch = 3
        #self.input_ch_views = opt.input_ch_views

        #self.feature_out_size = 

        self.output_features = True

    def sample_template_sdf(self, pts_):
        #print("use template 888 ")
        #exit()
        #torch.nn.functional.grid_sample
        #grid_sample_fix.grid_sample
        #with torch.no_grad():

        if self.transfer_to_orig:
            mean_shape_center = torch.from_numpy(np.array([-0.00179506, -0.22333345,  0.02821913])).float().to(pts_.device) 
            pts = pts_.reshape(-1, 3) + mean_shape_center 
        else:
            pts = pts_

        template_sdf = torch.nn.functional.grid_sample(
            self.sdf_voxels, pts.reshape(1, 1, 1, -1, 3) / 1.3,
            padding_mode = 'border', align_corners = True
        ).reshape(-1, 1)

        #if self.opt.rendering.sdf_template:
            #print("sdf  ", pts.shape, sdf_.shape, self.sample_template_sdf(pts).shape)
        #    sdf = sdf_ + self.sample_template_sdf(pts)
        
        return template_sdf


    def get_eikonal_term(self, pts, sdf_):
        #self.sdf_template
        
        sdf = sdf_
        eikonal_term = autograd.grad(outputs=sdf, inputs=pts,
                                    grad_outputs=torch.ones_like(sdf),
                                    create_graph=True)[0]

        return eikonal_term

    def sdf_activation(self, input):
        sigma = torch.sigmoid(input / self.sigmoid_beta.to(input)) / self.sigmoid_beta.to(input)

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
            features = None

        self.feature_out_size = raw.shape[-1] - 4
        sdf, rgb, features = torch.split(raw, [1, 3, self.feature_out_size], dim=self.channel_dim)

        noise = 0.

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
       
        #rgb_map = -1 + 2 * torch.sum(weights * torch.sigmoid(rgb), self.samples_dim)  # switch to [-1,1] value range

        rgb_map = torch.sum(weights * torch.sigmoid(rgb), self.samples_dim) [0, 1]
        depth_map = torch.sum(weights * z_vals, -1)


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


        if return_eikonal:
            eikonal_term = self.get_eikonal_term(
                cur_xyz, tmp_output[..., -1]
            )
            eikonal_term_list.append(eikonal_term)


        sdf = raw[..., -1]
        sdf[counter.squeeze(-1) == 0] = 1
        raw[..., -1] = sdf
        
        
        raw = raw / counter


        if self.with_sdf:
            sigma = self.sdf_activation(-raw[..., -1].view(batch_size, valid_rays_num, -1))
            sigma = 1 - torch.exp(-sigma * dists)
        else:
            raise NotImplementedError


        visibility = torch.cumprod(torch.cat([torch.ones_like(torch.index_select(sigma, 2, self.zero_idx)), 1.-sigma + 1e-10], 2), 2)
        visibility = visibility[...,:-1]
        _weights = sigma * visibility
        _rgb_map = torch.sum(_weights.unsqueeze(-1) * torch.sigmoid(raw[..., :3].view(batch_size, valid_rays_num, -1, 3)), 2)
        _xyz_map = torch.sum(_weights.unsqueeze(-1) * rays_pts_global.view(batch_size, valid_rays_num, -1, 3), 2)
        _depth_map = torch.sum(_weights * z_vals.view(batch_size, valid_rays_num, -1), -1)


        return rgb_map, feature_map, sdf_out, mask, xyz, eikonal_term

        if return_sdf_xyz:
            sdf = _sdf[0]
            sdf_xyz = _sdf[1]
        else:
            sdf = _sdf[0]

        if self.full_pipeline:
            raise NotImplementedError
        else:
            rgb = None

        out = (rgb, thumb_rgb)
        if return_xyz:
            out += (xyz,)
        if return_sdf:
            out += (sdf,)
        if return_eikonal:
            out += (eikonal_term,)
        if return_mask:
            out += (mask,)
        if return_normal:
            out += (features, )
        if return_sdf_xyz:
            out += (sdf_xyz, )

        return out

        #thumb_rgb, features, _sdf, mask, xyz, eikonal_term

    def sdf_raw2outputs(self, raw, z_vals, rays_d, pts, white_bkgd=False, return_eikonal = True, normal = None):
        #self, raw, z_vals, rays_d, pts, return_eikonal=False

        """Transforms model's predictions to semantically meaningful values.
        Args:
            raw: [num_rays, num_samples along ray, 4]. Prediction from model.
            z_vals: [num_rays, num_samples along ray]. Integration time.
            rays_d: [num_rays, 3]. Direction of each ray.
        Returns:
            rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
            disp_map: [num_rays]. Disparity map. Inverse of depth map.
            acc_map: [num_rays]. Sum of weights along each ray.
            weights: [num_rays, num_samples]. Weights assigned to each sampled color.
            depth_map: [num_rays]. Estimated distance to object.
        """

        #return_eikonal
        #calc eikonal outside
        assert not return_eikonal

        dists = z_vals[..., 1:] - z_vals[..., :-1]
        
        dists = torch.cat(
            [dists,
            torch.Tensor([1e10]).expand(dists[..., :1].shape).to(dists)],
            -1)  # [N_rays, N_samples]
        
        dists = dists * torch.norm(rays_d[..., None, :], dim=-1)
        
        sdf = raw[..., 0]
        #rgb = raw[..., 1:]  # [N_rays, N_samples, 3]
        rgb = torch.sigmoid(raw[..., 1:])

        noise = 0.

        if self.with_sdf:
            sigma = self.sdf_activation(-sdf)

            if return_eikonal:
                #print(pts.requires_grad, sdf.requires_grad)
                eikonal_term = self.get_eikonal_term(pts, sdf.view(-1, 1))
            else:
                eikonal_term = None

            sigma = 1 - torch.exp(-sigma * dists)
        else:
            sigma = sdf
            eikonal_term = None
            sigma = 1 - torch.exp(-F.softplus(sigma + noise) * dists.unsqueeze(self.channel_dim))

        alpha = sigma
        cum = torch.cumprod(
            torch.cat(
                [torch.ones((alpha.shape[0], 1)).to(alpha), 1. - alpha + 1e-10],
                -1), -1)[:, :-1]

        weights = alpha * torch.cumprod(
            torch.cat(
                [torch.ones((alpha.shape[0], 1)).to(alpha), 1. - alpha + 1e-10],
                -1), -1)[:, :-1]

        rgb_map = torch.sum(weights[..., None] * rgb, -2)  # [N_rays, 3]
        depth_map = torch.sum(weights * z_vals, -1)
        

        disp_map = 1. / torch.max(1e-10 * torch.ones_like(depth_map).to(depth_map),
                                depth_map / torch.sum(weights, -1))
        acc_map = torch.sum(weights, -1)

        if normal is not None:
            normal = torch.sum(weights[..., None] * normal.reshape(rgb.shape[0], rgb.shape[1], 3), -2)

        if white_bkgd:
            rgb_map = rgb_map + (1. - acc_map[..., None])

        return rgb_map, acc_map, depth_map, sdf, normal



def extract_mesh_with_marching_cubes(sdf, level_set=0):
    # b, h, w, d, _ = sdf.shape

    # change coordinate order from (y,x,z) to (x,y,z)
    # sdf_vol = sdf[0,...,0].permute(1,0,2).cpu().numpy()


    w, h, d = sdf.shape
    sdf_vol = sdf.cpu().numpy()

    # scale vertices
    verts, faces, _, _ = marching_cubes(sdf_vol, level_set, mask=sdf_vol!=10086)
    verts[:,0] = (verts[:,0]/float(w)-0.5)
    verts[:,1] = (verts[:,1]/float(h)-0.5)
    verts[:,2] = (verts[:,2]/float(d)-0.5)

    # # fix normal direction
    verts[:,2] *= -1; verts[:,1] *= -1
    mesh = trimesh.Trimesh(verts, faces)

    return mesh, verts, faces

def eikonal_loss(eikonal_term, sdf=None, beta=100, deltasdf=False):
    #print("deltasdf ***  ", deltasdf)
    if eikonal_term == None:
        eikonal_loss = 0
    else:
        if deltasdf:
            eikonal_loss = ((eikonal_term.norm(dim=-1)) ** 2).mean()
        else:
            eikonal_loss = ((eikonal_term.norm(dim=-1) - 1) ** 2).mean()

    if sdf == None:
        minimal_surface_loss = torch.tensor(0.0, device=eikonal_term.device)
    else:
        if deltasdf:
            minimal_surface_loss = torch.nn.functional.smooth_l1_loss(sdf, torch.zeros_like(sdf)).mean()
        else:
            minimal_surface_loss = torch.exp(-beta * torch.abs(sdf)).mean()

    return eikonal_loss, minimal_surface_loss

def get_sdf_loss(opt, sdf, eikonal_term):
    
    if opt.eikonal_lambda > 0:
        g_eikonal, g_minimal_surface = eikonal_loss(eikonal_term, sdf=sdf if opt.min_surf_lambda > 0 else None,
                                                    beta=opt.min_surf_beta, deltasdf = not opt.not_deltasdf)
        g_eikonal = opt.eikonal_lambda * g_eikonal
        if opt.min_surf_lambda > 0:
            g_minimal_surface = opt.min_surf_lambda * g_minimal_surface

    if opt.eikonal_lambda <= 0 and opt.min_surf_lambda > 0:
        g_minimal_surface = opt.min_surf_lambda * torch.exp(-opt.min_surf_beta * torch.abs(sdf)).mean()

    loss_dict={}
    loss_dict["g_eikonal"] = g_eikonal
    loss_dict["g_minimal_surface"] = g_minimal_surface
    return loss_dict

class VoxelSDFRenderer(nn.Module):
    def __init__(self, opt, xyz_min, xyz_max, style_dim=256, mode='train'):
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

        # print(opt.depth, opt.width)

        # set Siren Generator model
        # self.network = SirenGenerator(D=opt.depth, W=opt.width, style_dim=style_dim, input_ch=self.input_ch,
        #                               output_ch=4, input_ch_views=self.input_ch_views,
        #                               output_features=self.output_features)

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