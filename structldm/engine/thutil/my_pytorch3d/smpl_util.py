
import torch
import torch.nn.functional as F
import smplx as sx
import numpy as np

import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import imageio

import os

from pytorch3d.structures import Pointclouds, meshes

from pytorch3d.ops.knn import knn_gather, knn_points            


import torch

import pytorch3d
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.structures import Meshes
from pytorch3d.io import load_obj, save_obj

import cv2

import torch.nn as nn



from pytorch3d.renderer import (
    TexturesUV,
    SfMPerspectiveCameras,
    DirectionalLights,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
)

import sys
sys.path.append("..")

from pytorch3d.renderer.mesh.shader import (
    SoftPhongShader,
)
from pytorch3d.io import IO

from ..num import nearest_k_idx, index_high_dimension

json_path = ""
smpl_model = "smpl/smpl_uv.obj"


from .textures import get_smpl_uv_vts
from .textures import get_default_texture_maps_3
from ..load_smpl_tmp import load_smpl

from .util.sample_points_from_meshes import sample_points_from_meshes
from ..grid_sample_fix import grid_sample

class SMPL_Util:

    def __init__(
        self,        
        gender="neutral", verts_uvs = None, faces_uvs = None, smpl_uv_vts = None, face_pix = 1
    ) -> None:
        #if verts_uvs, faces_uvs are provided, they are bf-smpl.
        
        self.is_mesh = True
        self.mesh = 0
        
        self.face_pix = face_pix
        
        if (verts_uvs is not None) and (faces_uvs is not None) and (smpl_uv_vts is not None):                        
            self.verts_uvs = verts_uvs.cuda()  # (V, 2)
            self.faces_uvs = faces_uvs.cuda()  # (F, 3)
            texture_map = np.random.rand(256, 256, 3).astype(np.float32)            
            self.default_tex_maps = TexturesUV(maps=[torch.from_numpy(texture_map).cuda()], \
                faces_uvs=[faces_uvs.cuda()], verts_uvs=[verts_uvs.cuda()])
            self.smpl_uv_vts = smpl_uv_vts.float().cuda().unsqueeze(0) #.permute(0,2,
        else:
            self.default_tex_maps, self.verts_uvs, self.faces_uvs = get_default_texture_maps_3()
            
            self.smpl_uv_vts = get_smpl_uv_vts().float().cuda().unsqueeze(0) #.permute(0,2,
                
        self.smpl_model = load_smpl(gender) #load_smpl("neutral")
        self.smpl_faces = self.smpl_model.faces_tensor[None,...].expand(1, -1, -1) 
        self.camera = None
        
        self.verts_uvs_numpy = self.verts_uvs.detach().cpu().numpy()
        self.faces_uvs_numpy = self.faces_uvs.detach().cpu().numpy()
        self.smpl_faces_numpy = self.smpl_model.faces_tensor.detach().cpu().numpy()
    
    def get_smpl_faces(self):
        return self.smpl_model.faces_tensor
        
    
    def get_head_vts(self):
        return self.head_vertices_index, self.head_faces


    def load_notex_mesh(self, mesh_file):
        
        from pytorch3d.renderer import Textures
        
        device = torch.device("cuda:0")
                
        verts, faces_idx, _ = load_obj(mesh_file)
        faces = faces_idx.verts_idx

        #-Initialize each vertex to be white in color.
        verts_rgb = torch.ones_like(verts)[None] # (1, V, 3)
        textures = Textures(verts_rgb=verts_rgb.to(device))

        #-Create a Meshes object for the teapot. Here we have only one mesh in the batch.
        mesh = Meshes(
        verts=[verts.to(device)],
        faces=[faces.to(device)],
        textures=textures
        )
        
        return mesh


    def get_nearest_pts_in_mesh_torch(self, posed_smpl_vertices, query_points_in_posed, k=1, num_samples = 5):
                
        device = posed_smpl_vertices.device

        if posed_smpl_vertices.ndimension()==2:
            posed_smpl_vertices = posed_smpl_vertices[None,...]
        
        batch_size = posed_smpl_vertices.shape[0]
        if not torch.is_tensor(posed_smpl_vertices):
            posed_smpl_vertices = torch.from_numpy(posed_smpl_vertices).float().to(device)

        faces = self.smpl_faces.expand(batch_size, -1, -1)
        
        tex_maps = self.default_tex_maps
        mesh_obj = Meshes(verts = posed_smpl_vertices, faces = faces,
                                textures=tex_maps).to(device)
        
        t = num_samples
        num_samples *= posed_smpl_vertices.shape[1]
        
        with torch.no_grad():
            sampled_pts, sampled_normals, sampled_uv = sample_points_from_meshes(
                mesh_obj, num_samples = num_samples, return_normals = True, return_uv = True, verts_uvs = self.verts_uvs, faces_uvs = self.faces_uvs
            )
       
        chunk = 1024 * 48 # 16 #chunk_dim
        with torch.no_grad():
            vtx_id = torch.cat([knn_points(query_points_in_posed[[0], i: i + chunk, :], sampled_pts.reshape(1, -1, 3), K=k).idx 
                for i in range(0, query_points_in_posed.shape[1], chunk)],
                dim = 1)
                
        closest_smpl_verts = index_high_dimension(sampled_pts, vtx_id, dim=1)
        closest_smpl_normals = index_high_dimension(sampled_normals, vtx_id, dim=1)
        closest_smpl_uvs = index_high_dimension(sampled_uv, vtx_id, dim=1)

        closest_smpl_uvs[...,[0,1]] = closest_smpl_uvs[...,[1,0]]
        assert k==1 
        if k==1:
            closest_smpl_verts = closest_smpl_verts[:,:,0,:]
            closest_smpl_normals = closest_smpl_normals[:,:,0,:]
            closest_smpl_uvs = closest_smpl_uvs[:,:,0,:]

            height = torch.sum(closest_smpl_normals * (query_points_in_posed - closest_smpl_verts), -1)[...,None]
            
            return closest_smpl_uvs, height
    
        
    def sample_dense_smpl(self):
        pass


    def get_nearest_pts_in_mesh_torch_infer(self, posed_smpl_vertices, query_points_in_posed, k=1, num_samples = 5):
        
        #num_samples = 2
        
        device = posed_smpl_vertices.device

        if posed_smpl_vertices.ndimension()==2:
            posed_smpl_vertices = posed_smpl_vertices[None,...]
        
        batch_size = posed_smpl_vertices.shape[0]
        if not torch.is_tensor(posed_smpl_vertices):
            posed_smpl_vertices = torch.from_numpy(posed_smpl_vertices).float().to(device)

        faces = self.smpl_faces.expand(batch_size, -1, -1)
        
        tex_maps = self.default_tex_maps
        #print(posed_smpl_vertices.shape, faces.shape)
        mesh_obj = Meshes(verts = posed_smpl_vertices, faces = faces,
                                textures=tex_maps).to(device)
        
        t = num_samples
        num_samples *= posed_smpl_vertices.shape[1]
        
        with torch.no_grad():
            sampled_pts, sampled_normals, sampled_uv = sample_points_from_meshes(
                mesh_obj, num_samples = num_samples, return_normals = True, return_uv = True, verts_uvs = self.verts_uvs, faces_uvs = self.faces_uvs
            )

        chunk = 1024 * 480 # 16 #chunk_dim
        with torch.no_grad():
            vtx_id = torch.cat([knn_points(query_points_in_posed[[0], i: i + chunk, :], sampled_pts.reshape(1, -1, 3), K=k).idx 
                for i in range(0, query_points_in_posed.shape[1], chunk)],
                dim = 1)

        closest_smpl_verts = index_high_dimension(sampled_pts, vtx_id, dim=1)
        closest_smpl_normals = index_high_dimension(sampled_normals, vtx_id, dim=1)
        closest_smpl_uvs = index_high_dimension(sampled_uv, vtx_id, dim=1)

        closest_smpl_uvs[...,[0,1]] = closest_smpl_uvs[...,[1,0]]
       
        assert k==1 
        if k==1:
            closest_smpl_verts = closest_smpl_verts[:,:,0,:]
            closest_smpl_normals = closest_smpl_normals[:,:,0,:]
            closest_smpl_uvs = closest_smpl_uvs[:,:,0,:]

            height = torch.sum(closest_smpl_normals * (query_points_in_posed - closest_smpl_verts), -1)[...,None]
            
            return closest_smpl_uvs, height
    
        

        

    def get_nearest_pts_in_mesh(self, posed_smpl_vertices, query_points_in_posed, k=1, num_samples = 5):
        
        #num_samples = 2
        
        device = posed_smpl_vertices.device

        if posed_smpl_vertices.ndimension()==2:
            posed_smpl_vertices = posed_smpl_vertices[None,...]
        
        batch_size = posed_smpl_vertices.shape[0]
        if not torch.is_tensor(posed_smpl_vertices):
            posed_smpl_vertices = torch.from_numpy(posed_smpl_vertices).float().to(device)

        faces = self.smpl_faces.expand(batch_size, -1, -1)
        
        tex_maps = self.default_tex_maps
        #print(posed_smpl_vertices.shape, faces.shape)
        mesh_obj = Meshes(verts = posed_smpl_vertices, faces = faces,
                                textures=tex_maps).to(device)
        
        t = num_samples
        num_samples *= posed_smpl_vertices.shape[1]
        
        with torch.no_grad():
            sampled_pts, sampled_normals, sampled_uv = sample_points_from_meshes(
                mesh_obj, num_samples = num_samples, return_normals = True, return_uv = True, verts_uvs = self.verts_uvs, faces_uvs = self.faces_uvs
            )
                
        #chunk_dim = int(16 / t) - 1
        chunk = 1024 * 16 # 16 #chunk_dim
        
        # tmp=[]
        # for i in range(0, query_points_in_posed.shape[1], chunk):
        #     tmp.append(nearest_k_idx(src = query_points_in_posed[[0], i: i + chunk, :], tgt = sampled_pts, k=k))
        # vtx_id = torch.cat(tmp, 1)
        
        with torch.no_grad():
            vtx_id = torch.cat([nearest_k_idx(src = query_points_in_posed[[0], i: i + chunk, :] , \
                tgt = sampled_pts, k=k) 
                for i in range(0, query_points_in_posed.shape[1], chunk)], 
                dim = 1)
        
        #vtx_id = nearest_k_idx(src=query_points_in_posed, tgt = sampled_pts, k=k)
                
        closest_smpl_verts = index_high_dimension(sampled_pts, vtx_id, dim=1)
        closest_smpl_normals = index_high_dimension(sampled_normals, vtx_id, dim=1)
        closest_smpl_uvs = index_high_dimension(sampled_uv, vtx_id, dim=1)

        closest_smpl_uvs[...,[0,1]] = closest_smpl_uvs[...,[1,0]]
        #sampled_uv[...,[0,1]] = sampled_uv[...,[1,0]]
        #self.test_sample_uv_list(sampled_pts, sampled_uv)

        #h = (query_points_in_posed - verts).norm(-1)
        assert k==1 
        if k==1:
            closest_smpl_verts = closest_smpl_verts[:,:,0,:]
            closest_smpl_normals = closest_smpl_normals[:,:,0,:]
            closest_smpl_uvs = closest_smpl_uvs[:,:,0,:]

            height = torch.sum(closest_smpl_normals * (query_points_in_posed - closest_smpl_verts), -1)[...,None]
            
            #return sampled_pts, closest_smpl_uvs, height, vtx_id
            return closest_smpl_uvs, height
    

class DPLookupRendererNormal(nn.Module):


    '''Given an IUV denspose image (iuvimage) and densepose texture(dptex), propogates the texture image by differentiable sampling (lookup_mode supports bilinear and nearest)'''
    def __init__(self, lookup_mode = 'bilinear'):
        super(DPLookupRendererNormal, self).__init__()
        self.lookup_mode = lookup_mode


    def forward(self, normal_tex, normal_flow_b):
        nbatch = normal_flow_b.shape[0]
        normal_flow_b = normal_flow_b.permute(0, 2, 3, 1)
        flowzero = torch.ones(nbatch, normal_flow_b.shape[1], normal_flow_b.shape[2], 2).float().cuda() * 5


        
        flow = torch.where(torch.unsqueeze(normal_flow_b[:, :, :, 0] == 1, -1), normal_flow_b[:, :, :, 1:], flowzero)
        input_t = normal_tex

        #print('flow,', flow.shape)
        out_t = grid_sample(input_t, flow, mode=self.lookup_mode, align_corners=True)

        return out_t

    def forward_single(self, normal_tex, normal_flow_b):
        nbatch = normal_flow_b.shape[0]
        flowzero = torch.ones(nbatch, normal_flow_b.shape[1], normal_flow_b.shape[2], 2) * 5

        flow = torch.where(torch.unsqueeze(normal_flow_b[:, :, :, 0] == 1, -1), normal_flow_b[:, :, :, 1:], flowzero)
        input_t = normal_tex.unsqueeze(0).repeat(nbatch, 1, 1, 1).permute(0, 3, 2, 1)
        out_t = grid_sample(input_t, flow, mode=self.lookup_mode, align_corners=True)

        return out_t