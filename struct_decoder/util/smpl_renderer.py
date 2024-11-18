
import torch
import numpy as np

from pytorch3d.ops.knn import knn_gather, knn_points            

import torch

from pytorch3d.structures import Meshes


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

from Engine.th_utils.my_pytorch3d.textures import get_default_texture_maps, get_smpl_uv_vts

from pytorch3d.renderer.mesh.shader import (
    SoftPhongShader,
)
from Engine.th_utils.num import nearest_k_idx,  index_high_dimension
from Engine.th_utils.load_smpl_tmp import load_smpl

from Engine.th_utils.my_pytorch3d.util.sample_points_from_meshes import sample_points_from_meshes

class SMPL_Renderer:

    def __init__(
        self,
        gender="neutral", verts_uvs = None, faces_uvs = None, smpl_uv_vts = None, face_pix = 1
    ) -> None:
        
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
            self.default_tex_maps, self.verts_uvs, self.faces_uvs = get_default_texture_maps()
            
            self.smpl_uv_vts = get_smpl_uv_vts().float().cuda().unsqueeze(0) #.permute(0,2,
                
        self.smpl_model = load_smpl(gender) #load_smpl("neutral")
        self.smpl_faces = self.smpl_model.faces_tensor[None,...].expand(1, -1, -1) 
        self.camera = None
        
        self.verts_uvs_numpy = self.verts_uvs.detach().cpu().numpy()
        self.faces_uvs_numpy = self.faces_uvs.detach().cpu().numpy()
        self.smpl_faces_numpy = self.smpl_model.faces_tensor.detach().cpu().numpy()
        
            
    def get_smpl_faces(self):
        return self.smpl_model.faces_tensor
        
    def save_mesh(self, smpl_vts, dir):

        if isinstance(smpl_vts, Meshes):
            IO().save_mesh(data=smpl_vts, path=dir, include_textures=False)
            return 
        
        if not torch.is_tensor(smpl_vts):
            smpl_vts = torch.from_numpy(smpl_vts).float().cuda()
        
        if smpl_vts.ndimension()!=3:
            smpl_vts = smpl_vts[None,...]
        
        batch_size = smpl_vts.shape[0]
        device = smpl_vts.device
        


        faces = self.smpl_faces.expand(batch_size, -1, -1)
        
        tex_maps = self.default_tex_maps
        #print(posed_smpl_vertices.shape, faces.shape)
        mesh_obj = Meshes(verts = smpl_vts, faces = faces,
                                textures=tex_maps).to(device)
        
        IO().save_mesh(data=mesh_obj, path=dir, include_textures=False)
    
    def get_smpl_uv_vts_forsampling(self):
        #for uv indexing
        with torch.no_grad():            
            tmp = self.smpl_uv_vts
            tmp[...,[0,1]] = tmp[...,[1,0]]
            return tmp.float().cuda()
            

    def sample_points(self, posed_smpl_vertices, num_samples = 5, return_uv = False, return_normal = False):
        
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
        
        num_samples *= posed_smpl_vertices.shape[1]
        
        if return_normal:
            sampled_pts, sampled_normals, sampled_uv = sample_points_from_meshes(
                mesh_obj, num_samples = num_samples, return_normals = return_normal, return_uv = return_uv, verts_uvs = self.verts_uvs, faces_uvs = self.faces_uvs
            )
        else:
            sampled_pts, sampled_uv = sample_points_from_meshes(
                mesh_obj, num_samples = num_samples, return_normals = return_normal, return_uv = return_uv, verts_uvs = self.verts_uvs, faces_uvs = self.faces_uvs
            )
        
        if return_uv and return_normal:
            return sampled_pts, sampled_normals, sampled_uv
        
        if return_uv:
            return sampled_pts, sampled_uv
        
        if return_normal:
            return sampled_pts, sampled_uv
        
        return sampled_pts
        

    def get_nearest_pts_in_mesh_torch(self, posed_smpl_vertices, query_points_in_posed, k=1, num_samples = 5):
        
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
        
        
        # tmp=[]
        # for i in range(0, query_points_in_posed.shape[1], chunk):
        #     tmp.append(nearest_k_idx(src = query_points_in_posed[[0], i: i + chunk, :], tgt = sampled_pts, k=k))
        # vtx_id = torch.cat(tmp, 1)
        
        #print("query ", query_points_in_posed.shape, chunk)

        chunk = 1024 * 48 # 16 #chunk_dim
        with torch.no_grad():
            #dist = knn_points(query_points_in_posed, sampled_pts.reshape(1, -1, 3), K=k)
            #vtx_id = dist.idx
            vtx_id = torch.cat([knn_points(query_points_in_posed[[0], i: i + chunk, :], sampled_pts.reshape(1, -1, 3), K=k).idx 
                for i in range(0, query_points_in_posed.shape[1], chunk)],
                dim = 1)

            # vtx_id_mine = torch.cat([nearest_k_idx(src = query_points_in_posed[[0], i: i + chunk, :] , \
            #     tgt = sampled_pts, k=k) 
            #     for i in range(0, query_points_in_posed.shape[1], chunk)], 
            #     dim = 1)
                
            #print(vtx_id.shape, vtx_id_mine.shape)
            #print(vtx_id[0,5:8,0], vtx_id_mine[0,5:8,0])
        
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
    


    def get_nearest_pts_in_mesh_torch_grad(self, posed_smpl_vertices, query_points_in_posed, k=1, num_samples = 5):
        
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
        
        
        # tmp=[]
        # for i in range(0, query_points_in_posed.shape[1], chunk):
        #     tmp.append(nearest_k_idx(src = query_points_in_posed[[0], i: i + chunk, :], tgt = sampled_pts, k=k))
        # vtx_id = torch.cat(tmp, 1)
        
        #print("query ", query_points_in_posed.shape, chunk)

        chunk = 1024 * 48 # 16 #chunk_dim
        #with torch.no_grad():
            #dist = knn_points(query_points_in_posed, sampled_pts.reshape(1, -1, 3), K=k)
            #vtx_id = dist.idx
        vtx_id = torch.cat([knn_points(query_points_in_posed[[0], i: i + chunk, :], sampled_pts.reshape(1, -1, 3), K=k).idx 
            for i in range(0, query_points_in_posed.shape[1], chunk)],
            dim = 1)
                
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
    



    def get_nearest_pts_in_mesh_torch_part_2d(self, posed_smpl_vertices, query_points_in_posed, k=1, num_samples = 5, uv = None):
        
        #num_samples = 2
        
        device = posed_smpl_vertices.device

        if posed_smpl_vertices.ndimension()==2:
            posed_smpl_vertices = posed_smpl_vertices[None,...]
        
        batch_size = posed_smpl_vertices.shape[0]
        if not torch.is_tensor(posed_smpl_vertices):
            posed_smpl_vertices = torch.from_numpy(posed_smpl_vertices).float().to(device)
        
        sampled_pts, sampled_uv = posed_smpl_vertices, uv

        chunk = 1024 * 48 # 16 #chunk_dim
        with torch.no_grad():
            #dist = knn_points(query_points_in_posed, sampled_pts.reshape(1, -1, 3), K=k)
            #vtx_id = dist.idx
            vtx_id = torch.cat([knn_points(query_points_in_posed[[0], i: i + chunk, :], sampled_pts.reshape(1, -1, 3), K=k).idx 
                for i in range(0, query_points_in_posed.shape[1], chunk)],
                dim = 1)
                
        closest_smpl_verts = index_high_dimension(sampled_pts, vtx_id, dim=1)
        closest_smpl_uvs = index_high_dimension(sampled_uv, vtx_id, dim=1)

        closest_smpl_uvs[...,[0,1]] = closest_smpl_uvs[...,[1,0]]
        assert k==1 
        if k==1:
            closest_smpl_verts = closest_smpl_verts[:,:,0,:]
            closest_smpl_uvs = closest_smpl_uvs[:,:,0,:]
            return closest_smpl_uvs
    
        

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
        
        #print("query ", query_points_in_posed.shape, chunk)

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
  

        smpl_output = self.smpl_mesh.forward(betas=betas, body_pose=thetas[:,3:],
                                             transl=trans,
                                             global_orient=thetas[:,:3],
                                             return_verts=True)

        verts_rgb = torch.ones_like(smpl_output.vertices)

        mask_texture = Textures(verts_rgb=verts_rgb.cuda())

        mask_mesh = Meshes(verts=smpl_output.vertices,faces=self.faces.expand(b,-1,-1),textures=mask_texture).cuda()

        mask = self.mask_renderer(mask_mesh)[...,3]

        return mask