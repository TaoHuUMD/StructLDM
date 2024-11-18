'''
This includes the module to realize the transformation between mesh and location map.
Part of the codes is adapted from https://github.com/Lotayou/densebody_pytorch
'''


from os.path import join
import torch
from torch import nn
import os

import numpy as np

'''
Index_UV_Generator is used to transform mesh and location map
The verts is in shape (B * V *C)
The UV map is in shape (B * H * W * C)
B: batch size;     V: vertex number;   C: channel number
H: height of uv map;  W: width of uv map
'''
#import sys
#sys.path.extend("..")

from ..num import index_2d_UVList, index_2d_UVMap, index_3d_volume
#from ..pointcloud import *
from ..load_smpl_tmp import *

from ..grid_sample_fix import grid_sample

from pytorch3d.io import load_obj

class Index_UV_Generator(nn.Module):
    def __init__(self, UV_height, UV_width=-1, uv_type='SMPL', data_dir=None, vtx_uv_mode = "firsthit", device="cpu"): #cuda
        #or avg
        super(Index_UV_Generator, self).__init__()

        if uv_type == 'SMPL':
            obj_file = 'smpl_fbx_template.obj'
            face_file = ""
        elif uv_type == 'BF':
            obj_file = 'smpl_boundry_free_template.obj'

        self.uv_type = uv_type

        if data_dir is None:
            d = os.path.dirname(__file__)
            data_dir = os.path.join(d, '..', 'data', 'uv_sampler')
        self.data_dir = data_dir
        self.h = UV_height
        self.w = self.h if UV_width < 0 else UV_width
        self.obj_file = obj_file
        self.para_file = 'paras_h{:04d}_w{:04d}_{}.npz'.format(self.h, self.w, self.uv_type)

        if not os.path.isfile(join(data_dir, self.para_file)):
            print(join(data_dir, self.para_file), "not exist")
            exit()

        device = torch.device("cuda:0" if device=="cuda" else "cpu")

        para = np.load(join(data_dir, self.para_file))

      
        self.v_index = torch.LongTensor(para['v_index']).to(device)
        self.bary_weights = torch.FloatTensor(para['bary_weights']).to(device)
        self.vt2v = torch.LongTensor(para['vt2v'])
        self.vt_count = torch.FloatTensor(para['vt_count'])
        self.texcoords = torch.FloatTensor(para['texcoords'])
        self.texcoords = 2 * self.texcoords - 1
        self.mask = torch.ByteTensor(para['mask'].astype('uint8'))

        self.vts_uv = torch.zeros((6890, 2)).float()
        self.vts_uv_cnt = torch.zeros((6890,1)).float()
        
        verts_uvs, faces_uvs = self.get_faces_uvs(os.path.join(self.data_dir, self.obj_file))
        self.faces_uvs = faces_uvs
        self.verts_uvs = verts_uvs

        if vtx_uv_mode == "firsthit":
            for i in range(len(self.vt2v)):
                id = self.vt2v[i]
                if self.vts_uv_cnt[id] == 0 :
                    self.vts_uv[id] = self.texcoords[i]
                self.vts_uv_cnt[id] += 1
            self.vts_uv = (self.vts_uv + 1.0)/2
            
        elif vtx_uv_mode == "avg":
            for i in range(len(self.vt2v)):
                id = self.vt2v[i]
                self.vts_uv[id] += self.texcoords[i]
                self.vts_uv_cnt[id] += 1
            self.vts_uv /= self.vts_uv_cnt
            self.vts_uv = (self.vts_uv + 1.0)/2
        
        self.vts_uv.requires_grad = False
        
    def get_faces_uvs(self, obj_path):
        verts, faces, aux = load_obj(obj_path)        
        verts_uvs = aux.verts_uvs.cuda()  # (V, 2)
        faces_uvs = faces.textures_idx.cuda()  # (F, 3)
        return verts_uvs, faces_uvs
 
    def posmap_to_vts(self, posmap, vts_uv_ = None):

        assert posmap.ndimension() == 4
        
        device = posmap.device
        batch_size = posmap.shape[0]

        if vts_uv_ is None:
            vts_uv_ = self.vts_uv
        
        vts_uv = vts_uv_.clone()
        vts_uv[...,[0,1]] = vts_uv[...,[1,0]]
        if self.uv_type == 'SMPL':
            imgs_tmp = torch.flip(posmap, [0, 1])
            rot_posmap = torch.rot90(imgs_tmp, -1, (1,2))
        elif self.uv_type == 'BF':
            imgs_tmp = torch.rot90(posmap, 1, (1,2))
            rot_posmap = torch.flip(imgs_tmp, [0, 1])
                
        vts_6890 = index_2d_UVList(rot_posmap, vts_uv)[0].permute(1,0)
        
        return vts_6890
    

    def index_posmap_by_vts_rotated_uvmap(self, posmap_, vts_uv_):
        #pos map: N, C, H, W
        
        assert posmap_.ndimension() == 4
        
        device = posmap_.device
        batch_size = posmap_.shape[0]
        
        #to N, H, W, C
        if posmap_.shape[2] == posmap_.shape[-1]:
            posmap = posmap_.permute(0,2,3,1)
        else:
            posmap = posmap_
                
        vts_uv = vts_uv_.clone().expand(batch_size, -1, -1)
        if vts_uv.shape[-1] != 2: #to B, N, 2
            vts_uv = vts_uv.permute(0, 2, 1)
            
        vts_uv[...,[0,1]] = vts_uv[...,[1,0]]
        if self.uv_type == 'SMPL':
            imgs_tmp = torch.flip(posmap, [0, 1])
            rot_posmap = torch.rot90(imgs_tmp, -1, (1,2))
        elif self.uv_type == 'BF':
            imgs_tmp = torch.rot90(posmap, 1, (1,2))
            rot_posmap = torch.flip(imgs_tmp, [0, 1])
                
        vts_features = index_2d_UVList(rot_posmap, vts_uv)
        
        return vts_features
    



    def index_posmap_by_vts(self, posmap_, vts_uv_, to_rotate = True):
        #pos map: N, C, H, W
        
        assert posmap_.ndimension() == 4
        
        device = posmap_.device
        batch_size = posmap_.shape[0]

        #to N, H, W, C
        if posmap_.shape[2] == posmap_.shape[-1]:
            posmap = posmap_.permute(0,2,3,1)
        else:
            posmap = posmap_
                
        vts_uv = vts_uv_.clone().expand(batch_size, -1, -1)
        if vts_uv.shape[-1] != 2: #to B, N, 2
            vts_uv = vts_uv.permute(0, 2, 1)

        
        vts_uv[...,[0,1]] = vts_uv[...,[1,0]]    
        if to_rotate:
            if self.uv_type == 'SMPL':
                imgs_tmp = torch.flip(posmap, [0, 1])
                rot_posmap = torch.rot90(imgs_tmp, -1, (1,2))
            elif self.uv_type == 'BF':
                imgs_tmp = torch.rot90(posmap, 1, (1,2))
                rot_posmap = torch.flip(imgs_tmp, [0, 1])
        else:
            rot_posmap = posmap
                
        vts_features = index_2d_UVList(rot_posmap, vts_uv)
        
        return vts_features
    
    def index_3dposmap_by_vts(self, posmap_, vts_uv_, new_3d_shape = (1,1,1,1,1)):
        assert posmap_.ndimension() == 4
        
        device = posmap_.device
        batch_size = posmap_.shape[0]

        #to N, H, W, C
        if posmap_.shape[2] == posmap_.shape[-1]:
            posmap = posmap_.permute(0,2,3,1)
        else:
            posmap = posmap_
            
        vts_uv = vts_uv_.clone()
        if vts_uv.shape[-1] != 3: #to B, N, 2
            vts_uv = vts_uv.permute(0, 2, 1)
            
        vts_uv[...,[1,2]] = vts_uv[...,[2,1]]
        if self.uv_type == 'SMPL':
            imgs_tmp = torch.flip(posmap, [0, 1])
            rot_posmap = torch.rot90(imgs_tmp, -1, (1,2))
        elif self.uv_type == 'BF':
            imgs_tmp = torch.rot90(posmap, 1, (1,2))
            rot_posmap = torch.flip(imgs_tmp, [0, 1])
            
        vts_features = index_3d_volume(rot_posmap.reshape(new_3d_shape), vts_uv)
        
        return vts_features
                
    def index_posmap_by_uvmap(self, posmap_, vts_uv_):

        assert posmap_.ndimension() == 4
        
        device = posmap_.device
        batch_size = posmap_.shape[0]
        
        if posmap_.shape[2] == posmap_.shape[-1]:
            posmap = posmap_.permute(0,2,3,1)
        else:
            posmap = posmap_
                    
        vts_uv = vts_uv_.clone()
        if vts_uv.shape[-1] != 2: #to B, H, W, 2
            vts_uv = vts_uv.permute(0, 2, 3, 1)
        vts_uv[...,[0,1]] = vts_uv[...,[1,0]]
                
        if self.uv_type == 'SMPL':
            imgs_tmp = torch.flip(posmap, [0, 1])
            rot_posmap = torch.rot90(imgs_tmp, -1, (1,2))
        elif self.uv_type == 'BF':
            imgs_tmp = torch.rot90(posmap, 1, (1,2))
            rot_posmap = torch.flip(imgs_tmp, [0, 1])
                
        uvmap_feature = index_2d_UVMap(rot_posmap.permute(0,3,1,2), vts_uv.permute(0,3,1,2))
        
        return uvmap_feature
    
    def gen_transf_mtx_from_vtransf(self, vtransf, scaling=1.0):
        '''
        interpolate the local -> global coord transormation given such transformations defined on 
        the body verts (pre-computed) and barycentric coordinates of the query points from the uv map.
        Note: The output of this function, i.e. the transformation matrix of each point, is not a pure rotation matrix (SO3).
        
        args:
            vtransf: [batch, #verts, 3, 3] # per-vertex rotation matrix
            bary_coords: [uv_size, uv_size, 3] # barycentric coordinates of each query point (pixel) on the query uv map 
            faces: [uv_size, uv_size, 3] # the vert id of the 3 vertices of the triangle where each uv pixel locates
        returns: 
            [batch, uv_size, uv_size, 3, 3], transformation matrix for points on the uv surface
        '''
        #  
        device = vtransf.device
        faces = self.v_index.to(device)
        bary_coords = self.bary_weights.to(device)
        vtransf_by_tris = vtransf[:, faces] # shape will be [batch, uvsize, uvsize, 3, 3, 3], where the the last 2 dims being the transf (pure rotation) matrices, the other "3" are 3 points of each triangle
        

        transf_mtx_uv_pts = torch.einsum('bpqijk,pqi->bpqjk', vtransf_by_tris, bary_coords) # [batch, uvsize, uvsize, 3, 3], last 2 dims are the rotation matix
        transf_mtx_uv_pts *= scaling
        return transf_mtx_uv_pts
    
    
    def gen_pts_from_uvmap(self, vt6890, scaling=1.0):
        '''
        interpolate the local -> global coord transormation given such transformations defined on 
        the body verts (pre-computed) and barycentric coordinates of the query points from the uv map.
        Note: The output of this function, i.e. the transformation matrix of each point, is not a pure rotation matrix (SO3).
        
        args:
            vtransf: [batch, #verts, 3, 3] # per-vertex rotation matrix
            bary_coords: [uv_size, uv_size, 3] # barycentric coordinates of each query point (pixel) on the query uv map 
            faces: [uv_size, uv_size, 3] # the vert id of the 3 vertices of the triangle where each uv pixel locates
        returns: 
            [batch, uv_size, uv_size, 3, 3], transformation matrix for points on the uv surface
        '''
        device = vt6890.device
        faces = self.v_index.to(device)
        bary_coords = self.bary_weights.to(device)
        vtransf_by_tris = vt6890[:, faces] # shape will be [batch, uvsize, uvsize, 3, 3, 3], where the the last 2 dims being the transf (pure rotation) matrices, the other "3" are 3 points of each triangle

        uv_pts = torch.einsum('bpqij,pqi->bpqj', vtransf_by_tris, bary_coords) # [batch, uvsize, uvsize, 3, 3], last 2 dims are the rotation matix
        uv_pts *= scaling
        return uv_pts
    
    def get_uv_mask(self):
        return self.mask
    
    def get_UV_map(self, verts):
        bary_weights = self.bary_weights.type(verts.dtype).to(verts.device)
        v_index = self.v_index.to(verts.device)

        if verts.dim() == 2:
            verts = verts.unsqueeze(0)

        im = verts[:, v_index, :]
        bw = bary_weights[:, :, None, :]

        im = torch.matmul(bw, im).squeeze(dim=3)
        
        batch = im.shape[0]
                    
        return im #, mask, self.vts_uv.clone().detach().to(verts.device)[None,...].expand(batch, -1, -1)

    def get_UV_map_org(self, verts):
        self.bary_weights = self.bary_weights.type(verts.dtype).to(verts.device)
        self.v_index = self.v_index.to(verts.device)

        if verts.dim() == 2:
            verts = verts.unsqueeze(0)

        im = verts[:, self.v_index, :]
        bw = self.bary_weights[:, :, None, :]
                
        im = torch.matmul(bw, im).squeeze(dim=3)
        
        batch = im.shape[0]
                    
        return im #, mask, self.vts_uv.clone().detach().to(verts.device)[None,...].expand(batch, -1, -1)



    def get_vts_uv(self):
        with torch.no_grad():
            return self.vts_uv.clone().detach()[None,...]

    def resample(self, uv_map):
        batch_size, _, _, channel_num = uv_map.shape
        v_num = self.vt_count.shape[0]
        self.texcoords = self.texcoords.type(uv_map.dtype).to(uv_map.device)
        self.vt2v = self.vt2v.to(uv_map.device)
        self.vt_count = self.vt_count.type(uv_map.dtype).to(uv_map.device)

        uv_grid = self.texcoords[None, None, :, :].expand(batch_size, -1, -1, -1)

        vt = grid_sample(uv_map.permute(0, 3, 1, 2), uv_grid, mode='bilinear')
        vt = vt.squeeze(2).permute(0, 2, 1)
        v = vt.new_zeros([batch_size, v_num, channel_num])
        v.index_add_(1, self.vt2v, vt)
        v = v / self.vt_count[None, :, None]
        return v

    # just used for the generation of GT UVmaps
    def forward(self, verts):
        return self.get_UV_map(verts)
    
    def render_feature_map(self, verts, reso):

        # 
        if self.h == reso:
            bary_weights = self.bary_weights.type(verts.dtype).to(verts.device)
            v_index = self.v_index.to(verts.device)
        else:
            bary_weights = getattr(self, "bary_weights_%d" % reso).type(verts.dtype).to(verts.device)
            v_index = getattr(self, "v_index_%d" % reso).to(verts.device)

        if verts.dim() == 2:
            verts = verts.unsqueeze(0)

        im = verts[:, v_index, :]
        bw = bary_weights[:, :, None, :]
                
        batch_size = im.shape[0]#.expand(batch_size, -1, -1, -1)
        im = torch.matmul(bw, im).squeeze(dim=3)
        
        batch = im.shape[0]

        return im

    def gen_part_map(self, face_part):
        part_map = np.zeros(self.bary_id.shape) - 1
        mask = self.bary_id >= 0
        part_map[mask] = face_part[self.bary_id[mask]]
        return part_map
