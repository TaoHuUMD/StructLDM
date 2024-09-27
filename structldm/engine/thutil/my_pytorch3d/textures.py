from pytorch3d.io import load_obj, save_obj
import numpy as np
from torch._C import device
import torch

import os

from pytorch3d.renderer import (
    TexturesUV,
    SfMPerspectiveCameras,
    DirectionalLights,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
)


def get_smpl_face_uv():
    smpl_model_path = "../asset/smpl_uv.obj"
    verts, faces, aux = load_obj(
        smpl_model_path
    )
    tex = None
    #7576*2, 6890*3, 6889, 7575; faces:13000

    return aux.verts_uvs, faces.textures_idx

def get_smpl_uv():
    
    smpl_model_path = "../asset/smpl_uv.obj"
    uv = np.load(smpl_model_path)
    return torch.from_numpy(uv)

def get_smpl_uv_vts():
    
    smpl_model_path = "../asset/smpl_uv.obj"

    smpl_model_path = None
    flist = [smpl_model_path]
    for f in flist:
        if os.path.isfile(f):
            smpl_model_path = f
            break
    if smpl_model_path is None:
        print("3 no smlp uv obj file!")
        exit()

    uv = np.load(smpl_model_path) #allow_pickle=True
    return torch.from_numpy(uv)


def get_default_texture_maps_3():

    smpl_model_path = "../asset/smpl_uv.obj"

    smpl_model_path = None
    flist = [smpl_model_path]
    for f in flist:
        if os.path.isfile(f):
            smpl_model_path = f
            break
    if smpl_model_path is None:
        print("no smlp uv obj file!")
        exit()

    #texture_map = np.zeros((256, 256, 3)).astype(np.uint8)
    texture_map = np.random.rand(256, 256, 3)
    texture_map = texture_map.astype(np.float32)

    verts, faces, aux = load_obj(
        smpl_model_path
    )
    tex = None

    device="cuda"

    verts_uvs = aux.verts_uvs.to(device)  # (V, 2)
    faces_uvs = faces.textures_idx.to(device)  # (F, 3)

    mesh_tex = TexturesUV(maps=[torch.from_numpy(texture_map).to(device=device)], \
        faces_uvs=[faces_uvs], verts_uvs=[verts_uvs])
    return mesh_tex, verts_uvs, faces_uvs