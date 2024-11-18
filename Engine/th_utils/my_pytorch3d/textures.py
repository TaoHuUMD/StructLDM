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


def get_smpl_uv_vts():
    
    smpl_model_path = None
    flist = ["./data/asset/uv_table.npy"]
    for f in flist:
        if os.path.isfile(f):
            smpl_model_path = f
            break
    if smpl_model_path is None:
        print("no smlp uv obj file!")
        exit()

    uv = np.load(smpl_model_path) #allow_pickle=True
    return torch.from_numpy(uv)

def get_default_texture_maps():

    smpl_model_path = None
    flist = ["./data/asset/smpl_uv.obj"]
    for f in flist:
        if os.path.isfile(f):
            smpl_model_path = f
            break
    if smpl_model_path is None:
        print("no smlp uv obj file!")
        exit()

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