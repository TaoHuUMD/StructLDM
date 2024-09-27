
import torch
import torch.nn as nn

from pytorch3d.renderer.blending import (
    BlendParams,
    hard_rgb_blend,
    sigmoid_alpha_blend,
    softmax_rgb_blend,
)
from pytorch3d.renderer.lighting import PointLights
from pytorch3d.renderer.materials import Materials
from pytorch3d.renderer.mesh.shading import flat_shading, gouraud_shading, phong_shading

from pytorch3d.ops import interpolate_face_attributes

from typing import List, Optional
from pytorch3d.io import load_obj, save_obj

import numpy as np

from pytorch3d.renderer import (
    TexturesUV,
    SfMPerspectiveCameras,
    DirectionalLights,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
)
from pytorch3d.structures import Meshes

def load_obj_mesh_tex(
    files: list,
    device = None
):
    mesh_list = []

    texture_map = np.zeros((256, 256, 3)).astype(np.uint8)
    texture_map = texture_map.astype(np.float32) / 255.


    for f_obj in files:
        verts, faces, aux = load_obj(
            f_obj
        )
        tex = None


        verts_uvs = aux.verts_uvs.to(device)  # (V, 2)
        faces_uvs = faces.textures_idx.to(device)  # (F, 3)

        #image = list(tex_maps.values())[0].to(device)[None]

        #tex = TexturesUV(
        #    verts_uvs=[verts_uvs], faces_uvs=[faces_uvs], maps=image
        #)

        mesh_tex = TexturesUV(maps=[torch.from_numpy(texture_map).to(device=device)], faces_uvs=[faces_uvs],
                              verts_uvs=[verts_uvs])
        mesh = Meshes(
            verts=[verts.to(device)], faces=[faces.verts_idx.to(device)], textures=mesh_tex
        )

        mesh_list.append(mesh)
    if len(mesh_list) == 1:
        return mesh_list[0]
    return join_meshes_as_batch(mesh_list)

def smpl_pkl_to_mesh():
    # Load SMPL and texture data
    with open(verts_filename, 'rb') as f:
        data = pickle.load(f, encoding='latin1')
        v_template = torch.Tensor(data['v_template']).to(device)  # (6890, 3)
    ALP_UV = loadmat(data_filename)
    tex = torch.from_numpy(_read_image(file_name=tex_filename, format='RGB') / 255.).unsqueeze(0).to(device)

    verts = torch.from_numpy((ALP_UV["All_vertices"]).astype(int)).squeeze().to(device)  # (7829, 1)
    U = torch.Tensor(ALP_UV['All_U_norm']).to(device)  # (7829, 1)
    V = torch.Tensor(ALP_UV['All_V_norm']).to(device)  # (7829, 1)
    faces = torch.from_numpy((ALP_UV['All_Faces'] - 1).astype(int)).to(device)  # (13774, 3)
    face_indices = torch.Tensor(ALP_UV['All_FaceIndices']).squeeze()

    # Map each face to a (u, v) offset
    offset_per_part = {}
    already_offset = set()
    cols, rows = 4, 6
    for i, u in enumerate(np.linspace(0, 1, cols, endpoint=False)):
        for j, v in enumerate(np.linspace(0, 1, rows, endpoint=False)):
            part = rows * i + j + 1  # parts are 1-indexed in face_indices
            offset_per_part[part] = (u, v)

    # iterate over faces and offset the corresponding vertex u and v values
    for i in range(len(faces)):
        face_vert_idxs = faces[i]
        part = face_indices[i]
        offset_u, offset_v = offset_per_part[int(part.item())]

        for vert_idx in face_vert_idxs:
            # vertices are reused, but we don't want to offset multiple times
            if vert_idx.item() not in already_offset:
                # offset u value
                U[vert_idx] = U[vert_idx] / cols + offset_u
                # offset v value
                # this also flips each part locally, as each part is upside down
                V[vert_idx] = (1 - V[vert_idx]) / rows + offset_v
                # add vertex to our set tracking offsetted vertices
                already_offset.add(vert_idx.item())

    # invert V values
    U_norm, V_norm = U, 1 - V


    # create our verts_uv values
    verts_uv = torch.cat([U_norm[None], V_norm[None]], dim=2)  # (1, 7829, 2)

    # There are 6890 xyz vertex coordinates but 7829 vertex uv coordinates. 
    # This is because the same vertex can be shared by multiple faces where each face may correspond to a different body part.  
    # Therefore when initializing the Meshes class,
    # we need to map each of the vertices referenced by the DensePose faces (in verts, which is the "All_vertices" field)
    # to the correct xyz coordinate in the SMPL template mesh.
    v_template_extended = torch.stack(list(map(lambda vert: v_template[vert - 1], verts))).unsqueeze(0).to(
        device)  # (1, 7829, 3)

    # add a batch dimension to faces
    faces = faces.unsqueeze(0)
    

#load pkl files
def load_smpl_pytorch3d(
        files: list,
        device = None
    ):

    mesh_list = []






    texture_map = np.zeros((256, 256, 3)).astype(np.uint8)
    texture_map = texture_map.astype(np.float32) / 255.

    for f_obj in files:
        verts, faces, aux = load_obj(
            f_obj
        )
        tex = None

        verts_uvs = aux.verts_uvs.to(device)  # (V, 2)
        faces_uvs = faces.textures_idx.to(device)  # (F, 3)

        # image = list(tex_maps.values())[0].to(device)[None]

        # tex = TexturesUV(
        #    verts_uvs=[verts_uvs], faces_uvs=[faces_uvs], maps=image
        # )

        mesh_tex = TexturesUV(maps=[torch.from_numpy(texture_map).to(device=device)], faces_uvs=[faces_uvs],
                              verts_uvs=[verts_uvs])
        mesh = Meshes(
            verts=[verts.to(device)], faces=[faces.verts_idx.to(device)], textures=mesh_tex
        )

        mesh_list.append(mesh)
    if len(mesh_list) == 1:
        return mesh_list[0]
    return join_meshes_as_batch(mesh_list)


def load_mesh(
    files: list,
    device=None,
    load_textures: bool = True,
    create_texture_atlas: bool = False,
    texture_atlas_size: int = 4,
    texture_wrap: Optional[str] = "repeat",
):
    """
    Load meshes from a list of .obj files using the load_obj function, and
    return them as a Meshes object. This only works for meshes which have a
    single texture image for the whole mesh. See the load_obj function for more
    details. material_colors and normals are not stored.

    Args:
        f: A list of file-like objects (with methods read, readline, tell,
        and seek), pathlib paths or strings containing file names.
        device: Desired device of returned Meshes. Default:
            uses the current device for the default tensor type.
        load_textures: Boolean indicating whether material files are loaded

    Returns:
        New Meshes object.
    """
    mesh_list = []
    for f_obj in files:
        verts, faces, aux = load_obj(
            f_obj,
            load_textures=load_textures,
            create_texture_atlas=create_texture_atlas,
            texture_atlas_size=texture_atlas_size,
            texture_wrap=texture_wrap,
        )
        tex = None
        if create_texture_atlas:
            # TexturesAtlas type
            tex = TexturesAtlas(atlas=[aux.texture_atlas.to(device)])
        else:
            # TexturesUV type
            tex_maps = aux.texture_images

            print(tex_maps)
            print("tex")
            exit()

            if tex_maps is not None and len(tex_maps) > 0:
                verts_uvs = aux.verts_uvs.to(device)  # (V, 2)
                faces_uvs = faces.textures_idx.to(device)  # (F, 3)
                image = list(tex_maps.values())[0].to(device)[None]
                tex = TexturesUV(
                    verts_uvs=[verts_uvs], faces_uvs=[faces_uvs], maps=image
                )

        mesh = Meshes(
            verts=[verts.to(device)], faces=[faces.verts_idx.to(device)], textures=tex
        )
        mesh_list.append(mesh)
    if len(mesh_list) == 1:
        return mesh_list[0]
    return join_meshes_as_batch(mesh_list)