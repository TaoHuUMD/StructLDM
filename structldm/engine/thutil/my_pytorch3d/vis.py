import warnings
import pytorch3d
from pytorch3d.renderer.cameras import PerspectiveCameras, look_at_view_transform
from pytorch3d.renderer.mesh import shader, SoftPhongShader

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

from .textures import get_default_texture_maps

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

from pytorch3d.structures import Pointclouds
#from pytorch3d.vis.plotly_vis import AxisArgs, plot_batch_individually, plot_scene
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVOrthographicCameras, 
    PointsRasterizationSettings,
    PointsRenderer,
    PulsarPointsRenderer,
    PointsRasterizer,
    AlphaCompositor,
    NormWeightedCompositor
)

def get_camera(R, T, K, device):
    cameras = PerspectiveCameras(device = device, \
                    R=R, T=T, K=K) # focal_length=focal_length


    fx, fy = K[0][0][0], K[0][1][1]
    px, py = K[0][0][2], K[0][1][2]
    image_size = (1024,1024)
    print(fx, fy, px, py)

    off_cameras = PerspectiveCameras(device=device, R=R, T=T, focal_length=\
        ((fx, fy),), \
            principal_point=((px, py),), image_size=(image_size,))

    return off_cameras

def get_pointcloud(vertices):

    vertices = vertices.reshape(-1,3)
    rgb = np.random.rand(vertices.cpu().numpy().shape[0],3)
    rgb = torch.from_numpy(rgb).float().cuda()
    #torch.Tensor(pointcloud['rgb']).to(device)

    point_cloud = Pointclouds(points=[vertices], features=[rgb])
    return point_cloud

def get_mesh(vertices, faces):
    tex_maps = get_default_texture_maps()

    if not torch.is_tensor(vertices):
        vertices = torch.from_numpy(vertices).float().cuda()

    if faces is not None and (not torch.is_tensor(faces)):
        faces = torch.from_numpy(faces).float().cuda()

    batch_size = vertices.shape[0]
    device = vertices.device

    is_render_mesh = True
    if faces is None:
        is_render_mesh = False
    else:
        faces = faces.expand(batch_size, -1, -1)

    mesh = Meshes(verts = vertices, faces = faces,
                        textures=tex_maps).cuda()

    from pytorch3d.io import IO
    IO().save_mesh(data=mesh, path='/home/th/projects/neural_body/data/me.obj', include_textures=True)

    return mesh

def vis(mesh=None, pointcloud=None, camera=None):

    if pointcloud is not None:
        scenes = {
            "subplot_title": {
                "mesh_trace_title": mesh,
                "pointcloud_trace_title": pointcloud,
                "cameras_trace_title": camera
            }
        }
    else:
        scenes = {
            "subplot_title": {
                "mesh_trace_title": mesh,
                "cameras_trace_title": camera
            }
        }

    from pytorch3d.vis.plotly_vis import plot_scene
    fig = plot_scene(scenes, viewpoint_cameras=camera)
    #fig = plot_scene(scenes)
    fig.show()