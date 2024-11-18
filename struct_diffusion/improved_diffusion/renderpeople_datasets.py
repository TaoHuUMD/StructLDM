from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np
import os, imageio, cv2, time, copy, math, json
from random import sample
from cv2 import Rodrigues as rodrigues

import blobfile as bf
# from mpi4py import MPI

from smpl.smpl_numpy import SMPL
import torch
import torch.nn.functional as F

def load_triplane_data(
    *,
    data_name,
    data_dir,
    batch_size,
    image_size,
    class_cond=False,
    num_subjects=1000,
    layer_idx=None,
    deterministic=False,
    world_size=1,
    rank=0,
):
    """
    For a dataset, create a generator over (images, kwargs) pairs.

    Each images is an NCHW float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for class labels, in which case the key is "y"
    and the values are integer tensors of class labels.

    :param data_dir: a dataset directory.
    :param batch_size: the batch size of each returned pair.
    :param image_size: the size to which images are resized.
    :param class_cond: if True, include a "y" key in returned dicts for class
                       label. If classes are not available and this is true, an
                       exception will be raised.
    :param deterministic: if True, yield results in a deterministic order.
    :param random_crop: if True, randomly crop the images for augmentation.
    :param random_flip: if True, randomly flip the images for augmentation.
    """
    if not data_dir:
        raise ValueError("unspecified data directory")

    if data_name == 'renderpeople':
        dataset = TriplaneDataset(
            image_size,
            data_dir,
            num_subjects,
            layer_idx=layer_idx,
            classes=None,
            # shard=MPI.COMM_WORLD.Get_rank(),
            # num_shards=MPI.COMM_WORLD.Get_size(),
        )

    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=True
        )
    else:
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
        loader = DataLoader(dataset, sampler=train_sampler, batch_size=batch_size, num_workers=3, drop_last=False, pin_memory=True)
        
    while True:
        yield from loader

class TriplaneDataset(Dataset):
    def __init__(
        self,
        resolution,
        data_dir,
        num_subjects,
        classes=None,
        layer_idx=None,
        # shard=0,
        # num_shards=1,
    ):
        super().__init__()
        self.resolution = resolution
        self.layer_idx = layer_idx
        ckpt_path = data_dir #os.path.join(data_dir, '050000.tar')
        self.num_subjects = num_subjects
        self.layer_num = 4
        print('Reloading from', ckpt_path)
       
        triplane_ft_dir = os.path.dirname(ckpt_path)
        self.tri_plane_lst = []
        with open(os.path.join(triplane_ft_dir, 'human_list.txt')) as f:
            for line in f.readlines()[0:num_subjects]:
                line = line.strip()
                self.tri_plane_lst.append(os.path.join(triplane_ft_dir, line))

    def __len__(self):
        return self.num_subjects * self.layer_num #self.local_images.shape[0] * self.local_images.shape[1]

    def __getitem__(self, idx):
        instance_idx = idx // self.layer_num#self.local_images.shape[1]
        layer_idx = idx % self.layer_num #self.local_images.shape[1]

        if self.layer_idx is not None:
            layer_idx = int(self.layer_idx)

        tri_plane = torch.load(self.tri_plane_lst[instance_idx], map_location='cpu')['network_fn_state_dict']['tri_planes'].squeeze(0)
        tri_plane = tri_plane.reshape(tri_plane.shape[0], -1, *tri_plane.shape[-2:])

        if layer_idx == 0:
            layer_condition = torch.zeros((tri_plane.shape[1], tri_plane.shape[2], tri_plane.shape[3]), dtype=tri_plane.dtype)
        else:
            layer_condition = tri_plane[layer_idx-1]
        out_dict = {}
        out_dict["y"] = np.array(layer_idx, dtype=np.int64)
        return tri_plane[layer_idx], layer_condition, out_dict
  
def get_rays(H, W, K, R, T):
    # calculate the camera origin
    rays_o = -np.dot(R.T, T).ravel()
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

def get_bound_corners(bounds):
    min_x, min_y, min_z = bounds[0]
    max_x, max_y, max_z = bounds[1]
    corners_3d = np.array([
        [min_x, min_y, min_z],
        [min_x, min_y, max_z],
        [min_x, max_y, min_z],
        [min_x, max_y, max_z],
        [max_x, min_y, min_z],
        [max_x, min_y, max_z],
        [max_x, max_y, min_z],
        [max_x, max_y, max_z],
    ])
    return corners_3d

def project(xyz, K, RT):
    """
    xyz: [N, 3]
    K: [3, 3]
    RT: [3, 4]
    """
    xyz = np.dot(xyz, RT[:, :3].T) + RT[:, 3:].T
    xyz = np.dot(xyz, K.T)
    xy = xyz[:, :2] / xyz[:, 2:]
    return xy

def get_bound_2d_mask(bounds, K, pose, H, W):
    corners_3d = get_bound_corners(bounds)
    corners_2d = project(corners_3d, K, pose)
    corners_2d = np.round(corners_2d).astype(int)
    mask = np.zeros((H, W), dtype=np.uint8)
    cv2.fillPoly(mask, [corners_2d[[0, 1, 3, 2, 0]]], 1)
    cv2.fillPoly(mask, [corners_2d[[4, 5, 7, 6, 4]]], 1) # 4,5,7,6,4
    cv2.fillPoly(mask, [corners_2d[[0, 1, 5, 4, 0]]], 1)
    cv2.fillPoly(mask, [corners_2d[[2, 3, 7, 6, 2]]], 1)
    cv2.fillPoly(mask, [corners_2d[[0, 2, 6, 4, 0]]], 1)
    cv2.fillPoly(mask, [corners_2d[[1, 3, 7, 5, 1]]], 1)
    return mask

def get_near_far(bounds, ray_o, ray_d):
    """calculate intersections with 3d bounding box"""
    bounds = bounds + np.array([-0.01, 0.01])[:, None]
    ray_d[ray_d==0.0] = 1e-8
    nominator = bounds[None] - ray_o[:, None]
    # calculate the step of intersections at six planes of the 3d bounding box
    d_intersect = (nominator / ray_d[:, None]).reshape(-1, 6)
    # calculate the six interections
    p_intersect = d_intersect[..., None] * ray_d[:, None] + ray_o[:, None]
    # calculate the intersections located at the 3d bounding box
    min_x, min_y, min_z, max_x, max_y, max_z = bounds.ravel()
    eps = 1e-6
    p_mask_at_box = (p_intersect[..., 0] >= (min_x - eps)) * \
                    (p_intersect[..., 0] <= (max_x + eps)) * \
                    (p_intersect[..., 1] >= (min_y - eps)) * \
                    (p_intersect[..., 1] <= (max_y + eps)) * \
                    (p_intersect[..., 2] >= (min_z - eps)) * \
                    (p_intersect[..., 2] <= (max_z + eps))
    # obtain the intersections of rays which intersect exactly twice
    mask_at_box = p_mask_at_box.sum(-1) == 2
    # TODO
    # mask_at_box = p_mask_at_box.sum(-1) >= 1

    p_intervals = p_intersect[mask_at_box][p_mask_at_box[mask_at_box]].reshape(
        -1, 2, 3)

    # calculate the step of intersections
    ray_o = ray_o[mask_at_box]
    ray_d = ray_d[mask_at_box]
    norm_ray = np.linalg.norm(ray_d, axis=1)
    d0 = np.linalg.norm(p_intervals[:, 0] - ray_o, axis=1) / norm_ray
    d1 = np.linalg.norm(p_intervals[:, 1] - ray_o, axis=1) / norm_ray
    near = np.minimum(d0, d1)
    far = np.maximum(d0, d1)

    return near, far, mask_at_box

def sample_ray_RenderPeople_batch(H, W, K, R, T, bounds):

    ray_o, ray_d = get_rays(H, W, K, R, T)
    pose = np.concatenate([R, T], axis=1)

    bound_mask = get_bound_2d_mask(bounds, K_scale, pose, H, W)

    ray_o = ray_o.reshape(-1, 3).astype(np.float32)
    ray_d = ray_d.reshape(-1, 3).astype(np.float32)
    near, far, mask_at_box = get_near_far(bounds, ray_o, ray_d)
    near = near.astype(np.float32)
    far = far.astype(np.float32)

    near_all = np.zeros_like(ray_o[:,0])
    far_all = np.ones_like(ray_o[:,0])
    near_all[mask_at_box] = near 
    far_all[mask_at_box] = far 
    near = near_all
    far = far_all

    coord = np.zeros([len(ray_o), 2]).astype(np.int64)

    return ray_o, ray_d, near, far, mask_at_box


class RenderPeopleDatasetBatch(Dataset):
    def __init__(self, data_root=None, image_scaling=0.5):
        super(RenderPeopleDatasetBatch, self).__init__()
        self.data_root = data_root
        self.image_scaling = image_scaling

        self.train_view = self.test_view = [x for x in range(382)]
        self.output_view = self.train_view if split == 'train' else self.test_view
   
        print("output view: ", self.output_view)

        camera_file = os.path.join(data_root, 'camera.json')
        self.camera = json.load(open(camera_file))


    def prepare_smpl_params(self, smpl_path, pose_index):
        params_ori = dict(np.load(smpl_path, allow_pickle=True))['smpl'].item()
        params = {}
        params['shapes'] = np.array(params_ori['betas']).astype(np.float32)
        params['poses'] = np.zeros((1,72)).astype(np.float32)
        params['poses'][:, :3] = np.array(params_ori['global_orient'][pose_index]).astype(np.float32)
        params['poses'][:, 3:] = np.array(params_ori['body_pose'][pose_index]).astype(np.float32)
        # params['R'] = np.array(rodrigues(params_ori['global_orient'][pose_index:pose_index+1])[0]).astype(np.float32)
        params['R'] = np.eye(3).astype(np.float32)
        # params['Th'] = np.zeros((1,3)).astype(np.float32)
        params['Th'] = np.array(params_ori['transl'][0:1]).astype(np.float32)
        return params

    def prepare_input(self, smpl_path, pose_index):

        params = self.prepare_smpl_params(smpl_path, pose_index)
        xyz, joints = self.smpl_model(params['poses'], params['shapes'].reshape(-1))
        xyz = (np.matmul(xyz, params['R'].transpose()) + params['Th']).astype(np.float32)
        
        vertices = xyz

        # obtain the original bounds for point sampling
        min_xyz = np.min(xyz, axis=0)
        max_xyz = np.max(xyz, axis=0)
        min_xyz -= 0.05
        max_xyz += 0.05
        world_bounds = np.stack([min_xyz, max_xyz], axis=0)

        return world_bounds, vertices, params

    def __getitem__(self, index):
        """
            pose_index: [0, number of used poses), the pose index of selected poses
            view_index: [0, number of all view)
            mask_at_box_all: mask of 2D bounding box which is the 3D bounding box projection 
                training: all one array, not fixed length, no use for training
                Test: zero and one array, fixed length which can be reshape to (H,W)
            bkgd_msk_all: mask of foreground and background
                trainning: for acc loss
                test: no use
        """

        view_index = index % len(self.output_view)

        K = np.array(self.cams[f'camera{str(view_index).zfill(2)}']['K'])
        R = np.array(self.cams[f'camera{str(view_index).zfill(2)}']['R'])
        T = np.array(self.cams[f'camera{str(view_index).zfill(2)}']['T']).reshape(-1, 1)

        # rescaling
        H, W = 512, 512
        K[:2] = K[:2]*self.image_scaling

        # Prepare the smpl input, including the current pose and canonical pose
        smpl_path = os.path.join(self.data_root, 'smpl.npz')   
        world_bounds, vertices, params = self.prepare_input(smpl_path, 0)

        # Sample rays in target space world coordinate
        ray_o, ray_d, near, far, mask_at_box = sample_ray_RenderPeople_batch(
                H, W, K, R, T, world_bounds)
      
        return ray_o, ray_d, near, far

    def __len__(self):
        return len(self.output_view)