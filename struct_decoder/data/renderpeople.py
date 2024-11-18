import os

import torch
from torch.utils.data.dataset import Dataset

import numpy as np
import cv2
import json
import random
from PIL import Image

import logging

from pdb import set_trace as st

logger = logging.getLogger(f"care.{__name__}")

from scipy.spatial.transform import Slerp
from scipy.spatial.transform import Rotation as R

def lerp(a, b, t):
    return a + (b - a) * t

def interpolate_matrices(rot1, rot2, trans1, trans2, t):
    # SLERP for rotations
    slerp = Slerp([0, 1], R.from_matrix([rot1, rot2]))
    slerped_rot = slerp([t])
    
    # LERP for translations
    lerped_trans = lerp(trans1, trans2, t)

    return slerped_rot.as_matrix()[0], lerped_trans

def interpolate_poses(poses, num_interpolations):
    interpolated_poses = []

    for i in range(len(poses)):
        rot1, trans1 = poses[i]
        rot2, trans2 = poses[(i + 1) % len(poses)]

        for t in np.linspace(0, 1, num_interpolations + 1, endpoint=False):
            slerped_rot, lerped_trans = interpolate_matrices(rot1, rot2, trans1, trans2, t)
            interpolated_poses.append((slerped_rot, lerped_trans))

    return interpolated_poses

def get_KRTD(camera, view_index = 0):
    camera = camera['camera{:04d}'.format(view_index)]
    K = np.array(camera['K'])
    R = np.array(camera['R'])
    R_flip = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    R = R @ R_flip
    T = np.array(camera['T'])
    D = None
    return K, R, T, D

def load_smplx_params(all_smpl, frame_id):
    # poses,
    # expression,
    # shapes,
    # Rh,
    # Th,
    return {
        k: np.array(v[frame_id], dtype=np.float32) for k, v in all_smpl.items() if k != "id"
    }


# root_dir: ${root_data_dir}
# subject_ids: ${root_data_dir}/human_list.txt
# image: ${root_data_dir}/{people_id}/img/{camera}/{frame:04d}.jpg
# image_mask: ${root_data_dir}/{people_id}/mask/{camera}/{frame:04d}.png
# image_part_mask: ${root_data_dir}/{people_id}/mask/{camera}/{frame:04d}.png
# smplx_poses: ${root_data_dir}/{people_id}/outputs_re_fitting/refit_smpl_2nd.npz
# cam_path: ${root_data_dir}/{people_id}/cameras.json
# smplx_topology: ${smplx_dir}/SMPL_${model.smplx_gender}.pkl
class RenderPeopleSViewDataset(Dataset):
    def __init__(
        self,        
        root_dir,
        subject_ids,
        smplx_poses,
        image,
        image_mask,
        image_part_mask,
        cam_path,
        transform,
        frame_list=None,
        cond_cameras=None,
        sample_cameras=True,
        camera_id=None,
        image_height=512,
        image_width=256,        
        opt = None,
        phase = "train"
    ):
        super().__init__()

        self.image_height = image_height
        self.image_width = image_width
        self.ref_frame = 0
        self.transform = transform
        self.opt = opt

        if subject_ids is None:
            hl = os.listdir(root_dir)
            human_list = []
            for i in hl:
                if os.path.isdir(os.path.join(root_dir, i)):
                    human_list.append(i)
            human_list = sorted(human_list)
        else:
            with open(subject_ids, 'r') as f:
                human_list = f.read().splitlines()

        self.subject_ids = human_list
        self.root_dir = root_dir

        self.smpl_folder = self.root_dir

        if frame_list is None:
            n_frames = len(os.listdir(os.path.join(self.root_dir, self.subject_ids[0], 'img', 'camera0000')))
            self.frame_list = [str(fid) for fid in range(0, n_frames, self.opt.data_step)]

        self.subject_ids = self.subject_ids[:self.opt.training.debug_data_size]
            
        self.subject_ids_full = self.subject_ids.copy()

        self.image_path = image
        self.image_mask_path = image_mask
        self.image_part_mask_path = image_part_mask

        all_cameras = self.load_all_cameras(cam_path)


        self.phase = phase

        # TODO: samling training camera logics

        self.cameras = all_cameras

        # these are ids
        self.cond_cameras = cond_cameras

        self.sample_cameras = sample_cameras
        self.camera_id = camera_id

        self.all_smpl = self.load_all_smpl(smplx_poses)

        self.geo_list = []

    def load_all_smpl(self, smplx_poses):
        all_smpl = {}
        #print("all ", len(self.subject_ids))
        #for i in range(len(self.subject_ids)):
        valid_subject_ids = []
        for people_id in self.subject_ids:
            #people_id = self.subject_ids[i]
            current_smplx_path = smplx_poses.format(people_id=people_id)
            if not os.path.isfile(current_smplx_path):
                continue
            valid_subject_ids.append(people_id)
            smpl_param = dict(np.load(current_smplx_path, allow_pickle=True))['smpl'].item()
            poses = np.zeros((smpl_param['body_pose'].shape[0], 72)).astype(np.float32)
            poses[:, :3] = np.array(smpl_param['global_orient']).astype(np.float32)
            poses[:, 3:] = np.array(smpl_param['body_pose']).astype(np.float32)

            shapes = np.array(smpl_param['betas']).astype(np.float32)
            shapes = np.repeat(shapes[:], poses.shape[0], axis=0)
            Rh = smpl_param['global_orient'].astype(np.float32)
            Th = smpl_param['transl'].astype(np.float32)
            current_smplx = {
                'shapes': shapes,
                'Rh': Rh * 0, #FIXME: hack
                'Th': Th,
                'poses': poses,
            }
            all_smpl[people_id] = current_smplx

        #print("valid ", len(all_smpl))
        self.subject_ids = valid_subject_ids
        return all_smpl

    def load_all_cameras(self, camera_path):
        # input path to camera.json under synbody sequence
        # all_cameras is dict of dict
        all_cameras = {}
        for people_id in self.subject_ids:
            current_camera_path = camera_path.format(people_id=people_id)
            current_camera = {}
            with open(current_camera_path) as f:
                camera = json.load(f)
            for view_index in range(len(camera.keys())):
                K, R, T, _ = get_KRTD(camera, view_index)
                current_camera['camera{:04d}'.format(view_index)] = {
                    "Rt": np.concatenate([R, T[..., None]], axis=1).astype(np.float32),
                    "K": K.astype(np.float32),
                }
            for c in current_camera.values():
                c["cam_pos"] = -np.dot(c["Rt"][:3, :3].T, c["Rt"][:3, 3])
                # c["Rt"][:, -1] *= 1000.0
            all_cameras[people_id] = current_camera
        return all_cameras

    def sample_smpl_param(self, n, device, sample_dict=None):
        sample_trans = []
        sample_beta = []
        sample_theta = []
        if sample_dict is not None:
            assert n == 1
            people_id = sample_dict["people_id"]
            frame = str(sample_dict["frame_id"])
            camera_id = sample_dict["camera_id"]
            smpl_param = load_smplx_params(self.all_smpl[people_id], int(frame))
            sample_trans.append(torch.from_numpy(smpl_param["Th"]))
            sample_beta.append(torch.from_numpy(smpl_param["shapes"]))
            sample_theta.append(torch.from_numpy(smpl_param["poses"]))
            return torch.stack(sample_trans, 0).to(device), \
                   torch.stack(sample_beta, 0).to(device), \
                   torch.stack(sample_theta, 0).to(device)

        for i in range(n):
            people_id = self.subject_ids[random.choice(range(len(self.subject_ids)))]
            frame = random.choice(self.frame_list)
            camera_id = random.choice(list(self.cameras[people_id].keys()))
            smpl_param = load_smplx_params(self.all_smpl[people_id], int(frame))
            
            sample_trans.append(torch.from_numpy(smpl_param["Th"]))
            sample_beta.append(torch.from_numpy(smpl_param["shapes"]))
            sample_theta.append(torch.from_numpy(smpl_param["poses"]))

        return torch.stack(sample_trans, 0).to(device), \
               torch.stack(sample_beta, 0).to(device), \
               torch.stack(sample_theta, 0).to(device)
    
    def get_camera_extrinsics(self, n, device, sample_dict=None):
        sample_ex = []
        sample_in = []
        if sample_dict is not None:
            assert n == 1
            people_id = sample_dict["people_id"]
            frame = str(sample_dict["frame_id"])
            camera_id = sample_dict["camera_id"]
            cam_param = self.cameras[people_id][camera_id]
            sample_ex.append(torch.from_numpy(cam_param["Rt"]))
            sample_in.append(torch.from_numpy(cam_param["K"]))
            return torch.stack(sample_ex, 0).to(device), \
                   torch.stack(sample_in, 0).to(device)

        for i in range(n):
            people_id = self.subject_ids[random.choice(range(len(self.subject_ids)))]
            frame = random.choice(self.frame_list)
            camera_id = random.choice(list(self.cameras[people_id].keys()))
            cam_param = self.cameras[people_id][camera_id]
            sample_ex.append(torch.from_numpy(cam_param["Rt"]))
            sample_in.append(torch.from_numpy(cam_param["K"]))

        return torch.stack(sample_ex, 0).to(device), \
               torch.stack(sample_in, 0).to(device)

    def get_gt_image(self, sample_dict):
        people_id = sample_dict["people_id"]
        frame = str(sample_dict["frame_id"])
        camera_id = sample_dict["camera_id"]
        fmts = dict(people_id=people_id, frame=int(frame), camera=camera_id)
        img = np.transpose(
            cv2.imread(self.image_path.format(**fmts))[..., ::-1],
            axes=(2, 0, 1),
        )
        image_part_mask = cv2.imread(self.image_part_mask_path.format(**fmts))
        border = 3
        kernel = np.ones((border, border), np.uint8)
        part_msk_erode = cv2.erode(image_part_mask.copy(), kernel)[np.newaxis, ..., 0]
        image_part_mask = part_msk_erode

        img = img * (image_part_mask != 0) + (~(image_part_mask != 0)) * 255
        img = Image.fromarray(np.transpose(img.astype(np.uint8), axes=(1, 2, 0))).convert("RGB")
        img = np.asarray(img.resize((self.image_height, self.image_height), Image.HAMMING))
        img = self.transform(img)

        return img

    def __len__(self):
        if self.opt.df.load_sampled_pose:
            return len(self.sampled_data)
        if self.phase == "evaluate":
            return len(self.subject_ids)
        
        return len(self.subject_ids) * 200

    def __getitem__(self, index):

        return self.__getitem__pack(index)

    def __getitem__pack(self, idx):
       
        if self.phase == "evaluate" or self.phase == "test":
            
            frame = idx % 20 #demo

            idx = idx % len(self.subject_ids)
            people_id = self.subject_ids[idx]
                        
            camera_id = frame #self.camera_index_eva[idx]
            subject_id_int = 0 #self.eva_abso_index[idx]
            camera_id  = 'camera%04d' % camera_id
        
        else:
            subject_id_int  = idx % len(self.subject_ids)
            people_id = self.subject_ids[subject_id_int]

            frame = (random.choice(self.frame_list))

            camera_id = (
                random.choice(list(self.cameras[people_id].keys()))
                if self.sample_cameras
                else self.camera_id
            )



        frame =  int(frame)

        fmts = dict(people_id=people_id, frame=int(frame), camera=camera_id)

        sample = {"index": idx, **fmts}

        sample.update(load_smplx_params(self.all_smpl[people_id], int(frame)))

        sample["image"] = np.transpose(
            cv2.imread(self.image_path.format(**fmts))[..., ::-1],
            axes=(2, 0, 1),
        )


        image_mask = cv2.imread(self.image_mask_path.format(**fmts))
        border = 3
        kernel = np.ones((border, border), np.uint8)
        msk_erode = cv2.erode(image_mask.copy(), kernel)[np.newaxis, ..., 0]
        sample["image_mask"] = (msk_erode != 0).astype(np.float32)

        image_part_mask = cv2.imread(self.image_part_mask_path.format(**fmts))
        part_msk_erode = cv2.erode(image_part_mask.copy(), kernel)[np.newaxis, ..., 0]
        sample["image_part_mask"] = part_msk_erode

        sample["image_bg"] = sample["image"] * ~(sample["image_part_mask"] != 0)

        sample.update(self.cameras[people_id][camera_id])

        img = sample['image'] * (sample["image_part_mask"] != 0) + (~(sample["image_part_mask"] != 0)) * 255
        img = Image.fromarray(np.transpose(img.astype(np.uint8), axes=(1, 2, 0))).convert("RGB")

        W, H = img.size
        ratio = self.image_height / H


        img = np.asarray(img.resize((self.image_height, self.image_height), Image.HAMMING))
        
        if self.image_height != self.image_width:
            gap = (self.image_height - self.image_width) // 2

        trans = sample["Th"]
        beta = sample["shapes"]
        theta = sample["poses"]
        intrinsics = sample["K"]
        extrinsics = sample["Rt"]

        img_name = f'{subject_id_int}_{people_id}_{camera_id}_{frame:04d}'
        meta = {
            "trans": trans,
            "cam_trans": 0,
            'beta': beta,
            'theta': theta,
            'extrinsics': extrinsics,
            'intrinsics': intrinsics,
            'image_name': img_name,
            #'org_img': (org_img.astype(np.float32) / 255) * 2 - 1.0,
            
            "frame_name": img_name,
            'ratio': ratio, #thumb_img.shape[0] / H,
            'org_img_size': (H, W),
            'crop_bbox': (int(gap), int(gap), int(0), int(0)),
            "cam_ind": camera_id,

            "subject_id_int_" : subject_id_int,
            "people_id_" : people_id,
            "frame_" : frame,
            "camera_id_" : camera_id,
        }

        if not self.opt.df.sample_random_pose:
            meta.update(
                {
                'img': (img.astype(np.float32) / 255) * 2 - 1.0,
                'msk': sample["image_part_mask"][0][...,None],

                'swap_id' : 0 if not self.opt.df.transfer_texture else self.swap_id_list[idx],
        
                "eva_index": self.eva_abso_index[idx] if self.phase == "evaluate" else -1,  
                "index": subject_id_int if self.opt.df.stage_1_fitting else idx,
            }
            )

        return meta
    
    def gen_inf_cameras(self, num_views = 5):
        training_views = self.cameras[self.subject_ids[0]]
        self.training_views = training_views
        num_training_views = len(training_views.keys())
        interpolation_anchors = []
        for view_index in range(num_training_views):
            Rt = training_views['camera{:04d}'.format(view_index)]['Rt']
            K = training_views['camera{:04d}'.format(view_index)]['K']
            rot = Rt[:, :3]
            trans = Rt[:, 3]
            interpolation_anchors.append((rot, trans))
        interpolated_poses = interpolate_poses(interpolation_anchors, num_views)

        inf_camera = {}
        for people_id in self.subject_ids:
            current_camera = {}
            for view_index in range(len(interpolated_poses)):
                R, T = interpolated_poses[view_index]
                current_camera['camera{:04d}'.format(view_index)] = {
                    "Rt": np.concatenate([R, T[..., None]], axis=1).astype(np.float32),
                    "K": K.astype(np.float32),
                }
            for c in current_camera.values():
                c["cam_pos"] = -np.dot(c["Rt"][:3, :3].T, c["Rt"][:3, 3])
                # c["Rt"][:, -1] *= 1000.0
            inf_camera[people_id] = current_camera
        self.inf_cameras = inf_camera


    def inf_sample(self, people_id, camera_id, frame_id, cond_sample):
        fmts = dict(people_id=people_id, frame=int(frame_id), camera=camera_id)
        sample = {}
        sample.update({**fmts})

        sample.update(load_smplx_params(self.all_smpl[people_id], int(frame_id)))

        sample.update(self.inf_cameras[people_id][camera_id])

        for k, v in sample.items():
            if isinstance(v, np.ndarray):
                sample[k] = v[None, ...]

        sample.update(cond_sample)
        return sample

    def cond_sample(self, people_id):
        sample = {}
        # reading all the cond images
        if self.cond_cameras:
            sample["cond_image"] = []
            sample["cond_Rt"] = []
            sample["cond_K"] = []
            # for cond_camera_id in self.cond_cameras:
            cond_camera_id = random.choice(list(self.cameras[people_id].keys()))
            if True:
                cond_image = np.transpose(
                    cv2.imread(
                        self.image_path.format(
                            people_id=people_id, frame=int(self.ref_frame), camera=cond_camera_id
                        )
                    )[..., ::-1].astype(np.float32),
                    axes=(2, 0, 1),
                )
                sample["cond_image"].append(cond_image)
                sample["cond_Rt"].append(self.cameras[people_id][cond_camera_id]["Rt"])
                sample["cond_K"].append(self.cameras[people_id][cond_camera_id]["K"])

            for key in ["image", "K", "Rt"]:
                sample[f"cond_{key}"] = np.stack(sample[f"cond_{key}"], axis=0)

            sample["cond_cameras"] = self.cond_cameras[:]
        for k, v in sample.items():
            if isinstance(v, np.ndarray):
                sample[k] = v[None, ...]
        return sample
    

    def inf_sample_wsmpl(self, people_id, camera_id, frame_id, cond_sample, smpl_param):
        fmts = dict(people_id=people_id, frame=int(frame_id), camera=camera_id)
        sample = {}
        sample.update({**fmts})

        sample.update(load_smplx_params(smpl_param, int(frame_id)))

        sample.update(self.inf_cameras[people_id][camera_id])

        for k, v in sample.items():
            if isinstance(v, np.ndarray):
                sample[k] = v[None, ...]

        sample.update(cond_sample)
        return sample