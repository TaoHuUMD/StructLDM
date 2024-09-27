import numpy as np
import torch

def get_can_pose_snapshot(init_pose_, is_dense = False):
    
    can_list=[]
    ds = [-1, 0, 1]
    if is_dense:
      step = 120
      ds = np.linspace(-1, 1, step)
    for arm_ele in ds:    
        new_poses = np.zeros((24, 3))
        init_pose = init_pose_.reshape(24, 3)

        new_poses[0] = init_pose[0]

        new_poses[13] = arm_ele * init_pose[13]
        new_poses[14] = arm_ele * init_pose[14]

        can_list.append(new_poses)
        
    return can_list

def map_normalized_dp_to_tex_pytorch(img, norm_iuv_img, tex_res, fillconst=0):

    device = img.device
    tex = torch.ones((tex_res, tex_res, img.shape[2])).to(device) * fillconst
    tex_mask = torch.zeros((tex_res, tex_res)).to(device)

    valid_iuv = norm_iuv_img[norm_iuv_img[:, :, 0] > 0]
    valid_iuv = valid_iuv.cpu().numpy()

    if valid_iuv.size==0:
        return tex, tex_mask

    #if valid_iuv[:, 2].max() > 1:
    #    valid_iuv[:, 2] /= 255.
    #    valid_iuv[:, 1] /= 255.

    u_I = np.round(valid_iuv[:, 0] * (tex.shape[1] - 1)).astype(np.int32)
    v_I = np.round((1 - valid_iuv[:, 1]) * (tex.shape[0] - 1)).astype(np.int32)

    data = img[norm_iuv_img[:, :, 0] > 0]

    tex[v_I, u_I] = data
    tex_mask[v_I, u_I] = 1

    return tex, tex_mask


def map_normalized_dp_to_tex(img, norm_iuv_img, tex_res, fillconst=128):
    tex = np.ones((tex_res, tex_res, img.shape[2])) * fillconst
    tex_mask = np.zeros((tex_res, tex_res)).astype(np.bool)

    # print('norm max, min', norm_iuv_img[:, :, 0].max(), norm_iuv_img[:, :, 0].min())
    valid_iuv = norm_iuv_img[norm_iuv_img[:, :, 0] > 0]

    if valid_iuv.size==0:
        return tex, tex_mask

    if valid_iuv[:, 2].max() > 1:
        valid_iuv[:, 2] /= 255.
        valid_iuv[:, 1] /= 255.

    u_I = np.round(valid_iuv[:, 1] * (tex.shape[1] - 1)).astype(np.int32)
    v_I = np.round((1 - valid_iuv[:, 2]) * (tex.shape[0] - 1)).astype(np.int32)

    data = img[norm_iuv_img[:, :, 0] > 0]

    tex[v_I, u_I] = data
    tex_mask[v_I, u_I] = 1

    return tex, tex_mask