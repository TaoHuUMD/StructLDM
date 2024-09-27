import torch
import numpy as np
import math

import engine.thutil.grid_sample_fix as grid_sample_fix

def rotationX(angle):
    return [
        [1, 0, 0],
        [0, math.cos(angle), -math.sin(angle)],
        [0, math.sin(angle), math.cos(angle)],
    ]


def rotationY(angle):
    return [
        [math.cos(angle), 0, math.sin(angle)],
        [0, 1, 0],
        [-math.sin(angle), 0, math.cos(angle)],
    ]


def rotationZ(angle):
    return [
        [math.cos(angle), -math.sin(angle), 0],
        [math.sin(angle), math.cos(angle), 0],
        [0, 0, 1],
    ]

def batch_cross_3d(a, b):
    c = torch.zeros(a.shape[0], 3)
    c[:, 0], c[:, 1], c[:, 2] = a[:, 1]*b[:, 2]-a[:, 2]*b[:, 1], b[:, 0]*a[:, 2]-a[:, 0]*b[:, 2], a[:, 0]*b[:, 1]-b[:, 0]*a[:, 1]
    return c

def cross_3d(a, b):
    return np.array([a[1]*b[2]-a[2]*b[1], b[0]*a[2]-a[0]*b[2], a[0]*b[1]-b[0]*a[1]])

def index(feat, uv, size=None):
    '''
    :param feat: [B, C, H, W] image features
    :param uv: [B, 2, N] uv coordinates in the image plane, range [-1, 1]
    :return: [B, C, N] image features at the uv coordinates
    '''
    uv = uv.transpose(1, 2)  # [B, N, 2]
    uv = uv.unsqueeze(2)  # [B, N, 1, 2]
    if size != None:
        uv = (uv - size / 2) / (size / 2)
    # NOTE: for newer PyTorch, it seems that training results are degraded due to implementation diff in grid_sample_fix.grid_sample
    # for old versions, simply remove the aligned_corners argument.
    samples = grid_sample_fix.grid_sample(feat, uv, align_corners=True)  # [B, C, N, 1]
    return samples[:, :, :, 0]  # [B, C, N]


def orthogonal(points, calibrations, transforms=None):
    '''
    Compute the orthogonal projections of 3D points into the image plane by given projection matrix
    :param points: [B, 3, N] Tensor of 3D points
    :param calibrations: [B, 4, 4] Tensor of projection matrix
    :param transforms: [B, 2, 3] Tensor of image transform matrix
    :return: xyz: [B, 3, N] Tensor of xyz coordinates in the image plane
    '''
    rot = calibrations[:, :3, :3]
    trans = calibrations[:, :3, 3:4]
    pts = torch.baddbmm(trans, rot, points)  # [B, 3, N]
    if transforms is not None:
        scale = transforms[:2, :2]
        shift = transforms[:2, 2:3]
        pts[:, :2, :] = torch.baddbmm(shift, scale, pts[:, :2, :])
    return pts


def perspective(points, calibrations, transforms=None):
    '''
    Compute the perspective projections of 3D points into the image plane by given projection matrix
    :param points: [Bx3xN] Tensor of 3D points
    :param calibrations: [Bx4x4] Tensor of projection matrix
    :param transforms: [Bx2x3] Tensor of image transform matrix
    :return: xy: [Bx2xN] Tensor of xy coordinates in the image plane
    '''
    B, _, N = points.shape
    device = points.device
    
    points = torch.cat([points, torch.ones((B, 1, N), device=device)], dim=1)

    #print(calibrations.shape, points.shape, calibrations.dtype, points.dtype)

    points = calibrations @ points
    points[:, :2, :] /= points[:, 2:, :]
    return points

def rotationMatrixToAngles(R):
    """
    R : (bs, 3, 3)
    """
    # print(R.shape)
    sy = torch.sqrt(R[:, 0, 0] * R[:, 0, 0] + R[:, 1, 0] * R[:, 1, 0])
    singular = sy < 1e-6
    mask = ~singular
    x = torch.zeros(R.shape[0])
    y = torch.zeros(R.shape[0])
    z = torch.zeros(R.shape[0])
    if torch.sum(mask):
        x[mask] = torch.atan2(R[mask, 2, 1], R[mask, 2, 2])
        y[mask] = torch.atan2(-R[mask, 2, 0], sy[mask])
        z[mask] = torch.atan2(R[mask, 1, 0], R[mask, 0, 0])
    if torch.sum(singular):
        x[singular] = math.atan2(-R[singular, 1, 2], R[singular, 1, 1])
        y[singular] = torch.atan2(-R[singular, 2, 0], sy[singular])
        z[singular] = 0
    return torch.cat([x.unsqueeze(1), y.unsqueeze(1), z.unsqueeze(1)], dim=1)# np.array([x, y, z])