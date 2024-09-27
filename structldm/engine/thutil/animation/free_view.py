import numpy as np
import os, sys
import torch
from icecream import ic
import cv2
import imageio


def normalize(x):
    return x / np.linalg.norm(x)


def viewmatrix(z, up, pos):
    vec2 = normalize(z)
    vec0_avg = up
    vec1 = normalize(np.cross(vec2, vec0_avg))
    vec0 = normalize(np.cross(vec1, vec2))
    m = np.stack([vec0, vec1, vec2, pos], 1)
    return m


def ptstocam(pts, c2w):
    tt = np.matmul(c2w[:3, :3].T, (pts - c2w[:3, 3])[..., np.newaxis])[..., 0]
    return tt


def gen_path(RT, center=None):
    lower_row = np.array([[0., 0., 0., 1.]])

    # transfer RT to camera_to_world matrix
    RT = np.array(RT)
    RT[:] = np.linalg.inv(RT[:])

    RT = np.concatenate([RT[:, :, 1:2], RT[:, :, 0:1],
                         -RT[:, :, 2:3], RT[:, :, 3:4]], 2)

    up = normalize(RT[:, :3, 0].sum(0))  # average up vector
    z = normalize(RT[0, :3, 2])
    vec1 = normalize(np.cross(z, up))
    vec2 = normalize(np.cross(up, vec1))
    z_off = 0

    if center is None:
        center = RT[:, :3, 3].mean(0)
        z_off = 1.3

    c2w = np.stack([up, vec1, vec2, center], 1)

    # get radii for spiral path
    tt = ptstocam(RT[:, :3, 3], c2w).T
    rads = np.percentile(np.abs(tt), 80, -1)
    rads = rads * 1.3
    rads = np.array(list(rads) + [1.])

    render_w2c = []
    for theta in np.linspace(0., 2 * np.pi, render_views + 1)[:-1]:
        # camera position
        cam_pos = np.array([0, np.sin(theta), np.cos(theta), 1] * rads)
        cam_pos_world = np.dot(c2w[:3, :4], cam_pos)
        # z axis
        z = normalize(cam_pos_world -
                      np.dot(c2w[:3, :4], np.array([z_off, 0, 0, 1.])))
        # vector -> 3x4 matrix (camera_to_world)
        mat = viewmatrix(z, up, cam_pos_world)

        mat = np.concatenate([mat[:, 1:2], mat[:, 0:1],
                              -mat[:, 2:3], mat[:, 3:4]], 1)
        mat = np.concatenate([mat, lower_row], 0)
        mat = np.linalg.inv(mat)
        render_w2c.append(mat)

    return render_w2c


trans_t = lambda t: torch.Tensor([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, t],
    [0, 0, 0, 1]]).float()

rot_phi = lambda phi: torch.Tensor([
    [1, 0, 0, 0],
    [0, np.cos(phi), -np.sin(phi), 0],
    [0, np.sin(phi), np.cos(phi), 0],
    [0, 0, 0, 1]]).float()

rot_theta = lambda th: torch.Tensor([
    [np.cos(th), 0, -np.sin(th), 0],
    [0, 1, 0, 0],
    [np.sin(th), 0, np.cos(th), 0],
    [0, 0, 0, 1]]).float()


def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi / 180. * np.pi) @ c2w
    c2w = rot_theta(theta / 180. * np.pi) @ c2w
    c2w = torch.Tensor(np.array([[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])) @ c2w
    return c2w


def genRenderViews():
    render_poses = torch.stack([pose_spherical(angle, 0.0, 2.5) for angle in np.linspace(-180, 180, 30 + 1)[:-1]], 0)


def angleAxis2Rot(ref_cam, angle):
    theta = angle / 180. * np.pi
    Axis = np.array([[0], [1], [0]])
    angle_axis = Axis * theta
    delta_pose = np.eye(4, dtype=np.float32)
    delta_pose[:3, :3] = cv2.Rodrigues(angle_axis)[0]
    # c2w = delta_pose @ ref_cam
    return delta_pose

def renderFvv2(ref_cam, angle, trans):
    theta = angle / 180. * np.pi
    Axis = np.array([[0], [1], [0]])
    angle_axis = Axis * theta
    delta_pose = np.eye(4, dtype=np.float32)
    delta_pose[:3, :3] = cv2.Rodrigues(angle_axis)[0]
    # pose_trans=trans_t(trans)
    pose_trans = np.array([[1, 0, 0, trans[0]],
                           [0, 1, 0, trans[1]],
                           [0, 0, 1, trans[2]],
                           [0, 0, 0, 1]])
    c2w = np.linalg.inv(pose_trans) @ ref_cam
    c2w = delta_pose @ c2w
    c2w = pose_trans @ c2w
    return c2w



def gen_path(RT, view_num = 120, center=None):
    lower_row = np.array([[0., 0., 0., 1.]])

    # transfer RT to camera_to_world matrix
    RT = np.array(RT)
    RT[:] = np.linalg.inv(RT[:])

    RT = np.concatenate([RT[:, :, 1:2], RT[:, :, 0:1],
                         -RT[:, :, 2:3], RT[:, :, 3:4]], 2)

    up = normalize(RT[:, :3, 0].sum(0))  # average up vector
    z = normalize(RT[0, :3, 2])
    vec1 = normalize(np.cross(z, up))
    vec2 = normalize(np.cross(up, vec1))
    z_off = 0

    if center is None:
        center = RT[:, :3, 3].mean(0)
        z_off = 1.3

    c2w = np.stack([up, vec1, vec2, center], 1)

    # get radii for spiral path
    tt = ptstocam(RT[:, :3, 3], c2w).T
    rads = np.percentile(np.abs(tt), 80, -1)
    rads = rads * 1.3
    rads = np.array(list(rads) + [1.])

    render_w2c = []
    for theta in np.linspace(0., 2 * np.pi, view_num + 1)[:-1]:
        # camera position
        cam_pos = np.array([0, np.sin(theta), np.cos(theta), 1] * rads)
        cam_pos_world = np.dot(c2w[:3, :4], cam_pos)
        # z axis
        z = normalize(cam_pos_world -
                      np.dot(c2w[:3, :4], np.array([z_off, 0, 0, 1.])))
        # vector -> 3x4 matrix (camera_to_world)
        mat = viewmatrix(z, up, cam_pos_world)

        mat = np.concatenate([mat[:, 1:2], mat[:, 0:1],
                              -mat[:, 2:3], mat[:, 3:4]], 1)
        mat = np.concatenate([mat, lower_row], 0)
        mat = np.linalg.inv(mat)
        render_w2c.append(mat)

    return render_w2c

