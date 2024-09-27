import sys

import pickle
import cv2
import os
import numpy as np
import matplotlib.pylab as plt
from ..kutils import utils as ku
import torch

global_lookup = np.load('../asset/dp_uv_lookup_256.npy')

global_parts = {
    'GARMENTS': [1, 2,  # torso
                 5, 6,
                 7, 9,
                 8, 10,
                 11, 13,
                 12, 14,
                 15, 17,
                 16, 18,
                 20, 22,
                 19, 21],
    'BODY': [
        3, 4,
        23, 24
    ]

}


# im2tex
def uvTransformNormal(input_image, iuv_img, tex_res, fillconst=128, upsamplefactor=0, is_normalized_iuv=False):
    if upsamplefactor != 0:
        nshape = (input_image.shape[0] * upsamplefactor, input_image.shape[1] * upsamplefactor)
        input_image = cv2.resize(input_image, nshape)
        iuv_img_uv = cv2.resize(iuv_img, nshape)
        iuv_img_parts = cv2.resize(iuv_img, nshape, interpolation=cv2.INTER_NEAREST)[:, :, 0]
        iuv_img = np.concatenate((iuv_img_parts[:, :, np.newaxis], iuv_img_uv[:, :, 1:]), axis=2)

    if is_normalized_iuv:
        return map_normalized_dp_to_tex(input_image, iuv_img, tex_res, fillconst)
    else:
        return map_densepose_to_tex(input_image, iuv_img, tex_res, fillconst)


def uvTransformNormalPartial(input_image, iuv_img, tex_res, fillconst=128, upsamplefactor=0, parts='GARMENTS'):
    if upsamplefactor != 0:
        nshape = (input_image.shape[0] * upsamplefactor, input_image.shape[1] * upsamplefactor)
        input_image = cv2.resize(input_image, nshape)
        iuv_img_uv = cv2.resize(iuv_img, nshape)
        iuv_img_parts = cv2.resize(iuv_img, nshape, interpolation=cv2.INTER_NEAREST)[:, :, 0]
        iuv_img = np.concatenate((iuv_img_parts[:, :, np.newaxis], iuv_img_uv[:, :, 1:]), axis=2)

    return map_densepose_to_tex_partial(input_image, iuv_img, tex_res, fillconst, parts)


def map_densepose_to_tex_partial(img, iuv_img, tex_res, fillconst=128, parts='GARMENTS'):
    global global_lookup
    global global_parts

    partids = global_parts[parts]
    indices = np.zeros((iuv_img.shape[0], iuv_img.shape[1])).astype(np.bool)

    for pi in partids:
        indices = indices | (iuv_img[:, :, 0] == pi)

    iuv_raw = iuv_img[indices]
    data = img[indices]
    i = iuv_raw[:, 0] - 1

    # print(iuv_raw.dtype)
    if iuv_raw.dtype == np.uint8:
        u = iuv_raw[:, 1] / 255.
        v = iuv_raw[:, 2] / 255.
    else:
        u = iuv_raw[:, 1]
        v = iuv_raw[:, 2]

    u[u > 1] = 1.
    v[v > 1] = 1.

    uv_smpl = global_lookup[
        i.astype(np.int),
        np.round(v * 255.).astype(np.int),
        np.round(u * 255.).astype(np.int)
    ]

    tex = np.ones((tex_res, tex_res, img.shape[2])) * fillconst
    tex_mask = np.zeros((tex_res, tex_res)).astype(np.bool)

    u_I = np.round(uv_smpl[:, 0] * (tex.shape[1] - 1)).astype(np.int32)
    v_I = np.round((1 - uv_smpl[:, 1]) * (tex.shape[0] - 1)).astype(np.int32)

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


def map_densepose_to_tex(img, iuv_img, tex_res, fillconst=128):
    global global_lookup

    iuv_raw = iuv_img[iuv_img[:, :, 0] > 0]  # mask.
    data = img[iuv_img[:, :, 0] > 0]
    i = iuv_raw[:, 0] - 1

    # print(iuv_raw.dtype)
    if iuv_raw.dtype == np.uint8:
        u = iuv_raw[:, 1] / 255.
        v = iuv_raw[:, 2] / 255.
    else:
        u = iuv_raw[:, 1]
        v = iuv_raw[:, 2]

    u[u > 1] = 1.
    v[v > 1] = 1.

    uv_smpl = global_lookup[
        i.astype(np.int),
        np.round(v * 255.).astype(np.int),
        np.round(u * 255.).astype(np.int)
    ]

    tex = np.ones((tex_res, tex_res, img.shape[2])) * fillconst
    tex_mask = np.zeros((tex_res, tex_res)).astype(np.bool)

    u_I = np.round(uv_smpl[:, 0] * (tex.shape[1] - 1)).astype(np.int32)
    v_I = np.round((1 - uv_smpl[:, 1]) * (tex.shape[0] - 1)).astype(np.int32)

    tex[v_I, u_I] = data
    tex_mask[v_I, u_I] = 1

    return tex, tex_mask


def getDPImg(fim_size, dp2_result):
    dpimage = np.zeros(fim_size)

    if len(dp2_result['scores']) == 0:
        print('WARNING: NO DETECTION!')
        return None
    # dp result has bbox, iuv map
    iuv_arr = DensePoseResult.decode_png_data(*dp2_result['pred_densepose'].results[0])
    iuv_img = iuv_arr.transpose(1, 2, 0)

    # bb = [int(v) for v in dp_result['pred_boxes_XYXY'][0]]
    bb = [v for v in dp2_result['pred_boxes_XYXY'][0]]

    boxes_XYWH = BoxMode.convert(
        bb, BoxMode.XYXY_ABS, BoxMode.XYWH_ABS
    )
    x, y, w, h = [int(v) for v in boxes_XYWH]

    dpimage[y:y + h, x:x + w, ...] = iuv_img.copy()
    dpimage = dpimage.astype(np.uint)
    return dpimage


def showDPtex(dptex):
    fig = plt.figure()

    count = 0
    for i in range(4):
        for j in range(6):
            plt.subplot(4, 6, count + 1)
            plt.imshow(dptex[count, :, :])
            count += 1
    for ax in fig.axes:
        ax.axis("off")


# im2texDP

def uvTransformDPdet2(input_image, dp, tex_res):
    sys.path.extend(['/HPS/impl_deep_volume/static00/detectron2/projects/DensePose/',
                     '/HPS/impl_deep_volume/static00/detectron2/projects/DensePose/densepose',
                     '/HPS/impl_deep_volume/static00/tex2shape/lib'])

    from structures import DensePoseResult
    from maps import map_densepose_to_tex, normalize
    from detectron2.structures.boxes import BoxMode

    # dp result has bbox, iuv map
    iuv_arr = DensePoseResult.decode_png_data(*dp['pred_densepose'].results[0])
    iuv_img = iuv_arr.transpose(1, 2, 0)

    # bb = [int(v) for v in dp_result['pred_boxes_XYXY'][0]]
    bb = [v for v in dp['pred_boxes_XYXY'][0]]
    boxes_XYWH = BoxMode.convert(
        bb, BoxMode.XYXY_ABS, BoxMode.XYWH_ABS
    )
    x, y, w, h = [int(v) for v in boxes_XYWH]

    tightimg = input_image[y:y + h, x:x + w, ...]

    # plt.subplot(121)
    # plt.imshow(iuv_img)
    # plt.subplot(122)
    # plt.imshow(tightimg)
    # plt.show()

    tightimg = cv2.resize(tightimg, (iuv_img.shape[1], iuv_img.shape[0]))

    dptex = np.zeros((24, tex_res, tex_res, 3))

    iuv_raw = iuv_img[iuv_img[:, :, 0] > 0]
    data = tightimg[iuv_img[:, :, 0] > 0]
    i = iuv_raw[:, 0] - 1

    if iuv_raw.dtype == np.uint8:
        u = iuv_raw[:, 1] / 255.
        v = iuv_raw[:, 2] / 255.
    else:
        u = iuv_raw[:, 1]
        v = iuv_raw[:, 2]

    dptex[
        i.astype(np.int), np.round(u * (tex_res - 1)).astype(np.int), np.round(v * (tex_res - 1)).astype(np.int)] = data

    return dptex


def uvTransformDP(input_image, iuv_img, tex_res, fillconst=0):
    dptex = np.ones((24, tex_res, tex_res, 3)) * fillconst

    iuv_raw = iuv_img[iuv_img[:, :, 0] > 0]
    data = input_image[iuv_img[:, :, 0] > 0]
    i = iuv_raw[:, 0] - 1

    if iuv_raw.dtype == np.uint8:
        u = iuv_raw[:, 1] / 255.
        v = iuv_raw[:, 2] / 255.
    else:
        u = iuv_raw[:, 1]
        v = iuv_raw[:, 2]

    dptex[
        i.astype(np.int), np.round(u * (tex_res - 1)).astype(np.int), np.round(v * (tex_res - 1)).astype(np.int)] = data

    return dptex


def renderDP(dptex, iuv_image):
    rendered = np.zeros((iuv_image.shape[0], iuv_image.shape[1], dptex.shape[-1]))

    iuv_raw = iuv_image[iuv_image[:, :, 0] > 0]

    i = iuv_raw[:, 0] - 1

    if iuv_raw.dtype == np.uint8 or iuv_raw.max() > 1:
        u = iuv_raw[:, 1] / 255.
        v = iuv_raw[:, 2] / 255.
    else:
        u = iuv_raw[:, 1]
        v = iuv_raw[:, 2]

    rendered[iuv_image[:, :, 0] > 0] = dptex[
        i.astype(np.int), np.round(u * (dptex.shape[1] - 1)).astype(np.int), np.round(v * (dptex.shape[2] - 1)).astype(
            np.int)]

    return rendered


def getCombinedDP(dptex):
    psize = dptex.shape[1]

    r, c = 4, 6
    combinedtex = np.zeros((psize * r, psize * c, dptex.shape[-1]))

    count = 0
    for i in range(r):
        for j in range(c):
            combinedtex[i * psize:i * psize + psize, j * psize:j * psize + psize] = dptex[count]
            count += 1

    return combinedtex


def getDPTexList(combinedtex):
    r, c = 4, 6
    psize = int(combinedtex.shape[0] / 4)
    print(psize)
    dptexlist = np.zeros((24, psize, psize, combinedtex.shape[-1]))

    count = 0
    for i in range(r):
        for j in range(c):
            dptexlist[count] = combinedtex[i * psize:i * psize + psize, j * psize:j * psize + psize]
            count += 1
    return dptexlist


def getDPTexListTensor(combinedtex_b):
    r, c = 4, 6
    psize = int(combinedtex_b.shape[1] / r)

    dptexlist = torch.zeros(combinedtex_b.shape[0], 24, psize, psize, combinedtex_b.shape[-1])
    count = 0
    for i in range(r):
        for j in range(c):
            dptexlist[:, count] = combinedtex_b[:, i * psize:i * psize + psize, j * psize:j * psize + psize, :]
            count += 1
    return dptexlist


def getFaceBB(iuv_img_t, maxwidth=256):
    pad = 4
    facemask = (iuv_img_t[:, 0, :, :] == 23) | (iuv_img_t[:, 0, :, :] == 24)
    bbt = torch.zeros(iuv_img_t.shape[0], 4)
    for i in range(iuv_img_t.shape[0]):
        t = torch.where(facemask[i, ...] != 0)

        if len(t[0]) == 0:
            bbt[i] = torch.zeros(4)
            continue

        rect = torch.stack([t[1].min() - pad, t[1].max() + pad, t[0].min() - pad, t[0].max()])
        rect[[0, 2]] = torch.clamp(rect[[0, 2]], 0)
        rect[[1, 3]] = torch.clamp(rect[[1, 3]], 0, maxwidth)
        #     [torch.clamp(t[0].min() - pad, 0), torch.clamp(t[0].max() + pad, 0),  torch.clamp(t[1].min() + pad, 0, mw), torch.clamp(t[1].max() + pad, 0, mh)]
        # #rect = torch.stack([rect[1], rect[0], rect[3], rect[2]])
        # rect = torch.stack([rect[2], rect[3], rect[0], rect[1]])
        bbt[i] = rect
        # bbt[i] = torch.stack([t[0].min(), t[1].min(), t[0].max(), t[1].max()])

    return bbt.int()


def transferTexGarments(source_tex, target_tex):
    # transfer source garments to target
    # copy source body area to target texture
    pass


def convertToNormal(iuv_image):
    global global_lookup

    normal_flow = np.zeros((iuv_image.shape[0], iuv_image.shape[1], 2))

    mask = np.zeros((iuv_image.shape[0], iuv_image.shape[1], 1))

    indices = iuv_image[:, :, 0] > 0
    iuv_raw = iuv_image[indices]

    i = iuv_raw[:, 0] - 1

    mask[indices] = 1

    if iuv_raw.dtype == np.uint8 or iuv_raw.max() > 1:
        u = iuv_raw[:, 1] / 255.
        v = iuv_raw[:, 2] / 255.
    else:
        u = iuv_raw[:, 1]
        v = iuv_raw[:, 2]

    u[u > 1] = 1.
    v[v > 1] = 1.

    uv_smpl = global_lookup[
        i.astype(np.int),
        np.round(v * 255.).astype(np.int),
        np.round(u * 255.).astype(np.int)
    ]

    normal_flow[indices] = uv_smpl
    together = np.concatenate((mask, normal_flow), axis=2)

    return together


def renderDPNormal(normal_tex, normal_flow):
    rendered = np.zeros((normal_flow.shape[0], normal_flow.shape[1], normal_tex.shape[-1]))

    indices = normal_flow[:, :, 0] > 0
    iuv_raw = normal_flow[indices]

    if iuv_raw.dtype == np.uint8 or iuv_raw.max() > 1:
        u = iuv_raw[:, 1] / 255.
        v = iuv_raw[:, 2] / 255.
    else:
        u = iuv_raw[:, 1]
        v = iuv_raw[:, 2]

    u_I = np.round(u * (normal_tex.shape[1] - 1)).astype(np.int32)
    v_I = np.round((1 - v) * (normal_tex.shape[0] - 1)).astype(np.int32)

    # rendered[indices] = normal_tex[np.round(u*(normal_tex.shape[0]-1)).astype(np.int), np.round(v*(normal_tex.shape[1]-1)).astype(np.int)]
    rendered[indices] = normal_tex[v_I, u_I]

    return rendered

#
# def renderDPNormal(normal_tex, normal_flow):
#     rendered = np.zeros((iuv_image.shape[0], iuv_image.shape[1], dptex.shape[-1]))
#
#     indices = normal_flow[:, :, 0] > 0
#     iuv_raw = normal_flow[indices]
#
#     print(iuv_raw[:, 2])
#
#     print(iuv_raw.dtype)
#     if iuv_raw.dtype == np.uint8 or iuv_raw.max() > 1:
#         u = iuv_raw[:, 1] / 255.
#         v = iuv_raw[:, 2] / 255.
#     else:
#         u = iuv_raw[:, 1]
#         v = iuv_raw[:, 2]
#
#     print(iuv_raw[:, 2])
#
#     rendered[indices] = normal_tex[
#         np.round(u * (normal_tex.shape[0] - 1)).astype(np.int), np.round(v * (normal_tex.shape[1] - 1)).astype(np.int)]
#     print([np.round(v * (normal_tex.shape[1] - 1)).astype(np.int)])
#
#     return rendered
#
#
# normal_uv = convertToNormal(dp_image)
# print(normal_uv.dtype, normal_uv.max())
#
# # plt.imshow(normal_uv[:,:,2])
# # plt.show()
# # plt.imshow(teximage)
# # plt.show()
# rendered = renderDPNormal(teximage, normal_uv)
# plt.imshow(rendered)
# plt.show()