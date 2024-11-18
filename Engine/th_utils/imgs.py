import numpy as np
import torch
import cv2
import os
from .io.prints import *

def vis_img(img, bbox = None):
    if bbox is not None:
        rmin, cmin, rmax, cmax = bbox[0], bbox[2], bbox[1], bbox[3]
        cv2.rectangle(img,(cmin, rmin),(cmax, rmax),(0,255,0),3)
    cv2.imshow("img", img)
    cv2.waitKey(0)


def img2tensor(img, device):
    if torch.is_tensor(img):
        return img.to(device)
    t = torch.from_numpy(img).float()
    if t.ndim == 2:
        t = t[None][None]
    elif t.ndim == 3:
        t = t[None].permute(0,3,1,2)
    return t.to(device)


def tensor2img(t):
    if not torch.is_tensor(t):
        return t
    if t.ndim == 4:
        t = t[0].permute(2,0,1).cpu().numpy()
    elif t.ndim == 3:
        t = t.permute(2,0,1).cpu().numpy()
    else: t = t.cpu().numpy()
    return t


def backed_img(img, msk, bg):
    bg_img = np.ones_like(img) * (bg + 1) * 127.5
    if len(msk.shape) == 3: msk = msk[..., 0]
    msk = msk.astype(np.bool)
    bg_img[msk] = img[msk] 
    return bg_img



    printd(video_name)

    cap = cv2.VideoCapture(video_name)

    fps = cap.get(cv2.CAP_PROP_FPS)

    frame_num = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    printy("fps", fps, frame_num)

    from Engine.th_utils.files import mkdirs_right    
    mkdirs_right(out_dir, exist_ok=True)
    #os.makedirs(out_dir, exist_ok=True)
    printy(out_dir)

    num = total_extracted_frames

    for i in range(num):
        which_frame = int(start_frame + i)

        cap.set(1, which_frame)
        res, frame = cap.read()

        w, h = frame.shape[:-1]

        #my_video_name = os.path.join(out_dir, '%d.jpg' % (img_id + i))
        img_name = '%d.png' % (which_frame) if pre is None else "%s_%d.png" % (pre, which_frame)
        my_video_name = os.path.join(out_dir, img_name)
        cv2.imwrite(my_video_name, frame)

    return 0