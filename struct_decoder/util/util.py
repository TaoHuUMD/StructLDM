from __future__ import print_function
import torch
import numpy as np
from PIL import Image
import numpy as np
import os
import cv2

fixed_mat = np.load('../asset/fixedmat_viz.npy')
# Converts a Tensor into a Numpy array
# |imtype|: the desired type of the converted numpy array
def tensor2im(image_tensor, imtype=np.uint8, normalize=True, is_list = False):
    if isinstance(image_tensor, list) or is_list:
        image_numpy = []
        for i in range(len(image_tensor)):
            image_numpy.append(tensor2im(image_tensor[i], imtype, normalize))
        return image_numpy
    image_numpy = image_tensor.cpu().float().numpy()
    if normalize:
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    else:
        image_numpy = np.transpose(image_numpy, (1, 2, 0)) * 255.0
    image_numpy = np.clip(image_numpy, 0, 255)
    if image_numpy.shape[2] == 1 or image_numpy.shape[2] > 3:
        #fixed_mat = np.random.random_sample((16, 3))

        image_numpy = image_numpy.sum(axis=-1)
        #image_numpy = np.matmul(image_numpy, fixed_mat)
    return image_numpy.astype(imtype)

def tensor2imProj(image_tensor, imtype=np.uint8, normalize=True, is_list = False):
    global fixed_mat
    if isinstance(image_tensor, list) or is_list:
        image_numpy = []
        for i in range(len(image_tensor)):
            image_numpy.append(tensor2imProj(image_tensor[i], imtype, normalize))
        return image_numpy
    image_numpy = image_tensor.cpu().float().numpy()

    #print(image_numpy.shape, fixed_mat.shape)

    image_numpy = np.transpose(image_numpy, (1, 2, 0))
    if image_numpy.shape[2] == 1 or image_numpy.shape[2] == 16:
        #image_numpy = image_numpy[:,:,0]
        image_numpy_proj = np.matmul(image_numpy, fixed_mat)

        max, min = image_numpy_proj.max(), image_numpy_proj.min()
        image_numpy_proj = (image_numpy_proj - min)/(max - min)*255
        image_numpy_proj = np.clip(image_numpy_proj, 0, 255)
        return image_numpy_proj.astype(imtype)
    elif image_numpy.shape[2] >3 :
        image_numpy = (image_numpy[:,:,3:6] + 1) / 2.0 * 255.0
        return image_numpy.astype(imtype)
    else:
        image_numpy = (image_numpy + 1) / 2.0 * 255.0
        return image_numpy.astype(imtype)

# Converts a one-hot tensor into a colorful label map
def tensor2label(label_tensor, n_label, imtype=np.uint8):
    if n_label == 0:
        return tensor2im(label_tensor, imtype)
    label_tensor = label_tensor.cpu().float()
    if label_tensor.size()[0] > 1:
        label_tensor = label_tensor.max(0, keepdim=True)[1]
    label_tensor = Colorize(n_label)(label_tensor)
    label_numpy = np.transpose(label_tensor.numpy(), (1, 2, 0))
    return label_numpy.astype(imtype)


def save_image(image_numpy, image_path):

    if not (image_numpy.size==256*256 or image_numpy.size==512*512 or image_numpy.size==1024*1024):
        cv2.imwrite(image_path, image_numpy[:,:,[2,1,0]])
    else:
        cv2.imwrite(image_path, image_numpy*255)
    return 0
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)

def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)

def mkdir(path):
    os.makedirs(path, exist_ok = True)
    
    # if not os.path.exists(path):
    #     os.makedirs(path)

###############################################################################
# Code from
# https://github.com/ycszen/pytorch-seg/blob/master/transform.py
# Modified so it complies with the Citscape label map colors
###############################################################################
def uint82bin(n, count=8):
    """returns the binary of integer n, count refers to amount of bits"""
    return ''.join([str((n >> y) & 1) for y in range(count-1, -1, -1)])

def labelcolormap(N):
    if N == 35: # cityscape
        cmap = np.array([(  0,  0,  0), (  0,  0,  0), (  0,  0,  0), (  0,  0,  0), (  0,  0,  0), (111, 74,  0), ( 81,  0, 81),
                     (128, 64,128), (244, 35,232), (250,170,160), (230,150,140), ( 70, 70, 70), (102,102,156), (190,153,153),
                     (180,165,180), (150,100,100), (150,120, 90), (153,153,153), (153,153,153), (250,170, 30), (220,220,  0),
                     (107,142, 35), (152,251,152), ( 70,130,180), (220, 20, 60), (255,  0,  0), (  0,  0,142), (  0,  0, 70),
                     (  0, 60,100), (  0,  0, 90), (  0,  0,110), (  0, 80,100), (  0,  0,230), (119, 11, 32), (  0,  0,142)],
                     dtype=np.uint8)
    else:
        cmap = np.zeros((N, 3), dtype=np.uint8)
        for i in range(N):
            r, g, b = 0, 0, 0
            id = i
            for j in range(7):
                str_id = uint82bin(id)
                r = r ^ (np.uint8(str_id[-1]) << (7-j))
                g = g ^ (np.uint8(str_id[-2]) << (7-j))
                b = b ^ (np.uint8(str_id[-3]) << (7-j))
                id = id >> 3
            cmap[i, 0] = r
            cmap[i, 1] = g
            cmap[i, 2] = b
    return cmap

class Colorize(object):
    def __init__(self, n=35):
        self.cmap = labelcolormap(n)
        self.cmap = torch.from_numpy(self.cmap[:n])

    def __call__(self, gray_image):
        size = gray_image.size()
        color_image = torch.ByteTensor(3, size[1], size[2]).fill_(0)

        for label in range(0, len(self.cmap)):
            mask = (label == gray_image[0]).cpu()
            color_image[0][mask] = self.cmap[label][0]
            color_image[1][mask] = self.cmap[label][1]
            color_image[2][mask] = self.cmap[label][2]

        return color_image


def plot2Fig(mainfigure):
    mainfigure.canvas.draw()
    # data = np.frombuffer(mainfigure.canvas.tostring_rgb(), dtype=np.uint8)
    data = np.fromstring(mainfigure.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    w, h = mainfigure.canvas.get_width_height()[::-1]
    data = data.reshape(mainfigure.canvas.get_width_height()[::-1] + (3,))
    # data2 = data[:, :h//nc*(nc-1),:]

    return data

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def showDPtexPlt(dptex):
    figure = plt.figure()

    count = 0
    for i in range(4):
        for j in range(6):
            ax = plt.subplot(4, 6, count + 1)
            fig = ax.imshow(dptex[count, :, :].sum(axis = -1))
            fig.axes.get_xaxis().set_visible(False)
            fig.axes.get_yaxis().set_visible(False)
            plt.axis('off')

            count += 1
    figure.subplots_adjust(wspace=0, hspace=0)
    return plot2Fig(figure)

def showDPtex(dptex):
    psize = dptex.shape[1]
    combinedtex = np.zeros((psize*4, psize*6))
    count = 0
    for i in range(4):
        for j in range(6):
            combinedtex[i*psize:i*psize+psize, j*psize:j*psize+psize] = dptex[count].sum(axis = -1)
            count += 1

    combinedtex = np.repeat(combinedtex[..., np.newaxis], 3, axis = -1)
    combinedtex = (combinedtex - combinedtex.min()) / (combinedtex.max() - combinedtex.min())*255
    return  combinedtex.astype(np.uint8)

def padSqH(img, size, pad = 0):

    if size==img.shape[0]:
        return img

    ratio = size / img.shape[0]
    h = size
    w = int(img.shape[1] * ratio)

    frame = np.ones((size, size, 3)).astype(np.uint8) * pad
    p = int((h - w) / 2)
    img = cv2.resize(img, (w, h))
    frame[:, p:p + w] = img
    return frame

def save_img_list(img_list, path, img_name, rnum = 2):
    img = np.concatenate(img_list, 1)
    w = img.shape[1] // rnum
    h = img.shape[0]
    img = img.reshape(-1, w, 3)
    cv2.imwrite(os.path.join(path, f'{img_name}.png'), img)


def split_dict_batch(d, n=1):
    keys = list(d.keys())
    batch_size = len(d[keys[0]])
    if batch_size==1: 
        yield d
        return
    #for i in range(0, len(keys), n):
    for i in range(0, batch_size, n):
        yield {k: d[k][i:i+n] for k in keys}
