
import numpy as np
import os
import pickle

import imageio
import cv2
#from uvm_lib.engine.thutil.networks.net_util import get_epoch

def save_dict(d, p):
    #print(p, os.path.dirname(p))
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, 'wb') as f:
        pickle.dump(d, f, pickle.HIGHEST_PROTOCOL)
    print("keys saved ", len(d.keys()))

def load_dict(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def get_test_epoch(opt):
    model_dir = os.path.join(opt.checkpoints_dir, opt.name)    
    return get_epoch(model_dir, opt.which_epoch)

def get_epoch(model_dir, epoch= ""):
    
    if not os.path.exists(model_dir):
        return -1

    tgt_model_name = 'latest'
    pth_file = "%s.pth" % tgt_model_name
    
    pths = [
        int(pth.split('.')[0]) for pth in os.listdir(model_dir)
        if pth != 'latest.pth' and pth.find('.pth')!=-1 and check_int(pth.split('.')[0])
    ]
    if len(pths) == 0 and pth_file not in os.listdir(model_dir):
        return -1

    if epoch == "" or epoch == "-1":
        if pth_file in os.listdir(model_dir):
            pth = tgt_model_name
        else:
            pth = max(pths)
    else:
        pth = epoch
    if not os.path.isfile(os.path.join(model_dir, '{}.pth'.format(pth))):
        return -1
    print('load model: {}'.format(os.path.join(model_dir, '{}.pth'.format(pth))))
    pretrained_model = torch.load(
        os.path.join(model_dir, '{}.pth'.format(pth)), 'cuda')
                
    return pretrained_model['epoch']
    