from genericpath import exists
import shutil
import numpy as np
import os
import pickle

import cv2
from collections import OrderedDict

from .io.prints import *
import torch

def mkdirs_right_main(path, exist_ok=True, mode=0o777):
    try:
        original_umask = os.umask(0)
        os.makedirs(path, mode=mode, exist_ok=True)
    finally:
        os.umask(original_umask)

def get_log_dir(opt):    
    subdirs = opt.checkpoints_dir.split("/")
    mode_name = subdirs[-1]
    if mode_name == "": mode_name = subdirs[-2]
    log_dir = os.path.join(get_log_base_dir(opt), "%s_%s" % (opt.name, mode_name)) 
    return log_dir

def get_log_base_dir(opt):
    base_dir = os.path.join(opt.log_dir, "logdirs/recent")
    return base_dir
