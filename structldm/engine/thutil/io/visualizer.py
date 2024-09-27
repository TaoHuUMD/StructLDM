import cv2
import numpy as np
import os
import ntpath
import time
from .. import util
from .. import html
import scipy.misc
try:
    from StringIO import StringIO  # Python 2.7
except ImportError:
    from io import BytesIO         # Python 3.x


from ..dirs import get_log_dir, mkdirs_right_main, tmp_save

class Visualizer():
    def __init__(self, dir, win_size = 3000):
        # self.opt = opt
        #self.tf_log = opt.tf_log        
        #self.use_html = opt.isTrain and not opt.no_html
        self.use_html = True #not (opt.phase == "test")
        self.win_size = win_size

        if self.use_html:
            self.web_dir = os.path.join(dir, 'web')            
            print('create web directory %s...' % self.web_dir)
            for d in [self.web_dir]:
                mkdirs_right_main(d, exist_ok=True)

        self.webpage = html.HTML(self.web_dir, '')
   
    def save_images(self, img, image_name, text="train"):
        
        image_dir = self.webpage.get_image_dir()

        self.webpage.add_header(image_name)
        ims = []
        txts = []
        links = []

        #save_path = os.path.join(image_dir, image_name)
        #util.save_image(img, save_path)
        tmp_save(img, image_name, "", image_dir)

        ims.append(image_name)
        txts.append(text)
        links.append(image_name)
        self.webpage.add_images(ims, txts, links, width=self.win_size)

    def save(self):
        self.webpage.save()
