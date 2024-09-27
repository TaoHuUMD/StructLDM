import cv2
import numpy as np
import os
import ntpath
import time
from . import util
from . import html
import scipy.misc
from ..kutils import utils as ku
try:
    from StringIO import StringIO  # Python 2.7
except ImportError:
    from io import BytesIO         # Python 3.x

from tensorboardX import SummaryWriter

from uvm_lib.engine.thutil.dirs import get_log_dir

class Visualizer():
    def __init__(self, opt):
        # self.opt = opt
        #self.tf_log = opt.tf_log        
        self.use_html = opt.isTrain and not opt.no_html
        #self.win_size = opt.display_winsize
        self.win_size = 3000
        self.name = opt.name
        self.tf_log = True

        if "checkpoints_dir" in opt.keys():
            checkpoints_dir = opt.checkpoints_dir
        else:
            checkpoints_dir = opt.training.checkpoints_dir

        #self.tmp_train_webpage = html.HTML(self.web_dir, 'Experiment name = %s' % self.name, refresh=30, pre="tmp_train")
        #self.tmp_eval_webpage = html.HTML(self.web_dir, 'Experiment name = %s' % self.name, refresh=30, pre="tmp_eva")

        if (opt.training.distributed and opt.local_rank == 0) or not opt.training.distributed:
            if self.tf_log:
               
                self.log_dir = get_log_dir(opt)
                os.makedirs(self.log_dir, exist_ok=True)

               
                self.writer = SummaryWriter(log_dir=self.log_dir)

            if self.use_html:
                self.web_dir = os.path.join(checkpoints_dir, self.name, 'web')
                self.train_dir = os.path.join(self.web_dir, 'train_images')
                self.eval_dir = os.path.join(self.web_dir, 'eva_images')
                print('create web directory %s...' % self.web_dir)
                for d in [self.web_dir, self.train_dir, self.eval_dir]:
                    os.makedirs(d, exist_ok=True)

            self.log_name = os.path.join(checkpoints_dir, self.name, 'loss_log.txt')

            os.system("chmod 707 -R %s" % os.path.join(checkpoints_dir, self.name))


            with open(self.log_name, "a") as log_file:
                now = time.strftime("%c")
                log_file.write('================ Training Loss (%s) ================\n' % now)

            self.train_webpage = html.HTML(self.web_dir, 'Experiment name = %s' % self.name, refresh=30, pre="train")
            self.eval_webpage = html.HTML(self.web_dir, 'Experiment name = %s' % self.name, refresh=30, pre="eva")

        
          
    # |visuals|: dictionary of images to display or save
    def display_current_results(self, visuals, epoch, flag=""):
        if self.tf_log: # show images in tensorboard output
            img_summaries = []
            for label, image_numpy in visuals.items():
                # Write the image to a string
                if image_numpy is None: 
                    continue       

        import torch
        if torch.is_tensor(flag): flag = str(flag.item())
  
        save_train = False
        save_eval = False
        img_dir = self.train_dir
        if flag.find("eva")!=-1:
            img_dir = self.eval_dir
            save_eval = True
        else: save_train = True

        if self.use_html: # save images to a html file
            for label, image_numpy in visuals.items():
                if image_numpy is None: 
                    #print(label, "is none" ) 
                    continue
                if isinstance(image_numpy, list):
                    for i in range(len(image_numpy)):
                        img_path = os.path.join(img_dir, 'epoch%.3d_%s_%s_%d.jpg' % (epoch, flag, label, i))
                        util.save_image(image_numpy[i], img_path)
                else:
                    #print(label, type(image_numpy))
                    img_path = os.path.join(img_dir, 'epoch%.3d_%s_%s.jpg' % (epoch, flag, label))
                    util.save_image(image_numpy, img_path)
                

            pre = "eva" if save_eval else "train"
            
            if save_eval:
                self.save_images(self.eval_webpage, visuals, 'epoch%.3d_%s' % (epoch, flag))
                self.eval_webpage.save()
                
            else: 
                self.save_images(self.train_webpage, visuals, 'epoch%.3d_%s' % (epoch, flag))
                self.train_webpage.save()

    def save_training(self):
        self.train_webpage.save()
        self.eval_webpage.save()

    def plot_current_errors(self, errors, step):
        if self.tf_log:
            for tag, value in errors.items():
                self.writer.add_scalar(tag,value,step)

    # errors: same format as |errors| of plotCurrentErrors
    def print_current_errors(self, epoch, i, errors, t):
        if isinstance(t, list):
            message = '(epoch: %d, iters: %d, timed: %.3f, timef: %.3f) ' % (epoch, i, t[0], t[1])
        else:
            message = '(epoch: %d, iters: %d, time: %.3f) ' % (epoch, i, t)
        for k, v in errors.items():
            if v != 0:
                message += '%s: %.3f ' % (k, v)

        print(message)
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)

    # save image to the disk
    def save_images_old(self, webpage, visuals, image_paths):
        image_path1 = image_paths[0]
        image_path2 = image_paths[1]

        image_dir = webpage.get_image_dir()
        short_path = ntpath.basename(image_path1[0])
        name1 = os.path.splitext(short_path)[0]

        image_dir = webpage.get_image_dir()
        short_path = ntpath.basename(image_path2[0])
        name2 = os.path.splitext(short_path)[0]

        name = name1 + '__' + name2

        webpage.add_header(name)
        ims = []
        txts = []
        links = []

        for label, image_numpy in visuals.items():
            image_name = '%s_%s.jpg' % (name, label)
            save_path = os.path.join(image_dir, image_name)
            util.save_image(image_numpy, save_path)

            ims.append(image_name)
            txts.append(label)
            links.append(image_name)
        webpage.add_images(ims, txts, links, width=self.win_size)

    def save_images(self, webpage, visuals, frame_idx, phase="train"):
        
        image_dir = webpage.get_image_dir()
        
        name = "%s" % frame_idx

        webpage.add_header(name)
        ims = []
        txts = []
        links = []

        for label, image_numpy in visuals.items():
            if image_numpy is None: continue            
            image_name = "%s_%s.jpg" % (name, label)
            if phase=="test": 
                image_name = "%s_%s.png" % (name, label)
            save_path = os.path.join(image_dir, image_name)
            util.save_image(image_numpy, save_path)

            ims.append(image_name)
            txts.append(label)
            links.append(image_name)
        webpage.add_images(ims, txts, links, width=self.win_size)

    def save_images_distinct(self, webpage, visuals, image_paths):


        image_path1 = image_paths[0]
        image_path2 = image_paths[1]

        image_dir = webpage.get_image_dir()
        short_path = ntpath.basename(image_path1[0])
        name1 = os.path.splitext(short_path)[0]

        short_path = ntpath.basename(image_path2[0])
        name2 = os.path.splitext(short_path)[0]
        name = name1 + '__' + name2

        image_dir_fake = ku.getSplitDir(image_dir)[0] + '/' + name1
        ku.createDirectory(image_dir_fake)
        image_dir_target = ku.getSplitDir(image_dir)[0] + '/' + name2[:11] + '_target'
        ku.createDirectory(image_dir_target)

        webpage.add_header(name)
        ims = []
        txts = []
        links = []

        for label, image_numpy in visuals.items():
            image_name = '%s_%s.jpg' % (name, label)
            if label == 'fake_image':
                save_path = os.path.join(image_dir_fake, image_name)
            elif label == 'real_image_t':
                save_path = os.path.join(image_dir_target, image_name)
            else:
                save_path = os.path.join(image_dir, image_name)

            util.save_image(image_numpy, save_path)

            ims.append(image_name)
            txts.append(label)
            links.append(save_path)
        webpage.add_images(ims, txts, links, width=self.win_size)
