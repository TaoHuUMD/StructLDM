import cv2
import numpy as np
import os
from . import util
try:
    from StringIO import StringIO  # Python 2.7
except ImportError:
    from io import BytesIO         # Python 3.x

from tensorboardX import SummaryWriter

import imageio
import torch

class Visualizer():
    def __init__(self, opt):

        self.use_html = True #not (opt.phase == "test")
        self.win_size = 3000
        self.name = opt.name
        self.tf_log = True

        if opt.phase == "test": return

    def save_eval_images(self, img_list, epoch, flag=""):
        if torch.is_tensor(flag): flag = str(flag.item())
        self.save_img_list(img_list, self.eval_dir, 'epoch%s_%s' % (epoch, flag))

    # |visuals|: dictionary of images to display or save
    def display_current_results(self, visuals, epoch, flag=""):
        if self.tf_log: # show images in tensorboard output
            img_summaries = []
            for label, image_numpy in visuals.items():
                # Write the image to a string
                if image_numpy is None: 
                    continue

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
                    continue
                if isinstance(image_numpy, list):
                    for i in range(len(image_numpy)):
                        img_path = os.path.join(img_dir, 'epoch%.3d_%s_%s_%d.jpg' % (epoch, flag, label, i))
                        util.save_image(image_numpy[i], img_path)
                else:
                    img_path = os.path.join(img_dir, 'epoch%.3d_%s_%s.jpg' % (epoch, flag, label))
                    util.save_image(image_numpy, img_path)
                

            # update website
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

    # errors: dictionary of error labels and values
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

    def save_img_list(self, img_list_, path, img_name, rnum = 2):
        img_list = []
        for l in img_list_:
            if isinstance(l, list): 
                img_list += l
            else: img_list.append(l)
        
        img = np.concatenate(img_list, 1)
        w = img.shape[1] // rnum
        img = np.concatenate([img[:, :w, :], img[:, w:, :]], 0)
        cv2.imwrite(os.path.join(path, f'{img_name}.png'), img[:,:,[2,1,0]])

    def img_to_video(self, webpage, video_name):
        image_dir = webpage.get_image_dir()
        video_dir = os.path.join(image_dir, "../")

        vfs = sorted(os.listdir(image_dir))

        video_pre = "%s_%s" % (video_name.split("_")[0], video_name.split("_")[1])

        video_list = []
        for f in vfs:
            if f.startswith(video_pre):
                video_list.append(os.path.join(image_dir, f))

        def img_to_video(file_list, video_dir, fps, is_img=True):
    
            writer = imageio.get_writer(video_dir, fps=fps)

            for file_name in file_list:        
                if is_img:
                    img = imageio.imread(file_name)
                else:
                    img = imageio.imread(file_name).astype(np.float32)

                writer.append_data(img)

            writer.close()

        img_to_video(video_list, os.path.join(video_dir, video_name), 25)        
        print("%s video saved" % video_name)

    def save_images_list(self, webpage, visuals, frame_idx, phase="train"):
        
        image_dir = webpage.get_image_dir()
        
        name = "%s" % frame_idx

        webpage.add_header(name)
        ims = []
        txts = []
        links = []
 
        image_numpy = visuals

        image_name = "%s.jpg" % (name)
        if phase=="test":
            image_name = "%s.png" % (name)

        save_path = os.path.join(image_dir, image_name)
        util.save_image(image_numpy, save_path)

        ims.append(image_name)
        txts.append(name)
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
            #print(label, image_numpy.shape)
            save_path = os.path.join(image_dir, image_name)
            util.save_image(image_numpy, save_path)

            ims.append(image_name)
            txts.append(label)
            links.append(image_name)
        webpage.add_images(ims, txts, links, width=self.win_size)
