import cv2
import numpy as np
import os
import ntpath
import time
from . import util
from . import html
import scipy.misc
try:
    from StringIO import StringIO  # Python 2.7
except ImportError:
    from io import BytesIO         # Python 3.x

from tensorboardX import SummaryWriter

from .dirs import get_log_dir, mkdirs_right
import torch
from Engine3D.th_utils.files import img_to_video

class Visualizer():
    def __init__(self, opt):
        # self.opt = opt
        #self.tf_log = opt.tf_log        
        #self.use_html = opt.isTrain and not opt.no_html
        self.use_html = True #not (opt.phase == "test")
        #self.win_size = opt.display_winsize
        self.win_size = 3000
        self.name = opt.name
        self.tf_log = True

        if opt.phase == "test": return

        checkpoints_dir = opt.training.checkpoints_dir
        mkdirs_right(checkpoints_dir, exist_ok=True)

        #self.tmp_train_webpage = html.HTML(self.web_dir, 'Experiment name = %s' % self.name, refresh=30, pre="tmp_train")
        #self.tmp_eval_webpage = html.HTML(self.web_dir, 'Experiment name = %s' % self.name, refresh=30, pre="tmp_eva")

        if (opt.training.distributed and opt.local_rank == 0) or not opt.training.distributed:
            if self.tf_log:
                             
                self.log_dir = get_log_dir(opt, checkpoints_dir)
                mkdirs_right(self.log_dir, exist_ok=True)
             
                self.writer = SummaryWriter(log_dir=self.log_dir)

            if self.use_html:
                self.web_dir = os.path.join(checkpoints_dir, 'web')
                self.train_dir = os.path.join(self.web_dir, 'train_images')
                self.eval_dir = os.path.join(self.web_dir, 'eva_images')
                self.mesh_dir = os.path.join(self.web_dir, 'eva_meshes')
                print('create web directory %s...' % self.web_dir)
                for d in [self.web_dir, self.train_dir, self.eval_dir, self.mesh_dir]:
                    mkdirs_right(d, exist_ok=True)

            self.log_name = os.path.join(checkpoints_dir, 'loss_log.txt')

            os.system("chmod 707 -R %s" % checkpoints_dir)


            with open(self.log_name, "a") as log_file:
                now = time.strftime("%c")
                log_file.write('================ Training Loss (%s) ================\n' % now)

            self.train_webpage = html.HTML(self.web_dir, 'Experiment name = %s' % self.name, refresh=30, pre="train")
            self.eval_webpage = html.HTML(self.web_dir, 'Experiment name = %s' % self.name, refresh=30, pre="eva")

    def save_mesh(self, mesh_list, epoch, img_name_list, mesh_dir = None):
        
        if mesh_dir is not None:
            mdir = mesh_dir
        if hasattr(self, "mesh_dir"):
            mdir = os.path.join(self.mesh_dir, "%s" % epoch)
            
        mkdirs_right(mdir, exist_ok=True)

        import trimesh
        from lib.models.utils import extract_mesh_with_marching_cubes

        for m, n in zip(mesh_list, img_name_list):
            marching_cubes_mesh, _, _ = extract_mesh_with_marching_cubes(m, level_set=0)
            marching_cubes_mesh = trimesh.smoothing.filter_humphrey(marching_cubes_mesh, beta=0.2, iterations=5)            
            marching_cubes_mesh_filename = os.path.join(mdir, f'{n}.obj')
            with open(marching_cubes_mesh_filename, 'w') as f:
                marching_cubes_mesh.export(f,file_type='obj')

    def save_eval_images(self, img_list, epoch, flag=""):
        if torch.is_tensor(flag): flag = str(flag.item())
        #print(self.eval_dir, 'epoch%.3d_%s' % (epoch, flag))
        self.save_img_list(img_list, self.eval_dir, 'epoch%s_%s' % (epoch, flag))

    # |visuals|: dictionary of images to display or save
    def display_current_results(self, visuals, epoch, flag=""):
        if self.tf_log: # show images in tensorboard output
            img_summaries = []
            for label, image_numpy in visuals.items():
                # Write the image to a string
                if image_numpy is None: 
                    #print(label, "is none" ) 
                    continue
                #try:
                #    s = StringIO()
                #except:
                #    s = BytesIO()
                #scipy.misc.toimage(image_numpy).save(s, format="jpeg")
                #cv2.imwrite(s, image_numpy )

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
                

            # update website
            pre = "eva" if save_eval else "train"
            #webpage = html.HTML(self.web_dir, 'Experiment name = %s' % self.name, refresh=30, pre=pre)
            
            if save_eval:
                self.save_images(self.eval_webpage, visuals, 'epoch%.3d_%s' % (epoch, flag))
                self.eval_webpage.save()
                #self.eval_webpage.save_tmp(idx=1)
                
            else: 
                self.save_images(self.train_webpage, visuals, 'epoch%.3d_%s' % (epoch, flag))
                self.train_webpage.save()
                #self.train_webpage.save_tmp(idx=1)

    def save_training(self):
        self.train_webpage.save()
        self.eval_webpage.save()

    # errors: dictionary of error labels and values
    def plot_current_errors(self, errors, step):
        if self.tf_log:
            for tag, value in errors.items():
                self.writer.add_scalar(tag,value,step)
                #summary = self.tf.Summary(value=[self.tf.Summary.Value(tag=tag, simple_value=value)])
                #self.writer.add_summary(summary, step)

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

        #print(image_dir, video_dir)
        #exit()

        vfs = sorted(os.listdir(image_dir))

        video_pre = "%s_%s" % (video_name.split("_")[0], video_name.split("_")[1])

        video_list = []
        for f in vfs:
            if f.startswith(video_pre):
                video_list.append(os.path.join(image_dir, f))

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
