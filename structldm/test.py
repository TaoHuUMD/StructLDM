import numpy

import os

import sys
sys.path.append("..")

import torch

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

from structldm.lib.models.create_model import create_model
from structldm.lib.data.data_loader import CreateDataLoader
from structldm.lib.options.test_option import ProjectOptions
from structldm.lib.util.visualizer import Visualizer

from structldm.engine.thutil import html
from structldm.engine.thutil.io.prints import *
from structldm.engine.thutil.dirs import *

from pdb import set_trace as st


if __name__ == "__main__":

    opt = ProjectOptions().parse_()
    opt.nThreads = 0  # test code only supports nThreads = 1
    opt.batchSize = 1  # test code only supports batchSize = 1
    opt.serial_batches = True  # no shuffle
    opt.isInfer = True
    opt.phase = "test"
    opt.isTrain = False
    opt.training.batch = 1
    opt.training.chunk = 1

    if len(opt.gpu_ids.split(",")) > 1:
        opt.gpu_ids = opt.gpu_ids.split(",")[0]
        gpu_id = int(opt.gpu_ids)
        opt.gpu_ids = [gpu_id] #int(opt.gpus[gpu_id])
    else:    
        if opt.gpus == opt.gpu_ids:
            opt.gpu_ids = [int(opt.gpu_ids[0])] #
        else:    
            opt.gpus = opt.gpus.split(",")
            gpu_id = int(opt.gpu_ids[0])
            opt.gpu_ids = [int(opt.gpus[gpu_id])] #
    
    printg("test  ", opt.gpu_ids)
    torch.cuda.set_device(int(opt.gpu_ids[0]))

    data_loader = CreateDataLoader(opt, opt.phase)
    dataset = data_loader.load_data()
    
    visualizer = Visualizer(opt)

    save_name = opt.name
    if opt.save_name is not None and opt.save_name != '':
        save_name = opt.save_name
    else:
        save_name = opt.expname

    printy(opt.save_name, save_name)

    which_epoch = opt.which_epoch
    
    #model = create_model(opt).cuda().module #opt.gpu_ids[0]
    model = create_model(opt).cuda()
    
    if opt.test_eval:
        model_name = "ema_latest" if opt.which_epoch == "-1" else "ema_%s" % opt.which_epoch
    else: model_name = opt.which_epoch

    test_epoch, epoch_iter = model.load_all(model_name, True)
    opt.which_epoch = test_epoch #- 1
    
    if test_epoch == -1:
        test_epoch, epoch_iter = 1, 0
        printy("test model not trained")
        exit()
        
    printb(test_epoch)
    which_epoch = test_epoch #- 1

    view_num = len(opt.multiview_ids)

    cnt = 0
    model.eval()

    if opt.df.visualize_normal: 
        model.requires_grad_G_normal(True)

    for view_id in opt.multiview_ids:
          
        cnt += 1
        if cnt > 1 and opt.output_geometry:
            printg("bug here")
            sys.exit()#correct exit 
    
        printg(save_name, which_epoch, opt.test_step_size)
        web_dir = get_web_dir(save_name, which_epoch, view_id, opt)

        webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (save_name, opt.phase, which_epoch))

        result_dir = get_img_dir(save_name, which_epoch, view_id, opt)
        result_image_num = len(os.listdir(result_dir))
        test_image_num = len(dataset)
        
        printy("test images: ", test_image_num, "already generated images", result_image_num)
        
        video_name_pre = None
        img_list = []
        part_id = 0

        cnn = 0

        for i, data in enumerate(dataset):
                            
            minibatch = 1     

            if opt.df.visualize_normal:
                generated = model.inference(data)
            else:
                with torch.no_grad(): 
                    generated = model.inference(data)

            if opt.rendering.output_geometry:
                eva_list = []
                eva_list.append(model.compute_visuals(phase="test"))#which_epoch, )
                
                visualizer.save_mesh(eva_list, 0, [data["frame_name"][0]], os.path.join(webpage.get_image_dir(), "../mesh"))
                printg("save mesh ", os.path.join(webpage.get_image_dir(), "../mesh"))
                continue
                #exit()
            else:
                model.compute_visuals(phase="test")#which_epoch, 

            img_idx = data["frame_name"][0]
            if opt.dataset.dataset_name == "thuman2":
                img_idx += "%d" % np.random.randint(1000)
            
            print('process image... %s' % img_idx)
            cnn += 1

            visualizer.save_images(webpage, model.get_current_visuals(), img_idx if not opt.transfer_texture else f"{img_idx}_{cnn}", "test")
            

        webpage.save()
            
        printg("test finished")

    sys.exit()#correct exit