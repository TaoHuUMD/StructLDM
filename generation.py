import os
import sys
sys.path.append("..")

import torch

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

from struct_decoder.util import html

from struct_decoder.models.create_model import create_model
from struct_decoder.data.data_loader import CreateDataLoader
from struct_decoder.options.test_option import ProjectOptions
from struct_decoder.util.visualizer import Visualizer

if __name__ == "__main__":

    opt = ProjectOptions().parse_()
    opt.nThreads = 0  
    opt.batchSize = 1  
    opt.serial_batches = True  # no shuffle
    opt.isInfer = True
    opt.phase = "test"
    opt.isTrain = False
    opt.training.batch = 1
    opt.training.chunk = 1

    if len(opt.gpu_ids.split(",")) > 1:
        opt.gpu_ids = opt.gpu_ids.split(",")[0]
        gpu_id = int(opt.gpu_ids)
        opt.gpu_ids = [gpu_id] 
    else:    
        if opt.gpus == opt.gpu_ids:
            opt.gpu_ids = [int(opt.gpu_ids[0])] #
        else:    
            opt.gpus = opt.gpus.split(",")
            gpu_id = int(opt.gpu_ids[0])
            opt.gpu_ids = [int(opt.gpus[gpu_id])] #
    
    print("test  ", opt.gpu_ids)
    torch.cuda.set_device(int(opt.gpu_ids[0]))

    data_loader = CreateDataLoader(opt, opt.phase)
    dataset = data_loader.load_data()
    
    visualizer = Visualizer(opt)

    save_name = opt.name
    if opt.save_name is not None and opt.save_name != '':
        save_name = opt.save_name
    else:
        save_name = opt.expname

    print(opt.save_name, save_name)

    which_epoch = opt.which_epoch    
    model = create_model(opt).cuda()
    
    if opt.test_eval:
        model_name = "decoder_ema_latest" if opt.which_epoch == "-1" else "decoder_ema_%s" % opt.which_epoch
    else: model_name = opt.which_epoch

    print("to load ", model_name)

    test_epoch, epoch_iter = model.load_all(model_name, True)
    opt.which_epoch = test_epoch #- 1
    
    if test_epoch == -1:
        test_epoch, epoch_iter = 1, 0
        print("test model not trained")
        exit()
        
    which_epoch = test_epoch #- 1
    cnt = 0
    model.eval()

    if opt.df.visualize_normal: 
        model.requires_grad_G_normal(True)

    for view_id in opt.multiview_ids:
          
        cnt += 1
    
        web_dir = os.path.join(opt.results_dir, save_name)

        webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (save_name, opt.phase, which_epoch))

        result_dir = os.path.join(opt.results_dir, save_name, "images")
        result_image_num = len(os.listdir(result_dir))
        test_image_num = len(dataset)
        
        print("test images: ", test_image_num, "already generated images", result_image_num)
        
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

            model.compute_visuals(phase="test")

            img_idx = data["frame_name"][0]            
            print('process image... %s' % img_idx)
            cnn += 1

            visualizer.save_images(webpage, model.get_current_visuals(), img_idx if not opt.transfer_texture else f"{img_idx}_{cnn}", "test")
            
            
        webpage.save()        
        print("test finished")

    sys.exit()