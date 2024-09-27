import torch
import importlib

def create_model(opt):
        
    #model = importlib.import_module(opt.model_module)()
    
    model_path = f'{opt.project_directory}.models.{opt.model_module}'
    print(model_path)

    if opt.isTrain:
        model = importlib.import_module(model_path).Model()
    else:
        model = importlib.import_module(model_path).Model().cuda()
        
    model.initialize(opt)
    
    if opt.verbose:
        print("model [%s] was created" % (model.name()))
   
    print('model opt, data parallel, gpu num ', len(opt.gpu_ids))

    if opt.phase=="test":
        model = torch.nn.DataParallel(model, device_ids=opt.gpu_ids).to(opt.gpu_ids[0])
        return model
    else: #opt.training.distributed:
        model.cuda(opt.gpu_ids[0])
        if(opt.phase=="evaluate"):
            return torch.nn.DataParallel(model, device_ids=opt.gpu_ids).to(opt.gpu_ids[0])

    return model
