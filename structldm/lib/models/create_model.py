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
   
    
    return model.to(opt.gpu_ids[0])
