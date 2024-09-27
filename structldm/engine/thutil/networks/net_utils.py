import torch
import os
from torch import nn
import numpy as np
import torch.nn.functional
from collections import OrderedDict
from termcolor import colored
from .load import load_with_key_mismatch

from collections import OrderedDict
def divide_network_params(net, net1_prefix="uv_embedding_list"):
    net_1 = OrderedDict()
    net_2 = OrderedDict()
    for k in net.keys():
        if k.startswith(net1_prefix):
            net_1[k] = net[k]
            continue                
        net_2[k] = net[k]
    return net_1, net_2
        
def print_key(net):
    par1 = dict(net.named_parameters())
    for k in par1.keys():
        print(k)



def check_int(s):
    if s[0] in ('-', '+'):
        return s[1:].isdigit()
    return s.isdigit()

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

def load_model(model_dir,
               net,               
               opt_G = None, opt_D = None,
               scheduler = None,
               recorder = None,
               resume = True,
               epoch= ""):
    
    if not resume:
        os.system('rm -rf {}'.format(model_dir))

    print(os.path.exists(model_dir), model_dir)
    if not os.path.exists(model_dir):
        return -1, 0, 0


    tgt_model_name = 'latest'
    pth_file = "%s.pth" % tgt_model_name

    
    pths = [
        int(pth.split('.')[0]) for pth in os.listdir(model_dir)
        if pth != 'latest.pth' and pth.find('.pth')!=-1 and check_int(pth.split('.')[0])
    ]
    if len(pths) == 0 and pth_file not in os.listdir(model_dir):
        return -1, 0, 0

    if epoch == "" or epoch == "-1":
        if pth_file in os.listdir(model_dir):
            pth = tgt_model_name
        else:
            pth = max(pths)
    else:
        pth = epoch
        
    if not os.path.isfile(os.path.join(model_dir,'{}.pth'.format(pth))):
         return -1, 0, 0
    
    print('load model: {}'.format(os.path.join(model_dir,
                                               '{}.pth'.format(pth))))
    pretrained_model = torch.load(
        os.path.join(model_dir, '{}.pth'.format(pth)), 'cuda')
     
    net.load_state_dict(pretrained_model['net'])

    if opt_G is not None:
        opt_G.load_state_dict(pretrained_model['optimG'])
    
    if opt_D is not None:
        opt_D.load_state_dict(pretrained_model['optimD'])

    if scheduler is not None:
        scheduler.load_state_dict(pretrained_model['scheduler'])
    
    if recorder is not None:
        recorder.load_state_dict(pretrained_model['recorder'])
        
    lr = pretrained_model['lr']
        
    return pretrained_model['epoch'], pretrained_model['iter']+1, lr


def save_model(model_dir, net, opt_G, opt_D, label, epoch, iter, lr):
    
    os.system('mkdir -p {}'.format(model_dir))

    pth_file = "%s.pth" % label
    
    #scheduler, recorder,
    model = {
        'net': remove_net_prefix(net.state_dict()),
        #'scheduler': scheduler.state_dict(),
        #'recorder': recorder.state_dict(),
        'epoch': epoch,
        'label': label,
        'iter': iter,
        'lr': lr
    }
    if opt_G is not None:
        model.update({'optimG': opt_G.state_dict()})
    if opt_D is not None:
        model.update({'optimD': opt_D.state_dict()})
    
    torch.save(model, os.path.join(model_dir, pth_file))
    
    pths = [
        int(pth.split('.')[0]) for pth in os.listdir(model_dir)
        if pth != 'latest.pth' and pth.find('.pth')!=-1 and check_int(pth.split('.')[0])
    ]
    if len(pths)<3:
        return 
    

    return 


def load_model_full(model_dir,
               net,               
               opt_G = None, opt_D = None,
               scheduler = None,
               recorder = None,
               resume = True,
               epoch= "", strict = True, device = None):
    
    if not resume:
        os.system('rm -rf {}'.format(model_dir))

    print(os.path.exists(model_dir), model_dir)
    if not os.path.exists(model_dir):
        return -1


    is_test = True if epoch.find("ema_") != -1 else False
    

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
    elif epoch == "-2":
        pth = max(pths)
    else:
        pth = epoch
        
    print(os.path.join(model_dir,'{}.pth'.format(pth)), pth, epoch)

    if not os.path.isfile(os.path.join(model_dir,'{}.pth'.format(pth))):
         return -1
    
    print('load model: {}'.format(os.path.join(model_dir,
                                               '{}.pth'.format(pth))))
    pretrained_model = torch.load(
        os.path.join(model_dir, '{}.pth'.format(pth)), 'cpu') #'cpu' if device is None else "cuda:%d" % device)
        
    if strict == False:
        from .load import _load_model
        print("tyle ", type(pretrained_model['net']))
        error = _load_model(net, pretrained_model['net'])

        #only handle misshape of D
        #if opt_D is not None and error == 0:
        #    opt_D.load_state_dict(pretrained_model['optimD'])

        try:
            if opt_D is not None and not is_test:
                opt_D.load_state_dict(pretrained_model['optimD'])
        except: pass

        try:
            if opt_G is not None and not is_test:
                opt_G.load_state_dict(pretrained_model['optimG'])
        except: pass
        
        if scheduler is not None:
            scheduler.load_state_dict(pretrained_model['scheduler'])
            #scheduler.load_state_dict(pretrained_model['scheduler'])
        
        if recorder is not None:
            recorder.load_state_dict(pretrained_model['recorder'])
            #recorder.load_state_dict(pretrained_model['recorder'])
        
    else:
        net.load_state_dict(pretrained_model['net'])

        if opt_G is not None and not is_test:
            opt_G.load_state_dict(pretrained_model['optimG'])
        
        if opt_D is not None and strict and not is_test:
            opt_D.load_state_dict(pretrained_model['optimD'])

        if scheduler is not None and not is_test:
            scheduler.load_state_dict(pretrained_model['scheduler'])
        
        if recorder is not None and not is_test:
            recorder.load_state_dict(pretrained_model['recorder'])
        
    
    g_lr = pretrained_model['g_lr'] if "g_lr" in pretrained_model.keys() else None
    d_lr = pretrained_model['d_lr'] if "d_lr" in pretrained_model.keys() else None
        
    return pretrained_model['epoch'], pretrained_model['iter']+1, g_lr, d_lr


    

def load_model_full_df3(model_dir,
               net,               
               opt_G = None, opt_D = None,
               scheduler = None,
               recorder = None,
               resume = True,
               epoch= "", strict = True):
    
    if not resume:
        os.system('rm -rf {}'.format(model_dir))

    print(os.path.exists(model_dir), model_dir)
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
        
    if not os.path.isfile(os.path.join(model_dir,'{}.pth'.format(pth))):
         return -1
    
    print('load model: {}'.format(os.path.join(model_dir,
                                               '{}.pth'.format(pth))))
    pretrained_model = torch.load(
        os.path.join(model_dir, '{}.pth'.format(pth)), 'cuda')
        
    if strict == False:
        from .load import _load_model

        load_with_key_mismatch(net, pretrained_model['net'])
    
        if scheduler is not None:
            scheduler.load_state_dict(pretrained_model['scheduler'])
            #scheduler.load_state_dict(pretrained_model['scheduler'])
        
        if recorder is not None:
            recorder.load_state_dict(pretrained_model['recorder'])
            #recorder.load_state_dict(pretrained_model['recorder'])
        
    else:
        net.load_state_dict(pretrained_model['net'])

        if opt_G is not None:
            opt_G.load_state_dict(pretrained_model['optimG'])
        
        if opt_D is not None:
            opt_D.load_state_dict(pretrained_model['optimD'])


        if scheduler is not None:
            scheduler.load_state_dict(pretrained_model['scheduler'])
        
        if recorder is not None:
            recorder.load_state_dict(pretrained_model['recorder'])
        
    
    g_lr = pretrained_model['g_lr'] if "g_lr" in pretrained_model.keys() else None
    d_lr = pretrained_model['d_lr'] if "d_lr" in pretrained_model.keys() else None
        
    return pretrained_model['epoch'], pretrained_model['iter']+1, g_lr, d_lr



def save_model_full(model_dir, net, opt_G, opt_D, label, epoch, iter, g_lr, d_lr):
    
    os.system('mkdir -p {}'.format(model_dir))

    pth_file = "%s.pth" % label
    
    #scheduler, recorder,
    model = {
        'net': remove_net_prefix(net.state_dict()),
        #'scheduler': scheduler.state_dict(),
        #'recorder': recorder.state_dict(),
        'epoch': epoch,
        'label': label,
        'iter': iter,
        'g_lr': g_lr,
        'd_lr': d_lr
    }

    if opt_G is not None:
        model.update({'optimG': opt_G.state_dict()})
    if opt_D is not None:
        model.update({'optimD': opt_D.state_dict()})
    
    torch.save(model, os.path.join(model_dir, pth_file))
    
    pths = [
        int(pth.split('.')[0]) for pth in os.listdir(model_dir)
        if pth != 'latest.pth' and pth.find('.pth')!=-1 and check_int(pth.split('.')[0])
    ]
    if len(pths)<3:
        return 
    
    return 
    
def load_network(net, model_dir, resume=True, epoch=-1, strict=True):
    if not resume:
        return 0

    if not os.path.exists(model_dir):
        print(colored('pretrained model does not exist', 'red'))
        return 0

    if os.path.isdir(model_dir):
        
        pths = [
                int(pth.split('.')[0]) for pth in os.listdir(model_dir)
                if pth != 'latest.pth' and pth != 'D_latest.pth' and (pth[0] !='D' and check_int(pth.split('.')[0]))
        ]
        
        if len(pths) == 0 and 'latest.pth' not in os.listdir(model_dir):
            return 0

        if epoch == -1:
            if 'latest.pth' in os.listdir(model_dir):
                pth = 'latest'
            else:
                pth = max(pths)
        else:
            pth = epoch

        model_path = os.path.join(model_dir, '{}.pth'.format(pth))
    else:
        model_path = model_dir

    print('load model: {}'.format(model_path))
    pretrained_model = torch.load(model_path)
    net.load_state_dict(pretrained_model['net'], strict=strict)
    return pretrained_model['epoch'] + 1

def remove_untrainable_net(net):
    net_ = OrderedDict()
    for k in net.keys():
        print(k, net[k].requires_grad)
        if not net[k].requires_grad:
            print(k)
            continue        
        net_[k] = net[k]
    return net_

def remove_net_prefix(net, prefix="criterion"):
    net_ = OrderedDict()
    for k in net.keys():
        if k.startswith(prefix):
            continue
        
        net_[k] = net[k]
    return net_

def remove_net_prefix_why(net, prefix="criterion"):
    net_ = OrderedDict()
    for k in net.keys():
        if k.startswith(prefix):
            net_[k[len(prefix):]] = net[k]
        else:
            net_[k] = net[k]
    return net_


def add_net_prefix(net, prefix):
    net_ = OrderedDict()
    for k in net.keys():
        net_[prefix + k] = net[k]
    return net_


def replace_net_prefix(net, orig_prefix, prefix):
    net_ = OrderedDict()
    for k in net.keys():
        if k.startswith(orig_prefix):
            net_[prefix + k[len(orig_prefix):]] = net[k]
        else:
            net_[k] = net[k]
    return net_


def remove_net_layer(net, layers):
    keys = list(net.keys())
    for k in keys:
        for layer in layers:
            if k.startswith(layer):
                del net[k]
    return net
