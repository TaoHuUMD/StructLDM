import smplx as sx
import os, torch

from . import config
import numpy as np
  
  
from .io.prints import *
   
def load_smpl(gender, device="cuda"): 
    #gender="female"
    #="neutral"
    if isinstance(gender, list):
        gender_ = gender[0]
    elif isinstance(gender, str):
        gender_ = gender
    else: gender_ = "neutral"
    model_path = config.cfg_smpl_paths[gender_]
    return sx.create(model_path=model_path, model_type='smpl').to(device)

def get_smpl_vertices_error(smpl_model, gender, shape_, pose_, trans_=None):
    
    if smpl_model is None:
        smpl_model = load_smpl(gender)
    
    if torch.is_tensor(shape_):
        betas = shape_.reshape(1, 10).float().cuda()
    else:
        betas = torch.from_numpy(shape_.copy().reshape(1, 10)).float().cuda() if shape_ is not None else None  # torch.from_numpy(beta).float(),
    if torch.is_tensor(pose_):
        body_pose = pose_.reshape(24,3)[1:, :].reshape(1, 69).float().cuda()
        global_orient = pose_.reshape(24,3)[0].reshape(1, 3).float().cuda()
    else:
        body_pose = torch.from_numpy(pose_.reshape(24,3)[1:, :].reshape(1, 69)).float().cuda() if pose_ is not None else None
        global_orient = torch.from_numpy(pose_.reshape(24,3)[0].reshape(1, 3)).float().cuda() if pose_ is not None else None
    
    if not torch.is_tensor(trans_): 
        transl = torch.from_numpy(trans_.reshape(1, 3)).float().cuda() if trans_ is not None else None
    else:
        transl = trans_.reshape(1, 3).float().cuda()
    with torch.no_grad():
        out = smpl_model(betas=betas,
                body_pose=body_pose,
                global_orient=global_orient,
                transl=transl
            )
    return out.vertices[0].detach().cpu().numpy(), smpl_model.faces_tensor.cpu().numpy()



def get_smpl_vertices_torch(smpl_model, gender, shape_, pose_, trans_=None, device="cuda", scaling_ = None):
    
    if smpl_model is None:
        smpl_model = load_smpl(gender)

    if torch.is_tensor(shape_):
        betas = shape_.reshape(1, 10).float().to(device)
    else:
        betas = torch.from_numpy(shape_.copy().reshape(1, 10)).float().to(device) if shape_ is not None else None  # torch.from_numpy(beta).float(),
    if torch.is_tensor(pose_):
        body_pose = pose_.reshape(24,3)[1:, :].reshape(1, 69).float().to(device)
        global_orient = pose_.reshape(24,3)[0].reshape(1, 3).float().to(device)
    else:
        body_pose = torch.from_numpy(pose_.reshape(24,3)[1:, :].reshape(1, 69)).float().to(device) if pose_ is not None else None
        global_orient = torch.from_numpy(pose_.reshape(24,3)[0].reshape(1, 3)).float().to(device) if pose_ is not None else None

    if torch.is_tensor(scaling_):
        scaling = scaling_.reshape(1, 1).float().to(device)
    else:
        scaling = torch.from_numpy(scaling_.copy().reshape(1, 1)).float().to(device) if scaling_ is not None else None  # torch.from_numpy(beta).float(),
    
    if not torch.is_tensor(trans_): 
        transl = torch.from_numpy(trans_.reshape(1, 3)).float().to(device) if trans_ is not None else None
    else:
        transl = trans_.reshape(1, 3).float().to(device)
    with torch.no_grad():
        out = smpl_model(betas=betas,
                body_pose=body_pose,
                global_orient=global_orient,
                transl=transl,
                scaling = scaling
            )
        
    return out.vertices, None
    return out.vertices[0].detach().cpu().numpy(), smpl_model.faces_tensor.cpu().numpy()


def get_smpl_vertices(smpl_model, gender, shape_, pose_, trans_=None, device="cuda", scaling_ = None):
    
    if smpl_model is None:
        smpl_model = load_smpl(gender)

    if torch.is_tensor(shape_):
        betas = shape_.reshape(1, 10).float().to(device)
    else:
        betas = torch.from_numpy(shape_.copy().reshape(1, 10)).float().to(device) if shape_ is not None else None  # torch.from_numpy(beta).float(),
    if torch.is_tensor(pose_):
        body_pose = pose_.reshape(24,3)[1:, :].reshape(1, 69).float().to(device)
        global_orient = pose_.reshape(24,3)[0].reshape(1, 3).float().to(device)
    else:
        body_pose = torch.from_numpy(pose_.reshape(24,3)[1:, :].reshape(1, 69)).float().to(device) if pose_ is not None else None
        global_orient = torch.from_numpy(pose_.reshape(24,3)[0].reshape(1, 3)).float().to(device) if pose_ is not None else None

    if torch.is_tensor(scaling_):
        scaling = scaling_.reshape(1, 1).float().to(device)
    else:
        scaling = torch.from_numpy(scaling_.copy().reshape(1, 1)).float().to(device) if scaling_ is not None else None  # torch.from_numpy(beta).float(),
    

    if not torch.is_tensor(trans_): 
        transl = torch.from_numpy(trans_.reshape(1, 3)).float().to(device) if trans_ is not None else None
    else:
        transl = trans_.reshape(1, 3).float().to(device)
    with torch.no_grad():
        out = smpl_model(betas=betas,
                body_pose=body_pose,
                global_orient=global_orient,
                transl=transl,
                scaling = scaling
            )
       
    return out.vertices[0].detach().cpu().numpy(), smpl_model.faces_tensor.cpu().numpy()

def get_smpl_vertices_tensor(smpl_model, shape_, pose_, trans_=None):
    
    pose_ = pose_.reshape(24,3)
    
    betas = shape_.reshape(1, 10)
    body_pose = pose_[1:, :].reshape(1, 69)
    global_orient = pose_[0].reshape(1, 3)
    transl = trans_.reshape(1, 3) if trans_ is not None else None

    with torch.no_grad():
        out = smpl_model(betas=betas,
                body_pose=body_pose,
                global_orient=global_orient,
                transl=transl
            )
      
    return out.vertices

def load_smpl_path_para_vts_face_joints(smpl_path, shape_, pose_, trans_):
    
    smplx_model = sx.create(smpl_path, model_type='smpl')
    
    pose_ = pose_.reshape(24,3) if pose_ is not None else np.zeros((24,3))
    trans_  = np.zeros((1,3)) if trans_ is None else trans_

    with torch.no_grad():
        out = smplx_model(betas=torch.from_numpy(shape_.reshape(1, 10)).float(),  # torch.from_numpy(beta).float(),
                    body_pose=torch.from_numpy(pose_[1:, :].reshape(1, 69)).float(),
                    global_orient=torch.from_numpy(pose_[0].reshape(1, 3)).float(),
                    transl=torch.from_numpy(trans_.reshape(1, 3)).float()
                    )
        
    return out.vertices[0].detach().cpu().numpy(), smplx_model.faces_tensor.cpu().numpy(), out.joints.detach().cpu().numpy()



def load_smpl_path_para(smpl_path, shape_, pose_, trans_):
    
    smplx_model = sx.create(smpl_path, model_type='smpl')
    
    pose_ = pose_.reshape(24,3) if pose_ is not None else np.zeros((24,3))
    trans_  = np.zeros((1,3)) if trans_ is None else trans_

    with torch.no_grad():
        out = smplx_model(betas=torch.from_numpy(shape_.reshape(1, 10)).float(),  # torch.from_numpy(beta).float(),
                    body_pose=torch.from_numpy(pose_[1:, :].reshape(1, 69)).float(),
                    global_orient=torch.from_numpy(pose_[0].reshape(1, 3)).float(),
                    transl=torch.from_numpy(trans_.reshape(1, 3)).float()
                    )
       
    return out.vertices[0].detach().cpu().numpy(), smplx_model.faces_tensor.cpu().numpy()

def toTensor(a):
    if not torch.is_tensor(a):
        return torch.from_numpy(a).float().cuda()
    else: return a.cuda()
