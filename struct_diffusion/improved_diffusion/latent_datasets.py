from PIL import Image
import blobfile as bf
#from mpi4py import MPI
import numpy as np
from torch.utils.data import DataLoader, Dataset

import sys
sys.path.append("..")

from .latent import LatentModule
import torch
import os

def load_data(
    *, data_dir, batch_size, latent_opt, class_cond=False, deterministic=False,
    world_size=1,
    rank=0, dataset_ = None
):
    #latent_size, latent_dim, latent_num

    """
    For a dataset, create a generator over (images, kwargs) pairs.

    Each images is an NCHW float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for class labels, in which case the key is "y"
    and the values are integer tensors of class labels.

    :param data_dir: a dataset directory.
    :param batch_size: the batch size of each returned pair.
    :param image_size: the size to which images are resized.
    :param class_cond: if True, include a "y" key in returned dicts for class
                       label. If classes are not available and this is true, an
                       exception will be raised.
    :param deterministic: if True, yield results in a deterministic order.
    """
    if not data_dir:
        raise ValueError("unspecified data directory")
    
    latent_dir = data_dir
    #all_files = _list_image_files_recursively(data_dir)
    classes = None
    if class_cond:
        # Assume classes are the first part of the filename,
        # before an underscore.
        class_names = [bf.basename(path).split("_")[0] for path in all_files]
        sorted_classes = {x: i for i, x in enumerate(sorted(set(class_names)))}
        classes = [sorted_classes[x] for x in class_names]

    if dataset_ is not None:
        dataset = dataset_
    else:
        dataset = LatentDataset(
            latent_opt,
            latent_dir,
            classes=classes
        )

    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=True
        )
    else:
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
        loader = DataLoader(dataset, sampler=train_sampler, batch_size=batch_size, num_workers=3, drop_last=False, pin_memory=True)
        # loader = DataLoader(
        #     dataset, batch_size=batch_size, shuffle=True, num_workers=3, drop_last=False, persistent_workers=True
        # )
    while True:
        yield from loader

class LatentDataset(Dataset):
    def __init__(self, opt, latent_dir, classes=None, shard=0, num_shards=1):
        super().__init__()
        
        opt["uv_reso"] = opt.lat_reso
        opt["uv_type"] = opt.lat_type
        opt["uv_dim"] = opt.lat_dim

        self.opt = opt
        print("opt", self.opt)
        
        self.latent_list = LatentModule(opt=self.opt)

        def load_part(model, pretrained_dict):
            model_dict = model.state_dict()

            updated_pretrained_dict = {}
            if True:
                for k, v in pretrained_dict.items():
                    if k.startswith("net_Latent."):
                        updated_pretrained_dict.update({k.replace("net_Latent.", ""): v})
            
            # 1. filter out unnecessary keys
            pretrained_dict = {k: v for k, v in updated_pretrained_dict.items() if k in model_dict}
            # 2. overwrite entries in the existing state dict

            assert len(pretrained_dict) > 0 
            
            model_dict.update(pretrained_dict) 
            # 3. load the new state dict
            model.load_state_dict(model_dict)
        
        assert os.path.isfile(os.path.join(latent_dir, 'decoder_ema_{}.pth'.format(opt.epoch)))
        pretrained_model = torch.load(os.path.join(latent_dir, 'decoder_ema_{}.pth'.format(opt.epoch)), 'cpu')        
        print("load model ", os.path.join(latent_dir, 'decoder_ema_{}.pth'.format(opt.epoch)))

        load_part(self.latent_list, pretrained_model["net"])
    
        self.latent_list.calc_latent()

        self.resolution = opt.uv_reso
        self.local_classes = None if classes is None else classes[shard:][::num_shards]

    def get_stas(self):
        return self.latent_list.get_stas()

    def denorm(self, sample):        
        return self.latent_list.denorm(sample)
    
    def normalize_input(self, sample):
        return self.latent_list.normalize_input(sample)

    def mix_latent(self, lat_list, mix_code):
        return self.latent_list.mix_latent(lat_list, mix_code)

    def mix_latent_crop_foot(self, src_lat, tgt_lat):
        return self.latent_list.mix_latent_crop_foot(src_lat, tgt_lat)

    

    def __len__(self):
        return self.latent_list.__len__()
    

    def __getitem__(self, idx):
        return self.latent_list.get_normalized_latent(idx)

