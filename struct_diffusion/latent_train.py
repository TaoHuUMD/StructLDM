"""
Train a diffusion model on images.
"""

import sys, os
current = os.path.dirname(os.path.realpath(__file__))
sys.path.append(current + "/../")

import argparse
import datetime
from munch import *

from improved_diffusion import dist_util, logger
from improved_diffusion.latent_datasets import load_data

import sys
sys.path.append("..")

from improved_diffusion.latent_datasets import LatentDataset

from improved_diffusion.resample import create_named_schedule_sampler
from improved_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from improved_diffusion.train_util import TrainLoop
import torch as th
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist

def main():
    args = create_argparser().parse_args()

    # dist_util.setup_dist()
    rank, world_size, gpu = dist_util.setup_dist()
    args.rank, args.world_size, args.gpu = rank, world_size, gpu
    th.cuda.set_device(args.gpu)
    th.backends.cudnn.benchmark = True
    dist.init_process_group(backend='nccl', world_size=args.world_size, rank=args.rank, timeout=datetime.timedelta(seconds=9000))
    print(args.rank, args.world_size, args.gpu)

    logger.configure(dir=args.log_dir)

    #                os.makedirs(self.log_dir, exist_ok=True)
    log_dir = "./data/trained_model/logdirs"
    log_dir = os.path.join(log_dir, f'diff_{args.data_name}_{args.model_name}')
    if dist.get_rank() == 0:
        os.makedirs(log_dir, exist_ok=True)

        writer = SummaryWriter(log_dir)
    else:
        writer = None

    if dist.get_rank() == 0:
        logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.to(dist_util.dev())
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    if dist.get_rank() == 0:
        logger.log("creating data loader...")
    if args.data_name == "imagenet":
        data = load_data(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            image_size=args.image_size,
            class_cond=args.class_cond,
        )
    else:

        latent_opt = Munch()
        for k in ["lat_reso", "lat_type", "lat_dim", "epoch", "lat_num", "lat_valid_num", "src_model_name", "normalization", "use_lat_mask", "no_mask_dilation", "data_name", "use_partial_latent", "remove_invalid_id", "topo", "bl32", "load_processing_net", "use_relu",
        "smooth_kernel_size", "scale_fc"]:
            latent_opt[k] = args.__dict__[k] 

        base_dir = "./data/result/trained_model/%s" % args.data_name
        args.data_dir = os.path.join(base_dir, args.src_model_name)

        logger.log("creating data loader...")

        dataset = LatentDataset(
            latent_opt,
            args.data_dir,
            classes=None
        )
        data_stas = dataset.get_stas()

        data = load_data(
            data_dir=args.data_dir,
            latent_opt=latent_opt,
            batch_size=args.batch_size,
            #image_size=args.image_size,
            #class_cond=args.class_cond,
            world_size=args.world_size,
            rank=args.rank,
            dataset_=dataset
        )

        clip_tuple = None
        if args.clip_x0_training:
            lmin, lmax = data_stas
            print("min max after norm ", lmin, lmax)
            clip_tuple = (True, lmin, lmax)

    if dist.get_rank() == 0:
        logger.log("training...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        use_amp=args.use_amp,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
        use_cond=args.use_cond,
        writer=writer,
        clip_tuple = clip_tuple
    ).run_loop()
    if dist.get_rank() == 0:
        writer.close()


def create_argparser():
    defaults = dict(
        data_name="fashion",
        data_dir="",
        log_dir="",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=1,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=10,
        save_interval=20000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
        use_amp=False,
        num_subjects=500,
        layer_idx=None,
        use_cond=False,
        local_rank=0,
        lat_reso= 128, 
        lat_type = "BF", 
        lat_dim = 16,
        use_lat_mask = True,
        no_mask_dilation = False, 
        epoch = 0,  
        model_name = "",
        src_model_name = "", 
        lat_num = 0,
        lat_valid_num = 0,
        normalization = "",
        timestep_respacing = 1000, 
        num_samples = 32,
        sample_batch_size = 1,

        clip_x0_training = False,
        use_partial_latent = False,
        remove_invalid_id = False,
        topo = False,
        bl32 = False,
        load_processing_net = False,
        use_relu = False,
        smooth_kernel_size = 5,
        scale_fc = 10
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
