"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import os

import numpy as np
import torch as th
import torch.distributed as dist

from improved_diffusion.latent_datasets import LatentDataset
import torch

from munch import *
import sys
sys.path.append("..")

from improved_diffusion import dist_util, logger
from improved_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)


def main():
    args = create_argparser().parse_args()


    args.all_gpus = [int(g) for g in args.all_gpus.split(",")]
    
    #
    org_ddp = False
    if org_ddp:
        rank, world_size, gpu = dist_util.setup_dist()
        args.rank, args.world_size, args.gpu = rank, world_size, gpu
        
        th.cuda.set_device(args.gpu)
        
        th.backends.cudnn.benchmark = True
        dist.init_process_group(backend='nccl', world_size=args.world_size, rank=args.rank, timeout=datetime.timedelta(seconds=9000))
        print(args.rank, args.world_size, args.gpu)
    else:
        th.cuda.set_device(args.all_gpus[int(args.gpu_ids)])

    logger.configure(dir=args.log_dir)

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )

    
    def find_checkpoint(dir):
        pts = os.listdir(dir)
        max_pt = -1
        print(dir)
        for pt in pts:
            if pt.startswith("diffusion_ema") and pt.endswith(".pt"):
                iters = pt.split("_")[2].split(".")[0]
                iters = int(iters)
                if iters > max_pt: max_pt = iters
        if max_pt == -1: 
            print("model not trained")
            return -1
        return max_pt
        
    #70000
    dir = os.getenv("OPENAI_LOGDIR")
    dir = args.log_dir
    if args.model_path == "":
        if args.which_epoch == -1:
            trained_epoch = find_checkpoint(dir)            
        else:
            trained_epoch = args.which_epoch
        args.model_path = os.path.join(dir, "diffusion_ema_%06d.pt" % trained_epoch)
    
    print("load pretrained model ", args.model_path, args.which_epoch)

    if org_ddp:
        model.load_state_dict(
            dist_util.load_state_dict(args.model_path, map_location="cpu")
        )
        model.to(dist_util.dev()) #.clamp(-1, 1)
    else:    
        model.load_state_dict(torch.load(args.model_path, 'cpu'))
    #model = torch.load(args.model_path, 'cpu')

        model.to("cuda")
    
    model.eval()


    logger.log("sampling...")
    all_images = []
    all_labels = []

    latent_opt = Munch()
    for k in ["lat_reso", "lat_type", "lat_dim", "epoch", "lat_num", "lat_valid_num", "src_model_name", "normalization", "use_lat_mask", "no_mask_dilation", "data_name", "use_partial_latent", "remove_invalid_id", "topo", "is_test", "load_processing_net"]:
        latent_opt[k] = args.__dict__[k] 
    base_dir = "./data/result/trained_model/%s" % args.data_name
    args.data_dir = os.path.join(base_dir, args.src_model_name)
    latent_dir = args.data_dir
    dataset = LatentDataset(
            latent_opt,
            latent_dir,
            classes=None
    )

    gpu_num = len(args.all_gpus)
    
    total_iter = args.num_samples / (args.sample_batch_size * gpu_num)
    if args.num_samples % (args.sample_batch_size * gpu_num) != 0:
        total_iter += 1

    total_iter = int(total_iter)
    cnt = 0
    
    ddim="" if not args.use_ddim else "ddim"
    if args.save_rand:
        ddim += "rand"

    if (org_ddp and dist.get_rank() == 0) or (not org_ddp):
        #shape_str = "x".join([str(x) for x in arr.shape])
        shape_str = f"{args.block_size}x{latent_opt.lat_dim}x{latent_opt.lat_reso}x{latent_opt.lat_reso}"
        outdir = os.path.join(logger.get_dir(), "samples", f"{trained_epoch}_{shape_str}_{ddim}_{args.timestep_respacing}")
        os.makedirs(outdir)



    # using datetime module
    import datetime;

    all_images = []
    while True:
        cnt += 1
        model_kwargs = {}
        if args.class_cond:
            classes = th.randint(
                low=0, high=NUM_CLASSES, size=(args.sample_batch_size,), device=dist_util.dev()
            )
            model_kwargs["y"] = classes
        sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )

        #assert not args.clip_denoised
        if args.save_rand:
            noise = th.randn((args.sample_batch_size, latent_opt.lat_dim, latent_opt.lat_reso, latent_opt.lat_reso)).cuda()
            sample = sample_fn(
                model,
                (args.sample_batch_size, latent_opt.lat_dim, latent_opt.lat_reso, latent_opt.lat_reso),
                clip_denoised=args.clip_denoised,
                model_kwargs=model_kwargs,
                noise=noise
            )


        else:
            sample = sample_fn(
                model,
                (args.sample_batch_size, latent_opt.lat_dim, latent_opt.lat_reso, latent_opt.lat_reso),
                clip_denoised=args.clip_denoised,
                model_kwargs=model_kwargs,
            )

        print("sampled shape", sample.shape)        
        sample = dataset.denorm(sample)
        
        
        if org_ddp: 
            world_size = dist.get_world_size()
            gathered_samples = [th.zeros_like(sample) for _ in range(world_size)] #
            dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
        else:
            world_size = 1
            #all_images.extend([sample.cpu().numpy() for sample in gathered_samples])
            all_images.extend([sample.cpu().numpy()])

        if args.class_cond:
            gathered_labels = [
                th.zeros_like(classes) for _ in range(world_size) #dist.get_world_size()
            ]
            dist.all_gather(gathered_labels, classes)
            all_labels.extend([labels.cpu().numpy() for labels in gathered_labels])

        created_sample = len(all_images) * args.sample_batch_size
        logger.log(f"created {created_sample} samples")

        if created_sample >= args.block_size:        
            ct = datetime.datetime.now()
            ts = ct.timestamp()
            arr = np.concatenate(all_images, axis=0)
            if args.save_rand:
                arr = np.concatenate((arr, noise.cpu().numpy()), axis=0) 

            out_path = os.path.join(outdir, f"{arr.shape[0]}_{ts}.npz")
            logger.log(f"saving to {out_path}")        
            np.savez(out_path, arr)
            all_images = []

            total_samples = len(os.listdir(outdir)) * args.block_size
            logger.log("sampling complete %d / %d" % (total_samples, args.target_size))

            if  total_samples >= args.target_size:
                logger.log("to exit")
                exit()
                            
def create_argparser():
    defaults = dict(
        clip_denoised=False,
        clip_value = 1,

        num_samples=10000,
        batch_size=16,
        sample_batch_size=32,
        use_ddim=False,
        model_path="",
        data_name="fashion",
        data_dir="",

        which_epoch = -1,
        local_rank=0,
        lr=0.0,

        lat_reso= 128, 
        lat_type = "BF", 
        lat_dim = 16,
        use_lat_mask = True, 
        epoch = 0,  
        model_name = "",
        src_model_name = "", 
        lat_num = 0,
        lat_valid_num = 0,
        normalization = "",
        log_dir = "",
        gpu_ids = "0",
        pid = -1,
        tid = 24,
        all_gpus="",
        use_amp = False,
        no_mask_dilation = False, 

        block_size = 64,
        target_size = 50000,

        use_partial_latent = False,
        remove_invalid_id = False,
        topo = False,
        bl32 = False,
        save_rand = False,
        is_test = True,
        load_processing_net = False

    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
