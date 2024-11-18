import configargparse
from munch import *
import numpy as np
from pdb import set_trace as st
import yaml
from easydict import EasyDict as edict
import os

class BaseOptions():
    def __init__(self):
        self.parser = configargparse.ArgumentParser()
        self.initialized = False

    def initialize(self):
        # Dataset options
        dataset = self.parser.add_argument_group('dataset')
        dataset.add_argument("--dataset_path", type=str, default='./data/dataset/renderpeople')
        dataset.add_argument("--random_flip", action='store_true')
        dataset.add_argument("--gaussian_weighted_sampler", action='store_true')
        dataset.add_argument("--sampler_std", type=float, default=15)

        dataset.add_argument("--gender", type=str, default="neutral")

        dataset.add_argument("--dataset_name", type=str, default="renderpeople")
        dataset.add_argument("--dataset_size", type=int, default=8800, help='')

        dataset.add_argument("--img_H", type=int, default=1024, help='')
        dataset.add_argument("--img_W", type=int, default=512, help='')

        dataset.add_argument("--ratio", type=int, default=-1, help='')

        dataset.add_argument("--data_step", type=int, default=1, help='for video dataset')
        dataset.add_argument("--sample_num_per_video", type=int, default=-1, help='for test')

        dataset.add_argument("--fitting_test", action='store_true')

        # Experiment Options
        experiment = self.parser.add_argument_group('experiment')
        experiment.add_argument('--config', is_config_file=True, help='config file path')
        experiment.add_argument("--expname", type=str, default='debug', help='experiment name')
        experiment.add_argument("--ckpt", type=str, default='300000', help="path to the checkpoints to resume training")
        experiment.add_argument("--continue_training", action="store_true", help="continue training the model")

        # Training loop options
        training = self.parser.add_argument_group('training')

        training.add_argument("--checkpoints_dir", type=str, default='./data/result/trained_model/', help='checkpoints directory name')

        training.add_argument("--distributed", action="store_true", help="distributed training")

        training.add_argument("--iter", type=int, default=300000, help="total number of training iterations")
        training.add_argument("--batch", type=int, default=4, help="batch sizes for each GPU. A single RTX2080 can fit batch=4, chunck=1 into memory.")
        training.add_argument("--chunk", type=int, default=4, help='number of samples within a batch to processed in parallel, decrease if running out of memory')
        training.add_argument("--val_n_sample", type=int, default=8, help="number of test samples generated during training")
        training.add_argument("--d_reg_every", type=int, default=16, help="interval for applying r1 regularization to the StyleGAN generator")
        training.add_argument("--g_reg_every", type=int, default=4, help="interval for applying path length regularization to the StyleGAN generator")
        training.add_argument("--local_rank", type=int, default=0, help="local rank for distributed training")
        training.add_argument("--mixing", type=float, default=0.9, help="probability of latent code mixing")
        training.add_argument("--lr", type=float, default=0.002, help="learning rate")
        


        training.add_argument("--r1", type=float, default=300, help="weight of the r1 regularization")
        training.add_argument("--eikonal_lambda", type=float, default=0.5, help="weight of the eikonal regularization")
        training.add_argument("--min_surf_lambda", type=float, default=1.5, help="weight of the minimal surface regularization")
        training.add_argument("--min_surf_beta", type=float, default=100.0, help="weight of the minimal surface regularization")
        training.add_argument('--not_deltasdf',  action='store_true')

        training.add_argument("--path_regularize", type=float, default=2, help="weight of the path length regularization")
        training.add_argument("--path_batch_shrink", type=int, default=2, help="batch size reducing factor for the path length regularization (reduce memory consumption)")
        training.add_argument("--wandb", action="store_true", help="use weights and biases logging")
        training.add_argument("--small_aug", action='store_true')

        training.add_argument("--with_sdf", action='store_true')

        
        training.add_argument("--pool_size", type=int, default=0, help='weight of Discriminator')        
        training.add_argument('--face_model', type=str, default='../asset/spretrains/sphere20a_20171020.pth')
        training.add_argument('--no_lsgan', action='store_true', help='do *not* use least square GAN, if false, use vanilla GAN')
        training.add_argument('--no_ganFeat_loss', action='store_true', help='do *not* use least square GAN, if false, use vanilla GAN')
        training.add_argument('--no_vgg_loss', action='store_true', help='do *not* use least square GAN, if false, use vanilla GAN')

        training.add_argument("--debug_small_data", action='store_true', help='')
        training.add_argument("--debug_data_size", type=int, default=10, help='')
        
        training.add_argument("--test_overfitting", action='store_true')


        training.add_argument("--fix_embed", action='store_true')
        
        training.add_argument("--ablation_converge", action='store_true')



        training.add_argument("--glr", type=float, default=2e-5)
        training.add_argument("--dlr", type=float, default=2e-4)
        training.add_argument("--deltasdf", action='store_true', default=False)
        training.add_argument("--fid_path", type=str, default='')
        training.add_argument('--lr_embed', type=float, default=0, help="lr for embedding")
        training.add_argument('--lr_embed_same', action='store_true', help="same lr for embed and G")
        training.add_argument('--separate_opt_embed', action='store_true', help="a separate optimizer for embedding")
        

        # Inference Options
        inference = self.parser.add_argument_group('inference')
        
        inference.add_argument("--truncation_ratio", type=float, default=0.5, help="truncation ratio, controls the diversity vs. quality tradeoff. Higher truncation ratio would generate more diverse results")
        inference.add_argument("--truncation_mean", type=int, default=10000, help="number of vectors to calculate mean for the truncation")
        inference.add_argument("--identities", type=int, default=16, help="number of identities to be generated")
        inference.add_argument("--num_views_per_id", type=int, default=1, help="number of viewpoints generated per identity")
        inference.add_argument("--no_surface_renderings", action="store_true", help="when true, only RGB outputs will be generated. otherwise, both RGB and depth videos/renderings will be generated. this cuts the processing time per video")
        inference.add_argument("--fixed_camera_angles", action="store_true", help="when true, the generator will render indentities from a fixed set of camera angles.")
        inference.add_argument("--azim_video", action="store_true", help="when true, the camera trajectory will travel along the azimuth direction. Otherwise, the camera will travel along an ellipsoid trajectory.")

        # Generator options
        model = self.parser.add_argument_group('model')
        model.add_argument("--size", type=int, nargs="+", default=[256, 128], help="image sizes for the model")
        model.add_argument("--style_dim", type=int, default=128, help="number of style input dimensions")
        model.add_argument("--channel_multiplier", type=int, default=2, help="channel multiplier factor for the StyleGAN decoder. config-f = 2, else = 1")
        model.add_argument("--n_mlp", type=int, default=8, help="number of mlp layers in stylegan's mapping network")
        model.add_argument("--lr_mapping", type=float, default=0.01, help='learning rate reduction for mapping network MLP layers')
        model.add_argument("--renderer_spatial_output_dim", type=int, nargs="+", default=[128, 64], help='spatial resolution of the StyleGAN decoder inputs')
        model.add_argument("--project_noise", action='store_true', help='when true, use geometry-aware noise projection to reduce flickering effects (see supplementary section C.1 in the paper). warning: processing time significantly increases with this flag to ~20 minutes per video.')
        model.add_argument("--smpl_model_folder", type=str, default="smpl_models", help='path to smpl model folder')
        model.add_argument("--smpl_gender", type=str, default="neutral")
        model.add_argument("--voxhuman_name", type=str, default=None)

        model.add_argument('--texdecoder_down', type=int, default=3)
        model.add_argument('--texdecoder_resnet', type=int, default=3)
        model.add_argument('--texdecoder_outdim', type=int, default=-1)
        
        model.add_argument("--netG", type=str, default="global")

        #texdecoder_down texdecoder_resnet texdecoder_outdim
        model.add_argument('--n_blocks_local', type=int, default=3,
                                 help='number of residual blocks in the local enhancer network')
        model.add_argument('--n_local_enhancers', type=int, default=1, help='number of local enhancers to use')
        model.add_argument('--niter_fix_global', type=int, default=0,
                                 help='number of epochs that we only train the outmost local enhancer')

        model.add_argument('--ngf', type=int, default=64, help='# of gen filters in first conv layer')
        model.add_argument('--n_downsample_global', type=int, default=4,
                                 help='number of downsampling layers in netG')
        model.add_argument('--n_blocks_global', type=int, default=3,
                                 help='number of residual blocks in the global generator network')
        model.add_argument('--norm', type=str, default='instance',
                                 help='instance normalization or batch normalization')
        
        model.add_argument('--num_D', type=int, default=2, help='number of discriminators to use')
        model.add_argument('--n_layers_D', type=int, default=3, help='only used if which_model_netD==n_layers')
        model.add_argument('--ndf', type=int, default=64, help='# of discrim filters in first conv layer')    



        # Camera options
        camera = self.parser.add_argument_group('camera')
        camera.add_argument("--uniform", action="store_true", help="when true, the camera position is sampled from uniform distribution. Gaussian distribution is the default")
        camera.add_argument("--azim", type=float, default=0.3, help="camera azimuth angle std/range in Radians")
        camera.add_argument("--elev", type=float, default=0.15, help="camera elevation angle std/range in Radians")
        camera.add_argument("--fov", type=float, default=6, help="camera field of view half angle in Degrees")
        camera.add_argument("--dist_radius", type=float, default=0.12, help="radius of points sampling distance from the origin. determines the near and far fields")

        # Volume Renderer options
        rendering = self.parser.add_argument_group('rendering')
        # MLP model parameters
        rendering.add_argument("--depth", type=int, default=5, help='layers in network')
        rendering.add_argument("--width", type=int, default=128, help='channels per layer')
        # Volume representation options
        rendering.add_argument("--no_sdf", action='store_true', help='By default, the raw MLP outputs represent an underline signed distance field (SDF). When true, the MLP outputs represent the traditional NeRF density field.')
        rendering.add_argument("--no_z_normalize", action='store_true', help='By default, the model normalizes input coordinates such that the z coordinate is in [-1,1]. When true that feature is disabled.')
        rendering.add_argument("--static_viewdirs", action='store_true', help='when true, use static viewing direction input to the MLP')
        rendering.add_argument("--is_aist", action='store_true')
        # Ray intergration options
        rendering.add_argument("--no_offset_sampling", action='store_true', help='when true, use random stratified sampling when rendering the volume, otherwise offset sampling is used. (See Equation (3) in Sec. 3.2 of the paper)')
        rendering.add_argument("--perturb", type=float, default=1., help='set to 0. for no jitter, 1. for jitter')
        rendering.add_argument("--raw_noise_std", type=float, default=0., help='std dev of noise added to regularize sigma_a output, 1e0 recommended')
        rendering.add_argument("--force_background", action='store_true', help='force the last depth sample to act as background in case of a transparent ray')
        # Set volume renderer outputs
        rendering.add_argument("--return_xyz", action='store_true', help='when true, the volume renderer also returns the xyz point could of the surface. This point cloud is used to produce depth map renderings')
        rendering.add_argument("--return_sdf", action='store_true', help='when true, the volume renderer also returns the SDF network outputs for each location in the volume')
        rendering.add_argument("--stepsize", type=float, default=0.02, help='rendering step size')
        rendering.add_argument("--white_bg", action='store_true')
        rendering.add_argument("--input_ch_views", type=int, default=3)
        # inference options
        rendering.add_argument("--render_video", action='store_true')

        rendering.add_argument("--cat_style", action='store_true')
        rendering.add_argument("--no_pose_encoding", action='store_true')
        rendering.add_argument("--not_use_net", action='store_true')
        
        rendering.add_argument("--camera_cond", action='store_true')


        rendering.add_argument("--voxel_net", action='store_true')

        rendering.add_argument("--sdf_template", action='store_true')
        rendering.add_argument("--transfer_to_mean", action='store_true')
        
        rendering.add_argument("--output_geometry", action='store_true', help="app, pose transfer")

        model.add_argument('--vol_render_shallow_net', action="store_true", help='only used if which_model_netD==n_layers')

        rendering.add_argument("--up_layer", type=int, default=0)

        model.add_argument('--vol_render_big_net', action="store_true", help='only used if which_model_netD==n_layers')

        model.add_argument('--vol_render_siren_style', action="store_true", help='only used if which_model_netD==n_layers')

        model.add_argument('--vol_render_siren_layer', action="store_true", help='change last density/color layer')

        model.add_argument('--vol_render_siren_cat', action="store_true", help='only used if which_model_netD==n_layers')
        model.add_argument('--vol_render_mlp_cat', action="store_true", help='number of discriminators to use')

        model.add_argument("--disc_start", type=int, default=0)        
        model.add_argument('--patch_gan', action="store_true", help='number of discriminators to use')



        hypara = self.parser.add_argument_group('hypara')

        hypara.add_argument("--w_tv_embed", type=float, default=1e-4, help='weight of Discriminator')        
        hypara.add_argument("--w_G_Normal", type=float, default=1.0, help='weight of Discriminator')
        

        hypara.add_argument("--nerf_decay", action='store_true', help='weight of nerf decay')
        
        hypara.add_argument("--adjust_gamma", action='store_true', default=False)
        hypara.add_argument("--gamma_lb", type=float, default=20)

        hypara.add_argument("--no_D_grad", action='store_true', help='grad penalty in discrim')

        hypara.add_argument("--w_D", type=float, default=1.0, help='weight of Discriminator')
        hypara.add_argument("--w_D_grad", type=float, default=300, help='grad penalty in discrim')

        hypara.add_argument("--w_G_L1", type=float, default=0.5, help='weight of G_GAN')
        hypara.add_argument("--w_G_feat", type=float, default=1, help='weight of G_GAN')

        hypara.add_argument("--w_G_VGG", type=float, default=1, help='weight of G_VGG')
        hypara.add_argument("--w_nerf_VGG", type=float, default=1, help='weight of G_VGG')
        #hypara.add_argument("--nerf_vgg", action='store_true')
        
        hypara.add_argument("--w_sdf_ek", type=float, default=1, help='weight of G_VGG')

        hypara.add_argument("--w_tv_normal", type=float, default=1.0, help='mask loss of G_GAN')

        hypara.add_argument("--w_G_Mask", type=float, default=1.0, help='mask loss of G_GAN')
        
        hypara.add_argument("--w_G_GAN", type=float, default=1.0, help='weight of G_GAN')
        hypara.add_argument("--w_Face", type=float, default=5.0, help='face weight')
        
        hypara.add_argument("--w_tex_rec", type=float, default=2.0, help='weight of tec rec')
        hypara.add_argument("--w_nerf_rec", type=float, default=1.0, help='nerf rec weight')

        hypara.add_argument("--w_posmap", type=float, default=1.0, help='nerf rec weight')

        hypara.add_argument("--w_cross_tex", type=float, default=1.0, help='weight of tec rec')

        hypara.add_argument("--w_embed", type=float, default=0.01, help='weight of tec rec')

        hypara.add_argument("--update_emd_lr", action='store_true')

        #tv_latent w_tv_embed

        hypara.add_argument("--detach_sr", action='store_true')

        hypara.add_argument("--w_G_VGG_NeRF", type=float, default=0)
        hypara.add_argument("--nerf_vgg", action='store_true')


        self.parser.add_argument("--phase", type=str, help='train evaluate')
        self.parser.add_argument("--isTrain", action='store_true', help='weight of nerf decay')

        self.parser.add_argument("--world_size", type=int, default=0, help='weight of tec rec')
        self.parser.add_argument("--rank", type=int, default=0, help='weight of tec rec')
        
        self.parser.add_argument('--serial_batches', action='store_true',
                                        help='if true, takes images in order to make batches, otherwise takes them randomly')
        
        self.parser.add_argument('--nThreads', default=4, type=int, help='# threads for loading data')

        self.parser.add_argument('--niter', default=100, type=int, help='# threads for loading data')
        self.parser.add_argument('--niter_decay', default=0, type=int, help='# threads for loading data')

        self.parser.add_argument('--gen_ratio', default=1, type=int, help='# threads for loading data')
                
        self.parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        self.parser.add_argument('--gpus', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        
        self.parser.add_argument('--gpu_num', default=0, type=int, help='# threads for loading data')

        self.parser.add_argument('--load_pretrained_model', action='store_true', help = 'load model, generally low-reso trained model')
        self.parser.add_argument('--load_strict', action='store_true', help = '')        
        self.parser.add_argument('--load_model_name', type=str, help = 'model name')
        self.parser.add_argument('--confirm_pretrained_model', action='store_true', help = 'load model, generally low-reso trained model')
        

        

        self.parser.add_argument('--max_iters_epoch', default=0, type=int, help='# threads for loading data')
        self.parser.add_argument('--save_latest_freq', default=10, type=int, help='# threads for loading data')
        self.parser.add_argument('--eva_epoch_freq', default=10, type=int, help='# threads for loading data')


        self.parser.add_argument('--print_freq', default=0, type=int, help='# threads for loading data')
        self.parser.add_argument('--display_freq', default=10, type=int, help='# threads for loading data')
        self.parser.add_argument('--save_epoch_freq', default=10, type=int, help='# threads for loading data')
        
        self.parser.add_argument("--debug_val", action='store_true', help='weight of nerf decay')

        self.parser.add_argument("--no_html", action='store_true', help='weight of nerf decay')
        self.parser.add_argument("--display_winsize", type=int, default=3000, help='weight of nerf decay')
        
        self.parser.add_argument('--name', type=str, help='# threads for loading data')

        self.parser.add_argument('--multi_datasets', type=str, help='# threads for loading data')

        self.parser.add_argument('--log_dir', type=str, default='../data/result/nr/trained_model', help='add nr time latents.')
        
        self.parser.add_argument('--project_directory', type=str, default='StructLDMGit.struct_decoder', help='add nr time latents.')

        self.parser.add_argument('--model_module', type=str, default='model_eva', help='add nr time latents.')

        self.parser.add_argument('--verbose', action='store_true', help='add nr time latents.')

        self.parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')

        #self.parser.add_argument('--face_pix', type=int, default=1, help="")

        self.parser.add_argument('--data_config', nargs='+', default="data config")
        self.parser.add_argument('--project_config', nargs='+', default="project config")
        self.parser.add_argument('--method_config', nargs='+', default="method config")

        self.parser.add_argument('--isInfer', action='store_true')

        self.parser.add_argument('--release_mode', action='store_true')
        self.parser.add_argument('--config_dir', type=str, default="")

        
        self.parser.add_argument('--is_crop', action='store_true')
        self.parser.add_argument('--nerf_ratio', default=0, type=int, help='# threads for loading data')

        self.parser.add_argument('--one_layer_mix_style', action='store_true')
        self.parser.add_argument('--not_eva', action='store_true')
        

        self.parser.add_argument('--no_posmap', action='store_true')
        self.parser.add_argument('--use_posmap', action='store_true')
        self.parser.add_argument('--use_nerf', action='store_true')


        self.parser.add_argument('--inference_time', action='store_true')

        
        
        self.parser.add_argument('--sample_all_pixels', action='store_true', help="global nerf sampling")
        self.parser.add_argument('--vrnr', action='store_true', help="global nerf sampling")
        
        rendering.add_argument('--local_only_body', type=bool, default=True, help="sample points only on body local nerf")     
        rendering.add_argument('--use_dilate_model', type=bool, default=True, help="sample points only on body local nerf")
        rendering.add_argument('--use_small_dilation', type=bool, default=True, help="whether multiple identities")
        
        rendering.add_argument('--no_local_nerf', action='store_true', help="global nerf sampling")
        rendering.add_argument('--max_ray_interval', type=float, default=0.25*0.2, help="local sampling interval") 

        rendering.add_argument("--N_samples", type=int, default=28, help='number of samples per ray')
        rendering.add_argument('--not_even', type=bool, default=True, help="sample points only on body local nerf")     
        rendering.add_argument('--dila_dist', type=float, default=0.1)
        rendering.add_argument('--voxel_size', type=float, default=0.005, help="weight") 


        rendering.add_argument('--vrnr_voxel_factor', type=int, default=2, help="")

        rendering.add_argument('--nrays', type=int, default = 1024, help = "number of rays.. not used in vrnr")

        rendering.add_argument('--vrnr_mesh_demo', action='store_true', help="")
        rendering.add_argument('--nerf_mesh_demo', action='store_true', help="")
            
        rendering.add_argument('--check_mesh', action='store_true', help="global nerf sampling")
        rendering.add_argument('--check_can_mesh', action='store_true', help="global nerf sampling")

        rendering.add_argument('--use_sdf_render', action='store_true', help="global nerf sampling")
        rendering.add_argument('--meth_th', type=int, default=6, help="")


        
        rendering.add_argument('--uv_2dplane', action='store_true')
        rendering.add_argument('--plus_uvh_enc', action='store_true')
        rendering.add_argument('--learn_uv', action='store_true')

        rendering.add_argument("--uvVol_smpl_pts", type=int, default=128) #SMPL OR BF


        self.parser.add_argument("--multiview_ids", type=int, nargs="+", default=[0], help="image sizes for the model")

        self.parser.add_argument("--save_name", type=str)
        self.parser.add_argument("--demo_name", type=str, default="")
        self.parser.add_argument("--subdir", type=str, default="")

        self.parser.add_argument('--test_eval', action='store_true')
        self.parser.add_argument('--make_demo', action='store_true')
        self.parser.add_argument('--test_novel_pose', action='store_true')

        self.parser.add_argument('--new_pose', action='store_true')

        self.parser.add_argument("--test_step_size", type=int, default=1) #SMPL OR BF

        self.parser.add_argument("--data_dir", type=str, default='./data')
        self.parser.add_argument("--results_dir", type=str, default='./data/result')

        self.parser.add_argument('--new_view', action='store_true')
        self.parser.add_argument('--new_poseview', action='store_true')
        self.parser.add_argument('--denseframe', action='store_true')

        self.parser.add_argument('--normal_tv', action='store_true')
        self.parser.add_argument('--normal_recon', action='store_true')

        rendering.add_argument('--output_ch', type=int, default=4)

        hypara.add_argument('--tv_latent', action='store_true')

        self.parser.add_argument('--recon_on_mask', action='store_true')
        #self.parser.add_argument('--knn_part', action='store_true')
                
        self.initialized = True

    def merge_config_opt(self, config_file, opt):
        import os
        if not os.path.isfile(config_file): return
        conf = edict(yaml.load(open(config_file), Loader=yaml.SafeLoader))
        for k in conf:
            if isinstance(conf[k], edict):
                for i in conf[k]:
                    opt[k][i] = conf[k][i]
            else:
                setattr(opt, k, conf[k])


    def merge_cfg(self, opt):
        
        if opt.release_mode: 
            opt.config_dir = ""

        #import project config
        for p in opt.project_config:
            self.merge_config_opt(os.path.join(opt.config_dir, "configs/projects/%s" % p), opt)

        from Engine.th_utils.io.prints import printd
        
        #import method config
        for m in opt.method_config:
            self.merge_config_opt(os.path.join(opt.config_dir, "configs/methods/%s" % m), opt)

        for m in opt.data_config:
            self.merge_config_opt(os.path.join(opt.config_dir, "configs/datasets/%s" % m), opt)


    def parse(self):
        
        if not self.initialized:
            self.initialize()
        try:
            args = self.parser.parse_args()
        except: # solves argparse error in google colab
            args = self.parser.parse_args(args=[])

        self.opt = Munch()
        for k in args.__dict__:#basice options
            self.opt[k] = args.__dict__[k] 

        for group in self.parser._action_groups[2:]:
            title = group.title
            self.opt[title] = Munch()
            for action in group._group_actions:
                dest = action.dest
                self.opt[title][dest] = args.__getattribute__(dest)


        self.merge_cfg(self.opt)


        self.opt.name = self.opt.experiment.expname
        self.opt.multi_datasets = [0]
        
        import os

        self.opt.training.checkpoints_dir = os.path.join(self.opt.data_dir, "result/trained_model", self.opt.experiment.expname)

        if self.opt.dataset.ratio != -1:
            self.opt.gen_ratio = self.opt.dataset.ratio

            img_H, img_W = self.opt.dataset.img_H, self.opt.dataset.img_W
            ratio = self.opt.dataset.ratio
            self.opt.model.renderer_spatial_output_dim = (img_H // ratio, img_W // ratio)
        
        self.opt.posenet_setup.tex_latent_dim = self.opt.df.structured_dim
        self.opt.use_posmap = True if not self.opt.no_posmap else False
        self.opt.rendering.use_sdf_render = True
        self.opt.rendering.white_bg = True
        self.opt.use_nerf = True
        self.opt.vrnr = True

        return self.opt
