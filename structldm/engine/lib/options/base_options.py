import argparse
import os
#import util.util as util
import torch

import numpy as np
from munch import *

import sys
sys.path.extend("..")
#from configs.vrnr_setup import vrnr_init, vrnr_parse
from .vrnr_setup import vrnr_init, vrnr_parse
from configs.config_util import yaml_config, merge_config_opt

from uvm_lib.engine.thutil.io.prints import *

from easydict import EasyDict as edict


class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False
    
    def initialize(self):
        # experiment specifics


        self.parser.add_argument('--name', type=str, default='label2city',
                                 help='name of the experiment. It decides where to store samples and models')
        self.parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        self.parser.add_argument('--checkpoints_dir', type=str, default='../data/result/trained_model', help='models are saved here')
        
        self.parser.add_argument('--subdir', type=str, default='', help='subdir')
        
        self.parser.add_argument('--model', type=str, default='P_cond', help='which model to use')
        self.parser.add_argument('--norm', type=str, default='instance',
                                 help='instance normalization or batch normalization')
        self.parser.add_argument('--use_dropout', action='store_true', help='use dropout for the generator')
        self.parser.add_argument('--data_type', default=32, type=int, choices=[8, 16, 32],
                                 help="Supported data type i.e. 8, 16, 32 bit")
        self.parser.add_argument('--verbose', action='store_true', default=False, help='toggles verbose')
        self.parser.add_argument('--fp16', action='store_true', default=False, help='train with AMP')
        #self.parser.add_argument('--local_rank', type=int, default=0, help='local rank for distributed training')


        self.parser.add_argument('--log_dir', type=str, default='../data/result/trained_model', help='add nr time latents.')

        
        self.parser.add_argument('--uv_2dplane', action='store_true')
        self.parser.add_argument('--plus_uvh_enc', action='store_true')

        self.parser.add_argument('--texlat_init_size', default=128, type=int)

        self.parser.add_argument('--face_id_loss', action='store_true', help='smooth real label in GAN')

        self.parser.add_argument('--scale_latent_128', action='store_true', help='smooth real label in GAN')

        self.parser.add_argument('--one_superreso_block', action='store_true', help='smooth real label in GAN')

        self.parser.add_argument('--white_bg', action='store_true', help='smooth real label in GAN')
        self.parser.add_argument('--smooth_label', action='store_true', help='smooth real label in GAN')

        # input/output sizes
        self.parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
        self.parser.add_argument('--loadSize', type=int, default=1024, help='Output Size')
        self.parser.add_argument('--fineSize', type=int, default=512, help='then crop to this size')
        self.parser.add_argument('--label_nc', type=int, default=35, help='# of input label channels')
        self.parser.add_argument('--input_nc', type=int, default=16, help='# of input image channels')
        self.parser.add_argument('--output_nc', type=int, default=3, help='# of output image channels')


        self.parser.add_argument('--body_sample_ratio', type=float, default=0.5, help='number of clusters for features')
        self.parser.add_argument('--face_sample_ratio', type=float, default=0, help='number of clusters for features')
        
        # for setting inputs
        self.parser.add_argument('--dataset', type=str, default='fashion')

        self.parser.add_argument('--dataroot', type=str, default='./datasets/cityscapes/')
        self.parser.add_argument('--resize_or_crop', type=str, default='scale_width',
                                 help='scaling and cropping of images at load time [resize_and_crop|crop|scale_width|scale_width_and_crop]')
        self.parser.add_argument('--serial_batches', action='store_true',
                                 help='if true, takes images in order to make batches, otherwise takes them randomly')
        self.parser.add_argument('--no_flip', action='store_true',
                                 help='if specified, do not flip the images for data argumentation')
        self.parser.add_argument('--nThreads', default=4, type=int, help='# threads for loading data')
        self.parser.add_argument('--max_dataset_size', type=int, default=float("inf"),
                                 help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')

        self.parser.add_argument('--max_iters_epoch', type=int, default=500,
                            help='Maximum number of samples each epoch.')        
        self.parser.add_argument('--iters_each_sample', type=int, default=15,
                                 help='optimize 10 iterations for each image.')

        self.parser.add_argument('--update_nr_time_lat', action='store_true',
                                 help='add nr time latents.')

        self.parser.add_argument('--nr_time_lat_loc', type=str, default = "bottom", help='bottom: in latent code, or: top: when predict RGB.')
        
        self.parser.add_argument('--use_nr_time_latent', action='store_true',
                                 help='add nr time latents.')

        self.parser.add_argument('--use_nerf_time_latent', action='store_true',
                                 help='add nerf time latents.')
        
        self.parser.add_argument('--debug_small_data', action='store_true',
                            help='debug on small dataset.')

        # for displays
        self.parser.add_argument('--display_winsize', type=int, default=256, help='display window size')
        self.parser.add_argument('--tf_log', action='store_true',
                                 help='if specified, use tensorboard logging. Requires tensorflow installed')

        self.parser.add_argument('--fuse_mode', type=int, default=0,
                            help='fuse nerf features with nr, 0: att; 1, max; 2: cat.')
                
        # for generator
        self.parser.add_argument('--netG', type=str, default='global', help='selects model to use for netG')
        self.parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in first conv layer')
        self.parser.add_argument('--n_downsample_global', type=int, default=4,
                                 help='number of downsampling layers in netG')
        self.parser.add_argument('--n_blocks_global', type=int, default=3,
                                 help='number of residual blocks in the global generator network')
        self.parser.add_argument('--n_blocks_local', type=int, default=3,
                                 help='number of residual blocks in the local enhancer network')
        self.parser.add_argument('--n_local_enhancers', type=int, default=1, help='number of local enhancers to use')
        self.parser.add_argument('--niter_fix_global', type=int, default=0,
                                 help='number of epochs that we only train the outmost local enhancer')

        # for instance-wise features
        self.parser.add_argument('--no_instance', action='store_true',
                                 help='if specified, do *not* add instance map as input')
        self.parser.add_argument('--instance_feat', action='store_true',
                                 help='if specified, add encoded instance features as input')
        self.parser.add_argument('--label_feat', action='store_true',
                                 help='if specified, add encoded label features as input')
        self.parser.add_argument('--feat_num', type=int, default=3, help='vector length for encoded features')
        self.parser.add_argument('--load_features', action='store_true',
                                 help='if specified, load precomputed feature maps')
        self.parser.add_argument('--n_downsample_E', type=int, default=4, help='# of downsampling layers in encoder')
        self.parser.add_argument('--nef', type=int, default=16, help='# of encoder filters in the first conv layer')
        self.parser.add_argument('--n_clusters', type=int, default=10, help='number of clusters for features')

        self.parser.add_argument('--n_dptex_s', type=int, default=64, help='number of clusters for features')
        self.parser.add_argument('--lambda_tex', type=float, default=0, help='number of clusters for features')
        self.parser.add_argument('--lambda_rerender', type=float, default=0, help='number of clusters for features')
        self.parser.add_argument('--lambda_L1', type=float, default=0.5, help='number of clusters for features')
        self.parser.add_argument('--lambda_contrastive', type=float, default=0, help='number of clusters for features')
        self.parser.add_argument('--use_face', action='store_true', help='if specified, use pretrained face loss')
        self.parser.add_argument('--face_model', type=str, default='../asset/spretrains/sphere20a_20171020.pth')
        self.parser.add_argument('--lambda_face', type=float, default=5, help='number of clusters for features')
        self.parser.add_argument('--test_source', type=str)
        self.parser.add_argument('--inpaint_path', type=str)
        self.parser.add_argument('--renderer', type=str, default='dp_lookup')
        self.parser.add_argument('--pad_input', action='store_true')

        self.parser.add_argument('--multiview_ids', type=str, default='0', help='multiview list: e.g. 0  0,1,2, 0,2.')
        self.parser.add_argument('--bbx_w', type=int, default='0', help='max bbox width')
        self.parser.add_argument('--bbx_h', type=int, default='0', help='multiview list: e.g. 0  0,1,2, 0,2.')
        self.parser.add_argument('--pad_size', type=int, default='0', help='multiview list: e.g. 0  0,1,2, 0,2.')

        self.parser.add_argument('--texture_stack', action='store_true')
        self.parser.add_argument('--texture_stack_fixed', action='store_true')

        # self.parser.add_argument('--direct_sampling', action='store_true')
        self.parser.add_argument('--style_transfer', action='store_true', help='whether include style transfer net.')
        self.parser.add_argument('--texture_stack_channels', type=int, default=3)  # 3 channels

        self.parser.add_argument('--texture_stack_lr_factor', type=float, default=10.0)  # 3 channels

        self.parser.add_argument('--fix_texture_stack', action='store_true')  # 3 channels

        self.parser.add_argument('--render_with_dp_label', action='store_true')  # 3 channels
        self.parser.add_argument('--random_drop_prob', type=float, default=0.05, help='the probability to randomly drop each pose segment during training')

        self.parser.add_argument('--data_aug', action='store_true')  # 3 channels
        self.parser.add_argument('--direct_phase', type=str, default=' phase in direct rendering')

        self.parser.add_argument('--not_norm_label', action='store_true')  # 3 channels

        self.parser.add_argument('--not_data_aug', action='store_true')  # no data aug

        self.parser.add_argument('--dataset_name', type=str, default=' phase in direct rendering')


        self.parser.add_argument('--pix2pix', action='store_true', help='dp to avatar')

        
        self.parser.add_argument('--old_data_aug', action='store_true', help='no ego texture')
        self.parser.add_argument('--old_tex_stack', action='store_true', help='dp to avatar')

        #self.parser.add_argument('--std_data_aug', action='store_true', help='no ego texture')

        self.parser.add_argument('--debug_out', action='store_true', help='no ego texture')

        self.parser.add_argument('--fix_rgb', action='store_true', help='no ego texture')

        self.parser.add_argument('--direct_ego', action='store_true', help='outside texture')

        self.parser.add_argument('--direct_rendernet', action='store_true', help='no ego texture')
        self.parser.add_argument('--no_label', action='store_true', help='outside texture')

        self.parser.add_argument('--views_in_training', type=int, default=1, help='view number in training')

        self.parser.add_argument('--direct_sample', action='store_true')

        
        self.parser.add_argument('--niter', type=int, default=30, help='# of iter at starting learning rate')
        self.parser.add_argument('--niter_decay', type=int, default=0, help='# of iter to linearly decay learning rate to zero')
        
        self.parser.add_argument('--train_part', action='store_true')
        self.parser.add_argument('--add_render', action='store_true')
        self.parser.add_argument('--part_num', type=int, default=24)

        self.parser.add_argument('--rotate', action='store_true')

        self.parser.add_argument('--ego_aug', action='store_true')
        self.parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')

        self.parser.add_argument('--make_overview', action='store_true')
        self.parser.add_argument('--overview_id', type=int, default=234)

        self.parser.add_argument('--noresize', action='store_true')
        self.parser.add_argument('--w_tv', type=float, default=0.0001, help="weight")


        self.parser.add_argument('--geonr', action='store_true')

        self.parser.add_argument('--style_uvdim', type=int, default=128)

        
        self.parser.add_argument('--debug', action='store_true', help="debug")
        self.parser.add_argument('--debug_nr', action='store_true', help="debug")
                
        self.parser.add_argument('--D_UV', action='store_true')
                
        
        self.parser.add_argument('--pred_nerf_depth', action='store_true')

        self.parser.add_argument('--resize_nerf', action='store_true')
        
        self.parser.add_argument('--w_nr_pred_smpl_uv', type=float, default=2.0)
        self.parser.add_argument('--w_nr_pred_smpl_normal', type=float, default=0.3)
        self.parser.add_argument('--w_nr_pred_smpl_depth', type=float, default=2.0)
        
        self.parser.add_argument('--w_nr_pred_nerf_depth', type=float, default=2.0)
        
        self.parser.add_argument('--nr_insert_smpl_depth', action='store_true')                
        self.parser.add_argument('--nr_insert_smpl_uv', action='store_true')
        self.parser.add_argument('--nr_insert_smpl_normal', action='store_true')
        self.parser.add_argument('--nr_pred_smpl_normal', action='store_true')
        self.parser.add_argument('--nr_pred_smpl_uv', action='store_true')
        
        self.parser.add_argument('--nr_pred_smpl_depth', action='store_true')
        
        self.parser.add_argument('--nr_pred_real_normal', action='store_true')
        

        self.parser.add_argument('--w_nr_pred_mask', type=float, default=3.0)
                
        self.parser.add_argument('--nr_adduvlayer', action='store_true')
        
        self.parser.add_argument('--clone_nerf', action='store_true')
        self.parser.add_argument('--detach_nerf', action='store_true')
        

        #smpl_D NeRF
        
        self.parser.add_argument('--nr_pose_dep_uv', action='store_true',  help="use pose dependent style, features from conved posemap")
        self.parser.add_argument('--nr_static_uv', action='store_true', help="use uv latent in neural rendering")

        self.parser.add_argument('--loss_smpld_normal_smoothness', action='store_true')

        #self.parser.add_argument('--loss_rot_posmap_feat', action='store_true', help="rot posmap feature, 3D from 2D")
        #ablation study
        # self.parser.add_argument('--loss_rot_normal', action='store_true', help="smpl normal loss in posmap")
        self.parser.add_argument('--smpld_nr', action='store_true')

        self.parser.add_argument('--multi_identity_list', nargs='+', required = False)
        self.parser.add_argument('--dataset_id', nargs='*',  default=[])
        
        self.parser.add_argument('--gender', type=str, default="neutral") 

        self.parser.add_argument('--ratio', type=float, default=0.5, help="")
        
        self.parser.add_argument('--debug_mode', action='store_true')
        
        self.parser.add_argument('--denseframe', action='store_true', help="dense sampling frames") 
        self.parser.add_argument('--new_pose', action='store_true', help="test new pose seen view") 
        self.parser.add_argument('--new_view', action='store_true', help="test known pose unseen view") 
        self.parser.add_argument('--new_poseview', action='store_true', help="new pose new view")
                           
        self.parser.add_argument('--only_opt_mask', action='store_true')
        
        self.parser.add_argument('--debug_mask_opt', action='store_true')
        
        self.parser.add_argument('--w_tv_uvmap', type=float, default=0.0, help="total variation loss to supervise mask")
        self.parser.add_argument('--w_uvmap', type=float, default=0.0, help="does the same as w_reg_offset")
                
        self.parser.add_argument('--not_blur_nerf', action='store_true')
        
        self.parser.add_argument('--opt_static_model', action='store_true')

        
        self.parser.add_argument('--img_size', nargs='*', default=256, help="")


        self.parser.add_argument('--output_smpld_mesh', action='store_true')
        
        self.parser.add_argument('--test313_pose', action='store_true')
        
        
        #genderator        
        self.parser.add_argument('--generator_input_dim', type=int, default = 3, help="")        
            
        #uv nerf
        self.parser.add_argument('--nerf_uvh_feat_dim', type=int, default = 3, help="")        
        self.parser.add_argument('--nerf_output_rgb_dim', type=int, default = 3, help="")        
        self.parser.add_argument('--add_layer_density_color', action='store_true', help="")
        self.parser.add_argument('--add_layer_geometry', action='store_true', help="")
                                        
        self.parser.add_argument('--uvVol_2d', action='store_true',  help="no posmap feature loss")
        self.parser.add_argument('--uvVol_3d', action='store_true',  help="")
        self.parser.add_argument('--uvDepth', type=int, default=512, help="in 3d uv volume")
        
        self.parser.add_argument('--uvvol3dDim', type=int, default=1024, help="in 3d uv volume")
        
        #neural rendering
        #self.parser.add_argument('--add_uv_coord', action='store_true',  help="feature loss")


        self.parser.add_argument('--norm_in_feat', action='store_true', help="predict norm in posmap feature")
                
        self.parser.add_argument('--mlp_norm_offset', action='store_true', help = "query (feat, uv) to get norm and offset")
        self.parser.add_argument('--mlp_norm', action='store_true', help = "seperate norm and offest")
        self.parser.add_argument('--mlp_offset', action='store_true', help = "seperate norm and offest")
        #self.parser.add_argument('--rot_posmap', action='store_true')
        
        
        self.parser.add_argument('--num_D', type=int, default=2, help='number of discriminators to use')
        self.parser.add_argument('--n_layers_D', type=int, default=3, help='only used if which_model_netD==n_layers')
        self.parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in first conv layer')    
        self.parser.add_argument('--lambda_feat', type=float, default=10.0, help='weight for feature matching loss')                
        self.parser.add_argument('--no_ganFeat_loss', action='store_true', help='if specified, do *not* use discriminator feature matching loss')
        self.parser.add_argument('--no_vgg_loss', action='store_true', help='if specified, do *not* use VGG feature matching loss')        
        self.parser.add_argument('--no_lsgan', action='store_true', help='do *not* use least square GAN, if false, use vanilla GAN')
        self.parser.add_argument('--pool_size', type=int, default=0, help='the size of image buffer that stores previously generated images')

        self.parser.add_argument('--train_view', type=str, default = "", help="")
        self.parser.add_argument('--test_view', type=str, default = "", help="")

        self.parser.add_argument('--old_scheme', action='store_true', help="old training scheme")
                
        self.parser.add_argument('--aug_nr', action='store_true', help="predict norm in posmap feature")
        self.parser.add_argument('--small_rot', action='store_true', help="predict norm in posmap feature")
        
        self.parser.add_argument('--multi_datasets', type=str, default="")        

        
        self.parser.add_argument('--dataset_basedir', type=str, default="")        
                
        self.parser.add_argument('--rot_input_normal', action='store_true', help="uv + conv3d nerf")
                
        self.parser.add_argument('--conv3d_rot_scale', type=float, default=1.0)        
        
        self.parser.add_argument('--unif_conv', action='store_true', help="")
        
        self.parser.add_argument('--PosNetNum', type=int, default=3)     
        self.parser.add_argument('--PosNetNumRes', type=int, default=3)
        
        self.parser.add_argument('--sysmetric_uvlat', action='store_true', help="")
        
        self.parser.add_argument('--vis_lat', action='store_true', help="")
         
        self.parser.add_argument('--add_epoch', type=int, default=0)
            
        
        self.parser.add_argument('--dila_dist', type=float, default=0.06)
        
        self.parser.add_argument('--debug_train', action='store_true', help="")
                
        
        self.parser.add_argument('--train_nb', action='store_true', help="")
        
        
        self.parser.add_argument('--depth_layer', type=int, default=1, help="")
        
        self.parser.add_argument('--face_pix', type=int, default=1, help="")
        
        self.parser.add_argument('--is_crop', action='store_true', help="")
        self.parser.add_argument('--x0_crop', type=int, default=0, help="")
        self.parser.add_argument('--x1_crop', type=int, default=0, help="")
        self.parser.add_argument('--y0_crop', type=int, default=0, help="")
        self.parser.add_argument('--y1_crop', type=int, default=0, help="")
        
        self.parser.add_argument('--img_H', type=int, default=0, help="")
        self.parser.add_argument('--img_W', type=int, default=0, help="")
        
        self.parser.add_argument('--train_dress', action='store_true', help="")
        
        
        self.parser.add_argument('--vrnr_mesh_demo', action='store_true', help="")
        
        self.parser.add_argument('--nerf_mesh_demo', action='store_true', help="")
        
        self.parser.add_argument('--no_face', action='store_true', help="")
        
        self.parser.add_argument('--meth_th', type=int, default=6, help="")
        self.parser.add_argument('--vrnr_voxel_factor', type=int, default=2, help="")
        
        


        self.parser.add_argument('--inference_time', action='store_true', help="")
        self.parser.add_argument('--inference_time_sampled', type=int, default=4, help="")
        
        self.parser.add_argument('--is_inference', action='store_true', help="")

        self.parser.add_argument('--ab_latent', action='store_true', help="")
        
        self.parser.add_argument('--shape_editing', type=str, default="", help="fat, thin, tall, short, thin2fat,short2tall")
        
        self.parser.add_argument('--shape_interp_thin2fat', action='store_true', default="")
        self.parser.add_argument('--shape_interp_short2tall', action='store_true', default="")
        
        self.parser.add_argument('--free_view_fly_camera', action='store_true', help="")
        self.parser.add_argument('--static_scene', action='store_true', help="")
                
        self.parser.add_argument('--free_view_rot_smpl', action='store_true', help="")
        #self.parser.add_argument('--rotate_smpl', action='store_true', help="")
        
        self.parser.add_argument('--free_view_interp', action='store_true', help="")
        
        self.parser.add_argument('--free_view_num_interval', type = int, default = 5, help="")
        
        self.parser.add_argument('--view_interp_start', nargs='+', required = False)
        self.parser.add_argument('--view_interp_end', nargs='+', required = False)
        
        self.parser.add_argument('--test_ext_pose', action='store_true', help="")
        
        self.parser.add_argument('--test_ext_pose_mode', type=int, default = 0, help="")
        
        self.parser.add_argument('--vis_skeleton', action='store_true', help="")
        
        self.parser.add_argument('--freeview_mid', action='store_true', help="")
        
        self.parser.add_argument('--learn_uv', action='store_true', help="")        
        self.parser.add_argument('--vis_uv', action='store_true', help="")

        self.parser.add_argument('--dual_d', action='store_true', help="+ nerf feature in Discrim")
        self.parser.add_argument('--remove_tex', action='store_true', help="")

        self.parser.add_argument('--learn_uv_feature', action='store_true', help="")
 
        self.parser.add_argument('--direct_uv_feature', action='store_true', help="")

        self.parser.add_argument('--sample_high_reso', action='store_true', help="")

        self.parser.add_argument('--w_multi_nerf', type=float, default = 15.0, help="")

        self.parser.add_argument('--use_transformer', action='store_true', help="")

        self.parser.add_argument('--swap_uv', type=int, default = 0, help="")

        self.parser.add_argument('--is_insert_id_latent', action='store_true', help="")

        self.parser.add_argument('--use_nerf_uv', action='store_true', help="")
 
        self.parser.add_argument('--shape_memory_uv_ada', action='store_true', help="")
        self.parser.add_argument('--shape_memory_layer', type=int, default = 1, help="")
        self.parser.add_argument('--mem_fuse_mode', type=int, default = 0, help="0: none; 1: mean; 2:max; 3: att")
        
        self.parser.add_argument('--shape_memory_uv', action='store_true', help="")
        self.parser.add_argument('--shape_memory_size', type=int, default = 64, help="")
        self.parser.add_argument('--shape_memory_chan', type=int, default = 32, help="")

        self.parser.add_argument('--not_old', action='store_true', help="")

        self.parser.add_argument('--pred_uv_full', action='store_true', help="")


        self.parser.add_argument('--img_encoder_dim', type=int, default = 16, help="")

        self.parser.add_argument('--source_cam_id', type=int, help="", default=0)

        self.parser.add_argument('--img_encoder_loss', action='store_true', help="")
        


        self.parser.add_argument('--render_avg_uv', action='store_true', help="")
        self.parser.add_argument('--uv_view_num', type=int, default = 1, help="how many views to be averaged in uv")

        self.parser.add_argument('--input_tgt_view', action='store_true', help="")

        self.parser.add_argument('--online_uv_img', action='store_true', help="")
        self.parser.add_argument('--test_views', type=str, default="0", help="")

        self.parser.add_argument('--random_num', action='store_true', help="take random number of images")

        self.parser.add_argument('--use_pixel_aligned_img', action='store_true', help="pixel aligned features")

        self.parser.add_argument('--use_style_gan_gen', action='store_true', help="use style gan generator")

        self.parser.add_argument('--direct_1', action='store_true', help="direct input view")


        self.parser.add_argument('--ab_no_normal_loss', action='store_true', help="direct input view")
        self.parser.add_argument('--ab_no_tex_latent', action='store_true', help="direct input view")
        self.parser.add_argument('--ab_1c_geometry', action='store_true', help="direct input view")
        self.parser.add_argument('--ab_pose_para', action='store_true', help="direct input view")

        self.parser.add_argument('--uvh_feat_dim', type=int, default = 1)


        self.parser.add_argument('--ab_Dism', action='store_true', help="no disciminator or loss is zero")

        self.parser.add_argument('--ab_no_geoLat', action='store_true', help="no geo latent")

        self.parser.add_argument('--cat_pos_geo_tex', action='store_true', help="directly concatentate")

        self.parser.add_argument('--ab_cat_poseEd_nerf', action='store_true', help="no tex encoder")

        self.parser.add_argument('--ab_D_on_nerf', action='store_true', help="condition D on nerf")

        self.parser.add_argument('--UV_NeRF', action='store_true', help="hash nerf")

        
        self.parser.add_argument('--debug_val', action='store_true', help="hash nerf by uv")

      
        self.parser.add_argument('--total_parallel', type=int, default = 1, help="parallel in test or evaluation")
        self.parser.add_argument('--pid', type=int, default = 1, help="parallel id")

        self.parser.add_argument('--add_tex_noise', action='store_true', help="add texture noise")
        self.parser.add_argument('--noise_sigma', type=float, default = 0.01, help="pose noise")

        self.parser.add_argument('--wrong_input_dir', type=str, default = "", help="")


        self.parser.add_argument('--test_novel_pose', action='store_true', help="add texture noise")


        training = self.parser.add_argument_group('training')
        training.add_argument('--data_parallel', action='store_true', help="data parallel")
        training.add_argument('--distributed', action='store_true', help="distributed data parallel")

        self.parser.add_argument('--gpu', default=None, type=int)
        self.parser.add_argument('--world_size', default=-1, type=int, 
                            help='number of nodes for distributed training')
        self.parser.add_argument('--rank', default=-1, type=int, 
                            help='node rank for distributed training')
        self.parser.add_argument('--dist_url', default='env://', type=str, 
                            help='url used to set up distributed training')
        self.parser.add_argument('--dist_backend', default='nccl', type=str, 
                            help='distributed backend')
        self.parser.add_argument('--local_rank', default=-1, type=int, 
                            help='local rank for distributed training')
        self.parser.add_argument('--gpu_num', default=0, type=int)
        


        self.parser.add_argument('--dataset_res_name', default="", type=str)


        self.parser.add_argument('--nr_pred_mask', action='store_true')
        self.parser.add_argument('--masked_loss', action='store_true')

        self.parser.add_argument('--use_smpl_scaling', action='store_true')

        self.parser.add_argument('--fps', default=25, type=int)

        self.parser.add_argument('--dilation_in_smpl', action='store_true')

        self.parser.add_argument('--force_evaluation', action='store_true', help="overwrite previous evaluation results")


        #configs
        self.parser.add_argument('--config', nargs='+', default="data config")
        self.parser.add_argument('--project_config', nargs='+', default="project config")
        self.parser.add_argument('--method_config', nargs='+', default="method config")
        self.parser.add_argument('--config_dir', nargs='+', default="")
        self.parser.add_argument('--default_config', type=str, default="configs/defaults.yml")

        self.parser.add_argument("--is_pad_img", action='store_true', help='pad rectangle image to square')

        #release
        self.parser.add_argument('--release_mode', action='store_true', help="release mode")

        self.set_methods()

        #--superreso "StyleSup" "Eg3dSup" "GeneralSup"
        ##from .motion_setup import set_motion
        #set_motion(self.parser)

        self.set_sdf()
        self.set_posenet()

        vrnr_init(self.parser)
        self.patch_nerf()
        self.hash_nerf()

        self.setup_hypara()
        self.debug_setup()
        self.network_setup()

        self.demo_setup()

        self.project_setup()
        

        self.initialized = True

    def project_setup(self):
        #self.parser.add_argument('--option_module', type=str, default="", help="")
        self.parser.add_argument('--project_directory', type=str, default="", help="")
        self.parser.add_argument('--model_module', type=str, default="", help="")
        
        self.parser.add_argument('--posenet_module', type=str, default="", help="")
        self.parser.add_argument('--nerf_module', type=str, default="", help="")

    def set_methods(self):

        self.parser.add_argument('--uv_3d_nerf', action='store_true', help="patch nerf")
        self.parser.add_argument('--patch_nerf', action='store_true', help="patch nerf")
        self.parser.add_argument('--uv_conv3d_nerf', action='store_true', help="uv + conv3d nerf")
                
        self.parser.add_argument('--uv_2d_vol', action='store_true', help="uv + conv3d nerf")
        self.parser.add_argument('--uv_3d_vol', action='store_true', help="uv + conv3d nerf")
    
        
        self.parser.add_argument('--uv_mvp', action='store_true', help="")

        self.parser.add_argument('--uv3dNR', action='store_true', help="")
        
        self.parser.add_argument('--dnr3d', action='store_true', help="")
        
        self.parser.add_argument('--vrnr', action='store_true', help="vrnr")
        self.parser.add_argument('--ab_nonerf', action='store_true', help="ablation study, no nerf")
        self.parser.add_argument('--img_tel', action='store_true', help="")

        self.parser.add_argument('--motion_mode', action='store_true', help="learn motions")
        self.parser.add_argument('--ab_cond_pose_time', action='store_true', help="learn motions")
        
        
        self.parser.add_argument('--Hash_NeRF', action='store_true', help="dnr compare")

        self.parser.add_argument('--uv_hash', action='store_true', help="hash nerf by uv")

    def demo_setup(self):
        
        self.parser.add_argument("--demo_real_seq", action='store_true', help='input is real sequence')
        self.parser.add_argument("--demo_pose_dir", type=str, help='demo real pose dir')
        self.parser.add_argument("--demo_croped", action='store_true', help='input is real sequence')
        self.parser.add_argument("--save_demos_attach_name", type=str, default=None)

        self.parser.add_argument('--demo_all', action='store_true', help="")

        self.parser.add_argument('--demo_frame', type=str, default="", help="")

        self.parser.add_argument('--demo_view', type=str, default = "", help="")
        
        self.parser.add_argument('--demo_vid', type=str, default ="", help="")
        self.parser.add_argument('--demo_frame_id', type=str, default ="", help="1,2,3")

        self.parser.add_argument('--make_demo', action='store_true', help="")
        self.parser.add_argument('--demo_name', type=str, default="", help="")

    def debug_setup(self):
        self.parser.add_argument('--debug_only_nerf', action='store_true', help="only debug nerf")
        self.parser.add_argument('--debug_df',  action='store_true')
        self.parser.add_argument('--debug_one_image',  action='store_true')
        self.parser.add_argument('--debug_only_enc',  action='store_true')
        self.parser.add_argument('--test_overfitting', action='store_true', help="test overfitting")

    def set_sdf(self):
        sdf_setup = self.parser.add_argument_group('sdf_setup')
        sdf_setup.add_argument("--eikonal_lambda", type=float, default=0.5, help='')
        sdf_setup.add_argument("--min_surf_lambda", type=float, default=1.5, help='')
        sdf_setup.add_argument("--min_surf_beta", type=float, default=100, help='')

        sdf_setup.add_argument('--not_deltasdf',  action='store_true')

    def set_posenet(self):

        posenet_setup = self.parser.add_argument_group('posenet_setup')

        posenet_setup.add_argument('--uv_type', type=str, default="SMPL")

        posenet_setup.add_argument("--posenet_outdim", type=int, default=64, help='')
        posenet_setup.add_argument('--uv_reso', type=int, default=128)
        posenet_setup.add_argument("--tex_latent_dim", type=int, default=16, help='')

        posenet_setup.add_argument("--size_motion_window", type=int, default=5, help='')

        posenet_setup.add_argument("--pred_texture_dim", type=int, default=16, help='')

        posenet_setup.add_argument("--velocity", type=int, default=1, help="velocity, step size on dataset")
        posenet_setup.add_argument("--c_velo", action='store_true', help='with c_velo')
        posenet_setup.add_argument("--c_acce", action='store_true', help='trajectory')
        posenet_setup.add_argument("--c_traj", action='store_true', help='trajectory')


        posenet_setup.add_argument("--new_dynamics", action='store_true', help='new data dynamics')


        posenet_setup.add_argument("--ab_c_norm", action='store_true', help='condition on normal')
        posenet_setup.add_argument("--ab_c_v10", action='store_true', help='condition on past 10 frames')
        posenet_setup.add_argument("--ab_baseline_motionapp", action='store_true', help='condition on past 10 frames')

        posenet_setup.add_argument("--w_pred_pose_uv", type=float, default=1.0, help='nerf rec weight')
        posenet_setup.add_argument("--w_pred_velocity_uv", type=float, default=1.0, help='nerf rec weight')
        posenet_setup.add_argument("--w_pred_normal_uv", type=float, default=1.0, help='nerf rec weight')
        posenet_setup.add_argument("--w_pred_texture_uv", type=float, default=1.0, help='nerf rec weight')



        posenet_setup.add_argument("--w_rot_normal", type=float, default=1.0, help='nerf rec weight')
        posenet_setup.add_argument("--w_posmap_feat", type=float, default=0, help='nerf rec weight')
        posenet_setup.add_argument('--w_rot_offset', type=float, default=0.0, help="")
        posenet_setup.add_argument('--w_offset_normal_smoothes', type=float, default=0.0, help="")
        posenet_setup.add_argument('--w_offset_reg', type=float, default=0.0, help="")
        posenet_setup.add_argument('--w_img_encoder_loss', type=float, default = 1.0, help="")
        posenet_setup.add_argument('--w_img_encoder_loss_rot', type=float, default = 1.0, help="")


        posenet_setup.add_argument('--pred_current_state', action='store_true',  help="predict current state")

        posenet_setup.add_argument('--pred_texture_uv', action='store_true',  help="predict full texture in uv")
        posenet_setup.add_argument('--pred_normal_uv', action='store_true',  help="no posmap feature loss")
        posenet_setup.add_argument('--pred_offset_uv', action='store_true',  help="no posmap feature loss")
        posenet_setup.add_argument("--pred_pose_uv", action='store_true', help='pred next pose')
        posenet_setup.add_argument("--pred_velocity_uv", action='store_true', help='next trajectory')

        #--pred_normal_uv_rot --pred_pose_uv_rot --pred_velocity_uv_rot
        posenet_setup.add_argument('--pred_normal_uv_rot', action='store_true',  help="no posmap feature loss")        
        posenet_setup.add_argument("--pred_pose_uv_rot", action='store_true', help='pred next pose')
        posenet_setup.add_argument("--pred_velocity_uv_rot", action='store_true', help='next trajectory')


        posenet_setup.add_argument("--ab_pred_pose_by_velocity", action='store_true', help='pred velocity then get pose')
        posenet_setup.add_argument("--ab_sparse_pose", action='store_true', help='predict the offsets of vertex in 3d instead of normal')

        posenet_setup.add_argument('--rot_posmap', action='store_true')
        posenet_setup.add_argument('--conv_norm', action='store_true', help="conv norm as supervision, output 3 channel")
        posenet_setup.add_argument('--conv_offset', action='store_true', help="conv offset, output 3 channel")
       
        posenet_setup.add_argument('--learn_3d', action='store_true',  help="no posmap feature loss")
        posenet_setup.add_argument('--learn_local', action='store_true',  help="feature loss")

        posenet_setup.add_argument('--combine_pose_style', action='store_true',  help="combine the pose and style latent in posmap conv")        
        posenet_setup.add_argument('--use_style', action='store_true',  help="whether use a sepearate style part")              
        posenet_setup.add_argument('--smooth_latent', action='store_true',  help="a shallow net to smooth style uv latent")
        posenet_setup.add_argument('--feat_all', action='store_true',  help="")        
        
        posenet_setup.add_argument('--only_img_enc', action='store_true', help="")


        posenet_setup.add_argument('--pos_input_normal', action='store_true', help="")


        posenet_setup.add_argument('--use_img_encoder', action='store_true', help="")
        posenet_setup.add_argument('--use_img_pose_enc', action='store_true', help="")
        posenet_setup.add_argument('--rot_img_pose', action='store_true', help="")

        posenet_setup.add_argument('--posemap_down', type=int, default=3)
        posenet_setup.add_argument('--posemap_resnet', type=int, default=3)

        posenet_setup.add_argument('--input_norm_uv', action='store_true', help="")
        posenet_setup.add_argument('--posmap_rot_scale', type=float, default=1.0)

        #only_rot_pose, rot_all_same, rot_all_diff
        posenet_setup.add_argument('--only_rot_pose', action='store_true', help="")
        posenet_setup.add_argument('--rot_all_same', action='store_true', help="")
        posenet_setup.add_argument('--rot_all_diff', action='store_true', help="")


        posenet_setup.add_argument('--ab_PoseTime_XYZ', action='store_true', help="")


    def renderer_setup(self):
        t = 0
        

    def network_setup(self):
        self.parser.add_argument('--use_sdf_render', action='store_true', help="distributed data parallel")
        self.parser.add_argument('--superreso', type=str, default = "", help="StyleSup, Eg3dSup, GeneralSup, LightSup")

        self.parser.add_argument('--use_posmap', action='store_true',  help="whether use posmap net, geometry part")


    def setup_hypara(self):
    
        hypara = self.parser.add_argument_group('hypara')

        hypara.add_argument("--gamma_lb", type=int, default=20, help='weight of Discriminator')
        hypara.add_argument("--adjust_gamma", action='store_true', help='uvh tri-plane')

        hypara.add_argument("--nerf_decay", action='store_true', help='weight of nerf decay')
        

        hypara.add_argument("--mixing", type=float, default=0.9, help="probability of latent code mixing")

        hypara.add_argument("--w_posmap", type=float, default=1.0, help='nerf rec weight')

        hypara.add_argument("--w_D", type=float, default=1.0, help='weight of Discriminator')
        hypara.add_argument("--w_D_grad", type=float, default=1.0, help='grad penalty in discrim')

        hypara.add_argument("--w_G_L1", type=float, default=0.5, help='weight of G_GAN')
        hypara.add_argument("--w_G_feat", type=float, default=10, help='weight of G_GAN')

        hypara.add_argument("--w_G_Mask", type=float, default=1.0, help='mask loss of G_GAN')
        
        hypara.add_argument("--w_G_GAN", type=float, default=1.0, help='weight of G_GAN')
        hypara.add_argument("--w_Face", type=float, default=5.0, help='face weight')
        hypara.add_argument("--w_tex_rec", type=float, default=3.0, help='weight of tec rec')
        hypara.add_argument("--w_nerf_rec", type=float, default=20.0, help='nerf rec weight')

        hypara.add_argument("--gaussian_weighted_sampler", action='store_true', help='pad rectangle image to square')

        hypara.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')


    def hash_nerf(self):
        #parser = argparse.ArgumentParser()
        #parser.add_argument('path', type=str)
        #parser.add_argument('-O', action='store_true', help="equals --fp16 --cuda_ray --preload")
        #parser.add_argument('--test', action='store_true', help="test mode")
        #parser.add_argument('--workspace', type=str, default='workspace')
        #parser.add_argument('--seed', type=int, default=0)

        #self.parser.add_argument('--uv_hash', action='store_true', help="uv hash")

        ### training options
        #parser.add_argument('--iters', type=int, default=30000, help="training iters")
        #parser.add_argument('--lr', type=float, default=1e-2, help="initial learning rate")
        #parser.add_argument('--ckpt', type=str, default='latest')
        #parser.add_argument('--num_rays', type=int, default=4096, help="num rays sampled per image for each training step")
        self.parser.add_argument('--cuda_ray', action='store_true', help="use CUDA raymarching instead of pytorch")
        #self.parser.add_argument('--max_steps', type=int, default=1024, help="max num steps sampled per ray (only valid when using --cuda_ray)")
        #self.parser.add_argument('--num_steps', type=int, default=512, help="num steps sampled per ray (only valid when NOT using --cuda_ray)")
        self.parser.add_argument('--upsample_steps', type=int, default=0, help="num steps up-sampled per ray (only valid when NOT using --cuda_ray)")
        self.parser.add_argument('--update_extra_interval', type=int, default=16, help="iter interval to update extra status (only valid when using --cuda_ray)")
        self.parser.add_argument('--max_ray_batch', type=int, default=4096, help="batch size of rays at inference to avoid OOM (only valid when NOT using --cuda_ray)")
        self.parser.add_argument('--patch_size', type=int, default=1, help="[experimental] render patches in training, so as to apply LPIPS loss. 1 means disabled, use [64, 32, 16] to enable")

        #bound=opt.bound,
        #cuda_ray=opt.cuda_ray,
        #density_scale=1,
        #min_near=opt.min_near,
        #density_thresh=opt.density_thresh,
        #bg_radius=opt.bg_radius,
        
        ### network backbone options
        #self.parser.add_argument('--fp16', action='store_true', help="use amp mixed precision training")
        #self.parser.add_argument('--ff', action='store_true', help="use fully-fused MLP")
        #self.parser.add_argument('--tcnn', action='store_true', help="use TCNN backend")

        ### dataset options
        #self.parser.add_argument('--color_space', type=str, default='srgb', help="Color space, supports (linear, srgb)")
        #self.parser.add_argument('--preload', action='store_true', help="preload all data into GPU, accelerate training but use more GPU memory")
        # (the default value is for the fox dataset)
        self.parser.add_argument('--bound', type=float, default=2, help="assume the scene is bounded in box[-bound, bound]^3, if > 1, will invoke adaptive ray marching.")
        #self.parser.add_argument('--scale', type=float, default=0.33, help="scale camera location into box[-bound, bound]^3")
        #self.parser.add_argument('--offset', type=float, nargs='*', default=[0, 0, 0], help="offset of camera location")
        #self.parser.add_argument('--dt_gamma', type=float, default=1/128, help="dt_gamma (>=0) for adaptive ray marching. set to 0 to disable, >0 to accelerate rendering (but usually with worse quality)")
        self.parser.add_argument('--min_near', type=float, default=0.2, help="minimum near distance for camera")
        self.parser.add_argument('--density_thresh', type=float, default=10, help="threshold for density grid to be occupied")
        self.parser.add_argument('--bg_radius', type=float, default=-1, help="if positive, use a background model at sphere(bg_radius)")

        ### GUI options
        self.parser.add_argument('--gui', action='store_true', help="start a GUI")
        self.parser.add_argument('--W', type=int, default=1920, help="GUI width")
        self.parser.add_argument('--H', type=int, default=1080, help="GUI height")
        self.parser.add_argument('--radius', type=float, default=5, help="default GUI camera radius from center")
        self.parser.add_argument('--fovy', type=float, default=50, help="default GUI camera fovy")
        self.parser.add_argument('--max_spp', type=int, default=64, help="GUI rendering max sample per pixel")

        ### experimental
        self.parser.add_argument('--error_map', action='store_true', help="use error map to sample rays")
        self.parser.add_argument('--clip_text', type=str, default='', help="text input for CLIP guidance")
        self.parser.add_argument('--rand_pose', type=int, default=-1, help="<0 uses no rand pose, =0 only uses rand pose, >0 sample one rand pose every $ known poses")


    def patch_nerf(self):
        
        self.parser.add_argument('--patchNerf_xyzdim', type=int, default=4)        
        self.parser.add_argument('--p_mask_input', action='store_true', help="")
        self.parser.add_argument('--p_save_mask', action='store_true', help="")
        self.parser.add_argument('--patch_num', type=int, default=16*16)        
        self.parser.add_argument('--hidden_dim_for_mask', type=int, default=10)        

        self.parser.add_argument('--is_shape_in_selector', action='store_true', help="")
        self.parser.add_argument('--is_feat_in_selector', action='store_true', help="")

        self.parser.add_argument('--with_place_holder', action='store_true', help="")

        self.parser.add_argument('--p_no_feat', action='store_true', help="")
        
        self.parser.add_argument('--train_org_nb', action='store_true', help="")

        self.parser.add_argument('--res_lighting', action='store_true', help="")
        self.parser.add_argument('--w_res_lighting', type=float, default = 0.0, help="")
        self.parser.add_argument('--infer_no_lighting', action='store_true', help="")


        self.parser.add_argument('--vox_size', type=int, default = 16, help="")

        self.parser.add_argument('--uv_view_random', action='store_true', help="random augmentation")
        
        self.parser.add_argument('--add_noise_new', action='store_true', help="random augmentation")

        self.parser.add_argument('--new_split_tvcg', action='store_true', help="random augmentation")

        self.parser.add_argument('--add_one_test_frame', action='store_true', help="random augmentation")

        self.parser.add_argument('--D_lr', type=float, default=0.0002, help="weight")

        self.parser.add_argument('--test_step_size', type=int, default=30, help="test step size")

        self.parser.add_argument('--test_eval', action='store_true', help="load ema model in testing")

    def posmap_setup(self, opt):
        if opt.combine_pose_style:
            assert opt.use_posmap
        if opt.smooth_latent:
            assert opt.use_style
  
            
    def uvnerf_setup(self, opt):
        #nerf
        if opt.vrnr:
            opt.nerf_output_rgb_dim = max(opt.nerf_output_rgb_dim, 64)
        
        if opt.uv3dNR:
            opt.uvh_feat_dim = opt.posenet_outdim
        #elif opt.dnr3d:
        #    opt.uvh_feat_dim = opt.posenet_outdim
        
        if opt.ab_1c_geometry:
            opt.posenet_setup.tex_latent_dim = 1

        if opt.combine_pose_style:
            opt.uvh_feat_dim = opt.posenet_outdim 
            
        else: 
            input_latent_dim = opt.posenet_outdim if opt.use_posmap else 0
            if opt.use_style and not opt.ab_no_tex_latent:
                input_latent_dim += opt.posenet_setup.tex_latent_dim

            opt.uvh_feat_dim = input_latent_dim

        if self.opt.posenet_setup.use_img_encoder:
            if self.opt.posenet_setup.pred_normal_uv and self.opt.posenet_setup.input_norm_uv: opt.uvh_feat_dim += 3
            if not self.opt.use_img_pose_enc:
                opt.uvh_feat_dim += self.opt.img_encoder_dim

        elif self.opt.posenet_setup.pred_normal_uv and self.opt.posenet_setup.input_norm_uv: opt.uvh_feat_dim += 3

        if self.opt.cat_pos_geo_tex:
            opt.uvh_feat_dim = opt.posenet_setup.tex_latent_dim * 2 + 3 

        opt.generator_input_dim = opt.uvh_feat_dim

        if opt.ab_pose_para:
            opt.uvh_feat_dim = 216
        
    def dataset_setup(self, cfg):
    
        cfg.multi_identity_list = cfg.multi_identity_list.split(" ")                
        dataset_id = {}    
        for i in range(len(cfg.multi_identity_list)):        
            dataset_id.update({cfg.multi_identity_list[i]: i})        
        cfg.dataset_id = [dataset_id]

    def geo_nr(self):
        self.parser.add_argument('--uvEncoder', action='store_true', help="whether encode uv features")

    def add_vrnr_g_setup(opt):
        vrnr_g_opt = argparse.Namespace()
        vrnr_g_opt.fx = True

    def removed_config(self, opt):
        pass

    def merge_cfg(self, opt):
        
        if opt.release_mode: opt.config_dir = ""

        #import project config
        for p in opt.project_config:
            merge_config_opt(os.path.join(opt.config_dir, "configs/projects/%s" % p), opt)

        #--project_config uvm.yml --method_config motion.yml

        #import method config
        method_config = opt.method_config
        for m in method_config:
            merge_config_opt(os.path.join(opt.config_dir, "configs/methods/%s" % m), opt)


        #import data config
        opt.multi_datasets = []
        
        dataset_base_cfg = os.path.join(opt.config_dir, "configs/datasets/", opt.config[0].split("/")[0], "base.yml")

        dataset_base_dir = os.path.join(opt.config_dir, "configs/datasets/")
        
        dataset_id = [ i for i in range(len(opt.config))]
        if opt.swap_tex:
            for i in len(dataset_id):
                dataset_id[i] = (dataset_id[i] + 1) % len(dataset_id)
        
        #import dataset config
        self.dataset_id = dataset_id
        subdir = []
        for i in range(len(opt.config)):
            config_file = os.path.join(dataset_base_dir, opt.config[i])
            data_config = yaml_config(config_file, opt.default_config)
            merge_config_opt(dataset_base_cfg, data_config.dataset)
            
            if i==0: subdir.append(data_config.dataset.dirname) 
            
            for k in data_config:
                if k == "dataset": continue
                setattr(opt, k, data_config[k])
            
            data_config.dataset["id"] += self.dataset_id[i]
            opt.multi_datasets.append(data_config.dataset)
            subdir.append(data_config.dataset.resname)
            opt.dataset_res_name = data_config.dataset.resname
                
        opt.subdir = ""
        dir_name = ""
        for i in range(1, len(subdir)):
            if i >=2 : dir_name += "_"
            dir_name += subdir[i]
            
        opt.subdir = "%s/%s" % (subdir[0], dir_name)
        
        merge_config_opt(dataset_base_cfg, opt)
        
                
    def parse(self, save=True):

        if not self.initialized:
            self.initialize()

        arg = self.parser.parse_args()

        self.opt = Munch()
        for k in arg.__dict__:#basice options
            self.opt[k] = arg.__dict__[k] 
        
        #motion, posenet ect.
        for group in self.parser._action_groups[2:]:
            title = group.title
            self.opt[title] = Munch()
            for action in group._group_actions:
                dest = action.dest
                self.opt[title][dest] = arg.__getattribute__(dest)
    
        #import project config


        if self.opt.train_snap:
            self.dataset_setup(self.opt)
        else: 
            self.merge_cfg(self.opt)
        
        if self.opt.use_nerf or self.opt.dnr3d:
            self.uvnerf_setup(self.opt)

        self.posmap_setup(self.opt)
        vrnr_parse(self.opt)
        

        self.opt.debug_mode = True
        if self.opt.make_demo:
            self.opt.test_step_size = 1

        if self.opt.no_face:
            self.opt.use_face = False

        #self.opt.log_dir = self.opt.checkpoints_dir

        if self.opt.subdir !='':
            self.opt.checkpoints_dir = os.path.join(self.opt.checkpoints_dir, self.opt.subdir)

        self.opt.wrong_input_dir = os.path.join(self.opt.checkpoints_dir, "wrong_input")
        #os.makedirs(self.opt.wrong_input_dir, exist_ok=True)

        self.opt.isTrain = self.isTrain  # train or test
        if self.isTrain:
            self.opt.multiview_ids = self.opt.multi_datasets[0].train_view
            charsplit = " "
            if self.opt.multiview_ids.find(charsplit) == -1:
                charsplit = ","
            self.opt.multiview_ids = self.opt.multiview_ids.split(charsplit)
        else:
            self.opt.multiview_ids = self.opt.multi_datasets[0].test_view
            if self.opt.make_demo:
                self.opt.multiview_ids = self.opt.multi_datasets[0].demo_view
                if self.opt.demo_vid != "":
                    self.opt.multiview_ids = self.opt.demo_vid

            charsplit = " "
            if self.opt.multiview_ids.find(charsplit) == -1:
                charsplit = ","
            self.opt.multiview_ids = self.opt.multiview_ids.split(charsplit)

        str_ids = self.opt.gpu_ids.split(',')
        self.opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                self.opt.gpu_ids.append(id)#always start from 0  - int(str_ids[0])

        is_local = False
        cpu_num = 4 #local machine for each task, 4 in vulcan
        self.opt.nThreads = cpu_num #ddp for each

        gpunum = len(self.opt.gpu_ids)

        self.opt.D_lr *= np.sqrt(self.opt.batchSize)
        self.opt.lr *= np.sqrt(self.opt.batchSize) 

    
        args = self.opt

    
        # save to the disk
        expr_dir = os.path.join(self.opt.checkpoints_dir, self.opt.name)
        os.makedirs(expr_dir, exist_ok=True)

        if save and not self.opt.continue_train:
            file_name = os.path.join(expr_dir, 'opt.txt')
            with open(file_name, 'wt') as opt_file:
                opt_file.write('------------ Options -------------\n')
                for k, v in sorted(args.items()):
                    opt_file.write('%s: %s\n' % (str(k), str(v)))
                opt_file.write('-------------- End ----------------\n')

            os.system("chmod -R 707 %s" % expr_dir)   
  
        return self.opt
