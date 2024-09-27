#import open3d as o3d
from .yacs import CfgNode as CN
import argparse
import os
import numpy as np
import pprint

cfg = CN()


cfg.out_latent = False

cfg.load_latent = False
cfg.nb_swap_tex = False

cfg.lat_dim = 16

cfg.use_smpl_vts =  False

cfg.big_nerf = False

cfg.Din3 = False


cfg.multi_id = False
cfg.id_num_total = 6
cfg.id_num = 3
cfg.train_snap = False


cfg.train_new = False

cfg.sample_all_pixels = False
cfg.N_local_ray_samples = 32

cfg.use_small_reso = False
cfg.use_sparse_train = False
cfg.N_sparse_pts = 20

cfg.vis_per_epoch = False

cfg.load_sparse = False
cfg.res256 = False

cfg.use_base = True

cfg.tmp_eva = False
cfg.white_vgg = False


cfg.print1 = False

#cfg.norm_l1_weight = 100.0
#cfg.norm_lp_weight = 1.0

#cfg.norm_l1_weight = 10.0
#cfg.norm_lp_weight = 0.1
#cfg.norm_D_weight = 0.1

cfg.fineturn_nb = False

cfg.norm_l1_weight = 10.0
cfg.norm_lp_weight = 0.1
cfg.norm_D_weight = 0.3
cfg.norm_tv_weight = 1.0 

cfg.norm_face_weight = 0.1

cfg.loss_scale_for_record = 10.0

if True:
    cfg.norm_l1_weight = 1.0
    cfg.norm_lp_weight = 1
    cfg.norm_D_weight = 1
    cfg.norm_tv_weight = 1.0 

    cfg.norm_face_weight = 1

    cfg.loss_scale_for_record = 1.0


cfg.use_tv_loss = False
cfg.use_lpip_vgg = False
cfg.use_lpip_alex = False
cfg.use_old_vgg = False
cfg.use_face_loss = False
cfg.face_model_pth = "../asset/spretrains/sphere20a_20171020.pth"

cfg.tv_dir = "hw"

cfg.use_percept_loss = False

cfg.weight_tv = 0.0001
cfg.weight_pep = 1.0
cfg.weight_face = 0.5

cfg.use_pep_l1 = False
cfg.weight_pep_l1 = 1.0

cfg.use_D = False
cfg.weight_D = 0.5

cfg.start_fullimg_epoch = 199

cfg.small_lr = False

#add discriminator
cfg.norm = 'instance' #help='instance normalization or batch normalization'
cfg.netD_input_nc = 3
cfg.pool_size = 0
cfg.num_D = 2 # 'number of discriminators to use'
cfg.n_layers_D =3 # 'only used if which_model_netD==n_layers')
cfg.ndf =64 # '# of discrim filters in first conv layer')
cfg.no_ganFeat_loss = False
cfg.no_lsgan = False
cfg.beta1=0.5
cfg.D_lr = 1e-4

cfg.niter_decay = 500
cfg.niter = 200

cfg.weight_vgg = 1.0
cfg.weight_l1 = 1.0
cfg.not_even = True

cfg.fineturn = False

cfg.diff_org_smpl = True

cfg.use_deform_net = False
cfg.par_deform = 0.05
cfg.pose_res = 1

cfg.new_local = False
 
cfg.state = 'Train'

#cfg.inter_eva_epoch = [300, 600, 900, 1200]
cfg.inter_eva_epoch = 200

cfg.check_mesh = False
cfg.check_can_mesh = False

cfg.share_diff = False

cfg.add_normal = False

cfg.normal_sup = False

cfg.can_architecture = False

cfg.use_pose_cond = False

cfg.mode=""
cfg.epoch=-1

cfg.debug_eva = False

cfg.eva_box = True

cfg.test_epoch=-1 #0

cfg.debug_per_frame = False

cfg.test_seen_pose = False

cfg.overfit_oneview = False

cfg.is_save_depth = False

cfg.tmp_path = '../data/tmp'

cfg.eval_whole_img = False

cfg.USE_NEW_VERTICES = True
cfg.Test_KnownPose=False
cfg.T_pose=False

cfg.mix_can_pose = False

cfg.max_can_pose = False
cfg.cat_can_pose = False
cfg.avg_can_pose = False
cfg.pose_conv_layers = 2

#local nerf
cfg.use_small_dilation = True

cfg.conv_smpl_vertices = False

cfg.org_pose=True

cfg.use_max_color = False

cfg.use_dilate_model = True

cfg.no_can = False

cfg.check_masks = False

cfg.debug_canonical_transform = False

cfg.debug_output = False

cfg.debug_1pose_all_views = False

cfg.use_multiview = False

cfg.is_opt_mask = False
 
cfg.is_opt_skin_weights = False
cfg.is_opt_pose = False
cfg.is_opt_vertics = False
cfg.use_verts_regularier = False

cfg.rm_lightpts_zero = False


#local Nerf based on depth
cfg.local_nerf = False #whether local nerf, default false.
cfg.local_only_body = True # only sample rays on human body

cfg.local_inverse_skin_mid = False
#cfg.local_inverse_skin_all = True

cfg.use_new_vertices = True
cfg.mesh_smooth = False

cfg.use_density_th = False
cfg.use_new_dist = False

cfg.use_new_lr = True #should be false 

cfg.normal_offsets = True
cfg.par_verts_reg = 0.1

cfg.only_optimize_mask = False
cfg.output_intermediate_mesh = True


cfg.par_laplacian_can = 0.1
cfg.par_laplacian_posed = 0.1
cfg.par_mask_loss = 2.0



cfg.max_ray_interval = 0.25*0.2

cfg.deform_skin = False

cfg.density_th = 0.5
cfg.par_density_norm = 1000
#cfg.par_nerf_dist = 7.5

cfg.par_nerf_dist = 1.0



cfg.min_ray_interval = cfg.max_ray_interval * 0.2
#cfg.par_interval_decay = -1.0/200
cfg.par_interval_decay = 0



# experiment name
cfg.exp_name = 'hello'

cfg.debug_mode = True
cfg.viewpoint_ele = 0.8
if not cfg.debug_mode:
    cfg.viewpoint_ele = 0

cfg.change_pose = True
if not cfg.debug_mode:
    cfg.change_pose = False

cfg.In_Canonical = False
cfg.voxel_factor = 1

cfg.invSkin_mode="NN_IVS_POINTS" #NN_PROJECTION

# network
cfg.point_feature = 9
cfg.distributed = False

# data
cfg.human = 313
cfg.training_view = [0, 6, 12, 18]
cfg.intv = 1
cfg.begin_i = 0  # the first smpl
cfg.ni = 1  # number of smpls

cfg.render_ni = -1

cfg.i = 1  # the i-th smpl
cfg.i_intv = 1
cfg.nv = 6890  # number of vertices
cfg.smpl = 'smpl_4views_5e-4'
cfg.vertices = 'vertices'
cfg.params = 'params_4views_5e-4'
cfg.mask_bkgd = True
cfg.sample_smpl = False
cfg.sample_grid = False
cfg.sample_fg_ratio = 0.7
cfg.H = 1024
cfg.W = 1024
cfg.add_pointcloud = False

cfg.big_box = False

cfg.rot_ratio = 0.
cfg.rot_range = np.pi / 32

cfg.new_test_rule=True
cfg.evaluate_views = 4
cfg.evaluate_imgs = 20 #evaluate 20 images for each

cfg.debug_one_frame = False
cfg.debug_two_frame = False

#cfg.evaluate_frame_interval = 10000 #org 30.

# mesh
#cfg.mesh_th = 50  # threshold of alpha
cfg.mesh_th = 10

# task
cfg.task = 'nerf4d'

# gpus
cfg.gpus = list(range(8))
# if load the pretrained network
cfg.resume = True

# epoch
cfg.ep_iter = -1
cfg.save_ep = 100
cfg.save_latest_ep = 5
cfg.eval_ep = 100

cfg.pn = 2

# -----------------------------------------------------------------------------
# train
# -----------------------------------------------------------------------------
cfg.train = CN()

cfg.train.dataset = 'CocoTrain'
cfg.train.epoch = 10000
cfg.train.num_workers = 4
cfg.train.collator = ''
cfg.train.batch_sampler = 'default'
cfg.train.sampler_meta = CN({'min_hw': [256, 256], 'max_hw': [480, 640], 'strategy': 'range'})
cfg.train.shuffle = True

# use adam as default
cfg.train.optim = 'adam'

cfg.train.lr = 5e-4

cfg.train.weight_decay = 0

linear_sche = CN({'type': 'multi_step', 'milestones': [1, 2, 3, 4], 'gamma': 0.5})

cfg.train.scheduler = CN({'type': 'multi_step', 'milestones': [80, 120, 200, 240], 'gamma': 0.5})

cfg.train.batch_size = 4

cfg.train.acti_func = 'relu'

cfg.train.use_vgg = False
cfg.train.vgg_pretrained = ''
cfg.train.vgg_layer_name = [0,0,0,0,0]

cfg.train.use_ssim = False
cfg.train.use_d = False

# test
cfg.test = CN()
cfg.test.dataset = 'CocoVal'
cfg.test.batch_size = 1
cfg.test.epoch = -1
cfg.test.sampler = 'default'
cfg.test.batch_sampler = 'default'
cfg.test.sampler_meta = CN({'min_hw': [480, 640], 'max_hw': [480, 640], 'strategy': 'origin'})

base_dir = '../'
#vp="/vulcanscratch/taohu/projects/neural_body/neuralbody/"

#if os.path.isdir(vp):
#    base_dir = vp

# trained model
cfg.trained_model_dir = base_dir + 'data/trained_model'

# recorder
cfg.record_dir = base_dir + 'data/record'
cfg.log_interval = 100
cfg.record_interval = 100

# result
cfg.result_dir = base_dir + 'data/result'

# evaluation
cfg.skip_eval = False

cfg.test_novel_pose = False
cfg.novel_pose_ni = 100

cfg.fix_random = False
 
cfg.vis = 'mesh'

cfg.begin_newpose_id = 30

cfg.local_mach = False

# data
cfg.body_sample_ratio = 0.5
cfg.face_sample_ratio = 0.

cfg.train.epoch = 200

cfg.opt = CN()

cfg.nerf = CN()
cfg.nerf.latentUV = False

cfg.nerf.mult_stage1_terminate = 20
cfg.nerf.use_conv3d = False
cfg.nerf.deUVNerf = True 

cfg.nerf.nolighting = False

cfg.nerf.one_stage = False
cfg.nerf.two_stage_uv = False
cfg.nerf.s1_density = False
cfg.nerf.s2_uv = False
cfg.nerf.uv_3d = True
cfg.nerf.implicit_trans = False
cfg.nerf.uvdim = 8

cfg.nerf.use_bk = False

cfg.nerf.multi_in_uvh = False
cfg.nerf.multi_in_pose = False
cfg.nerf.multi_in_shape = False

cfg.nerf.swap1 = True

cfg.nerf.fix_uv = False
cfg.nerf.pred_uv_offset = False
cfg.nerf.exp_tex = False #no texture translation

cfg.nerf.id_code = True

cfg.nerf.save_tex = False
cfg.nerf.swap_tex = False

#uv is based on 3d feature, lighting, view direction.
cfg.nerf.cond_uv = False

cfg.nerf.debug = False

cfg.nerf.sup_all_uv = False

cfg.nerf.big_nerf = True
cfg.nerf.use_pose_cond=True
cfg.nerf.add1layer=True

cfg.nerf.normal_sup = False
cfg.nerf.uv_sup = True

cfg.nerf.uv_w = 1.0
cfg.nerf.norm_w = 1.0
cfg.nerf.only_sup_smpl = False

cfg.vrnr = False
cfg.VRNR = CN()

#uv latent
cfg.VRNR.share_3D=False
cfg.VRNR.not_3d_uvlatent=False
cfg.VRNR.uv_normal=False
cfg.VRNR.use_light = False
cfg.VRNR.not_latent_uv = False

cfg.VRNR.use_conv3d = False

#VRNR, generator
cfg.VRNR.netG = 'global'
cfg.VRNR.ngf = 64
cfg.VRNR.n_downsample_global = 3

cfg.VRNR.no_encoder = False

cfg.snap_step = 2

cfg.VRNR.n_blocks_global = 9
#help='number of residual blocks in the global generator network')
cfg.VRNR.n_blocks_local = 3
cfg.VRNR.n_local_enhancers =1 # help='number of local enhancers to use')
cfg.VRNR.niter_fix_global =0 # help='number of epochs that we only train the outmost local enhancer')
cfg.VRNR.no_instance = True  # help='if specified, do *not* add instance map as input')
cfg.VRNR.output_nc = 3
cfg.VRNR.norm ='instance'

cfg.VRNR.uv2pix = False

cfg.VRNR.beta1 = 0.5
cfg.VRNR.D_lr = 1e-4

#VRNR, discriminator
cfg.VRNR.netD_input_nc = cfg.netD_input_nc
cfg.VRNR.ndf = cfg.ndf
cfg.VRNR.n_layers_D = cfg.n_layers_D
cfg.VRNR.norm = 'instance' #help='instance normalization or batch normalization'
cfg.VRNR.num_D = 2 # 'number of discriminators to use'
cfg.VRNR.no_ganFeat_loss = False
#cfg.old_lr = cfg.VRNR.train.lr

cfg.VRNR.lr = 2e-4
cfg.VRNR.D_lr = 2e-4

cfg.VRNR.no_lsgan = False
cfg.VRNR.netD_input_nc = 3
cfg.VRNR.pool_size = 0

cfg.VRNR.only_nerf = False
cfg.VRNR.only_gen = False

cfg.VRNR.nerf_reso = 64
cfg.VRNR.gen_reso = 256

if cfg.VRNR.nerf_reso == 64:
    cfg.VRNR.n_downsample_global = 2
elif cfg.VRNR.nerf_reso == 32:
    cfg.VRNR.n_downsample_global = 3
elif cfg.VRNR.nerf_reso == 16:
    cfg.VRNR.n_downsample_global = 4

cfg.VRNR.n_downsample_global = 4

#cfg.VRNR.gen_ratio = 0.5



cfg.VRNR.weight_pep_vgg = 10.0
cfg.VRNR.weight_pep_l1 = 0.0

cfg.VRNR.weight_nerf = 1.0
cfg.VRNR.weight_tv = 0.0001

cfg.VRNR.weight_face = 5.0


cfg.VRNR.use_D = True
cfg.VRNR.weight_D = 1.0

cfg.VRNR.debug = False

cfg.org_img_reso = 1024

cfg.freeze_img_encoder = False
cfg.freezeIE_after = 0

cfg.mixImgFeat = False
cfg.attentionFeat = True

cfg.add1layer = True

cfg.use_img_feature = False
cfg.num_views = 4
cfg.debug_no_img = False

cfg.dir_pre=""

cfg.save_ep = 100
cfg.eval_vis_ep = 50

def parse_cfg(cfg, args):
    if len(cfg.task) == 0:
        raise ValueError('task must be specified')

    # assign the gpus
    #os.environ['CUDA_VISIBLE_DEVICES'] = ', '.join([str(gpu) for gpu in cfg.gpus])

    if cfg.dir_pre=="":
        cfg.trained_model_dir = os.path.join(cfg.trained_model_dir, cfg.task, cfg.exp_name)
        cfg.result_dir = os.path.join(cfg.result_dir, cfg.task, cfg.exp_name)
    else:
        cfg.trained_model_dir = os.path.join(cfg.trained_model_dir, cfg.task, cfg.dir_pre, cfg.exp_name)
        cfg.result_dir = os.path.join(cfg.result_dir, cfg.task, cfg.dir_pre, cfg.exp_name)        
        
    cfg.record_dir = os.path.join(cfg.record_dir, cfg.task, cfg.dir_pre, cfg.exp_name)

    cfg.tmp_result_dir = os.path.join(cfg.result_dir, cfg.task, cfg.exp_name, 'intermediate')
    

    cfg.local_rank = args.local_rank
    
    cfg.distributed = cfg.distributed or args.launcher not in ['none']

    cfg.default_test_epoch = cfg.train.epoch - 1
    
    if cfg.test_epoch == 0: cfg.test.epoch = cfg.train.epoch - 1 
    else: cfg.test.epoch = cfg.test_epoch

    cfg.train.scheduler.decay_epochs = 800

    #os.makedirs(cfg.tmp_result_dir, exist_ok=True)
  
    if cfg.mode == "CanVol" \
        or cfg.mode == "CanDiff" \
        or cfg.mode == "CanDiffLocal" \
            or cfg.mode == "CanDiffLocal2":
        cfg.In_Canonical = True

    if cfg.mode == "NBNew" or cfg.mode == "NoCan":
        cfg.In_Canonical = False
        
    #if cfg.epoch !=-1:
    #    cfg.test.epoch = cfg.epoch
    

    if cfg.use_img_feature:
        add_img_encoder_setup(cfg)
        cfg.save_ep = 50
        cfg.eval_vis_ep = 10
        cfg.save_latest_ep = 5
        
        if cfg.mixImgFeat:
            cfg.attentionFeat = False
        
        cfg.inter_eva_epoch = -2

    cfg.log_interval = 20
    cfg.record_interval = 20

    if cfg.multi_id:
        cfg.eval_vis_ep = 10
        cfg.save_ep = 50

    if cfg.debug_one_frame:
        cfg.eval_vis_ep = 1
    if cfg.overfit_oneview or cfg.debug_1pose_all_views:
        if cfg.eval_vis_ep > 5:
            cfg.eval_vis_ep = 5
        cfg.save_ep = 10
    elif cfg.debug_output:
        cfg.eval_vis_ep = 1
        cfg.save_ep = 1

    if cfg.sample_all_pixels:
        cfg.save_ep_all_pix = 5
        cfg.eval_vis_ep_all_pix = 3

        cfg.save_latest_ep_all_pix = 1

        cfg.test_epoch_all_pix = cfg.start_fullimg_epoch + 50
        cfg.inter_eva_epoch_all_pix = cfg.start_fullimg_epoch + 30

        cfg.niter_decay = 100 
        cfg.niter = 200 + 10


        cfg.log_interval = 10
        cfg.record_interval = 10

        cfg.train.epoch = cfg.start_fullimg_epoch + 100
        cfg.default_test_epoch = cfg.train.epoch - 1

        cfg.debug_per_frame = True
        
        if cfg.mode =='NoCan':
            cfg.save_ep_all_pix = 10
            cfg.eval_vis_ep_all_pix = 5

            cfg.save_latest_ep_all_pix = 5

            cfg.save_latest_ep = 5

            cfg.debug_per_frame = False
        #cfg.eval_ep = 1
    
    
        #cfg.eval_ep = 1

    if cfg.vis_per_epoch:
        cfg.eval_vis_ep_all_pix = 1
        cfg.debug_per_frame = True


    if cfg.debug_per_frame or cfg.print1:
        cfg.log_interval = 1
        cfg.record_interval = 1
        

    if cfg.USE_NEW_VERTICES:
        cfg.vertices="new_vertices"
        cfg.params="new_params"

    if cfg.use_multiview:
        cfg.training_view = [0, 1, 2, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20, 21, 22]
        cfg.test_view = [0, 3, 9, 15]

    if cfg.debug_1pose_all_views:
        #cfg.training_view = [0, 1, 2, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20, 21, 22]
        cfg.training_view = [0, 1, 2, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 16, 17, 18]
        #cfg.training_view = [15]


        cfg.test_view = [0, 3, 9, 15]
        cfg.evaluate_views = 4
        #test 3, 9, 15, 

    if cfg.check_masks:
        cfg.is_opt_mask = True
        
    cfg.is_check_mesh = cfg.check_mesh or cfg.check_can_mesh

    if cfg.new_local:
        #cfg.N_local_ray_samples = 8
        cfg.par_nerf_dist = 10

    if cfg.small_lr:
        cfg.train.lr = 5e-5 
        cfg.D_lr = 1e-5

    if cfg.res256:
        cfg.ratio = 256/1024
        cfg.pn = 12
        if cfg.use_D: cfg.pn += 2
        
        if cfg.local_mach:
            cfg.pn += 12
        
    if cfg.use_small_reso:
        #cfg.ratio = 384/1024
        cfg.ratio = 320/1024
        cfg.pn = 4
        if cfg.use_D: cfg.pn += 2
        if cfg.local_mach:
            cfg.pn += 2
    else:
        cfg.pn = 2
        if cfg.use_D: cfg.pn += 1
        if cfg.use_deform_net:
            cfg.pn += 2
        if cfg.local_mach:
            cfg.pn +=1
        
    if cfg.no_can:
        cfg.pn -= 6


    if cfg.use_lpip_vgg or cfg.use_lpip_alex or cfg.use_old_vgg or cfg.use_pep_l1:
        cfg.use_percept_loss = True    
        
    #cfg.train.scheduler = linear_sche
        
    if cfg.multi_id:

        if cfg.vrnr:
            cfg.network_module = 'engine.thutil.loss.net_VR_DNR'
            cfg.network_path = 'engine.thutil/loss/net_VR_DNR.py'
        else:
            cfg.network_module = 'feature_lib.model.Multi_DeNerf'
            cfg.network_path = 'feature_lib/model/Multi_DeNerf.py'

        cfg.train.multi_dataset = []
        cfg.test.multi_dataset = []

        if cfg.id_num == 1:
            cfg.zju_id_list = ["xyzc_394"]
            cfg.snap_id_list = ["snapshot_f4c"]
        else:    
            cfg.zju_id_list = ["xyzc_394", "xyzc_377", "xyzc_313"]#
            #cfg.zju_id_list = ["xyzc_394"]#
            cfg.snap_id_list = ["snapshot_f4c", "snapshot_f3c", "snapshot_m2c"]
            
        make_dataset_id(cfg)

        if cfg.train_snap:
            cfg.multi_identity_list = cfg.snap_id_list
            make_multi_identity_list_snap(cfg)
        else: 
            cfg.multi_identity_list = cfg.zju_id_list
            
            make_multi_identity_list_zju(cfg)

        if cfg.debug_1pose_all_views:
            #cfg.train.epoch = 200
            cfg.nerf.mult_stage1_terminate = 20

    vrnr_setup(cfg)
    
    #cfg.data_root = '../data/people_snapshot/female-3-casual'

def make_dataset_id(cfg):
    if cfg.train_snap:
        cfg.dataset_id = [{
            "snapshot_f4c": 0,
            "snapshot_f3c": 1,
            "snapshot_m2c": 2,
            "snap_386": 3,
            "snap_392": 4,
            "snap_311": 5
        }]

        cfg.dataset_id_swap1 = [{
            "snapshot_f4c": 1,
            "snapshot_f3c": 0,
            "snapshot_m2c": 0,
            "snap_386": 3,
            "snap_392": 4,
            "snap_311": 5
        }]

        cfg.dataset_id_swap2 = [{
            "snapshot_f4c": 2,
            "snapshot_f3c": 2,
            "snapshot_m2c": 1,
            "snap_386": 3,
            "snap_392": 4,
            "snap_311": 5
        }]

    else:
          
        cfg.dataset_id = [{
                "xyzc_394": 0,
                "xyzc_377": 1,
                "xyzc_313": 2,
                "xyzc_386": 3,
                "xyzc_392": 4,
                "xyzc_311": 5
            }]

        cfg.dataset_id_swap1 = [{
            "xyzc_394": 1,
            "xyzc_377": 0,
            "xyzc_313": 0,
            "xyzc_386": 3,
            "xyzc_392": 4,
            "xyzc_311": 5
        }]

        cfg.dataset_id_swap2 = [{
            "xyzc_394": 2,
            "xyzc_377": 2,
            "xyzc_313": 1,
            "xyzc_386": 3,
            "xyzc_392": 4,
            "xyzc_311": 5
        }]
    
    cfg.dataset_id_swap = cfg.dataset_id_swap1
    if not cfg.nerf.swap1:
        cfg.dataset_id_swap = cfg.dataset_id_swap2

def make_multi_identity_list_zju(cfg):   

    for d in cfg.multi_identity_list:
        human_id = d.split('_')[1]
        cfg.train.multi_dataset.append('Human{}_0001_Train'.format(human_id))
        cfg.test.multi_dataset.append('Human{}_0001_Test'.format(human_id))


def make_multi_identity_list_snap(cfg):
    identity_list = cfg.multi_identity_list
    #["f4c", "f3c",  "m2c"]
    cfg.train.multi_dataset.append('Female_4_casual_Train')
    cfg.train.multi_dataset.append('Female_3_casual_Train')
    cfg.train.multi_dataset.append('Male_2_casual_Train')

    cfg.test.multi_dataset.append('Female_4_casual_Test')
    cfg.test.multi_dataset.append('Female_3_casual_Test')
    cfg.test.multi_dataset.append('Male_2_casual_Test')
    
def make_multi_identity_list(cfg):    
    if cfg.train_snap:
        make_multi_identity_list_snap(cfg)
    else: 
        make_multi_identity_list_zju(cfg)

def vrnr_setup(cfg):

    cfg.train.lr = 2e-4

    if cfg.train_snap:
        cfg.org_img_reso = 1080

    cfg.VRNR.nerf_ratio = cfg.VRNR.nerf_reso / cfg.org_img_reso
    cfg.VRNR.gen_ratio = cfg.VRNR.gen_reso / cfg.org_img_reso

    if cfg.vrnr:
        cfg.ratio = cfg.VRNR.nerf_ratio

    if not cfg.vrnr:
        return
    
    if cfg.VRNR.debug:
        cfg.train.epoch = 50
        cfg.eval_vis_ep = 5
        cfg.save_ep = 10
        
    cfg.train.epoch = 600
        
    if cfg.VRNR.only_nerf:
        cfg.use_percept_loss = False
        cfg.use_old_vgg = False 
        cfg.use_tv_loss = False 
        cfg.use_D = False
                    
        cfg.VRNR.nerf_reso = 64
        cfg.VRNR.nerf_ratio = cfg.VRNR.nerf_reso / cfg.org_img_reso

        
    #if cfg.VRNR.only_gen:
    #    cfg.VRNR.only_gen = False    

def add_img_encoder_setup(cfg):
        
    #network_module: 'lib.networks.latent_xyzc'
    #network_path: 'lib/networks/latent_xyzc.py'
    cfg.network_module = 'feature_lib.model.FeatureNerf'
    cfg.network_path = 'feature_lib/model/FeatureNerf.py'
        
    cfg.num_views = 4
        
    
def setup_opt():
    
    cfg.opt.num_views = 4

    cfg.opt.SDF = False
    cfg.opt.Nerf = True

    cfg.opt.use_image_normal = False
    cfg.opt.input_smpl_normal = False
    cfg.opt.fine_part = False
    cfg.opt.coarse_part = True
    cfg.opt.preserve_single = False
    
    cfg.opt.loadSize = 512
    
    cfg.opt.nerf_xyz = False
    
    cfg.opt.num_stack =4
    cfg.opt.num_hourglass = 2
    cfg.opt.fine_num_stack =  1
    cfg.opt.fine_num_hourglass = 2
    cfg.opt.fine_hourglass_dim = 32 
    cfg.opt.skip_hourglass = False
    cfg.opt.hg_down ='ave_pool'# help='ave pool || conv64 || conv128')
    cfg.opt.hourglass_dim = 256 #', type=int, default='256', help='256 | 512')
    cfg.opt.input_smpl_normal = False #', action='store_true')
    cfg.opt.wo_smpl = False #', action='store_true')
    cfg.opt.wo_smpl_normal = False #', action='store_true')
    cfg.opt.w_smpl_normal = False #', action='store_true')
    cfg.opt.wo_xyz = False #', action='store_true')
    cfg.opt.no_fill = False #', action='store_true')
    
    cfg.opt.norm ='group' #,help='instance normalization or batch normalization or group normalization')
    cfg.opt.pts_threshold = 0.1 
    #', type=float, default=0.1, help='distance threshold between feature points and query points')

def make_cfg(args):
    
    setup_opt()
    
    cfg.merge_from_file(args.cfg_file)
    cfg.merge_from_list(args.opts)
    parse_cfg(cfg, args)
    # pprint.pprint(cfg)
    return cfg

