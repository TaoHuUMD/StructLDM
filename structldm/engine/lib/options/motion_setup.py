import argparse
import os


def set_motion(parser):

    motion = parser.add_argument_group('motion')

    motion.add_argument("--infer_velocity", type=float, default=1, help='')


    motion.add_argument("--motion_chain", action='store_true', help='')
    motion.add_argument("--motion_steps", type=str, default="-25 -20 -15 -10 -5 -1 5 10 15 20 25", help='previous motion status')

    motion.add_argument("--motion_point", type=int, default=0, help='start or end point of a motion, <0, start, >0 end')


    motion.add_argument("--style_dim", type=int, default=256, help='')
    #motion.add_argument("--tex_latent_dim", type=int, default=16, help='')
    #motion.add_argument("--pose_tex_dim", type=int, default=64, help='dim of posmap out')
    motion.add_argument("--ab_uvh_plane_c", type=int, default=16, help='dim of vh, uh plane')
    motion.add_argument("--nerf_dim", type=int, default=32, help='dim of nerf input')

    motion.add_argument('--aug_random_flip', action='store_true', help="")

    #motion.add_argument("--supnet_dim", type=int, default=32, help='input dim of super reso net')

    motion.add_argument("--use_global_posemap", action='store_true', help='whether use global verts in posemap')

    motion.add_argument("--is_pad_img", action='store_true', help='pad rectangle image to square')

    
    #1
    #--is_pad_img --style_dim 256 --tex_latent_dim 16 --pose_tex_dim 64 --ab_uvh_plane_c 16 --nerf_dim 32
    #--ab_uvh_plane --ab_nerf_rec --ab_Ddual --ab_D_pose --ab_tex_rec --ab_Dtex --ab_Dtex_pose
    #--w_D 1 --w_D_grad 0 --w_G_GAN 1.0 --w_Face 5.0 --w_nerf_rec 3.0 --w_tex_rec 3.0 --w_posmap 1.0
    
    #2 no nerf
    #--ab_only_sup_dynamic_tex
    motion.add_argument("--ab_sup_only_dynamic_tex", action='store_true', help='no nerf, only supreso posemap out')
    
    #3 no nerf, supp. cond on static style.        
    motion.add_argument("--ab_sup_only_static_style", action='store_true', help='no nerf, only supreso on style')


    motion.add_argument("--ab_sup_2d_style", action='store_true', help='sup net only condition on 2d style latent')


    motion.add_argument("--dual_discrim_eg3d", action='store_true', help='#whether D on tex field')


    motion.add_argument("--use_org_gan_loss", action='store_true', help='#')
    motion.add_argument("--use_org_discrim", action='store_true', help='#')


    motion.add_argument("--ab_Dtex", action='store_true', help='#whether D on tex field')
    motion.add_argument("--ab_Dtex_pose", action='store_true', help='#whether Dtex is conditioned on pose map')

    motion.add_argument("--ab_uvh_plane", action='store_true', help='uvh tri-plane')
    motion.add_argument("--ab_nerf_rec", action='store_true', help='whether rec loss on nerf')

    motion.add_argument("--ab_Ddual", action='store_true', help='dual discriminator')
    motion.add_argument("--ab_D_pose", action='store_true', help='D cond on pose')
    motion.add_argument("--ab_tex_rec", action='store_true', help='uv tex recon')

    motion.add_argument("--D_label_noise", action='store_true', help='add label noise for D')
    motion.add_argument("--D_noise_factor", type=float, default=0.05, help='add label noise for D')

    motion.add_argument("--debug_data_size", type=int, default=10, help='debug small dataset')

    motion.add_argument("--abandon", action='store_true', help='not used')
    

    motion.add_argument("--ab_cond_uv_latent", action='store_true', help='super reso cond on 2d uv lat')
    motion.add_argument("--general_superreso", action='store_true', help='super reso cond on 2d uv lat')
    
    #in the future        
    motion.add_argument("--deep_nerf", action='store_true', help='3 layer nerf network')
    motion.add_argument("--ab_cond_1d_lat", action='store_true', help='super reso cond on 1d style')
