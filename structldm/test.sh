#!/bin/bash
. scripts/unit/methods.sh

method="FAD2DHybrid"
gpu=$1

RNUM=24 #512 28
SNUM=2
RATIO=1
NerfR=4
UVRESO=128
UVTYPE="BF"
SRESO=128
SDIM=16
WIDTH=32


config="--data_config renderpeople.yml --method_config df_trip_GAN.yml"

render="--uv_type ${UVTYPE} --uv_reso ${UVRESO} --ratio -1  --batch 1 --chunk 1 --gen_ratio ${RATIO} --nerf_ratio ${NerfR} " 

setup="--new_pose_est --data_step 3 --up_layer 3 --structured_dim ${SDIM} --structured_reso ${SRESO} --stage1_s1_rec_fitting --load_pretrained_model --load_model_name FAD2DHybrid_4S_RPGAN_4_BF128_S1281632  --which_epoch 4260"
#

net="--voxel_net --texdecoder_outdim ${SDIM} --df_nerf_dim 16 --output_ch 16 --width ${WIDTH}" 
netD="--with_D --patch_gan --disc_start 0 --w_G_GAN 1 --w_D 1"

ab="--use_face --knn_part --tv_latent --w_tv_embed 1e-6" 

fixembed="--fix_embed --lr_embed 0 --w_embed 0"
nerf_vgg_fix="--w_G_L1 2 --w_G_VGG 4 --w_G_feat 0 --nerf_vgg --w_nerf_VGG 2.0 --normal_tv --normal_recon --w_tv_normal 2e-5 --w_G_Normal 1 --eikonal_lambda 0.5 --min_surf_lambda 1.5 "


debug="--visualize_depth --visualize_normal --no_visualize_pose" 

modelname="4S_RPGAN_Fix_vggnerf_${NerfR}_${UVTYPE}${UVRESO}_S${SRESO}${SDIM}${WIDTH}" #

aug="" 

epoch="--niter 8000 --which_epoch 1380 " #

basic="--eikonal_lambda 0.5 --adjust_gamma --gamma_lb 20 --min_surf_lambda 1.5 --deltasdf --sampler_std 15 --input_ch_views 3 --white_bg --depth 4 --style_dim 64 --input_ch_views 3 --N_samples ${RNUM} --uvVol_smpl_pts ${SNUM} --w_D_grad 0 --glr 2e-4 --dlr 4e-5 --df_ab_nerf_rec --w_tex_rec 0.1"

w="--w_Face 5 --w_nerf_rec 2"

cmd="${config} ${basic} ${setup} ${net} ${nerf_vgg_fix} ${fixembed} ${netD} ${ab} ${w} ${debug} ${!method} ${aug} ${render} ${epoch} --expname ${method}_${modelname}"

diff_name="r2_x0Clip_point"

diff="--load_sampled_pose --load_diffused_sample --test_diff_part 20 --diffusion_name ${diff_name} --eval_metrics --diffusion_epoch 240000 --diffusion_step 350" #

cmd="${config} ${basic} ${setup} ${net} ${netD} ${ab} ${w} ${debug} ${!method} ${aug} ${render} ${epoch} --expname ${method}_${modelname} ${diff} --save_name demo_geo7_${method}_${modelname}_${diff_name}_5w"

batch_api "${gpu}" "${cmd}" "test" "1380" 

