#!/bin/bash

gpu=$1

RNUM=24
SNUM=2
RATIO=1
NerfR=4
UVRESO=128
UVTYPE="BF"
SRESO=128
SDIM=16
WIDTH=32

config="--data_config renderpeople.yml --method_config df_trip_GAN.yml"
modelname="renderpeople" #

data_dir="./data"

render="--uv_type ${UVTYPE} --uv_reso ${UVRESO} --ratio -1  --batch 1 --chunk 1 --gen_ratio ${RATIO} --nerf_ratio ${NerfR} "

setup="--new_pose_est --data_step 3 --up_layer 3 --structured_dim ${SDIM} --structured_reso ${SRESO}  --stage_1_fitting --stage_1_ad_rec --stage1_s1_rec_fitting --uvh_trip_direct --tex_2dp"

net="--voxel_net --texdecoder_outdim ${SDIM} --df_nerf_dim 16 --output_ch 16 --width ${WIDTH}" 
netD="--with_D --patch_gan --disc_start 0 --w_G_GAN 1 --w_D 1"

ab="--use_face --knn_part --tv_latent --w_tv_embed 1e-6"

fixembed="--fix_embed --lr_embed 0 --w_embed 0"

debug="--visualize_depth --visualize_normal --no_visualize_pose"

epoch="--niter 8000 --which_epoch 1380" #

basic="--eikonal_lambda 0.5 --adjust_gamma --gamma_lb 20 --min_surf_lambda 1.5 --deltasdf --sampler_std 15 --input_ch_views 3 --white_bg --depth 4 --style_dim 64 --input_ch_views 3 --N_samples ${RNUM} --uvVol_smpl_pts ${SNUM} --w_D_grad 0 --glr 2e-4 --dlr 4e-5 --df_ab_nerf_rec --w_tex_rec 0.1 --w_Face 5 --w_nerf_rec 2"

diff=" --load_test_samples --test_diff_part 20 --diffusion_name r2_x0Clip_point --eval_metrics --diffusion_epoch 240000 --diffusion_step 350 --samples_dir $data_dir/result/trained_model/${modelname}/samples"

cmd="${config} ${basic} ${setup} ${net} ${netD} ${ab} ${debug} ${render} ${epoch} --expname ${modelname} ${diff} --save_name test_results_${modelname} --data_dir $data_dir"

if [ -z "$gpu" ]
  then
    gpu=$CUDA_VISIBLE_DEVICES
fi

#assume 8 gpus
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export NVIDIA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python -W ignore generation.py ${cmd} --test_eval --gpu_ids $gpu --gpus $gpu 
