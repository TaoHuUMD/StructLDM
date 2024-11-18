#!/bin/bash
. scripts/unit.sh

train_sample=$1
gpu=$2

norm_method="point"
model_name="r2_x0Clip_${norm_method}"
data_name="renderpeople"
OPENAI_LOGDIR="./data/result/trained_model/${data_name}"

MODEL_FLAGS="--image_size 128 --num_channels 128 --num_res_blocks 3 --learn_sigma True --log_dir ${OPENAI_LOGDIR}"
DIFFUSION_FLAGS="--diffusion_steps 1000 --noise_schedule cosine"
TRAIN_FLAGS="--lr 1e-4 --batch_size 8"

LATENT_FLAGS="--data_name ${data_name} --lat_reso 128 --lat_type BF --lat_dim 16 --in_channels 16 --epoch 4260 --src_model_name FAD2DHybrid_4S_RPGAN_4_BF128_S1281632 --model_name ${model_name} --lat_num 700 --lat_valid_num 670 --predict_xstart True --clip_x0_training True"

cmd="${MODEL_FLAGS} ${DIFFUSION_FLAGS} ${TRAIN_FLAGS} ${LATENT_FLAGS} --normalization ${norm_method}" 
batch_api "${gpu}" "${cmd}" "${train_sample}" "-1"

exit