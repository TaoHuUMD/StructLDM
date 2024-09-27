. scripts/unit/method_setup.sh
. scripts/unit/train_test.sh

none=""
debug="train_test"
test="test_pose_dense"

tep="-1"

## direct
train_err="../log/${trained_model}_train_err.txt"
test_err="../log/${trained_model}_test_err.txt"
demo_err="../log/${trained_model}_demo_err.txt"

free_view="--make_demo --free_view_rot_smpl"

motion_="--debug_mode" 
motion_aug="--debug_mode --aug_nr --small_rot"


batch_motion_api () {

  gpu=$1
  method=$2
  modelname=$3
  othercmd=$4
  flag=$5

  if [ -z "$gpu" ]
    then
      gpu=$CUDA_VISIBLE_DEVICES
  fi

  if [ $flag = "mtrain" ]
    then 
    batch_motion ${gpu} ${method} "mtrain" "" "${modelname}" "${othercmd}" "-1"

  elif [ $flag = "test" ]    
    then
    batch_motion ${gpu} ${method} "test" "" "${modelname}" "${othercmd} --test_step_size 30 --test_eval" "-1" "6"
  fi

}



batch_motion () {
  gpu=$1
  flag=$2
  train_test=$3
  isaug=$4
  pre=$5
  other=$6
  testepoch=$7 
  total_parallel=$8 #in testing

  augs="motion_${isaug}" #_${augs}

  vrnr_sets="--vrnr ${!augs} ${other} --debug_mode" #${!flag}

  batch $gpu $train_test "${vrnr_sets}" "${pre}_${flag}" "${testepoch}" "${total_parallel}"

  RETURN=$?
  echo "$RETURN FOR BATCH"
}

batch () {

  gpu=$1
  train_test=$2
  other=$3
  pre=$4
  testepoch=$5
  total_parallel=$6

  trained_model="${pre}"
  vrnr $gpu "${train_test}" "$trained_model" "${testepoch}" "${other}" "${demo_name}" "$total_parallel"

  RETURN=$?
  echo "$RETURN FOR VRNR"

}


vrnr () {

  gpu=$1
  train_test_demo=$2
  trained_model=$3
  test_epochs=$4
  input_cmd=$5
  total_parallel=$7

  other="${input_cmd}"
  extra_opt="--name ${trained_model} ${other}"

  echo "${train_test_demo}"

  if [ $train_test_demo = "mtrain" ]
    then
      export NVIDIA_VISIBLE_DEVICES=${gpu}
      export CUDA_VISIBLE_DEVICES=${gpu}

      MASTER_PORT=$((12000 + $RANDOM % 20000))
      gpures="${gpu//[^,]}"      
      NUM_GPU="${#gpures}"
      NUM_GPU=$((NUM_GPU+1))
      echo ${NUM_GPU}
      extra_opt="${extra_opt} --gpu_ids ${gpu}"

      python -W ignore -m torch.distributed.launch --nproc_per_node=${NUM_GPU} --master_port=${MASTER_PORT} train_dist.py ${extra_opt}
      RETURN=$?
      is_continue $RETURN "this mtrain"
  
  elif [ $train_test_demo = "train" ]
    then
      export CUDA_DEVICE_ORDER=PCI_BUS_ID
      export NVIDIA_VISIBLE_DEVICES=${gpu}
      export CUDA_VISIBLE_DEVICES=${gpu}

      MASTER_PORT=$((12000 + $RANDOM % 20000))
      gpures="${gpu//[^,]}"      
      NUM_GPU="${#gpures}"
      NUM_GPU=$((NUM_GPU+1))
      #set -x
      echo ${NUM_GPU}
      extra_opt="${extra_opt} --gpu_ids ${gpu}"

      python -W ignore train_dist.py ${extra_opt}
      
      RETURN=$?
      is_continue $RETURN "train"

  else
      export CUDA_DEVICE_ORDER=PCI_BUS_ID
      export NVIDIA_VISIBLE_DEVICES=${gpu}
      export CUDA_VISIBLE_DEVICES=${gpu}
  
      if [ $train_test_demo = "test" ]
        then
        sh scripts/unit/train_test.sh "test" "${extra_opt}" "${test_epochs}" "${total_parallel}" "${gpu}"
  
  
      fi

  fi
}