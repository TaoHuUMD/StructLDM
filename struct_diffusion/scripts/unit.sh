
batch_api () {

  gpu=$1  
  othercmd=$2
  flag=$3

  echo $SAMPLING_FLAGS

  if [ -z "$flag" ] || [ "${flag}" = "" ]
    then
      echo "train or test"
      exit
  fi

  echo "${flag}"

  if [ -z "$gpu" ]
    then
      gpu=$CUDA_VISIBLE_DEVICES
  fi
  
  cmd_train="${othercmd}"

  cmd_test_s1_fit_eval="${othercmd} --test_eval" #--force_evaluation
  cmd_test_s1_fit="${othercmd}" #--force_evaluation

  if [ "${flag}" = "" ];
  then
    flag="all"
  fi

  if [ $flag = "mtrain" ] || [ $flag = "t" ]
    then 
    batch "${gpu}" "mtrain" "${cmd_train}"   #"${test_epoch}"

  elif [ $flag = "sample" ] || [ $flag = "s" ]
    then 
    batch "${gpu}" "sample" "${othercmd} ${SAMPLING_FLAGS}" #"${test_epoch}"

  fi

}

is_continue () {
  RETURN=$1
  script_name=$2
  if [ $RETURN -eq 0 ];
  then
    echo "The script ${script_name} was executed successfuly $RETURN"
  else
    echo "The script ${script_name} failed and returned the code $RETURN"
    exit $RETURN
  fi 
}

batch () {

  gpu=$1
  train_test_demo=$2
  extra_opt=$3
  
  echo "${train_test_demo}"

  if [ $train_test_demo = "mtrain" ]
    then
      export NVIDIA_VISIBLE_DEVICES=${gpu}
      export CUDA_VISIBLE_DEVICES=${gpu}

      MASTER_PORT=$((12000 + $RANDOM % 20000))
      gpures="${gpu//[^,]}"      
      NUM_GPU="${#gpures}"
      NUM_GPU=$((NUM_GPU+1))
      #set -x
      echo ${NUM_GPU}

      extra_opt="${extra_opt}"

      python -W ignore -m torch.distributed.launch --nproc_per_node=${NUM_GPU} --master_port=${MASTER_PORT} latent_train.py ${extra_opt}

      RETURN=$?
      is_continue $RETURN "this mtrain"


    elif [ $train_test_demo = "sample" ]
      then

        MASTER_PORT=$((12000 + $RANDOM % 20000))
        gpures="${gpu//[^,]}"      
        NUM_GPU="${#gpures}"
        NUM_GPU=$((NUM_GPU+1))
        echo ${NUM_GPU}

        export NVIDIA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
        export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
        for gpuid in $(seq 0 $((NUM_GPU - 1))); do          
            python -W ignore latent_sample.py ${extra_opt} --gpu_ids $gpuid --all_gpus $gpu & 
        done
        wait
        wait
        exit

        
      fi
  fi
}

