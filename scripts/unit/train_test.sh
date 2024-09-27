
test_train=$1

extra_opt=$2

test_epochs=$3

total_parallel=$4

gpu=$5

#gpures="${gpu//[^,]}"      
#NUM_GPU="${#gpures}"
#NUM_GPU=$((NUM_GPU+1))


demo_name="demo_"


test_debug="0"
demo_debug="1"

#not run test, only evaluate
only_eva_in_test="0"

is_continue () {
  RETURN=$1
  script_name=$2
  if [ $RETURN -eq 0 ];
  then
    echo "The script ${script_name} was executed successfuly $RETURN"
    #exit 0
  else
    echo "The script ${script_name} failed and returned the code $RETURN"
    exit $RETURN
  fi 
}

mtrain () {

  python -m torch.distributed.run -W ignore --nnodes=1 ${extra_opt}

  RETURN=$?
  is_continue $RETURN "mtrain"

}


test () {
  
  is_crop_bbox="--crop_bbox"

  cmds="${extra_opt} --dataroot notknown --label_nc 0 --loadSize 1056 --n_dptex_s 512 --renderer normal_lookup  --no_instance " #--subdir $subdir --gpu_ids 0
  
  #test on single gpu
  python -W ignore test.py ${cmds} --which_epoch $test_epochs --pid 0 --total_parallel 1 --gpu_ids 0 &


RETURN=$?
is_continue $RETURN "test"

}






if [ $test_train = "mtrain" ]
  then 
    mtrain
  else    
      if [ $test_train = "test" ]
        then
          test
      
  fi
fi