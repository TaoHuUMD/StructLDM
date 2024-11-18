gpu=$1
TEST_SAMPLES=64

export SAMPLING_FLAGS="--timestep_respacing 100 --num_samples ${TEST_SAMPLES} --sample_batch_size 32 --use_ddim True"
sh scripts/exec.sh "sample" $gpu
wait
exit