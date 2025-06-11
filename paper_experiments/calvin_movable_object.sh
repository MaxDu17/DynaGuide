#!/bin/bash


# this bash script should handle the full experiment upon a single run 

cleanup() {
    echo "Terminating all background processes..."
    pkill -P $$  # Kill all child processes of this script
    exit 1
}

# Trap Ctrl+C (SIGINT) and call cleanup function
trap cleanup SIGINT 

num_trials=50 # how many times to evaluate 

dynamics_model="your dynamics model here"
base_policies=(
    "your base policies here"
)
goal_conditioning_policies=(
    "your goal conditioned policies here"
)
count=0
gpudevice=4
exp_root="your directory here"

cp calvin_articulated_and_block_exps.sh ${exp_root}calvin_articulated_and_block_exps.sh

######## FOR THE CONTROLS
count=0
for ckpt_dir in "${base_policies[@]}"; do
    echo "On checkpoint $ckpt_dir"
    run_name=Control_$count
    output_folder=$exp_root$run_name
    CUDA_VISIBLE_DEVICES=1 python run_dynaguide.py  --video_path $output_folder/$run_name.mp4 \
        --dataset_path $output_folder/$run_name.hdf5 --dataset_obs --json_path $output_folder/$run_name.json --horizon 400 --n_rollouts $num_trials \
        --agent $ckpt_dir --output_folder $exp_root/$run_name --video_skip 2  \
        --exp_setup_config $exp_setup_config --guidance $dynamics_model --camera_names third_person --scale 0 --ss 1 & 
    wait 
    ((count++))

done 
wait 

######## FOR THE CUBES
count=0
for ckpt_dir in "${base_policies[@]}"; do
    echo "On checkpoint $ckpt_dir"

    run_name=red_lift_$count
    exp_setup_config=yourconfigdirectory/red_lift.json
    output_folder=$exp_root$run_name
    CUDA_VISIBLE_DEVICES=$gpudevice python run_dynaguide.py  --video_path $output_folder/$run_name.mp4 \
        --dataset_path $output_folder/$run_name.hdf5 --dataset_obs --json_path $output_folder/$run_name.json --horizon 400 --n_rollouts $num_trials \
        --agent $ckpt_dir --output_folder $output_folder  --video_skip 2  \
        --exp_setup_config $exp_setup_config --guidance $dynamics_model --camera_names third_person --scale 2 --ss 4 --alpha 30 & 
    echo $run_name 

    run_name=blue_lift_$count
    exp_setup_config=yourconfigdirectory/blue_lift.json
    output_folder=$exp_root$run_name
    CUDA_VISIBLE_DEVICES=$gpudevice python run_dynaguide.py  --video_path $output_folder/$run_name.mp4 \
        --dataset_path $output_folder/$run_name.hdf5 --dataset_obs --json_path $output_folder/$run_name.json --horizon 400 --n_rollouts $num_trials \
        --agent $ckpt_dir --output_folder $output_folder  --video_skip 2  \
        --exp_setup_config $exp_setup_config --guidance $dynamics_model --camera_names third_person --scale 2 --ss 4 --alpha 30 & 
    echo $run_name 

    run_name=pink_lift_$count
    exp_setup_config=yourconfigdirectory/pink_lift.json
    output_folder=$exp_root$run_name
    CUDA_VISIBLE_DEVICES=$gpudevice python run_dynaguide.py  --video_path $output_folder/$run_name.mp4 \
        --dataset_path $output_folder/$run_name.hdf5 --dataset_obs --json_path $output_folder/$run_name.json --horizon 400 --n_rollouts $num_trials \
        --agent $ckpt_dir --output_folder $output_folder  --video_skip 2  \
        --exp_setup_config $exp_setup_config --guidance $dynamics_model --camera_names third_person --scale 2 --ss 4 --alpha 30 & 
    echo $run_name 
    wait # make sure that you don't set off too many at once 

    ((count++))

done 
wait 


### CUBES SAMPLING BASELINE
count=0
for ckpt_dir in "${base_policies[@]}"; do
    echo "On checkpoint $ckpt_dir"

    run_name=red_lift_planner_$count
    exp_setup_config=yourconfigdirectory/red_lift.json
    output_folder=$exp_root$run_name
    CUDA_VISIBLE_DEVICES=$gpudevice python run_sampling_baseline.py  --video_path $output_folder/$run_name.mp4 \
        --dataset_path $output_folder/$run_name.hdf5 --dataset_obs --json_path $output_folder/$run_name.json --horizon 200 --n_rollouts 50 \
        --agent $ckpt_dir --output_folder $output_folder --video_skip 5  \
        --exp_setup_config $exp_setup_config --guidance $dynamics_model --camera_names third_person --num_samples 5 &
    echo $run_name 

    run_name=blue_lift_planner_$count
    exp_setup_config=yourconfigdirectory/blue_lift.json
    output_folder=$exp_root$run_name
    CUDA_VISIBLE_DEVICES=$gpudevice python run_sampling_baseline.py  --video_path $output_folder/$run_name.mp4 \
        --dataset_path $output_folder/$run_name.hdf5 --dataset_obs --json_path $output_folder/$run_name.json --horizon 200 --n_rollouts 50 \
        --agent $ckpt_dir --output_folder $output_folder --video_skip 5  \
        --exp_setup_config $exp_setup_config --guidance $dynamics_model --camera_names third_person --num_samples 5 &
    echo $run_name

    run_name=pink_lift_planner_$count
    exp_setup_config=yourconfigdirectory/pink_lift.json
    output_folder=$exp_root$run_name
    CUDA_VISIBLE_DEVICES=$gpudevice python run_sampling_baseline.py  --video_path $output_folder/$run_name.mp4 \
        --dataset_path $output_folder/$run_name.hdf5 --dataset_obs --json_path $output_folder/$run_name.json --horizon 200 --n_rollouts 50 \
        --agent $ckpt_dir --output_folder $output_folder --video_skip 5  \
        --exp_setup_config $exp_setup_config --guidance $dynamics_model --camera_names third_person --num_samples 5 &
    echo $run_name 
    wait # make sure that you don't set off too many at once 

    ((count++))

done 
wait 

count=0
for ckpt_dir in "${goal_conditioning_policies[@]}"; do
    echo "On checkpoint $ckpt_dir"
    run_name=red_lift_gc_$count
    exp_setup_config=yourconfigdirectory/red_lift.json
    output_folder=$exp_root$run_name
    goal_dir=/store/real/maxjdu/repos/robotrainer/dataset/Calvin_Goals/red_lift
    CUDA_VISIBLE_DEVICES=$gpudevice python run_dynaguide.py  --video_path $output_folder/$run_name.mp4 \
        --dataset_path $exp_root$run_name/$run_name.hdf5 --dataset_obs --json_path $output_folder/$run_name.json --horizon 400 --n_rollouts $num_trials \
        --agent $ckpt_dir --output_folder $exp_root/$run_name --video_skip 2  \
        --exp_setup_config $exp_setup_config --goal_dir $goal_dir --guidance $dynamics_model --camera_names third_person --scale 0 --ss 1 --alpha 30 &
    echo $run_name 

    run_name=blue_lift_gc_$count
    exp_setup_config=yourconfigdirectory/blue_lift.json
    output_folder=$exp_root$run_name
    goal_dir=/store/real/maxjdu/repos/robotrainer/dataset/Calvin_Goals/blue_lift
    CUDA_VISIBLE_DEVICES=$gpudevice python run_dynaguide.py  --video_path $output_folder/$run_name.mp4 \
        --dataset_path $output_folder/$run_name.hdf5 --dataset_obs --json_path $output_folder/$run_name.json --horizon 400 --n_rollouts $num_trials \
        --agent $ckpt_dir --output_folder $exp_root/$run_name --video_skip 2  \
        --exp_setup_config $exp_setup_config --goal_dir $goal_dir --guidance $dynamics_model --camera_names third_person --scale 0 --ss 1 --alpha 30 &
    echo $run_name 

    run_name=pink_lift_gc_$count
    exp_setup_config=yourconfigdirectory/pink_lift.json
    output_folder=$exp_root$run_name
    goal_dir=/store/real/maxjdu/repos/robotrainer/dataset/Calvin_Goals/pink_lift
    CUDA_VISIBLE_DEVICES=$gpudevice python run_dynaguide.py  --video_path $output_folder/$run_name.mp4 \
        --dataset_path $output_folder/$run_name.hdf5 --dataset_obs --json_path $output_folder/$run_name.json --horizon 400 --n_rollouts $num_trials \
        --agent $ckpt_dir --output_folder $output_folder --video_skip 2  \
        --exp_setup_config $exp_setup_config --goal_dir $goal_dir --guidance $dynamics_model --camera_names third_person --scale 0 --ss 1 --alpha 40 & 
    echo $run_name 
    wait 

    ((count++))

done 
wait 

echo "All configurations processed successfully."