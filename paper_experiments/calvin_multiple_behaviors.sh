#!/bin/bash

# FORMATTING 
# -[behavior name]_[modifier]_[seed]
# -modifier includes none (ours), rejection, and goal 

# this bash script should handle the full experiment upon a single run 

cleanup() {
    echo "Terminating all background processes..."
    pkill -P $$  # Kill all child processes of this script
    exit 1
}

# Trap Ctrl+C (SIGINT) and call cleanup function
trap cleanup SIGINT 

num_trials=50


dynamics_model="your dynamics model here"
base_policies=(
    "your base policies here"
)

count=0
gpudevice=3
exp_root="your directory here"



############ FOR THE ARTICULATED OBJECTS 
for ckpt_dir in "${base_policies[@]}"; do
    echo "On checkpoint $ckpt_dir"

    run_name=drawer_dont_open_$count
    exp_setup_config=yourconfigdirectory/drawer_dont_open_positives.json
    output_folder=$exp_root$run_name
    CUDA_VISIBLE_DEVICES=4 python run_dynaguide.py  --video_path $output_folder/$run_name.mp4 \
        --dataset_path $exp_root$run_name/$run_name.hdf5 --dataset_obs --json_path $output_folder/$run_name.json --horizon 400 --n_rollouts $num_trials \
        --agent $ckpt_dir --output_folder $exp_root/$run_name --video_skip 2  \
        --exp_setup_config $exp_setup_config --guidance $dynamics_model --camera_names third_person --scale 1.5 --ss 4 --alpha 30 &
    echo $run_name 

    run_name=door_dont_right_$count
    exp_setup_config=yourconfigdirectory/door_dont_right_positives.json
    output_folder=$exp_root$run_name
    CUDA_VISIBLE_DEVICES=4 python run_dynaguide.py  --video_path $output_folder/$run_name.mp4 \
        --dataset_path $output_folder/$run_name.hdf5 --dataset_obs --json_path $output_folder/$run_name.json --horizon 400 --n_rollouts $num_trials \
        --agent $ckpt_dir --output_folder $exp_root/$run_name --video_skip 2  \
        --exp_setup_config $exp_setup_config --guidance $dynamics_model --camera_names third_person --scale 1.5 --ss 4 --alpha 30 &
    echo $run_name 

    run_name=drawer_dont_open_door_dont_right_$count
    exp_setup_config=yourconfigdirectory/drawer_dont_open_door_dont_right_positives.json
    output_folder=$exp_root$run_name
    CUDA_VISIBLE_DEVICES=1 python run_dynaguide.py  --video_path $output_folder/$run_name.mp4 \
        --dataset_path $output_folder/$run_name.hdf5 --dataset_obs --json_path $output_folder/$run_name.json --horizon 400 --n_rollouts $num_trials \
        --agent $ckpt_dir --output_folder $output_folder --video_skip 2  \
        --exp_setup_config $exp_setup_config --guidance $dynamics_model --camera_names third_person --scale 1 --ss 4 --alpha 40 & 
    echo $run_name 


    run_name=door_left_button_on_$count
    exp_setup_config=yourconfigdirectory/door_left_button_on_negs.json
    output_folder=$exp_root$run_name
    CUDA_VISIBLE_DEVICES=2 python run_dynaguide.py  --video_path $output_folder/$run_name.mp4 \
        --dataset_path $output_folder/$run_name.hdf5 --dataset_obs --json_path $output_folder/$run_name.json --horizon 400 --n_rollouts $num_trials \
        --agent $ckpt_dir --output_folder $output_folder --video_skip 2  \
        --exp_setup_config $exp_setup_config --guidance $dynamics_model --camera_names third_person --scale 1 --ss 4 --alpha 40 & 
    echo $run_name 

    run_name=button_on_switch_on_$count
    exp_setup_config=yourconfigdirectory/button_on_switch_on_negs.json
    output_folder=$exp_root$run_name
    CUDA_VISIBLE_DEVICES=5 python run_dynaguide.py  --video_path $output_folder/$run_name.mp4 \
        --dataset_path $output_folder/$run_name.hdf5 --dataset_obs --json_path $output_folder/$run_name.json --horizon 400 --n_rollouts $num_trials \
        --agent $ckpt_dir --output_folder $output_folder --video_skip 2  \
        --exp_setup_config $exp_setup_config --guidance $dynamics_model --camera_names third_person --scale 1 --ss 4 --alpha 30 & 
    echo $run_name 

    run_name=switch_on_door_left_button_on_$count
    exp_setup_config=yourconfigdirectory/switch_on_door_left_button_on_negs.json
    output_folder=$exp_root$run_name
    CUDA_VISIBLE_DEVICES=3 python run_dynaguide.py  --video_path $output_folder/$run_name.mp4 \
        --dataset_path $output_folder/$run_name.hdf5 --dataset_obs --json_path $output_folder/$run_name.json --horizon 400 --n_rollouts $num_trials \
        --agent $ckpt_dir --output_folder $output_folder --video_skip 2  \
        --exp_setup_config $exp_setup_config --guidance $dynamics_model --camera_names third_person --scale 1 --ss 4 --alpha 30 & 
    echo $run_name 

    wait # wait this batch to stop 

    ((count++))
done
wait


######## ARTICULATED SAMPLING BASELINE
count=0
for ckpt_dir in "${base_policies[@]}"; do
    echo "On checkpoint $ckpt_dir"

    run_name=drawer_dont_open_planner_$count
    exp_setup_config=yourconfigdirectory/drawer_dont_open_positives.json
    output_folder=$exp_root$run_name
    CUDA_VISIBLE_DEVICES=1 python run_sampling_baseline.py  --video_path $output_folder/$run_name.mp4 \
        --dataset_path $output_folder/$run_name.hdf5 --dataset_obs --json_path $output_folder/$run_name.json --horizon 200 --n_rollouts 50 \
        --agent $ckpt_dir --output_folder $output_folder --video_skip 5  \
        --exp_setup_config $exp_setup_config --guidance $dynamics_model --camera_names third_person --num_samples 5 &
    echo $run_name 

    run_name=door_dont_right_planner_$count
    exp_setup_config=yourconfigdirectory/door_dont_right_positives.json
    output_folder=$exp_root$run_name
    CUDA_VISIBLE_DEVICES=1 python run_sampling_baseline.py  --video_path $output_folder/$run_name.mp4 \
        --dataset_path $output_folder/$run_name.hdf5 --dataset_obs --json_path $output_folder/$run_name.json --horizon 200 --n_rollouts 50 \
        --agent $ckpt_dir --output_folder $output_folder --video_skip 5  \
        --exp_setup_config $exp_setup_config --guidance $dynamics_model --camera_names third_person --num_samples 5 &
    echo $run_name

    run_name=drawer_dont_open_door_dont_right_planner_$count
    exp_setup_config=yourconfigdirectory/drawer_dont_open_door_dont_right_positives.json
    output_folder=$exp_root$run_name
    CUDA_VISIBLE_DEVICES=2 python run_sampling_baseline.py  --video_path $output_folder/$run_name.mp4 \
        --dataset_path $output_folder/$run_name.hdf5 --dataset_obs --json_path $output_folder/$run_name.json --horizon 200 --n_rollouts 50 \
        --agent $ckpt_dir --output_folder $output_folder --video_skip 5  \
        --exp_setup_config $exp_setup_config --guidance $dynamics_model --camera_names third_person --num_samples 5 &
    echo $run_name 

    run_name=door_left_button_on_planner_$count
    exp_setup_config=yourconfigdirectory/door_left_button_on_negs.json
    output_folder=$exp_root$run_name
    CUDA_VISIBLE_DEVICES=2 python run_sampling_baseline.py  --video_path $output_folder/$run_name.mp4 \
        --dataset_path $output_folder/$run_name.hdf5 --dataset_obs --json_path $output_folder/$run_name.json --horizon 200 --n_rollouts 50 \
        --agent $ckpt_dir --output_folder $output_folder --video_skip 5  \
        --exp_setup_config $exp_setup_config --guidance $dynamics_model --camera_names third_person --num_samples 5 &
    echo $run_name 

    run_name=button_on_switch_on_planner_$count
    exp_setup_config=yourconfigdirectory/button_on_switch_on_negs.json
    output_folder=$exp_root$run_name
    CUDA_VISIBLE_DEVICES=3 python run_sampling_baseline.py  --video_path $output_folder/$run_name.mp4 \
        --dataset_path $output_folder/$run_name.hdf5 --dataset_obs --json_path $output_folder/$run_name.json --horizon 200 --n_rollouts 50 \
        --agent $ckpt_dir --output_folder $output_folder --video_skip 5  \
        --exp_setup_config $exp_setup_config --guidance $dynamics_model --camera_names third_person --num_samples 5 &
    echo $run_name 

    run_name=switch_on_door_left_button_on_planner_$count
    exp_setup_config=yourconfigdirectory/switch_on_door_left_button_on_negs.json
    output_folder=$exp_root$run_name
    CUDA_VISIBLE_DEVICES=3 python run_sampling_baseline.py  --video_path $output_folder/$run_name.mp4 \
        --dataset_path $output_folder/$run_name.hdf5 --dataset_obs --json_path $output_folder/$run_name.json --horizon 200 --n_rollouts 50 \
        --agent $ckpt_dir --output_folder $output_folder --video_skip 5  \
        --exp_setup_config $exp_setup_config --guidance $dynamics_model --camera_names third_person --num_samples 5 &
    echo $run_name 

    wait # make sure that you don't set off too many at once 

    ((count++))

done
wait


echo "All configurations processed successfully."
