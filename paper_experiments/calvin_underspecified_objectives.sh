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

dynamics_model="dynamics model here"
base_policies=(
   "your base policies here"
)
goal_conditioning_policies=(
    "your goal conditioned policies here"
)
count=0
gpudevice=5
exp_root="your experiment directory here" 


############ FOR THE ARTICULATED OBJECTS 
for ckpt_dir in "${base_policies[@]}"; do
    echo "On checkpoint $ckpt_dir"

            run_name=switch_on_$count
            exp_setup_config=yourconfigdirectory/ood_switch_on.json
            output_folder=$exp_root$run_name
            CUDA_VISIBLE_DEVICES=$gpudevice python run_dynaguide.py  --video_path $output_folder/$run_name.mp4 \
                --dataset_path $exp_root$run_name/$run_name.hdf5 --dataset_obs --json_path $output_folder/$run_name.json --horizon 400 --n_rollouts $num_trials \
                --agent $ckpt_dir --output_folder $exp_root/$run_name --video_skip 2  \
                --exp_setup_config $exp_setup_config --guidance $dynamics_model --camera_names third_person --scale 1.5 --ss 4 --alpha 30 &
            echo $run_name 

            run_name=switch_off_$count
            exp_setup_config=yourconfigdirectory/ood_switch_off.json
            output_folder=$exp_root$run_name
            CUDA_VISIBLE_DEVICES=$gpudevice python run_dynaguide.py  --video_path $output_folder/$run_name.mp4 \
                --dataset_path $output_folder/$run_name.hdf5 --dataset_obs --json_path $output_folder/$run_name.json --horizon 400 --n_rollouts $num_trials \
                --agent $ckpt_dir --output_folder $exp_root/$run_name --video_skip 2  \
                --exp_setup_config $exp_setup_config --guidance $dynamics_model --camera_names third_person --scale 1.5 --ss 4 --alpha 30 &
            echo $run_name 

    run_name=drawer_open_$count
    exp_setup_config=yourconfigdirectory/ood_drawer_open.json
    output_folder=$exp_root$run_name
    CUDA_VISIBLE_DEVICES=$gpudevice python run_dynaguide.py  --video_path $output_folder/$run_name.mp4 \
        --dataset_path $output_folder/$run_name.hdf5 --dataset_obs --json_path $output_folder/$run_name.json --horizon 400 --n_rollouts $num_trials \
        --agent $ckpt_dir --output_folder $output_folder --video_skip 2  \
        --exp_setup_config $exp_setup_config --guidance $dynamics_model --camera_names third_person --scale 1.5 --ss 4 --alpha 40 & 
    echo $run_name 

    run_name=drawer_close_$count
    exp_setup_config=yourconfigdirectory/ood_drawer_close.json
    output_folder=$exp_root$run_name
    CUDA_VISIBLE_DEVICES=$gpudevice python run_dynaguide.py  --video_path $output_folder/$run_name.mp4 \
        --dataset_path $output_folder/$run_name.hdf5 --dataset_obs --json_path $output_folder/$run_name.json --horizon 400 --n_rollouts $num_trials \
        --agent $ckpt_dir --output_folder $output_folder --video_skip 2  \
        --exp_setup_config $exp_setup_config --guidance $dynamics_model --camera_names third_person --scale 1.5 --ss 4 --alpha 40 & 
    echo $run_name 

        run_name=button_on_$count
        exp_setup_config=yourconfigdirectory/ood_button_on.json
        output_folder=$exp_root$run_name
        CUDA_VISIBLE_DEVICES=$gpudevice python run_dynaguide.py  --video_path $output_folder/$run_name.mp4 \
            --dataset_path $output_folder/$run_name.hdf5 --dataset_obs --json_path $output_folder/$run_name.json --horizon 400 --n_rollouts $num_trials \
            --agent $ckpt_dir --output_folder $output_folder --video_skip 2  \
            --exp_setup_config $exp_setup_config --guidance $dynamics_model --camera_names third_person --scale 1.6 --ss 4 --alpha 20 & 
        echo $run_name 

        run_name=button_off_$count
        exp_setup_config=yourconfigdirectory/ood_button_off.json
        output_folder=$exp_root$run_name
        CUDA_VISIBLE_DEVICES=$gpudevice python run_dynaguide.py  --video_path $output_folder/$run_name.mp4 \
            --dataset_path $output_folder/$run_name.hdf5 --dataset_obs --json_path $output_folder/$run_name.json --horizon 400 --n_rollouts $num_trials \
            --agent $ckpt_dir --output_folder $output_folder --video_skip 2  \
            --exp_setup_config $exp_setup_config --guidance $dynamics_model --camera_names third_person --scale 1.6 --ss 4 --alpha 20 & 
        echo $run_name 

    run_name=door_left_$count
    exp_setup_config=yourconfigdirectory/ood_door_left.json
    output_folder=$exp_root$run_name
    CUDA_VISIBLE_DEVICES=$gpudevice python run_dynaguide.py  --video_path $output_folder/$run_name.mp4 \
        --dataset_path $output_folder/$run_name.hdf5 --dataset_obs --json_path $output_folder/$run_name.json --horizon 400 --n_rollouts $num_trials \
        --agent $ckpt_dir --output_folder $output_folder --video_skip 2  \
        --exp_setup_config $exp_setup_config --guidance $dynamics_model --camera_names third_person --scale 1.2 --ss 4 --alpha 40 & 
    echo $run_name 

    used to be scale 1.8 and alpha 20 but we were a bit too aggressive I think 
    
    run_name=door_right_$count
    exp_setup_config=yourconfigdirectory/ood_door_right.json
    output_folder=$exp_root$run_name
    CUDA_VISIBLE_DEVICES=$gpudevice python run_dynaguide.py  --video_path $output_folder/$run_name.mp4 \
        --dataset_path $output_folder/$run_name.hdf5 --dataset_obs --json_path $output_folder/$run_name.json --horizon 400 --n_rollouts $num_trials \
        --agent $ckpt_dir --output_folder $output_folder  --video_skip 2  \
        --exp_setup_config $exp_setup_config --guidance $dynamics_model --camera_names third_person --scale 1.2 --ss 4 --alpha 15 & 
    echo $run_name 

    wait # wait this batch to stop 

    ((count++))
done
wait

######## ARTICULATED SAMPLING 
count=0
for ckpt_dir in "${base_policies[@]}"; do
    echo "On checkpoint $ckpt_dir"
    run_name=switch_on_planner_$count
    exp_setup_config=yourconfigdirectory/ood_switch_on.json
    output_folder=$exp_root$run_name
    CUDA_VISIBLE_DEVICES=$gpudevice python run_sampling_baseline.py  --video_path $output_folder/$run_name.mp4 \
        --dataset_path $output_folder/$run_name.hdf5 --dataset_obs --json_path $output_folder/$run_name.json --horizon 200 --n_rollouts 50 \
        --agent $ckpt_dir --output_folder $output_folder --video_skip 5  \
        --exp_setup_config $exp_setup_config --guidance $dynamics_model --camera_names third_person --num_samples 5 &
    echo $run_name 

    run_name=switch_off_planner_$count
    exp_setup_config=yourconfigdirectory/ood_switch_off.json
    output_folder=$exp_root$run_name
    CUDA_VISIBLE_DEVICES=$gpudevice python run_sampling_baseline.py  --video_path $output_folder/$run_name.mp4 \
        --dataset_path $output_folder/$run_name.hdf5 --dataset_obs --json_path $output_folder/$run_name.json --horizon 200 --n_rollouts 50 \
        --agent $ckpt_dir --output_folder $output_folder --video_skip 5  \
        --exp_setup_config $exp_setup_config --guidance $dynamics_model --camera_names third_person --num_samples 5 &
    echo $run_name`

    run_name=button_off_planner_$count
    exp_setup_config=yourconfigdirectory/ood_button_off.json
    output_folder=$exp_root$run_name
    CUDA_VISIBLE_DEVICES=$gpudevice python run_sampling_baseline.py  --video_path $output_folder/$run_name.mp4 \
        --dataset_path $output_folder/$run_name.hdf5 --dataset_obs --json_path $output_folder/$run_name.json --horizon 200 --n_rollouts 50 \
        --agent $ckpt_dir --output_folder $output_folder --video_skip 5  \
        --exp_setup_config $exp_setup_config --guidance $dynamics_model --camera_names third_person --num_samples 5 &
    echo $run_name 

    run_name=button_on_planner_$count
    exp_setup_config=yourconfigdirectory/ood_button_on.json
    output_folder=$exp_root$run_name
    CUDA_VISIBLE_DEVICES=$gpudevice python run_sampling_baseline.py  --video_path $output_folder/$run_name.mp4 \
        --dataset_path $output_folder/$run_name.hdf5 --dataset_obs --json_path $output_folder/$run_name.json --horizon 200 --n_rollouts 50 \
        --agent $ckpt_dir --output_folder $output_folder --video_skip 5  \
        --exp_setup_config $exp_setup_config --guidance $dynamics_model --camera_names third_person --num_samples 5 &
    echo $run_name 

    run_name=door_left_planner_$count
    exp_setup_config=yourconfigdirectory/ood_door_left.json
    output_folder=$exp_root$run_name
    CUDA_VISIBLE_DEVICES=$gpudevice python run_sampling_baseline.py  --video_path $output_folder/$run_name.mp4 \
        --dataset_path $output_folder/$run_name.hdf5 --dataset_obs --json_path $output_folder/$run_name.json --horizon 200 --n_rollouts 50 \
        --agent $ckpt_dir --output_folder $output_folder --video_skip 5  \
        --exp_setup_config $exp_setup_config --guidance $dynamics_model --camera_names third_person --num_samples 4 &
    echo $run_name 

    run_name=door_right_planner_$count
    exp_setup_config=yourconfigdirectory/ood_door_right.json
    output_folder=$exp_root$run_name
    CUDA_VISIBLE_DEVICES=$gpudevice python run_sampling_baseline.py  --video_path $output_folder/$run_name.mp4 \
        --dataset_path $output_folder/$run_name.hdf5 --dataset_obs --json_path $output_folder/$run_name.json --horizon 200 --n_rollouts 50 \
        --agent $ckpt_dir --output_folder $output_folder --video_skip 5  \
        --exp_setup_config $exp_setup_config --guidance $dynamics_model --camera_names third_person --num_samples 4 &
    echo $run_name 

    run_name=drawer_open_planner_$count
    exp_setup_config=yourconfigdirectory/ood_drawer_open.json
    output_folder=$exp_root$run_name
    CUDA_VISIBLE_DEVICES=$gpudevice python run_sampling_baseline.py  --video_path $output_folder/$run_name.mp4 \
        --dataset_path $output_folder/$run_name.hdf5 --dataset_obs --json_path $output_folder/$run_name.json --horizon 200 --n_rollouts 50 \
        --agent $ckpt_dir --output_folder $output_folder --video_skip 5  \
        --exp_setup_config $exp_setup_config --guidance $dynamics_model --camera_names third_person --num_samples 5 &
    echo $run_name 

    run_name=drawer_close_planner_$count
    exp_setup_config=yourconfigdirectory/ood_drawer_close.json
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

########### GOAL CONDITIONING
count=0
for ckpt_dir in "${goal_conditioning_policies[@]}"; do
    echo "On checkpoint $ckpt_dir"

    run_name=switch_on_gc_$count
    exp_setup_config=yourconfigdirectory/ood_switch_on.json
    output_folder=$exp_root$run_name
    goal_dir=yourgoaldirectory/ood_switch_on
    CUDA_VISIBLE_DEVICES=$gpudevice python run_dynaguide.py  --video_path $output_folder/$run_name.mp4 \
        --dataset_path $exp_root$run_name/$run_name.hdf5 --dataset_obs --json_path $output_folder/$run_name.json --horizon 400 --n_rollouts $num_trials \
        --agent $ckpt_dir --output_folder $exp_root/$run_name --video_skip 2  \
        --exp_setup_config $exp_setup_config --goal_dir $goal_dir --guidance $dynamics_model --camera_names third_person --scale 0 --ss 1 --alpha 30 &
    echo $run_name 

    run_name=switch_off_gc_$count
    exp_setup_config=yourconfigdirectory/ood_switch_off.json
    output_folder=$exp_root$run_name
    goal_dir=yourgoaldirectory/ood_switch_off
    CUDA_VISIBLE_DEVICES=$gpudevice python run_dynaguide.py  --video_path $output_folder/$run_name.mp4 \
        --dataset_path $output_folder/$run_name.hdf5 --dataset_obs --json_path $output_folder/$run_name.json --horizon 400 --n_rollouts $num_trials \
        --agent $ckpt_dir --output_folder $exp_root/$run_name --video_skip 2  \
        --exp_setup_config $exp_setup_config --goal_dir $goal_dir --guidance $dynamics_model --camera_names third_person --scale 0 --ss 1 --alpha 30 &
    echo $run_name 

    run_name=drawer_open_gc_$count
    exp_setup_config=yourconfigdirectory/ood_drawer_open.json
    output_folder=$exp_root$run_name
    goal_dir=yourgoaldirectory/ood_drawer_open
    CUDA_VISIBLE_DEVICES=$gpudevice python run_dynaguide.py  --video_path $output_folder/$run_name.mp4 \
        --dataset_path $output_folder/$run_name.hdf5 --dataset_obs --json_path $output_folder/$run_name.json --horizon 400 --n_rollouts $num_trials \
        --agent $ckpt_dir --output_folder $output_folder --video_skip 2  \
        --exp_setup_config $exp_setup_config --goal_dir $goal_dir --guidance $dynamics_model --camera_names third_person --scale 0 --ss 1 --alpha 40 & 
    echo $run_name 

    run_name=drawer_close_gc_$count
    exp_setup_config=yourconfigdirectory/ood_drawer_close.json
    output_folder=$exp_root$run_name
    goal_dir=yourgoaldirectory/ood_drawer_close
    CUDA_VISIBLE_DEVICES=$gpudevice python run_dynaguide.py  --video_path $output_folder/$run_name.mp4 \
        --dataset_path $output_folder/$run_name.hdf5 --dataset_obs --json_path $output_folder/$run_name.json --horizon 400 --n_rollouts $num_trials \
        --agent $ckpt_dir --output_folder $output_folder --video_skip 2  \
        --exp_setup_config $exp_setup_config --goal_dir $goal_dir --guidance $dynamics_model --camera_names third_person --scale 0 --ss 1 --alpha 40 & 
    echo $run_name 

    run_name=button_on_gc_$count
    exp_setup_config=yourconfigdirectory/ood_button_on.json
    output_folder=$exp_root$run_name
    goal_dir=yourgoaldirectory/ood_button_on
    CUDA_VISIBLE_DEVICES=$gpudevice python run_dynaguide.py  --video_path $output_folder/$run_name.mp4 \
        --dataset_path $output_folder/$run_name.hdf5 --dataset_obs --json_path $output_folder/$run_name.json --horizon 400 --n_rollouts $num_trials \
        --agent $ckpt_dir --output_folder $output_folder --video_skip 2  \
        --exp_setup_config $exp_setup_config --goal_dir $goal_dir --guidance $dynamics_model --camera_names third_person --scale 0 --ss 1 --alpha 30 & 
    echo $run_name 

    run_name=button_off_gc_$count
    exp_setup_config=yourconfigdirectory/ood_button_off.json
    output_folder=$exp_root$run_name
    goal_dir=yourgoaldirectory/ood_button_off
    CUDA_VISIBLE_DEVICES=$gpudevice python run_dynaguide.py  --video_path $output_folder/$run_name.mp4 \
        --dataset_path $output_folder/$run_name.hdf5 --dataset_obs --json_path $output_folder/$run_name.json --horizon 400 --n_rollouts $num_trials \
        --agent $ckpt_dir --output_folder $output_folder --video_skip 2  \
        --exp_setup_config $exp_setup_config --goal_dir $goal_dir --guidance $dynamics_model --camera_names third_person --scale 0 --ss 1 --alpha 30 & 
    echo $run_name 

    run_name=door_left_gc_$count
    exp_setup_config=yourconfigdirectory/ood_door_left.json
    output_folder=$exp_root$run_name
    goal_dir=yourgoaldirectory/ood_door_left
    CUDA_VISIBLE_DEVICES=$gpudevice python run_dynaguide.py  --video_path $output_folder/$run_name.mp4 \
        --dataset_path $output_folder/$run_name.hdf5 --dataset_obs --json_path $output_folder/$run_name.json --horizon 400 --n_rollouts $num_trials \
        --agent $ckpt_dir --output_folder $output_folder --video_skip 2  \
        --exp_setup_config $exp_setup_config --goal_dir $goal_dir --guidance $dynamics_model --camera_names third_person --scale 0 --ss 1 --alpha 30 & 
    echo $run_name 

    run_name=door_right_gc_$count
    exp_setup_config=yourconfigdirectory/ood_door_right.json
    output_folder=$exp_root$run_name
    goal_dir=yourgoaldirectory/ood_door_right
    CUDA_VISIBLE_DEVICES=$gpudevice python run_dynaguide.py  --video_path $output_folder/$run_name.mp4 \
        --dataset_path $output_folder/$run_name.hdf5 --dataset_obs --json_path $output_folder/$run_name.json --horizon 400 --n_rollouts $num_trials \
        --agent $ckpt_dir --output_folder $output_folder  --video_skip 2  \
        --exp_setup_config $exp_setup_config --goal_dir $goal_dir --guidance $dynamics_model --camera_names third_person --scale 0 --ss 1 --alpha 30 & 
    echo $run_name 

    wait # wait this batch to stop 

    ((count++))
done
wait


echo "All configurations processed successfully."