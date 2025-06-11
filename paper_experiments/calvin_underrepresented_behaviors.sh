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

dynamics_model=/store/real/maxjdu/repos/robotrainer/results/classifiers/CALVINABCD_blockseg_newmodel_8head_6_depth_NOISED_wproprio_padsame/3000.pth


# TRAINING CODE 
datasets=(
"your datasets here"
)

splits=(1 3 5 10 20 40 60 80)

count=0
gpudevice=4
exp_root=/store/real/maxjdu/repos/robotrainer/FINAL_EXPERIMENTS/Multi_Neg/
base_folder=/store/real/maxjdu/repos/robotrainer/results
count=0
for dataset in "${datasets[@]}"; do
    python split_train_val.py --dataset $dataset --ratio 0.03
    output_name=calvin_door_left_gc_${splits[$count]}_percent
    CUDA_VISIBLE_DEVICES=4 python train_base_policy.py --config configs/diffusion_policy_image_calvin_ddim.json --dataset $dataset --output $base_folder  --name $output_name &

    echo $output_name
    ((count++))
    echo $count 
    if [ $count = 4 ]; then 
        wait 
    fi 
done 
wait 

exit 


exp_root="your exp dir"
splits=(1 3 5 10 20 40 60 80)
gpudevice=5

base_policies=(
"your trained policies here (after you've trained them with script above)"
)

# # ############ FOR THE ARTICULATED OBJECTS 

count=0
for ckpt_dir in "${base_policies[@]}"; do
    for i in $(seq 0 5); do
        echo "On checkpoint $ckpt_dir"
        run_name=switch_on_${splits[$count]}_$i
        exp_setup_config=yourconfigdirectory/switch_on.json
        output_folder=$exp_root$run_name
        CUDA_VISIBLE_DEVICES=$gpudevice python run_dynaguide.py  --video_path $output_folder/$run_name.mp4 \
            --dataset_path $exp_root$run_name/$run_name.hdf5 --dataset_obs --json_path $output_folder/$run_name.json --horizon 400 --n_rollouts $num_trials \
            --agent $ckpt_dir --output_folder $exp_root/$run_name --video_skip 2  \
            --exp_setup_config $exp_setup_config --guidance $dynamics_model --camera_names third_person --scale 1.5 --ss 4 --alpha 30 &
        echo $run_name 
    done
    wait
     ((count++))
done

count=0
for ckpt_dir in "${base_policies[@]}"; do
    for i in $(seq 0 5); do  
        echo "On checkpoint $ckpt_dir"

        run_name=switch_on_control_${splits[$count]}_$i
        exp_setup_config=yourconfigdirectory/switch_on.json
        output_folder=$exp_root$run_name
        CUDA_VISIBLE_DEVICES=$gpudevice python run_dynaguide.py  --video_path $output_folder/$run_name.mp4 \
            --dataset_path $exp_root$run_name/$run_name.hdf5 --dataset_obs --json_path $output_folder/$run_name.json --horizon 400 --n_rollouts $num_trials \
            --agent $ckpt_dir --output_folder $exp_root/$run_name --video_skip 2  \
            --exp_setup_config $exp_setup_config --guidance $dynamics_model --camera_names third_person --scale 0 --ss 1 --alpha 30 &
        echo $run_name 
    done
    wait
    ((count++))
done

count=0
for ckpt_dir in "${base_policies[@]}"; do
    for i in $(seq 0 5); do 
        echo "On checkpoint $ckpt_dir"
        run_name=switch_on_planner_${splits[$count]}_$i
        exp_setup_config=yourconfigdirectory/switch_on.json
        output_folder=$exp_root$run_name
        CUDA_VISIBLE_DEVICES=$gpudevice python run_sampling_baseline.py  --video_path $output_folder/$run_name.mp4 \
            --dataset_path $output_folder/$run_name.hdf5 --dataset_obs --json_path $output_folder/$run_name.json --horizon 200 --n_rollouts $num_trials \
            --agent $ckpt_dir --output_folder $output_folder --video_skip 5  \
            --exp_setup_config $exp_setup_config --guidance $dynamics_model --camera_names third_person --num_samples 5 &
        echo $run_name 
    done
    wait
    ((count++))
done 
