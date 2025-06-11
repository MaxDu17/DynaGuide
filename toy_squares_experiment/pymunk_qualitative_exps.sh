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

dynamics_model=/store/real/maxjdu/repos/robotrainer/results/classifiers/Pymunk_classifier_FROMSCRATCH_100k_noised_ddim/11900.pth
base_policies=(
    "/store/real/maxjdu/repos/robotrainer/results/MunkCubeBasePolicy_ddim/20250203105846/models/model_epoch_200.pth"
    "/store/real/maxjdu/repos/robotrainer/results/MunkCubeBasePolicy_ddim_1/20250308135059/models/model_epoch_200.pth"
    "/store/real/maxjdu/repos/robotrainer/results/MunkCubeBasePolicy_ddim_2/20250309085109/models/model_epoch_200.pth"
    "/store/real/maxjdu/repos/robotrainer/results/MunkCubeBasePolicy_ddim_3/20250310030814/models/model_epoch_200.pth"
    "/store/real/maxjdu/repos/robotrainer/results/MunkCubeBasePolicy_ddim_4/20250310211754/models/model_epoch_200.pth"
    "/store/real/maxjdu/repos/robotrainer/results/MunkCubeBasePolicy_ddim_5/20250311152658/models/model_epoch_200.pth"
)

gpudevice=0
exp_root=/store/real/maxjdu/repos/robotrainer/FINAL_EXPERIMENTS/Pymunk_Qualitative/

count=0
for base_policy in "${base_policies[@]}"; do
    run_name=Avoid_Wall_0_5_$count
    output_folder=$exp_root$run_name
    CUDA_VISIBLE_DEVICES=$gpudevice python classifier_guidance_pymunk.py --video_path $output_folder/$run_name.mp4 \
        --dataset_path $output_folder/$run_name.hdf5 --dataset_obs --json_path $output_folder/$run_name.json --horizon 400 --n_rollouts 100 \
        --agent $base_policy --output_folder $output_folder --video_skip 1  \
        --guidance $dynamics_model --scale 0.5 --camera_names image --target_list 1,-0.33,-0.33,-0.33 --ss 4 --setup cube_blockade & 
    echo $run_name 

    run_name=Avoid_Wall_Control_$count
    output_folder=$exp_root$run_name
    CUDA_VISIBLE_DEVICES=$gpudevice python classifier_guidance_pymunk.py --video_path $output_folder/$run_name.mp4 \
        --dataset_path $output_folder/$run_name.hdf5 --dataset_obs --json_path $output_folder/$run_name.json --horizon 400 --n_rollouts 100 \
        --agent $base_policy --output_folder $output_folder --video_skip 1  \
        --guidance $dynamics_model --scale 0 --camera_names image --target_list 1,-0.33,-0.33,-0.33 --ss 4 --setup cube_blockade & 
    echo $run_name 


    wait 
    ((count++))


done
wait


# count=0
# for base_policy in "${base_policies[@]}"; do
#     run_name=EarlyDecision_0_5_$count
#     output_folder=$exp_root$run_name
#     CUDA_VISIBLE_DEVICES=$gpudevice python classifier_guidance_pymunk.py --video_path $output_folder/$run_name.mp4 \
#         --dataset_path $output_folder/$run_name.hdf5 --dataset_obs --json_path $output_folder/$run_name.json --horizon 400 --n_rollouts 100 \
#         --agent $base_policy --output_folder $output_folder --video_skip 1  \
#         --guidance $dynamics_model --scale 0.5 --camera_names image --target_list 1,-0.33,-0.33,-0.33 --ss 4 --setup early_decision &
#     echo $run_name 


#     run_name=EarlyDecision_Control_$count
#     output_folder=$exp_root$run_name
#     CUDA_VISIBLE_DEVICES=$gpudevice python classifier_guidance_pymunk.py --video_path $output_folder/$run_name.mp4 \
#         --dataset_path $output_folder/$run_name.hdf5 --dataset_obs --json_path $output_folder/$run_name.json --horizon 400 --n_rollouts 100 \
#         --agent $base_policy --output_folder $output_folder --video_skip 1  \
#         --guidance $dynamics_model --scale 0 --camera_names image --target_list 1,-0.33,-0.33,-0.33 --ss 1 --setup early_decision & 
#     echo $run_name 


#     wait 
#     ((count++))


# done
# wait


count=0
for base_policy in "${base_policies[@]}"; do
    run_name=LateDecision_0_5_$count
    output_folder=$exp_root$run_name
    CUDA_VISIBLE_DEVICES=$gpudevice python classifier_guidance_pymunk.py --video_path $output_folder/$run_name.mp4 \
        --dataset_path $output_folder/$run_name.hdf5 --dataset_obs --json_path $output_folder/$run_name.json --horizon 400 --n_rollouts 100 \
        --agent $base_policy --output_folder $output_folder --video_skip 1  \
        --guidance $dynamics_model --scale 0.5 --camera_names image --target_list 1,-0.33,-0.33,-0.33 --ss 4 --setup late_decision &
    echo $run_name 

    run_name=LateDecision_Control_$count
    output_folder=$exp_root$run_name
    CUDA_VISIBLE_DEVICES=$gpudevice python classifier_guidance_pymunk.py --video_path $output_folder/$run_name.mp4 \
        --dataset_path $output_folder/$run_name.hdf5 --dataset_obs --json_path $output_folder/$run_name.json --horizon 400 --n_rollouts 100 \
        --agent $base_policy --output_folder $output_folder --video_skip 1  \
        --guidance $dynamics_model --scale 0 --camera_names image --target_list 1,-0.33,-0.33,-0.33 --ss 1 --setup late_decision & 
    echo $run_name 


    wait 
    ((count++))


done
wait