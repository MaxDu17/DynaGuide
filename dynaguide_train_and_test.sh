#!/bin/bash
base_folder=results

cleanup() {
    echo "Terminating all background processes..."
    pkill -P $$  # Kill all child processes of this script
    exit 1
}

# Trap Ctrl+C (SIGINT) and call cleanup function
trap cleanup SIGINT 

# run_name=TestName
# output_folder=youroutputfolder/$run_name
# checkpoint_dir=path_to_base_policy
# exp_setup_config=path_to_setup_configs/switch_off.json
# embedder=path_to_dynamics_model
# CUDA_VISIBLE_DEVICES=0 python run_dynaguide.py  --video_path $output_folder/$run_name.mp4 \
#     --dataset_path $output_folder/$run_name.hdf5 --dataset_obs --json_path $output_folder/$run_name.json --horizon 400 --n_rollouts 100 \
#     --agent $checkpoint_dir --output_folder $output_folder --video_skip 2  \
#     --exp_setup_config $exp_setup_config --guidance $embedder --camera_names third_person --scale 1 --save_frames

# experiment_name=TestName
# CUDA_VISIBLE_DEVICES=1 python train_dynaguide.py --exp_dir results/dynamics_models/$experiment_name/ \
#     --train_hdf5 path_to_train_data  \
#     --test_hdf5 path_to_test_data \
#     --cameras third_person --action_dim 7 --proprio_key proprio --proprio_dim 15 \
#     --num_epochs 6000 --action_chunk_length 16 --batch_size 16 

# experiment_name=name_of_trained_dynamics_model
# checkpoint=3000
# CUDA_VISIBLE_DEVICES=7 python test_dynaguide_embedding.py --exp_dir path_to_experiment_directory/$experiment_name/ \
#     --good_hdf5 path_to_desired_behavior_dataset  \
#     --mixed_hdf5 path_to_labeled_test_set  \
#     --checkpoint path_to_dynamics_model_directory$experiment_name/$checkpoint.pth  \
#     --action_chunk_length 16 --key button_off 


# Train base policy 

# Train and test dynamics model 
# experiment_name=TESTEST
# CUDA_VISIBLE_DEVICES=1 python train_dynaguide.py --exp_dir results/dynamics_models/$experiment_name/ \
#     --train_hdf5 /store/real/maxjdu/repos/robotrainer/dataset/CalvinABCD_betterseg/data.hdf5  \
#     --test_hdf5 /store/real/maxjdu/repos/robotrainer/dataset/CalvinDD_validation_better_seg/data.hdf5 \
#     --cameras third_person --action_dim 7 --proprio_key proprio --proprio_dim 15 \
#     --num_epochs 6000 --action_chunk_length 16 --batch_size 16 

# experiment_name=CALVINABCD_blockseg_newmodel_8head_6_depth_withlearnedposembed_NOISED_wproprio
# checkpoint=3000
# CUDA_VISIBLE_DEVICES=7 python test_dynaguide_embedding.py --exp_dir /store/real/maxjdu/repos/robotrainer/results/classifiers/$experiment_name/ \
#     --good_hdf5 /store/real/maxjdu/repos/robotrainer/dataset/CalvinDD_validation_by_category_wcubes/button_off.hdf5 \
#     --mixed_hdf5 /store/real/maxjdu/repos/robotrainer/dataset/CalvinDD_validation_by_category_wcubes/labeled_test_set.hdf5  \
#     --checkpoint /store/real/maxjdu/repos/robotrainer/results/classifiers/$experiment_name/$checkpoint.pth  \
#     --action_chunk_length 16 --key button_off 


# run Dynaguide 
run_name=TESTTESTETS
output_folder=/store/real/maxjdu/repos/robotrainer/results/outputs/$run_name
checkpoint_dir=/store/real/maxjdu/repos/robotrainer/results/CalvinFullDataTrain_ddim/20250227201015/models/model_epoch_1000.pth
exp_setup_config=/store/real/maxjdu/repos/robotrainer/calvin_exp_configs/switch_off.json
embedder=/store/real/maxjdu/repos/robotrainer/results/classifiers/CALVINABCD_blockseg_newmodel_8head_6_depth_NOISED_wproprio_padsame/3000.pth
CUDA_VISIBLE_DEVICES=0 python run_dynaguide.py  --video_path $output_folder/$run_name.mp4 \
    --dataset_path $output_folder/$run_name.hdf5 --dataset_obs --json_path $output_folder/$run_name.json --horizon 400 --n_rollouts 100 \
    --agent $checkpoint_dir --output_folder $output_folder --video_skip 2  \
    --exp_setup_config $exp_setup_config --guidance $embedder --camera_names third_person --scale 1.5 --ss 4 --alpha 30 --save_frames

