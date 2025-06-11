#!/bin/bash
base_folder=results

cleanup() {
    echo "Terminating all background processes..."
    pkill -P $$  # Kill all child processes of this script
    exit 1
}

# Trap Ctrl+C (SIGINT) and call cleanup function
trap cleanup SIGINT 


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


# run_name=pymunk_touch_res128_largercubes_repeated_100ktrain
# output_folder=/store/real/maxjdu/repos/robotrainer/dataset/pymunktouch/$run_name
# python collect_scripted_data_pymunk.py --video_path $output_folder/$run_name.mp4 \
#     --dataset_path $output_folder/data.hdf5 --dataset_obs --json_path $output_folder/config.json --horizon 150 --n_rollouts 100000 \
#     --env_config /store/real/maxjdu/repos/robotrainer/configs/touchcubes.json --output_folder $output_folder --keep_only_successful \
#     --camera_names image --video_skip 5 --repeat_environment

# experiment_name=Pymunk_classifier_FROMSCRATCH_100k_noised_ddim
# CUDA_VISIBLE_DEVICES=6 python train_end_state_classifier.py --exp_dir /store/real/maxjdu/repos/robotrainer/results/classifiers/$experiment_name/ \
#     --train_hdf5 /store/real/maxjdu/repos/robotrainer/dataset/pymunktouch/pymunk_touch_res128_largercubes_repeated_100ktrain/data.hdf5 \
#     --test_hdf5 /store/real/maxjdu/repos/robotrainer/dataset/pymunktouch/pymunk_touch_res128_largercubes_repeated_valid/data.hdf5 \
#     --num_epochs 12000 --action_chunk_length 16 --batch_size 16 --noised


### TESTING DYNAMICS
# experiment_name=Pymunk_classifier_FROMSCRATCH_100k_noised_ddim
# checkpoint=11900
# CUDA_VISIBLE_DEVICES=6 python test_end_state_classifier.py --exp_dir /store/real/maxjdu/repos/robotrainer/results/classifiers/$experiment_name/ \
#     --mixed_hdf5 /store/real/maxjdu/repos/robotrainer/dataset/pymunktouch/pymunk_touch_res128_largercubes_repeated_valid/data.hdf5  \
#     --checkpoint /store/real/maxjdu/repos/robotrainer/results/classifiers/$experiment_name/$checkpoint.pth  \
#     --action_chunk_length 16

