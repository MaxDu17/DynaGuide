#!/bin/bash
base_folder=/store/real/maxjdu/repos/robotrainer/results

## COLLECTING DATA
# run_name=pymunk_touch_res128_largercubes_repeated_100ktrain
# output_folder=/store/real/maxjdu/repos/robotrainer/dataset/pymunktouch/$run_name
# python collect_scripted_data_pymunk.py --video_path $output_folder/$run_name.mp4 \
#     --dataset_path $output_folder/data.hdf5 --dataset_obs --json_path $output_folder/config.json --horizon 150 --n_rollouts 100000 \
#     --env_config /store/real/maxjdu/repos/robotrainer/configs/touchcubes.json --output_folder $output_folder --keep_only_successful \
#     --camera_names image --video_skip 5 --repeat_environment

# run_name=pymunk_touch_200_blue
# output_folder=/store/real/maxjdu/repos/robotrainer/dataset/pymunktouch/$run_name
# python collect_scripted_data_pymunk.py --video_path $output_folder/$run_name.mp4 \
#     --dataset_path $output_folder/data.hdf5 --dataset_obs --json_path $output_folder/config.json --horizon 150 --n_rollouts 200 \
#     --env_config /store/real/maxjdu/repos/robotrainer/configs/touchcubes.json --output_folder $output_folder --keep_only_successful \
#     --camera_names image --video_skip 5


## BASE POLICY TRAINING 
# CUDA_VISIBLE_DEVICES=6 python train.py --config configs/diffusion_policy_image_munk_ddim.json --dataset dataset/pymunktouch/pymunk_touch_res128_largercubes_repeated_10ktrain/data.hdf5 --output $base_folder  --name MunkCubeBasePolicy_25_percent
# CUDA_VISIBLE_DEVICES=3 python train.py --config configs/diffusion_policy_image_munk_ddim.json \
#     --dataset dataset/pymunktouch/pymunk_touch_0_percent_blue_10k/data.hdf5 --output $base_folder  --name MunkCubeBasePolicy_0_percent_10k


## DYNAMICS MODEL TRAINING
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

# PERFORMANCE: REJECTION SAMPLING

# run_name=REJECTION_pymunk_not_blue
# output_folder=/store/real/maxjdu/repos/robotrainer/results/outputs/$run_name
# checkpoint_dir=/store/real/maxjdu/repos/robotrainer/results/MunkCubeBasePolicy_onepointREPEAT_normed/20250128135649/models/model_epoch_1000.pth
# embedder=/store/real/maxjdu/repos/robotrainer/results/classifiers/Pymunk_classifier_repeated_normalized_oldencoder/5900.pth
# CUDA_VISIBLE_DEVICES=5 python rejection_sampling_pymunk.py --video_path $output_folder/$run_name.mp4 \
#     --dataset_path $output_folder/$run_name.hdf5 --dataset_obs --json_path $output_folder/$run_name.json --horizon 200 --n_rollouts 50 \
#     --agent $checkpoint_dir --output_folder $output_folder --video_skip 1  \
#     --guidance $embedder --camera_names image


####### PERFORMANCE: GUIDANCE
# run_name=GUIDANCE_pymunk_blue 
# output_folder=/store/real/maxjdu/repos/robotrainer/results/outputs/$run_name
# checkpoint_dir=/store/real/maxjdu/repos/robotrainer/results/MunkCubeBasePolicy_ddim/20250203105846/models/model_epoch_1000.pth
# embedder=/store/real/maxjdu/repos/robotrainer/results/classifiers/Pymunk_classifier_FROMSCRATCH_100k_noised_ddim/11900.pth
# CUDA_VISIBLE_DEVICES=5 python classifier_guidance_pymunk.py --video_path $output_folder/$run_name.mp4 \
#     --dataset_path $output_folder/$run_name.hdf5 --dataset_obs --json_path $output_folder/$run_name.json --horizon 400 --n_rollouts 100 \
#     --agent $checkpoint_dir --output_folder $output_folder --video_skip 1  \
#     --guidance $embedder --scale 0.5 --camera_names image --target_list 1,-0.33,-0.33,-0.33 --render_visuals --setup early_decision
