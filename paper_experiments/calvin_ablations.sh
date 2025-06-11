#!/bin/bash

# this script runs ablation experiments 

cleanup() {
    echo "Terminating all background processes..."
    pkill -P $$  # Kill all child processes of this script
    exit 1
}

# Trap Ctrl+C (SIGINT) and call cleanup function
trap cleanup SIGINT 

num_trials=50

dynamics_model="add your dynamics model here"
base_policies=(
    "add your policy paths here"
)

count=0
gpudevice=4
exp_root="add your experiment root here"


######## FOR THE CONTROLS
# count=0
# for ckpt_dir in "${base_policies[@]}"; do
#     echo "On checkpoint $ckpt_dir"
#     run_name=Control_$count
#     output_folder=$exp_root$run_name
#     CUDA_VISIBLE_DEVICES=1 python run_dynaguide.py  --video_path $output_folder/$run_name.mp4 \
#         --dataset_path $output_folder/$run_name.hdf5 --dataset_obs --json_path $output_folder/$run_name.json --horizon 400 --n_rollouts $num_trials \
#         --agent $ckpt_dir --output_folder $exp_root/$run_name --video_skip 2  \
#         --exp_setup_config $exp_setup_config --guidance $dynamics_model --camera_names third_person --scale 0 --ss 1 & 
#     wait 
#     ((count++))

# done 
# wait 
ss=(
    1
    2
    3
    4
    5
    6
)

scales_a=(
    0.5
    0.7
    0.9
    1
    1.2
)
scales_b=(
    1.2
    1.4
    1.6
    1.8
)
scales_c=(
    2
    2.2
    2.4
    2.6
)

# scales_a=(
#    2.8
#    3
#    3.4
#    3.8
#    4.2
# )
# scales_b=(
#     4.6 
#     5
#     5.4
#     5.8
#     6.2
# )

alphas_a=(
    1
    2
    5
    10
)
alphas_b=(
    20
    30
    40
    50
)

# max_examples
examples_a=(
    1
    2
    4
    6
    8
)
examples_b=(
    10
    12
    14
    16
    18
    20
)
#### FOR SAMPLE LIMITS
for ckpt_dir in "${base_policies[@]}"; do
    echo "On checkpoint $ckpt_dir"
    for examples in "${examples_a[@]}"; do
        run_name=examples_${examples}_$count
        exp_setup_config=yourconfigdirectory/switch_on.json
        output_folder=$exp_root$run_name
        CUDA_VISIBLE_DEVICES=0 python run_dynaguide.py  --video_path $output_folder/$run_name.mp4 \
            --dataset_path $exp_root$run_name/$run_name.hdf5 --dataset_obs --json_path $output_folder/$run_name.json --horizon 400 --n_rollouts $num_trials \
            --agent $ckpt_dir --output_folder $exp_root/$run_name --video_skip 2  \
            --exp_setup_config $exp_setup_config --guidance $dynamics_model --camera_names third_person --scale 1.5 --ss 4 --alpha 30 --max_examples $examples &
        echo $run_name 
    done 

    for examples in "${examples_b[@]}"; do
        run_name=examples_${examples}_$count
        exp_setup_config=yourconfigdirectory/switch_on.json
        output_folder=$exp_root$run_name
        CUDA_VISIBLE_DEVICES=1 python run_dynaguide.py  --video_path $output_folder/$run_name.mp4 \
            --dataset_path $exp_root$run_name/$run_name.hdf5 --dataset_obs --json_path $output_folder/$run_name.json --horizon 400 --n_rollouts $num_trials \
            --agent $ckpt_dir --output_folder $exp_root/$run_name --video_skip 2  \
            --exp_setup_config $exp_setup_config --guidance $dynamics_model --camera_names third_person --scale 1.5 --ss 4 --alpha 30 --max_examples $examples &
        echo $run_name 
    done 

    wait # wait this batch to stop 
    ((count++))
done
wait



############ FOR ALPHA
for ckpt_dir in "${base_policies[@]}"; do
    echo "On checkpoint $ckpt_dir"
    for alpha in "${alphas_a[@]}"; do
        run_name=TESTalpha_${alpha}_$count
        exp_setup_config=yourconfigdirectory/switch_on.json
        output_folder=$exp_root$run_name
        CUDA_VISIBLE_DEVICES=0 python run_dynaguide.py  --video_path $output_folder/$run_name.mp4 \
            --dataset_path $exp_root$run_name/$run_name.hdf5 --dataset_obs --json_path $output_folder/$run_name.json --horizon 400 --n_rollouts $num_trials \
            --agent $ckpt_dir --output_folder $exp_root/$run_name --video_skip 2  \
            --exp_setup_config $exp_setup_config --guidance $dynamics_model --camera_names third_person --scale 1.5 --ss 4 --alpha $alpha &
        echo $run_name 
    done 

    for alpha in "${alphas_b[@]}"; do
        run_name=alpha_${alpha}_$count
        exp_setup_config=yourconfigdirectory/switch_on.json
        output_folder=$exp_root$run_name
        CUDA_VISIBLE_DEVICES=1 python run_dynaguide.py  --video_path $output_folder/$run_name.mp4 \
            --dataset_path $exp_root$run_name/$run_name.hdf5 --dataset_obs --json_path $output_folder/$run_name.json --horizon 400 --n_rollouts $num_trials \
            --agent $ckpt_dir --output_folder $exp_root/$run_name --video_skip 2  \
            --exp_setup_config $exp_setup_config --guidance $dynamics_model --camera_names third_person --scale 1.5 --ss 4 --alpha $alpha &
        echo $run_name 
    done 

    wait # wait this batch to stop 
    ((count++))
done
wait

# ###  FOR SCALE 
for ckpt_dir in "${base_policies[@]}"; do
    echo "On checkpoint $ckpt_dir"
    for scale in "${scales_a[@]}"; do
        run_name=scale_${scale}_$count
        exp_setup_config=yourconfigdirectory/switch_on.json
        output_folder=$exp_root$run_name
        CUDA_VISIBLE_DEVICES=0 python run_dynaguide.py  --video_path $output_folder/$run_name.mp4 \
            --dataset_path $exp_root$run_name/$run_name.hdf5 --dataset_obs --json_path $output_folder/$run_name.json --horizon 400 --n_rollouts $num_trials \
            --agent $ckpt_dir --output_folder $exp_root/$run_name --video_skip 2  \
            --exp_setup_config $exp_setup_config --guidance $dynamics_model --camera_names third_person --scale $scale --ss 4 --alpha 30 &
        echo $run_name 
    done 

    for scale in "${scales_b[@]}"; do
        run_name=scale_${scale}_$count
        exp_setup_config=yourconfigdirectory/switch_on.json
        output_folder=$exp_root$run_name
        CUDA_VISIBLE_DEVICES=1 python run_dynaguide.py  --video_path $output_folder/$run_name.mp4 \
            --dataset_path $exp_root$run_name/$run_name.hdf5 --dataset_obs --json_path $output_folder/$run_name.json --horizon 400 --n_rollouts $num_trials \
            --agent $ckpt_dir --output_folder $exp_root/$run_name --video_skip 2  \
            --exp_setup_config $exp_setup_config --guidance $dynamics_model --camera_names third_person --scale $scale --ss 4 --alpha 30 &
        echo $run_name 
    done 

    for scale in "${scales_c[@]}"; do
        run_name=scale_${scale}_$count
        exp_setup_config=yourconfigdirectory/switch_on.json
        output_folder=$exp_root$run_name
        CUDA_VISIBLE_DEVICES=5 python run_dynaguide.py  --video_path $output_folder/$run_name.mp4 \
            --dataset_path $exp_root$run_name/$run_name.hdf5 --dataset_obs --json_path $output_folder/$run_name.json --horizon 400 --n_rollouts $num_trials \
            --agent $ckpt_dir --output_folder $exp_root/$run_name --video_skip 2  \
            --exp_setup_config $exp_setup_config --guidance $dynamics_model --camera_names third_person --scale $scale --ss 4 --alpha 30 &
        echo $run_name 
    done 

    wait # wait this batch to stop 
    ((count++))
done
wait

# STOCHASTIC SAMPLING
for ckpt_dir in "${base_policies[@]}"; do
    echo "On checkpoint $ckpt_dir"
    for ss_val in "${ss[@]}"; do
        run_name=ss_s3_${ss_val}_$count
        exp_setup_config=yourconfigdirectory/switch_on.json
        output_folder=$exp_root$run_name
        CUDA_VISIBLE_DEVICES=2 python run_dynaguide.py  --video_path $output_folder/$run_name.mp4 \
            --dataset_path $exp_root$run_name/$run_name.hdf5 --dataset_obs --json_path $output_folder/$run_name.json --horizon 400 --n_rollouts $num_trials \
            --agent $ckpt_dir --output_folder $exp_root/$run_name --video_skip 2  \
            --exp_setup_config $exp_setup_config --guidance $dynamics_model --camera_names third_person --scale 3 --ss $ss_val --alpha 30 &
        echo $run_name 
    done 
    wait # wait this batch to stop 
    ((count++))
done 

NOISED
for ckpt_dir in "${base_policies[@]}"; do
    echo "On checkpoint $ckpt_dir"
    run_name=no_noise_pretrain_$count
    exp_setup_config=yourconfigdirectory/switch_on.json
    output_folder=$exp_root$run_name
    CUDA_VISIBLE_DEVICES=7 python run_dynaguide.py  --video_path $output_folder/$run_name.mp4 \
        --dataset_path $exp_root$run_name/$run_name.hdf5 --dataset_obs --json_path $output_folder/$run_name.json --horizon 400 --n_rollouts $num_trials \
        --agent $ckpt_dir --output_folder $exp_root/$run_name --video_skip 2  \
        --exp_setup_config $exp_setup_config --guidance /store/real/maxjdu/repos/robotrainer/results/classifiers/CALVINABCD_blockseg_newmodel_8head_6_depth_NOISED_wproprio_NoNoise/5900.pth \
        --camera_names third_person --scale 1.5 --ss 4 --alpha 30 &
    echo $run_name 
    ((count++))
done 
wait 

for ckpt_dir in "${base_policies[@]}"; do
    echo "On checkpoint $ckpt_dir"
    run_name=noise_pretrain_$count
    exp_setup_config=yourconfigdirectory/switch_on.json
    output_folder=$exp_root$run_name
    CUDA_VISIBLE_DEVICES=7 python run_dynaguide.py  --video_path $output_folder/$run_name.mp4 \
        --dataset_path $exp_root$run_name/$run_name.hdf5 --dataset_obs --json_path $output_folder/$run_name.json --horizon 400 --n_rollouts $num_trials \
        --agent $ckpt_dir --output_folder $exp_root/$run_name --video_skip 2  \
        --exp_setup_config $exp_setup_config --guidance $dynamics_model --camera_names third_person --scale 1.5 --ss 4 --alpha 30 &
    echo $run_name 
    ((count++))
done 
wait 

echo "All configurations processed successfully."