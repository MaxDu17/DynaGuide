# DynaGuide: Steering Diffusion Polices with Active Dynamic Guidance
[Maximilian Du](https://maximiliandu.com/), [Shuran Song](https://shurans.github.io/)

arXiv | [Project Website](https://dynaguide.github.io/)

This repository contains the `DynaGuide` implementation code. This includes code to train the dynamics model and use the model during inference-time to steer a diffusion policy. We include example uses of `DynaGuide` as seen in our paper experiments: a toy block environment and the Calvin environment. Finally, we include code snippets that can be added to any diffusion policy codebase to enable `DynaGuide`. 

## Code Release Status
- 6/16/25: Uploaded all code and example of pretrained dynamics model, base policy, and guidance condition. 

# Installation and Starting
1. Install pytorch 
1. Go to `robomimic/` and run `pip install -e .`
1. Go to `calvin/` and run `install.sh`
1. Go to this directory and run `pip install -e .`
1. Download the calvin datasets and process them according to this [section](#workflow-for-calvin-experiments). Or, for the toy experiments, collect the data according to this [section](#workflow-for-toy-environment-experiment). For your own dataset / diffusion policy, refer to this [section](#workflow-for-applying-dynaguide-to-any-diffusion-policy) for general deployment of DynaGuide. 


## Overview: The Important Code
The core of DynaGuide is found in the `core/` folder in this directory. Experiment configs are found in `configs/` (base policy training) and `calvin_exp_configs_examples/` (DynaGuide evals). Auxiliary data processing scripts are found in `data_processing_calvin/`. See "Generating the Dataset Splits" for instructions on using them. Scripts for getting the paper results are found in `paper_experiments/`, and the toy experiment can be found in `toy_squares_experiment`. Figure generation can be found in `figure_generation/` that takes raw experiment output and creates the figures in the paper. 

The modified Calvin environment can be installed from the `Calvin/` folder, and the diffusion policy can be installed from the `robomimic/` folder. For more details, refer to the next section. 

The experiment code can be found in the main directory here
- `analyze_calvin_touch.py`: the code that looks at the DynaGuide results and gives statistics on individual behaviors; this code is repeated with modifications in the figure generation folder. 
- `run_dynaguide.py`: the code that evaluates dynaguide and records the data for analysis. It also runs the base policy directly and goal conditioned policies. 
- `run_sampling_baseline.py`: the code that runs the baseline where the base policy is sampled multiple times and the best action is kept 
- `run_trained_agent.py`: not used often, but this is the default code for evaluating trained base policies
- `test_dynaguide_embedding.py`: the code that takes a trained dynamics model and runs a suite of tests to ensure that is functional before applying DynaGuide 
- `train_base_policy.py`: this is the code that trains the base diffusion policy 
- `train_dynaguide.py`: this is the code that trains the dynamics model 

Additionally, the diffusion policy in `robomimic/robomimic/algo/diffusion_policy.py` is modified to support guidance. You can modify any DDIM noise prediction diffusion policy to support guidance (see this [section](#workflow-for-applying-dynaguide-to-any-diffusion-policy))

# Workflow for Calvin Experiments
This codebase contains the DynaGuide method, and we test it on the Calvin benchmark environment. 

## Generating the Dataset Splits
You will find the scripts for processing CALVIN data in the `data_processing_calvin` folder. You will run these scripts as follows: 

1. Download Calvin dataset (train and validation) using their provided instructions. Use the ABCD split for dynamics model training, and the D split for the base policy training. 
1. Run `calvin_to_labeled_hdf5.py` to segment the behaviors into a .h5 file. Do it once for the Calvin train split, and this will be your training .h5 file you use in `train_dynaguide.py`. Do it again for the Calvin validation split, and this will be used for the guidance conditions and the tests for the dynamics model. 
1. For the validation set: run `split_behavioral_validation_datasets_calvin.py` to separate these segmented behaviors to individual h5 files and a mixed test file for testing the dynamics model. 
1. To create the guidance conditions, use these individual h5 files in `hdf5_combiner.py`, which can create single h5 files with multiple behaviors with a set number of conditions per behavior 

Additional notes 
- To remove individual trajectories from guidance conditions files (e.g. there's a bad segmentation), use `hdf5_filter.py`. 
- To extract initial robot states for use in experiment resets (found in the config file), use `retrieve_initial_robot_states.py`
- To extract goal images for the goal conditioning baseline, use `generate_goal_images.py` 

## Training the Model
At this point, you should have 
- An h5 dataset for base policy training, downloaded & parsed from the Calvin D train split
- An h5 dataset for dynamics model training, downloaded & parsed from the Calvin ABCD train split.
- h5 datasets for dynamics model testing, including a multi-behavior test h5 file and individual behavior h5 files, created from the `split_behavioral_validation_datsets_calvin.py`

**Base Policy** 
The base policy is a diffusion policy implemented in Robomimic. We can use the provided robomimic script `train.py` to train the base policy. We provide the configs in the `configs/` folder. 

```
python split_train_val.py --dataset path_to_calvin_dataset --ratio 0.03
python train.py --config configs/diffusion_policy_image_calvin_ddim.json \
    --dataset path_to_calvin_dataset --output output_folder  --name run_name
```

**Dynamics Model**
The dynamics model is trained using this DynaGuide codebase. For the calvin environment, you can train the model with this configuration: 

```
python train_dynaguide.py --exp_dir directory_to_save_dynamics_model \
    --train_hdf5 your_calvin_dynamics_dataset  \
    --test_hdf5 your_calvin_validation_dataset \
    --cameras third_person --action_dim 7 --proprio_key proprio --proprio_dim 15 \
    --num_epochs 6000 --action_chunk_length 16 --batch_size 16 
```

To test the model, `test_dynaguide.py` contains a set of visualizations to analyze the ability for the dynamics model to predict the future and take actions into account. To run these tests, use this configuation: 

```
python test_dynaguide_embedding.py --exp_dir directory_of_dynamics_model \
     --good_hdf5 single_target_behavior_h5_file \
     --mixed_hdf5 mixed_behavior_test_h5_file  \
     --checkpoint path_to_dynamics_model_checkpoint  \
     --action_chunk_length 16 --key button_off 
```



## Generating the Experiment Config Files
The DynaGuide experiments in the Calvin environment require an experiment config that dictates the environment setup, guidance conditions, and others. Examples of these are included in `calvin_exp_configs_examples`, including the ones used in the paper experiments. 

```
{
    "env_setup" : {"green_light" : 1}, -> This is how to set up the environment for a particular test. See below for more details 
    "use_neg" : false, -> make true if you're including negative guidance conditions 
    "pos_examples": "/yourfolderhere/dataset/CalvinDD_validation_by_category_wcubes/button_off_20.hdf5", -> path to positive guidance conditions. 
    "loc_target": [ -> For the position guidance baseline only 
      -0.11221751715334842,
      -0.11465078614523619,
      0.4966200809334477
    ],
    "reset_poses": "initial_calvin_robot_states_midpoint.json" -> file containing reset poses for the robot (provided)
}
```

The `env_setup` tells the environment how to configure the articulated elements. This is important for testing the ability for DynaGuide to guide towards one particular behavior. 
- `green_light` {0, 1} sets the green light to either on or off 
- `sliding_door` [0, 0.28] sets the sliding door to a position. The base posiion (0) is the door fully right. At 0.28, the door is left. 
- `drawer` [0, 0.16] sets the drawer to a position. T he base position (0) is the drawer closed. At 0.16, the drawer is open. 
- `switch` sets the lever switch to a position (and the light is set automatically). The base position (0) is the switch in the off position. At 0.085, the switch is in the on position. 

## Running DynaGuide
To run DynaGuide, you need 1) the experiment config file 2) trained base policy 3) trained dynamics model and 4) guidance conditions (h5 files). Once you have these components, you can run `run_dynaguide.py` using the following code:

```
python run_dynaguide.py  --video_path $output_folder/$run_name.mp4 \
    --dataset_path output_hdf5_path --dataset_obs --json_path output_json_path --horizon 400 --n_rollouts 100 \
    --agent checkpoint_dir --output_folder output_folder --video_skip 2  \
    --exp_setup_config exp_config_path --guidance dynamics_model_path \
    --camera_names third_person --scale 1.5 --ss 4 --alpha 30 --save_frames
```
For examples of this evaluation call, look at the shell files in `paper_experiments`. Note that the scale, ss, and alpha can be modified here. 


## Conducting Experiments from Paper
For your convenience, the scripts used to run the paper experiments are included in `paper_experiments/`. Modifications will be needed for different system capacities and paths. 

- Experiment 1: `calvin_articulated_object.sh` and generate plots with `experiment_1_graphs.py`
- Experiment 2: `calvin_movable_object.sh` and generate plots with `experiment_2_blocks.py`
- Experiment 3: `calvin_underspecified_objectives.sh` and generate plots with `experiment_3_partialgoals_graphs.py`
- Experiment 4: `calvin_multiple_behaviors.sh` and generate plots with `experiment_4_multiobjective.py`
- Experiment 5: `calvin_underrepresented_behaviors.sh` and generate plots with `experiment_5_deprivation_graphs.py` 


## Calvin DynaGuide: Immediate Example

1. Download the DynaGuide model from this [link](https://drive.google.com/file/d/1lcI_PBgFIYDsoK4T4qO7SGJJgI2lw0Kd/view?usp=drive_link)
1. Download the Base policy from this [link](https://drive.google.com/file/d/1DeOnoDacXjBHgy1DGoRJpYIR8SyDy5fJ/view?usp=drive_link)
1. Download the Guidance conditions for SWITCH_ON from this [link](https://drive.google.com/file/d/1wtEGnG87Y-imqD2MygcqbJwp_RGi7YNB/view?usp=drive_link)
1. Modify the `calvin_exp_configs_examples/switch_on.json` by changing the `pos_examples` file path to the downloaded guidance condition 
1. Make a folder called `results` in this directory 
Run the following code: 

```
run_name=SwitchOnDynaGuide
output_folder=results/$run_name
checkpoint_dir=path_to_base_policy
exp_setup_config=calvin_exp_configs_examples/switch_on.json
embedder=path_to_embedder
python run_dynaguide.py  --video_path $output_folder/$run_name.mp4 \
    --dataset_path $output_folder/$run_name.hdf5 --dataset_obs --json_path $output_folder/$run_name.json --horizon 400 --n_rollouts 100 \
    --agent $checkpoint_dir --output_folder $output_folder --video_skip 2  \
    --exp_setup_config $exp_setup_config --guidance $embedder --camera_names third_person --scale 1.5 --ss 4 --alpha 30 --save_frames
```


To compare with base policy, run the control using this code 
```
run_name=BasePolicy
output_folder=results/$run_name
checkpoint_dir=path_to_base_policy
exp_setup_config=calvin_exp_configs_examples/switch_off.json
embedder=path_to_embedder
python run_dynaguide.py  --video_path $output_folder/$run_name.mp4 \
    --dataset_path $output_folder/$run_name.hdf5 --dataset_obs --json_path $output_folder/$run_name.json --horizon 400 --n_rollouts 100 \
    --agent $checkpoint_dir --output_folder $output_folder --video_skip 2  \
    --exp_setup_config $exp_setup_config --guidance $embedder --camera_names third_person --scale 0 --ss 1  --save_frames
```

Finally, fill the experiment name and directory in `analyze_calvin_touch.py` and run the code to see the behavior distribution before and after DynaGuide 


See above instsructions for how to run the experiments seen in the DynaGuide paper. 

# Workflow for Toy Environment Experiment
The toy square touching experiment is found in `toy_squares_experiment/` and the environment is found in `robomimic/robomimic/envs/env_flat_cube.py`. Unlike the main DynaGuide method, this dynamics model is actually a classifier and it is trained end-to-end with synthetic data. To collect this data, run this code:

```
run_name=pymunk_touch_res128_largercubes_repeated_100ktrain
output_folder=output_folder_here
python collect_scripted_data_pymunk.py --video_path $output_folder/$run_name.mp4 \
    --dataset_path $output_folder/data.hdf5 --dataset_obs --json_path $output_folder/config.json --horizon 150 --n_rollouts 100000 \
    --env_config /store/real/maxjdu/repos/robotrainer/configs/touchcubes.json --output_folder $output_folder --keep_only_successful \
    --camera_names image --video_skip 5 --repeat_environment
```

Then you can train the dynamics model using this code: 

```
experiment_name=Pymunk_classifier_FROMSCRATCH_100k_noised_ddim
CUDA_VISIBLE_DEVICES=6 python train_end_state_classifier.py --exp_dir /store/real/maxjdu/repos/robotrainer/results/classifiers/$experiment_name/ \
    --train_hdf5 /store/real/maxjdu/repos/robotrainer/dataset/pymunktouch/pymunk_touch_res128_largercubes_repeated_100ktrain/data.hdf5 \
    --test_hdf5 /store/real/maxjdu/repos/robotrainer/dataset/pymunktouch/pymunk_touch_res128_largercubes_repeated_valid/data.hdf5 \
    --num_epochs 12000 --action_chunk_length 16 --batch_size 16 --noised
```

and run tests with this code: 

```
experiment_name=Pymunk_classifier_FROMSCRATCH_100k_noised_ddim
checkpoint=11900
CUDA_VISIBLE_DEVICES=6 python test_end_state_classifier.py --exp_dir /store/real/maxjdu/repos/robotrainer/results/classifiers/$experiment_name/ \
    --mixed_hdf5 /store/real/maxjdu/repos/robotrainer/dataset/pymunktouch/pymunk_touch_res128_largercubes_repeated_valid/data.hdf5  \
    --checkpoint /store/real/maxjdu/repos/robotrainer/results/classifiers/$experiment_name/$checkpoint.pth  \
    --action_chunk_length 16
```

Finally, you can run an experiment by running code like this: 

```
CUDA_VISIBLE_DEVICES=5 python classifier_guidance_pymunk.py --video_path output_video_path \
    --dataset_path output_data_path --dataset_obs --json_path output_json --horizon 400 --n_rollouts 100 \
    --agent base_policy_path --output_folder output_folder_path --video_skip 1  \
    --guidance classifier_ckpt_path --scale 0.5 --camera_names image --target_list 1,-0.33,-0.33,-0.33 --render_visuals
```

The `target_list` allows you to specify which colors to seek and avoid. Positive means seek, and negative means avoid. 

# Workflow for applying DynaGuide to Any Diffusion Policy 
The code found in `core/` should be general and applicable to any diffusion-based policy. To deploy it on any such policy, you can train the dynamics model using the existing DynaGuide scripts and compute the guidance function using the existing code in `core/dynaguide.py`. To deploy it to any diffusion policy: 

1. Locate the function that does the inference-time denoising process
1. Verify that the model is indeed predicting *noise* 
1. Add this code: 

```
scaled_grad = guidance_function(state, naction)
noise_pred = noise_pred - (1 - noise_scheduler.alphas_cumprod[k]).sqrt() * scaled_grad 
```
where `guidance_function` is the function computed by `core/dynaguide.py` and `noise_pred` is the predicted noise outputted by the noise estimation network. Use the noise scheduler of the diffusion policy as `noise_scheduler`. If you use `diffusers.schedulers.scheduling_ddim`, nothing needs to be changed. 

Add this after the `noise_pred` is updated by the network and *before* the noise scheduler operates. To enable stochastic sampling, simply wrap the denoising step in an inner for-loop. An example is shown here: 

```
 for ss_iter in range(ss): # stochastic sampling; notice how the "k" doesn't decrease in this loop. 
    noise_pred = nets['policy']['noise_pred_net'](
        sample=naction, 
        timestep=k,
        global_cond=obs_cond
    )
    

    state = inputs["obs"]
    scaled_grad = guidance_function(state, naction)
    noise_pred = noise_pred - (1 - self.noise_scheduler.alphas_cumprod[k]).sqrt() * scaled_grad 
    # inverse diffusion step (remove noise)
    naction = self.noise_scheduler.step(
        model_output=noise_pred,
        timestep=k,
        sample=naction
    ).prev_sample # this gives the mu from the paper 
```

# Cite our paper 
```
@misc{du2025dynaguidesteeringdiffusionpolices,
    title={DynaGuide: Steering Diffusion Polices with Active Dynamic Guidance}, 
    author={Maximilian Du and Shuran Song},
    year={2025},
    eprint={2506.13922},
    archivePrefix={arXiv},
    primaryClass={cs.RO},
    url={https://arxiv.org/abs/2506.13922}, 
}
```

# Code References 
We are grateful for the Calvin [benchmark codebase](https://github.com/mees/calvin) duplicated here in this repository with minor modifications. The diffusion policy is adapted from the [Robomimic codebase](https://robomimic.github.io/) also duplicated here with modifications. The dynamics model was inspired by the [Dino-WM implementation](https://dino-wm.github.io/) and leverages visuals representations from [Dino-V2](https://dinov2.metademolab.com/). 

# Feedback
Are you trying to get DynaGuide working and something's wrong? The numbers don't look right? Send me an email at `maxjdu@stanford.edu` !

