# DynaGuide Code

## What is here? 
This repository contains the `DynaGuide` implementation code. This includes code to train the dynamics model and use the model during inference-time to steer a diffusion policy. We include example uses of `DynaGuide` as seen in our paper experiments: a toy block environment and the Calvin environment. Finally, we include code snippets that can be added to any diffusion policy codebase to enable `DynaGuide`. 

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
- `train_dynaguide.py`: this is the 

# Installation and Starting
TODO


# Workflow for Calvin Experiments
This codebase contains the DynaGuide method, and we test it on the Calvin benchmark environment 

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
python train.py --config configs/diffusion_policy_image_calvin_ddim.json --dataset path_to_calvin_dataset --output output_folder  --name run_name
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

## Running DynaGuide


## Conducting Experiments from Paper
For your convenience, the scripts used to run the paper experiments are included in `paper_experiments/`. Modifications will be needed for different system capacities and paths. 

- Experiment 1: `calvin_articulated_object.sh` and generate plots with `experiment_1_graphs.py`
- Experiment 2: `calvin_movable_object.sh` and generate plots with `experiment_2_blocks.py`
- Experiment 3: `calvin_underspecified_objectives.sh` and generate plots with `experiment_3_partialgoals_graphs.py`
- Experiment 4: `calvin_multiple_behaviors.sh` and generate plots with `experiment_4_multiobjective.py`
- Experiment 5: `calvin_underrepresented_behaviors.sh` and generate plots with `experiment_5_deprivation_graphs.py` 


## Calvin DynaGuide: Immediate Example

1. Download the DynaGuide model from this link: 
1. Download the Base policy from this link: 
1. Download the Guidance conditions for SWITCH_ON from this link: 
1. Modify the `calvin_exp_configs_examples/switch_on.json` by changing the `pos_examples` file path to the downloaded guidance condition 
1. Make a folder called `results` in this directory 
Run the following code: 

```
run_name=SwitchOnDynaGuide
output_folder=results/$run_name
checkpoint_dir=path_to_base_policy
exp_setup_config=calvin_exp_configs_examples/switch_off.json
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

# Workflow for applying DynaGuide to Any Diffusion Policy 

done: 
dynamics models
embedder datasets
image_models 
train dynaguide 
test dynaguide embedding
run dynaguide 

in progress
toy experiment 
baselines 

todo: 
diffusion policy repo 
calvin repo 