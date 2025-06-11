import argparse
import os
import json
import h5py
import imageio
import sys
import time
import traceback
import numpy as np
from copy import deepcopy
from tqdm import tqdm
import math

import shutil 
import torch
import pickle 

import robomimic
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.obs_utils as ObsUtils
from robomimic.utils.log_utils import log_warning
from robomimic.envs.env_base import EnvBase
from robomimic.envs.wrappers import EnvWrapper
from robomimic.algo import RolloutPolicy
from robomimic.scripts.playback_dataset import DEFAULT_CAMERAS
import io 

from core.dynamics_models import FinalStateClassification
from core.image_models import VAE

from core.embedder_datasets import MultiviewDataset
import cv2 
import matplotlib.pyplot as plt 

# the suite of scenarios 
import random 
def jiggle(deterministic = False):
    if deterministic:
        return 0
    return 2 * (np.random.random() - 0.5)
    

def early_decision_cube_setup(deterministic = False):
    # used to be 70 jiggle and 20 jiggle for agent 
    # -> 90 jiggle and this time, also 70 jiggle for agent! 
    cube_1 = [128 + 90 * jiggle(), 128 + 90 * jiggle()]
    cube_2 = [128 + 90 * jiggle(), 384 + 90 * jiggle()]
    cube_4 = [384 + 90 * jiggle(), 128 + 90 * jiggle()]
    cube_3 = [384 + 90 * jiggle(), 384 + 90 * jiggle()]
    # agent = [255 + 100 * jiggle(), 255 + 100 * jiggle()]
    agent = [255 + 70 * jiggle(), 255 + 70 * jiggle()]
    cube_list = [cube_1, cube_2, cube_3, cube_4]
    if not deterministic:
        random.shuffle(cube_list)

    loc_list = [agent,]
    loc_list.extend(cube_list)
    # to keep the cubes in place 
    # if not deterministic:
    rand_rots = np.random.rand(4) * 2 * np.pi - np.pi 

    # rand_rots = np.random.rand(4) * 2 * np.pi - np.pi 
    loc_list.append(rand_rots)
    state = np.concatenate(loc_list)
    return state 


def late_decision_cube_setup(close = False, deterministic = False):
    # used to be 10 jiggle and agent 30, 15 jiggle 
    # now is 25 jiggle and agent 70 jiggle 
    # if deterministic:
    #     displacement = 0
    # else:
        # displacement = 100 * np.random.random()
    displacement = 0
    # cube_1 = [60, 340 + 10 * jiggle()]
    # cube_2 = [60, 420 + 10 * jiggle()]
    # cube_3 = [100, 480 + 10 * jiggle()]
    # cube_4 = [180, 480 + 10 * jiggle()]

    cube_1 = [100, 340 + 25 * jiggle()]
    cube_2 = [100, 420 + 25 * jiggle()]
    cube_3 = [150 + 25 * jiggle(), 460]
    cube_4 = [230 + 25 * jiggle(), 460]
    # if close:
    #     agent = [220 + 10 * jiggle(deterministic), 210 + displacement]
    # else:
    #     agent = [480 + 10 * jiggle(deterministic), 210 + displacement]

    # agent = [460 + 30 * jiggle(), 60 + 15 * jiggle()]
    agent = [420 + 70 * jiggle(), 120 + 70 * jiggle()]

    cube_list = [cube_1, cube_2, cube_3, cube_4]
    if not deterministic: 
        random.shuffle(cube_list)
    loc_list = [agent,]
    loc_list.extend(cube_list)

    # if not deterministic:
    rand_rots = np.random.rand(4) * 2 * np.pi - np.pi 
    # else:
    #     rand_rots = np.zeros(4)

    loc_list.append(rand_rots)
    state = np.concatenate(loc_list)
    return state 

def cube_blockade_setup(deterministic = False):
    jiggle_scale = 30
    # cube_4 = [240 + jiggle_scale * jiggle(), 100 + jiggle_scale * jiggle()]
    # cube_2 = [240 + jiggle_scale * jiggle(), 250 + jiggle_scale * jiggle()]
    # cube_3 = [240 + jiggle_scale * jiggle(), 400 + jiggle_scale * jiggle()]
    # cube_1 = [60 + 2 * jiggle_scale * jiggle(), 250 + jiggle_scale * jiggle()]
    # agent = [420 + 2 * jiggle_scale * jiggle(), 250 + jiggle_scale * jiggle()]

    cube_4 = [320 + 2 * jiggle_scale * jiggle(), 60 + 2 * jiggle_scale * jiggle()]
    cube_2 = [320 + 2 * jiggle_scale * jiggle(), 250 + 2 * jiggle_scale * jiggle()] # barrier cube 
    cube_3 = [320 + 2 * jiggle_scale * jiggle(), 450 + 2 * jiggle_scale * jiggle()]
    cube_1 = [60 + 2 * jiggle_scale * jiggle(), 250 + 2 * jiggle_scale * jiggle()] # target cube 
    agent = [450 + 2 * jiggle_scale * jiggle(), 250 + 2 * jiggle_scale * jiggle()]
    # used to have jiggle_scale = 10 

    cube_list = [cube_1, cube_2, cube_3, cube_4]
    # if not deterministic:
    #     random.shuffle(cube_list)
    loc_list = [agent] 
    loc_list.extend(cube_list)

    # if not deterministic:
    #     rand_rots = np.random.rand(4) * 2 * np.pi - np.pi 
    # else:
        # rand_rots = np.zeros(4)
    rand_rots = np.random.rand(4) * 2 * np.pi - np.pi 
    loc_list.append(rand_rots)
    state = np.concatenate(loc_list)
    return state 


def around_the_column_setup(deterministic = False):
    if deterministic:
        displacement_x, displacement_y = 0, 0 
    else:
        displacement_x = 25 * np.random.random()
        displacement_y = 25 * np.random.random()
    cube_1 = [240 + displacement_x, 240 + displacement_y]
    cube_2 = [300 + displacement_x, 240 + displacement_y]
    cube_3 = [240 + displacement_x, 300 + displacement_y]
    cube_4 = [300 + displacement_x, 300 + displacement_y]
    agent_list = [
        [450 + 10 * jiggle(deterministic), 450 + 10 * jiggle(deterministic)],
        [60 + 10 * jiggle(deterministic), 450 + 10 * jiggle(deterministic)], 
        [450 + 10 * jiggle(deterministic), 60 + 10 * jiggle(deterministic)],
        [60 + 10 * jiggle(deterministic), 60 + 10 * jiggle(deterministic)],
    ]
    ordering = [0, 1, 2, 3]
    if not deterministic:
        random.shuffle(ordering)
    cube_list = [cube_1, cube_2, cube_3, cube_4]
    shuffled_cube_list = [cube_list[i] for i in ordering]
    random.shuffle(cube_list)
    loc_list = [agent_list[ordering[0]],] # picks the appropiate agent for the cube setup 
    loc_list.extend(shuffled_cube_list)

    if not deterministic:
        rand_rots = np.random.rand(4) * 2 * np.pi - np.pi 
    else:
        rand_rots = np.zeros(4)
    # rand_rots = np.random.rand(4) * 2 * np.pi - np.pi 
    loc_list.append(rand_rots)
    state = np.concatenate(loc_list)
    return state 



def prepare_np(data, device = "cuda"):
    if type(data) == dict:
        return {k : torch.tensor(v).to(device).to(torch.float32) for k, v in data.items()}
    return torch.tensor(data).to(device).to(torch.float32)

def calculate_classifier_guidance(model, scale, target):
    # target = target 
    ce_loss = torch.nn.CrossEntropyLoss()
    def guidance(states, actions):
        relevant_state = {"image" : states["image"][:, -1] * 255} # needs to be 0 -> 255 not 0 -> 1 
        # actions.requires_grad = True
        predicted_end = model(relevant_state, actions) # S X D
        softmaxed = torch.softmax(predicted_end, dim = 1) 

        guidance_value = 0
        for i, weight in enumerate(target):
            # print(i, weight)
            guidance_value += weight * torch.log(softmaxed[0, i])

        gradient = torch.autograd.grad(guidance_value, actions)[0].detach() # janky gradient lol 

        gradient = scale * gradient        
        return gradient    
    
    return guidance


def get_img_from_fig(fig, dpi=80):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    img = cv2.imdecode(img_arr, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img 

# what I want to see: denosing progression (essentially the same thing)
# difference: Superimpose the envionment onto it? Le3t's not start there 

def visualize_corrections(samples_list, corrections_list, diffusion_list, state, exp_dir, step, rollout_count, target_list):
    # TODO: variable targets 
    current_xy = state["agent_pos"][-1]
    to_skip = 2 #len(samples_list) // 20 if len(samples_list) > 20 else 1
    plt.rcParams["figure.figsize"] = (30, 5)
    # fig, axs = plt.subplots(2, 5)
    fig, axs = plt.subplots(nrows = 1, ncols = 6)
    fig.tight_layout()
    SCALE = 1

    video_writer = imageio.get_writer(exp_dir + f"/{rollout_count}_{step}_guidance.gif") #, fps=20)

    plot_count = 0 
    for i, action in enumerate(samples_list):
        if i < 5:
            continue 
        ax = axs[plot_count] 
        ax.set_xticks([])
        ax.set_yticks([])
        plot_count += 1 
        action = action[0, :].detach().cpu().numpy()
        color_list = ["black" if i < 8 else "gray" for i in range(action.shape[0])]
        cropped_image = state["image"][-1][:, 5:-5, 5:-5] # remove the grey boarder 
        ax.imshow(np.flipud(np.transpose(cropped_image, (1, 2, 0))), extent = [-1, 1, -1, 1]) #flip ud important for the matplotlib quirk 

        # ax.scatter(action[:, 0], action[:, 1], zorder = 2, color = color_list, s = 15)
        ax.scatter(action[:, 0], action[:, 1], zorder = 2, color = "black", s = 15)
        # ax.scatter((current_xy[0],), (current_xy[1],), color = "orange", s = 40, zorder = 10)
        corrections = corrections_list[i].detach().cpu().numpy()[0]
        diffusion = diffusion_list[i].detach().cpu().numpy()[0]
        MAX_LEN = 0.8
        if np.max(np.linalg.norm(diffusion, axis = 1)) > MAX_LEN:
            diffusion = MAX_LEN * diffusion / np.max(np.linalg.norm(diffusion, axis = 1))
        if np.max(np.linalg.norm(corrections, axis = 1)) > MAX_LEN: 
            corrections = MAX_LEN * corrections / np.max(np.linalg.norm(corrections, axis = 1))

        for j in range(action.shape[0]):
            ax.arrow(action[j, 0], action[j, 1], SCALE * corrections[j, 0], SCALE * corrections[j, 1], head_width = 0.05, width = 0.02, color = "#89C5EC", length_includes_head = True) # , head_width = 2, width = 0.5, color = "black")
            ax.arrow(action[j, 0], action[j, 1], SCALE * diffusion[j, 0], SCALE * diffusion[j, 1], head_width = 0.05, width = 0.02, color = "#E37B3A", length_includes_head = True) # , head_width = 2, width = 0.5, color = "black")
        # ax.plot((current_xy[0], current_xy[0] + vec[0]), (current_xy[1], current_xy[1] + vec[1]), "--", color = "green", linewidth = 3, zorder = 3)
        # ax.text(0, 1.35, f"Denoising Step {len(samples_list)- i - 1}", horizontalalignment = 'center', fontfamily = 'sans-serif', fontsize = 'x-large', fontweight = 'medium')
        ax.set_xlim(-1.5, 1.5)
        # ax.set_ylim(1.5, -1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_xticks([])
        ax.set_yticks([])


        # this saves 
        extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=80,  bbox_inches=extent)
        buf.seek(0)
        img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
        buf.close()
        img = cv2.imdecode(img_arr, 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        video_writer.append_data(img)

    plt.tight_layout()
    plt.savefig(exp_dir + f"/{rollout_count}_{step}_guidance.pdf", transparent = True)
    plt.close()
    video_writer.close()


    # for visual purposes, render the last frame for the video 
    plt.rcParams["figure.figsize"] = (6,6) # need to make things square 
    plt.tight_layout()
    plt.axis('off')
    plot_count += 1 
    action = samples_list[-1] 
    action = action[0, :].detach().cpu().numpy()
    color_list = ["blue" if i < 8 else "cyan" for i in range(action.shape[0])]
    plt.imshow(np.flipud(np.transpose(state["image"][-1], (1, 2, 0))), extent = [-1, 1, -1, 1]) #flip ud important for the matplotlib quirk 
    plt.scatter(action[:, 0], action[:, 1], zorder = 2, color = color_list, s = 15)

    plt.scatter((current_xy[0],), (current_xy[1],), color = "magenta", s = 40, zorder = 10)
    corrections = corrections_list[-1].detach().cpu().numpy()[0]
    for j in range(action.shape[0]):
        plt.arrow(action[j, 0], action[j, 1], SCALE * corrections[j, 0], SCALE * corrections[j, 1], head_width = 0.05, width = 0.01, color = "magenta") # , head_width = 2, width = 0.5, color = "black")
    # ax.plot((current_xy[0], current_xy[0] + vec[0]), (current_xy[1], current_xy[1] + vec[1]), "--", color = "green", linewidth = 3, zorder = 3)
    
    plt.xlim(-1.5, 1.5)
    plt.ylim(1.5, -1.5)
    arr = get_img_from_fig(plt)
    plt.close()
    return arr




def rollout(policy, env, horizon, target_list, render=False, video_writer=None, video_skip=5, return_obs=False, camera_names=None, real=False, rate_measure=None,
            classifier_grad = None, reset_state = None, model = None, exp_dir = None, rollout_count = None, render_visuals = False, setup = None, ss = 4):
    """
    Helper function to carry out rollouts. Supports on-screen rendering, off-screen rendering to a video, 
    and returns the rollout trajectory.

    Args:
        policy (instance of RolloutPolicy): policy loaded from a checkpoint
        env (instance of EnvBase): env loaded from a checkpoint or demonstration metadata
        horizon (int): maximum horizon for the rollout
        render (bool): whether to render rollout on-screen
        video_writer (imageio writer): if provided, use to write rollout to video
        video_skip (int): how often to write video frames
        return_obs (bool): if True, return possibly high-dimensional observations along the trajectoryu. 
            They are excluded by default because the low-dimensional simulation states should be a minimal 
            representation of the environment. 
        camera_names (list): determines which camera(s) are used for rendering. Pass more than
            one to output a video with multiple camera views concatenated horizontally.
        real (bool): if real robot rollout
        rate_measure: if provided, measure rate of action computation and do not play actions in environment

    Returns:
        stats (dict): some statistics for the rollout - such as return, horizon, and task success
        traj (dict): dictionary that corresponds to the rollout trajectory
    """
    rollout_timestamp = time.time()
    assert isinstance(env, EnvBase) or isinstance(env, EnvWrapper)
    assert isinstance(policy, RolloutPolicy)
    assert not (render and (video_writer is not None))

    policy.start_episode()
    state = None 
    if setup == "early_decision":
        state = early_decision_cube_setup(deterministic = True)
    if setup == "around_column":
        state = around_the_column_setup()
    if setup == "cube_blockade":
        state = cube_blockade_setup()
    if setup == "late_decision":
        state = late_decision_cube_setup(deterministic = True)

    obs = env.reset()
    if state is not None:
        obs = env.reset_to(state)


    state_dict= {}

    results = {}
    video_count = 0  # video frame counter
    total_reward = 0.
    got_exception = False
    success = env.is_success()["task"]
    traj = dict(actions=[], rewards=[], dones=[], states=[], initial_state_dict=state_dict)
    collage = None
    if return_obs:
        # store observations too
        traj.update(dict(obs=[], next_obs=[]))
    try:
        for step_i in range(horizon):
            print(step_i)
            # # HACK: some keys on real robot do not have a shape (and then they get frame stacked)
            # for k in obs:
            #     if len(obs[k].shape) == 1:
            #         obs[k] = obs[k][..., None] 

            # get action from policy
            t1 = time.time()
            act = policy(ob=obs, guidance_function = classifier_grad, guidance_type = "diffusion", ss = ss)
            # print(act)
            t2 = time.time()
            # if real and (not env.base_env.controller_type == "JOINT_IMPEDANCE") and (policy.policy.global_config.algo_name != "diffusion_policy"):
            #     # joint impedance actions and diffusion policy actions are absolute in the real world
            #     act = np.clip(act, -1., 1.)

            if rate_measure is not None:
                rate_measure.measure()
                print("time: {}s".format(t2 - t1))
                # dummy reward and done
                r = 0.
                done = False
                next_obs = obs
            else:
                # play action
                # print(act)
                t1 = time.time()
                next_obs, r, done, _ = env.step(act)

            # compute reward
            total_reward += r
            success = env.is_success()["task"]

            if step_i % 14 == 0: 
                denoising_list = policy.policy.corrections_list # action before correction 
                guidance_list = policy.policy.guidance_list # scaled guidance 
                diffusion_list = policy.policy.diffusion_list 
                print("Guidance norms: ", [np.linalg.norm(guidance.detach().cpu().numpy()).item() for guidance in guidance_list])
                if render_visuals:
                    collage = visualize_corrections(denoising_list, guidance_list, diffusion_list, obs, exp_dir, step_i, rollout_count, target_list = target_list)
            # visualization
            if render:
                env.render(mode="human", camera_name=camera_names[0])
            
            if video_writer is not None:
                if video_count % video_skip == 0:
                    video_img = []
                    for cam_name in camera_names:
                        img = env.render(mode="rgb_array", height=128 * 5, width=128 * 5, camera_name=cam_name)
                        video_img.append(img)
                        video_img = np.concatenate(video_img, axis=0) # concatenate horizontally

                    if render_visuals:
                        video_img = cv2.resize(video_img, (128 * 5, 128 * 5))
                        collage_r = cv2.resize(collage, (128 * 5, 128 * 5))
                        # video_img = np.concatenate((before_collage, collage, video_img), axis = 0)
                        # corrections = cv2.resize(corrections, (128 * 5, 128 * 5))
                        # video_img = np.concatenate((collage, video_img, corrections), axis = 0)
                        video_img = np.concatenate((collage_r, video_img), axis = 0)
                        video_writer.append_data(video_img)
                    else:
                        video_writer.append_data(video_img)
                    # if step_i % 8 == 0:
                    #     plt.imsave(exp_dir + f"/{rollout_count}_{step_i}.png", video_img)
                video_count += 1
            # collect transition
            traj["actions"].append(act)
            traj["rewards"].append(r)
            traj["dones"].append(done)
            # if not real:
            #     traj["states"].append(state_dict["states"])
            if return_obs:
                # Note: We need to "unprocess" the observations to prepare to write them to dataset.
                #       This includes operations like channel swapping and float to uint8 conversion
                #       for saving disk space.
                traj["obs"].append(ObsUtils.unprocess_obs_dict(obs))
                traj["next_obs"].append(ObsUtils.unprocess_obs_dict(next_obs))

            # break if done or if success
            if done or success:
                break

            # update for next iter
            obs = deepcopy(next_obs)
            # if not real:
            #     state_dict = env.get_state()

    except env.rollout_exceptions as e:
        print("WARNING: got rollout exception {}".format(e))
        got_exception = True

    stats = dict(
        Return=total_reward,
        Horizon=(step_i + 1),
        Success_Rate=float(success),
        Exception_Rate=float(got_exception),
        time=(time.time() - rollout_timestamp),
    )

    if return_obs:
        # convert list of dict to dict of list for obs dictionaries (for convenient writes to hdf5 dataset)
        traj["obs"] = TensorUtils.list_of_flat_dict_to_dict_of_list(traj["obs"])
        traj["next_obs"] = TensorUtils.list_of_flat_dict_to_dict_of_list(traj["next_obs"])

    # list to numpy array
    for k in traj:
        if k == "initial_state_dict":
            continue
        if isinstance(traj[k], dict):
            for kp in traj[k]:
                traj[k][kp] = np.array(traj[k][kp])
        else:
            traj[k] = np.array(traj[k])

    return stats, traj


def run_trained_agent(args):
    # classifier_grad = None
    states_list, actions_list, labels_list = list(), list(), list()

    if args.output_folder is not None and not os.path.isdir(args.output_folder):
        os.mkdir(args.output_folder)

    # some arg checking
    write_video = (args.video_path is not None)
    assert not (args.render and write_video) # either on-screen or video but not both

    rate_measure = None
    if args.hz is not None:
        import RobotTeleop
        from RobotTeleop.utils import Rate, RateMeasure, Timers
        rate_measure = RateMeasure(name="control_rate_measure", freq_threshold=args.hz)
    
    # load ckpt dict and get algo name for sanity checks
    algo_name, ckpt_dict = FileUtils.algo_name_from_checkpoint(ckpt_path=args.agent)

    if args.dp_eval_steps is not None:
        assert algo_name == "diffusion_policy"
        log_warning("setting @num_inference_steps to {}".format(args.dp_eval_steps))

        # HACK: modify the config, then dump to json again and write to ckpt_dict
        tmp_config, _ = FileUtils.config_from_checkpoint(ckpt_dict=ckpt_dict)
        with tmp_config.values_unlocked():
            if tmp_config.algo.ddpm.enabled:
                tmp_config.algo.ddpm.num_inference_timesteps = args.dp_eval_steps
            elif tmp_config.algo.ddim.enabled:
                tmp_config.algo.ddim.num_inference_timesteps = args.dp_eval_steps
            else:
                raise Exception("should not reach here")
        ckpt_dict['config'] = tmp_config.dump()
    # device
    device = TorchUtils.get_torch_device(try_to_use_cuda=True)

    ## HACK: change DDPM to DDIM  
    tmp_config, _ = FileUtils.config_from_checkpoint(ckpt_dict=ckpt_dict)
    with tmp_config.values_unlocked():
        tmp_config.algo.ddpm.enabled = False 
        tmp_config.algo.ddim.enabled = True
    ckpt_dict['config'] = tmp_config.dump()
    # restore policy
    policy, ckpt_dict = FileUtils.policy_from_checkpoint(ckpt_dict=ckpt_dict, device=device, verbose=True)


    ############# classifier guidance ################
  # this needs to be aligned with the action chunk length in the trained model 
    ACTION_DIM = 2
    ACTION_CHUNK_LENGTH = 16 # this is how long the action predictions are
    cameras = ["image"] # you can change this; it's hardcoded

    state_vae = VAE(64)
    # so this should be able to reload the model without loading the state vae directly 
    model = FinalStateClassification(ACTION_DIM, ACTION_CHUNK_LENGTH, cameras=cameras, state_vae = state_vae, classes = 4)
    model.load_state_dict(torch.load(args.guidance))
    model.to("cuda")
    model.eval() 

    # target = 0 

    classifier_grad = calculate_classifier_guidance(model, args.scale, target = args.target_list) # adjustable!! 

    shutil.copy("classifier_guidance_pymunk.py", args.output_folder + "/classifier_guidance_pymunk.py") # because there are variants 
    with open(args.output_folder + "/args.json", "w") as f:
        json.dump(vars(args), f, indent=2) # tracks everything that runs this program 

    # read rollout settings
    rollout_num_episodes = args.n_rollouts
    rollout_horizon = args.horizon
    config, _ = FileUtils.config_from_checkpoint(ckpt_dict=ckpt_dict)
    if rollout_horizon is None:
        # read horizon from config
        rollout_horizon = config.experiment.rollout.horizon

    # HACK: assume absolute actions for now if using diffusion policy on real robot
    if (algo_name == "diffusion_policy") and EnvUtils.is_real_robot_gprs_env(env_meta=ckpt_dict["env_metadata"]):
        ckpt_dict["env_metadata"]["env_kwargs"]["absolute_actions"] = True

    # create environment from saved checkpoint
    env, _ = FileUtils.env_from_checkpoint(
        ckpt_dict=ckpt_dict, 
        env_name=args.env, 
        render=args.render, 
        render_offscreen=(args.video_path is not None), 
        verbose=True,
    )

    universal_state = env.get_state() if args.same_env else None

    # Auto-fill camera rendering info if not specified
    if args.camera_names is None:
        # We fill in the automatic values
        env_type = EnvUtils.get_env_type(env=env)
        args.camera_names = DEFAULT_CAMERAS[env_type]
    if args.render:
        # on-screen rendering can only support one camera
        assert len(args.camera_names) == 1

    is_real_robot = EnvUtils.is_real_robot_env(env=env) or EnvUtils.is_real_robot_gprs_env(env=env)
    if is_real_robot:
        # on real robot - log some warnings
        need_pause = False
        if "env_name" not in ckpt_dict["env_metadata"]["env_kwargs"]:
            log_warning("env_name not in checkpoint...proceed with caution...")
            need_pause = True
        if ckpt_dict["env_metadata"]["env_name"] != "EnvRealPandaGPRS":
            # we will load EnvRealPandaGPRS class by default on real robot even if agent was collected with different class
            log_warning("env name in metadata appears to be class ({}) different from EnvRealPandaGPRS".format(ckpt_dict["env_metadata"]["env_name"]))
            need_pause = True
        if need_pause:
            ans = input("continue? (y/n)")
            if ans != "y":
                exit()

    # maybe set seed
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    # maybe create video writer
    video_writer = None
    if write_video:
        video_writer = imageio.get_writer(args.video_path, fps=20)

    # maybe open hdf5 to write rollouts
    write_dataset = (args.dataset_path is not None)
    if write_dataset:
        data_writer = h5py.File(args.dataset_path, "w")
        data_grp = data_writer.create_group("data")
        total_samples = 0

    rollout_stats = []
    for i in tqdm(range(rollout_num_episodes)):
        try:
            stats, traj = rollout(
                policy=policy, 
                env=env, 
                horizon=rollout_horizon, 
                render=args.render, 
                video_writer=video_writer, 
                video_skip=args.video_skip, 
                return_obs=(write_dataset and args.dataset_obs),
                camera_names=args.camera_names,
                real=is_real_robot,
                rate_measure=rate_measure,
                classifier_grad = classifier_grad,
                reset_state = universal_state,
                model = model,
                exp_dir = args.output_folder,
                rollout_count = i,
                target_list = args.target_list,
                render_visuals = args.render_visuals,
                setup = args.setup,
                ss = args.ss 
            )
        except KeyboardInterrupt:
            if is_real_robot:
                print("ctrl-C catched, stop execution")
                print("env rate measure")
                print(env.rate_measure)
                ans = input("success? (y / n)")
                rollout_stats.append((1 if ans == "y" else 0))
                print("*" * 50)
                print("have {} success out of {} attempts".format(np.sum(rollout_stats), len(rollout_stats)))
                print("*" * 50)
                continue
            else:
                sys.exit(0)
        
        if is_real_robot:
            print("TERMINATE WITHOUT KEYBOARD INTERRUPT...")
            ans = input("success? (y / n)")
            rollout_stats.append((1 if ans == "y" else 0))
            continue
        rollout_stats.append(stats)

        if write_dataset:
            # store transitions
            ep_data_grp = data_grp.create_group("demo_{}".format(i))
            ep_data_grp.create_dataset("actions", data=np.array(traj["actions"]))
            # ep_data_grp.create_dataset("states", data=np.array(traj["states"]))
            ep_data_grp.create_dataset("rewards", data=np.array(traj["rewards"]))
            ep_data_grp.create_dataset("dones", data=np.array(traj["dones"]))
            if args.dataset_obs:
                for k in traj["obs"]:
                    ep_data_grp.create_dataset("obs/{}".format(k), data=np.array(traj["obs"][k]))
                    ep_data_grp.create_dataset("next_obs/{}".format(k), data=np.array(traj["next_obs"][k]))

            # episode metadata
            if "model" in traj["initial_state_dict"]:
                ep_data_grp.attrs["model_file"] = traj["initial_state_dict"]["model"] # model xml for this episode
            ep_data_grp.attrs["num_samples"] = traj["actions"].shape[0] # number of transitions in this episode
            total_samples += traj["actions"].shape[0]

    rollout_stats = TensorUtils.list_of_flat_dict_to_dict_of_list(rollout_stats)
    avg_rollout_stats = { k : np.mean(rollout_stats[k]) for k in rollout_stats }
    avg_rollout_stats["Num_Success"] = np.sum(rollout_stats["Success_Rate"])
    avg_rollout_stats["Time_Episode"] = np.sum(rollout_stats["time"]) / 60. # total time taken for rollouts in minutes
    avg_rollout_stats["Num_Episode"] = len(rollout_stats["Success_Rate"]) # number of episodes attempted
    print("Average Rollout Stats")
    stats_json = json.dumps(avg_rollout_stats, indent=4)
    print(stats_json)
    if args.json_path is not None:
        json_f = open(args.json_path, "w")
        json_f.write(stats_json)
        json_f.close()

    if write_video:
        video_writer.close()

    if write_dataset:
        # global metadata
        data_grp.attrs["total"] = total_samples
        data_grp.attrs["env_args"] = json.dumps(env.serialize(), indent=4) # environment info
        data_writer.close()
        print("Wrote dataset trajectories to {}".format(args.dataset_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Path to trained model
    parser.add_argument(
        "--agent",
        type=str,
        default = None,
        help="path to saved checkpoint pth file",
    )

    # Path to trained model
    parser.add_argument(
        "--guidance",
        type=str,
        default = None,
        help="path to saved checkpoint pth file",
    )

    # number of rollouts
    parser.add_argument(
        "--n_rollouts",
        type=int,
        default=27,
        help="number of rollouts",
    )

    # maximum horizon of rollout, to override the one stored in the model checkpoint
    parser.add_argument(
        "--horizon",
        type=int,
        default=None,
        help="(optional) override maximum horizon of rollout from the one in the checkpoint",
    )

    # Env Name (to override the one stored in model checkpoint)
    parser.add_argument(
        "--env",
        type=str,
        default=None,
        help="(optional) override name of env from the one in the checkpoint, and use\
            it for rollouts",
    )

    # Whether to render rollouts to screen
    parser.add_argument(
        "--render",
        action='store_true',
        help="on-screen rendering",
    )

    # Dump a video of the rollouts to the specified path
    parser.add_argument(
        "--video_path",
        type=str,
        default=None,
        help="(optional) render rollouts to this video file path",
    )

     # Dump a video of the rollouts to the specified path
    parser.add_argument(
        "--setup",
        type=str,
        default=None,
        help="(optional) render rollouts to this video file path",
    )


    # How often to write video frames during the rollout
    parser.add_argument(
        "--video_skip",
        type=int,
        default=1,
        help="render frames to video every n steps",
    )

        # How often to write video frames during the rollout
    parser.add_argument(
        "--ss",
        type=int,
        default=4,
        help="render frames to video every n steps",
    )

    # camera names to render
    parser.add_argument(
        "--camera_names",
        type=str,
        nargs='+',
        default=None,
        help="(optional) camera name(s) to use for rendering on-screen or to video",
    )

    # If provided, an hdf5 file will be written with the rollout data
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        help="(optional) if provided, an hdf5 file will be written at this path with the rollout data",
    )

    # If True and @dataset_path is supplied, will write possibly high-dimensional observations to dataset.
    parser.add_argument(
        "--dataset_obs",
        action='store_true',
        help="include possibly high-dimensional observations in output dataset hdf5 file (by default,\
            observations are excluded and only simulator states are saved)",
    )

    parser.add_argument(
        "--render_visuals",
        action='store_true',
        help="include possibly high-dimensional observations in output dataset hdf5 file (by default,\
            observations are excluded and only simulator states are saved)",
    )

        # If True and @dataset_path is supplied, will write possibly high-dimensional observations to dataset.
    parser.add_argument(
        "--same_env",
        action='store_true',
        help="reset to the same environment every time",
    )

    # for seeding before starting rollouts
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="(optional) set seed for rollouts",
    )

        # for seeding before starting rollouts
    parser.add_argument(
        "--scale",
        type=float,
        default=None,
        help="How much to influence",
    )
    def list_of_floats(arg):
        float_list= list(map(float, arg.split(',')))
        if len(float_list) == 5:
            return float_list[1:]
        return float_list 
        # return list(map(float, arg.split(',')))
            # for seeding before starting rollouts
    parser.add_argument(
        "--target_list",
        type=list_of_floats,
        default=None,
        help="specifying the target color",
    )


    # Dump a json of the rollout results stats to the specified path
    parser.add_argument(
        "--json_path",
        type=str,
        default=None,
        help="(optional) dump a json of the rollout results stats to the specified path",
    )

    
    # Dump a json of the rollout results stats to the specified path
    parser.add_argument(
        "--output_folder",
        type=str,
        default=None,
        help="",
    )


    # Dump a file with the error traceback at this path. Only created if run fails with an error.
    parser.add_argument(
        "--error_path",
        type=str,
        default=None,
        help="(optional) dump a file with the error traceback at this path. Only created if run fails with an error.",
    )

    # TODO: clean up this arg
    # If provided, do not run actions in env, and instead just measure the rate of action computation
    parser.add_argument(
        "--hz",
        type=int,
        default=None,
        help="If provided, do not run actions in env, and instead just measure the rate of action computation and raise warnings if it dips below this threshold",
    )

    # TODO: clean up this arg
    # If provided, set num_inference_timesteps explicitly for diffusion policy evaluation
    parser.add_argument(
        "--dp_eval_steps",
        type=int,
        default=None,
        help="If provided, set num_inference_timesteps explicitly for diffusion policy evaluation",
    )

    args = parser.parse_args()
    res_str = None
    try:
        run_trained_agent(args)
    except Exception as e:
        res_str = "run failed with error:\n{}\n\n{}".format(e, traceback.format_exc())
        if args.error_path is not None:
            # write traceback to file
            f = open(args.error_path, "w")
            f.write(res_str)
            f.close()
        raise e
