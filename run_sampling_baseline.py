
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
import random 

from core.dynamics_models import FinalStatePredictionDino

from core.embedder_datasets import MultiviewDataset
import cv2 
import matplotlib.pyplot as plt 
import copy 
import shutil 

# robot_initial_positions = None # uncomment this if you want to start the robot at an initial position all the time 
robot_initial_positions = list() 
with open("/store/real/maxjdu/repos/robotrainer/dataset/initial_calvin_robot_states_constrained.json", "r") as f:
    poses = json.load(f) 
    robot_initial_positions = [np.array(k) for k in poses["robot_states"]]


def generate_reset_state(sim_hold = {}):
    # {"sliding_door" : state_obs[0],
    #         "drawer" : state_obs[1],
    #         "button" : state_obs[2],
    #         "switch" : state_obs[3],
    #         "lightbulb" : state_obs[4],
    #         "green_light" : state_obs[5],
    #         "red_block": state_obs[6:9], # we are ignoring rotations 
    #         "blue_block" : state_obs[12 : 15],
    #         "pink_block" : state_obs[18 :21]}, {
    #         "red_rot": R.from_euler("XYZ", state_obs[9:12]).as_matrix(), # we are ignoring rotations 
    #         "blue_rot" : R.from_euler("XYZ", state_obs[15:18]).as_matrix(),
    #         "pink_rot" : R.from_euler("XYZ", state_obs[21:]).as_matrix()
    #         }
    adjustables = ["sliding_door", "drawer", "switch", "green_light"]
    adjustable_index = [0, 1, 3, 5]
    adjustable_limits = [[0, 0.27], [0, 0.16], [0, 0.09], [0, 1]]
    state = np.zeros((24,))
    binary_list = list()
    # return state, [False, False, False, False]
    for adjustable, idx, limits in zip(adjustables, adjustable_index, adjustable_limits):
        if adjustable in sim_hold:
            state[idx] = float(sim_hold[adjustable])
            binary_list.append(sim_hold[adjustable] > (limits[1] - limits[0]) / 2)
        # are just doing binaries because it's more in distribution 
        else:
            select = random.random() > 0.5 
            state[idx] = limits[1] if select else limits[0]
            binary_list.append(select)
    return state, binary_list 

def articulated_binaries_from_start_state(start_state):
    adjustable_index = [0, 1, 3, 5]
    adjustable_limits = [[0, 0.27], [0, 0.16], [0, 0.09], [0, 1]]
    binary_list = list()
    for idx, limits in zip(adjustable_index, adjustable_limits):
        midpoint = (limits[1] - limits[0]) / 2
        binary_list.append(start_state[idx] > midpoint)
    return binary_list 


def check_state_difference(start_state, state, robot_pos, binaries, for_display = False):
    # if you want to use this code for clean display, we use different thresholds 
    # adjustables = ["sliding_door", "drawer", "switch", "green_light"]
    adjustable_index = [0, 1, 3, 5]
    adjustable_limits = [[0, 0.27], [0, 0.16], [0, 0.09], [0, 1]]
    for binary, idx, limits in zip(binaries, adjustable_index, adjustable_limits):
        midpoint = (limits[1] - limits[0]) / 2
        near_low = limits[0] + 0.1 * (limits[1] - limits[0])
        near_high = limits[0] + 0.9 * (limits[1] - limits[0])

        if not for_display:
            if binary and state[idx] < midpoint:
                return True 
            if not binary and state[idx] > midpoint:
                return True 
        else: # this is for display where we want the task to fully finish 
            if binary and state[idx] < near_low:
                return True 
            if not binary and state[idx] > near_high:
                return True 

    if np.linalg.norm(robot_pos - state[6:9]) < 0.06 and np.linalg.norm(state[6:8] - start_state[6:8]) > 0.001:
        return True 

    if np.linalg.norm(robot_pos - state[12:15]) < 0.06 and np.linalg.norm(state[12:14] - start_state[12:14]) > 0.001:
        return True 

    if np.linalg.norm(robot_pos - state[18:21]) < 0.06 and np.linalg.norm(state[18:20] - start_state[18:20]) > 0.001:
        return True 
    
    return False 


def prepare_np(data, device = "cuda"):
    if type(data) == dict:
        return {k : torch.tensor(v).to(device).to(torch.float32) for k, v in data.items()}
    return torch.tensor(data).to(device).to(torch.float32)

def precompute_good_final_states(model, good_dataset):
    good_embeddings_list = list()
    print("Precomputing good")
    idx = 0
    good_img_list = list() 
    for length in tqdm(good_dataset.lengths_list):
        idx += length 
        sample = good_dataset.get_labeled_item(idx - 1, flatten_action = False)
        # sample = good_dataset.get_labeled_item(idx - 16, flatten_action = False)
        state, action, label = prepare_np(sample[0]), prepare_np(sample[1]), sample[2]
        good_img_list.append(np.transpose(state["third_person"].detach().cpu().numpy(), (1, 2, 0)) / 255)
        state = {k : torch.unsqueeze(v, dim = 0) for k, v in state.items()}
        action = torch.unsqueeze(action, dim = 0) # compensates for the batch dimension 
        with torch.no_grad():
            good_embedding = model.state_embedding(state, normalize = False).flatten(start_dim = 1) # gets the s, a embedding only 
            # good_embedding = model.state_action_embedding(state, action, normalize = False) # gets the s, a embedding only 
        good_embeddings_list.append(good_embedding.clone())
    good_embeddings = torch.concatenate(good_embeddings_list, dim = 0)
    return good_embeddings 

def run_rejection_sampling(model, actions_list, good_embeddings, state):
    # returns the best action and annotates the image with the similiarty score 
    fed_state = {"third_person": state["third_person"][:, -1] * 255}
    fed_state["proprio"] = state["proprio"][:, -1]

    predicted_image_list = list() 
    scores_list = list() 
    visualized_state = np.transpose(state["third_person"][0, -1].detach().cpu().numpy(), (1, 2, 0)) * 255
    visualized_state = visualized_state.astype(np.uint8)
    required_padding = (5 - (len(actions_list) % 5)) % 5
    pad = np.zeros((128, 128, 3))
    for action in actions_list:
        with torch.no_grad():
            # s_end_embedding = model.state_action_embedding(fed_state, action).flatten(start_dim = 1) # gets the s, a embedding only 
            s_end_embedding, reco_image = model(fed_state, action) # gets the s, a embedding only 
        
        s_end_embedding = s_end_embedding.flatten(start_dim = 1)
        s_norm = torch.cdist(good_embeddings, s_end_embedding, p=2.0)
        sa_average_norm = torch.mean(s_norm).detach().cpu().numpy()
        scores_list.append(sa_average_norm)
        # reco_image = model.image_reconstruct(s_end_embedding)[0].detach().cpu().numpy() # batch removal 

        reco_image = np.transpose(np.clip(reco_image[0].detach().cpu().numpy(), 0, 1), (1, 2, 0)).copy() * 255
        reco_image = reco_image.astype(np.uint8)
        reco_image = cv2.resize(reco_image, (128, 128))
        reco_image = cv2.putText(reco_image, str(int(sa_average_norm.item())), (10, 100), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 1, cv2.LINE_AA)
        predicted_image_list.append(reco_image.copy()) # copy is needed for the tiling 

    closest =  np.argmin(np.array(scores_list))
    predicted_image_list[closest] = cv2.rectangle(predicted_image_list[closest], (1, 1), (127, 127), (0, 255, 0), 5)

    
    grid_list = list()
    row_list = list() 
    for i in range(len(predicted_image_list)):
        if i % 5 == 0 and i > 0:
            grid_list.append(np.concatenate(row_list, axis = 1))
            row_list = list()
        row_list.append(predicted_image_list[i])
    for i in range(required_padding):
        row_list.append(pad)
    grid_list.append(np.concatenate(row_list, axis = 1)) # final row might have black padding 
    collage = np.concatenate(grid_list, axis = 0)

    visualized_state = cv2.resize(visualized_state, (128 * 5, 128 * 5))
    final_collage = np.concatenate((collage, visualized_state), axis = 0) 
    return closest, final_collage, scores_list, collage

        

def rollout(policy, env, horizon, render=False, video_writer=None, video_skip=5, return_obs=False, camera_names=None, real=False, rate_measure=None,
            reset_state = None, good_embeddings = None, bad_embeddings = None, end_predictor = None, exp_dir = None, rollout = None, save_image = True, 
            setup_config = None, num_samples = 10):
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

    # normal
    # obs = env.reset()     
    env.reset()
    
    # this is for calvin envrionment     
    special_state, articulated_binaries = generate_reset_state(sim_hold = setup_config["env_setup"]) # and keep the switch off 
    if robot_initial_positions is not None:
        state_to_reset = {"scene" : special_state, "robot" : random.choice(robot_initial_positions)}
    else:
        state_to_reset = special_state 

    obs = env.reset_to(state_to_reset)
    state_dict = env.get_state()
    start_state = obs.copy()


    results = {}
    video_count = 0  # video frame counter
    total_reward = 0.
    got_exception = False
    success = env.is_success()["task"]
    traj = dict(actions=[], rewards=[], dones=[], states=[], initial_state_dict=state_dict)
    if return_obs:
        # store observations too
        traj.update(dict(obs=[], next_obs=[]))

    topframe = None 
    try:
        for step_i in range(horizon):
            print(step_i)
            # HACK: some keys on real robot do not have a shape (and then they get frame stacked)
            for k in obs:
                if len(obs[k].shape) == 1:
                    obs[k] = obs[k][..., None] 

            # get action from policy
            t1 = time.time()

            # INJECTION OF REJECTION SAMPLING
            recompute_interval = 14 #8
            log_obs = copy.deepcopy(obs)
            obs = TensorUtils.to_tensor(obs)
            obs = TensorUtils.to_batch(obs)
            obs = TensorUtils.to_device(obs, "cuda")
            obs = TensorUtils.to_float(obs)
            if step_i % recompute_interval == 0:
                # this is when you want to recompute 
                actions_list = list() 
                for i in tqdm(range(num_samples)):
                    with torch.no_grad():
                        actions = policy.policy.get_full_action(obs)
                    actions_list.append(actions)

                selection, collage, scores_list, topframe = run_rejection_sampling(end_predictor, actions_list, good_embeddings, obs)

                selected_chunk = actions_list[selection][0].detach().cpu() # this takes the batch away and turns it into np
                policy.policy.set_full_action(selected_chunk) # forcing the policy to adopt this action 
                print(scores_list) 
                print(selection)

                if save_image:
                    plt.imsave(exp_dir + f"/{rollout}_{step_i}.png", collage)

            act = policy(ob=obs)
            #######################################

            # print(act)
            t2 = time.time()
            if real and (not env.base_env.controller_type == "JOINT_IMPEDANCE") and (policy.policy.global_config.algo_name != "diffusion_policy"):
                # joint impedance actions and diffusion policy actions are absolute in the real world
                act = np.clip(act, -1., 1.)

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
                next_obs, r, done, _ = env.step(act)

            # compute reward
            total_reward += r
            success = env.is_success()["task"]

            # visualization
            if render:
                env.render(mode="human", camera_name=camera_names[0])
            if video_writer is not None:
                if video_count % video_skip == 0:
                    video_img = []
                    for cam_name in camera_names:
                        video_img.append(env.render(mode="rgb_array", height=512, width=512, camera_name=cam_name))
                    video_img = np.concatenate(video_img, axis=1) # concatenate horizontally
                    # this logic adds the action selection annotation 
                    video_img = cv2.resize(video_img, (128 * 5, 128 * 5))
                    video_img = np.concatenate((topframe, video_img), axis = 0)
                    video_writer.append_data(video_img)
                video_count += 1

            # collect transition
            traj["actions"].append(act)
            traj["rewards"].append(r)
            traj["dones"].append(done)
            if not real:
                # traj["states"].append(state_dict["states"])
                # traj["states"].append(state_dict)
                traj["states"].append(0)
            if return_obs:
                # Note: We need to "unprocess" the observations to prepare to write them to dataset.
                #       This includes operations like channel swapping and float to uint8 conversion
                #       for saving disk space.
                # traj["obs"].append(ObsUtils.unprocess_obs_dict(obs))
                traj["obs"].append(ObsUtils.unprocess_obs_dict(log_obs))
                traj["next_obs"].append(ObsUtils.unprocess_obs_dict(next_obs))

            # break if done or if success
            fed_state = log_obs["states"]
            fed_start_state = start_state["states"]
            fed_proprio = log_obs["proprio"]
            if len(log_obs["states"].shape) == 2: # collapse this if there's framestacking for the policy 
                fed_state = fed_state[-1] #obs["states"][-1]
                fed_start_state = fed_start_state[-1]
                fed_proprio = fed_proprio[-1]
            done = check_state_difference(fed_start_state, fed_state, fed_proprio[0:3], articulated_binaries)

            if done or success:
                break

            # update for next iter
            obs = deepcopy(next_obs)
            if not real:
                state_dict = env.get_state()

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
        try:
            if isinstance(traj[k], dict):
                for kp in traj[k]:
                    traj[k][kp] = np.array(traj[k][kp])
            else:
                traj[k] = np.array(traj[k])
        except:
            import ipdb 
            ipdb.set_trace()

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

    # restore policy
    policy, ckpt_dict = FileUtils.policy_from_checkpoint(ckpt_dict=ckpt_dict, device=device, verbose=True)

    shutil.copy("exp.sh", args.output_folder + "/exp.sh") # because there are variants 


    ############# REJECTION SAMPLING ################
    # this needs to be aligned with the action chunk length in the trained model 
    ACTION_DIM = 7 
    ACTION_CHUNK_LENGTH = 16 # this is how long the action predictions are
    # cameras = ["agentview_image", "third_person"] # you can change this; it's hardcoded
    cameras = ["third_person"] # you can change this; it's hardcoded
    proprio_dim = 15 
    proprio = "proprio" # set to None if you want to exclude propriorception 
    padding = True
    pad_mode = "repeat" #"zeros" #"repeat" # "zeros" for calvin 

    model = FinalStatePredictionDino(ACTION_DIM, ACTION_CHUNK_LENGTH, cameras=cameras, reconstruction = True, \
                                     proprio = proprio, proprio_dim = proprio_dim)    
    model.load_state_dict(torch.load(args.guidance))
    model.to("cuda")
    model.eval()

    exp_setup_config = None 
    if args.exp_setup_config is not None: 
        with open(args.exp_setup_config, "r") as f:
            exp_setup_config = json.load(f)
    # ensure backward compatibility 
    good_dataset_path = args.good_states if args.exp_setup_config is None else exp_setup_config["pos_examples"]

    good_dataset = MultiviewDataset(good_dataset_path, action_chunk_length = ACTION_CHUNK_LENGTH, cameras = cameras, \
                                    padding = padding, pad_mode = pad_mode, proprio = proprio)
    
    bad_dataset = None 
    bad_embeddings = None 
    if exp_setup_config["use_neg"]:
        bad_dataset_path = exp_setup_config["neg_examples"]

        bad_dataset = MultiviewDataset(bad_dataset_path, action_chunk_length = ACTION_CHUNK_LENGTH, cameras = cameras, \
                                        padding = padding, pad_mode = pad_mode, proprio = proprio)
    
        bad_dataset = precompute_good_final_states(model, bad_dataset)
    good_embeddings = precompute_good_final_states(model, good_dataset)
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

    shutil.copy("rejection_sampling_dino.py", args.output_folder + "/rejection_sampling_dino.py") # because there are variants 
    with open(args.output_folder + "/args.json", "w") as f:
        json.dump(vars(args), f) # tracks everything that runs this program 

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
                reset_state = universal_state,
                good_embeddings = good_embeddings,
                bad_embeddings = bad_embeddings,
                end_predictor = model,
                exp_dir = args.output_folder,
                rollout = i,
                save_image = args.save_image,
                setup_config  = exp_setup_config,
                num_samples = args.num_samples
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
            ep_data_grp.create_dataset("states", data=np.array(traj["states"]))
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

    # Path to trained model
    parser.add_argument(
        "--good_states",
        type=str,
        default = None,
        help="examples of good states",
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

    # maximum horizon of rollout, to override the one stored in the model checkpoint
    parser.add_argument(
        "--num_samples",
        type=int,
        default=10,
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

    # Whether to render rollouts to screen
    parser.add_argument(
        "--save_image",
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

    # How often to write video frames during the rollout
    parser.add_argument(
        "--video_skip",
        type=int,
        default=1,
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
        help="(optional) dump a json of the rollout results stats to the specified path",
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

    parser.add_argument(
        "--exp_setup_config",
        type=str,
        default=None,
        help="(optional) dump a json of the rollout results stats to the specified path",
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
