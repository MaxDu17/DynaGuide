"""
The main script for evaluating a policy in an environment.

Args:
    agent (str): path to saved checkpoint pth file

    horizon (int): if provided, override maximum horizon of rollout from the one 
        in the checkpoint

    env (str): if provided, override name of env from the one in the checkpoint,
        and use it for rollouts

    render (bool): if flag is provided, use on-screen rendering during rollouts

    video_path (str): if provided, render trajectories to this video file path

    video_skip (int): render frames to a video every @video_skip steps

    camera_names (str or [str]): camera name(s) to use for rendering on-screen or to video

    dataset_path (str): if provided, an hdf5 file will be written at this path with the
        rollout data

    dataset_obs (bool): if flag is provided, and @dataset_path is provided, include 
        possible high-dimensional observations in output dataset hdf5 file (by default,
        observations are excluded and only simulator states are saved).

    seed (int): if provided, set seed for rollouts

Example usage:

    # Evaluate a policy with 50 rollouts of maximum horizon 400 and save the rollouts to a video.
    # Visualize the agentview and wrist cameras during the rollout.
    
    python run_trained_agent.py --agent /path/to/model.pth \
        --n_rollouts 50 --horizon 400 --seed 0 \
        --video_path /path/to/output.mp4 \
        --camera_names agentview robot0_eye_in_hand 

    # Write the 50 agent rollouts to a new dataset hdf5.

    python run_trained_agent.py --agent /path/to/model.pth \
        --n_rollouts 50 --horizon 400 --seed 0 \
        --dataset_path /path/to/output.hdf5 --dataset_obs 

    # Write the 50 agent rollouts to a new dataset hdf5, but exclude the dataset observations
    # since they might be high-dimensional (they can be extracted again using the
    # dataset_states_to_obs.py script).

    python run_trained_agent.py --agent /path/to/model.pth \
        --n_rollouts 50 --horizon 400 --seed 0 \
        --dataset_path /path/to/output.hdf5
"""
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




class ReachingPolicyKey:
    def __init__(self, num_cubes = 4, noise = True):
        self.tolerance = 0.005
        self.num_cubes = num_cubes 
        self.noise = noise

    def select_random_waypoint(self, obs):
        threshold = 20 / 512 # 30 pixels but everything is 0->1 so we make it like this 
        counter = 0 
        # print("selecting random waypoint away from other cubes")
        while True:
            counter += 1 
            waypoint = np.random.rand(2) 
        
            for i in range(self.num_cubes):
                cube_pos = obs["states"][2 * i : 2 * i + 2]
                if np.linalg.norm(cube_pos - waypoint) > threshold:
                    # print(f"\t Took {counter} iterations.")
                    return waypoint 
                
    def start_episode(self, target, obs):
        if target is not None: 
            self.target_cube = target 
        else: 
            self.target_cube = random.randint(1, self.num_cubes - 1)
        
        # TODO: the obs yielded by reset is not the same as step in terms of dimension 
        target_cube = obs["states"][2 * self.target_cube : 2 * self.target_cube + 2]
        num_waypoints = random.randint(0, 1) #random.randint(0, 3) 

        current_pos = obs["agent_pos"]

        waypoint_list = [current_pos]
        dist_est = 0 
        last_waypoint = current_pos
        for i in range(num_waypoints):
            waypoint = self.select_random_waypoint(obs)
            waypoint_list.append(waypoint)
            dist_est += np.linalg.norm(waypoint - last_waypoint)
            last_waypoint = waypoint 
        dist_est += np.linalg.norm(target_cube - last_waypoint)
        waypoint_list.append(target_cube)
        # waypoint = self.select_random_waypoint(obs)
        # estimated_distance = np.linalg.norm(waypoint - target_cube) + np.linalg.norm(current_pos - waypoint) 
        num_steps = int(dist_est * 50) # this calculates how fast we should go based on an estimate of our length 
    
        self.step_counter = 0 
        # self.curve = self.bezier_curve(current_pos, waypoint, target_cube, num_steps)
        self.curve = self.reparameterize(waypoint_list, num_steps)

    def bezier_point(self, control_points, t):
        """Calculate a point on a Bézier curve for a given t."""
        import math 
        n = len(control_points) - 1
        point = np.zeros_like(control_points[0])
        for i in range(n + 1):
            coefficient = math.comb(n, i) * (t ** i) * ((1 - t) ** (n - i))
            point += coefficient * control_points[i]
        return point

    def arc_length(self, control_points, steps=100):
        """Calculate the arc length of a Bézier curve."""
        t_values = np.linspace(0, 1, steps)
        points = np.array([self.bezier_point(control_points, t) for t in t_values])
        distances = np.linalg.norm(np.diff(points, axis=0), axis=1)
        return np.cumsum(distances)

    def reparameterize(self, control_points, num_points):
        """Reparameterize the curve to generate equally spaced points."""
        # total_length = self.arc_length(control_points, steps=100)[-1]
        arc_lengths = self.arc_length(control_points, steps=500) # distances you travel on the line 
        total_length = arc_lengths[-1]
        target_lengths = np.linspace(0, total_length, num_points) # this is how far you want to travel in each point 


        t_values = np.linspace(0, 1, 500)
        reparameterized_t = []
        for length in target_lengths:
            closest_index = np.argmin(np.abs(arc_lengths - length))
            reparameterized_t.append(t_values[closest_index])

        points = np.array([self.bezier_point(control_points, t) for t in reparameterized_t])
        return points
    
    def __call__(self, ob):
        target_cube = ob["states"][2 * self.target_cube : 2 * self.target_cube + 2][:, 0]  
        current_pos = ob["agent_pos"][:, 0]
        # target_cube = ob["cubes_pos"][self.target_cube]
        # target_cube[-1] += 0.02 # so we don't collide
        # target_cube[0] -= 0.012 #so we touch on the lower part of the cube 
        # action = current_pos #np.zeros(current_pos)
        # delta = target_cube - ob["robot0_eef_pos"][:, 0]
        if self.step_counter >= len(self.curve):
            action = self.curve[-1]
        else:
            action = self.curve[self.step_counter]
            self.step_counter += 1 
        
        if self.noise:
            action += 0.03 * (np.random.rand(2) - 0.5)  
        # action = current_pos 
        # action[0] += -0.05
    
        # delta_unit_vector = delta / np.linalg.norm(delta)
        # delta_unit_vector += self.wind * np.linalg.norm(delta) # this skews the movement and less towards the end. Constant "wind" in the environment
        
        # sent_delta = 0.3 * delta_unit_vector
        # if self.noise:
        #     sent_delta = sent_delta + 0.1 * np.random.rand(sent_delta.shape[0])

        # sent_delta = np.clip(sent_delta, -1, 1)
        # if self.noise:
        #     action = np.concatenate((sent_delta, 0.15 * (np.random.rand(3) - 0.5), np.array([1])), axis = 0)
        # else:
        #     action = np.concatenate((sent_delta, np.zeros(3), np.array([1])), axis = 0)
        # # action = np.concatenate((sent_delta, np.zeros(3), np.array([1])), axis = 0)
        return action 

def rollout(policy, env, horizon, render=False, video_writer=None, video_skip=5, return_obs=False, camera_names=None, real=False, rate_measure=None, key = None, reset_to = None):
    SAVE_ALL = False # save all videos 
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
    # assert isinstance(policy, RolloutPolicy)
    assert not (render and (video_writer is not None))

    # want to reset every time we hit key = 0 

    if reset_to is not None:
        obs = env.reset_to(reset_to)
    else:
        obs = env.reset()


    # key = count % 4 # this will cycle through the cubes to touch 
    policy.start_episode(key, obs)


    state_dict = dict()
    if real:
        input("ready for next eval? hit enter to continue")
    # else:
    #     state_dict = env.get_state()
    #     # hack that is necessary for robosuite tasks for deterministic action playback
    #     obs = env.reset_to(state_dict)

    results = {}
    video_count = 0  # video frame counter
    total_reward = 0.
    got_exception = False
    success = env.is_success()["task"]
    traj = dict(actions=[], rewards=[], dones=[], states=[], initial_state_dict=state_dict, label=[])
    if return_obs:
        # store observations too
        traj.update(dict(obs=[], next_obs=[]))
    try:
        video_list = list()
        for step_i in range(horizon):
            # HACK: some keys on real robot do not have a shape (and then they get frame stacked)
            for k in obs:
                if len(obs[k].shape) == 1:
                    obs[k] = obs[k][..., None] 

            # get action from policy
            t1 = time.time()
            act = policy(ob=obs)
            t2 = time.time()
            # THIS IS COMMENTED OUT SO DON'T USE A REAL ROBOT ON THIS
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
                next_obs, r, done, info = env.step(act)
            # compute reward
            total_reward += r
            success = r > 0 # very specific to this task 
            # success = env.is_success()["task"]

            # visualization
            if render:
                env.render(mode="human", camera_name=camera_names[0])
            if video_writer is not None:
                if video_count % video_skip == 0:
                    video_img = []
                    for cam_name in camera_names:
                        video_img.append(env.render(mode="rgb_array", height=512, width=512, camera_name=cam_name))
                    video_img = np.concatenate(video_img, axis=1) # concatenate horizontally
                    video_list.append(video_img)
                video_count += 1

            # collect transition
            traj["actions"].append(act)
            traj["rewards"].append(r)
            traj["dones"].append(done)

            traj["label"].append(key) #this is for the classifier 
            # if not real:
            #     traj["states"].append(state_dict["states"])
            if return_obs:
                # Note: We need to "unprocess" the observations to prepare to write them to dataset.
                #       This includes operations like channel swapping and float to uint8 conversion
                #       for saving disk space.
                obs = {k: np.squeeze(v) for k, v in obs.items()} # HACK: this will remove the additional dimension which is incorrect for later training 
                next_obs = {k: np.squeeze(v) for k, v in next_obs.items()}
                traj["obs"].append(ObsUtils.unprocess_obs_dict(obs))
                traj["next_obs"].append(ObsUtils.unprocess_obs_dict(next_obs))


            # break if done or if success
            if done or success:
                break
            
            # if step_i == horizon - 1:
            #     import ipdb 
            #     ipdb.set_trace() 
            # update for next iter
            obs = deepcopy(next_obs)
            # if not real:
            #     state_dict = env.get_state()

    except env.rollout_exceptions as e:
        print("WARNING: got rollout exception {}".format(e))
        got_exception = True
    
    if info["cube_contacted"] != key or step_i < 8: # don't hit cubes that are too close 
        success = False # we hit something along the way, call this a failure 
    
    if success or SAVE_ALL:
        for frame in video_list:
            video_writer.append_data(frame)

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
    # some arg checking
    write_video = (args.video_path is not None)
    assert not (args.render and write_video) # either on-screen or video but not both

    if args.output_folder is not None and not os.path.isdir(args.output_folder):
        os.mkdir(args.output_folder)

    rate_measure = None
    if args.hz is not None:
        import RobotTeleop
        from RobotTeleop.utils import Rate, RateMeasure, Timers
        rate_measure = RateMeasure(name="control_rate_measure", freq_threshold=args.hz)
    
    # load ckpt dict and get algo name for sanity checks

    # algo_name, ckpt_dict = FileUtils.algo_name_from_checkpoint(ckpt_path=args.agent)

    # if args.dp_eval_steps is not None:
    #     assert algo_name == "diffusion_policy"
    #     log_warning("setting @num_inference_steps to {}".format(args.dp_eval_steps))

    #     # HACK: modify the config, then dump to json again and write to ckpt_dict
    #     tmp_config, _ = FileUtils.config_from_checkpoint(ckpt_dict=ckpt_dict)
    #     with tmp_config.values_unlocked():
    #         if tmp_config.algo.ddpm.enabled:
    #             tmp_config.algo.ddpm.num_inference_timesteps = args.dp_eval_steps
    #         elif tmp_config.algo.ddim.enabled:
    #             tmp_config.algo.ddim.num_inference_timesteps = args.dp_eval_steps
    #         else:
    #             raise Exception("should not reach here")
    #     ckpt_dict['config'] = tmp_config.dump()
    # device
    device = TorchUtils.get_torch_device(try_to_use_cuda=True)

    # restore policy
    # policy, ckpt_dict = FileUtils.policy_from_checkpoint(ckpt_dict=ckpt_dict, device=device, verbose=True)

    # read rollout settings
    rollout_num_episodes = args.n_rollouts
    rollout_horizon = args.horizon
    # config, _ = FileUtils.config_from_checkpoint(ckpt_dict=ckpt_dict)
    # if rollout_horizon is None:
    #     # read horizon from config
    #     rollout_horizon = config.experiment.rollout.horizon

    # HACK: assume absolute actions for now if using diffusion policy on real robot
    # if (algo_name == "diffusion_policy") and EnvUtils.is_real_robot_gprs_env(env_meta=ckpt_dict["env_metadata"]):
    #     ckpt_dict["env_metadata"]["env_kwargs"]["absolute_actions"] = True

    # create environment from saved checkpoint
    with open(args.env_config, "r") as f:
        cfg = json.load(f)
        env_meta = cfg["env_meta"]
        shape_meta = cfg["shape_meta"]
        env = EnvUtils.create_env_from_metadata(
            env_meta=env_meta, 
            env_name=env_meta["env_name"],
            render=False,
            render_offscreen=(args.video_path is not None),
            use_image_obs=shape_meta.get("use_images", False),
            use_depth_obs=shape_meta.get("use_depths", False),
        )
        obs_specs = cfg["obs_specs"]
        ObsUtils.initialize_obs_utils_with_obs_specs(obs_specs)
    
    policy = ReachingPolicyKey(num_cubes = 4, noise = True) # TODO: THIS IS HARDCODED
        
    # env, _ = FileUtils.env_from_checkpoint(
    #     ckpt_dict=ckpt_dict, 
    #     env_name=args.env, 
    #     render=args.render, 
    #     render_offscreen=(args.video_path is not None), 
    #     verbose=True,
    # )

    # Auto-fill camera rendering info if not specified
    if args.camera_names is None:
        # We fill in the automatic values
        env_type = EnvUtils.get_env_type(env=env)
        args.camera_names = DEFAULT_CAMERAS[env_type]
    if args.render:
        # on-screen rendering can only support one camera
        assert len(args.camera_names) == 1

    is_real_robot = EnvUtils.is_real_robot_env(env=env) or EnvUtils.is_real_robot_gprs_env(env=env)
    # if is_real_robot:
    #     # on real robot - log some warnings
    #     need_pause = False
    #     if "env_name" not in ckpt_dict["env_metadata"]["env_kwargs"]:
    #         log_warning("env_name not in checkpoint...proceed with caution...")
    #         need_pause = True
    #     if ckpt_dict["env_metadata"]["env_name"] != "EnvRealPandaGPRS":
    #         # we will load EnvRealPandaGPRS class by default on real robot even if agent was collected with different class
    #         log_warning("env name in metadata appears to be class ({}) different from EnvRealPandaGPRS".format(ckpt_dict["env_metadata"]["env_name"]))
    #         need_pause = True
    #     if need_pause:
    #         ans = input("continue? (y/n)")
    #         if ans != "y":
    #             exit()

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
    i = 0

    REPEAT_ENVIRONMENT = args.repeat_environment # if you want uniform coverage 

    MAX_TRIES = 10
    try_count = 0 
    obs = env.reset()
    current_state = env.get_state() 

    BLUE_PROPORTION = 0.25 # this is for the 

    while i < rollout_num_episodes:
        print(f"On {i} out of {rollout_num_episodes}")
        key = 0
        #  ["Blue", "Red", "Green", "Yellow"]
        # if i < BLUE_PROPORTION * rollout_num_episodes: # ensure that correct proportion; ok because we shuffle 
        #     key = 0 
        # else:
        #     key = random.randint(1, 3)

        # key = i % 4 # for normal collection
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
                key = key,
                reset_to = current_state if REPEAT_ENVIRONMENT else None 
            )
        except KeyboardInterrupt:
            sys.exit(0)
        
        rollout_stats.append(stats)

        if args.keep_only_successful and stats["Success_Rate"] < 1:
            print("Failed run, not saved")
            try_count += 1 
            if try_count > MAX_TRIES: # sometimes you can't reach a cube in the environment even with one waypoint 
                env.reset() 
                current_state = env.get_state() 
                print("\t\tFAILED TOO MANY TIMES, TRYING ANOTHER ENVIRONMENT")
            continue
        else:
            # we are going to save 
            try_count = 0 

        if write_dataset:
            # store transitions
            ep_data_grp = data_grp.create_group("demo_{}".format(i))
            ep_data_grp.create_dataset("actions", data=np.array(traj["actions"]))
            ep_data_grp.create_dataset("states", data=np.array(traj["states"]))
            ep_data_grp.create_dataset("rewards", data=np.array(traj["rewards"]))
            ep_data_grp.create_dataset("dones", data=np.array(traj["dones"]))
            ep_data_grp.create_dataset("label", data=np.array(traj["label"]))
            if args.dataset_obs:
                for k in traj["obs"]:
                    ep_data_grp.create_dataset("obs/{}".format(k), data=np.array(traj["obs"][k]))
                    ep_data_grp.create_dataset("next_obs/{}".format(k), data=np.array(traj["next_obs"][k]))

            # episode metadata
            if "model" in traj["initial_state_dict"]:
                ep_data_grp.attrs["model_file"] = traj["initial_state_dict"]["model"] # model xml for this episode
            ep_data_grp.attrs["num_samples"] = traj["actions"].shape[0] # number of transitions in this episode
            total_samples += traj["actions"].shape[0]

        i += 1 
        # if we've made it this far, we're successful 
        if i % 4 == 0: # touch every cube 
            obs = env.reset()
            current_state = env.get_state() 

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
        "--env_config",
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

    # Whether to render rollouts to screen
    parser.add_argument(
        "--repeat_environment",
        action='store_true',
        help="on-screen rendering",
    )

        # Whether to render rollouts to screen
    parser.add_argument(
        "--keep_only_successful",
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
        default=5,
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
