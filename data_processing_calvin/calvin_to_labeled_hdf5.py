import torch
import h5py
import tqdm
import json
import numpy as np
import imageio
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.file_utils as FileUtils
from pathlib import Path
import pickle
import os
import matplotlib.pyplot as plt 
from scipy.spatial.transform import Rotation as R
import random 
"""

"""


# TASK_NAME = "CalvinDD_door_left_80_percent"
# TASK_NAME = "CalvinDD_turn_on_green_light"
# TASK_NAME = "CalvinDD_open_drawer"
# TASK_NAME = "CalvinDD_door_move_right"

# ORIGINAL_DIR = "/store/real/maxjdu/repos/calvin/dataset/task_D_D/validation/"

# TASK_NAME = "CalvinDD_betterseg"
TASK_NAME = "CalvinABCD_betterseg"
# ORIGINAL_DIR = "/store/real/maxjdu/repos/calvin/dataset/task_D_D/training/"
ORIGINAL_DIR = "/store/real/maxjdu/repos/calvin/dataset/task_ABCD_D/training/"

SCALING_FACTOR =  {"sliding_door" : 10,
            "drawer" : 10,
            "button" : 10,
            "switch" : 10,
            "lightbulb" : 0, # ignore the light because it's represented by the switch 
            "green_light" : 0,
            "red_block": 1,
            "blue_block" : 1,
            "pink_block" : 1}

# switch: goes from 0 to 0.08 ish

def index_to_label(index):
    if index < 6:
        element_dict = {0 : "sliding_door",
                        1 : "drawer",
                        2: "button",
                        3: "switch",
                        4: "lightbulb",
                        5: "green_light"}
        return element_dict[index]
    elif index < 12:
        return "red_block"
    elif index < 15:
        return "blue_block"
    return "pink_block"

def segment_states(state_obs):
    return {"sliding_door" : state_obs[0],
            "drawer" : state_obs[1],
            "button" : state_obs[2],
            "switch" : state_obs[3],
            "lightbulb" : state_obs[4],
            "green_light" : state_obs[5],
            "red_block": state_obs[6:9], # we are ignoring rotations 
            "blue_block" : state_obs[12 : 15],
            "pink_block" : state_obs[18 :21]}, {
            "red_rot": R.from_euler("XYZ", state_obs[9:12]).as_matrix(), # we are ignoring rotations 
            "blue_rot" : R.from_euler("XYZ", state_obs[15:18]).as_matrix(),
            "pink_rot" : R.from_euler("XYZ", state_obs[21:]).as_matrix()
            }

env_meta = json.load(open("../configs/calvin.json", 'r'))

Path(f'../dataset/{TASK_NAME}').mkdir(parents=True, exist_ok=True)
Path(f'../dataset/{TASK_NAME}/videos').mkdir(parents=True, exist_ok=True)
data_writer = h5py.File(f'../dataset/{TASK_NAME}/data.hdf5', 'w')

data_grp = data_writer.create_group("data")
data_grp.attrs["env_args"] =  json.dumps(env_meta)

ep_count = 0
total_samples = 0

video_skip = 20 #00 #250 # 250 #250 #50 #used to be 10

count = 0 
mag_list = list() 
listed_steps = os.listdir(ORIGINAL_DIR)
listed_steps = [x for x in listed_steps if ".npz" in x]
listed_steps.sort(key = lambda x: int(x.split(".")[0].split("_")[-1]))


waiting = True
moving = False # the two states needed for segmentation 
to_segment = False
move_count = 0
last_state, last_rot = segment_states(np.load(ORIGINAL_DIR + listed_steps[0])["scene_obs"])
ACTIVE_EPSILON = 0.001 # more sensitive to initial touching 
# RELEASE_EPSILON = 0.00001 # sensitivity to not moving 
RELEASE_EPSILON = 0.001 # sensitivity to not moving 
MIN_LENGTH = 16 #30

action_list = list()
eye_in_hand_list = list()
third_person_list = list()
state_list = list()
proprio_list = list() 

ep_data_grp = data_grp.create_group("demo_{}".format(ep_count))

behavior = None 
start_interaction_state = None 
end_interaction_state = None 

inclusion_count = 0 
for step_file in tqdm.tqdm(listed_steps): #[0:10]
    if ".npz" not in step_file:
        continue 
    try:
        data = np.load(ORIGINAL_DIR + step_file)
    except KeyboardInterrupt:
        print("quitting")
        quit()
    except:
        print("Problem with ", ORIGINAL_DIR + folder + "/" + step_file)
        continue
    # rewards = [step[0].reward for step in timesteps]
    # rewards[0] = 0 # remove none conditions
    # mag_list.append(np.linalg.norm(data["rel_actions"][0:3]))

    current_state, current_rot = segment_states(data["scene_obs"])
    if start_interaction_state is None:
        start_interaction_state = current_state 

    # print(data["robot_obs"][-1])
    delta_state = {k : SCALING_FACTOR[k] * np.linalg.norm(current_state[k] - last_state[k]).item() for k in current_state.keys()}
    delta_rot = {k : np.linalg.norm(R.from_matrix(current_rot[k] @ np.linalg.inv(last_rot[k])).as_euler("XYZ")).item() for k in current_rot.keys()} # sees how much we are rotating 
    # print(delta_rot)
    # to_segment = delta_state["lightbulb"] > 0.1 or delta_state["green_light"] > 0.1 # if there is a change in light, you should segment 
    to_segment = False
    active_touching = np.any(np.array(list(delta_state.values())) > ACTIVE_EPSILON) or np.any(np.array(list(delta_rot.values())) > ACTIVE_EPSILON)
    active_touching_object = np.where(np.array(list(delta_state.values())) > ACTIVE_EPSILON)
    active_touching_rotation = np.where(np.array(list(delta_rot.values())) > ACTIVE_EPSILON)
    release_touching = np.any(np.array(list(delta_state.values())) > RELEASE_EPSILON) or np.any(np.array(list(delta_rot.values())) > RELEASE_EPSILON)
    # print(np.linalg.norm(current_state))
    # print(delta_state)
    if moving:
        move_count += 1 
    if waiting and active_touching: # you've touched an object and now we are moving something 
        moving = True 
        waiting = False
        if delta_state["red_block"] > ACTIVE_EPSILON or delta_state["blue_block"] > ACTIVE_EPSILON or delta_state["pink_block"] > ACTIVE_EPSILON:
            eef_pos = data["robot_obs"][0:3]
            # this checks if the robot is close to the cube, so we don't false trigger on drawer 
            if np.linalg.norm(eef_pos - current_state["red_block"]) < 0.05 or np.linalg.norm(eef_pos - current_state["blue_block"]) < 0.05 \
                  or np.linalg.norm(eef_pos - current_state["pink_block"]) < 0.05:
                print("robot close!")
                to_segment = True 
            else:
                print("robot far!")

    elif moving and not release_touching: # you've stopped touching something, meaning that you should segment
        waiting = True 
        moving = False 
        to_segment = True # this is when you segment 
        if move_count < 10:
            # print("too short of a motion; stitching")
            to_segment = False # basically ignore this touch 

        move_count = 0
        if len(action_list) < MIN_LENGTH:
            # print("too short; stitching")
            to_segment = False # basically ignore this touch 
        
        if current_state["green_light"] != start_interaction_state["green_light"]: # always segment light turning color 
            to_segment = True 

    if to_segment:
        end_interaction_state = current_state 
        if end_interaction_state["green_light"] > 0.8 and start_interaction_state["green_light"] < 0.2:
            behavior = "button_on"

        elif end_interaction_state["green_light"] < 0.2 and start_interaction_state["green_light"] > 0.8: 
            behavior = "button_off"

        elif end_interaction_state["switch"] < 0.01 and start_interaction_state["switch"] > 0.07:
            behavior = "switch_off"
        
        elif end_interaction_state["switch"] > 0.07 and start_interaction_state["switch"] < 0.01:
            behavior = "switch_on"

        elif end_interaction_state["drawer"] < 0.05 and start_interaction_state["drawer"] > 0.10:
            behavior = "drawer_close"
        
        elif end_interaction_state["drawer"] > 0.10 and start_interaction_state["drawer"] < 0.05:
            behavior = "drawer_open"
        
        elif end_interaction_state["sliding_door"] < 0.05 and start_interaction_state["sliding_door"] > 0.25:
            behavior = "door_right"
        
        elif end_interaction_state["sliding_door"] > 0.25 and start_interaction_state["sliding_door"] < 0.05:
            behavior = "door_left"
        
        elif np.linalg.norm(end_interaction_state["red_block"] - start_interaction_state["red_block"] > 0.01):
            behavior = "red_displace"
        
        elif np.linalg.norm(end_interaction_state["blue_block"] - start_interaction_state["blue_block"] > 0.01):
            behavior = "blue_displace"

        elif np.linalg.norm(end_interaction_state["pink_block"] - start_interaction_state["pink_block"] > 0.01):
            behavior = "pink_displace"
        
        elif moving and delta_state["red_block"] > ACTIVE_EPSILON: # special case of touching 
            behavior = "red_lift"
        
        elif moving and delta_state["pink_block"] > ACTIVE_EPSILON:
            behavior = "pink_lift"
        
        elif moving and delta_state["blue_block"] > ACTIVE_EPSILON:
            behavior = "blue_lift"
        
        else:
            behavior = "other"
        
        start_interaction_state = current_state 
            

    if np.max(np.abs(data["rel_actions"]) > 1.0):
        print("clipping!")
    clipped_rel_actions = np.clip(data["rel_actions"], -1, 1)
    action_list.append(clipped_rel_actions)
    eye_in_hand_list.append(data["rgb_gripper"])
    third_person_list.append(data["rgb_static"])
    state_list.append(data["scene_obs"])
    proprio_list.append(data["robot_obs"])

    
    save_probability = random.random() > 0 # 0.9 #0.9 # THIS IS FOR VALIDATION

    # if behavior == "door_left": # for reduced data experiments 
    #     save_probability = random.random() > 0.20
    #     if save_probability:
    #         inclusion_count += 1 

    if to_segment and save_probability:
        # save everything to the current data grp, created new one, segment videos! 
        action = np.stack(action_list, axis = 0)
        eye_in_hand = np.stack(eye_in_hand_list, axis = 0)
        third_person = np.stack(third_person_list, axis = 0)
        states = np.stack(state_list, axis = 0)
        proprio = np.stack(proprio_list, axis = 0)
        ep_data_grp.attrs["behavior"] = behavior if behavior is not None else "other"
        ep_data_grp.create_dataset("actions", data=action)
        ep_data_grp.create_dataset("obs/eye_in_hand", data=eye_in_hand)
        ep_data_grp.create_dataset("obs/third_person", data=third_person)
        ep_data_grp.create_dataset("obs/proprio", data=proprio)
        ep_data_grp.create_dataset("obs/states", data=states)
        
        rewards = np.zeros(len(action_list))
        ep_data_grp.create_dataset("rewards", data=rewards)
        dones = np.zeros(len(action_list))
        dones[-1] = 1 
        ep_data_grp.create_dataset("dones", data= dones)
        print(ep_count)
        if ep_count % video_skip == 0:
            behavior_label = behavior if behavior is not None else "other"
            video_writer = imageio.get_writer(f"../dataset/{TASK_NAME}/videos/{ep_count}_{behavior_label}.gif")  # , fps=20)

            for img in range(0, len(third_person_list), 4):
                video_writer.append_data(third_person_list[img])
            video_writer.close()

        ep_data_grp.attrs["num_samples"] = len(action_list) #.shape[0]
        total_samples += ep_data_grp.attrs["num_samples"]

        action_list.clear()
        eye_in_hand_list.clear()
        third_person_list.clear()
        state_list.clear()
        proprio_list.clear() 
        behavior = None 
        # start_interaction_state = end_interaction_state = {}

        ep_count += 1 
        ep_data_grp = data_grp.create_group("demo_{}".format(ep_count))

    elif to_segment: # gets called when we want to clear 
        print("random reject!")
        action_list.clear()
        eye_in_hand_list.clear()
        third_person_list.clear()
        state_list.clear()
        proprio_list.clear() 
        # start_interaction_state = end_interaction_state = {}
        behavior = None 

    last_state = current_state
    last_rot = current_rot
    to_segment = False 


# SO I DON'T HAVE A BLANK FINAL DEMONSTRATION
action = np.stack(action_list, axis = 0)
eye_in_hand = np.stack(eye_in_hand_list, axis = 0)
third_person = np.stack(third_person_list, axis = 0)
states = np.stack(state_list, axis = 0)
proprio = np.stack(proprio_list, axis = 0)
ep_data_grp.create_dataset("actions", data=action)
ep_data_grp.create_dataset("obs/eye_in_hand", data=eye_in_hand)
ep_data_grp.create_dataset("obs/third_person", data=third_person)
ep_data_grp.create_dataset("obs/proprio", data=proprio)
ep_data_grp.create_dataset("obs/states", data=states)

rewards = np.zeros(len(action_list))
ep_data_grp.create_dataset("rewards", data=rewards)
dones = np.zeros(len(action_list))
dones[-1] = 1 
ep_data_grp.create_dataset("dones", data= dones)
print(ep_count)
if ep_count % video_skip == 0:
    video_writer = imageio.get_writer(f"../dataset/{TASK_NAME}/videos/{ep_count}.gif")  # , fps=20)

    for img in range(0, len(third_person_list), 4):
        video_writer.append_data(third_person_list[img])
    video_writer.close()

ep_data_grp.attrs["num_samples"] = len(action_list) #.shape[0]
total_samples += ep_data_grp.attrs["num_samples"]


data_grp.attrs["total"] = total_samples
print(total_samples)
print(ep_count)
data_writer.close()
print("Inclusion of target: ", inclusion_count)
