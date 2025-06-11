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
#

# make a list of different behaviors and sorting 1/2 into these bins, and adding 1/2 into the "test"
relevant_behaviors = ["button_on", "button_off", "switch_on", "switch_off", "drawer_open", "drawer_close", 
                      "door_left", "door_right", "red_lift", "blue_lift", "pink_lift", "other"]

relevant_behaviors_count = {k : 0 for k in relevant_behaviors}
relevant_behaviors_test_count = {k : 0 for k in relevant_behaviors}
relevant_behavior_datasets = {} 
relevant_behavior_video_writers = {} 
datawriter_list = list() # for housekeeping 

# def index_to_label(index):
#     if index < 6:
#         element_dict = {0 : "sliding_door",
#                         1 : "drawer",
#                         2: "button",
#                         3: "switch",
#                         4: "lightbulb",
#                         5: "green_light"}
#         return element_dict[index]
#     elif index < 12:
#         return "red_block"
#     elif index < 15:
#         return "blue_block"
#     return "pink_block"

# def segment_states(state_obs):
#     return {"sliding_door" : state_obs[0],
#             "drawer" : state_obs[1],
#             "button" : state_obs[2],
#             "switch" : state_obs[3],
#             "lightbulb" : state_obs[4],
#             "green_light" : state_obs[5],
#             "red_block": state_obs[6:9], # we are ignoring rotations 
#             "blue_block" : state_obs[12 : 15],
#             "pink_block" : state_obs[18 :21]}, {
#             "red_rot": R.from_euler("XYZ", state_obs[9:12]).as_matrix(), # we are ignoring rotations 
#             "blue_rot" : R.from_euler("XYZ", state_obs[15:18]).as_matrix(),
#             "pink_rot" : R.from_euler("XYZ", state_obs[21:]).as_matrix()
#             }

# THIS CODE TAKES IN HDF5 AND SPLITS IT INTO
# - untouched validation (50%)
# - segmented target behavior (50%)


# ORIGINAL_DIR = "/store/real/maxjdu/repos/robotrainer/dataset/CalvinDD_validation_all/data.hdf5"
ORIGINAL_DIR = "/store/real/maxjdu/repos/robotrainer/dataset/CalvinDD_validation_better_seg_all/data.hdf5" # this is where the segmented dataset lies 

dataset = h5py.File(ORIGINAL_DIR, 'r')
robot_state_list = list() 

behavior_distribution = {}
total = 0 
# for demo in tqdm.tqdm(dataset["data"]):
#     demo_grp = dataset["data"][demo]
#     if "behavior" not in demo_grp.attrs:
#         print("Problem with demo ", demo)
#         continue 
#     if demo_grp.attrs["behavior"] not in behavior_distribution:
#         behavior_distribution[demo_grp.attrs["behavior"]] = 0

#     behavior_distribution[demo_grp.attrs["behavior"]] += 1 
#     total += 1 
# behavior_distribution = {k : v / total for k, v in behavior_distribution.items()}


for demo in tqdm.tqdm(dataset["data"]):
    demo_grp = dataset["data"][demo]
    if "behavior" not in demo_grp.attrs:
        print("Problem with demo ", demo)
        continue 
    if demo_grp.attrs["behavior"] == "other": # or demo_grp.attrs["behavior"] == "button_on" or demo_grp.attrs["behavior"] == "button_on":
        continue 
    # table': [[0.0, -0.15, 0.46], [0.35, -0.03, 0.46]],    
    #                 valid_surfaces.append([[-0.32, -0.15, 0.46], [-0.16, 0.03, 0.46]]) # this is the area in front of left slider 

    # initial_robot_state = demo_grp["obs"]["proprio"][0]
    # midpoint = demo_grp["obs"]["proprio"].shape[0] // 5 # finding a point a little after the initial state 
    if demo_grp["obs"]["proprio"].shape[0] < 8: 
        continue 
    initial_robot_state = demo_grp["obs"]["proprio"][8]
    # if initial_robot_state[6] < 0.065: # robot should have open gripper 
    if initial_robot_state[6] < 0.075: # robot should have open gripper 
        continue 
    if initial_robot_state[2] < 0.53: # robot should be high enough on the table 
        continue 
    
    if initial_robot_state[0] < 0 or initial_robot_state[0] > 0.35: # robot should be between the areas 
        continue 
    

    if demo_grp.attrs["behavior"] not in behavior_distribution:
        behavior_distribution[demo_grp.attrs["behavior"]] = 0
    behavior_distribution[demo_grp.attrs["behavior"]] += 1 
    total += 1 


 
    # import ipdb 
    # ipdb.set_trace()

    robot_state_list.append(initial_robot_state.tolist())

print(len(robot_state_list))
behavior_distribution = {k : v / total for k, v in behavior_distribution.items()}
print(behavior_distribution)
with open(f"/store/real/maxjdu/repos/robotrainer/dataset/initial_calvin_robot_states_right_side_midpoint.json", "w") as f:
    json.dump({"robot_states": robot_state_list}, f)