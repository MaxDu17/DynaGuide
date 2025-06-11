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

TASK_NAME = "CalvinDD_validation_by_category_wcubes"

for behavior in relevant_behaviors:
    Path(f'../dataset/{TASK_NAME}').mkdir(parents=True, exist_ok=True)
    Path(f'../dataset/{TASK_NAME}/{behavior}_videos').mkdir(parents=True, exist_ok=True)
    data_writer = h5py.File(f'../dataset/{TASK_NAME}/{behavior}.hdf5', 'w')
    data_grp = data_writer.create_group("data")
    relevant_behavior_datasets[behavior] = data_grp 
    datawriter_list.append(data_writer)



Path(f'../dataset/{TASK_NAME}').mkdir(parents=True, exist_ok=True)
Path(f'../dataset/{TASK_NAME}/test_set_videos').mkdir(parents=True, exist_ok=True)
data_writer = h5py.File(f'../dataset/{TASK_NAME}/labeled_test_set.hdf5', 'w')
test_set_grp = data_writer.create_group("data")
test_set_count = 0
datawriter_list.append(data_writer)


dataset = h5py.File(ORIGINAL_DIR, 'r')

for demo in tqdm.tqdm(dataset["data"]):
    demo_grp = dataset["data"][demo]
    if "behavior" not in demo_grp.attrs:
        print("Problem with demo ", demo)
        continue 

    behavior = demo_grp.attrs["behavior"]
    if behavior not in relevant_behaviors:
        continue 
    # ep_grp = relevant_behavior_datasets[behavior].create_group("demo_{}".format(relevant_behaviors_count[behavior]))
    if random.random() > 0.3:
        video_writer = imageio.get_writer(f"../dataset/{TASK_NAME}/{behavior}_videos/{relevant_behaviors_count[behavior]}.gif")  # , fps=20)
        demo_grp.copy(demo_grp, relevant_behavior_datasets[behavior], "demo_{}".format(relevant_behaviors_count[behavior]))
        relevant_behaviors_count[behavior] += 1 
        for img in range(0, len(demo_grp["obs/third_person"]), 4):
            video_writer.append_data(demo_grp["obs/third_person"][img])
        video_writer.close()
    else:
        demo_grp.copy(demo_grp, test_set_grp, "demo_{}".format(test_set_count))
        test_set_grp["demo_{}".format(test_set_count)].attrs["behavior"] = behavior 
        relevant_behaviors_test_count[behavior] += 1 
        test_set_count += 1 

print(relevant_behaviors_count)
print(relevant_behaviors_test_count)
for data_writer in datawriter_list:
    data_writer.close()
dataset.close()

with open(f"../dataset/{TASK_NAME}/stats.json", "w") as f:
    json.dump(relevant_behaviors_count, f)

with open(f"../dataset/{TASK_NAME}/test_stats.json", "w") as f:
    json.dump(relevant_behaviors_test_count, f) 