import h5py 
import numpy as np 
import matplotlib.pyplot as plt 
import tqdm 
import os 
from pathlib import Path
import json 

file_list = [
    "/store/real/maxjdu/repos/robotrainer/dataset/CalvinDD_validation_by_category_wcubes/button_on_20.hdf5",
    "/store/real/maxjdu/repos/robotrainer/dataset/CalvinDD_validation_by_category_wcubes/button_off_20.hdf5",
    "/store/real/maxjdu/repos/robotrainer/dataset/CalvinDD_validation_by_category_wcubes/door_left_20.hdf5",
    "/store/real/maxjdu/repos/robotrainer/dataset/CalvinDD_validation_by_category_wcubes/door_right_20.hdf5",
    "/store/real/maxjdu/repos/robotrainer/dataset/CalvinDD_validation_by_category_wcubes/switch_on_20.hdf5",
    "/store/real/maxjdu/repos/robotrainer/dataset/CalvinDD_validation_by_category_wcubes/switch_off_20.hdf5",
    "/store/real/maxjdu/repos/robotrainer/dataset/CalvinDD_validation_by_category_wcubes/drawer_open_20.hdf5",
    "/store/real/maxjdu/repos/robotrainer/dataset/CalvinDD_validation_by_category_wcubes/drawer_close_20.hdf5"
]

results_dict = {} 

for file in file_list:
    dataset = h5py.File(file, 'r')

    avg_goal_loc = list()
    for demo in dataset["data"]:
        demo_grp = dataset["data"][demo]
        end_loc = demo_grp["obs"]["proprio"][-1][0:3]
        avg_goal_loc.append(end_loc)
    avg_goal = np.mean(np.stack(avg_goal_loc, axis = 0), axis = 0)
    results_dict[file] = avg_goal.tolist() 
    print(file, avg_goal)

with open("/store/real/maxjdu/repos/robotrainer/dataset/CalvinDD_validation_by_category_wcubes/end_loc.json", "w") as f:
    json.dump(results_dict, f, indent = 2)
