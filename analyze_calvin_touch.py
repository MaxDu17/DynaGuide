import matplotlib.pyplot as plt 
import h5py
import numpy as np 
import torch 
import imageio.v2 as iio
import imageio 
import pickle 
import tqdm 
import json 
from scipy.spatial.transform import Rotation as R

RESULTS_DIR = "/store/real/maxjdu/repos/robotrainer/analysis"
# RESULTS_DIR = "/store/real/maxjdu/repos/robotrainer/paper_figs"
import cv2 


SCALING_FACTOR =  {"sliding_door" : 10,
            "drawer" : 10,
            "button" : 10,
            "switch" : 10,
            "lightbulb" : 0, # ignore the light because it's represented by the switch 
            "green_light" : 0,
            "red_block": 1,
            "blue_block" : 1,
            "pink_block" : 1}

def plot_double_bar(values1, values2, labels, name, label_1, label_2):
    x = np.arange(len(labels))  # Label locations
    width = 0.35  # Width of the bars
    
    fig, ax = plt.subplots()
    bars1 = ax.bar(x - width/2, values1, width, label=label_1)
    bars2 = ax.bar(x + width/2, values2, width, label=label_2)
    
    # ax.set_xlabel('Labels')
    ax.set_ylabel('Success Rate')
    ax.set_title('Double Bar Plot')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation = 45)
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/{name}_CONTROL_COMPARISON.pdf")
    plt.close()


def segment_state_batched(state_obs):
    return {"sliding_door" : state_obs[:, 0],
            "drawer" : state_obs[:, 1],
            "button" : state_obs[:, 2],
            "switch" : state_obs[:, 3],
            "lightbulb" : state_obs[:, 4],
            "green_light" : state_obs[:, 5],
            "red_block": state_obs[:, 6:9], # we are ignoring rotations 
            "blue_block" : state_obs[:, 12 : 15],
            "pink_block" : state_obs[:, 18 :21]}, {
            "red_rot": R.from_euler("XYZ", state_obs[:, 9:12]).as_matrix(), # we are ignoring rotations 
            "blue_rot" : R.from_euler("XYZ", state_obs[:, 15:18]).as_matrix(),
            "pink_rot" : R.from_euler("XYZ", state_obs[:, 21:]).as_matrix()
            }

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

def generate_detailed_behavior_distribtuion(name, hdf5_name, save_data = True):
    relevant_behaviors = ["button_on", "button_off", "switch_on", "switch_off", "drawer_open", "drawer_close", "door_left", "door_right", 
                          "red_displace", "blue_displace", "pink_displace", "no_behavior"] #, "other"]
    behavior_dict = {k : 0 for k in relevant_behaviors}
    # adjustable_limits = {"sliding_door" : [0, 0.27], "drawer" : [0, 0.16], "switch" : [0, 0.09], "green_light" : [0, 1]}
    adjustable_limits = {"sliding_door" : [0, 0.27], "drawer" : [0, 0.16], "switch" : [0, 0.08], "green_light" : [0, 1]}
    half_thresholds = {k : (v[1] - v[0]) / 2 for k, v in adjustable_limits.items()}

    dataset = h5py.File(hdf5_name, 'r')
    data_grp = dataset["data"]
    steps_list = list()

    demo_labels = {} 
    for demo in tqdm.tqdm(data_grp.keys()):

        positions = data_grp[demo]["obs"]["proprio"]#[:, -1]
        env_states = data_grp[demo]["obs"]["states"]#[:, -1]
        if len(env_states.shape) == 3:
            env_states = env_states[:, -1]
        if len(positions.shape) == 3:
            positions=  positions[:, -1]
        
        first_state, first_rot = segment_states(env_states[0])
        something_happened = False 
        # this searches for the first small motion, which is useful for behaviors that set and reset 
        for step in range(env_states.shape[0]):
            last_state, last_rot = segment_states(env_states[step])

            delta_state = {k : np.linalg.norm(last_state[k] - first_state[k]) for k in last_state.keys()}
            robot_pos = positions[step, 0:3]
            # TODO: if multiple are touched at the same time 
            if np.linalg.norm(robot_pos - last_state["red_block"]) < 0.1 and delta_state["red_block"] > 0.001:
                behavior_dict["red_displace"] += 1 
                something_happened = True 
                demo_labels[demo] = "red_displace"

            if np.linalg.norm(robot_pos - last_state["pink_block"]) < 0.1 and delta_state["pink_block"] > 0.001:
                behavior_dict["pink_displace"] += 1 
                something_happened = True 
                demo_labels[demo] = "pink_displace"


            if np.linalg.norm(robot_pos - last_state["blue_block"]) < 0.1 and delta_state["blue_block"] > 0.001:
                behavior_dict["blue_displace"] += 1 
                something_happened = True 
                demo_labels[demo] = "blue_displace"

            
            if something_happened: # this allows for multiple-counting 
                break 

            # if delta_state["red_block"] > 0.02:
            #     behavior_dict["red_displace"] += 1 
            #     something_happened = True 
            #     break 
            # if delta_state["blue_block"] > 0.02:
            #     behavior_dict["blue_displace"] += 1 
            #     something_happened = True 
            #     break 
            # if delta_state["pink_block"] > 0.02:
            #     behavior_dict["pink_displace"] += 1 
            #     something_happened = True 
            #     break 
            # print(delta_state["switch"])

            if delta_state["sliding_door"] > 0.05: 
                if first_state["sliding_door"] < last_state["sliding_door"]:
                    behavior_dict["door_left"] += 1 
                    demo_labels[demo] = "door_left"

                else:
                    behavior_dict["door_right"] += 1 
                    demo_labels[demo] = "door_right"

                something_happened = True
 
                break 
            elif delta_state["drawer"] > 0.05: 
                if first_state["drawer"] > last_state["drawer"]:
                    behavior_dict["drawer_close"] += 1 
                    demo_labels[demo] = "drawer_close"

                else:
                    behavior_dict["drawer_open"] += 1
                    demo_labels[demo] = "drawer_open"
 
                something_happened = True 

                break 
            elif delta_state["switch"] > 0.02:
                if first_state["switch"] > last_state["switch"]:
                    behavior_dict["switch_off"] += 1 
                    demo_labels[demo] = "switch_off"
                else:
                    behavior_dict["switch_on"] += 1 
                    demo_labels[demo] = "switch_on"

                something_happened = True 
                break 
            elif delta_state["green_light"] > 0.01:
                if first_state["green_light"] > last_state["green_light"]:
                    behavior_dict["button_off"] += 1 
                    demo_labels[demo] = "button_off"
                else:
                    behavior_dict["button_on"] += 1 
                    demo_labels[demo] = "button_off"

                something_happened = True 
                break 
        if not something_happened:
            behavior_dict["no_behavior"] += 1  # nothing happened 
            demo_labels[demo] = "no_behavior"
        steps_list.append(step)
    total_sum = sum(behavior_dict.values())
    behavior_dict = {k : v / total_sum for k, v in behavior_dict.items()}
    # print("here")
    full_report = {"Full distribution" : behavior_dict, "Average Length" : np.mean(steps_list), "demo_labels" : demo_labels}

    if save_data:
        # print("here")

        plt.bar(behavior_dict.keys(), behavior_dict.values())
        plt.xticks(rotation=90)
        plt.title("Behavior Distribution " + name)
        plt.tight_layout()
        plt.savefig(f"{RESULTS_DIR}/{name}_full_dist.png")
        plt.close()
        # print("here")



        with open(f"{RESULTS_DIR}/{name}_dist.json", "w") as f:
            json.dump(full_report, f, indent = 2)
    
    return full_report


# this is absolutely terrible forgive me 
def _generate_behavior_distribution(name, hdf5_name):
    relevant_behaviors = ["button_on", "button_off", "switch_on", "switch_off", "drawer_open", "drawer_close", "door_left", "door_right", "other"]
    behavior_dict = {k : 0 for k in relevant_behaviors}
    first_behavior_dict = {k : 0 for k in relevant_behaviors}
    dataset = h5py.File(hdf5_name, 'r')
    data_grp = dataset["data"]
    ACTIVE_EPSILON = 0.001 # more sensitive to initial touching 
    # RELEASE_EPSILON = 0.00001 # sensitivity to not moving 
    RELEASE_EPSILON = 0.001 # sensitivity to not moving 
    MIN_LENGTH = 30

    for demo in tqdm.tqdm(data_grp.keys()):
        # TODO: figure out if the robot is close! 
        waiting = True
        moving = False # the two states needed for segmentation 
        to_segment = False
        move_count = 0
        positions = data_grp[demo]["obs"]["proprio"][:, -1]
        env_states = data_grp[demo]["obs"]["states"][:, -1]
        last_state, last_rot = segment_states(env_states[0])
        step_count = 0 
        is_first = True 

        for step in range(env_states.shape[0]):
            current_step = env_states[step]

            current_state, current_rot = segment_states(current_step)
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
            step_count += 1 
            if moving:
                move_count += 1 
            if waiting and active_touching_object: # you've touched an object for hte first time 
                start_interaction_state = current_state 
                moving = True 
                waiting = False
            elif moving and not release_touching: # you've stopped touching something 
                waiting = True 
                moving = False 
                to_segment = True # this is when you segment 
                if move_count < 10:
                    # print("too short of a motion; stitching")
                    to_segment = False # basically ignore this touch 

                move_count = 0
                if step_count < MIN_LENGTH:
                    # print("too short; stitching")
                    to_segment = False # basically ignore this touch 
                
                if to_segment:
                    end_interaction_state = current_state 
                    if end_interaction_state["green_light"] > 0.8 and start_interaction_state["green_light"] < 0.2:
                        behavior = "button_on"

                    elif end_interaction_state["green_light"] < 0.2 and start_interaction_state["green_light"] > 0.8: 
                        behavior = "button_off"

                    elif end_interaction_state["switch"] < 0.01 and start_interaction_state["switch"] > 0.05:
                        behavior = "switch_off"
                    
                    elif end_interaction_state["switch"] > 0.05 and start_interaction_state["switch"] < 0.01: # used to be 0.07 
                        behavior = "switch_on"

                    elif end_interaction_state["drawer"] < 0.05 and start_interaction_state["drawer"] > 0.10:
                        behavior = "drawer_close"
                    
                    elif end_interaction_state["drawer"] > 0.10 and start_interaction_state["drawer"] < 0.05:
                        behavior = "drawer_open"
                    
                    elif end_interaction_state["sliding_door"] < 0.05 and start_interaction_state["sliding_door"] > 0.25:
                        behavior = "door_right"
                    
                    elif end_interaction_state["sliding_door"] > 0.25 and start_interaction_state["sliding_door"] < 0.05:
                        behavior = "door_left"
                    
                    else:
                        behavior = "other"
                    print("detected behavior ", behavior)
                    behavior_dict[behavior] += 1
                    step_count = 0 # reset steps since reset 

                    if is_first and behavior != "other":
                        print("here!")
                        first_behavior_dict[behavior] += 1 
                        is_first = False 

            last_state = current_state
            last_rot = current_rot

    plt.bar(behavior_dict.keys(), behavior_dict.values())
    plt.xticks(rotation=90)
    plt.title("Full Distribution " + name)
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/{name}_full_dist.png")
    plt.close()

    plt.bar(first_behavior_dict.keys(), first_behavior_dict.values())
    plt.xticks(rotation=90)
    plt.title("First Distribution " + name)
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/{name}_first_dist.png")

    plt.close()
    full_report = {"Full distribution" : behavior_dict, "First distribution" : first_behavior_dict}
    with open(f"{RESULTS_DIR}/{name}_dist.json", "w") as f:
        json.dump(full_report, f, indent = 2)
    
    return full_report

def calculate_success_rates(name, hdf5_name, target, bounds):
    # TODO: make this more general to lists of targets and bounds 
    # plots the scatter of all of the cubes in the trial and highlights the pressed cube in red (others are in gray)
    # 
    dataset = h5py.File(hdf5_name, 'r')
    data_grp = dataset["data"]
    total = len(data_grp.keys())
    x_list = list()
    y_list = list()
    color_list = list()
    successes = 0 
    target_count = 0
    near_target = 0 


    for demo in tqdm.tqdm(data_grp.keys()):
        # TODO: figure out if the robot is close! 
        positions = data_grp[demo]["obs"]["proprio"][:, -1]
        env_states = data_grp[demo]["obs"]["states"][:, -1]
        segmented_states, cubes = segment_state_batched(env_states)
        target_value = segmented_states[target]
        if np.any(np.logical_and(np.array(target_value) < bounds[1], np.array(target_value) > bounds[0])):
            successes += 1 

    with open(f"{RESULTS_DIR}/{name}.json", "w") as f:
        json.dump({"hdf5_name" : hdf5_name, "target": target, "bounds" : bounds, 
                   "target_success_rate": successes / len(data_grp.keys())}, f, indent = 2)
    plt.close()
    return successes / len(data_grp.keys())

names = [
   "Calvin_PaperPullfigVisual_BasePolicy2"
]


for name in names:
    hdf5_name = f"/store/real/maxjdu/repos/robotrainer/results/outputs/{name}/{name}.hdf5"
    # hdf5_name = f"/store/real/maxjdu/repos/robotrainer/FINAL_EXPERIMENTS/ArticulatedObjects/{name}/{name}.hdf5"
    # target = {"switch" : [0.01, 0.1]} # acceptable tolerance 
    # succ = calculate_success_rates(name, hdf5_name, "switch", [0.01, 0.1])
    # dist = generate_behavior_distribution(name, hdf5_name)
    # try:
    dist = generate_detailed_behavior_distribtuion(name, hdf5_name)
    print(name, dist)
    sorted_keys = sorted(dist["demo_labels"].keys(), key = lambda x : int(x.split("_")[-1]))
    behavior_list = [dist["demo_labels"][k] for k in sorted_keys]

    with open(f"traj_labels_{name}.json", "w") as f:
        json.dump(behavior_list, f)
        
    
    # except:
    #     print(f"Skipping {hdf5_name}")
    # succ = calculate_success_rates(name, hdf5_name, "green_light", [0.9, 1.1]) # should only accept on
    # print(name, succ)

exit()
# plot success rates for each behavior TODO

names = [
    "SwitchOnDynaGuide"
]


relevant_behaviors = ["button_on", "button_off", "switch_on", "switch_off", "drawer_open", "drawer_close", "door_left", "door_right",  
                           "red_displace", "blue_displace", "pink_displace"] #, "no_behavior"] #, "other"]

our_values = list()
mean_length = list() 
for i, name in enumerate(names):
    hdf5_name = f"path_to_experiments/{name}/{name}.hdf5"
    dist = generate_detailed_behavior_distribtuion(name, hdf5_name, save_data = False)
    our_values.append(dist["Full distribution"][relevant_behaviors[i]])
    print(name, dist)
    mean_length.append(dist["Average Length"])