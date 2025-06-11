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
import cv2 


RESULTS_DIR = "/store/real/maxjdu/repos/robotrainer/final_figure_generation"
# plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 12
plt.rcParams['ytick.labelsize'] = 12


control_file =  "CALVINABCD_CONTROL_block_ss1"

relevant_behaviors = ["red_lift", "blue_lift", "pink_lift"] #, "no_behavior"] #, "other"]

pretty_labels = [ "Red Block Move", "Blue Block Move", "Pink Block Move"]
num_seeds = 6

DUMMY = False # this is when you just want to generate a figure without computing everything 

if DUMMY: 
    gc_means = {k : 0.4 * np.random.random() for k in pretty_labels}
    gc_vars = {k : 0.01 for k in pretty_labels}
    planner_means = {k : 0.4 * np.random.random() for k in pretty_labels}
    planner_vars = {k : 0.01 for k in pretty_labels}
    our_method_means = {k : 0.4 * np.random.random() for k in pretty_labels}
    our_method_vars = {k : 0.01 for k in pretty_labels}
    ctrl_means = {k : 0.4 * np.random.random() for k in pretty_labels}
    ctrl_vars = {k : 0.01 for k in pretty_labels}
    posguide_means = {k : 0.4 * np.random.random() for k in pretty_labels}
    posguide_vars = {k : 0.01 for k in pretty_labels}

SCALING_FACTOR =  {"sliding_door" : 10,
            "drawer" : 10,
            "button" : 10,
            "switch" : 10,
            "lightbulb" : 0, # ignore the light because it's represented by the switch 
            "green_light" : 0,
            "red_block": 1,
            "blue_block" : 1,
            "pink_block" : 1}

def combined_std(std_list, mean_list):
    mean_of_means = mean(mean_list)
    squared_means = [(mu - mean_of_means)**2 + sig**2 for sig, mu in zip(std_list, mean_list)]
    return mean(squared_means)**0.5


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
                          "red_lift", "blue_lift", "pink_lift", "no_behavior"] #, "other"]
    behavior_dict = {k : 0 for k in relevant_behaviors}
    # adjustable_limits = {"sliding_door" : [0, 0.27], "drawer" : [0, 0.16], "switch" : [0, 0.09], "green_light" : [0, 1]}
    adjustable_limits = {"sliding_door" : [0, 0.27], "drawer" : [0, 0.16], "switch" : [0, 0.08], "green_light" : [0, 1]}
    half_thresholds = {k : (v[1] - v[0]) / 2 for k, v in adjustable_limits.items()}

    dataset = h5py.File(hdf5_name, 'r')
    data_grp = dataset["data"]
    steps_list = list()

    for demo in data_grp.keys():

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
                behavior_dict["red_lift"] += 1 
                something_happened = True 

            if np.linalg.norm(robot_pos - last_state["pink_block"]) < 0.1 and delta_state["pink_block"] > 0.001:
                behavior_dict["pink_lift"] += 1 
                something_happened = True 

            if np.linalg.norm(robot_pos - last_state["blue_block"]) < 0.1 and delta_state["blue_block"] > 0.001:
                behavior_dict["blue_lift"] += 1 
                something_happened = True 
            
            if something_happened: # this allows for multiple-counting 
                break 

            if delta_state["sliding_door"] > 0.05: 
                if first_state["sliding_door"] < last_state["sliding_door"]:
                    behavior_dict["door_left"] += 1 
                else:
                    behavior_dict["door_right"] += 1 
                something_happened = True 
                break 
            elif delta_state["drawer"] > 0.05: 
                if first_state["drawer"] > last_state["drawer"]:
                    behavior_dict["drawer_close"] += 1 
                else:
                    behavior_dict["drawer_open"] += 1 
                something_happened = True 
                break 
            elif delta_state["switch"] > 0.02:
                if first_state["switch"] > last_state["switch"]:
                    behavior_dict["switch_off"] += 1 
                else:
                    behavior_dict["switch_on"] += 1 
                something_happened = True 
                break 
            elif delta_state["green_light"] > 0.01:
                if first_state["green_light"] > last_state["green_light"]:
                    behavior_dict["button_off"] += 1 
                else:
                    behavior_dict["button_on"] += 1 
                something_happened = True 
                break 
        if not something_happened:
            behavior_dict["no_behavior"] += 1  # nothing happened 
        steps_list.append(step)
    
    total_sum = sum(behavior_dict.values())
    behavior_dict = {k : v / total_sum for k, v in behavior_dict.items()}
    full_report = {"Full distribution" : behavior_dict, "Average Length" : np.mean(steps_list)}

    if save_data:
        plt.bar(behavior_dict.keys(), behavior_dict.values())
        plt.xticks(rotation=90)
        plt.title("Behavior Distribution " + name)
        plt.tight_layout()
        plt.savefig(f"{RESULTS_DIR}/{name}_full_dist.png", transparent=True)
        plt.close()
        with open(f"{RESULTS_DIR}/{name}_dist.json", "w") as f:
            json.dump(full_report, f, indent = 2)
    
    return full_report




# TODO: try making this graph by grouping the articulated objects together 
if not DUMMY:
    our_method_means = {behavior : 0 for behavior in relevant_behaviors} 
    our_method_vars = {behavior : 0 for behavior in relevant_behaviors} 
    target_vals = {k : list() for k in relevant_behaviors}
    for behavior in relevant_behaviors:
        print("Ours: ", behavior)
        for seed in range(num_seeds):
            name = behavior + "_" + str(seed)
            record_file =  f"/store/real/maxjdu/repos/robotrainer/FINAL_EXPERIMENTS/ArticulatedObjects/{name}/{name}.hdf5"
            try:
                dist = generate_detailed_behavior_distribtuion(name, record_file, save_data = False)
                print(seed, " : ", dist["Full distribution"][behavior])
            except KeyboardInterrupt:
                exit() 
            except:
                print("Error with ", record_file)
                continue 
            target_vals[behavior].append(dist["Full distribution"][behavior])
        our_method_means[behavior] = np.mean(np.array(target_vals[behavior])) if len(target_vals[behavior]) > 0 else 0
        our_method_vars[behavior] = np.std(np.array(target_vals[behavior]))  / np.sqrt(num_seeds - 1) if len(target_vals[behavior]) > 0 else 0
        

    planner_means = {behavior : 0 for behavior in relevant_behaviors} 
    planner_vars = {behavior : 0 for behavior in relevant_behaviors} 
    target_vals = {k : list() for k in relevant_behaviors}
    for behavior in relevant_behaviors:
        print("Planner: ", behavior)
        for seed in range(num_seeds):
            name = behavior + "_planner_" + str(seed)
            record_file =  f"/store/real/maxjdu/repos/robotrainer/FINAL_EXPERIMENTS/ArticulatedObjects/{name}/{name}.hdf5"
            try:
                dist = generate_detailed_behavior_distribtuion(name, record_file, save_data = False)
                print(seed, " : ", dist["Full distribution"][behavior])
            except KeyboardInterrupt:
                exit() 
            except:
                print("Error with ", record_file)
                continue 
            target_vals[behavior].append(dist["Full distribution"][behavior])
        planner_means[behavior] = np.mean(np.array(target_vals[behavior])) if len(target_vals[behavior]) > 0 else 0
        planner_vars[behavior] = np.std(np.array(target_vals[behavior]))  / np.sqrt(num_seeds - 1) if len(target_vals[behavior]) > 0 else 0

    gc_means = {behavior : 0 for behavior in relevant_behaviors} 
    gc_vars = {behavior : 0 for behavior in relevant_behaviors} 
    target_vals = {k : list() for k in relevant_behaviors}
    for behavior in relevant_behaviors:
        print("Goal Conditioned: ", behavior)
        for seed in range(num_seeds):
            name = behavior + "_gc_" + str(seed)
            record_file =  f"/store/real/maxjdu/repos/robotrainer/FINAL_EXPERIMENTS/ArticulatedObjects/{name}/{name}.hdf5"
            try:
                dist = generate_detailed_behavior_distribtuion(name, record_file, save_data = False)
                print(seed, " : ", dist["Full distribution"][behavior])
            except KeyboardInterrupt:
                exit() 
            except:
                print("Error with ", record_file)
                continue 
            target_vals[behavior].append(dist["Full distribution"][behavior])
        gc_means[behavior] = np.mean(np.array(target_vals[behavior])) if len(target_vals[behavior]) > 0 else 0
        gc_vars[behavior] = np.std(np.array(target_vals[behavior]))  / np.sqrt(num_seeds - 1) if len(target_vals[behavior]) > 0 else 0



    ctrl_means = {behavior : 0 for behavior in relevant_behaviors} 
    ctrl_vars = {behavior : 0 for behavior in relevant_behaviors}
    target_vals = {k : list() for k in relevant_behaviors}

    for seed in range(num_seeds):
        name = "Control" + "_" + str(seed)
        record_file =  f"/store/real/maxjdu/repos/robotrainer/FINAL_EXPERIMENTS/ArticulatedObjects/{name}/{name}.hdf5"
        try:
            dist = generate_detailed_behavior_distribtuion(name, record_file, save_data = False)
            print(seed, " : ", dist["Full distribution"][behavior])
        except KeyboardInterrupt:
            exit() 
        except:
            print("Error with ", record_file)
            continue 
        for behavior in relevant_behaviors:
            target_vals[behavior].append(dist["Full distribution"][behavior])

    for behavior in relevant_behaviors:
        ctrl_means[behavior] = np.mean(np.array(target_vals[behavior])) if len(target_vals[behavior]) > 0 else 0
        ctrl_vars[behavior] = np.std(np.array(target_vals[behavior]))  / np.sqrt(num_seeds - 1) if len(target_vals[behavior]) > 0 else 0


# colors = ["#4357AD", "#48A9A6", "#E15A97", "#94CF26", "#FF9F1C"]
# colors = ["#414067", "#89C5EC", "#E37B3A", "#EFCD76", "#9397A4"] # ours guidance, ours sampling, goal conditioning, felix, baseline 
colors = ["#5753A3", "#89C5EC", "#E37B3A", "#FAAF6F", "#EFCD76"] # ours guidance, ours sampling, goal conditioning, felix, baseline 
shades_of_gray = ["#888888", "#BABABA", "#222222"]
mean = lambda x : sum(x) / len(x)

fig, ax2 = plt.subplots() # figsize=(8, 5)) #, gridspec_kw={'width_ratios': [4/7, 3/7]})
fig.set_size_inches(7.2, 3, forward=True)

width = 0.20  # Width of the bars
# x_left = np.array([0, 1, 2, 3, 4.25]) # a little extra distance for the average bar 
x_right = np.array([0, 1, 2, 3.125])

label_fontsize = 7
error_kw = {'elinewidth': 1, 'capthick': 1, 'ecolor': 'black'}
capsize = 2
padding = 3

ours_bar = ax2.bar(x_right - 1.5 * width, list(our_method_means.values()) + [mean(our_method_means.values())], width, capsize=capsize, error_kw=error_kw,
                    yerr = list(our_method_vars.values()) + [combined_std(our_method_vars.values(), our_method_means.values())], label="Ours-Guidance", color = colors[0])
ax2.bar_label(ours_bar, labels = [f"{val:.2f}" for val in list(our_method_means.values()) + [mean(our_method_means.values())]], padding=padding, fontsize=label_fontsize, fontname="serif")
planning_bar = ax2.bar(x_right - 0.5 * width, list(planner_means.values()) + [mean(planner_means.values())], width,  capsize=capsize, error_kw=error_kw,
                       yerr = list(planner_vars.values()) + [combined_std(planner_vars.values(), planner_means.values())], label="Ours-Sampling", color = colors[1])
ax2.bar_label(planning_bar, labels = [f"{val:.2f}" for val in list(planner_means.values()) + [mean(planner_means.values())]], padding=padding, fontsize=label_fontsize, fontname="serif")
gc_bar = ax2.bar(x_right + 0.5 * width, list(gc_means.values()) + [mean(gc_means.values())], width,  capsize=capsize, error_kw=error_kw,
                 yerr = list(gc_vars.values()) + [combined_std(gc_vars.values(), gc_means.values())], label="Goal Conditioning", color = colors[2])
ax2.bar_label(gc_bar, labels = [f"{val:.2f}" for val in list(gc_means.values()) + [mean(gc_means.values())]], padding=padding, fontsize=label_fontsize, fontname="serif")
baseline_bar = ax2.bar(x_right + 1.5 * width, list(ctrl_means.values()) + [mean(ctrl_means.values())], width,  capsize=capsize, error_kw=error_kw,
                       yerr = list(ctrl_vars.values()) + [combined_std(ctrl_vars.values(), ctrl_means.values())], label="Base Policy", color = colors[4])
ax2.bar_label(baseline_bar, labels = [f"{val:.2f}" for val in list(ctrl_means.values()) + [mean(ctrl_means.values())]], padding=padding, fontsize=label_fontsize, fontname="serif")

ax2.set_ylim(0, 0.4)
# ax2.legend()
# # ax.set_xlabel('Labels')
ax2.set_xticks(x_right)
ax2.set_xticklabels(pretty_labels + ["Average"], fontname = 'DejaVu Sans')# to include average
ax2.set_yticks([0.1, 0.2, 0.3, 0.4])

ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
# ax2.spines['bottom'].set_visible(False)

# ax2.spines['bottom'].set_position(('outward', 5))


# yticks2 = ax2.get_yticks()
# ax2.set_yticks(yticks2[1:])

plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/Exp2_cubes.pdf", transparent=True)
plt.close()

