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

RESULTS_DIR = "/store/real/maxjdu/repos/robotrainer/final_figure_generation"

relevant_atomic_behaviors = ["button_on", "button_off", "switch_on", "switch_off", "drawer_open", "drawer_close", "door_left", "door_right"] #, "no_behavior"] #, "other"]
condensed_atomic_behaviors = ["button", "switch", "drawer", "door", "blocks", "no_behavior"] 

behaviors_list = ["button_on_switch_on", "door_left_button_on", "switch_on_door_left_button_on", "drawer_dont_open", "door_dont_right", "drawer_dont_open_door_dont_right"]
pretty_behaviors_list = ["Button \n OR Switch", "Door \n OR Button", "Switch OR \n Button OR Door","NO Drawer", "NO Door", "NO Door \n and NO Drawer"]

DUMMY = False # for graph iteration 

if DUMMY:
    ours_means = {complex_behavior : {behavior : 0.1 * np.random.random() for behavior in condensed_atomic_behaviors} for complex_behavior in behaviors_list}
    ours_vars =  {complex_behavior : {behavior : 0.01 for behavior in condensed_atomic_behaviors} for complex_behavior in behaviors_list}
    planner_means = {complex_behavior : {behavior : 0.1 * np.random.random() for behavior in condensed_atomic_behaviors} for complex_behavior in behaviors_list}
    planner_vars =  {complex_behavior : {behavior : 0.01 for behavior in condensed_atomic_behaviors} for complex_behavior in behaviors_list}
    ctrl_means = {complex_behavior : {behavior : 0.1 * np.random.random() for behavior in condensed_atomic_behaviors} for complex_behavior in behaviors_list}
    ctrl_vars =  {complex_behavior : {behavior : 0.01 for behavior in condensed_atomic_behaviors} for complex_behavior in behaviors_list}


plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 11
plt.rcParams['ytick.labelsize'] = 11


num_seeds = 6


SCALING_FACTOR =  {"sliding_door" : 10,
            "drawer" : 10,
            "button" : 10,
            "switch" : 10,
            "lightbulb" : 0, # ignore the light because it's represented by the switch 
            "green_light" : 0,
            "red_block": 1,
            "blue_block" : 1,
            "pink_block" : 1}



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
    relevant_atomic_behaviors = ["button_on", "button_off", "switch_on", "switch_off", "drawer_open", "drawer_close", "door_left", "door_right", 
                          "red_lift", "blue_lift", "pink_lift", "no_behavior"] #, "other"]
    condensed_atomic_behaviors = ["button", "switch", "drawer", "door", "blocks", "no_behavior"] #, "no_behavior"] #, "other"]

    behavior_dict = {k : 0 for k in relevant_atomic_behaviors}
    condensed_behavior_dict = {k : 0 for k in condensed_atomic_behaviors}
    # adjustable_limits = {"sliding_door" : [0, 0.27], "drawer" : [0, 0.16], "switch" : [0, 0.09], "green_light" : [0, 1]}
    adjustable_limits = {"sliding_door" : [0, 0.27], "drawer" : [0, 0.16], "switch" : [0, 0.08], "green_light" : [0, 1]}
    half_thresholds = {k : (v[1] - v[0]) / 2 for k, v in adjustable_limits.items()}

    dataset = h5py.File(hdf5_name, 'r')
    data_grp = dataset["data"]
    steps_list = list()

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
                behavior_dict["red_lift"] += 1 
                condensed_behavior_dict["blocks"] += 1 
                something_happened = True 

            if np.linalg.norm(robot_pos - last_state["pink_block"]) < 0.1 and delta_state["pink_block"] > 0.001:
                behavior_dict["pink_lift"] += 1 
                condensed_behavior_dict["blocks"] += 1 
                something_happened = True 

            if np.linalg.norm(robot_pos - last_state["blue_block"]) < 0.1 and delta_state["blue_block"] > 0.001:
                behavior_dict["blue_lift"] += 1 
                condensed_behavior_dict["blocks"] += 1 
                something_happened = True 
            
            if something_happened: # this allows for multiple-counting 
                break 

            if delta_state["sliding_door"] > 0.05: 
                if first_state["sliding_door"] < last_state["sliding_door"]:
                    behavior_dict["door_left"] += 1 
                else:
                    behavior_dict["door_right"] += 1 
                condensed_behavior_dict["door"] += 1 
                something_happened = True 
                break 
            elif delta_state["drawer"] > 0.05: 
                if first_state["drawer"] > last_state["drawer"]:
                    behavior_dict["drawer_close"] += 1 
                else:
                    behavior_dict["drawer_open"] += 1 
                condensed_behavior_dict["drawer"] += 1 

                something_happened = True 
                break 
            elif delta_state["switch"] > 0.02:
                if first_state["switch"] > last_state["switch"]:
                    behavior_dict["switch_off"] += 1 
                else:
                    behavior_dict["switch_on"] += 1 
                condensed_behavior_dict["switch"] += 1 

                something_happened = True 
                break 
            elif delta_state["green_light"] > 0.01:
                if first_state["green_light"] > last_state["green_light"]:
                    behavior_dict["button_off"] += 1 
                else:
                    behavior_dict["button_on"] += 1 
                condensed_behavior_dict["button"] += 1 

                something_happened = True 
                break 
        if not something_happened:
            behavior_dict["no_behavior"] += 1  # nothing happened 
            condensed_behavior_dict["no_behavior"] += 1 

        steps_list.append(step)
    
    total_sum = sum(behavior_dict.values())
    behavior_dict = {k : v / total_sum for k, v in behavior_dict.items()}
    condensed_behavior_dict = {k : v / total_sum for k, v in condensed_behavior_dict.items()}
    full_report = {"Full distribution" : behavior_dict, "Condensed distribution" : condensed_behavior_dict,  "Average Length" : np.mean(steps_list)}

    if save_data:
        plt.bar(behavior_dict.keys(), behavior_dict.values())
        plt.xticks(rotation=90)
        plt.title("Behavior Distribution " + name)
        plt.tight_layout()
        plt.savefig(f"{RESULTS_DIR}/{name}_full_dist.png")
        plt.close()
        with open(f"{RESULTS_DIR}/{name}_dist.json", "w") as f:
            json.dump(full_report, f, indent = 2)
    
    return full_report






if not DUMMY:
    ours_means = {complex_behavior : {behavior : 0 for behavior in condensed_atomic_behaviors} for complex_behavior in behaviors_list}
    ours_vars =  {complex_behavior : {behavior : 0 for behavior in condensed_atomic_behaviors} for complex_behavior in behaviors_list}
    for behavior in behaviors_list:
        target_vals = {k : list() for k in condensed_atomic_behaviors}
        print(behavior)
        for seed in range(num_seeds):
            name = behavior + "_" + str(seed)
            record_file =  f"/store/real/maxjdu/repos/robotrainer/FINAL_EXPERIMENTS/Multi_Neg_w_negs/{name}/{name}.hdf5"
            try:
                dist = generate_detailed_behavior_distribtuion(name, record_file, save_data = False)
            except KeyboardInterrupt:
                exit() 
            except:
                print("Error with ", record_file)
                continue 

            for behavior_atom in condensed_atomic_behaviors:
                target_vals[behavior_atom].append(dist["Condensed distribution"][behavior_atom])
        # at the end of this, i have a distionary {behavior primitives : distribution }

        for behavior_atom in condensed_atomic_behaviors:
            ours_means[behavior][behavior_atom] = np.mean(np.array(target_vals[behavior_atom])) if len(target_vals[behavior_atom]) > 0 else 0
            ours_vars[behavior][behavior_atom] = np.std(np.array(target_vals[behavior_atom]))  / np.sqrt(num_seeds - 1) if len(target_vals[behavior_atom]) > 0 else 0

    planner_means = {complex_behavior : {behavior : 0 for behavior in condensed_atomic_behaviors} for complex_behavior in behaviors_list}
    planner_vars =  {complex_behavior : {behavior : 0 for behavior in condensed_atomic_behaviors} for complex_behavior in behaviors_list}
    for behavior in behaviors_list:
        target_vals = {k : list() for k in condensed_atomic_behaviors}
        print(behavior)
        for seed in range(num_seeds):
            name = behavior + "_planner_" + str(seed)
            record_file =  f"/store/real/maxjdu/repos/robotrainer/FINAL_EXPERIMENTS/Multi_Neg_w_negs/{name}/{name}.hdf5"
            try:
                dist = generate_detailed_behavior_distribtuion(name, record_file, save_data = False)
            except KeyboardInterrupt:
                exit() 
            except:
                print("Error with ", record_file)
                continue 

            for behavior_atom in condensed_atomic_behaviors:
                target_vals[behavior_atom].append(dist["Condensed distribution"][behavior_atom])
        # at the end of this, i have a distionary {behavior primitives : distribution }

        for behavior_atom in condensed_atomic_behaviors:
            planner_means[behavior][behavior_atom] = np.mean(np.array(target_vals[behavior_atom])) if len(target_vals[behavior_atom]) > 0 else 0
            planner_vars[behavior][behavior_atom] = np.std(np.array(target_vals[behavior_atom]))  / np.sqrt(num_seeds - 1) if len(target_vals[behavior_atom]) > 0 else 0


    # ctrl_means = {behavior : 0 for behavior in condensed_atomic_behaviors} 
    # ctrl_vars = {behavior : 0 for behavior in condensed_atomic_behaviors}
    # target_vals = {k : list() for k in condensed_atomic_behaviors}
    # for seed in range(num_seeds):
    #     name = "Control" + "_" + str(seed)
    #     record_file =  f"/store/real/maxjdu/repos/robotrainer/FINAL_EXPERIMENTS/Multi_Neg/{name}/{name}.hdf5"
    #     try:
    #         dist = generate_detailed_behavior_distribtuion(name, record_file, save_data = False)
    #     except KeyboardInterrupt:
    #         exit() 
    #     except:
    #         print("Error with ", record_file)
    #         continue 
    #     for behavior in condensed_atomic_behaviors:
    #         target_vals[behavior].append(dist["Condensed distribution"][behavior])

    # for behavior in condensed_atomic_behaviors:
    #     ctrl_means[behavior] = np.mean(np.array(target_vals[behavior])) if len(target_vals[behavior]) > 0 else 0
    #     ctrl_vars[behavior] = np.std(np.array(target_vals[behavior]))  / np.sqrt(num_seeds - 1) if len(target_vals[behavior]) > 0 else 0

    # graph_ctrl_means = {b : {"control" : ctrl_means[b]} for b in condensed_atomic_behaviors}
    # graph_ctrl_vars = {b : {"control" : ctrl_vars[b]} for b in condensed_atomic_behaviors}


# at the end of this, I have a dict with a distribution of behaviors per complex behaviors 
# turn this into a dict key = primitive behavior, value = dictionary with complex behaviors 

graph_ours_means = {b : {k : ours_means[k][b] for k in behaviors_list} for b in condensed_atomic_behaviors}
graph_ours_vars = {b : {k : ours_vars[k][b] for k in behaviors_list} for b in condensed_atomic_behaviors}

graph_planner_means = {b : {k : planner_means[k][b] for k in behaviors_list} for b in condensed_atomic_behaviors}
graph_planner_vars = {b : {k : planner_vars[k][b] for k in behaviors_list} for b in condensed_atomic_behaviors}


# color_dict = {"button" : "#4357AD", "switch": "#48A9A6", "drawer":  "#E15A97", "door":  "#94CF26", "blocks":  "#FF9F1C", "no_behavior": "#222222"}
color_dict_planner = {"blocks" : "#C3D8E6", "button": "#A1CAE5", "door":  "#74B9E6", "drawer":  "#4CA5DC", "switch":  "#3792D0"}#, "no_behavior": "#222222"}
color_dict_ours = {"blocks" : "#D1CFE6", "button": "#A19FCE", "door":  "#7471B4", "drawer":  "#5753A3", "switch":  "#45429A"}#, "no_behavior": "#222222"}
    
    

relevant_atomic_behaviors = {behavior : sorted([k for k in behavior.split("_") if k in condensed_atomic_behaviors]) for behavior in behaviors_list}
width = 0.33  # Width of the bars
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10, 4), gridspec_kw={'width_ratios': [1/2, 1/2]})
# fig.set_size_inches(12, 6, forward=True)

# if you find this cursed, you are absolutely correct 
irrelevant_alpha = 0.3
label_fontsize = 10
error_kw = {'elinewidth': 1, 'capthick': 1, 'ecolor': 'black'}
capsize = 2
padding = 3

for i, full_behavior in enumerate(behaviors_list[:3]): 
    irrelevant_atomic_behaviors = [behavior for behavior in condensed_atomic_behaviors if behavior not in relevant_atomic_behaviors[full_behavior] and behavior != "no_behavior"]
    bottom_ours = 0
    bottom_planner = 0
    for relevant_behavior in relevant_atomic_behaviors[full_behavior]:
        value = graph_ours_means[relevant_behavior][full_behavior]
        std = graph_ours_vars[relevant_behavior][full_behavior]
        ours_bar = ax1.bar([i - 0.5 * width], [value], width, bottom = [bottom_ours], label=relevant_behavior, 
                           error_kw=error_kw, yerr = [std], capsize=capsize,  color = "#B4DCB5", edgecolor = "#278B4D", linewidth = 1, alpha = 0.7) # ) #  edgecolor = "green", linewidth = 3) #color = "green") #color_dict_ours[relevant_behavior]) 
        bottom_ours += value 

        value = graph_planner_means[relevant_behavior][full_behavior]
        planner_bar = ax1.bar([i + 0.5 * width], [value], width, bottom = [bottom_planner], label=relevant_behavior, 
                           error_kw=error_kw, yerr = [std], capsize=capsize, color = "#B4DCB5", edgecolor = "#278B4D", linewidth = 1, alpha = 0.7) # edgecolor = "green", linewidth = 3) #, color =  "green") #color_dict_planner[relevant_behavior]) 
        bottom_planner += value 
    bottom_ours_relevant = bottom_ours
    bottom_planner_relevant = bottom_planner

    for irrelevant_behavior in irrelevant_atomic_behaviors:
        value = graph_ours_means[irrelevant_behavior][full_behavior]
        ours_bar = ax1.bar([i - 0.5 * width], [value], width, bottom = [bottom_ours], label=irrelevant_behavior, color = "#FAA41C", edgecolor = "black", linewidth = 0, alpha = 0.3) # color = color_dict_ours[irrelevant_behavior], edgecolor = "yellow", linewidth = 3)#color = "yellow") #color = color_dict_ours[irrelevant_behavior], alpha = irrelevant_alpha, hatch = "xx") 
        bottom_ours += value 

        value = graph_planner_means[irrelevant_behavior][full_behavior]
        planner_bar = ax1.bar([i + 0.5 * width], [value], width, bottom = [bottom_planner], label=irrelevant_behavior, color = "#FAA41C", edgecolor = "black", linewidth = 0, alpha = 0.3)#color = color_dict_ours[irrelevant_behavior], edgecolor = "yellow", linewidth = 3) #, color = "yellow", linewidth = 2) # color = color_dict_planner[irrelevant_behavior], alpha = irrelevant_alpha, hatch = "xx") 
        bottom_planner += value 
    ours_bar = ax1.bar([i - 0.5 * width], [1 - bottom_ours], width, bottom = [bottom_ours], color = "#C8525D", edgecolor = "black", linewidth = 0, alpha = 0.3) # "gray", edgecolor = "red", linewidth = 3) #  color = "red") #color = color_dict_ours[irrelevant_behavior], alpha = irrelevant_alpha, hatch = "xx") 
    ours_bar = ax1.bar([i + 0.5 * width], [1 - bottom_planner], width, bottom = [bottom_planner], color = "#C8525D", edgecolor = "black", linewidth = 0, alpha = 0.3) #color = "gray", edgecolor = "red", linewidth = 3) #color = color_dict_ours[irrelevant_behavior], alpha = irrelevant_alpha, hatch = "xx") 

    
    # ax1.bar_label(ours_bar, labels = [f"{bottom_ours_relevant:.2f}"], padding=padding, fontsize=label_fontsize, fontname="serif")
    # ax1.bar_label(planner_bar, labels = [f"{bottom_planner_relevant:.2f}"], padding=padding, fontsize=label_fontsize, fontname="serif")

    # add label 

for i, full_behavior in enumerate(behaviors_list[3:]): 
    irrelevant_atomic_behaviors = [behavior for behavior in condensed_atomic_behaviors if behavior not in relevant_atomic_behaviors[full_behavior] and behavior != "no_behavior"]
    bottom_ours = 0
    bottom_planner = 0

    for irrelevant_behavior in irrelevant_atomic_behaviors:
        value = graph_ours_means[irrelevant_behavior][full_behavior]
        std = graph_ours_vars[irrelevant_behavior][full_behavior]

        ours_bar = ax2.bar([i - 0.5 * width], [value], width, bottom = [bottom_ours], label=irrelevant_behavior,  error_kw=error_kw, yerr = [std], capsize=capsize, 
                           color = "#B4DCB5", edgecolor = "#278B4D", linewidth = 1, alpha = 0.7)
        bottom_ours += value 

        value = graph_planner_means[irrelevant_behavior][full_behavior]
        planner_bar = ax2.bar([i + 0.5 * width], [value], width, bottom = [bottom_planner], label=irrelevant_behavior,  error_kw=error_kw, yerr = [std], capsize=capsize, 
                             color = "#B4DCB5", edgecolor = "#278B4D", linewidth = 1, alpha = 0.7)
        bottom_planner += value 
        
    for relevant_behavior in relevant_atomic_behaviors[full_behavior]:
        value = graph_ours_means[relevant_behavior][full_behavior]
        std = graph_ours_vars[relevant_behavior][full_behavior]
        ours_bar = ax2.bar([i - 0.5 * width], [value], width, bottom = [bottom_ours], label=relevant_behavior, 
                     color = "#FAA41C", edgecolor = "black", linewidth = 0, alpha = 0.3) 
        bottom_ours += value 

        value = graph_planner_means[relevant_behavior][full_behavior]
        planner_bar = ax2.bar([i + 0.5 * width], [value], width, bottom = [bottom_planner], label=relevant_behavior, 
                     color = "#FAA41C", edgecolor = "black", linewidth = 0, alpha = 0.3) 
        bottom_planner += value 


    ours_bar = ax2.bar([i - 0.5 * width], [1 - bottom_ours], width, bottom = [bottom_ours], color = "#C8525D", edgecolor = "black", linewidth = 0, alpha = 0.3) # "gray", edgecolor = "red", linewidth = 3) #  color = "red") #color = color_dict_ours[irrelevant_behavior], alpha = irrelevant_alpha, hatch = "xx") 
    ours_bar = ax2.bar([i + 0.5 * width], [1 - bottom_planner], width, bottom = [bottom_planner], color = "#C8525D", edgecolor = "black", linewidth = 0, alpha = 0.3) #color = "gray", edgecolor = "red", linewidth = 3) #color = color_dict_ours[irrelevant_behavior], alpha = irrelevant_alpha, hatch = "xx") 


# from matplotlib.patches import Patch
# custom_patches = [Patch(facecolor=color, label = label) for label, color in color_dict.items()] 
# plt.legend(handles=custom_patches)

# ax.set_ylabel('Success Rate')
# ax.set_title('Double Bar Plot')
ax1.set_xticks([0, 1, 2])
ax2.set_xticks([0, 1, 2])
# ax.legend()
ax2.set_xticklabels(pretty_behaviors_list[3:], fontname = 'DejaVu Sans') 
ax1.set_xticklabels(pretty_behaviors_list[:3], fontname = 'DejaVu Sans') 

ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

yticks = ax1.get_yticks()
ax1.set_yticks(yticks[1:])
yticks = ax2.get_yticks()
ax2.set_yticks(yticks[1:])

ax2.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
ax1.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])

# ax.legend()
plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/Exp4.pdf")
plt.close()