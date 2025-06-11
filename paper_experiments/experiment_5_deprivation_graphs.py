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
# plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 11
plt.rcParams['ytick.labelsize'] = 11


control_file =  "CALVINABCD_CONTROL_block_ss1"

relevant_behaviors = ["button_on", "button_off", "switch_on", "switch_off", "drawer_open", "drawer_close", "door_left", "door_right",  
                           "red_lift", "blue_lift", "pink_lift"] #, "no_behavior"] #, "other"]

target_behavior = "switch_on"
percentages = [1, 3, 5, 10, 20, 40, 60, 80] #, 100]
seeds = 6

DUMMY = False
if DUMMY: # mockup graph for figure iteration 
    # progression_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    ours_performance = {k : np.random.random() for k in percentages} 
    ours_performance_vars = {k : 0.1 for k in percentages} 
    planner_performance = {k : np.random.random() for k in percentages} 
    planner_performance_vars = {k : 0.1 for k in percentages} 
    gc_performance = {k : np.random.random() for k in percentages} 
    gc_performance_vars = {k : 0.1 for k in percentages} 
    ctrl_performance = {k : np.random.random() for k in percentages} 
    ctrl_performance_vars = {k : 0.1 for k in percentages} 

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
        plt.savefig(f"{RESULTS_DIR}/{name}_full_dist.png")
        plt.close()
        with open(f"{RESULTS_DIR}/{name}_dist.json", "w") as f:
            json.dump(full_report, f, indent = 2)
    
    return full_report



    
if not DUMMY: 
    ours_performance = {} 
    ours_performance_vars = {} 
    for percentage in percentages:
        performance_list = list() 
        for seed in range(seeds):
            name = target_behavior + "_" + str(percentage) + "_" + str(seed)
            record_file =  f"/store/real/maxjdu/repos/robotrainer/FINAL_EXPERIMENTS/Deprivation/{name}/{name}.hdf5"
            try:
                dist = generate_detailed_behavior_distribtuion(name, record_file, save_data = False)
                print(percentage, " : ", dist["Full distribution"][target_behavior])
            except KeyboardInterrupt:
                exit() 
            except:
                print("Error with ", record_file)
                continue 
            performance_list.append(dist["Full distribution"][target_behavior])
        ours_performance[percentage] = np.mean(np.array(performance_list)) #dist["Full distribution"][target_behavior]
        ours_performance_vars[percentage] = np.std(np.array(performance_list)) / np.sqrt(seeds - 1) #dist["Full distribution"][target_behavior]

    planner_performance = {} 
    planner_performance_vars = {} 
    for percentage in percentages:
        performance_list = list() 
        for seed in range(seeds):
            name = target_behavior + "_planner_" + str(percentage) + "_" + str(seed)
            record_file =  f"/store/real/maxjdu/repos/robotrainer/FINAL_EXPERIMENTS/Deprivation/{name}/{name}.hdf5"
            try:
                dist = generate_detailed_behavior_distribtuion(name, record_file, save_data = False)
                print(percentage, " : ", dist["Full distribution"][target_behavior])
            except KeyboardInterrupt:
                exit() 
            except:
                print("Error with ", record_file)
                continue 
            performance_list.append(dist["Full distribution"][target_behavior])
        planner_performance[percentage] = np.mean(np.array(performance_list)) #dist["Full distribution"][target_behavior]
        planner_performance_vars[percentage] = np.std(np.array(performance_list)) / np.sqrt(seeds - 1) #dist["Full distribution"][target_behavior]

    # gc_performance = {} 
    # gc_performance_vars = {} 
    # for percentage in percentages:
    #     performance_list = list() 
    #     for seed in range(seeds):
    #         name = target_behavior + "_gc_" + str(percentage) + "_" + str(seed)
    #         record_file =  f"/store/real/maxjdu/repos/robotrainer/FINAL_EXPERIMENTS/Deprivation/{name}/{name}.hdf5"
    #         try:
    #             dist = generate_detailed_behavior_distribtuion(name, record_file, save_data = False)
    #             print(percentage, " : ", dist["Full distribution"][target_behavior])
    #         except KeyboardInterrupt:
    #             exit() 
    #         except:
    #             print("Error with ", record_file)
    #             continue 
    #         performance_list.append(dist["Full distribution"][target_behavior])
    #     gc_performance[percentage] = np.mean(np.array(performance_list)) #dist["Full distribution"][target_behavior]
    #     gc_performance_vars[percentage] = np.std(np.array(performance_list)) / np.sqrt(seeds - 1) #dist["Full distribution"][target_behavior]

    ctrl_performance = {} 
    ctrl_performance_vars = {} 
    for percentage in percentages:
        performance_list = list() 
        for seed in range(seeds):
            name = target_behavior + "_control_" + str(percentage) + "_" + str(seed)
            record_file =  f"/store/real/maxjdu/repos/robotrainer/FINAL_EXPERIMENTS/Deprivation/{name}/{name}.hdf5"
            try:
                dist = generate_detailed_behavior_distribtuion(name, record_file, save_data = False)
                print(percentage, " : ", dist["Full distribution"][target_behavior])
            except KeyboardInterrupt:
                exit() 
            except:
                print("Error with ", record_file)
                continue 
            performance_list.append(dist["Full distribution"][target_behavior])
        ctrl_performance[percentage] = np.mean(np.array(performance_list)) #dist["Full distribution"][target_behavior]
        ctrl_performance_vars[percentage] = np.std(np.array(performance_list)) / np.sqrt(seeds - 1) #dist["Full distribution"][target_behavior]


# colors = ["#414067", "#89C5EC", "#E37B3A", "#EFCD76", "#9397A4"] # ours guidance, ours sampling, goal conditioning, felix, baseline 
colors = ["#5753A3", "#89C5EC", "#E37B3A", "#FAAF6F", "#EFCD76"] # ours guidance, ours sampling, goal conditioning, felix, baseline 

fig, ax = plt.subplots()
fig.set_size_inches(4, 4, forward=True)

ax.plot(ours_performance.keys(), ours_performance.values(), color = colors[0], label = "Ours: Guidance", linewidth=2, markersize=6, marker='o')
ax.fill_between(ours_performance.keys(), np.array(list(ours_performance.values())) - np.array(list(ours_performance_vars.values())), 
                np.array(list(ours_performance.values())) + np.array(list(ours_performance_vars.values())), color=colors[0], alpha=0.3)

ax.plot(planner_performance.keys(), planner_performance.values(), color = colors[1], label = "Ours: Sampling", linewidth=2, markersize=6, marker='o')
ax.fill_between(planner_performance.keys(), np.array(list(planner_performance.values())) - np.array(list(ours_performance_vars.values())), 
                np.array(list(planner_performance.values())) + np.array(list(planner_performance_vars.values())), color=colors[1], alpha=0.3)

# ax.plot(gc_performance.keys(), gc_performance.values(), color = colors[2], label = "Goal Conditioned")
# ax.fill_between(gc_performance.keys(), np.array(gc_performance.values()) - np.array(gc_performance_vars.values()), 
#                 np.array(gc_performance.values()) + np.array(gc_performance_vars.values()), color=colors[2], alpha=0.3)

ax.plot(ctrl_performance.keys(), ctrl_performance.values(), color = colors[4], label = "Control", linewidth=2, markersize=6, marker='o')
ax.fill_between(ctrl_performance.keys(), np.array(list(ctrl_performance.values())) - np.array(list(ctrl_performance_vars.values())), 
                np.array(list(ctrl_performance.values())) + np.array(list(ctrl_performance_vars.values())), color=colors[4], alpha=0.3)


ax.set_ylim(0, 1)


ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
# ax.spines['bottom'].set_visible(False)

yticks = ax.get_yticks()
ax.set_yticks(yticks[1:])

# may disable for the final paper 
# ax.legend()
# ax.set_xlabel("Percentage Data Present")
# ax.set_ylabel("Success Rate")

plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/Exp3_Deprivation.pdf")
plt.close()
