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
    ss_performance = {k : np.random.random() for k in percentages} 
    ss_performance_vars = {k : 0.1 for k in percentages} 
    alpha_performance = {k : np.random.random() for k in percentages} 
    alpha_performance_vars = {k : 0.1 for k in percentages} 
    scale_performance = {k : np.random.random() for k in percentages} 
    scale_performance_vars = {k : 0.1 for k in percentages} 
    not_noised_mean = np.random.random()
    not_noised_var = 0.1 
    noised_mean = np.random.random()
    noised_var = 0.1 

    examples_performance = {k : np.random.random() for k in percentages} 
    examples_performance_vars = {k : 0.1 for k in percentages} 

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




# wait 
ss=[ 1, 2, 3, 4, 5, 6]
scales = [ 0.5, 0.7, 0.9, 1, 1.2, 1.2, 1.4, 1.6, 1.8, 2, 2.2, 2.4, 2.6, 2.8, 3, 3.4, 3.8, 4.2, 4.6, 5, 5.4, 5.8, 6.2]
alphas = [ 1, 2, 5, 10, 20, 30, 40, 50]
examples = [ 1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
  
if not DUMMY: 
    target_behavior = "switch_on"

    examples_performance = {} 
    examples_performance_vars = {} 
    for example in examples: 
        performance_list = list() 
        for seed in range(seeds):
            name = "examples_" + str(example) + "_" + str(seed)
            record_file =  f"/store/real/maxjdu/repos/robotrainer/FINAL_EXPERIMENTS/Ablations/{name}/{name}.hdf5"
            try:
                dist = generate_detailed_behavior_distribtuion(name, record_file, save_data = False)
                print(example, " : ", dist["Full distribution"][target_behavior])
            except KeyboardInterrupt:
                exit() 
            except:
                print("Error with ", record_file)
                continue 
            performance_list.append(dist["Full distribution"][target_behavior])
        examples_performance[example] = np.mean(np.array(performance_list)) #dist["Full distribution"][target_behavior]
        examples_performance_vars[example] = np.std(np.array(performance_list)) / np.sqrt(seeds - 1) #dist["Full distribution"][target_behavior]

    alpha_performance = {} 
    alpha_performance_vars = {} 
    for alpha in alphas: 
        performance_list = list() 
        for seed in range(seeds):
            name = "alpha_" + str(alpha) + "_" + str(seed)
            record_file =  f"/store/real/maxjdu/repos/robotrainer/FINAL_EXPERIMENTS/Ablations/{name}/{name}.hdf5"
            try:
                dist = generate_detailed_behavior_distribtuion(name, record_file, save_data = False)
                print(alpha, " : ", dist["Full distribution"][target_behavior])
            except KeyboardInterrupt:
                exit() 
            except:
                print("Error with ", record_file)
                continue 
            performance_list.append(dist["Full distribution"][target_behavior])
        alpha_performance[alpha] = np.mean(np.array(performance_list)) #dist["Full distribution"][target_behavior]
        alpha_performance_vars[alpha] = np.std(np.array(performance_list)) / np.sqrt(seeds - 1) #dist["Full distribution"][target_behavior]

    scale_performance = {} 
    scale_performance_vars = {} 
    for scale in scales: 
        performance_list = list() 
        for seed in range(seeds):
            name = "scale_" + str(scale) + "_" + str(seed)
            record_file =  f"/store/real/maxjdu/repos/robotrainer/FINAL_EXPERIMENTS/Ablations/{name}/{name}.hdf5"
            try:
                dist = generate_detailed_behavior_distribtuion(name, record_file, save_data = False)
                print(scale, " : ", dist["Full distribution"][target_behavior])
            except KeyboardInterrupt:
                exit() 
            except:
                print("Error with ", record_file)
                continue 
            performance_list.append(dist["Full distribution"][target_behavior])
        scale_performance[scale] = np.mean(np.array(performance_list)) #dist["Full distribution"][target_behavior]
        scale_performance_vars[scale] = np.std(np.array(performance_list)) / np.sqrt(seeds - 1) #dist["Full distribution"][target_behavior]

    ss_performance = {} 
    ss_performance_vars = {} 
    for ss_value in ss: 
        performance_list = list() 
        for seed in range(seeds):
            name = "ss_s3_" + str(ss_value) + "_" + str(seed)
            record_file =  f"/store/real/maxjdu/repos/robotrainer/FINAL_EXPERIMENTS/Ablations/{name}/{name}.hdf5"
            try:
                dist = generate_detailed_behavior_distribtuion(name, record_file, save_data = False)
                print(ss_value, " : ", dist["Full distribution"][target_behavior])
            except KeyboardInterrupt:
                exit() 
            except:
                print("Error with ", record_file)
                continue 
            performance_list.append(dist["Full distribution"][target_behavior])
        ss_performance[ss_value] = np.mean(np.array(performance_list)) #dist["Full distribution"][target_behavior]
        ss_performance_vars[ss_value] = np.std(np.array(performance_list)) / np.sqrt(seeds - 1) #dist["Full distribution"][target_behavior]

  

    performance_list = list() 
    for seed in range(seeds):
        name = "noised_" + str(seed)
        record_file =  f"/store/real/maxjdu/repos/robotrainer/FINAL_EXPERIMENTS/Ablations/{name}/{name}.hdf5"
        try:
            dist = generate_detailed_behavior_distribtuion(name, record_file, save_data = False)
            print("noised", " : ", dist["Full distribution"][target_behavior])
        except KeyboardInterrupt:
            exit() 
        except:
            print("Error with ", record_file)
            continue 
        performance_list.append(dist["Full distribution"][target_behavior])
    noised_mean = np.mean(np.array(performance_list)) #dist["Full distribution"][target_behavior]
    noised_var = np.std(np.array(performance_list)) / np.sqrt(seeds - 1) #dist["Full distribution"][target_behavior]

    performance_list = list() 
    for seed in range(seeds):
        name = "not_noised_" + str(seed)
        record_file =  f"/store/real/maxjdu/repos/robotrainer/FINAL_EXPERIMENTS/Ablations/{name}/{name}.hdf5"
        try:
            dist = generate_detailed_behavior_distribtuion(name, record_file, save_data = False)
            print("not noised", " : ", dist["Full distribution"][target_behavior])
        except KeyboardInterrupt:
            exit() 
        except:
            print("Error with ", record_file)
            continue 
        performance_list.append(dist["Full distribution"][target_behavior])
    not_noised_mean = np.mean(np.array(performance_list)) #dist["Full distribution"][target_behavior]
    not_noised_var = np.std(np.array(performance_list)) / np.sqrt(seeds - 1) #dist["Full distribution"][target_behavior]


    

# colors = ["#414067", "#89C5EC", "#E37B3A", "#EFCD76", "#9397A4"] # ours guidance, ours sampling, goal conditioning, felix, baseline 
colors = ["#5753A3", "#89C5EC", "#E37B3A", "#FAAF6F", "#EFCD76"] # ours guidance, ours sampling, goal conditioning, felix, baseline 

fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4)
fig.set_size_inches(16, 4, forward=True)

ax1.plot(scale_performance.keys(), scale_performance.values(), color = colors[0], label = "Ours: Guidance", linewidth=2, markersize=6, marker='o')
ax1.fill_between(scale_performance.keys(), np.array(list(scale_performance.values())) - np.array(list(scale_performance_vars.values())), 
                np.array(list(scale_performance.values())) + np.array(list(scale_performance_vars.values())), color=colors[0], alpha=0.3)

ax2.plot(alpha_performance.keys(), alpha_performance.values(), color = colors[0], label = "Ours: Guidance", linewidth=2, markersize=6, marker='o')
ax2.fill_between(alpha_performance.keys(), np.array(list(alpha_performance.values())) - np.array(list(alpha_performance_vars.values())), 
                np.array(list(alpha_performance.values())) + np.array(list(alpha_performance_vars.values())), color=colors[0], alpha=0.3)

ax3.plot(ss_performance.keys(), ss_performance.values(), color = colors[0], label = "Ours: Guidance", linewidth=2, markersize=6, marker='o')
ax3.fill_between(ss_performance.keys(), np.array(list(ss_performance.values())) - np.array(list(ss_performance_vars.values())), 
                np.array(list(ss_performance.values())) + np.array(list(ss_performance_vars.values())), color=colors[0], alpha=0.3)

ax4.plot(examples_performance.keys(), examples_performance.values(), color = colors[0], label = "Ours: Guidance", linewidth=2, markersize=6, marker='o')
ax4.fill_between(examples_performance.keys(), np.array(list(examples_performance.values())) - np.array(list(examples_performance_vars.values())), 
                np.array(list(examples_performance.values())) + np.array(list(examples_performance_vars.values())), color=colors[0], alpha=0.3)



ax1.set_ylim(0.4, 0.8)
ax2.set_ylim(0.4, 0.8)
ax3.set_ylim(0.4, 0.8)
ax4.set_ylim(0.4, 0.8)


ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
yticks = ax1.get_yticks()
ax1.set_yticks(yticks[1:])

ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
yticks = ax2.get_yticks()
ax2.set_yticks(yticks[1:])
ax2.set_xticks([1, 10, 20, 30, 40, 50])


ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)
yticks = ax3.get_yticks()
ax3.set_yticks(yticks[1:])

ax4.spines['top'].set_visible(False)
ax4.spines['right'].set_visible(False)
yticks = ax4.get_yticks()
ax4.set_yticks(yticks[1:])
ax4.set_xticks([1, 5, 10, 15, 20])

plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/Parameters.pdf")
plt.close()



fig, ax = plt.subplots()
fig.set_size_inches(4, 4, forward=True)
width = 0.75
error_kw = {'elinewidth': 1, 'capthick': 1, 'ecolor': 'black'}
ax.bar([0, 1], [not_noised_mean, noised_mean], width, yerr = [not_noised_var, noised_var], label="Goal Conditioning", 
                        capsize=2, error_kw=error_kw, color = colors[0])

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_xticks([0, 1])
ax.set_xticklabels(["Noise Pretrain", "No Noise Pretrain"], fontname = 'DejaVu Sans')# to include average
plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/Ablations.pdf")
plt.close()