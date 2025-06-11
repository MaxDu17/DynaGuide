import numpy as np
import random 
import torch 
from tqdm import tqdm 

def generate_reset_state(sim_hold = {}):
    """
    This function creates a reset state that is randomized while maintaining certain states required for tasks
    """
    """    
    Reference: this is what the lowdim states represent: 
    {"sliding_door" : state_obs[0],
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
    """

    adjustables = ["sliding_door", "drawer", "switch", "green_light"]
    adjustable_index = [0, 1, 3, 5]
    adjustable_limits = [[0, 0.27], [0, 0.16], [0, 0.09], [0, 1]] # hard coded from simulator 
    state = np.zeros((24,))
    binary_list = list()
    # return state, [False, False, False, False]
    for adjustable, idx, limits in zip(adjustables, adjustable_index, adjustable_limits):
        if adjustable in sim_hold:
            state[idx] = float(sim_hold[adjustable])
            binary_list.append(sim_hold[adjustable] > (limits[1] - limits[0]) / 2)
        else:
            select = random.random() > 0.5 
            state[idx] = limits[1] if select else limits[0]
            binary_list.append(select)
    return state, binary_list #binary list is used to detect behaviors 

def articulated_binaries_from_start_state(start_state): # given any start state, tell us what things are articulated in which direction 
    adjustable_index = [0, 1, 3, 5]
    adjustable_limits = [[0, 0.27], [0, 0.16], [0, 0.09], [0, 1]]
    binary_list = list()
    for idx, limits in zip(adjustable_index, adjustable_limits):
        midpoint = (limits[1] - limits[0]) / 2
        binary_list.append(start_state[idx] > midpoint)
    return binary_list 


def check_state_difference(start_state, state, robot_pos, binaries, for_display = False):
    """
    This function checks if something has changed (warranting the end of episode). 
    Normally for speed, it only checks for a slight change in state. For display purposes
    when you want to see a full articulation, the flag changes the end-of-episode criteria 
    """
    # if you want to use this code for clean display, we use different thresholds 
    # adjustables = ["sliding_door", "drawer", "switch", "green_light"]
    adjustable_index = [0, 1, 3, 5]
    adjustable_limits = [[0, 0.27], [0, 0.16], [0, 0.09], [0, 1]]
    for binary, idx, limits in zip(binaries, adjustable_index, adjustable_limits):
        midpoint = (limits[1] - limits[0]) / 2
        near_low = limits[0] + 0.25 * (limits[1] - limits[0])
        near_high = limits[0] + 0.75 * (limits[1] - limits[0])

        if not for_display:
            if binary and state[idx] < midpoint:
                return True 
            if not binary and state[idx] > midpoint:
                return True 
        else: # this is for display where we want the task to fully finish 
            if binary and state[idx] < near_low:
                return True 
            if not binary and state[idx] > near_high:
                return True 

    if not for_display and np.linalg.norm(robot_pos - state[6:9]) < 0.06 and np.linalg.norm(state[6:8] - start_state[6:8]) > 0.001:
        return True 

    if not for_display and np.linalg.norm(robot_pos - state[12:15]) < 0.06 and np.linalg.norm(state[12:14] - start_state[12:14]) > 0.001:
        return True 

    if not for_display and np.linalg.norm(robot_pos - state[18:21]) < 0.06 and np.linalg.norm(state[18:20] - start_state[18:20]) > 0.001:
        return True 
    
    if for_display and np.linalg.norm(robot_pos - state[6:9]) < 0.06 and np.linalg.norm(state[6:8] - start_state[6:8]) > 0.03:
        return True 

    if for_display and np.linalg.norm(robot_pos - state[12:15]) < 0.06 and np.linalg.norm(state[12:14] - start_state[12:14]) > 0.03:
        return True 

    if for_display and np.linalg.norm(robot_pos - state[18:21]) < 0.06 and np.linalg.norm(state[18:20] - start_state[18:20]) > 0.03:
        return True 
    

    return False 