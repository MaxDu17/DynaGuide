
from torch.utils.data import Dataset
import h5py
import torch 
import numpy as np 

"""
This the dataset used to train the dynamics model of DynaGuide 
"""
class MultiviewDataset(Dataset):
    def __init__(self, hdf5_file, action_chunk_length, cameras, padding = False, mode = "embedding", pad_mode = "zeros",
                 proprio = None): # = ["agentview_image", "robot0_eye_in_hand_image"]):
        super().__init__()
        print(f"Using camera {cameras}")
        self.padding = padding # padding allows you to query to the end of a demosntration regardless of chunk length, typically true 

        self.length = 0
        self.proprio = proprio 
        self.h5_file = h5py.File(hdf5_file, "r")
        self.lengths_list = list()
        self.action_chunk_length = action_chunk_length
        self.pad_mode = pad_mode 

        self.cameras = cameras 
        self.demos_list = sorted(list(self.h5_file["data"].keys()), key = lambda x: int(x.split("_")[-1]))
        self.build_lengths_list()
        self.length = self.__len__()
        self.mode = mode # "embedding" means returning s, a, s_T. "classifier" means return s, a, Y

        self.sample_distribution = {} 
    
    def build_lengths_list(self):
        for demo in self.demos_list: 
            try:
                if not self.padding: 
                    self.lengths_list.append(self.h5_file["data"][demo]["actions"].shape[0] - (self.action_chunk_length))
                else:
                    self.lengths_list.append(self.h5_file["data"][demo]["actions"].shape[0])
            except:
                print(f"Skipped demo {demo} because it is empty!")

    def __len__(self):
        return sum(self.lengths_list)
    
    def parse_idx(self, idx): # index to demo number and index in the demo 
        demo = 0
        remaining_idx = idx 
        while remaining_idx >= self.lengths_list[demo]:
            remaining_idx -= self.lengths_list[demo]
            demo += 1
        return demo, remaining_idx
    
    def get_bounds_of_demo(self, demo):
        cum_demo = np.cumsum(self.lengths_list) 
        start = cum_demo[demo] - self.lengths_list[demo] 
        end = cum_demo[demo]
        return start, end 

    def smooth_one_hot(self, size, index, epsilon=0.1): # generates a (potentially) soft one-hot vector 
        vector = np.full(size, epsilon / (size - 1))  # Spread probability
        vector[index] = 1 - epsilon  # Assign main probability to the target
        return vector

    def __getitem__(self, idx):
        demo, remaining_idx = self.parse_idx(idx)
        selected_demo = self.h5_file["data"][self.demos_list[demo]]
        selected_state = {camera: np.transpose(selected_demo["obs"][camera][remaining_idx], (2, 0, 1)) for camera in self.cameras}
        last_selected_state = {camera: np.transpose(selected_demo["obs"][camera][-1], (2, 0, 1)) for camera in self.cameras}
        if self.proprio is not None: 
            selected_state[self.proprio] = selected_demo["obs"][self.proprio][remaining_idx]
            last_selected_state[self.proprio] = selected_demo["obs"][self.proprio][-1]

        if "label" in selected_demo:
            task_label = int(selected_demo["label"][remaining_idx]) 

        if self.mode == "classifier": 
            target = self.smooth_one_hot(4, task_label, 0) # not smooth anymore 
        else:
            target = last_selected_state 

        if not self.padding or remaining_idx + self.action_chunk_length <= self.lengths_list[demo]: # if we don't need padding 
            selected_actions = selected_demo["actions"][remaining_idx  : remaining_idx  + self.action_chunk_length]
            return selected_state, selected_actions, target
         
         # pad the actions based on the scheme 
        amount_to_pad = remaining_idx + self.action_chunk_length - self.lengths_list[demo]
        selected_actions = np.zeros((self.action_chunk_length, selected_demo["actions"].shape[1]))
        if self.pad_mode == "repeat":
            selected_actions[-amount_to_pad:] = selected_demo["actions"][-1]
        selected_actions[:-amount_to_pad] = selected_demo["actions"][remaining_idx : ]
        return selected_state, selected_actions, target

    def _get_cube_pos(self, idx): # only for the toy experiment! 
        demo, remaining_idx = self.parse_idx(idx)
        selected_demo = self.h5_file["data"][self.demos_list[demo]]
        return selected_demo["obs"]["states"][remaining_idx], selected_demo["obs"]["agent_pos"][remaining_idx]

    def get_labeled_item(self, idx): 
        """
        This function retrieves like __getitem__ except that it will also get the behavior label, useful for diagnostics and tests. It is not used during 
        normal DynaGuide operation. 
        """
        demo, remaining_idx = self.parse_idx(idx)
        selected_demo = self.h5_file["data"][self.demos_list[demo]]
        if "label" in selected_demo:
            label = int(selected_demo["label"][remaining_idx]) #1 if selected_demo["label"][remaining_idx] else 0
        elif "behavior" in selected_demo.attrs:
            label = selected_demo.attrs["behavior"]
        else:
            label = None

        selected_state = {camera: np.transpose(selected_demo["obs"][camera][remaining_idx], (2, 0, 1)) for camera in self.cameras}
        if self.proprio is not None: 
            selected_state[self.proprio] = selected_demo["obs"][self.proprio][remaining_idx]

        if not self.padding or remaining_idx + self.action_chunk_length <= self.lengths_list[demo]: 
            selected_actions = selected_demo["actions"][remaining_idx  : remaining_idx  + self.action_chunk_length]
            return selected_state, selected_actions, label

        amount_to_pad = remaining_idx + self.action_chunk_length - self.lengths_list[demo]
        selected_actions = np.zeros((self.action_chunk_length, selected_demo["actions"].shape[1]))
        if amount_to_pad == 0:
            selected_actions = selected_demo["actions"][remaining_idx : ]
        else:
            selected_actions[:-amount_to_pad] = selected_demo["actions"][remaining_idx : ]
        
        # this supports the padding mode for repeating 
        if self.pad_mode == "repeat":
            selected_actions[-amount_to_pad:] = selected_demo["actions"][-1]

        return selected_state, selected_actions, label