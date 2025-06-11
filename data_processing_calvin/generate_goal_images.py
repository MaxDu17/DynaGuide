import h5py 
import numpy as np 
import matplotlib.pyplot as plt 
import tqdm 
import os 
from pathlib import Path



file_list = [
    "/store/real/maxjdu/repos/robotrainer/dataset/CalvinDD_ood_goals/OOD_Goal_Button_Off.hdf5",
    "/store/real/maxjdu/repos/robotrainer/dataset/CalvinDD_ood_goals/OOD_Goal_Button_On.hdf5",
    # "/store/real/maxjdu/repos/robotrainer/dataset/CalvinDD_ood_goals/OOD_Goal_Door_Left.hdf5",
    # "/store/real/maxjdu/repos/robotrainer/dataset/CalvinDD_ood_goals/OOD_Goal_Door_Right.hdf5",
    # "/store/real/maxjdu/repos/robotrainer/dataset/CalvinDD_ood_goals/OOD_Goal_Drawer_Open.hdf5",
    # "/store/real/maxjdu/repos/robotrainer/dataset/CalvinDD_ood_goals/OOD_Goal_Drawer_Close.hdf5",
    # "/store/real/maxjdu/repos/robotrainer/dataset/CalvinDD_ood_goals/FILTERED_OOD_Goal_Switch_Off.hdf5",
    # "/store/real/maxjdu/repos/robotrainer/dataset/CalvinDD_ood_goals/OOD_Goal_Switch_Off.hdf5",
    # "/store/real/maxjdu/repos/robotrainer/dataset/CalvinDD_ood_goals/OOD_Goal_Switch_On.hdf5"
]
save_locations = [
     "/store/real/maxjdu/repos/robotrainer/dataset/Calvin_Goals/ood_button_off/",
     "/store/real/maxjdu/repos/robotrainer/dataset/Calvin_Goals/ood_button_on/",
    #  "/store/real/maxjdu/repos/robotrainer/dataset/Calvin_Goals/ood_door_left/",
    #  "/store/real/maxjdu/repos/robotrainer/dataset/Calvin_Goals/ood_door_right/",
    #  "/store/real/maxjdu/repos/robotrainer/dataset/Calvin_Goals/ood_drawer_open/",
    #  "/store/real/maxjdu/repos/robotrainer/dataset/Calvin_Goals/ood_drawer_close/",
    #  "/store/real/maxjdu/repos/robotrainer/dataset/Calvin_Goals/ood_switch_off/",
    #  "/store/real/maxjdu/repos/robotrainer/dataset/Calvin_Goals/filtered_ood_switch_off/",
    #  "/store/real/maxjdu/repos/robotrainer/dataset/Calvin_Goals/ood_switch_on/"
]

for file, save_location in zip(file_list, save_locations):
    GOAL_MODALITY = "third_person"

    Path(save_location).mkdir(parents=True, exist_ok=True)

    dataset = h5py.File(file, 'r')

    count = 0 
    demo_list = dataset["data"].keys() 
    demo_list = sorted(demo_list, key=lambda x: int(x.split("_")[1]))
    for demo in tqdm.tqdm(demo_list):
        print(demo)
        demo_grp = dataset["data"][demo]
        img = demo_grp["obs"][GOAL_MODALITY][-1]
        if len(img.shape) == 4:
            img = img[0] # this is if there is saved frame stacking 
        plt.imsave(f"{save_location}{count}.png", img)
        count += 1 

