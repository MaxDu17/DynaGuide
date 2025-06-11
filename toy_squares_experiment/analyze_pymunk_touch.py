import matplotlib.pyplot as plt 
import h5py
import numpy as np 
import torch 
import imageio.v2 as iio
import imageio 
import pickle 
import tqdm 
import json 

from matplotlib.collections import LineCollection

# RESULTS_DIR = "/store/real/maxjdu/repos/robotrainer/analysis"
RESULTS_DIR = "/store/real/maxjdu/repos/robotrainer/paper_figs"
import cv2 


def blend_images(img1, img2):
    # Split RGBA channels
    b1, g1, r1, a1 = cv2.split(img1.astype(float) / 255)
    b2, g2, r2, a2 = cv2.split(img2.astype(float) / 255)

    # Compute alpha blend factor
    alpha_blend = a1 + a2 * (1 - a1)

    # Blend each channel
    b = (b1 * a1 + b2 * a2 * (1 - a1)) / (alpha_blend + 1e-5)
    g = (g1 * a1 + g2 * a2 * (1 - a1)) / (alpha_blend + 1e-5)
    r = (r1 * a1 + r2 * a2 * (1 - a1)) / (alpha_blend + 1e-5)

    # Scale back and merge
    blended = cv2.merge([b, g, r, alpha_blend]) * 255
    blended = blended.astype(np.uint8)

    return blended

def composite_on_white(img):
    # Ensure image has an alpha channel
    if img.shape[2] < 4:
        raise ValueError("Image must have an alpha channel (RGBA)")

    # Split channels
    b, g, r, a = cv2.split(img.astype(float) / 255)  # Normalize to [0,1]

    # Create a white background
    white_bg = np.ones((img.shape[0], img.shape[1], 3), dtype=np.float32)

    # Composite: Alpha blending formula
    b = (1 - a) * white_bg[:, :, 0] + a * b
    g = (1 - a) * white_bg[:, :, 1] + a * g
    r = (1 - a) * white_bg[:, :, 2] + a * r

    # Merge back to an RGB image
    composite_img = cv2.merge([b, g, r]) * 255  # Scale back to [0,255]
    return composite_img.astype(np.uint8)


def generate_trajectory_traces(name, hdf5_name):
    # SUGGESTIONs:
#     Transparency/color gradient based on time 
    # A little marker at the end 
    # Lines thicker [done]
    # Reduce transparency of the cubes [done]
    # Highlight one trajectory 


    # only works when there isn't randomness 
    dataset = h5py.File(hdf5_name, 'r')
    data_grp = dataset["data"]
    plt.tight_layout()
    average_image = None 

    fig, ax = plt.subplots()

    # for demo in tqdm.tqdm(data_grp.keys()):
    for demo in tqdm.tqdm(data_grp.keys()):
        positions = data_grp[demo]["obs"]["agent_pos"][:, -1]
        start_image =data_grp[demo]["obs"]["image"][0, -1] #.astype(np.float32) #/ 255.0
        
        # if average_image is None:
        #     average_image = start_image 
        # else:
        #     average_image = cv2.addWeighted(average_image, 0.7, start_image, 0.3, 0)
        start_image_alpha = cv2.cvtColor(start_image, cv2.COLOR_BGR2BGRA)
        gray_white = cv2.cvtColor(start_image, cv2.COLOR_BGR2GRAY)
        mask = cv2.threshold(gray_white, 180, 255, cv2.THRESH_BINARY)[1]  # Adjust threshold
        start_image_alpha[:, :, 3][mask == 255] = 0
        start_image_alpha[:, :, 3][mask != 255] = 50
        # Invert the mask so white areas become transparent
        # mask_inv = cv2.bitwise_not(mask)

        if average_image is None:
            average_image = start_image_alpha 
        else:
            average_image = blend_images(average_image, start_image_alpha)
        
        # plt.scatter(positions[:, 0], positions[:, 1], color = "black", alpha = 0.3, s = 3, zorder = 2)
        # plt.plot(positions[:, 0], positions[:, 1], color = "black", alpha = 0.3, zorder = 2) #, s = 3, zorder = 2)
        norm = plt.Normalize(0, positions.shape[0])
        cmap = plt.get_cmap("plasma")  # 'plasma' colormap transitions from purple to yellow

        # Create segments for a continuous color transition
        points = positions[:, np.newaxis, :] #reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        # Create the line collection with the gradient
        lc = LineCollection(segments, cmap=cmap, norm=norm, linewidth=2, alpha = 0.7, zorder = 2)
        lc.set_array(np.linspace(0, positions.shape[0], positions.shape[0]-1))

        ax.add_collection(lc)
        ax.scatter(positions[-1, 0:1], positions[-1, 1:], color = "black", s = 5, zorder = 3, alpha = 0.7)
        # ax.scatter([0],[0], color = "black", s = 20, zorder= 2)
        # ax.autoscale()


    # blended_image = average_image / len(data_grp.keys())
    # blended_image = average_image / 255
    blended_image = composite_on_white(average_image)
    ax.imshow(np.flipud(blended_image), extent = [-1, 1, -1, 1], zorder = 1)
    # plt.title("Cube Touch Traces")
   
    plt.axis("off")
    plt.savefig(f"{RESULTS_DIR}/{name}_traces.pdf", dpi = 300)
    plt.close()

        
        # cubes_pos = data_grp[demo]["obs"]["states"][:, -1]
    pass 


# this function creates the guidance CI graph for the paper 
def plot_cubes_color_ci(root_dir, name, target, num_runs):
    blue_list = [] 
    for i in range(num_runs):
        # name, root_dir, name, target, num_runs
        # plots the scatter of all of the cubes in the trial and highlights the pressed cube in red (others are in gray)
        name_mod = name + "_" + str(i)
        hdf5_name = root_dir + name_mod + "/" + name_mod + ".hdf5"
        hdf5_name_alt = root_dir + name_mod + "/" + name + ".hdf5"
        try:
            dataset = h5py.File(hdf5_name, 'r')
        except FileNotFoundError:
            dataset = h5py.File(hdf5_name_alt, 'r')

        data_grp = dataset["data"]
        total = len(data_grp.keys())
        x_list = list()
        y_list = list()
        color_list = list()
        successes = 0 
        target_count = 0
        near_target = 0 

        color_list = ["Blue", "Red", "Green", "Yellow"]
        cube_touch_distr = {k : 0 for k in color_list}

        for demo in tqdm.tqdm(data_grp.keys()):
            positions = data_grp[demo]["obs"]["agent_pos"][:, -1]
            cubes_pos = data_grp[demo]["obs"]["states"][:, -1]
            cubes_pos = np.reshape(cubes_pos, (cubes_pos.shape[0], 4, 2))
        
            if data_grp[demo]["rewards"][-1] < 0.5: #don't include bad touches 
                # print("Rejected fail!")
                continue

            successes += 1 
            last_position = positions[-1]
            last_cube_position = cubes_pos[-1]
            distances = np.linalg.norm(last_cube_position - last_position, axis = 1)
            closest_cube = np.argmin(distances)
            
            cube_touch_distr[color_list[closest_cube]] += 1 

            if target[closest_cube] == 1:
                target_count += 1 
            
            close = False
            for element in range(4):
                if target[element] == 1 and distances[element] < 0.35:
                    color_list.append("blue" if closest_cube == 0 else "gray") # show partial success 
                    near_target += 1 
                    close = True 
                    break 
            
            if not close:
                color_list.append("blue" if closest_cube == 0 else "black")

            x_list.append(last_cube_position[closest_cube, 0])
            y_list.append(last_cube_position[closest_cube, 1])

        cube_touch_distr = {k: v / successes for k, v in cube_touch_distr.items()}
        blue_list.append(cube_touch_distr["Blue"])        # WE ONLY CARE ABOUT BLUE 

    return np.mean(blue_list), np.std(blue_list) / (num_runs**0.5)

        
    return cube_touch_distr

def plot_cubes_color(name, hdf5_name, target):
    # plots the scatter of all of the cubes in the trial and highlights the pressed cube in red (others are in gray)
    dataset = h5py.File(hdf5_name, 'r')
    data_grp = dataset["data"]
    total = len(data_grp.keys())
    x_list = list()
    y_list = list()
    color_list = list()
    successes = 0 
    target_count = 0
    near_target = 0 

    color_list = ["Blue", "Red", "Green", "Yellow"]
    cube_touch_distr = {k : 0 for k in color_list}

    for demo in tqdm.tqdm(data_grp.keys()):
        positions = data_grp[demo]["obs"]["agent_pos"][:, -1]
        cubes_pos = data_grp[demo]["obs"]["states"][:, -1]
        cubes_pos = np.reshape(cubes_pos, (cubes_pos.shape[0], 4, 2))
    
        if data_grp[demo]["rewards"][-1] < 0.5: #don't include bad touches 
            print("Rejected fail!")
            continue

        successes += 1 
        last_position = positions[-1]
        last_cube_position = cubes_pos[-1]
        distances = np.linalg.norm(last_cube_position - last_position, axis = 1)
        closest_cube = np.argmin(distances)
        
        cube_touch_distr[color_list[closest_cube]] += 1 

        if target[closest_cube] == 1:
            target_count += 1 
        
        close = False
        for element in range(4):
            if target[element] == 1 and distances[element] < 0.35:
                color_list.append("blue" if closest_cube == 0 else "gray") # show partial success 
                near_target += 1 
                close = True 
                break 
        
        if not close:
            color_list.append("blue" if closest_cube == 0 else "black")

        x_list.append(last_cube_position[closest_cube, 0])
        y_list.append(last_cube_position[closest_cube, 1])

    cube_touch_distr = {k: v / successes for k, v in cube_touch_distr.items()}


    # Create a bar chart
    colors = ["black" if v == 0 else "green" for v in target]
    plt.bar(cube_touch_distr.keys(), cube_touch_distr.values(), color=colors)
    plt.title(name)
    plt.savefig(f"{RESULTS_DIR}/CubeTouch_{name}.png")

    with open(f"{RESULTS_DIR}/CubeTouch_{name}.json", "w") as f:
        json.dump({"hdf5_name" : hdf5_name, "target": target.tolist(), "target rate": target_count / successes, "near target rate" : near_target / successes, 
                   "success rate": successes / len(data_grp.keys()), "distr" : cube_touch_distr}, f, indent = 2)
    plt.close()
    return cube_touch_distr

names = [
    "GUIDANCE_pymunk_late_choice_deterministic_control_randomized",
    "GUIDANCE_pymunk_late_choice_deterministic_0_7_randomized",
    "GUIDANCE_pymunk_blue_four_corners_deterministic_0_5",
    "GUIDANCE_pymunk_blue_four_corners_deterministic_control",
    "GUIDANCE_pymunk_red_four_corners_deterministic",
    "GUIDANCE_pymunk_yellow_four_corners_deterministic",
    "GUIDANCE_pymunk_green_four_corners_deterministic"
         ]

for name in names:
    print(name)
    hdf5_name = f"/store/real/maxjdu/repos/robotrainer/results/outputs/{name}/{name}.hdf5"
    target = np.array([1, 0, 0, 0])
    plot_cubes_color(name, hdf5_name, target)
    generate_trajectory_traces(name, hdf5_name)