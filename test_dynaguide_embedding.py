import matplotlib.pyplot as plt 
import argparse 
import torch 
import tqdm 
import numpy as np 

from core.dynamics_models import FinalStatePredictionDino

from core.embedder_datasets import MultiviewDataset
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

import random 
import cv2 
import imageio
import torchvision 
import os 

torch.set_printoptions(sci_mode=False)

MAIN_CAMERA = "third_person"

def generate_TSNE_points(embeddings, dims):
    tsne = TSNE(dims, verbose=1)
    tsne_proj = tsne.fit_transform(embeddings)
    return tsne_proj

def generate_PCA_points(embeddings, dims):
    pca = PCA(n_components=dims)
    pca.fit(embeddings)
    print(pca.explained_variance_ratio_)
    pca_proj = pca.transform(embeddings)
    return pca_proj, pca 

def prepare(data, device = "cuda"):
    if type(data) == dict:
        return {k : v.to(device).to(torch.float32) for k, v in data.items()}
    return data.to(device).to(torch.float32)

def prepare_np(data, device = "cuda"):
    if type(data) == dict:
        return {k : torch.tensor(v).to(device).to(torch.float32) for k, v in data.items()}
    return torch.tensor(data).to(device).to(torch.float32)


def jiggle_action(model, mixed_dataset, output_dir, step):
    # take a demonstration
    # for every state: sample 10 random actions for one step, get the next state, plot average variance 
    # tests the impact of actions in a unit ball and the smoothness of the embedding 
    # to ensure reasonability, we jiggle the action in the unit ball as set by the real actions 
    print("Jiggling Action!")
    demo = random.randint(0, len(mixed_dataset.lengths_list) - 1)
    start, end = mixed_dataset.get_bounds_of_demo(demo)
    var_list = list() 
    dist_list = list() 
    for i in tqdm.tqdm(range(start, end)):
        sample = mixed_dataset.get_labeled_item(i) 
        state, action, label = prepare_np(sample[0]), prepare_np(sample[1]), sample[2]
        state = {k : torch.unsqueeze(v, dim = 0) for k, v in state.items()}
        action = torch.unsqueeze(action, dim = 0) # compensates for the batch dimension 
        with torch.no_grad():
            batch_state = {MAIN_CAMERA : torch.tile(state[MAIN_CAMERA], (11, 1, 1, 1)), "proprio" : torch.tile(state["proprio"], (11, 1))} 
            action_mags = torch.mean(torch.abs(action), dim = 1)
            batch_action = 2 * (torch.rand((11, 16, 7), device = action.device) - 0.5)
            batch_action = batch_action * action_mags # scaling 
            
            batch_action[0] = action # first slot is the base 
            z_hats = model.state_action_embedding(batch_state, batch_action, normalize =False).detach().cpu().numpy()
        
        base_z_hat = z_hats[0]
        other_z_hats = z_hats[1:]
        z_hat_variances = np.mean(np.std(other_z_hats, axis = 0))
        z_hat_distances = np.mean(np.abs(other_z_hats - base_z_hat))   
        var_list.append(z_hat_variances)
        dist_list.append(z_hat_distances)
    fig, axs = plt.subplots(2, 1)
    axs[0].plot(var_list)
    axs[0].set_title("Z hat noise variances (10 samples)")
    axs[1].plot(dist_list)
    axs[1].set_title("Z hat distance from accepted action (10 samples)")
    plt.tight_layout()
    plt.savefig(output_dir + str(step) + f"_action_jiggle_{demo}.png", dpi=300)
    plt.close()
    

def replay_through_reconstruction(model, mixed_dataset, output_dir, step):
    """
    This function will feed states through the dynamics model and decoder to see how well they are reconstructing. 
    You should be seeing a predcited last state taht gets better as the episode goes on 
    """
    print("Replay through reconstruction")
    color_output = imageio.get_writer(output_dir + str(step) +  "reconstructions.mp4")
    resizer = torchvision.transforms.Resize((200, 200))

    for trial in tqdm.tqdm(range(10)):
        demo = random.randint(0, len(mixed_dataset.lengths_list) - 1)
        start, end = mixed_dataset.get_bounds_of_demo(demo)
        for i in range(start, end):
            sample = mixed_dataset.__getitem__(i)
            state, action, last_state = prepare_np(sample[0]), prepare_np(sample[1]), prepare_np(sample[2])
            state = {k : torch.unsqueeze(v, dim = 0) for k, v in state.items()}
            last_state = {k : torch.unsqueeze(v, dim = 0) for k, v in last_state.items()}
            action = torch.unsqueeze(action, dim = 0) # compensates for the batch dimension 

            with torch.no_grad():
                pred_last, reco = model(state, action)
                reco = torch.clip(resizer(reco), 0, 1)
            combined_frame = torch.concatenate((state[MAIN_CAMERA][0]  / 255, reco[0], last_state[MAIN_CAMERA][0]  / 255), dim = 1).detach().cpu().numpy()
            combined_frame = np.transpose(combined_frame, (1, 2, 0))
            color_output.append_data(combined_frame)
    color_output.close()

def plot_prediction_similarity(model, mixed_dataset, output_dir, step, action_mod = None): 
    """
    This function will plot the l2 distance between the true end state and the predicted end state embedding. It should go down as the episode goes on
    This fuction also gives an option (action_mod) to mess with the actions to see if the model is learning to use the action properly
    """
    print("Plotting Prediction Similarity")
    for selection in range(10):
        demo = random.randint(0, len(mixed_dataset.lengths_list) - 1)
        start, end = mixed_dataset.get_bounds_of_demo(demo)
        mse_loss = torch.nn.MSELoss()
        error_list = list()
        variance_list = list()
        for j in tqdm.tqdm(range(start, end)):
            sample = mixed_dataset.__getitem__(j)
            state, action, last_state = prepare_np(sample[0]), prepare_np(sample[1]), prepare_np(sample[2])
            state = {k : torch.unsqueeze(v, dim = 0) for k, v in state.items()}
            last_state = {k : torch.unsqueeze(v, dim = 0) for k, v in last_state.items()}
            action = torch.unsqueeze(action, dim = 0) # compensates for the batch dimension 
            if action_mod == "noise":
                action_noised = torch.randn_like(action)
                action_noised[-1] = action[-1]
                action = action_noised 
            if action_mod == "negate":
                action[0:-1] = -action[0:-1]

            with torch.no_grad():
                # last_state_predict, reco_last = model(state, action) # the image is 200x200, we resize to 224 for dino 
                last_state_embed = model.state_embedding(last_state)#.detach().cpu().numpy()
                predicted_last_state_embed = model.state_action_embedding(state, action)#.detach().cpu().numpy() # gets the s, a embedding only
                mse_loss = torch.nn.MSELoss()
                loss = mse_loss(last_state_embed, predicted_last_state_embed)
            error_list.append(loss.item())
        
        plt.plot(error_list)
            
    plt.title("End State Prediction")
    plt.xlabel("Step")
    plt.ylabel("MSE Prediction Error")
    plt.legend()
    mod = action_mod if action_mod is not None else ""
    plt.savefig(output_dir + str(step) + f"_end_prediction_error_{mod}.png", dpi=300)
    plt.close()


def compare_final_state_similarity_by_category(model, good_dataset, mixed_dataset, output_dir, step, good_key):
    """
    This function plots the category-level similiarty compared to a target category. 
    You should see the lowest distance being same-cateogry and related categories. 
    """
    good_embeddings_list = list()
    print("Precomputing good")
    idx = 0
    good_img_list = list() 
    for length in tqdm.tqdm(good_dataset.lengths_list):
        idx += length 
        sample = good_dataset.get_labeled_item(idx - 1)#, flatten_action = False)
        state, action, label = prepare_np(sample[0]), prepare_np(sample[1]), sample[2]
        good_img_list.append(np.transpose(state[MAIN_CAMERA].detach().cpu().numpy(), (1, 2, 0)) / 255)
        state = {k : torch.unsqueeze(v, dim = 0) for k, v in state.items()}
        action = torch.unsqueeze(action, dim = 0) # compensates for the batch dimension 
        with torch.no_grad():
            good_embedding = model.state_embedding(state, normalize = False).flatten(start_dim = 1) 
        good_embeddings_list.append(good_embedding.clone())
    good_embeddings = torch.concatenate(good_embeddings_list, dim = 0)

    good_list = list()
    bad_list = list() 
    label_list = list()
    embed_list = list() 
    idx = 0
    mixed_img_list = list()
    category_lists = {} 

    print("Calculating mixed")
    for length in tqdm.tqdm(mixed_dataset.lengths_list):
        idx += length 
        sample = mixed_dataset.get_labeled_item(idx - 1) 
        state, action, label = prepare_np(sample[0]), prepare_np(sample[1]), sample[2]
        mixed_img_list.append(np.transpose(state[MAIN_CAMERA].detach().cpu().numpy(), (1, 2, 0)) / 255)
        state = {k : torch.unsqueeze(v, dim = 0) for k, v in state.items()}
        action = torch.unsqueeze(action, dim = 0) # compensates for the batch dimension 
        with torch.no_grad():
            s_embedding = model.state_embedding(state, normalize = False).flatten(start_dim = 1) 
        s_norm = torch.cdist(good_embeddings, s_embedding, p=2.0)
        sa_average_norm = torch.mean(s_norm).detach().cpu().numpy()

        label = str(label)

        embed_list.append(s_embedding[0])
        if label not in category_lists:
            category_lists[label] = []
        category_lists[label].append(sa_average_norm)
    
    averages = {k : sum(v) / len(v) for k, v in category_lists.items()}
    color_list = ["blue" if k != good_key else "green" for k in averages.keys()]
    plt.bar(averages.keys(), averages.values(), color = color_list)
    plt.title("End State L2 Distance Average")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.ylim(min(averages.values()) - 50, max(averages.values()) + 50)
    plt.savefig(output_dir + str(step) + "_end_state_category_distr.png")
    plt.close() 


def compare_final_state_similarity(model, good_dataset, mixed_dataset, output_dir, step , good_key):
    """
    This function takes a desired end state set and plots a histogram of end state similarities between in-class and out-of-class data
    Ideally, the embedding should be able to separate in-class and out-of-class data through L2 distance. Otherwise, DynaGuide may not work properly. 
    """
    good_embeddings_list = list()
    print("Precomputing good")
    idx = 0
    good_img_list = list() 
    for length in tqdm.tqdm(good_dataset.lengths_list):
        idx += length 
        sample = good_dataset.get_labeled_item(idx - 1) 
        state, action, label = prepare_np(sample[0]), prepare_np(sample[1]), sample[2]
        good_img_list.append(np.transpose(state[MAIN_CAMERA].detach().cpu().numpy(), (1, 2, 0)) / 255)
        state = {k : torch.unsqueeze(v, dim = 0) for k, v in state.items()}
        action = torch.unsqueeze(action, dim = 0) # compensates for the batch dimension 
        with torch.no_grad():
            good_embedding = model.state_embedding(state, normalize = False).flatten(start_dim = 1) 
        good_embeddings_list.append(good_embedding.clone())
    good_embeddings = torch.concatenate(good_embeddings_list, dim = 0)

    good_list = list()
    bad_list = list() 
    label_list = list()
    embed_list = list() 
    idx = 0
    mixed_img_list = list()
    print("Calculating mixed")
    for length in tqdm.tqdm(mixed_dataset.lengths_list):
        idx += length 
        sample = mixed_dataset.get_labeled_item(idx - 1) 
        state, action, label = prepare_np(sample[0]), prepare_np(sample[1]), sample[2]
        mixed_img_list.append(np.transpose(state[MAIN_CAMERA].detach().cpu().numpy(), (1, 2, 0)) / 255)
        state = {k : torch.unsqueeze(v, dim = 0) for k, v in state.items()}
        action = torch.unsqueeze(action, dim = 0) # compensates for the batch dimension 
        with torch.no_grad():
            s_embedding = model.state_embedding(state, normalize = False).flatten(start_dim = 1) # gets the s, a embedding only 
        s_norm = torch.cdist(good_embeddings, s_embedding, p=2.0)
        # sa_average_norm = torch.mean(s_norm).detach().cpu().numpy()
        sa_average_norm = torch.min(s_norm).detach().cpu().numpy() # get the closet one 

        label = str(label)
        embed_list.append(s_embedding[0])
        label_list.append(label)
        if label == good_key: # str is needed to adapt to numerical labels for the pymunk 
            good_list.append(sa_average_norm)
        else:
            bad_list.append(sa_average_norm)
    

    bins = np.linspace(min(min(good_list), min(bad_list)), max(max(good_list), max(bad_list)), 20)
    plt.hist([good_list, bad_list], bins = bins, color = ["green", "blue"], label = ["good", "bad"])
    plt.legend(loc='upper right')
    plt.title("End State L2 Distance Average (lower better)")
    plt.savefig(output_dir + str(step) + "_end_state_distr.png")
    plt.close() 
    


# this will visualize how the model processes individual and average trajectories 
def plot_valid_trajectory(model, good_dataset, mixed_dataset, output_dir, step, good_label): # this doesn't take in the dataloader 
    # precompute the good end state 
    good_embeddings_list = list()
    print("Precomputing good")
    idx = 0
    good_img_list = list() 
    for length in tqdm.tqdm(good_dataset.lengths_list):
        idx += length 
        # sample = good_dataset.get_labeled_item(idx - 1) 
        sample = good_dataset.get_labeled_item(idx - 16) # one chunk before 
        state, action, label = prepare_np(sample[0]), prepare_np(sample[1]), sample[2]
        good_img_list.append(np.transpose(state[MAIN_CAMERA].detach().cpu().numpy(), (1, 2, 0)) / 255)
        state = {k : torch.unsqueeze(v, dim = 0) for k, v in state.items()}
        action = torch.unsqueeze(action, dim = 0) # compensates for the batch dimension 
        with torch.no_grad():
            good_embedding = model.state_embedding(state).flatten(start_dim = 1)
        good_embeddings_list.append(good_embedding.clone())
    good_embeddings = torch.concatenate(good_embeddings_list, dim = 0)


    good_dict = {}
    bad_dict = {}
    good_step_list = list()
    bad_step_list = list()
    print("Evaluating trajectories")
    for i in tqdm.tqdm(range(len(mixed_dataset))):
        sample = mixed_dataset.get_labeled_item(i) #, flatten_action = False)
        demo, idx_in_demo = mixed_dataset.parse_idx(i)
        state, action, label = prepare_np(sample[0]), prepare_np(sample[1]), sample[2]
        if label == good_label and demo not in good_dict:
            good_dict[demo] = list()
            current_traj = good_dict[demo]
            current_step_list = good_step_list 
        elif label != good_label and demo not in bad_dict:
            bad_dict[demo] = list()
            current_traj = bad_dict[demo] 
            current_step_list = bad_step_list
        
        if len(current_step_list) <= idx_in_demo:
            current_step_list.append(list())
        

        # state = torch.unsqueeze(state, dim = 0)
        state = {k : torch.unsqueeze(v, dim = 0) for k, v in state.items()}

        action = torch.unsqueeze(action, dim = 0) # compensates for the batch dimension 
        with torch.no_grad():
            final_embedding = model.state_action_embedding(state, action).flatten(start_dim = 1) # gets the s, a embedding only 
        
        mse_distance = torch.mean(torch.square(good_embeddings - final_embedding)) # using broadcasting 
        # dot_product = good_embeddings @ embedding.T 
        # # average_dot_product = torch.mean(dot_product)
        # average_dot_product = torch.quantile(dot_product, 0.8)
        # outputs = torch.einsum('ik,jk->ij', s_a_embed, s_prime_embed)
        current_traj.append(mse_distance.item())
        current_step_list[idx_in_demo].append(mse_distance.item())

    print("Compute self-similarity")
    key_dict = dict()
    for i in tqdm.tqdm(range(len(good_dataset))):
        sample = good_dataset.get_labeled_item(i)#, flatten_action = False)
        demo, idx_in_demo = good_dataset.parse_idx(i)
        state, action, label = prepare_np(sample[0]), prepare_np(sample[1]), sample[2]
        state = {k : torch.unsqueeze(v, dim = 0) for k, v in state.items()}
        action = torch.unsqueeze(action, dim = 0) # compensates for the batch dimension 
        with torch.no_grad():
            embedding = model.state_action_embedding(state, action).flatten(start_dim = 1) # gets the next step embedding 
            
        mse_distance = torch.mean(torch.square(good_embeddings - embedding))

        if demo not in key_dict:
            key_dict[demo] = list()

        key_dict[demo].append(mse_distance.item())


    for demo, traj in good_dict.items():
        plt.plot(traj, color = "green")
    for demo, traj in bad_dict.items():
        plt.plot(traj, color = "red")

    plt.xlabel("Steps")
    plt.ylabel("Classifier Output")
    plt.title("Individual Trajectories")
    plt.savefig(output_dir + str(step) + "_individual_similarities.png")
    plt.close()

    # plot them individually 
    fig, ax_tuple = plt.subplots(ncols = 11, nrows = 10) #, figsize = (7, 7)) #use figure size to manually control how large the plot will be, in inches
    count_red = 0 
    count_green = 1 
    for demo, traj in good_dict.items():
        ax_tuple[count_green // 10, count_green % 10].plot(traj, color = "green")
        ax_tuple[count_green // 10, count_green % 10].get_xaxis().set_visible(False)
        ax_tuple[count_green // 10, count_green % 10].get_yaxis().set_visible(False)
        count_green += 2 
        if count_green > 99:
            break # don't overfill 
    for demo, traj in bad_dict.items():
        ax_tuple[count_red // 10, count_red % 10].plot(traj, color = "red")
        ax_tuple[count_red // 10, count_red % 10].get_xaxis().set_visible(False)
        ax_tuple[count_red // 10, count_red % 10].get_yaxis().set_visible(False)
        count_red += 2 
        if count_red > 99:
            break 
    
    count = 0
    for demo, traj in key_dict.items():
        ax_tuple[count, 10].plot(traj, color = "blue")
        ax_tuple[count, 10].get_xaxis().set_visible(False)
        ax_tuple[count, 10].get_yaxis().set_visible(False)
        count += 1 
        if count > 9:
            break

    plt.savefig(output_dir + str(step) + "_tiled_plots.pdf")
    plt.close()

    mean_good_list = [sum(k) / len(k) for k in good_step_list]
    std_good_list = [np.std(k) for k in good_step_list]
    mean_bad_list = [sum(k) / len(k) for k in bad_step_list]
    std_bad_list = [np.std(k) for k in bad_step_list]

    good_x = np.arange(len(mean_good_list))
    bad_x = np.arange(len(mean_bad_list))
    plt.plot(mean_good_list, "green")
    plt.plot(mean_bad_list, "red")
    plt.fill_between(good_x, [k-v for k, v in zip(mean_good_list, std_good_list)], [k+v for k, v in zip(mean_good_list, std_good_list)], color = "green", alpha = 0.2)
    plt.fill_between(bad_x, [k-v for k, v in zip(mean_bad_list, std_bad_list)], [k+v for k, v in zip(mean_bad_list, std_bad_list)], color = "red", alpha = 0.2)
    plt.xlabel("Steps")
    plt.ylabel("Classifier Output")
    plt.title("Average Trajectories")
    plt.savefig(output_dir + str(step) + "_average_similarities.png")
    plt.close()


def main(args):
    # this needs to be aligned with the action chunk length in the trained model 
    ACTION_DIM = 7 # for Calvin 

    # for calvin 
    proprio_dim = 15 
    proprio = "proprio" # set to None if you want to exclude propriorception 

    cameras = [MAIN_CAMERA] # you can change this; it's hardcoded
    padding = True
    pad_mode = "repeat" #"repeat" # "zeros" for calvin 

    model = FinalStatePredictionDino(ACTION_DIM, args.action_chunk_length, cameras=cameras, reconstruction = True, \
                                     proprio = proprio, proprio_dim = proprio_dim)

    model.load_state_dict(torch.load(args.checkpoint))
    model.to("cuda")
    model.eval()


    good_dataset = MultiviewDataset(args.good_hdf5, action_chunk_length = args.action_chunk_length, cameras = cameras, padding = padding,
                                    pad_mode = pad_mode, proprio = proprio)
    
    checkpoint_number = args.checkpoint.split("/")[-1].split(".")[0]
    # for calvin environment 
    mixed_dataset = MultiviewDataset(args.mixed_hdf5, action_chunk_length = args.action_chunk_length, cameras = cameras, padding = padding,
                                     pad_mode = pad_mode, proprio = proprio)


    log_dir = args.exp_dir + "tests/"
    if not os.path.isdir(log_dir):
        os.mkdir(log_dir)

    # these are the tests that you can run 
    compare_final_state_similarity(model, good_dataset, mixed_dataset, log_dir, checkpoint_number, args.key)
    compare_final_state_similarity_by_category(model, good_dataset, mixed_dataset, log_dir, checkpoint_number, args.key)
    plot_prediction_similarity(model, mixed_dataset, log_dir, checkpoint_number, action_mod = "noise")
    plot_prediction_similarity(model, mixed_dataset, log_dir, checkpoint_number, action_mod = "negate")
    plot_prediction_similarity(model, mixed_dataset, log_dir, checkpoint_number)
    replay_through_reconstruction(model, mixed_dataset, log_dir, checkpoint_number) #step hard coded for now 
    plot_valid_trajectory(model, good_dataset, mixed_dataset, log_dir, checkpoint_number, args.key) 
    jiggle_action(model, mixed_dataset, log_dir, checkpoint_number) #step hard coded for now



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp_dir",
        type=str,
        default=None,
        help="Where the trained model is",
    )
    parser.add_argument(
        "--key",
        type=str,
        default=None,
        help="This is the behavior you want to isolate",
    )

    parser.add_argument(
        "--good_hdf5",
        type=str,
        default=None,
        help="One behavior hdf5",
    )
    parser.add_argument(
        "--mixed_hdf5",
        type=str,
        default=None,
        help="Mixed (but labeled) hdf5",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="where to find the checkpoint for the dynamics model",
    )
    parser.add_argument(
        "--action_chunk_length",
        type=int,
        default=None,
        help="the length of the action input to the dynamics model",
    )
    args = parser.parse_args()

    main(args)