import matplotlib.pyplot as plt 
import argparse 
import torch 
import tqdm 
import numpy as np 

from embedder_models import FinalStateClassification, FinalStateClassificationMLP
from image_models import VAE

from embedder_datasets import MultiviewDataset
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression  # for Logistic Regression algorithm

import random 
import cv2 
import imageio
import torchvision 
import os 

torch.set_printoptions(sci_mode=False)

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

def calculate_simple_probe(features, labels):
    from sklearn.ensemble import GradientBoostingClassifier

    split = int(0.2 * labels.shape[0])
    train_features = features[:-split]
    test_features = features[-split:]

    train_labels = labels[:-split]
    test_labels = labels[-split:]


    # log = LogisticRegression()

    model = GradientBoostingClassifier(
        random_state=5, n_estimators=200, max_depth=10, learning_rate=0.2
    )

    model.fit(train_features, train_labels)
    bool_labels = model.predict(test_features)
    accuracy = np.sum(bool_labels == (test_labels > 0)) / test_features.shape[0]
    precision = np.sum(np.logical_and(bool_labels, test_labels > 0)) / np.sum(bool_labels)
    recall = np.sum(np.logical_and(bool_labels, test_labels > 0)) / np.sum(test_labels > 0)
    f1_score = (2 * precision * recall) / (precision + recall)
    print("Set Balance:", np.sum(test_labels) / np.size(test_labels), "Accuracy: ", accuracy.item(), "Precision: ", precision.item(), "Recall", recall.item(), "F1 Score", f1_score.item())

def compute_data_stack(mixed_dataset, output_dir, step):
    # this one is a diagnostic for the datset to make sure that the distribution is meaningful 
    image_arr = np.zeros((128, 128, 3), dtype = np.float32)
    for demo in tqdm.tqdm(range(len(mixed_dataset.lengths_list))):
        start, end = mixed_dataset.get_bounds_of_demo(demo)
        sample = mixed_dataset.__getitem__(start)
        image = sample[0]["image"]
        image= np.transpose(image, (1, 2, 0))
        white_pixels = (image == 255).all(axis=-1)
        image[white_pixels] = 0
        image_arr = np.maximum(image_arr, image)

    
    image_arr /= len(mixed_dataset.lengths_list)
  
    plt.imsave(output_dir + "dataset_avg.png", image_arr)


# ADAPTED
def compute_action_dependency(model, mixed_dataset, output_dir, step):
    prediction_ticker = {"closest_only" : 0, "action_closest_only" : 0, "the_same" : 0, "neither" : 0}

    for i in tqdm.tqdm(range(len(mixed_dataset))):
    # for i in tqdm.tqdm(range(100)):
        sample = mixed_dataset.__getitem__(i)
        state, action, label = prepare_np(sample[0]), prepare_np(sample[1]), prepare_np(sample[2])

        state = {k : torch.unsqueeze(v, dim = 0) for k, v in state.items()}
        action = torch.unsqueeze(action, dim = 0) # compensates for the batch dimension 
        # prediction = model(state, action).detach().cpu().numpy() 
        prediction = model(state, action).detach().cpu() 
        distribution_prediction = torch.nn.functional.softmax(prediction, dim = 1)[0] # (4, )
        locs, agent_pos = mixed_dataset._get_cube_pos(i)
   
        locs = locs.reshape(4, 2)

        last_action = action[0, -1].detach().cpu().numpy() 
        action_distances = np.linalg.norm(locs - last_action, axis = 1)
        action_closest = np.argmin(action_distances)

        state_distances = np.linalg.norm(locs - agent_pos, axis = 1)
        agent_closest = np.argmin(state_distances)

        model_prediction = np.argmax(distribution_prediction.detach().cpu().numpy())

        if action_closest == agent_closest:
            if model_prediction == action_closest:
                prediction_ticker["the_same"] += 1 
            else:
                prediction_ticker["neither"] += 1 
        else: 
            if model_prediction == action_closest:
                prediction_ticker["action_closest_only"] += 1 
            elif model_prediction == agent_closest:
                prediction_ticker["closest_only"] += 1 
            else:
                prediction_ticker["neither"] += 1 
    with open(output_dir + str(step) + "_action_dependency.json", "w") as f:
        print(prediction_ticker)
        import json 
        json.dump(prediction_ticker, f)
       

# ADAPRTED 
def visualize_action_shift(model, mixed_dataset, output_dir, step):
    print("Sweeping actions")
    # the vision: sweep the current action in a circle and visualize the color of the arrows 
    # for i in range(10):
    color_output = imageio.get_writer(output_dir + str(step) +  "action_shift_visualized.mp4")
    for k in range(30):
        demo = random.randint(0, len(mixed_dataset.lengths_list) - 1)
        start, end = mixed_dataset.get_bounds_of_demo(demo)
        steps = 8 
        for i in tqdm.tqdm(range(start, end)):
            sample = mixed_dataset.__getitem__(i)
            state, action, label = prepare_np(sample[0]), prepare_np(sample[1]), prepare_np(sample[2])
            locs, agent_pos = mixed_dataset._get_cube_pos(i)
            relative = action - torch.tensor(agent_pos).cuda() 
            
            image_visual = np.transpose(sample[0]["image"], (1, 2, 0))
            image_visual = cv2.resize(image_visual, (500, 500))

        

            color_ordering = [(0, 0, 255), (255, 0, 0), (0, 255, 0), (255, 255, 0)]
            state = {k : torch.unsqueeze(v, dim = 0) for k, v in state.items()}

            theta_jump = 2 * np.pi / steps 
            thetas = torch.arctan2(relative[:, 1], relative[:, 0])
            mags = torch.norm(relative, dim = 1)
            for j in range(steps):
                # this rotates the action chunk 
                mod_theta = thetas + j * theta_jump 
                reco_x = mags * torch.cos(mod_theta)
                reco_y = mags * torch.sin(mod_theta)
                reco_rel = torch.stack((reco_x, reco_y), dim = 1)
                reco_abs = reco_rel + torch.tensor(agent_pos).cuda() 

            
                fed_reco_abs = torch.unsqueeze(reco_abs, dim = 0).float() # compensates for the batch dimension
                prediction = model(state, fed_reco_abs).detach().cpu().numpy()
                arrow_color = color_ordering[np.argmax(prediction)]
                # annotate with lines 
                shifted_pos = ((agent_pos[0] + 1) / 2, (agent_pos[1] + 1) / 2)
                shifted_end_action = ((reco_abs[-1, 0] + 1) / 2, (reco_abs[-1, 1] + 1) / 2)
                image_visual = cv2.arrowedLine(image_visual, 
                                (int(shifted_pos[0] * image_visual.shape[0]), int(shifted_pos[1] * image_visual.shape[1])),
                                (int(shifted_end_action[0] * image_visual.shape[0]), int(shifted_end_action[1] * image_visual.shape[1])),
                                arrow_color, 5)
                # image_visual = cv2.arrowedLine(image_visual, 
                #                                 (int(agent_pos[0] * image_visual.shape[0]), int(agent_pos[1] * image_visual.shape[1])),
                #                                 (int(reco_abs[-1, 0] * image_visual.shape[0]), int(reco_abs[-1, 1]* image_visual.shape[1])),
                #                                 arrow_color, 5)

            color_output.append_data(image_visual)
        

# ADAPTED 
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
        sample = mixed_dataset.__getitem__(i)
        state, action, label = prepare_np(sample[0]), prepare_np(sample[1]), prepare_np(sample[2])

        state = {k : torch.unsqueeze(v, dim = 0) for k, v in state.items()}
        action = torch.unsqueeze(action, dim = 0) # compensates for the batch dimension

      
        # prediction = model(state, action).detach().cpu().numpy() 
        with torch.no_grad():
            batch_state = {"image" : torch.tile(state["image"], (11, 1, 1, 1))} # this is jank af 
            # take magnitude along each axis 
            batch_action = 0.05 * (torch.rand((11, 16, 2), device = action.device) - 0.5)
            batch_action += action 
            
            batch_action[0] = action # first slot is the base 
            batch_action = torch.clip(batch_action, -1, 1)

            logits = model(batch_state, batch_action).detach().cpu().numpy()
        
        base_z_hat = logits[0]
        other_z_hats = logits[1:]
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
    

# ADAPTED 
def replay_with_prediction(model, mixed_dataset, output_dir, step, vae = False): # this plots latent predictions and their true values over time 
    # plot similarity of true end state and predicted end state 
    ce_loss = torch.nn.CrossEntropyLoss() 
    color_output = imageio.get_writer(output_dir + str(step) +  "visualized_predictions.mp4")
    print("Plotting Prediction Similarity")
    for selection in range(20):
        demo = random.randint(0, len(mixed_dataset.lengths_list) - 1)
        start, end = mixed_dataset.get_bounds_of_demo(demo)
        error_list = list()
        variance_list = list()
        for j in tqdm.tqdm(range(start, end)):
            sample = mixed_dataset.__getitem__(j)
            state, action, label = prepare_np(sample[0]), prepare_np(sample[1]), prepare_np(sample[2])
            # TODO: visualize actions 

            state = {k : torch.unsqueeze(v, dim = 0) for k, v in state.items()}
            action = torch.unsqueeze(action, dim = 0) # compensates for the batch dimension 
            # prediction = model(state, action).detach().cpu().numpy() 
            prediction = model(state, action).detach().cpu() 
            distribution_prediction = torch.nn.functional.softmax(prediction, dim = 1)[0] # (4, )

            image_visual = np.transpose(sample[0]["image"], (1, 2, 0))
            image_visual = cv2.resize(image_visual, (500, 500))
            
            # display bar graph visualization 
            canvas = np.zeros_like(image_visual)
            # ordering: ["Blue", "Red", "Green", "Yellow"] 
            color_ordering = [(0, 0, 255), (255, 0, 0), (0, 255, 0), (255, 255, 0)]
            bar_width = canvas.shape[0] // len(color_ordering)
            for i in range(len(color_ordering)):
                fill_pixels = int(distribution_prediction[i] * canvas.shape[0])
                canvas = cv2.rectangle(canvas, (0, i * bar_width), (fill_pixels, (i+1) * bar_width), color_ordering[i], -1)
            for i in range(action.shape[1]):
                scaled_action = ((action[0, i, 0] + 1) / 2, (action[0, i, 1] + 1) / 2)
                image_visual = cv2.circle(image_visual, (int(scaled_action[0] * canvas.shape[0]), int(scaled_action[1] * canvas.shape[1])), 5, (0, 255, 255), -1) 

            combined_frame = np.concatenate([image_visual, canvas], axis = 0)
            color_output.append_data(combined_frame)
    color_output.close()


# ADAPTED 
def plot_prediction_similarity(model, mixed_dataset, output_dir, step, vae = False): # this plots latent predictions and their true values over time 
    # plot similarity of true end state and predicted end state 
    ce_loss = torch.nn.CrossEntropyLoss() 
    print("Plotting Prediction Similarity")
    for selection in range(10):
        demo = random.randint(0, len(mixed_dataset.lengths_list) - 1)
        start, end = mixed_dataset.get_bounds_of_demo(demo)
        error_list = list()
        variance_list = list()
        for j in tqdm.tqdm(range(start, end)):
            sample = mixed_dataset.__getitem__(j)
            state, action, label = prepare_np(sample[0]), prepare_np(sample[1]), prepare_np(sample[2])
            state = {k : torch.unsqueeze(v, dim = 0) for k, v in state.items()}
            action = torch.unsqueeze(action, dim = 0) # compensates for the batch dimension 
            # prediction = model(state, action).detach().cpu().numpy() 
            prediction = model(state, action).detach().cpu() 
            error = ce_loss(prediction[0], label.detach().cpu())
            error_list.append(error)
            # error_list.append(np.mean(np.square(predicted_last_state_embed[0] - last_state_embed[0])))

        plt.plot(error_list)
        # we're doing this maybe later, which can visualize uncertainty. Right now it's not doing a good job 
            
    # plt.tight_layout()
    plt.title("End State Prediction")
    plt.xlabel("Step")
    plt.ylabel("Cross Entropy Prediction Error")
    plt.legend()
    plt.savefig(output_dir + str(step) + f"_end_prediction_error.png", dpi=300)
    plt.close()


def main(args):
    # this needs to be aligned with the action chunk length in the trained model 
    ACTION_DIM = 2
    # cameras = ["agentview_image", "robot0_eye_in_hand_image"] # you can change this; it's hardcoded
    cameras = ["image"] # you can change this; it's hardcoded
    padding = True
    
    state_vae = VAE(64)

    # print("I'M INTENTIONALLY NOT LOADI NG THE WEIGHTS!!")
    # print("ABLATION: USING MLP INSTEAD OF TRANSFORMER")
    model = FinalStateClassification(ACTION_DIM, args.action_chunk_length, cameras=cameras, state_vae = state_vae, classes = 4 )
    # model = FinalStateClassificationMLP(ACTION_DIM, args.action_chunk_length, cameras=cameras, state_vae = state_vae, classes = 4 )
    model.load_state_dict(torch.load(args.checkpoint))
    model.to("cuda")
    model.eval() 
    
    
    checkpoint_number = args.checkpoint.split("/")[-1].split(".")[0]
    mixed_dataset = MultiviewDataset(args.mixed_hdf5, action_chunk_length = args.action_chunk_length, cameras = cameras, padding = padding, mode = "classifier", 
                                     pad_mode = "repeat", hard_label = False)

    log_dir = args.exp_dir + "tests/"
    if not os.path.isdir(log_dir):
        os.mkdir(log_dir)
    # these are the tests that you can run 
    # compute_data_stack(mixed_dataset, log_dir, checkpoint_number)
    jiggle_action(model, mixed_dataset, log_dir, checkpoint_number) 
    visualize_action_shift(model, mixed_dataset, log_dir, checkpoint_number) 
    plot_prediction_similarity(model, mixed_dataset, log_dir, checkpoint_number)
    replay_with_prediction(model, mixed_dataset, log_dir, checkpoint_number)
    # compute_action_dependency(model, mixed_dataset, log_dir, checkpoint_number)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp_dir",
        type=str,
        default=None,
        help="",
    )
    # parser.add_argument(
    #     "--model_name",
    #     type=str,
    #     default=None,
    #     help="",
    # )
    parser.add_argument(
        "--mixed_hdf5",
        type=str,
        default=None,
        help="",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="",
    )
    parser.add_argument(
        "--action_chunk_length",
        type=int,
        default=None,
        help="",
    )
    args = parser.parse_args()

    main(args)