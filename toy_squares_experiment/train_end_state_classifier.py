from PIL import Image
import numpy as np
import torch
import h5py
import tqdm 
import numpy as np
import torch.nn as nn

from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt 
import robomimic.utils.file_utils as FileUtils
import argparse 
import os 
from torch.utils.tensorboard import SummaryWriter
import time 

from embedder_models import FinalStateClassification, FinalStateClassificationMLP
from embedder_datasets import MultiviewDataset 
from image_models import VAE

import torchvision 
import shutil 
import json 
import random 

# resizer = torchvision.transforms.Resize((224, 224))
# resizer = torchvision.transforms.Resize((200, 200))
mse_loss = torch.nn.MSELoss()
ce_loss = torch.nn.CrossEntropyLoss()

def sigmoid(z):
    return 1/(1 + np.exp(-z))

def kld_prior(mean, log_var):
    # return -0.5 * torch.sum(1+ log_var - mean.pow(2) - log_var.exp())
    return torch.mean(-0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp(), dim = 1), dim = 0)

# def make_filmstrip(current_state, reco_states, true_states, save_dir):
#     reco_states = resizer(reco_states) 
#     reco_states = reco_states.detach().cpu().numpy()
#     current_state = current_state.detach().cpu().numpy()
#     true_states = true_states.detach().cpu().numpy()
#     reco_states = np.transpose(reco_states, (0, 2, 3, 1))
#     true_states = np.transpose(true_states, (0, 2, 3, 1))
#     current_state = np.transpose(current_state, (0, 2, 3, 1))
#     reco_img = np.concatenate([reco_states[i] for i in range(reco_states.shape[0])], axis = 0) # this is terrible; i'm sorry
#     real_img = np.concatenate([true_states[i] for i in range(true_states.shape[0])], axis = 0) # this is terrible; i'm sorry
#     current_img = np.concatenate([current_state[i] for i in range(current_state.shape[0])], axis = 0) # this is terrible; i'm sorry
#     final_img = np.concatenate((current_img, reco_img, real_img), axis = 1)
#     plt.imsave(save_dir, final_img)


def get_valid_stats(model, sampler, generator, exp_dir, step, camera = "robot0_eye_in_hand_image"): 
    loss_count = 0

    info = {"overall" : 0, "ce_loss" : 0, "accuracy": 0}
    # for sample in tqdm.tqdm(dataset):
    # sample_generator = iter(dataset) # could this be problematic! TODO

    for j in tqdm.tqdm(range(50)):
        try:
            sample = next(generator)
        except StopIteration:
            print("wrapping around!")
            generator = iter(sampler)
            sample = next(generator)
        state, action, label = prepare(sample[0]), prepare(sample[1]), prepare(sample[2]) # there may not be a negative (in that case, it give syou none)
        # label = label.long()
        with torch.no_grad():
            prediction_logit = model(state, action)
            prediction_dist = torch.nn.functional.softmax(prediction_logit, dim = 1)
            class_loss = ce_loss(prediction_dist, label)
            info["ce_loss"] += class_loss.item()
            prediction_results = torch.argmax(prediction_logit, dim = 1) == torch.argmax(label, dim = 1)
            info["accuracy"] += torch.mean(prediction_results.float()).item()
        
            loss = class_loss 
            info["overall"] += loss.item() 
            
        loss_count += 1
    info = {k : v / loss_count for k, v in info.items()}
    print(f"Average Validation Losses: {info}")

    return info, generator

def save_scalar_stats(writer, info_dict, epoch, mode = "train"):
    for key, value in info_dict.items():
        writer.add_scalar(f"{mode}/{key}", value, epoch)

def prepare(data, device = "cuda"):
    if data is None:
        return None # for passthroughs 
    
    if type(data) == dict:
        return {k : v.to(device).to(torch.float32) for k, v in data.items()}
    return data.to(device).to(torch.float32)

def loss_scheduler(step, weight, mode = "linear", start = 0, end = 2000):
    if mode == "linear":
        return min(max(0, ((step - start) / (end - start)) * weight), weight)
    if mode == "sigmoid":
        # when this reaches 4, I assume that it's close to 1 
        if step < start:
            return 0
        return weight * sigmoid(-4 + 8 * max(0, ((step - start) / (end - start))))


def main(args):
    ACTION_DIM = 2
    CAMERA = "image" # "robot0_eye_in_hand_image"

    # cameras = ["agentview_image", "robot0_eye_in_hand_image"] # you can change this 
    cameras = [CAMERA] # you can change this 
    padding = True 
    hard_label = True # use the epsilon scaling for labels or not 

    if args.exp_dir is not None and not os.path.isdir(args.exp_dir):
        os.mkdir(args.exp_dir)
    shutil.copy("train_end_state_classifier.py", args.exp_dir + "/train_end_state_classifier.py") # because there are variants 
    shutil.copy("embedder_models.py", args.exp_dir + "/embedder_models.py") # because there are variants 
    shutil.copy("embedder_datasets.py", args.exp_dir + "/embedder_datasets.py") # because there are variants 
    with open(args.exp_dir + "/args.json", "w") as f:
        json.dump(vars(args), f)
    
    state_vae = VAE(64)
    model = FinalStateClassification(ACTION_DIM, args.action_chunk_length, cameras=cameras, state_vae = state_vae, classes = 4 )
    # model = FinalStateClassificationMLP(ACTION_DIM, args.action_chunk_length, cameras=cameras, state_vae = state_vae, classes = 4 )

    model.to("cuda")
    print(model.trainable_parameters())
    # torch.save(model.state_dict(), args.exp_dir + "initial.pth") #saves everything from the state dictionary

    dataset = MultiviewDataset(args.train_hdf5, action_chunk_length = args.action_chunk_length, cameras = cameras, padding = padding, mode = "classifier", 
                               pad_mode = "repeat", hard_label = hard_label)
    sampler = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, sampler=None,
            batch_sampler=None, num_workers=4, collate_fn=None,
            pin_memory=False, drop_last=False, timeout=0,
            worker_init_fn=None, # prefetch_factor=2,
            persistent_workers=False)
    
    
    valid_dataset = MultiviewDataset(args.test_hdf5, action_chunk_length = args.action_chunk_length, cameras = cameras, padding = padding, mode = "classifier", 
                                     pad_mode = "repeat", hard_label = hard_label)
    
    valid_sampler = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=True, sampler=None,
            batch_sampler=None, num_workers=4, collate_fn=None,
            pin_memory=False, drop_last=False, timeout=0,
            worker_init_fn=None, # prefetch_factor=2,
            persistent_workers=False)
    valid_generator = iter(valid_sampler)

    if args.noised:
        # print("I am augmenting the transformer with diffusion action noise!")
        # from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
        # noise_scheduler = DDPMScheduler(
        #         num_train_timesteps=100,
        #         beta_schedule="squaredcos_cap_v2",
        #         clip_sample=True,
        #         prediction_type="epsilon"
        # )

        print("I am augmenting the transformer with diffusion action noise!")
        from diffusers.schedulers.scheduling_ddim import DDIMScheduler
        noise_scheduler = DDIMScheduler(
                num_train_timesteps=100,
                beta_schedule="squaredcos_cap_v2",
                clip_sample=True,
                prediction_type="epsilon",
                steps_offset=0,
                set_alpha_to_one=True
        )
        # noise_scheduler.set_timesteps(10)


    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    writer = SummaryWriter(args.exp_dir) #you can specify logging directory

    model.eval()
    stats, valid_generator = get_valid_stats(model, valid_sampler, valid_generator, args.exp_dir, step = 0, camera = CAMERA)
    model.train()
    save_scalar_stats(writer, stats, 0, "valid")

    sample_generator = iter(sampler)

    for i in range(args.num_epochs):
        loss_count = 0
        info = {"overall" : 0, "ce_loss" : 0, "accuracy": 0}

        print(f"----------------Training Step {i}------------------")
        

        # this code is for unlocking the encoder later to improve the latent space 
        if i == 1000 and args.unlock:
            print("UNFREEZING VISION MODELS!")
            model.unfreeze()

        for j in tqdm.tqdm(range(100)):
            try:
                sample = next(sample_generator)
            except StopIteration:
                sample_generator = iter(sampler)
                sample = next(sample_generator)
            

            state, action, label = prepare(sample[0]), prepare(sample[1]), prepare(sample[2])

            if args.noised:
                # this prepares 
                m = torch.distributions.geometric.Geometric(0.05 * torch.ones(action.shape[0])) 
                timesteps = torch.clip(m.sample(), 0, 99).long() # this samples from a geometric distribtuion of expected value 20 
                noise = torch.randn(action.shape, device=action.device)
                # timesteps = torch.randint(0, 100, (action.shape[0],), device=action.device).long() # this samples uniformly 
                noised_action = noise_scheduler.add_noise(action, noise, timesteps)
                mask = torch.rand(action.shape[0]) > 0.5 
                action[mask] = noised_action[mask] # this just noises a batch 
            
            # label = label.long()
            prediction_logit = model(state, action)
            prediction_dist = torch.nn.functional.softmax(prediction_logit, dim = 1)
            class_loss = ce_loss(prediction_dist, label)
            info["ce_loss"] += class_loss.item()

            prediction_results = torch.argmax(prediction_logit, dim = 1) == torch.argmax(label, dim = 1)
            info["accuracy"] += torch.mean(prediction_results.float()).item()
        
            loss = class_loss
            info["overall"] += loss.item() 

            optimizer.zero_grad() #gradients add up, so you must reset
            loss.backward() #backpropagation. Put a vector into the backward() to compute the jacobian product
            optimizer.step() #applies change
            # print(loss.item())

            loss_count += 1

        info = {k : v / loss_count for k, v in info.items()}
        save_scalar_stats(writer, info, i, "train")

        print("Average training losses: ", info)        
        print("--------------Evaluating-----------------")
        # print("NOT DOING MODEL EVAL MODE")
        model.eval()
        # print("I'M USING TRAIN SET AS EVAL")
        stats, valid_generator = get_valid_stats(model, valid_sampler, valid_generator, args.exp_dir, step = i+1, camera = CAMERA)
        # stats, valid_generator = get_valid_stats(model, sampler, sample_generator, args.exp_dir, step = i+1, camera = CAMERA)
        save_scalar_stats(writer, stats, i + 1, "valid")
        if i % 100 == 0:
            torch.save(model.state_dict(), args.exp_dir + str(i) + ".pth") #saves everything from the state dictionary

        model.train()

        
if __name__ == "__main__":
    torch.set_printoptions(sci_mode = False)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp_dir",
        type=str,
        default=None,
        help="",
    )
    parser.add_argument(
        "--train_hdf5",
        type=str,
        default=None,
        help="",
    )

        # Whether to render rollouts to screen
    parser.add_argument(
        "--unlock",
        action='store_true',
        help="unlocks the encoder at a set time",
    )

    parser.add_argument(
        "--noised",
        action='store_true',
        help="unlocks the encoder at a set time",
    )
    # parser.add_argument(
    #     "--cameras",
    #     type=list,
    #     default=["agentview"],
    #     help="",
    # )
    parser.add_argument(
        "--test_hdf5",
        type=str,
        default=None,
        help="",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=None,
        help="",
    )
    parser.add_argument(
        "--action_chunk_length",
        type=int,
        default=None,
        help="",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="",
    )
    args = parser.parse_args()

    main(args)