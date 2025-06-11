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
import argparse 
import os 
from torch.utils.tensorboard import SummaryWriter
import time 

from core.dynamics_models import FinalStatePredictionDino, FinalStatePredictionDinoCLS
from core.embedder_datasets import MultiviewDataset

import torchvision 
import shutil 
import json 
import random 


def make_filmstrip(current_state, reco_states, true_states, save_dir):
    """
    This code generates a "filmstrip" with three columns: current state, predicted future state, true future state.
    It is useful for monitoring the dynamic model's training progress 
    """
    imsize = current_state.shape[-1]
    resizer = torchvision.transforms.Resize((imsize, imsize))

    reco_states = resizer(reco_states) 
    reco_states = torch.clip(reco_states, 0, 1) # this makes it legal, although as the model gets better, it shouldn't need a lot of clipping 
    reco_states = reco_states.detach().cpu().numpy()
    current_state = current_state.detach().cpu().numpy()
    true_states = true_states.detach().cpu().numpy()
  
    reco_states = np.transpose(reco_states, (0, 2, 3, 1))
    true_states = np.transpose(true_states, (0, 2, 3, 1))
    current_state = np.transpose(current_state, (0, 2, 3, 1))
    reco_img = np.concatenate([reco_states[i] for i in range(reco_states.shape[0])], axis = 0) 
    real_img = np.concatenate([true_states[i] for i in range(true_states.shape[0])], axis = 0) 
    current_img = np.concatenate([current_state[i] for i in range(current_state.shape[0])], axis = 0) 
    final_img = np.concatenate((current_img, reco_img, real_img), axis = 1)
    plt.imsave(save_dir, final_img)


def get_valid_stats(model, sampler, generator, exp_dir, step, camera = "robot0_eye_in_hand_image"): 
    loss_count = 0
    embedding_list = list()
    mse_loss = torch.nn.MSELoss()

    info = {"overall" : 0, "mse_loss" : 0, "reco_loss" : 0}
    resizer = None 
    for j in tqdm.tqdm(range(50)): # takes 50 points for validation 
        try:
            sample = next(generator)
        except StopIteration:
            generator = iter(sampler)
            sample = next(generator)

        state, action, last_state = prepare(sample[0]), prepare(sample[1]), prepare(sample[2]) # there may not be a negative (in that case, it give syou none)
        kld_prior_loss = 0
        with torch.no_grad():      
            z_hat_last, reco_last = model(state, action) # the image is 200x200, we resize to 224 for dino 
            embedding_list.append(z_hat_last)
            if resizer is None:
                resizer = torchvision.transforms.Resize((last_state[camera].shape[-1], last_state[camera].shape[-1]))
           
            reco_loss = mse_loss(resizer(reco_last), last_state[camera] / 255) # inputs are at 0 to 255 but outputs are 0-1
            info["reco_loss"] += reco_loss.item()

            z_last = model.state_embedding(last_state)
            mse_loss_value = mse_loss(z_last, z_hat_last)
            info["mse_loss"] += mse_loss_value.item() 

            loss =  mse_loss_value + reco_loss
           

            info["overall"] += loss.item() 

            
        
        if loss_count % 50 == 0 and step % 40 == 0:
            print("Making filmstrip!")
            make_filmstrip(state[camera] / 255, reco_last, last_state[camera] / 255, exp_dir + f"/rc_{step}_{loss_count}.png")
 
        loss_count += 1

    cat_embed = torch.concatenate(embedding_list, dim = 0)
    variances = torch.std(cat_embed, dim = 0).detach().cpu().numpy() # batch variance
    info = {k : v / loss_count for k, v in info.items()}
    print(f"Average Validation Losses: {info}")
    info["min_var"] = np.min(variances)
    info["max_var"] = np.max(variances)
    print(f"Min batch variance is: {np.min(variances)} and maximum is {np.max(variances)}")
    z_hat_mean = z_hat_logvar = 0

    return info, variances, (z_hat_mean, z_hat_logvar), generator # return variances for logging purposes 

def save_scalar_stats(writer, info_dict, epoch, mode = "train"):
    for key, value in info_dict.items():
        writer.add_scalar(f"{mode}/{key}", value, epoch)

def prepare(data, device = "cuda"):
    if data is None:
        return None # for passthroughs 
    
    if type(data) == dict:
        return {k : v.to(device).to(torch.float32) for k, v in data.items()}
    return data.to(device).to(torch.float32)

def main(args):
    # ACTION_DIM = 7
    # CAMERA = "third_person"
    # cameras = [CAMERA] # you can potentially use multiple cameras although we only use one 
    # cameras = args.cameras 
    main_camera = args.cameras[0] # arbitrarily use the first camera for visualization 
    padding = True 
    pad_mode = "zeros" #"zeros"  # use "repeat" for absolute control, "zeros" for delta control to make it proper 

    # proprio_dim = 15 # enter the dimension of the proprioception you want to include 

    # proprio = "proprio" # the state key to use as the lowdim proprioception. Set to None if you want to exclude propriorception 

    if args.exp_dir is not None and not os.path.isdir(args.exp_dir):
        os.mkdir(args.exp_dir)
    with open(args.exp_dir + "/args.json", "w") as f: # saving the training parameters 
        json.dump(vars(args), f)

    model = FinalStatePredictionDino(args.action_dim, args.action_chunk_length, cameras=args.cameras, reconstruction = True, proprio = args.proprio_key, proprio_dim = args.proprio_dim)
    # model = FinalStatePredictionDinoCLS(ACTION_DIM, args.action_chunk_length, cameras=cameras, reconstruction = True, proprio = proprio, proprio_dim = proprio_dim) # alternative model 
    model.to("cuda")
    print(model.trainable_parameters())

    dataset = MultiviewDataset(args.train_hdf5, action_chunk_length = args.action_chunk_length, cameras = args.cameras, proprio = args.proprio_key,
        padding = padding, pad_mode = pad_mode)

    sampler = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, sampler=None,
            batch_sampler=None, num_workers=4, collate_fn=None,
            pin_memory=False, drop_last=False, timeout=0,
            worker_init_fn=None, # prefetch_factor=2,
            persistent_workers=False)
    sample_generator = iter(sampler)
    
    valid_dataset = MultiviewDataset(args.test_hdf5, action_chunk_length = args.action_chunk_length, cameras = args.cameras, proprio = args.proprio_key,
        padding = padding, pad_mode = pad_mode)

    valid_sampler = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=True, sampler=None,
            batch_sampler=None, num_workers=4, collate_fn=None,
            pin_memory=False, drop_last=False, timeout=0,
            worker_init_fn=None, # prefetch_factor=2,
            persistent_workers=False)
    valid_generator = iter(valid_sampler)

    if args.noised:
        from diffusers.schedulers.scheduling_ddim import DDIMScheduler
        noise_scheduler = DDIMScheduler(
                num_train_timesteps=100,
                beta_schedule="squaredcos_cap_v2",
                clip_sample=True,
                prediction_type="epsilon",
                steps_offset=0,
                set_alpha_to_one=True
        )

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    writer = SummaryWriter(args.exp_dir) #you can specify logging directory

    mse_loss = torch.nn.MSELoss()
    resizer = None 


    for i in range(args.num_epochs):
        info = {"overall" : 0, "mse_loss" : 0, "reco_loss" : 0} #this was misplaced 
        loss_count = 0
        print(f"----------------Training Step {i}------------------")
        for j in tqdm.tqdm(range(100)):
            try:
                sample = next(sample_generator)
            except StopIteration:
                sample_generator = iter(sampler)
                sample = next(sample_generator)
            

            state, action, last_state = prepare(sample[0]), prepare(sample[1]), prepare(sample[2])

            if args.noised:
                # this logic will add noise to the actions to simulate the noised action inputs during inference time 
                m = torch.distributions.geometric.Geometric(0.05 * torch.ones(action.shape[0])) 
                timesteps = torch.clip(m.sample(), 0, 99).long() # this samples from a geometric distribtuion of expected value 20 
                noise = torch.randn(action.shape, device=action.device)
                noised_action = noise_scheduler.add_noise(action, noise, timesteps)

                chance_of_mask = min(0.5, i / args.num_epochs) # this ensures a ramping effect of the noise operation 
                mask = torch.rand(action.shape[0]) < chance_of_mask 
                action[mask] = noised_action[mask] 
            
            # the actual dynamics model call 
            z_hat_last, reco_last = model(state, action)
            if resizer is None: # resizing the reconstruction so they can be compared 
                resizer = torchvision.transforms.Resize((last_state[main_camera].shape[-1], last_state[main_camera].shape[-1]))
           
            # computing model loss 
            reco_loss = mse_loss(resizer(reco_last), last_state[main_camera] / 255)
            info["reco_loss"] += reco_loss.item()

            z_last = model.state_embedding(last_state)
            mse_loss_value = mse_loss(z_last, z_hat_last)
            info["mse_loss"] += mse_loss_value.item() 

            loss =  mse_loss_value + reco_loss # recall that these two losses are not connected through gradient 

            info["overall"] += loss.item() 

            optimizer.zero_grad() #gradients add up, so you must reset
            loss.backward() #backpropagation. Put a vector into the backward() to compute the jacobian product
            optimizer.step() #applies change

            loss_count += 1

        info = {k : v / loss_count for k, v in info.items()}
        save_scalar_stats(writer, info, i, "train")

        print("Average training losses: ", info)  
        if i % 5 == 0:  # so we don't have to spend that much time evaluating something 
            print("--------------Evaluating-----------------")
            model.eval()
            stats, embeddings_std, (mean, logvar), valid_generator = get_valid_stats(model, valid_sampler, valid_generator, args.exp_dir, step = i,camera = main_camera)
            model.train()
            save_scalar_stats(writer, stats, i, "valid")
            writer.add_histogram("valid/embeddings_std", embeddings_std, i)
            writer.add_histogram("valid/mean", mean, i)
            writer.add_histogram("valid/logvar", logvar, i)
        if i % 100 == 0:
            torch.save(model.state_dict(), args.exp_dir + str(i) + ".pth") #saves everything from the state dictionary

        
if __name__ == "__main__":
    torch.set_printoptions(sci_mode = False)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp_dir",
        type=str,
        default=None,
        help="This is the experiment directory where everything is stored.",
    )
    parser.add_argument(
        "--train_hdf5",
        type=str,
        default=None,
        help="The data for training",
    )

    parser.add_argument(
        "--test_hdf5",
        type=str,
        default=None,
        help="The data for testing",
    )

    parser.add_argument(
        "--noised",
        action='store_true',
        help="enables noise augmentation for actions",
    )

    parser.add_argument(
        "--cameras",
        nargs='+',
        default=["third_person"],
        help="List of camera names",
    )


    parser.add_argument(
        "--num_epochs",
        type=int,
        default=None,
        help="Number of training steps x100 to run",
    )

    parser.add_argument(
        "--action_dim",
        type=int,
        default=None,
        help="How large the actions are",
    )
    
    parser.add_argument(
        "--proprio_key",
        type=str,
        default=None,
        help="The key to use in the states dict for proprio. Leave blank to ignore proprio",
    )

    parser.add_argument(
        "--proprio_dim",
        type=int,
        default=None,
        help="How large the proprioception is",
    )
    parser.add_argument(
        "--action_chunk_length",
        type=int,
        default=16,
        help="How long to take an action chunk",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="",
    )
    args = parser.parse_args()

    main(args)