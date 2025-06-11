import numpy as np 
from tqdm import tqdm
import torch 
import random

def prepare_np(data, device = "cuda"):
    if type(data) == dict:
        return {k : torch.tensor(v).to(device).to(torch.float32) for k, v in data.items()}
    return torch.tensor(data).to(device).to(torch.float32)

def calculate_position_guidance(target_position, scale):
    """
    This function calculates position-based guidance as seen in this paper https://arxiv.org/abs/2411.16627
    """
    target_position_tensor = torch.tensor(target_position, device = "cuda", requires_grad = True)
    target_position_tensor = torch.unsqueeze(target_position_tensor, axis = 0)
    def guidance(states, actions): 
        DOWNSCALING = 40
        start_position = states["proprio"][:, -1, 0:3]
        gradient = torch.zeros_like(actions)
        cumulative_positions = torch.cumsum(actions[0, :, 0:3] / DOWNSCALING, axis = 0) + start_position  #T X 3 
        deltas = target_position_tensor - cumulative_positions 
        # print(start_position.cpu().numpy(), cumulative_positions[-1].detach().cpu().numpy())
        # print(deltas)
        gradient[0, :, 0:3] = deltas 
        gradient = scale * gradient 
        # print(deltas)
        
        return gradient 

    return guidance, None, None
  

def calculate_classifier_guidance(model, good_dataset, scale, main_camera, bad_dataset = None, alpha = 20, max_examples = None):
    # max_examples is for ablation test 
    good_embeddings_list = list()
    bad_embeddings_list = list()
    bad_embeddings = None 
    good_embeddings = None 
    print("Precomputing embeddings")
    idx = 0
    if good_dataset is not None:
        for length in tqdm(good_dataset.lengths_list):
            idx += length 
            sample = good_dataset.get_labeled_item(idx - 1) 
            state, action, label = prepare_np(sample[0]), prepare_np(sample[1]), sample[2]
            state = {k : torch.unsqueeze(v, dim = 0) for k, v in state.items()}
            action = torch.unsqueeze(action, dim = 0) # compensates for the batch dimension 
            with torch.no_grad():
                good_embedding = model.state_action_embedding(state, action).flatten(start_dim = 1) # gets the s, a embedding only 
                # good_embedding = model.state_embedding(state, normalize = False).flatten(start_dim = 1) # gets the s, a embedding only 
            good_embeddings_list.append(good_embedding.clone())
        good_embeddings = torch.concatenate(good_embeddings_list, dim = 0) # K X D
        if max_examples is not None: 
            print("LIMITING GOOD EMBEDDINGS TO", max_examples)
            good_embeddings = good_embeddings[:max_examples]

    idx = 0
    if bad_dataset is not None:
        for length in tqdm(bad_dataset.lengths_list):
            idx += length 
            sample = bad_dataset.get_labeled_item(idx - 1) 
            state, action, label = prepare_np(sample[0]), prepare_np(sample[1]), sample[2]
            state = {k : torch.unsqueeze(v, dim = 0) for k, v in state.items()}
            action = torch.unsqueeze(action, dim = 0) # compensates for the batch dimension 
            with torch.no_grad():
                bad_embedding = model.state_action_embedding(state, action).flatten(start_dim = 1) # gets the s, a embedding only 
                # bad_embedding = model.state_embedding(state, normalize = False).flatten(start_dim = 1) # gets the s, a embedding only 
            bad_embeddings_list.append(bad_embedding.clone())
        bad_embeddings = torch.concatenate(bad_embeddings_list, dim = 0) # K X D
    print("I AM USING SCALE ",  scale)
    print("I AM USING ALPHA ", alpha)
    
    def guidance(states, actions):
        # relevant_state = {MAIN_CAMERA : states[MAIN_CAMERA][:, -1] * 255}
        relevant_state = {main_camera : states[main_camera][:, -1] * 255}
        relevant_state["proprio"] = states["proprio"][:, -1]
        predicted_end = model.state_action_embedding(relevant_state, actions).flatten(start_dim = 1) # S X D 
        annealing_factor  = 1 #compute_anneal_weights(latent_prob_distance) # for legacy annealing approach 
        latent_prob_distance = 0

        if good_embeddings is not None:
            norm_pairwise_dist = torch.cdist(good_embeddings, predicted_end, p=2.0)

            # latent_prob_distance = -torch.mean(norm_pairwise_dist)
            # alpha = #20 #80 # used to be 20 for mixed tasks # to scale the distances away 
            latent_prob_distance = torch.logsumexp(-norm_pairwise_dist / alpha, dim = 0) #alpha * torch.logsumexp(-norm_pairwise_dist / alpha, dim = 0)
            # latent_prob_distance = -torch.min(norm_pairwise_dist) # find the closest

        if bad_embeddings is not None: # this adds the average negative guidance to the model 
            norm_pairwise_dist = torch.cdist(bad_embeddings, predicted_end, p=2.0)
            bad_dists = torch.logsumexp(-norm_pairwise_dist / alpha, dim = 0) #alpha * torch.logsumexp(-norm_pairwise_dist / alpha, dim = 0)

            # neg_annealing_factor = compute_negative_anneal_weights(bad_dists)
            # latent_prob_distance -= bad_dists 
            latent_prob_distance = latent_prob_distance - bad_dists # -= bad_dists 


        with torch.no_grad():
            gradient = torch.autograd.grad(latent_prob_distance, actions)[0]
        gradient = scale * gradient       
        gradient = annealing_factor * gradient 

        return gradient
    
    assert good_embeddings is not None or bad_embeddings is not None, "You provided no good or bad embeddings"
    return guidance, good_embeddings, bad_embeddings
