import robomimic.utils.file_utils as FileUtils
import argparse 
import os 
import torch.nn as nn
from torchvision import transforms
import torch 
import numpy as np
import math 

from core.image_models import ResNet18Dec, VQVAE
import einops

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000, batch_first = True):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        self.batch_first = batch_first 

    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        if self.batch_first:
            x = torch.transpose(x, 0, 1)
        x = x + self.pe[:x.size(0)]
        if self.batch_first:
            x = torch.transpose(x, 0, 1)
        return self.dropout(x)

# this is for action embedding 
class ActionEmbedding(nn.Module):
    def __init__(
        self,
        num_frames=16, # horizon
        tubelet_size=1,
        in_chans=8, # action_dim
        emb_dim=384, # output_dim
        use_3d_pos=False # always False for now
    ):
        super().__init__()

        # Map input to predictor dimension
        self.num_frames = num_frames
        self.tubelet_size = tubelet_size
        self.in_chans = in_chans
        self.emb_dim = emb_dim

        output_dim = emb_dim // num_frames

        # just downsampling 
        self.patch_embed = nn.Conv1d(
            in_chans,
            output_dim,
            kernel_size=tubelet_size,
            stride=tubelet_size)
        self.out_project = nn.Linear(num_frames * output_dim, emb_dim)

        # TODO: can consider adding a 1d conv sequence at lower dimension 

        # self.conv_layers = nn.Sequential(
        #     nn.Conv1d(in_chans, emb_dim // 4, kernel_size=3, stride=2),  # [B, D, 16] → [B, emb_dim//4, 8]
        #     nn.ReLU(),
        #     nn.Conv1d(emb_dim // 4, emb_dim // 2, kernel_size=3, stride=2),  # [B, emb_dim//4, 8] → [B, emb_dim//2, 4]
        #     nn.ReLU(),
        #     nn.Conv1d(emb_dim // 2, emb_dim, kernel_size=2, stride=2),  # [B, emb_dim//2, 4] → [B, emb_dim, 2]
        #     nn.ReLU(),
        #     nn.Conv1d(emb_dim, emb_dim, kernel_size=2, stride=2)  # [B, emb_dim, 2] → [B, emb_dim, 1]
        # )

    def forward(self, x):
        # x: proprioceptive vectors of shape [B T D]
        x = x.permute(0, 2, 1) # [b, d, t]
        # x = self.conv_layers(x)
        x = self.patch_embed(x)
        x = einops.rearrange(x, "b d t -> b 1 (d t)")
        # x = x.permute(0, 2, 1)

        return x

"""
This is the main dynamics model of DynaGuide that uses the Dino patch embeddings as the latent space. 
The DINO-WM paper and codebase was referenced while designing this dynamics model. Go check them out: https://github.com/gaoyuezhou/dino_wm 
"""
class FinalStatePredictionDino(nn.Module):
    def __init__(self, action_dim, action_horizon, cameras, proprio = None, proprio_dim = None, reconstruction = True):
        super().__init__()
        chunk_size = 384 
        emb_dropout = 0 

        self.action_embedder = ActionEmbedding(num_frames = action_horizon, in_chans = action_dim, emb_dim = chunk_size)

        self.image_transform = transforms.Compose([ # assumes given 0 - 255 
            transforms.Resize((224, 224)),
            transforms.Normalize(
                mean=(123.675, 116.28, 103.53),
                std=(58.395, 57.12, 57.375),
            ), 
        ])

        self.state_encoder = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').to("cuda")

        self.proprio = proprio # how we add robot proprioception 
        if proprio is not None:
            assert proprio_dim is not None 
            self.pos_embedding = nn.Parameter(torch.randn(1, 258, chunk_size)) # learned position embeddings 
            self.proprio_embedder = nn.Linear(proprio_dim, chunk_size)

        else:
            self.pos_embedding = nn.Parameter(torch.randn(1, 257, chunk_size)) # learned position embeddings 

        # freezing and making the dino model into eval mode 
        for parameter in self.state_encoder.parameters():
            parameter.requires_grad = False # freezing the encoder 
        self.state_encoder.eval()

        # the main dynamics model 
        state_decoder_layer = nn.TransformerEncoderLayer(d_model=chunk_size, nhead=8, batch_first = True)# you can try 16 heads too 
        self.state_decoder_transformer = nn.TransformerEncoder(state_decoder_layer, num_layers=6) # you can try smaller if you have less data 
        self.dropout = nn.Dropout(emb_dropout)

        self.cameras = cameras 
        self.mask = torch.nn.Transformer().generate_square_subsequent_mask(action_horizon) # TODO: verify that this is correct 
        
        self.reconstruction = reconstruction 
        if reconstruction:
            # this model is *not* used to train the dynamics model; it is only for visualization. 
            self.reconstruction_model = VQVAE(in_channel = 3, channel = 384, n_res_block = 4, n_res_channel = 128, emb_dim = 128, quantize = False)

    def trainable_parameters(self): # counting the parameters in the network 
        count = 0 
        for parameter in self.parameters():
            if parameter.requires_grad:
                count += np.prod(parameter.size())
        return count 

    def compute_image_state_patches(self, state):
        """
        This function takes in an image and computes the dino embeddings, used internally. 
        """
        patch_list = list()
        for camera in self.cameras:
            transformed_state = self.image_transform(state[camera])
            embed = self.state_encoder.forward_features(transformed_state)["x_norm_patchtokens"]
            patch_list.append(embed) # batch, patch, dim 
        return torch.concatenate(patch_list, dim = 1) #batch, patches (sequential), dim

    def forward(self, states, actions): # takes in 0-255 image, regular action chunk       
        """
        This function takes (s, a) and outputs z-hat and the reconstructed image if present. 
        Input: state dict with image / proprioception, action 
        Output: z-hat vector 
        We use the forward() while training the dynamics model. To compute the state-action embeddings directly, see the functions below. 
        """
        B, S = actions.shape[0], actions.shape[1]

        image_embed = self.compute_image_state_patches(states) # [B, Patches, D]
        action_embed = self.action_embedder(actions) #[B, 1, D]
        if self.proprio is not None:
            proprio_embed = torch.unsqueeze(self.proprio_embedder(states[self.proprio]), axis = 1) # note that this is a bit jank but for experiments it's ok 
            combined_embed = torch.concatenate([image_embed, proprio_embed, action_embed], axis = 1)
        else:
            combined_embed = torch.concatenate([image_embed, action_embed], axis = 1)

        combined_embed += self.pos_embedding 
        predicted_state = self.state_decoder_transformer(combined_embed)

        predicted_z_end = predicted_state[:, :256] # The first 256 output tokens are the 16x16 predicted output patch. Ignore the others 
        
        if self.reconstruction:
            reco_image = self.image_reconstruct(predicted_z_end.detach()) # CRITICAL: the image reconstructor doesn't influence the actual dynamics model b/c detach 
            return predicted_z_end, reco_image 

        return predicted_z_end # returns zhat sequence and z sequence, then you can compute something like MSE loss 
    
    def image_reconstruct(self, embedding):
        """
        This function envokes the trained reconstruction model on a DINO embedding, useful for the reconstruction tests 
        """
        return self.reconstruction_model(embedding)

    def state_embedding(self, state, normalize = False):
        """
        This function takes an image state and returns the latent embedding. 
        This is useful for computing the guidance conditions. 
        """
        embed = self.compute_image_state_patches(state)
        return embed 
        

    def state_action_embedding(self, state, actions, normalize = False):
        """
        This function takes in (s, a) and computes z-hat, which is the dynamics mode. 
        This function is useful during DynaGuide to compute the z-hat that we use for the guidance signal. 
        """
        image_embed = self.compute_image_state_patches(state) # [B, Patches, D]
        action_embed = self.action_embedder(actions) #[B, 1, D]
        if self.proprio is not None:
            proprio_embed = torch.unsqueeze(self.proprio_embedder(state[self.proprio]), axis = 1) # note that this is a bit jank but for experiments it's ok 
            combined_embed = torch.concatenate([image_embed, proprio_embed, action_embed], axis = 1)

        else:
            combined_embed = torch.concatenate([image_embed, action_embed], axis = 1)

        combined_embed += self.pos_embedding 
        predicted_state = self.state_decoder_transformer(combined_embed)
        predicted_z_end = predicted_state[:, :256]  # The first 256 output tokens are the 16x16 predicted output patch. Ignore the others 
        return predicted_z_end 

"""
This is an alternative DynaGuide implementaion that uses the Dino CLS token as the embedding space. Because it doesn't perform as well,
we don't use it in the paper. However, the implementation is here as an other example of a dynamics model 
"""
class FinalStatePredictionDinoCLS(nn.Module):
    def __init__(self, action_dim, action_horizon, cameras, proprio = None, proprio_dim = None, reconstruction = True):
        super().__init__()
        repr_dim = 384

        assert proprio is None, "proprioception is not implemented for this model!"

        self.action_projector = nn.Sequential(
            nn.Linear(action_dim, repr_dim),
        )
        self.image_transform = transforms.Compose([ # assumes given 0 - 255 
            transforms.Resize((224, 224)),
            transforms.Normalize(
                mean=(123.675, 116.28, 103.53),
                std=(58.395, 57.12, 57.375),
            ), 
        ])

        self.state_encoder = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').to("cuda")

        for parameter in self.state_encoder.parameters():
            parameter.requires_grad = False # freezing the encoder Dino 


        self.prediction_token = torch.nn.Parameter(torch.zeros(1, repr_dim), requires_grad = True) #this is what we feed into the network for the final output prediction
        self.decoding_position_embedding = PositionalEncoding(repr_dim, max_len = action_horizon + 1, batch_first = True)
    
        state_decoder_layer = nn.TransformerDecoderLayer(d_model=repr_dim, nhead = 8, batch_first = True)
        self.state_decoder_transformer = nn.TransformerDecoder(state_decoder_layer, num_layers = 6)

        self.cameras = cameras 
        if reconstruction:
            self.decoder = ResNet18Dec(z_dim = repr_dim)


    def trainable_parameters(self):
        count = 0 
        for parameter in self.parameters():
            if parameter.requires_grad:
                count += np.prod(parameter.size())
        return count 

    def compute_image_state_patches(self, state):
        patch_list = list()
        for camera in self.cameras:
            transformed_state = self.image_transform(state[camera])
            embed = self.state_encoder.forward_features(transformed_state)["x_norm_clstoken"]
            patch_list.append(embed) # batch, patch, dim 
        return torch.concatenate(patch_list, dim = 1) #batch, patches (sequential), dim

    def forward(self, states, actions): # takes in 0-255 image, regular action chunk       
        embed = self.compute_image_state_patches(states)
        projected_actions = self.action_projector(actions) # B X S X embedding depth 
        projected_actions = self.decoding_position_embedding(projected_actions)
        prediction_token = torch.tile(self.prediction_token, dims = (embed.shape[0], 1)) #tiling for batch 
        prediction_token = torch.unsqueeze(prediction_token, dim = 1)
        projected_actions = torch.concatenate((projected_actions, prediction_token), dim = 1)
        embed = torch.unsqueeze(embed, dim = 1)
        predicted_final_state = self.state_decoder_transformer(projected_actions, memory = embed)[:, -1] # , tgt_mask=self.mask, tgt_is_causal = True)[:, -1] # causal transformer 
        reco_final_state = self.decoder(predicted_final_state.detach()) # CRITICAL: NOT PASSSING GRADIENT TO RECONSTRUCTOR
        return predicted_final_state, reco_final_state # returns zhat sequence and z sequence, then you can compute something like MSE loss 

    
    def image_reconstruct(self, embedding):
        return self.decoder(embedding)

    def state_embedding(self, state, normalize = False):
        """
        This function takes an image state and returns the latent embedding. 
        This is useful for computing the guidance conditions. 
        """
        s_embedding = self.compute_image_state_patches(state)
        if normalize: 
            return torch.nn.functional.normalize(s_embedding, dim = 1)
        return s_embedding

    def state_action_embedding(self, state, actions): 
        """
        This function takes in (s, a) and computes z-hat, which is the dynamics mode. 
        This function is useful during DynaGuide to compute the z-hat that we use for the guidance signal. 
        """
        embed = self.compute_image_state_patches(state)
        projected_actions = self.action_projector(actions) # B X S X D
        projected_actions = self.decoding_position_embedding(projected_actions)
        prediction_token = torch.tile(self.prediction_token, dims = (embed.shape[0], 1)) #tiling for batch 
        prediction_token = torch.unsqueeze(prediction_token, dim = 1)
        projected_actions = torch.concatenate((projected_actions, prediction_token), dim = 1)
        embed = torch.unsqueeze(embed, dim = 1)
        predicted_final_state = self.state_decoder_transformer(projected_actions, memory = embed)[:, -1] # , tgt_mask=self.mask, tgt_is_causal = True)[:, -1] # causal transformer 
        return predicted_final_state
       
"""
This class is a dynamics model for the toy squares experiment that directly classifies based on square color 
"""
class FinalStateClassification(nn.Module):
    def __init__(self, action_dim, action_horizon, cameras, state_vae, classes):
        super().__init__()
        repr_dim = 64

        self.action_projector = nn.Sequential(
            nn.Linear(action_dim, repr_dim),
            nn.ReLU(),
            nn.Linear(repr_dim, repr_dim),
        )

        self.state_vae = state_vae # must be a loaded model 

        # if you want to start the model with a loaded vae, this is how to freeze it 
        # print("Froze ", self.state_vae.trainable_parameters(), " parameters")
        # for parameter in self.state_vae.parameters():
        #     parameter.requires_grad = False # freezing the encoder 


        self.prediction_token = torch.nn.Parameter(torch.zeros(1, repr_dim), requires_grad = True) #this is what we feed into the network for the final output prediction
        self.decoding_position_embedding = PositionalEncoding(repr_dim, max_len = action_horizon + 1, batch_first = True)
     
        state_decoder_layer = nn.TransformerDecoderLayer(d_model=repr_dim, nhead = 4, batch_first = True)
        self.state_decoder_transformer = nn.TransformerDecoder(state_decoder_layer, num_layers = 4)

        self.prediction_head = nn.Linear(repr_dim, classes)

        self.cameras = cameras 

    def unfreeze(self): # unfreezes the model if you decided to freeze the encoder at the start 
        print("UNFREEZING the encoder!")
        for parameter in self.state_vae.parameters():
            parameter.requires_grad = True # freezing the encoder 
        print("Unfroze ", self.state_vae.trainable_parameters(), " parameters")
       
    def trainable_parameters(self): # calculates the number of parameters 
        count = 0 
        for parameter in self.parameters():
            if parameter.requires_grad:
                count += np.prod(parameter.size())
        return count 

    def compute_image_state_patches(self, state): # run the encoder 
        patch_list = list()
        for camera in self.cameras:
            assert torch.max(state[camera]) > 1, "you are feeding in an already-normalized image" # sanity check 
            transformed_state = state[camera] / 255 # assuming that the camera is 0-255 
            embed = self.state_vae.encode(transformed_state)
            patch_list.append(embed) # batch, patch, dim 
        return torch.concatenate(patch_list, dim = 1) #batch, patches (sequential), dim

    def forward(self, states, actions): 
        """
        This function takes in (s,a) and outputs category prediction. It is the main function we call on the dynamics model for both training and inference-time. 
        Input: 0-255 image, regular action chunk  
        Output: color prediction logit vector 
        """
        embed = self.compute_image_state_patches(states)
        projected_actions = self.action_projector(actions) # B X S X 128 
        projected_actions = self.decoding_position_embedding(projected_actions)
        prediction_token = torch.tile(self.prediction_token, dims = (embed.shape[0], 1)) #tiling for batch 
        prediction_token = torch.unsqueeze(prediction_token, dim = 1)
        projected_actions = torch.concatenate((projected_actions, prediction_token), dim = 1)
        embed = torch.unsqueeze(embed, dim = 1)
        predicted_latent = self.state_decoder_transformer(projected_actions, memory = embed)[:, -1] # , tgt_mask=self.mask, tgt_is_causal = True)[:, -1] # causal transformer 
        output_logit = self.prediction_head(predicted_latent)

        return output_logit 