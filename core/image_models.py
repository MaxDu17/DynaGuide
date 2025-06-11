import torch
from torch import nn, optim
import torch.nn.functional as F
import torchvision 
import numpy as np 
from torchvision.transforms import v2

class ResizeConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, scale_factor, mode='nearest'):
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        x = self.conv(x)
        return x

class BasicBlockEnc(nn.Module):

    def __init__(self, in_planes, stride=1):
        super().__init__()

        planes = in_planes*stride

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        if stride == 1:
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out

class BasicBlockDec(nn.Module):

    def __init__(self, in_planes, stride=1):
        super().__init__()

        planes = int(in_planes/stride)

        self.conv2 = nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(in_planes)
        # self.bn1 could have been placed here, but that messes up the order of the layers when printing the class

        if stride == 1:
            self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(planes)
            self.shortcut = nn.Sequential()
        else:
            self.conv1 = ResizeConv2d(in_planes, planes, kernel_size=3, scale_factor=stride)
            self.bn1 = nn.BatchNorm2d(planes)
            self.shortcut = nn.Sequential(
                ResizeConv2d(in_planes, planes, kernel_size=3, scale_factor=stride),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = torch.relu(self.bn2(self.conv2(x)))
        out = self.bn1(self.conv1(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out

class ResNet18Enc(nn.Module):

    def __init__(self, num_Blocks=[2,2,2,2], z_dim=10, nc=3):
        super().__init__()
        self.in_planes = 64
        self.z_dim = z_dim
        self.conv1 = nn.Conv2d(nc, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(BasicBlockEnc, 64, num_Blocks[0], stride=1)
        self.layer2 = self._make_layer(BasicBlockEnc, 128, num_Blocks[1], stride=2)
        self.layer3 = self._make_layer(BasicBlockEnc, 256, num_Blocks[2], stride=2)
        self.layer4 = self._make_layer(BasicBlockEnc, 512, num_Blocks[3], stride=2)
        self.linear = nn.Linear(512, 2 * z_dim)

    def _make_layer(self, BasicBlockEnc, planes, num_Blocks, stride):
        strides = [stride] + [1]*(num_Blocks-1)
        layers = []
        for stride in strides:
            layers += [BasicBlockEnc(self.in_planes, stride)]
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        mu = x[:, :self.z_dim]
        logvar = x[:, self.z_dim:]
        return mu, logvar


class ResNet18EncLite(nn.Module):

    def __init__(self, num_Blocks=[2,2,2,2], z_dim=10, nc=3):
        super().__init__()
        self.in_planes = 8
        self.z_dim = z_dim
        self.conv1 = nn.Conv2d(nc, 8, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(8)
        self.layer0_1 = self._make_layer(BasicBlockEnc, 8, num_Blocks[0], stride=1)
        self.layer0_2 = self._make_layer(BasicBlockEnc, 8, num_Blocks[0], stride=1)
        self.layer1 = self._make_layer(BasicBlockEnc, 16, num_Blocks[0], stride=2)
        self.layer2 = self._make_layer(BasicBlockEnc, 32, num_Blocks[1], stride=2)
        self.layer3 = self._make_layer(BasicBlockEnc, 64, num_Blocks[2], stride=2)
        self.layer4 = self._make_layer(BasicBlockEnc, 128, num_Blocks[3], stride=2)
        self.linear = nn.Linear(128, 2 * z_dim)

    def _make_layer(self, BasicBlockEnc, planes, num_Blocks, stride):
        strides = [stride] + [1]*(num_Blocks-1)
        layers = []
        for stride in strides:
            layers += [BasicBlockEnc(self.in_planes, stride)]
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.layer0_1(x)
        x = self.layer0_2(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = F.adaptive_avg_pool2d(x, 1) 
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        mu = x[:, :self.z_dim]
        logvar = x[:, self.z_dim:]
        return mu, logvar

class ResNet18Dec(nn.Module):

    def __init__(self, num_Blocks=[2,2,2,2], z_dim=10, nc=3):
        super().__init__()
        self.in_planes = 512

        self.linear = nn.Linear(z_dim, 512)

        self.layer4 = self._make_layer(BasicBlockDec, 256, num_Blocks[3], stride=2)
        self.layer3 = self._make_layer(BasicBlockDec, 128, num_Blocks[2], stride=2)
        self.layer2 = self._make_layer(BasicBlockDec, 64, num_Blocks[1], stride=2)
        # self.layer1_5 = self._make_layer(BasicBlockDec, 32, num_Blocks[1], stride=2)
        # self.layer1 = self._make_layer(BasicBlockDec, 32, num_Blocks[1], stride=2)
        self.layer0 = self._make_layer(BasicBlockDec, 32, num_Blocks[0], stride=2)
        self.conv1 = ResizeConv2d(32, nc, kernel_size=3, scale_factor=2)

    def _make_layer(self, BasicBlockDec, planes, num_Blocks, stride):
        strides = [stride] + [1]*(num_Blocks-1)
        layers = []
        for stride in reversed(strides):
            layers += [BasicBlockDec(self.in_planes, stride)]
        self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, z):
        x = self.linear(z)
        x = x.view(z.size(0), 512, 1, 1)
        x = F.interpolate(x, scale_factor=4)
        x = self.layer4(x)
        x = self.layer3(x)
        x = self.layer2(x)
        # x = self.layer1_5(x)
        # x = self.layer1(x)
        x = self.layer0(x)
        x = torch.sigmoid(self.conv1(x))
        x = x.view(x.size(0), 3, 128, 128)
        return x


# class ResNet18DecPatch(nn.Module):
#     def __init__(self, num_Blocks=[2,2,2,2], nc=3):
#         super().__init__()
#         self.in_planes = 384

#         # self.linear = nn.Linear(z_dim, 512)
#         self.layer_5 = 
#         self.layer4 = self._make_layer(BasicBlockDec, 192, num_Blocks[3], stride=2)
#         self.layer3 = self._make_layer(BasicBlockDec, 96, num_Blocks[2], stride=2)
#         self.layer2 = self._make_layer(BasicBlockDec, 48, num_Blocks[1], stride=2)
#         # self.layer1_5 = self._make_layer(BasicBlockDec, 32, num_Blocks[1], stride=2)
#         # self.layer1 = self._make_layer(BasicBlockDec, 24, num_Blocks[1], stride=2)
#         self.layer0 = self._make_layer(BasicBlockDec, 24, num_Blocks[0], stride=2)
#         self.conv1 = ResizeConv2d(24, nc, kernel_size=3, scale_factor=2)

#     def _make_layer(self, BasicBlockDec, planes, num_Blocks, stride):
#         strides = [stride] + [1]*(num_Blocks-1)
#         layers = []
#         for stride in reversed(strides):
#             layers += [BasicBlockDec(self.in_planes, stride)]
#         self.in_planes = planes
#         return nn.Sequential(*layers)

#     def forward(self, z):
#         # x = self.linear(z)
#         # x = x.view(z.size(0), 512, 1, 1)
#         num_patches = z.shape[1]
#         num_side_patches = int(num_patches ** 0.5)    
#         z_fed = rearrange(z, "b (h w) e -> b h w e", h=num_side_patches, w=num_side_patches)
#         z_fed = z_fed.permute(0, 3, 1, 2)
#         # x = F.interpolate(x, scale_factor=4)
#         import ipdb 
#         ipdb.set_trace()
#         x = self.layer4(z_fed)
#         x = self.layer3(x)
#         x = self.layer2(x)
#         # x = self.layer1_5(x)
#         # x = self.layer1(x)
#         x = self.layer0(x)
#         x = torch.sigmoid(self.conv1(x))
#         x = x.view(x.size(0), 3, 128, 128)
#         return x


class ResNet18DecLite(nn.Module):

    def __init__(self, num_Blocks=[2,2,2,2], z_dim=10, nc=3):
        super().__init__()
        self.in_planes = 128

        self.linear = nn.Linear(z_dim, 128)

        self.layer4 = self._make_layer(BasicBlockDec, 64, num_Blocks[3], stride=2)
        self.layer3 = self._make_layer(BasicBlockDec, 32, num_Blocks[2], stride=2)
        self.layer2 = self._make_layer(BasicBlockDec, 16, num_Blocks[1], stride=2)
        # self.layer1_5 = self._make_layer(BasicBlockDec, 32, num_Blocks[1], stride=2)
        # self.layer1 = self._make_layer(BasicBlockDec, 32, num_Blocks[1], stride=2)
        self.layer0 = self._make_layer(BasicBlockDec, 8, num_Blocks[0], stride=2)
        self.layer0_1 = self._make_layer(BasicBlockDec, 8, num_Blocks[0], stride=1)
        self.layer0_2 = self._make_layer(BasicBlockDec, 8, num_Blocks[0], stride=1)
        self.conv1 = ResizeConv2d(8, nc, kernel_size=3, scale_factor=2)

    def _make_layer(self, BasicBlockDec, planes, num_Blocks, stride):
        strides = [stride] + [1]*(num_Blocks-1)
        layers = []
        for stride in reversed(strides):
            layers += [BasicBlockDec(self.in_planes, stride)]
        self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, z):
        x = self.linear(z)
        x = x.view(z.size(0), 128, 1, 1)
        x = F.interpolate(x, scale_factor=4)
        x = self.layer4(x)
        x = self.layer3(x)
        x = self.layer2(x)
        x = self.layer0(x)
        x = self.layer0_1(x)
        x = self.layer0_2(x)
        x = self.conv1(x)
        # x = torch.sigmoid(self.conv1(x))
        print("no sigmoid!")
        x = x.view(x.size(0), 3, 128, 128)
        return x


class ResNet18Torch(nn.Module):
    def __init__(self, z_dim, pretrained = False):
        super().__init__()
        self.model = torchvision.models.resnet18(pretrained = pretrained)
        self.pretrained = pretrained 
        if pretrained: # try freezing 
            for parameter in self.model.parameters():
                parameter.requires_grad = False # freezing the encoder 

        self.model.fc = nn.Linear(512, 2 * z_dim)
        self.z_dim = z_dim 
        self.resizer = torchvision.transforms.Resize((224, 224))
        self.normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

    def forward(self, x):
        x = self.resizer(x)
        if self.pretrained:
            x = self.normalize(x)
        out = self.model(x)
        mu = out[:, :self.z_dim]
        logvar = out[:, self.z_dim:]
        return mu, logvar

class VAE(nn.Module):
    def __init__(self, z_dim):
        super().__init__()
        # self.encoder = ResNet18EncLite(z_dim=z_dim) # TODO: older versoins used this. If you're seeing an error, enable this one 
        # self.encoder = ResNet18Enc(z_dim=z_dim) # TODO: older versoins used this. If you're seeing an error, enable this one 

        self.encoder = ResNet18Torch(z_dim = z_dim, pretrained=False) 

        # self.decoder = VQVAE(in_channel = 3, channel = z_dim, n_res_block = 4, n_res_channel = 128, emb_dim = 128, quantize = False)

        self.decoder = ResNet18Dec(z_dim=z_dim)
        # self.decoder = ResNet18DecLite(z_dim=z_dim)
        train_transforms = v2.Compose([
            v2.GaussianNoise(mean = 0, sigma = 0.05), # used to be 0.02 
            v2.ColorJitter(brightness = 0.2, contrast = 0.2, saturation = 0.1, hue = 0),
            # v2.RandomCrop(size=(224, 224), pad_if_needed = True)
            v2.RandomAffine(degrees=(-10, 10), translate=(0, 0.1), scale=(0.95, 1.05))
        ])
        self.train_transforms = v2.Lambda(lambda x: torch.stack([train_transforms(x_) for x_ in x])) # this applys transformations individualy

    def forward(self, x):
        assert torch.max(x) < 1.1, "You are probably feeding in the wrong data!"
        if self.training: 
            # print("I shouldn't be here!")
            x = self.train_transforms(x)
        mean, logvar = self.encoder(x)
        # print("TEMPORARY: NOT A VAE ANYMORE") # TODO: 
        z = self.reparameterize(mean, logvar)
        # z = mean 
        # z = z.unsqueeze(1).repeat(1, 16, 1)  # Resulting shape: N x K x D
        x = self.decoder(z)
        return x, mean, logvar
    
    def encode(self, x):
        assert torch.max(x) < 1.1, "You are probably feeding in the wrong data!"
        mean, logvar = self.encoder(x)
        return mean 
    
    def decode(self, z):
        return self.decoder(z)
    
    def trainable_parameters(self):
        count = 0 
        for parameter in self.parameters():
            if parameter.requires_grad:
                count += np.prod(parameter.size())
        return count 
    
    @staticmethod
    def reparameterize(mean, logvar):
        std = torch.exp(logvar / 2) # in log-space, squareroot is divide by two
        epsilon = torch.randn_like(std)
        return epsilon * std + mean
    
# class PretrainedVAE(nn.Module):
#     def __init__(self):
#         super().__init__()
#         from diffusers.models import AutoencoderKL
#         self.vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse")
#         self.resizer = torchvision.transforms.Resize((256, 256))
#         self.down_resizer = torchvision.transforms.Resize((224, 224))
#         # self.normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
#         #                          std=[0.229, 0.224, 0.225])

#     def forward(self, x):
#         # convert from 0->1 to -1 -> 1 
#         # TODO: I'M USING THE WRONG NORMALIZATION I THINK
#         # return 0.5 * (self.down_resizer(self.vae(self.resizer((x - 0.5) * 2)).sample) / 2)
#         return self.down_resizer(self.vae(self.resizer(x)).sample)
#         # return self.down_resizer(self.vae(self.resizer(self.normalize(x))).sample)
        
#     def encode(self, x):
#         return torch.flatten(self.vae.encode(self.resizer(x))["latent_dist"].mean, start_dim = 1)
        
#     def decode(self, x):
#         B = x.shape[0]
#         x = x.view(B, 4, 32, 32)
#         return self.down_resizer(self.vae.decode(x).sample)

# model = VAE(128)
# img = torch.ones((8, 3, 224, 224))
# latent = torch.ones((8, 128))
# out = model.decoder(latent)
# print(out.shape)

############ VQVAE DECODER FROM DINO-WM ################
import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange

# Copyright 2018 The Sonnet Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================


# Borrowed from https://github.com/deepmind/sonnet and ported it to PyTorch


class Quantize(nn.Module):
    def __init__(self, dim, n_embed, decay=0.99, eps=1e-5):
        super().__init__()

        self.dim = dim
        self.n_embed = n_embed
        self.decay = decay
        self.eps = eps

        embed = torch.randn(dim, n_embed)
        self.register_buffer("embed", embed)
        self.register_buffer("cluster_size", torch.zeros(n_embed))
        self.register_buffer("embed_avg", embed.clone())

    def forward(self, input):
        flatten = input.reshape(-1, self.dim)
        import ipdb 
        ipdb.set_trace()

        dist = (
            flatten.pow(2).sum(1, keepdim=True)
            - 2 * flatten @ self.embed
            + self.embed.pow(2).sum(0, keepdim=True)
        )
        _, embed_ind = (-dist).max(1)
        embed_onehot = F.one_hot(embed_ind, self.n_embed).type(flatten.dtype)
        embed_ind = embed_ind.view(*input.shape[:-1])
        quantize = self.embed_code(embed_ind)

        if self.training:
            embed_onehot_sum = embed_onehot.sum(0)
            embed_sum = flatten.transpose(0, 1) @ embed_onehot

            self.cluster_size.data.mul_(self.decay).add_(
                embed_onehot_sum, alpha=1 - self.decay
            )
            self.embed_avg.data.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)
            n = self.cluster_size.sum()
            cluster_size = (
                (self.cluster_size + self.eps) / (n + self.n_embed * self.eps) * n
            )
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(0)
            self.embed.data.copy_(embed_normalized)

        diff = (quantize.detach() - input).pow(2).mean()
        quantize = input + (quantize - input).detach()

        return quantize, diff, embed_ind

    def embed_code(self, embed_id):
        return F.embedding(embed_id, self.embed.transpose(0, 1))


class ResBlock(nn.Module):
    def __init__(self, in_channel, channel):
        super().__init__()

        self.conv = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channel, channel, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, in_channel, 1),
        )

    def forward(self, input):
        out = self.conv(input)
        out += input

        return out


class Encoder(nn.Module):
    def __init__(self, in_channel, channel, n_res_block, n_res_channel, stride):
        super().__init__()

        if stride == 4:
            blocks = [
                nn.Conv2d(in_channel, channel // 2, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 2, channel, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel, channel, 3, padding=1),
            ]

        elif stride == 2:
            blocks = [
                nn.Conv2d(in_channel, channel // 2, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 2, channel, 3, padding=1),
            ]

        for i in range(n_res_block):
            blocks.append(ResBlock(channel, n_res_channel))

        blocks.append(nn.ReLU(inplace=True))

        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        return self.blocks(input)


class Decoder(nn.Module):
    def __init__(
        self, in_channel, out_channel, channel, n_res_block, n_res_channel, stride
    ):
        # in channel: what we expect the data to be 
        # channel: latent diensions 
        # out channel: what we expect the output to have 
        super().__init__()

        blocks = [nn.Conv2d(in_channel, channel, 3, padding=1)]
        for i in range(n_res_block):
            blocks.append(ResBlock(channel, n_res_channel))

        blocks.append(nn.ReLU(inplace=True))

        if stride == 4:
            blocks.extend(
                [
                    nn.ConvTranspose2d(channel, channel // 2, 4, stride=2, padding=1),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(
                        channel // 2, out_channel, 4, stride=2, padding=1
                    ),
                ]
            )

        elif stride == 2:
            blocks.append(
                nn.ConvTranspose2d(channel, out_channel, 4, stride=2, padding=1)
            )

        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        return self.blocks(input)


# taken from Dino-WM codebase 
class VQVAE(nn.Module):
    def __init__(
        self,
        in_channel=3, # this is the output channel 
        channel=128, # this is the input channel 
        n_res_block=2,
        n_res_channel=32, # this is the latent for resnet 
        emb_dim=64, # this is the expected embedding of the input 
        n_embed=512, # this is number of embedding vectors 
        decay=0.99,
        quantize=True,
    ):
        super().__init__()

        self.quantize = quantize
        self.quantize_b = Quantize(emb_dim, n_embed)

        if not quantize:
            for param in self.quantize_b.parameters():
                param.requires_grad = False
        # input, desired output, latent 
        self.upsample_b = Decoder(channel, emb_dim, emb_dim, n_res_block, n_res_channel, stride=4)
        self.dec = Decoder(
            emb_dim,
            in_channel,
            emb_dim,
            n_res_block,
            n_res_channel,
            stride=4,
        )
        self.info = f"in_channel: {in_channel}, channel: {channel}, n_res_block: {n_res_block}, n_res_channel: {n_res_channel}, emb_dim: {emb_dim}, n_embed: {n_embed}, decay: {decay}"

    def forward(self, input):
        '''
            input: (b, t, num_patches, emb_dim)
        '''
        num_patches = input.shape[1]
        num_side_patches = int(num_patches ** 0.5)    
        input = rearrange(input, "b (h w) e -> b h w e", h=num_side_patches, w=num_side_patches)

        if self.quantize:
            quant_b, diff_b, id_b = self.quantize_b(input)
        else:
            quant_b, diff_b = input, torch.zeros(1).to(input.device)

        quant_b = quant_b.permute(0, 3, 1, 2)
        diff_b = diff_b.unsqueeze(0)
        dec = self.decode(quant_b)
        return dec #, diff_b # diff is 0 if no quantization, WE DO NOT USE QUANTIZATION

    def decode(self, quant_b):
        upsample_b = self.upsample_b(quant_b) 
        dec = self.dec(upsample_b) # quant: (128, 64, 64)
        return dec