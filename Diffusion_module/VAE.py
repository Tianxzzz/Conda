
# from torch.nn.init import xavier_normal_, constant_, xavier_uniform_
   
import numpy as np
import math
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.nn.init import xavier_normal_, constant_

class VAE(nn.Module):

    def __init__(self, input_dim, hidden_dims, latent_dim, act_func='relu', dropout=0.1):
        super(VAE, self).__init__()

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.latent_dim = latent_dim
        self.act_func = act_func
        self.dropout = nn.Dropout(dropout)

        # Build Encoder
        encoder_modules = []

        encoder_modules.append(nn.Linear(self.input_dim, self.hidden_dims))
        if act_func == 'relu':
            encoder_modules.append(nn.ReLU())
        elif act_func == 'sigmoid':
            encoder_modules.append(nn.Sigmoid())
        elif act_func == 'tanh':
            encoder_modules.append(nn.Tanh())
        else:
            raise ValueError("Unsupported activation function")
        
        self.encoder = nn.Sequential(*encoder_modules)
        self.fc_mu = nn.Linear(self.hidden_dims, self.latent_dim)
        self.fc_logvar = nn.Linear(self.hidden_dims, self.latent_dim)

        # Build Decoder
        decoder_modules = []
        
        decoder_modules.append(nn.Linear(self.latent_dim, self.hidden_dims))
        if act_func == 'relu':
            decoder_modules.append(nn.ReLU())
        elif act_func == 'sigmoid':
            decoder_modules.append(nn.Sigmoid())
        elif act_func == 'tanh':
            decoder_modules.append(nn.Tanh())
        else:
            raise ValueError("Unsupported activation function")

        decoder_modules.append(nn.Linear(self.hidden_dims, self.input_dim))
        decoder_modules.append(nn.Sigmoid())  

        self.decoder = nn.Sequential(*decoder_modules)

        self.apply(xavier_normal_initialization)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, x):
        x = self.dropout(x)
        
        hidden = self.encoder(x)
    
        mu = self.fc_mu(hidden)
        logvar = self.fc_logvar(hidden)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar


    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
      
        z, mu, logvar = self.encode(x)

        recon_x = self.decode(z)
        return recon_x, mu, logvar

def xavier_normal_initialization(m):
    if isinstance(m, nn.Linear):
        xavier_normal_(m.weight)
        if m.bias is not None:
            constant_(m.bias, 0)


def compute_vae_loss(recon_x, x, mu, logvar):
    # Reconstruction loss (Binary Cross-Entropy)
    recon_loss = F.binary_cross_entropy(recon_x, x, reduction='sum')
    # KL Divergence
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_loss
     
                