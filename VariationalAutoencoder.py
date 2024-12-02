import torch
import pytorch_lightning as PL
import matplotlib.pyplot as plt
import torchvision
from torchvision import transforms
from torch import nn
from tqdm import tqdm
import torch.nn.functional as F
import os
from torchmetrics import Accuracy, F1Score


class VariationalAutoencoder(PL.LightningModule):
    def __init__(self, latent_dim = 64):
        super().__init__()
        self.latent_dim = latent_dim
        self.KL_coef = 0.5
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size = (3, 3), stride = 2, padding = 1),
            nn.LeakyReLU(),
            
            nn.Conv2d(64, 128, kernel_size = (3, 3), stride = 2, padding = 1),
            nn.LeakyReLU(),
            
            nn.Conv2d(128, 256, kernel_size = (3, 3), stride = 2, padding = 1),
            nn.LeakyReLU(),
            
            nn.Conv2d(256, 512, kernel_size = (3, 3), stride = 2, padding = 1),
            nn.LeakyReLU(),
            
            nn.Flatten(),
            nn.Linear(4*4*512, 512),
            nn.LeakyReLU()
        )
        
        self.get_mean = nn.Linear(512, self.latent_dim)
        self.get_logvar = nn.Linear(512, self.latent_dim)
        
        self.decoder_linear = nn.Sequential(
            nn.Linear(self.latent_dim, 512),
            nn.LeakyReLU(),
            
            nn.Linear(512, 4*4*512),
            nn.LeakyReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size = (3, 3), stride = 2, padding = 1, output_padding = 1, bias = False),
            nn.LeakyReLU(),
            
            nn.ConvTranspose2d(256, 128, kernel_size = (3, 3), stride = 2, padding = 1, output_padding = 1, bias = False),
            nn.LeakyReLU(),
            
            nn.ConvTranspose2d(128, 64, kernel_size = (3, 3), stride = 2, padding = 1, output_padding = 1, bias = False),
            nn.LeakyReLU(),
            
            nn.ConvTranspose2d(64, 3, kernel_size = (3, 3), stride = 2, padding = 1, output_padding = 1, bias = False),
            nn.Sigmoid()
        )
    
    def encode(self, x):
        return self.encoder(x)
        
    def decode(self, x):
        return self.decoder(self.decoder_linear(x).reshape(-1, 512, 4, 4))
    
    def sample(self, mean, logvar):
        epsilon = torch.randn_like(mean)
        return mean + torch.exp(0.5 * logvar) * epsilon
        
    def forward(self, x):
        out = self.encode(x)
        mean, logvar = self.get_mean(out), self.get_logvar(out)
        z = self.sample(mean, logvar)
        out = self.decode(z)
        return out, mean, logvar
    
    def vae_loss(self, fake_images, real_images, z_mean, z_log_var):
        reconstruction_loss = torch.mean(torch.square(real_images))
        kl_loss = -self.KL_coef * torch.mean(1 + z_log_var - torch.square(z_mean) - torch.exp(z_log_var))
        total_loss = reconstruction_loss + kl_loss
        return kl_loss, reconstruction_loss, total_loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr = 0.0003)
    
    def training_step(self, batch, batch_idx):
        x = batch
        x_recon, mu, log_var = self(x)
        recon_loss = F.binary_cross_entropy(x_recon.view(x.size(0), -1), x.view(x.size(0), -1), reduction='sum')
        kl_divergence = -self.KL_coef * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        loss = recon_loss + kl_divergence
        self.log_dict({'KL_Loss' : kl_divergence, 'reconstruction_loss' : recon_loss, 'total_loss':loss}, prog_bar = True, on_epoch = True)
        return loss