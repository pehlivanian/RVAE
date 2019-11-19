import math
import torch
import torch.nn as nn
import torch.utils
import torch.utils.data
from torchvision import datasets, transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt 
from RVAE import RVAE

import numpy as np

"""implementation of the Variational Recurrent
Neural Network (VRNN) from https://arxiv.org/abs/1506.02216
using unimodal isotropic gaussian distributions for 
inference, prior, and generating models."""

#hyperparameters
n_features = 28
n_phi_x_hidden = 100
n_phi_z_hidden = 100
n_latent = 16
n_hidden_prior = 100
n_rec_hidden = 100
n_encoder_hidden = 100
n_decoder_hidden = 100
n_rec_layers = 1
bias = False
#######
n_epochs = 10
clip = 10
learning_rate = 1e-3
batch_size = 128
seed = 128
print_every = 100
save_every = 10

#manual seed
torch.manual_seed(seed)
plt.ion()

#init model + optimizer + datasets
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data', train=True, download=True,
		transform=transforms.ToTensor()),
    batch_size=batch_size, shuffle=True)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data', train=False, 
		transform=transforms.ToTensor()),
    batch_size=batch_size, shuffle=True)

model = RVAE(n_features,
             n_phi_x_hidden,
             n_phi_z_hidden,
             n_latent,
             n_hidden_prior,
             n_rec_hidden,
             n_encoder_hidden,
             n_decoder_hidden,
             n_rec_layers,
             bias
             )
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

epoch = 1

train_loss = 0
enum = list(enumerate(train_loader))
batch_idx, (data, _) = enum[0]

#################################################
###  At this point data has shape (128, 1, 28, 28)
##################################################
data = Variable(data.squeeze().transpose(0, 1))
##############################################
### At this point data has shape (28, 128, 28)
##############################################
            
data = (data - data.min().data.item()) / (data.max().data.item() - data.min().data.item())		

#forward + backward + optimize
optimizer.zero_grad()

self = model
x = data

all_encoder_mu, all_encoder_sigma = [], []
all_decoder_mu, all_decoder_sigma = [], []
kld_loss = 0
nll_loss = 0

h = Variable(torch.zeros(self.n_rec_layers, x.size(1), self.n_rec_hidden))

t = 0


def train(epoch):
	train_loss = 0
	for batch_idx, (data, _) in enumerate(train_loader):

            #################################################
            ###  At this point data has shape (128, 1, 28, 28)
            ##################################################
            data = Variable(data.squeeze().transpose(0, 1))
            ##############################################
            ### At this point data has shape (28, 128, 28)
            ##############################################
            
            data = (data - data.min().data.item()) / (data.max().data.item() - data.min().data.item())		
            
            #forward + backward + optimize
            optimizer.zero_grad()
		
            kld_loss, nll_loss, _, _ = model(data)
            loss = kld_loss + nll_loss
            loss.backward()
            optimizer.step()
            
            #grad norm clipping, only in pytorch version >= 1.10
            nn.utils.clip_grad_norm(model.parameters(), clip)
            
            #printing
            if batch_idx % print_every == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\t KLD Loss: {:.6f} \t NLL Loss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader),
                    kld_loss.data.item() / batch_size,
                    nll_loss.data.item() / batch_size))


            train_loss += loss.data.item()

	print('====> Epoch: {} Average loss: {:.4f}'.format(
		epoch, train_loss / len(train_loader.dataset)))

