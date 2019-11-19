import math
import torch
import torch.nn as nn
import torch.utils
import torch.utils.data
from torchvision import datasets, transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt

import numpy as np


"""implementation of the Variational Recurrent
Neural Network (RVAE) from https://arxiv.org/abs/1506.02216
using unimodal isotropic gaussian distributions for 
inference, prior, and generating models."""

class RVAE(nn.Module):
	def __init__(self,
		   n_features,
		   n_phi_x_hidden,
		   n_phi_z_hidden,
		   n_latent,
		   n_hidden_prior,
		   n_rec_hidden,
		   n_encoder_hidden,
		   n_decoder_hidden,
		   n_rec_layers=2,
		   bias=False
		   ):
		
            super(RVAE, self).__init__()

            self.n_features = n_features
            self.n_phi_x_hidden = n_phi_x_hidden
            self.n_phi_z_hidden = n_phi_z_hidden
            self.n_latent = n_latent
            self.n_hidden_prior = n_hidden_prior
            self.n_rec_hidden = n_rec_hidden
            self.n_encoder_hidden = n_encoder_hidden
            self.n_decoder_hidden = n_decoder_hidden
            self.n_rec_layers = n_rec_layers

            self.phi_x = nn.Sequential(
                nn.Linear(n_features, n_phi_x_hidden),
                nn.ReLU(),
                nn.Linear(n_phi_x_hidden, n_phi_x_hidden),
                nn.ReLU()
                )
            self.phi_z = nn.Sequential(
                nn.Linear(n_latent, n_phi_z_hidden),
                nn.ReLU(),
                nn.Linear(n_phi_z_hidden, n_phi_z_hidden),
                nn.ReLU()
                )

            self.main_encoder = nn.Sequential(
                nn.Linear(n_phi_x_hidden + n_rec_hidden, n_encoder_hidden),
                nn.ReLU(),
                nn.Linear(n_rec_hidden, n_rec_hidden),
                nn.ReLU()
                )
            self.main_encoder_mu = nn.Sequential(
                nn.Linear(n_rec_hidden, n_latent),
                nn.Sigmoid()
                )
            self.main_encoder_sigma = nn.Sequential(
                nn.Linear(n_rec_hidden, n_latent),
                nn.Softplus()
                )

            self.prior = nn.Sequential(
                nn.Linear(n_rec_hidden, n_hidden_prior),
                nn.ReLU()
                )
            self.prior_mu = nn.Sequential(
                nn.Linear(n_hidden_prior, n_latent),
                nn.Sigmoid()
                )
            self.prior_sigma = nn.Sequential(
                nn.Linear(n_hidden_prior, n_latent),
                nn.Softplus()
                )

            self.main_decoder = nn.Sequential(
                nn.Linear(n_phi_z_hidden + n_rec_hidden, n_decoder_hidden),
                nn.ReLU(),
                nn.Linear(n_decoder_hidden, n_decoder_hidden),
                nn.ReLU()
                )
            self.main_decoder_mu = nn.Sequential(
                nn.Linear(n_decoder_hidden, n_features),
                nn.Sigmoid()
                )
            self.main_decoder_sigma = nn.Sequential(
                nn.Linear(n_decoder_hidden, n_features),
                nn.Softplus()
                )

            self.recurrent_cell = nn.GRU( n_phi_x_hidden + n_phi_z_hidden,
                                          n_rec_hidden,
                                          n_rec_layers,
                                          bias)
            
	def forward(self, x):

            all_encoder_mu, all_encoder_sigma = [], []
            all_decoder_mu, all_decoder_sigma = [], []
            kld_loss = 0
            nll_loss = 0
            
            h = Variable(torch.zeros(self.n_rec_layers, x.size(1), self.n_rec_hidden))
            # At this point x has shape (28, 128, 28)
            
            for t in range(x.size(0)):

                phi_x = self.phi_x(x[t])
                
                encoder_input = torch.cat([phi_x, h[-1]], 1)
                encoder_output = self.main_encoder(encoder_input)
                encoder_mu = self.main_encoder_mu(encoder_output)
                encoder_sigma = self.main_encoder_sigma(encoder_output)
                
                prior = self.prior(h[-1])
                prior_mu = self.prior_mu(prior)
                prior_sigma = self.prior_sigma(prior)
                
                z = self._reparameterized_sample( encoder_mu, encoder_sigma)
                phi_z = self.phi_z(z)
                
                decoder_input = torch.cat([phi_z, h[-1]], 1)
                decoder_output = self.main_decoder(decoder_input)
                decoder_mu = self.main_decoder_mu(decoder_output)
                decoder_sigma = self.main_decoder_sigma(decoder_output)
                
                # recurrent cell
                _, h = self.recurrent_cell(torch.cat([phi_x, phi_z], 1).unsqueeze(0), h)
                
                kld_loss += self._kld_gauss(encoder_mu, encoder_sigma, prior_mu, prior_sigma)
                nll_loss += self._nll_bernoulli(decoder_mu, x[t])
                
                all_encoder_mu.append(encoder_mu)
                all_encoder_sigma.append(encoder_sigma)
                all_decoder_mu.append(decoder_mu)
                all_decoder_sigma.append(decoder_sigma)
                
            return kld_loss, nll_loss, (all_encoder_mu, all_encoder_sigma), (all_decoder_mu, all_decoder_sigma)

	def sample(self, seq_len):
            
            sample = torch.zeros(seq_len, self.n_features)

            h = Variable(torch.zeros(self.n_rec_layers, 1, self.n_rec_hidden))
            for t in range(seq_len):
                prior = self.prior(h[-1])
                prior_mu = self.prior_mu(prior)
                prior_sigma = self.prior_sigma(prior)
                
                # sampling and reparameterization
                z = self._reparameterized_sample(prior_mu, prior_sigma)
                phi_z = self.phi_z(z)
                
                # decoder
                decoder = self.main_decoder(torch.cat([phi_z, h[-1]], 1))
                decoder_mu = self.main_decoder_mu(decoder)
                dec_sigma = self.main_decoder_sigma(decoder)
                
                phi_x = self.phi_x(decoder_mu)

                # recurrence
                _, h = self.recurrent_cell(torch.cat([phi_x, phi_z], 1).unsqueeze(0), h)
                
                sample[t] = decoder_mu.data
	
            return sample

	def reset_parameters(self, stdv=1e-1):
		for weight in self.parameters():
			weight.data.normal_(0, stdv)

	def _init_weights(self, stdv):
		pass

	def _reparameterized_sample(self, mean, std):
		"""using std to sample"""
		eps = torch.FloatTensor(std.size()).normal_()
		eps = Variable(eps)
		return eps.mul(std).add_(mean)


	def _kld_gauss(self, mean_1, std_1, mean_2, std_2):
		"""Using std to compute KLD"""

		kld_element =  (2 * torch.log(std_2) - 2 * torch.log(std_1) + 
			(std_1.pow(2) + (mean_1 - mean_2).pow(2)) /
			std_2.pow(2) - 1)
		return	0.5 * torch.sum(kld_element)


	def _nll_bernoulli(self, theta, x):
		val = torch.sum(x*torch.log(theta) + (1-x)*torch.log(1-theta))
		if np.isnan(val.detach().numpy()):
			import pdb
			pdb.set_trace()
			SMALL_NUMBER = -35.
			log_theta = torch.log(theta)
			log_1_minus_theta = torch.log(1-theta)
			log_theta[log_theta == -np.inf] = SMALL_NUMBER
			log_1_minus_theta[log_1_minus_theta == -np.inf] = SMALL_NUMBER
			return - torch.sum(x*log_theta + (1-x)*log_1_minus_theta)
		return - val


	def _nll_gauss(self, mean, std, x):
		val1 = torch.sum(torch.log( 1 / torch.sqrt( 2 * 3.1415926535 * std**2)))
		val2 = - torch.sum((x - mean)**2 / 2 / (std**2))
		return - (val1 + val2)
