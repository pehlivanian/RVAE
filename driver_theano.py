import numpy as np
from numpy.linalg import cholesky
import time
import os
import pickle
import gzip
import theano
import theano.tensor as T
from theano_utils import (gd_solver, rmsprop_solver, negative_feedback,
                          gd_optimizer, rmsprop_optimizer, adam_optimizer, load_data,)

import RVAE_theano

########
# Data #
########
# MNIST dataset
all_data = load_data('mnist.pkl.gz')
train_set_x = all_data[0][0] # row major
train_set_x0 = np.swapaxes(train_set_x.get_value().reshape(-1, 28, 28), 0, 1)
train_set_x = theano.shared(name='train_set_x', value=train_set_x0)
(m, N, n) = train_set_x0.shape


# Simulated correlated data
# num_features = 8
# num_trials = 1000
# rows_to_flatten = 25
# np_rng = np.random.RandomState(44)
# raw_samp = np_rng.uniform(0.025, 1.15, (num_features,num_trials))
# covar = np.cov(raw_samp)
# L = cholesky(covar)
# z = np.dot( np_rng.uniform( 0.025, 1.15, (500000, num_features)), L)

# Flatten z
# shp = z.shape
# num = shp[0]*shp[1]
# sample_size = num_features*rows_to_flatten
# stride = num_features
# z_flat = z.reshape(1, num)
# z_out = np.zeros((int(1+(num-sample_size)/stride), sample_size))
# for ind in range(int(1+(num-sample_size)/stride)):
#     z_out[ind,:] = z_flat[0,(ind*stride):(ind*stride)+sample_size]
# train_set_x = theano.shared(np.asarray(z_out, dtype=theano.config.floatX),
#                             borrow=True)

###############
# Layer specs #
###############
# Pytorch
# num_features        = 28
# batch_size          = 150
# n_phi_x_hidden      = 150
# n_phi_z_hidden      = 150
# n_latent            = 28
# n_hidden_prior      = 150
# n_rec_hidden        = 150
# n_encoder_hidden    = 150
# n_decoder_hidden    = 150
# n_rec_layers        = 3
# bias                = False

# Theano
n_features       = 28
n_hidden_encoder = [150]
n_phi_x_hidden   = [150]
n_phi_z_hidden   = [150]
n_latent         = [28]
n_hidden_prior   = 150
n_hidden_decoder = [150]
n_rec_hidden     = [150]
n_rec_layers     = 3



#################
# Solver params #
#################
# Solver params, by type
gd_solver_kwargs = dict(learning_rate=0.1)
rmsprop_solver_kwargs = dict(eta=1.e-4,beta=.7,epsilon=1.e-6)
adam_solver_kwargs = dict(learning_rate=0.001,beta1=0.95,beta2=0.999,epsilon=1e-8)

#########
# Model #
#########
x = T.matrix('x')
index = T.iscalar('index')
num_batches = N
batch_size = int(N / num_batches)



import RVAE_theano

# train_set_x0 = train_set_x.get_value()[14,:,:]
# train_set_x = theano.shared(name='train_set_x', value=train_set_x0)

model = RVAE_theano.RVAE(n_features,
                         n_hidden_encoder,
                         n_hidden_decoder,
                         n_latent,
                         n_hidden_prior,
                         n_rec_hidden,
                         n_phi_x_hidden,
                         n_phi_z_hidden,
                         train_set_x,
                         batch_size=batch_size,
                         solver='rmsprop',
                         solverKwargs=dict(eta=1.e-4, beta=.7, epsilon=1.e-6),
                         L=1,
                         n_rec_layers=n_rec_layers,
                         rng=None)



self = model
h_shape = (self.n_rec_layers, model.x.get_value().shape[0], self.n_rec_hidden[-1])
h = theano.shared(name='h', value=np.zeros(h_shape))

phi_x = self.phi_x.output()

encoder_input = T.concatenate([phi_x, h[-1]], axis=1)        
encoder_output = self.main_encoder.output_from_input(encoder_input)

mu = self.mu_encoder.output_from_input(encoder_output)
logSigma = self.log_sigma_encoder.output_from_input(encoder_output)

prior = self.prior.output_from_input(h[-1])
prior_mu = self.prior_mu.output_from_input(prior)
prior_logSigma = self.prior_log_sigma.output_from_input(prior)

z = self.sample(mu, logSigma)

phi_z = self.phi_z.output_from_input(z)

decoder_input = T.concatenate([phi_z, h[-1]], axis=1)
decoder_output = self.main_decoder.output_from_input(decoder_input)
decoder_mu = self.main_decoder_mu.output_from_input(decoder_output)
decoder_logSigma = self.main_decoder_log_sigma.output_from_input(decoder_output)

recurrent_input = T.concatenate([phi_x, phi_z], axis=1)
recurrent_input0 = theano.function([], recurrent_input)()

train_step = theano.function([],
                             self.recurrent_layer.hidden_output(),
                             givens=[(self.recurrent_layer.x, recurrent_input0.astype(theano.config.floatX)),
                                     ]
                             )
