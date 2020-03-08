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
train_set_x0 = train_set_x.get_value()[:1000, :]
train_set_x0 = np.swapaxes(train_set_x0.reshape(-1, 28, 28), 0, 1)
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

train_set_x0 = train_set_x.get_value()[14,:,:]
train_set_x = theano.shared(name='train_set_x', value=train_set_x0)

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
