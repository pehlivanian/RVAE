import numpy as np
from numpy.linalg import cholesky
import matplotlib.pyplot as plot
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
# OLD PARAMS
n_features       = 28
n_hidden_encoder = [150, 150]
n_phi_x_hidden   = [150, 150]
n_phi_z_hidden   = [150, 150]
n_latent         = [28]
n_hidden_prior   = 150
n_hidden_decoder = [150, 150]
n_rec_hidden     = [150]
n_rec_layers     = 2

# n_features       = 28
# n_hidden_encoder = [25]
# n_phi_x_hidden   = [25]
# n_phi_z_hidden   = [25]
# n_latent         = [28]
# n_hidden_prior   = 25
# n_hidden_decoder = [25]
# n_rec_hidden     = [25]
# n_rec_layers     = 2

# n_features       = 28
# n_hidden_encoder = [28]
# n_phi_x_hidden   = [28]
# n_phi_z_hidden   = [28]
# n_latent         = [28]
# n_hidden_prior   = 28
# n_hidden_decoder = [28]
# n_rec_hidden     = [28]
# n_rec_layers     = 3

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
# OLD PARAMS
batch_size = 2

# This represents one minibatch, take care of it later with (index, givens) logic
train_set_x_batch0 = train_set_x.get_value()[:,0:batch_size, :]
train_set_x_batch = theano.shared(name='train_set_x', value=train_set_x_batch0)

model = RVAE_theano.RVAE(n_features,
                         n_hidden_encoder,
                         n_hidden_decoder,
                         n_latent,
                         n_hidden_prior,
                         n_rec_hidden,
                         n_phi_x_hidden,
                         n_phi_z_hidden,
                         train_set_x_batch,
                         batch_size=batch_size,
                         solver='rmsprop',
                         solverKwargs=dict(eta=1.e-4, beta=.7, epsilon=1.e-6),
                         L=1,
                         n_rec_layers=n_rec_layers,
                         rng=None)


# Test
self = model
h = self.h
x_in = train_set_x_batch
x_step = train_set_x_batch[0]
dev = self.srng.normal((self.batch_size, self.n_latent[-1]))

index = T.lscalar('index')
model.x = train_set_x[:, 0:batch_size, :]
cost, updates = model.compute_cost_updates()
cost0 = theano.function([], cost)()
train_rvae = theano.function(
    [index],
    cost,
    updates=updates,
    givens=[(model.x, train_set_x[:, index*batch_size:(index+1)*batch_size, :])],
    )

num_epochs = range(10)
num_batches = range(int(N/batch_size))
report_each = 100
costs = list()

for epoch in num_epochs:
    print('EPOCH {}'.format(epoch))
    for i in num_batches:
        costs.append(train_rvae(i))
        if not i % report_each:
            print('Minibatch: {} Avg Cost: {}'.format(i, np.mean(costs)))
    filename = '/home/charles/git/theano_RVAE/VRAE_epoch_{}'.format(epoch)
    f = open(filename, 'wb')
    pickle.dump(model, f)
    f.close()
        
        
