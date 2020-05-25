import numpy as np
from numpy.linalg import cholesky
import matplotlib.pyplot as plot
import time
import os
import pickle
import theano
import theano.tensor as T
from theano_utils import (gd_solver, rmsprop_solver, negative_feedback,
                          gd_optimizer, rmsprop_optimizer, adam_optimizer, load_data,)

import RVAE_theano


########
# Data #
########
# MNIST dataset
# all_data = load_data('mnist.pkl.gz')
# train_set_x = all_data[0][0] # row major
# train_set_x0 = np.swapaxes(train_set_x.get_value().reshape(-1, 28, 28), 0, 1)
# train_set_x = theano.shared(name='train_set_x', value=train_set_x0)
# (m, N, n) = train_set_x0.shape


# Simulated correlated data
np_rng = np.random.RandomState(51)
low, high = 0.025, 1.15
num_features = 28
num_rows = 10000
num_trials = 1
# raw_samp = np_rng.normal(low, high, (num_features,num_trials))
# covar = np.cov(raw_samp)
# L = cholesky(covar)
# z = np_rng.normal(low, high, (num_rows, num_trials, num_features))
z = np_rng.uniform(low, high, (num_rows, num_trials, num_features))
# for ind in range(num_trials):
#     z[:, ind, :] = np.dot(z[:, ind, :], L)    
train_set_x = theano.shared(np.asarray(z, dtype=theano.config.floatX),
                            borrow=True)
(m, N, n) = z.shape

###############
# Layer specs #
###############
# Theano
# OLD PARAMS
# n_features       = 28
# n_hidden_encoder = [150, 150]
# n_phi_x_hidden   = [150, 150]
# n_phi_z_hidden   = [150, 150]
# n_latent         = [28]
# n_hidden_prior   = 150
# n_hidden_decoder = [150, 150]
# n_rec_hidden     = [150]
# n_rec_layers     = 3

# n_features       = 25
# n_hidden_encoder = [15]
# n_phi_x_hidden   = [15]
# n_phi_z_hidden   = [15]
# n_latent         = [28]
# n_hidden_prior   = 20
# n_hidden_decoder = [15]
# n_rec_hidden     = [15]
# n_rec_layers     = 3

# large
# If GMM_nll == True, we probably need a large
# n_hidden_decoder as decoder output needs to
# encode {mu, sigma, coeff} for the GMM distribution
# n_features       = 28
# n_hidden_encoder = [150, 150]
# n_phi_x_hidden   = [150, 150]
# n_phi_z_hidden   = [150, 150]
# n_latent         = [28]
# n_hidden_prior   = 150
# n_hidden_decoder = [150, 150]
# n_rec_hidden     = [150]
# n_rec_layers     = 3
# GMM_nll          = True
# n_coeff          = 2

# small
n_features       = 28
n_hidden_encoder = [50]
n_phi_x_hidden   = [50]
n_phi_z_hidden   = [50]
n_latent         = [50]
n_hidden_prior   = 50
n_hidden_decoder = [50]
n_rec_hidden     = [50]
n_rec_layers     = 5
GMM_nll          = False
n_coeff          = 1


#################
# Solver params #
#################
# Solver params, by type
gd_solver_kwargs = dict(learning_rate=0.1)
rmsprop_solverKwargs = dict(eta=0.001,beta=0.8,epsilon=1.e-6)
adam_solverKwargs = dict(learning_rate=0.001,beta1=0.9,beta2=0.999,epsilon=1e-8)

#########
# Model #
#########
x = T.matrix('x')
index = T.iscalar('index')
batch_size = 1

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
                         rmsprop_solverKwargs=rmsprop_solverKwargs,
                         adam_solverKwargs=adam_solverKwargs,
                         L=1,
                         n_rec_layers=n_rec_layers,
                         GMM_nll=GMM_nll,
                         n_coeff=n_coeff,
                         rng=None)


# Test
self = model
h = self.h
x_in = train_set_x_batch
x_step = train_set_x_batch[0]
dev = self.srng.normal((self.batch_size, self.n_latent[-1]))

index = T.lscalar('index')
model.x = train_set_x[:, 0:batch_size, :]
# rmsprop updates
# cost, updates, log_p_x_z, KLD = model.compute_rmsprop_cost_updates()
# adam updates
cost, updates, log_p_x_z, KLD = model.compute_adam_cost_updates()
cost0 = theano.function([], cost)()
train_rvae = theano.function(
    [index],
    [cost, log_p_x_z, KLD],
    updates=updates,
    givens=[(model.x, train_set_x[:, index*batch_size:(index+1)*batch_size, :])],
    )

epochs = range(0,10000)
num_batches = range(int(N/batch_size))
report_each = 200

for epoch in epochs:
    print('EPOCH {}'.format(epoch))
    costs = list()
    log_p_x_zs = list()
    KLDs = list()
    for i in num_batches:
        cost, log_p_x_z, KLD = train_rvae(i)
        costs.append(cost)
        log_p_x_zs.append(log_p_x_z)
        KLDs.append(KLD)
        if not i % report_each:
            print('{} Minibatch: {} Cost: {:.8} Avg Cost: {:.8} log_p_x_z: {:.8} KLD: {:.8}'.format(time.ctime(time.time()), i, costs[-1], np.mean(costs), np.mean(log_p_x_zs), np.mean(KLDs)))
            # if model.GMM_nll:
            #     x_in = train_set_x[:, i*batch_size:(i+1)*batch_size, :]
            #     plot.imshow(model.GMM_predict(x_in))
            #     plot.pause(1e-6)
            # else:
            #     plot.imshow(model.sample(28))
            #     plot.pause(1e-6)
            # filename = '/home/charles/git/theano_RVAE/RVAE_global_large_epoch_{}_minibatch_{}'.format(epoch, i)
            # f = open(filename, 'wb')
            # pickle.dump(model, f)
            # f.close()

# index = 293
# x_in = train_set_x[:, index*batch_size:(index+1)*batch_size, :]
# x_step = x_in[0]
