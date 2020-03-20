from __future__ import division

import numpy as np
import matplotlib.pyplot as plt
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

import pickle
from collections import OrderedDict

from theano_utils import (gd_solver, rmsprop_solver, negative_feedback, gd_optimizer,
                          rmsprop_optimizer, adam_optimizer, load_data)

import RNN
import dA
import hiddenlayer

SEED = 483
epsilon = 1e-8

def relu(x):
    return T.switch(x<0, 0, x)

def _create_weight(dim_input, dim_output, rng, sigma_init):
    return rng.normal(0,
                      sigma_init,
                      (dim_input,
                       dim_output)).astype(theano.config.floatX)
def _create_bias(dim_output):
    return np.zeros(dim_output).astype(theano.config.floatX)

def _create_initial_mu(dim_input, dim_output, rng, var=0.01 ):
    return rng.normal(0,
                      var,
                      (dim_input, dim_output)
                      ).astype(theano.config.floatX)

def _create_initial_sigma(dim_input, dim_output, rng, var=0.01):
    return rng.normal(1.0,
                      var,
                      (dim_input, dim_output)
                      ).astype(theano.config.floatX)

def _get_activation(fn_name):
    if fn_name == 'tanh':
        return T.tanh
    elif fn_name == 'sigmoid':
        return T.nnet.sigmoid
    elif fn_name == 'symmetric_sigmoid':
        return lambda x: -1 + 2 * T.nnet.sigmoid(x)
    elif fn_name == 'identity':
        return lambda x: x
    elif fn_name == 'relu':
        return T.nnet.relu
    else:
        raise RuntimeError( 'activation function %s not supported' % (fn_name, ))

class RVAE:
    ''' Recurrent Variational Auto Encoder implementation
    '''
    def __init__(self,
                 n_features,
                 n_hidden_encoder,
                 n_hidden_decoder,
                 n_latent,
                 n_hidden_prior,
                 n_rec_hidden,
                 n_phi_x_hidden,
                 n_phi_z_hidden,
                 input,
                 batch_size=100,
                 solver='rmsprop',
                 solverKwargs=dict(eta=1.e-4,beta=.7,epsilon=1.e-6),
                 L=1,
                 n_rec_layers=2,
                 bptt_truncate=-1,
                 rng=None):

        self.params = list()

        rng = rng or np.random.RandomState(SEED)
        if 'gpu' in theano.config.device:
            theano_rng = theano.sandbox.cuda.rng_curand.CURAND_RandomStreams(seed=SEED)
        else:
            theano_rng = RandomStreams(rng.randint(2 ** 30))
        self.np_rng = rng
        self.srng = T.shared_randomstreams.RandomStreams(seed=SEED)        
        self.rng = theano_rng

        self.bptt_truncate = bptt_truncate

        if input is None:
            self.x = T.dmatrix(name='input')
        else:
            # We wire the layers so that the dimensions are set up for an
            # iteration over input as in
            #
            # for input_step in input:
            #     train(input_step)
            #
            # where input_step represents, in the MNIST case,1 row of the digit
            # batch_size times
            # This will get overrided in a givens method
            self.x = input[0]

        z = T.matrix('z')
        self.z = z

        self.solver = solver
        self.solverKwargs = solverKwargs

        self.global_activation_fn = 'sigmoid'

        self.n_features = n_features
        self.n_hidden_encoder = n_hidden_encoder
        self.n_hidden_decoder = n_hidden_decoder
        self.n_latent = n_latent
        self.n_hidden_prior = n_hidden_prior
        self.n_rec_hidden = n_rec_hidden
        self.n_phi_x_hidden = n_phi_x_hidden
        self.n_phi_z_hidden = n_phi_z_hidden

        # number of samples z^(i,l) per datapoint:
        self.L = L

        # number of layers in recurrent cell
        self.n_rec_layers = n_rec_layers

        self.batch_size = batch_size

        self.sigma_init = 0.01

        epoch = T.scalar('eopch')
        #########################
        # START CREATE ENCODERS #
        #########################
        # The network that determines the latent inputs to the
        # mu, logSigma generation hidden layers.
        # mu, sigma are both (n_latent) x 1 vectors, as
        # logSigma is an entrywise log of the multiplier for the
        # diagaonal covariance matrix - no covariance between
        # latent factors is assumed.

        self.phi_x = dA.SdA(
            numpy_rng=self.np_rng,
            theano_rng=self.rng,
            input=self.x,
            symmetric=False,
            n_visible=n_features,
            dA_layers_sizes=self.n_phi_x_hidden,
            tie_weights=False,
            tie_biases=False,
            encoder_activation_fn='relu',
            decoder_activation_fn='relu',
            global_decoder_activation_fn='relu',
            initialize_W_as_identity=False,
            initialize_W_prime_as_W_transpose=False,
            add_noise_to_W=False,
            noise_limit=0.,
            solver_type='rmsprop',
            predict_modifier_type='negative_feedback',
            solver_kwargs=dict(eta=1.e-4,beta=.7,epsilon=1.e-6)            
            )        

        self.h_shape = (self.n_rec_layers, input.get_value().shape[1], self.n_rec_hidden[-1])
        self.h = theano.shared(name='h', value=np.zeros(self.h_shape))
        encoder_input = T.concatenate([self.phi_x.output(), self.h[-1]], axis=1)
        
        self.main_encoder = dA.SdA(
            numpy_rng=self.np_rng,
            theano_rng=self.rng,
            input=encoder_input,
            symmetric=False,
            n_visible=self.n_phi_x_hidden[-1] + self.n_rec_hidden[-1],
            dA_layers_sizes=self.n_hidden_encoder,
            tie_weights=False,
            tie_biases=False,
            encoder_activation_fn='relu',
            decoder_activation_fn='relu',
            global_decoder_activation_fn='relu',
            initialize_W_as_identity=False,
            initialize_W_prime_as_W_transpose=False,
            add_noise_to_W=False,
            noise_limit=0.,
            solver_type='rmsprop',
            predict_modifier_type='negative_feedback',
            solver_kwargs=dict(eta=1.e-4,beta=.7,epsilon=1.e-6)
            )

        self.phi_z = dA.SdA(
            numpy_rng=self.np_rng,
            theano_rng=self.rng,
            input=self.z,
            symmetric=False,
            n_visible=n_latent[-1],
            dA_layers_sizes=self.n_phi_z_hidden,
            tie_weights=False,
            tie_biases=False,
            encoder_activation_fn='relu',
            decoder_activation_fn='relu',
            global_decoder_activation_fn='relu',
            initialize_W_as_identity=False,
            initialize_W_prime_as_W_transpose=False,
            add_noise_to_W=False,
            noise_limit=0.,
            solver_type='rmsprop',
            predict_modifier_type='negative_feedback',
            solver_kwargs=dict(eta=1.e-4,beta=.7,epsilon=1.e-6)            
            )

        self.mu_encoder = dA.SdA(
            numpy_rng=self.np_rng,
            theano_rng=self.rng,
            input=self.main_encoder.output(),
            symmetric=False,
            n_visible=self.n_hidden_encoder[-1],
            dA_layers_sizes=self.n_latent,
            tie_weights=False,
            tie_biases=False,
            encoder_activation_fn='sigmoid',
            decoder_activation_fn='sigmoid',
            global_decoder_activation_fn='sigmoid',
            initialize_W_as_identity=False,
            initialize_W_prime_as_W_transpose=False,
            add_noise_to_W=False,
            noise_limit=0.,
            solver_type='rmsprop',
            predict_modifier_type='negative_feedback',
            solver_kwargs=dict(eta=1.e-4,beta=.7,epsilon=1.e-6)
            )

        self.log_sigma_encoder = dA.SdA(
            numpy_rng=self.np_rng,
            theano_rng=self.rng,
            input=self.main_encoder.output(),
            symmetric=False,
            n_visible=self.n_hidden_encoder[-1],
            dA_layers_sizes=self.n_latent,
            tie_weights=False,
            tie_biases=False,
            encoder_activation_fn='softplus',
            decoder_activation_fn='softplus',
            global_decoder_activation_fn='softplus',
            initialize_W_as_identity=False,
            initialize_W_prime_as_W_transpose=False,
            add_noise_to_W=False,
            noise_limit=0.,
            solver_type='rmsprop',
            predict_modifier_type='negative_feedback',
            solver_kwargs=dict(eta=1.e-4,beta=.7,epsilon=1.e-6),
            )

        for layer in self.main_encoder.mlp_layers:
            self.params = self.params + layer.params
        for layer in self.mu_encoder.mlp_layers:
            self.params = self.params + layer.params
        for layer in self.log_sigma_encoder.mlp_layers:
            self.params = self.params + layer.params
        for layer in self.phi_x.mlp_layers:
            self.params = self.params + layer.params
        for layer in self.phi_z.mlp_layers:
            self.params = self.params + layer.params
        #######################
        # END CREATE ENCODERS #
        #######################

        ################
        # CREATE PRIOR #
        ################

        self.prior = hiddenlayer.HiddenLayer(rng=self.np_rng,
                                             input=self.h[-1],
                                             n_in=self.n_rec_hidden[-1],
                                             n_out=self.n_hidden_prior,
                                             activation=T.nnet.relu,
                                             )
        self.prior_mu = dA.SdA(
            numpy_rng=self.np_rng,
            theano_rng=self.rng,
            input=self.prior.output(),
            symmetric=False,
            n_visible=self.n_hidden_prior,
            dA_layers_sizes=self.n_latent,
            tie_weights=False,
            tie_biases=False,
            encoder_activation_fn='sigmoid',
            decoder_activation_fn='sigmoid',
            global_decoder_activation_fn='sigmoid',
            initialize_W_as_identity=False,
            initialize_W_prime_as_W_transpose=False,
            add_noise_to_W=False,
            noise_limit=0.,
            solver_type='rmsprop',
            predict_modifier_type='negative_feedback',
            solver_kwargs=dict(eta=1.e-4,beta=.7,epsilon=1.e-6)
            )

        self.prior_log_sigma = dA.SdA(
            numpy_rng=self.np_rng,
            theano_rng=self.rng,
            input=self.prior.output(),
            symmetric=False,
            n_visible=self.n_hidden_prior,
            dA_layers_sizes=self.n_latent,
            tie_weights=False,
            tie_biases=False,
            encoder_activation_fn='softplus',
            decoder_activation_fn='softplusy',
            global_decoder_activation_fn='softplus',
            initialize_W_as_identity=False,
            initialize_W_prime_as_W_transpose=False,
            add_noise_to_W=False,
            noise_limit=0.,
            solver_type='rmsprop',
            predict_modifier_type='negative_feedback',
            solver_kwargs=dict(eta=1.e-4,beta=.7,epsilon=1.e-6),
            )
        
        self.params = self.params + self.prior.params
        for layer in self.prior_mu.mlp_layers:
            self.params = self.params + layer.params
        for layer in self.prior_log_sigma.mlp_layers:
            self.params = self.params + layer.params
        ####################
        # END CREATE PRIOR #
        ####################

        ########################
        # START CREATE DECODER #
        ########################

        decoder_input = T.concatenate([self.phi_z.output(), self.h[-1]], axis=1)
        self.main_decoder = dA.SdA(
            numpy_rng=self.np_rng,
            theano_rng=theano_rng,
            input=decoder_input,
            symmetric=False,
            n_visible=self.n_phi_z_hidden[-1] + self.n_rec_hidden[-1],
            dA_layers_sizes=self.n_hidden_decoder,
            tie_weights=False,
            tie_biases=False,
            encoder_activation_fn='relu',
            decoder_activation_fn='relu',
            global_decoder_activation_fn='relu',
            initialize_W_as_identity=False,
            initialize_W_prime_as_W_transpose=False,
            add_noise_to_W=False,
            noise_limit=0.,
            solver_type='rmsprop',
            predict_modifier_type='negative_feedback',
            solver_kwargs=dict(eta=1.e-4,beta=.7,epsilon=1.e-6),
            )

        self.main_decoder_mu = dA.SdA(
            numpy_rng=self.np_rng,
            theano_rng=theano_rng,
            input=self.main_decoder.output(),
            symmetric=False,
            n_visible=self.n_hidden_decoder[-1],
            dA_layers_sizes=[self.n_features],
            tie_weights=False,
            tie_biases=False,
            encoder_activation_fn='sigmoid',
            decoder_activation_fn='sigmoid',
            global_decoder_activation_fn='sigmoid',
            initialize_W_as_identity=False,
            initialize_W_prime_as_W_transpose=False,
            add_noise_to_W=False,
            noise_limit=0.,
            solver_type='rmsprop',
            predict_modifier_type='negative_feedback',
            solver_kwargs=dict(eta=1.e-4,beta=.7,epsilon=1.e-6),
            )

        self.main_decoder_log_sigma = dA.SdA(
            numpy_rng=self.np_rng,
            theano_rng=theano_rng,
            input=self.main_decoder.output(),
            symmetric=False,
            n_visible=self.n_hidden_decoder[-1],
            dA_layers_sizes=[self.n_features],
            tie_weights=False,
            tie_biases=False,
            encoder_activation_fn='softplus',
            decoder_activation_fn='softplus',
            global_decoder_activation_fn='softplus',
            initialize_W_as_identity=False,
            initialize_W_prime_as_W_transpose=False,
            add_noise_to_W=False,
            noise_limit=0.,
            solver_type='rmsprop',
            predict_modifier_type='negative_feedback',
            solver_kwargs=dict(eta=1.e-4,beta=.7,epsilon=1.e-6),
            )

        for layer in self.main_decoder.mlp_layers:
            self.params = self.params + layer.params
        for layer in self.main_decoder_mu.mlp_layers:
            self.params = self.params + layer.params
        # Not used in output calibraion, output generation
        # for layer in self.main_decoder_log_sigma.mlp_layers:
        #     self.params = self.params + [layer.W, layer.b]
        ######################
        # END CREATE DECODER #
        ######################

        #########################
        # CREATE RECURRENT UNIT #
        #########################

        recurrent_input = T.concatenate([self.phi_x.output(), self.phi_z.output()], axis=1)
        self.recurrent_layer = RNN.GRU_for_RVAE(self.np_rng,
                                                recurrent_input.T,
                                                recurrent_input.T,
                                                self.n_phi_x_hidden[-1] + self.n_phi_z_hidden[-1],
                                                self.n_rec_hidden[-1],
                                                self.n_rec_layers,
                                                batch_size=self.batch_size,
                                                theano_rng=None,
                                                bptt_truncate=-1)

        self.params = self.params + self.recurrent_layer.hidden_params        
        #############################
        # END CREATE RECURRENT UNIT #
        #############################

        ############################################
        # START CREATE GLOBAL HIDDEN LAYER WEIGHTS #
        ############################################
        # The HiddenLayer class accepts None to specify linear activation
        # We will use sigmoid however as cost involves binary_crossentropy
        global_activation_fn = _get_activation('sigmoid')
        self.global_decoder = hiddenlayer.HiddenLayer(rng=self.np_rng,
                                                      input=self.main_decoder_mu.output(),
                                                      n_in=self.n_features,
                                                      n_out=self.n_features,
                                                      activation=global_activation_fn,
                                                      )

        # XXX
        # self.params = self.params + self.global_decoder.params
        ##########################################
        # END CREATE GLOBAL HIDDEN LAYER WEIGHTS #
        ##########################################

    def reconstruction_sample_old(self, mu, logSigma):
        global SEED
        srng = T.shared_randomstreams.RandomStreams(seed=SEED)
        dev = 0.0025 * T.ones((self.batch_size, self.n_latent[-1]))
        # dev = srng.normal((self.batch_size, self.n_latent[-1]))
        z = mu + T.exp(0.5 * logSigma) * dev
        return z

    def reconstruction_sample(self, mu, logSigma):
        global SEED
        srng = T.shared_randomstreams.RandomStreams(seed=SEED)
        dev = 0.0025 * T.ones((self.batch_size, self.n_latent[-1]))
        # dev = srng.normal((self.batch_size, self.n_latent[-1]))
        z = mu + logSigma * dev
        return z

    def KLDivergence_old(self, mu, logSigma, prior_mu, prior_logSigma):
        kld_element = (prior_logSigma - logSigma + (T.exp(logSigma) + (mu - prior_mu)**2) / T.exp(prior_logSigma) - 1)
        return 0.5 * kld_element

    def KLDivergence(self, mu, logSigma, prior_mu, prior_logSigma):
        kld_element = (2. * T.log(prior_logSigma) - 2*T.log(logSigma) + (logSigma**2 + (mu - prior_mu)**2) / prior_logSigma**2 - 1)
        return 0.5 * kld_element

    def get_hidden_cost_output_from_input(self, x_in):

        def iter_step(x_step, h, dev):
            phi_x = self.phi_x.output_from_input(x_step)

            encoder_input = T.concatenate([phi_x, h[-1]], axis=1)        
            encoder_output = self.main_encoder.output_from_input(encoder_input)

            mu = self.mu_encoder.output_from_input(encoder_output)
            logSigma = self.log_sigma_encoder.output_from_input(encoder_output)

            prior = self.prior.output_from_input(h[-1])
            prior_mu = self.prior_mu.output_from_input(prior)
            prior_logSigma = self.prior_log_sigma.output_from_input(prior)

            # z = mu + T.exp(0.5 * logSigma) * dev
            z = mu + logSigma * dev
                                    
            phi_z = self.phi_z.output_from_input(z)

            decoder_input = T.concatenate([phi_z, h[-1]], axis=1)
            decoder_output = self.main_decoder.output_from_input(decoder_input)
            decoder_mu = self.main_decoder_mu.output_from_input(decoder_output)
            # Not used
            # decoder_logSigma = self.main_decoder_log_sigma.output_from_input(decoder_output)

            recurrent_input = T.shape_padaxis(T.concatenate([phi_x, phi_z], axis=1), axis=2)            
            h_t = T.swapaxes(self.recurrent_layer.hidden_output_from_input(recurrent_input), 1, 2)

            # KL Divergence
            KLD = self.KLDivergence(mu, logSigma, prior_mu, prior_logSigma)

            # log(p(x|z))
            # XXX
            # x_tilde = self.global_decoder.output_from_input(decoder_mu)
            x_tilde = decoder_mu
            log_p_x_z = -1 * T.sum( x_step * T.log(x_tilde) + (1 - x_step)*T.log(1 - x_tilde), axis=0)

            obj = T.mean(log_p_x_z + KLD)
        
            return h_t, obj, x_tilde

        [h_n, obj, x],inner_updates = theano.scan(
            fn=iter_step,
            sequences=[x_in],
            truncate_gradient=self.bptt_truncate,
            outputs_info=[theano.shared(np.zeros(self.h_shape, dtype=self.h.dtype), broadcastable=self.h.broadcastable),
                          None,
                          None],
            non_sequences=[T.as_tensor_variable(self.srng.normal((self.batch_size, self.n_latent[-1])), self.h.dtype)],
            )

        h_out = T.swapaxes(h_n, 0, 1)[-1]
        obj_out = T.sum(obj)
        x_out = x

        return h_out, obj_out, x_out

    def get_hidden_cost_output(self):
        return self.get_hidden_cost_output_from_input(self.x)
    
    def output_from_input(self, input):
        _, _, x = self.get_hidden_cost_output_from_input(input)
        return x

    def output(self):
        return self.output_from_input(self.x)

    def predict_from_input(self, input):
        _, _, x = self.get_hidden_cost_output_from_input(input)
        return x

    def predict(self):
        return self.predict_form_input(self.x)

    def objective_from_input(self, input):
        _, obj, _ = self.get_hidden_cost_output_from_input(input)
        return obj

    def objective(self):
        return self.objective_from_input(self.x)

    def compute_cost_updates(self):
        ''' simple rmsprop
        '''
        _, cost, _ = self.get_hidden_cost_output()

        grads = T.grad(cost, self.params)

        one = T.constant(1.0)

        def _updates(param, cache, df, beta=0.8, eta=1.e-2, epsilon=1.e-6):
            cache_val = beta * cache + (one-beta) * df**2
            x = T.switch(T.abs_(cache_val) < epsilon,
                         cache_val,
                         eta * df / (T.sqrt(cache_val) + epsilon))
            updates = (param, param-x), (cache, cache_val)

            return updates

        caches = [theano.shared(name='c%s' % param,
                                value=param.get_value() * 0,
                                broadcastable=param.broadcastable)
                  for param in self.params]

        updates = []

        for param, cache, grad in zip(self.params, caches, grads):
            param_updates, cache_updates = _updates(param,
                                                    cache,
                                                    grad,
                                                    beta=self.solverKwargs['beta'],
                                                    eta=self.solverKwargs['eta'],
                                                    epsilon=self.solverKwargs['epsilon'])

            updates.append(param_updates)
            updates.append(cache_updates)

        return (cost, updates)


    def sample(self, seq_len):
        def sample_one(h):
            prior = self.prior.output_from_input(h[-1])
            prior_mu = self.prior_mu.output_from_input(prior)
            prior_logSigma = self.prior_log_sigma.output_from_input(prior)

            dev = T.as_tensor_variable(self.srng.normal((1, self.n_latent[-1])), self.h.dtype)
            # z = prior_mu + T.exp(0.5 * prior_logSigma) * dev
            z = prior_mu + prior_logSigma * dev
                                    
            phi_z = self.phi_z.output_from_input(z)

            decoder_input = T.concatenate([phi_z, h[-1]], axis=1)
            decoder_output = self.main_decoder.output_from_input(decoder_input)
            decoder_mu = self.main_decoder_mu.output_from_input(decoder_output)

            phi_x = self.phi_x.output_from_input(decoder_mu)

            recurrent_input = T.shape_padaxis(T.concatenate([phi_x, phi_z], axis=1), axis=2)            
            h_t = T.swapaxes(self.recurrent_layer.hidden_output_from_input(recurrent_input), 1, 2)

            # log(p(x|z))
            # XXX
            # x_tilde = self.global_decoder.output_from_input(decoder_mu)
            x_tilde = decoder_mu

            # return [h_t[:, -2:-1, :], x_tilde]
            # for batch_size == 1
            return [h_t, x_tilde]

        [h_all, x_all], sample_updates = theano.scan(
            fn=sample_one,
            outputs_info=[theano.shared(np.zeros((self.h_shape[0], 1, self.h_shape[2])), broadcastable=self.h.broadcastable),
                          None],
            n_steps=seq_len,            
            )
        
        x_all0 = theano.function([], x_all[:, 0, :])()
        
        return x_all0


    def reconstruct_all(self, train_set_x):
        encoder_output = self.main_encoder.output()
        mu = self.mu_encoder.output()
        logSigma = self.log_sigma_encoder.output()
        z = self.sample(mu, logSigma).dimshuffle(1,2)
        hidden_decoded = self.main_decoder.output_from_input(z)
        x_hat = self.global_decoder.output_from_input(hidden_decoded)

        # Generate entire set
        fn = theano.function([], x_hat, givens={self.x:train_set_x})
        return fn()

    #########################################################
    # START PRETRAINING FUNCTIONS FOR EMBEDDED AUTOENCODERS #
    #########################################################
    def encoder_pretraining_functions(self, train_set_x, batch_size):
        return self.main_encoder.pretraining_functions(train_set_x, batch_size, update=True)

    def mu_encoder_pretraining_functions(self, train_set_x, batch_size):
        encoder_shared = theano.shared(value=theano.function([], self.main_encoder.output(), givens={self.x:train_set_x})(),
                                       name='encoder_shared',
                                       borrow=True)
        fns = self.mu_encoder.pretraining_functions(train_set_x=encoder_shared,
                                                     batch_size=batch_size)
        return fns
    
    def log_sigma_encoder_pretraining_functions(self, train_set_x, batch_size):
        encoder_shared = theano.shared(value=theano.function([], self.main_encoder.output(), givens={self.x:train_set_x})(),
                                       name='encoder_shared',
                                       borrow=True)
        fns = self.log_sigma_encoder.pretraining_functions(train_set_x=encoder_shared,
                                                     batch_size=batch_size)
        return fns

    def decoder_pretraining_functions(self, train_set_x, batch_size):
        encoder_output = self.main_encoder.output()
        mu = self.mu_encoder.output()
        logSigma = self.log_sigma_encoder.output()
        z = self.sample(mu, logSigma).dimshuffle(1,2)
        z_shared = theano.shared(value=theano.function([], z, givens={self.x:train_set_x})(), name='z_shared', borrow=True)
        return self.main_decoder.pretraining_functions(z_shared, batch_size, update=True)

    def pretraining_functions(self, train_set_x, batch_size):
        return self.encoder_pretraining_functions(train_set_x, batch_size) + \
               self.mu_encoder_pretraining_functions(train_set_x, batch_size) + \
               self.log_sigma_encoder_pretraining_functions(train_set_x, batch_size) + \
               self.decoder_pretraining_functions(train_set_x, batch_size)
    #######################################################
    # END PRETRAINING FUNCTIONS FOR EMBEDDED AUTOENCODERS #
    #######################################################

    def generate(self, num_devs, seed):
        np_rng = np.random.RandomState(seed)
        z = T.matrix('z')
        hidden_decoded = self.main_decoder.output_from_input(z)
        x_hat = self.global_decoder.output_from_input(hidden_decoded)
        fn = theano.function([z], x_hat )
        z_emp = np_rng.normal(0, 1, (num_devs, self.n_latent))
        x_out = fn(z_emp)
        return x_out

    def describe(self, train_set_x, title=''):
        # main encoder
        main_encoder_desc = self.main_encoder.describe(train_set_x, title='MAIN_ENCODER_'+title)

        # mu encoder
        encoder_shared = theano.shared(value=theano.function([],
                                                             self.main_encoder.output(),
                                                             givens={self.x:train_set_x})(),
                                       name='encoder_shared',
                                       borrow=True)
        mu_encoder_desc   = self.mu_encoder.describe(encoder_shared, title='MU_ENCODER'+title)


        # log sigma encoder
        log_sigma_encoder_desc   = self.log_sigma_encoder.describe(encoder_shared, title='LOGSIGMA_ENCODER_'+title)

        # main decoder
        # main_encoder_desc   = self.main_decoder.describe(train_set_x, title='MAIN_DECODER_'+title)
        return main_encoder_desc + mu_encoder_desc + log_sigma_encoder_desc
