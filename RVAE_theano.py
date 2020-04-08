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
import gpu_utils as utils

import RNN
import dA
import hiddenlayer

SEED = 483
epsilon = 1e-8

def relu(x):
    return T.switch(x<0, 0, x)

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
                 rmsprop_solverKwargs=dict(eta=1.e-3,beta=.7,epsilon=1.e-6),
                 adam_solverKwargs=dict(),
                 L=1,
                 n_rec_layers=2,
                 bptt_truncate=-1,
                 GMM_nll=False,
                 n_coeff=1,
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
        self.GMM_nll = GMM_nll
        self.n_coeff = 1
        if self.GMM_nll:
            self.mu_summand_size = n_features
            self.n_coeff = n_coeff            

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
        self.rmsprop_solverKwargs = rmsprop_solverKwargs
        self.adam_solverKwargs = adam_solverKwargs

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
        encoder_input = utils.concatenate([self.phi_x.output(), self.h[-1]], axis=1)
        
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

        decoder_input = utils.concatenate([self.phi_z.output(), self.h[-1]], axis=1)
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
            dA_layers_sizes=[self.n_features * self.n_coeff],
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

        if self.GMM_nll:
            self.main_decoder_log_sigma = dA.SdA(
                numpy_rng=self.np_rng,
                theano_rng=theano_rng,
                input=self.main_decoder.output(),
                symmetric=False,
                n_visible=self.n_hidden_decoder[-1],
                dA_layers_sizes=[self.n_features * self.n_coeff],
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
            self.main_decoder_coeff = dA.SdA(
                numpy_rng=self.np_rng,
                theano_rng=theano_rng,
                input=self.main_decoder.output(),
                symmetric=False,
                n_visible=self.n_hidden_decoder[-1],
                dA_layers_sizes=[self.n_coeff],
                tie_weights=False,
                tie_biases=False,
                encoder_activation_fn='softmax',
                decoder_activation_fn='softmax',
                global_decoder_activation_fn='softmax',
                initialize_W_as_identity=False,
                initialize_W_prime_as_W_transpose=False,
                add_noise_to_W=False,
                noise_limit=0.,
                solver_type='rmsprop',
                predict_modifier_type='negative_feedback',
                solver_kwargs=dict(eta=1.e-4,beta=.7,epsilon=1.e-6),
                )
                
            for layer in self.main_decoder_log_sigma.mlp_layers:
                self.params = self.params + layer.params
            for layer in self.main_decoder_coeff.mlp_layers:
                self.params = self.params + layer.params
                
        for layer in self.main_decoder.mlp_layers:
            self.params = self.params + layer.params
        for layer in self.main_decoder_mu.mlp_layers:
            self.params = self.params + layer.params
        ######################
        # END CREATE DECODER #
        ######################

        #########################
        # CREATE RECURRENT UNIT #
        #########################

        recurrent_input = utils.concatenate([self.phi_x.output(), self.phi_z.output()], axis=1)
        self.recurrent_layer = RNN.GRU(self.np_rng,
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
        # global_activation_fn = _get_activation('sigmoid')
        # XXX
        # Skip for now
        # self.global_decoder = hiddenlayer.HiddenLayer(rng=self.np_rng,
        #                                               input=self.main_decoder_mu.output(),
        #                                               n_in=self.n_features,
        #                                               n_out=self.n_features,
        #                                               activation=global_activation_fn,
        #                                               )
        # self.params = self.params + self.global_decoder.params
        ##########################################
        # END CREATE GLOBAL HIDDEN LAYER WEIGHTS #
        ##########################################

    def KLDivergence(self, mu, logSigma, prior_mu, prior_logSigma):
        kld_element = T.sum(2. * T.log(prior_logSigma) - 2*T.log(logSigma) + (logSigma**2 + (mu - prior_mu)**2) / prior_logSigma**2 - 1)
        return 0.5 * kld_element

    def get_hidden_cost_output_from_input(self, x_in):

        def iter_step(x_step, dev, h):
            phi_x = self.phi_x.output_from_input(x_step)

            encoder_input = utils.concatenate([phi_x, h[-1]], axis=1)        
            encoder_output = self.main_encoder.output_from_input(encoder_input)

            mu = self.mu_encoder.output_from_input(encoder_output)
            logSigma = self.log_sigma_encoder.output_from_input(encoder_output)

            prior = self.prior.output_from_input(h[-1])
            prior_mu = self.prior_mu.output_from_input(prior)
            prior_logSigma = self.prior_log_sigma.output_from_input(prior)

            # z = mu + T.exp(0.5 * logSigma) * dev
            z = mu + logSigma * dev
                                    
            phi_z = self.phi_z.output_from_input(z)

            decoder_input = utils.concatenate([phi_z, h[-1]], axis=1)
            decoder_output = self.main_decoder.output_from_input(decoder_input)
            decoder_mu = self.main_decoder_mu.output_from_input(decoder_output)

            recurrent_input = T.shape_padaxis(utils.concatenate([phi_x, phi_z], axis=1), axis=2)            
            h_t = T.swapaxes(self.recurrent_layer.hidden_output_from_input(recurrent_input), 1, 2)
            # KL Divergence
            KLD = self.KLDivergence(mu, logSigma, prior_mu, prior_logSigma)

            # log(p(x|z))
            # XXX
            # x_tilde = self.global_decoder.output_from_input(decoder_mu)
            x_tilde = decoder_mu

            if self.GMM_nll:
                decoder_logSigma = self.main_decoder_log_sigma.output_from_input(decoder_output)
                decoder_coeff = self.main_decoder_coeff.output_from_input(decoder_output)
                nll = GMM_diag(x_step, decoder_mu, decoder_logSigma, decoder_coeff,
                          self.mu_summand_size, self.n_coeff)
            else:
                nll = cross_entropy( x_step, x_tilde)
            
            obj = nll + KLD
        
            return h_t, obj, x_tilde, nll, KLD

        [h_n, obj, x, log_p_x_z, KLD],inner_updates = theano.scan(
            fn=iter_step,
            sequences=[x_in,
                       T.as_tensor_variable(self.srng.normal((self.n_features, self.batch_size, self.n_latent[-1])), self.h.dtype)],
            truncate_gradient=self.bptt_truncate,
            outputs_info=[theano.shared(1/self.h_shape[-1] * np.ones(self.h_shape, dtype=self.h.dtype), broadcastable=self.h.broadcastable),
                          None,
                          None,
                          None,
                          None],
            )

        h_out = T.swapaxes(h_n, 0, 1)[-1]
        obj_out = T.sum(obj)
        x_out = x
        log_p_x_z_out = T.sum(log_p_x_z)
        KLD_out = T.sum(KLD)

        return h_out, obj_out, x_out, log_p_x_z_out, KLD_out

    def get_hidden_cost_output(self):
        return self.get_hidden_cost_output_from_input(self.x)
    
    def output_from_input(self, input):
        _, _, x, _, _ = self.get_hidden_cost_output_from_input(input)
        return x

    def output(self):
        return self.output_from_input(self.x)

    def predict_from_input(self, input):
        _, _, x, _, _ = self.get_hidden_cost_output_from_input(input)
        return x

    def predict(self):
        return self.predict_form_input(self.x)

    def objective_from_input(self, input):
        _, obj, _, _, _ = self.get_hidden_cost_output_from_input(input)
        return obj

    def objective(self):
        return self.objective_from_input(self.x)

    def compute_adam_cost_updates(self):
        ''' simple adam
        '''

        _, cost, _, log_p_x_z, KLD = self.get_hidden_cost_output()

        grads = T.grad(cost, self.params)

        updates = list()
        
        beta1 = self.adam_solverKwargs['beta1']
        beta2 = self.adam_solverKwargs['beta2']
        epsilon = self.adam_solverKwargs['epsilon']
        learning_rate = self.adam_solverKwargs['learning_rate']
        
        epoch_pre = theano.shared(np.asarray(.0, dtype=theano.config.floatX))
        epoch_num = epoch_pre + 1
        gamma = learning_rate * T.sqrt(1 - beta2 ** epoch_num) / (1 - beta1 ** epoch_num)

        grads = T.grad(cost, self.params)

        for param, grad in zip(self.params, grads):
            v = param.get_value(borrow=True)
            m_pre = theano.shared(np.zeros(v.shape, dtype=v.dtype), broadcastable=param.broadcastable)
            v_pre = theano.shared(np.zeros(v.shape, dtype=v.dtype), broadcastable=param.broadcastable)

            m_t = beta1 * m_pre + (1 - beta1) * grad
            v_t = beta2 * v_pre + (1 - beta2) * grad ** 2
            step = gamma * m_t / (T.sqrt(v_t) + epsilon)

            updates.append((m_pre, m_t))
            updates.append((v_pre, v_t))
            updates.append((param, param - step))

        updates.append((epoch_pre, epoch_num))
        return cost, updates, log_p_x_z, KLD        
        
    def compute_rmsprop_cost_updates(self):
        ''' simple rmsprop
        '''
        _, cost, _, log_p_x_z, KLD = self.get_hidden_cost_output()

        grads = T.grad(cost, self.params)

        one = T.constant(1.0)

        def _updates(param, cache, df, beta=0.8, eta=1.e-3, epsilon=1.e-6):
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
                                                    beta=self.rmsprop_solverKwargs['beta'],
                                                    eta=self.rmsprop_solverKwargs['eta'],
                                                    epsilon=self.rmsprop_solverKwargs['epsilon'])

            updates.append(param_updates)
            updates.append(cache_updates)

        return (cost, updates, log_p_x_z, KLD)

    def sample(self, seq_len):
        if self.GMM_nll:
            return self.GMM_sample(seq_len)
        else:
            return self.CE_sample(seq_len)

    def CE_sample(self, seq_len):
        def sample_one(h):
            prior = self.prior.output_from_input(h[-1])
            prior_mu = self.prior_mu.output_from_input(prior)
            prior_logSigma = self.prior_log_sigma.output_from_input(prior)

            dev = T.as_tensor_variable(self.srng.normal((1, self.n_latent[-1])), self.h.dtype)
            # z = prior_mu + T.exp(0.5 * prior_logSigma) * dev
            z = prior_mu + prior_logSigma * dev
                                    
            phi_z = self.phi_z.output_from_input(z)

            decoder_input = utils.concatenate([phi_z, h[-1]], axis=1)
            decoder_output = self.main_decoder.output_from_input(decoder_input)
            decoder_mu = self.main_decoder_mu.output_from_input(decoder_output)

            phi_x = self.phi_x.output_from_input(decoder_mu)

            recurrent_input = T.shape_padaxis(utils.concatenate([phi_x, phi_z], axis=1), axis=2)            
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
            outputs_info=[theano.shared(1/self.h_shape[-1]*np.ones((self.h_shape[0], 1, self.h_shape[2])), broadcastable=self.h.broadcastable),
                          None],
            n_steps=seq_len,            
            )
        
        x_all0 = theano.function([], x_all[:, 0, :])()
        
        return x_all0

    def GMM_predict(self, x_in):

        def iter_step(x_step, dev, h):
            phi_x = self.phi_x.output_from_input(x_step)

            encoder_input = utils.concatenate([phi_x, h[-1]], axis=1)        
            encoder_output = self.main_encoder.output_from_input(encoder_input)

            mu = self.mu_encoder.output_from_input(encoder_output)
            logSigma = self.log_sigma_encoder.output_from_input(encoder_output)

            prior = self.prior.output_from_input(h[-1])
            prior_mu = self.prior_mu.output_from_input(prior)
            prior_logSigma = self.prior_log_sigma.output_from_input(prior)

            # z = mu + T.exp(0.5 * logSigma) * dev
            z = mu + logSigma * dev
                                    
            phi_z = self.phi_z.output_from_input(z)

            decoder_input = utils.concatenate([phi_z, h[-1]], axis=1)
            decoder_output = self.main_decoder.output_from_input(decoder_input)
            decoder_mu = self.main_decoder_mu.output_from_input(decoder_output)

            recurrent_input = T.shape_padaxis(utils.concatenate([phi_x, phi_z], axis=1), axis=2)            
            h_t = T.swapaxes(self.recurrent_layer.hidden_output_from_input(recurrent_input), 1, 2)

            decoder_logSigma = self.main_decoder_log_sigma.output_from_input(decoder_output)
            decoder_coeff = self.main_decoder_coeff.output_from_input(decoder_output)

            dev_norm = self.srng.normal((self.n_coeff, self.n_features))
            gaussians = decoder_mu.reshape((self.n_coeff, self.n_features)) + \
                        decoder_logSigma.reshape((self.n_coeff, self.n_features)) * dev_norm
            x_tilde = T.dot( decoder_coeff, gaussians)
            
            return h_t, x_tilde

        [h_n, x],inner_updates = theano.scan(
            fn=iter_step,
            sequences=[x_in,
                       T.as_tensor_variable(self.srng.normal((self.n_features, self.batch_size, self.n_latent[-1])), self.h.dtype)],
            truncate_gradient=self.bptt_truncate,
            outputs_info=[theano.shared(1/self.h_shape[-1] * np.ones(self.h_shape, dtype=self.h.dtype), broadcastable=self.h.broadcastable),
                          None],
            )

        x_out = theano.function([], x[:, 0, :])()

        return x_out

        
    def GMM_sample(self, seq_len):
        def sample_one(dev, h):
            prior = self.prior.output_from_input(h[-1])
            prior_mu = self.prior_mu.output_from_input(prior)
            prior_logSigma = self.prior_log_sigma.output_from_input(prior)

            # z = prior_mu + T.exp(0.5 * prior_logSigma) * dev
            z = prior_mu + prior_logSigma * dev
                                    
            phi_z = self.phi_z.output_from_input(z)

            decoder_input = utils.concatenate([phi_z, h[-1]], axis=1)
            decoder_output = self.main_decoder.output_from_input(decoder_input)
            decoder_mu = self.main_decoder_mu.output_from_input(decoder_output)
            decoder_logSigma = self.main_decoder_log_sigma.output_from_input(decoder_output)
            decoder_coeff = self.main_decoder_coeff.output_from_input(decoder_output)

            # Method 1
            # Treat decoder_coeff as categorical random variables
            # In this case, the weights are a probability of selection
            # of that GMM summand
            # dev_unif = self.srng.uniform()
            # decoder_coeff_prob = T.cumsum(decoder_coeff)
            # ind = T.sum(decoder_coeff_prob <= dev_unif)
            # dev_norm = self.srng.normal((1, self.n_features))
            # output_mu       = decoder_mu[:, (ind*self.n_features):((ind+1)*self.n_features)]
            # output_logSigma = decoder_logSigma[:, (ind*self.n_features):((ind+1)*self.n_features)]
            # x_tilde = output_mu + output_logSigma * dev_norm

            # Method 2            
            dev_norm = self.srng.normal((self.n_coeff, self.n_features))
            gaussians = decoder_mu.reshape((self.n_coeff, self.n_features)) + \
                        decoder_logSigma.reshape((self.n_coeff, self.n_features)) * dev_norm
            x_tilde = T.dot( decoder_coeff, gaussians)
            
            phi_x = self.phi_x.output_from_input(x_tilde)

            recurrent_input = T.shape_padaxis(utils.concatenate([phi_x, phi_z], axis=1), axis=2)            
            h_t = T.swapaxes(self.recurrent_layer.hidden_output_from_input(recurrent_input), 1, 2)

            # return [h_t[:, -2:-1, :], x_tilde]
            # for batch_size == 1
            return [h_t, x_tilde]

        [h_all, x_all], sample_updates = theano.scan(
            fn=sample_one,
            sequences=[T.as_tensor_variable(self.srng.normal((self.n_features, self.batch_size, self.n_latent[-1])), self.h.dtype)],
            outputs_info=[theano.shared(1/self.h_shape[-1]*np.ones((self.h_shape[0], 1, self.h_shape[2])), broadcastable=self.h.broadcastable),
                          None],
            n_steps=seq_len,            
            )
        
        x_all0 = theano.function([], x_all[:, 0, :])()
        
        return x_all0
    
def cross_entropy(y, y_hat):
    nll = T.sum(-1 * T.sum( y * T.log(y_hat) + (1 - y)*T.log(1 - y_hat), axis=0))
    return nll

def logsumexp(x, axis=None):
    ''' logsumexp trick
    '''
    x_max = T.max(x, axis=axis, keepdims=True)
    z = T.sum(T.log(T.sum(T.exp(x - x_max), axis=axis, keepdims=True)) + x_max)
    return z
    
def GMM_diag(y, mu, sig, coeff, mu_summand_size, n_coeff):
    ''' GMM for diagonal covariance matrix, cross-correlations = 0
    '''
    y = T.concatenate([y]*n_coeff, axis=0).T.reshape((y.shape[0],
                                                     mu_summand_size,
                                                     n_coeff))
    mu = mu.reshape((mu.shape[0],
                     mu_summand_size,
                     n_coeff))
    sig = sig.reshape((sig.shape[0],
                       mu_summand_size,
                       n_coeff))
    # Ran into a problem related to penstroke extending to edge of image,
    # in this case first or last term of inner_terms is large
    # inner_terms =  (y - mu)**2 / sig**2 + T.log(sig**2) + T.log(2 * np.pi)
    # inner_terms = inner_terms * T.lt(T.abs_(inner_terms), 100)
    # inner = -0.5 * T.sum(inner_terms, axis=1)
    inner = -0.5 * T.sum((y - mu)**2 / sig**2 + T.log(sig**2) +
                         T.log(2 * np.pi), axis=1)
    nll = -logsumexp(T.log(coeff) + inner, axis=0)

    return nll

