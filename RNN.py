from __future__ import division

import numpy as np
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

import pickle
from collections import OrderedDict


SEED = 483
epsilon = 1e-8

class GRU(object):
    ''' GRU implementation
    '''
    def __init__(self,
                 rng,
                 input,
                 output,
                 n_visible,
                 n_hidden,
                 n_layers,
                 batch_size,
                 theano_rng=None,
                 bptt_truncate=-1
                ):
        
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.bptt_truncate = bptt_truncate

        self.params = []

        self.numpy_rng = rng
        if not theano_rng:
            theano_rng = RandomStreams(self.numpy_rng.randint(2 ** 30))

        self.x = input
        self.batch_size = batch_size

        if output is None:
            self.y = T.dvector(name='output')
        else:
            self.y = output

        U0_shape = (self.n_hidden, self.n_visible)
        W0_shape = (self.n_hidden, self.n_hidden)
        Un_shape = (self.n_hidden, self.n_hidden)
        Wn_shape = (self.n_hidden, self.n_hidden)
        b_shape =  (self.n_hidden, self.batch_size)

        V_shape =  (self.n_visible, self.n_hidden)
        c_shape =  (self.n_visible, self.batch_size)
        
        initial_Uzn = np.zeros((-1 + self.n_layers,) + Un_shape)
        initial_Urn = np.zeros((-1 + self.n_layers,) + Un_shape)
        initial_Uhn = np.zeros((-1 + self.n_layers,) + Un_shape)
        initial_Wzn = np.zeros((-1 + self.n_layers,) + Wn_shape)
        initial_Wrn = np.zeros((-1 + self.n_layers,) + Wn_shape)
        initial_Whn = np.zeros((-1 + self.n_layers,) + Wn_shape)
        
        initial_bzn = np.zeros((-1 + self.n_layers,) + b_shape)
        initial_brn = np.zeros((-1 + self.n_layers,) + b_shape)
        initial_bhn = np.zeros((-1 + self.n_layers,) + b_shape)

        # initial layer, shapes are generally (visible x hidden),
        # (hidden x hidden) after that
        layer = 0
        initial_Uz0 = self.numpy_rng.uniform( -np.sqrt(1./self.n_visible),
                                              np.sqrt(1./self.n_visible),
                                              U0_shape)
        initial_Ur0 = self.numpy_rng.uniform( -np.sqrt(1./self.n_visible),
                                              np.sqrt(1./self.n_visible),
                                              U0_shape)
        initial_Uh0 = self.numpy_rng.uniform( -np.sqrt(1./self.n_visible),
                                              np.sqrt(1./self.n_visible),
                                              U0_shape)
        initial_Wz0 = self.numpy_rng.uniform( -np.sqrt(1./self.n_visible),
                                              np.sqrt(1./self.n_visible),
                                              W0_shape)
        initial_Wr0 = self.numpy_rng.uniform( -np.sqrt(1./self.n_visible),
                                              np.sqrt(1./self.n_visible),
                                              W0_shape)
        initial_Wh0 = self.numpy_rng.uniform( -np.sqrt(1./self.n_visible),
                                              np.sqrt(1./self.n_visible),
                                              W0_shape)
        initial_bz0 = np.zeros(b_shape)
        initial_br0 = np.zeros(b_shape)
        initial_bh0 = np.zeros(b_shape)
        
        for layer in range(-1 + self.n_layers):
            initial_Uzn[layer] = self.numpy_rng.uniform( -np.sqrt(1./self.n_visible),
                                                         np.sqrt(1./self.n_visible),
                                                         Un_shape)
            initial_Urn[layer] = self.numpy_rng.uniform( -np.sqrt(1./self.n_visible),
                                                         np.sqrt(1./self.n_visible),
                                                         Un_shape)
            initial_Uhn[layer] = self.numpy_rng.uniform( -np.sqrt(1./self.n_visible),
                                                        np.sqrt(1./self.n_visible),
                                                        Un_shape)
            initial_Wzn[layer] = self.numpy_rng.uniform( -np.sqrt(1./self.n_visible),
                                                        np.sqrt(1./self.n_visible),
                                                        Wn_shape)
            initial_Wrn[layer] = self.numpy_rng.uniform( -np.sqrt(1./self.n_visible),
                                                        np.sqrt(1./self.n_visible),
                                                        Wn_shape)
            initial_Whn[layer] = self.numpy_rng.uniform( -np.sqrt(1./self.n_visible),
                                                        np.sqrt(1./self.n_visible),
                                                        Wn_shape)
            initial_bzn[layer] = np.zeros(b_shape)
            initial_brn[layer] = np.zeros(b_shape)
            initial_bhn[layer] = np.zeros(b_shape)
            
        initial_V = self.numpy_rng.uniform( -np.sqrt(1./self.n_visible),
                                            np.sqrt(1./self.n_visible),
                                            V_shape)
        initial_c = np.zeros(c_shape)

        self.Uz0 = theano.shared(name='Uz0', value=initial_Uz0.astype(theano.config.floatX))
        self.Ur0 = theano.shared(name='Ur0', value=initial_Ur0.astype(theano.config.floatX))
        self.Uh0 = theano.shared(name='Uh0', value=initial_Uh0.astype(theano.config.floatX))
        self.Wz0 = theano.shared(name='Wz0', value=initial_Wz0.astype(theano.config.floatX))
        self.Wr0 = theano.shared(name='Wr0', value=initial_Wr0.astype(theano.config.floatX))
        self.Wh0 = theano.shared(name='Wh0', value=initial_Wh0.astype(theano.config.floatX))
        self.bz0 = theano.shared(name='bz0', value=initial_bz0.astype(theano.config.floatX))
        self.br0 = theano.shared(name='br0', value=initial_br0.astype(theano.config.floatX))
        self.bh0 = theano.shared(name='bh0', value=initial_bh0.astype(theano.config.floatX))
        self.Uzn = theano.shared(name='Uzn', value=initial_Uzn.astype(theano.config.floatX))
        self.Urn = theano.shared(name='Urn', value=initial_Urn.astype(theano.config.floatX))
        self.Uhn = theano.shared(name='Uhn', value=initial_Uhn.astype(theano.config.floatX))
        self.Wzn = theano.shared(name='Wzn', value=initial_Wzn.astype(theano.config.floatX))
        self.Wrn = theano.shared(name='Wrn', value=initial_Wrn.astype(theano.config.floatX))
        self.Whn = theano.shared(name='Whn', value=initial_Whn.astype(theano.config.floatX))
        self.bzn = theano.shared(name='bzn', value=initial_bzn.astype(theano.config.floatX))
        self.brn = theano.shared(name='brn', value=initial_brn.astype(theano.config.floatX))
        self.bhn = theano.shared(name='bhn', value=initial_bhn.astype(theano.config.floatX))
        
        self.V = theano.shared(name='V', value=initial_V.astype(theano.config.floatX))
        self.c = theano.shared(name='c', value=initial_c.astype(theano.config.floatX))

        self.hidden_params = self.params + [self.Uz0, self.Ur0, self.Uh0,
                                            self.Uzn, self.Urn, self.Uhn,
                                            self.Wz0, self.Wr0, self.Wh0,
                                            self.Wzn, self.Wrn, self.Whn,
                                            self.bz0, self.br0, self.bh0,
                                            self.bzn, self.brn, self.bhn]
        self.params = self.hidden_params + [self.V, self.c]
        
    def _get_hidden_values(self, input):
        def forward_step(s_t0_in, s_t0_prev, s_tn_prev):

            # Layer 0
            z_t0 = T.nnet.sigmoid(T.dot(self.Uz0, s_t0_in) + T.dot(self.Wz0, s_t0_prev)        + self.bz0)
            r_t0 = T.nnet.sigmoid(T.dot(self.Ur0, s_t0_in) + T.dot(self.Wr0, s_t0_prev)        + self.br0)
            h_t0 = T.tanh(        T.dot(self.Uh0, s_t0_in) + T.dot(self.Wh0, s_t0_prev * r_t0) + self.bh0)
            s_t0 = (T.ones_like(z_t0) - z_t0) * h_t0 + z_t0 * s_t0_prev

            def iter_step(lyr, s_t_in, s_tn_prev):
                z_t = T.nnet.sigmoid(T.dot(self.Uzn[lyr], s_t_in) + T.dot(self.Wzn[lyr], s_tn_prev[lyr])       + self.bzn[lyr])
                r_t = T.nnet.sigmoid(T.dot(self.Urn[lyr], s_t_in) + T.dot(self.Wrn[lyr], s_tn_prev[lyr])       + self.brn[lyr])
                h_t = T.tanh(        T.dot(self.Uhn[lyr], s_t_in) + T.dot(self.Whn[lyr], s_tn_prev[lyr] * r_t) + self.bhn[lyr])
                s_t = (T.ones_like(z_t) - z_t) * h_t + z_t * s_tn_prev[lyr]

                return s_t

            s_tn, inner_updates = theano.scan(
                fn=iter_step,
                sequences=[T.arange(-1 + self.n_layers)],
                outputs_info=[s_t0],
                non_sequences=[s_tn_prev]
                )

            s_all = T.concatenate([T.shape_padaxis(s_t0, axis=0), s_tn])
            s_t = s_tn[-1]
            o_t = T.dot(self.V, s_t) + self.c
 
            return [s_t0, s_tn, o_t, s_all]
        
        [s, s_prev, o, s_prev_all], outer_updates = theano.scan(
            fn=forward_step,
            sequences=input,
            truncate_gradient=self.bptt_truncate,
            outputs_info=[theano.shared(value=(1./self.n_hidden)*np.ones((self.n_hidden, self.batch_size))),
                          theano.shared(value=(1./self.n_hidden)*np.ones((-1 + self.n_layers, self.n_hidden, self.batch_size))),
                          None,
                          None]
            )
        
        return o, s_prev_all[-1]
        
    def predict_from_input(self, input):
        o, _ = self._get_hidden_values(input)
        return o

    def predict(self):
        o = self.predict_from_input(self.x)
        return o

    def output_from_input(self, input):
        o = self.predict_from_input(input)
        return o
        
    def output(self):
        return self.output_from_input(self.x)

    def hidden_output_from_input(self, input):
        _, h = self._get_hidden_values(input)
        return h

    def hidden_output(self):
        h = self.hidden_output_from_input(self.x)
        return h

    def number_of_params(self):
        return sum(np.prod(param.get_value().shape) for param in self.params)

