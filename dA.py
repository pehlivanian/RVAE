import io
import os
import sys
import timeit

import numpy

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

import theano_utils
from theano_utils import create_weight, create_bias

from hiddenlayer import HiddenLayer


class dA(object):
    ''' Denoising autoencoder, with binomial noise corruption
    '''

    def __init__(
        self,
        numpy_rng,
        theano_rng=None,
        input=None,
        n_visible=784,
        n_hidden=500,
        W=None,
        b=None,
        tie_weights=True,
        tie_biases=False,
        initial_W=None,
        initial_b=None,
        initial_W_prime=None,
        initial_b_prime=None,
        encoder_activation_fn='tanh',
        decoder_activation_fn='tanh',
        initialize_W_as_identity=False,
        initialize_W_prime_as_W_transpose=False,
        add_noise_to_W=False,
        noise_limit=0.01,
    ):
        """
        Initialize the dA class by specifying the number of visible units (the
        dimension d of the input ), the number of hidden units ( the dimension
        d' of the latent or hidden space ) and the corruption level. The
        constructor also receives symbolic variables for the input, weights and
        bias. Such a symbolic variables are useful when, for example the input
        is the result of some computations, or when weights are shared between
        the dA and an MLP layer. When dealing with SdAs this always happens,
        the dA on layer 2 gets as input the output of the dA on layer 1,
        and the weights of the dA are used in the second stage of training
        to construct an MLP.

        :type numpy_rng: numpy.random.RandomState
        :param numpy_rng: number random generator used to generate weights

        :type theano_rng: theano.tensor.shared_randomstreams.RandomStreams
        :param theano_rng: Theano random generator; if None is given one is
                     generated based on a seed drawn from `rng`

        :type input: theano.tensor.TensorType
        :param input: a symbolic description of the input or None for
                      standalone dA

        :type n_visible: int
        :param n_visible: number of visible units

        :type n_hidden: int
        :param n_hidden:  number of hidden units

        :type W: theano.tensor.TensorType
        :param W: Theano variable pointing to a set of weights that should be
                  shared belong the dA and another architecture; if dA should
                  be standalone set this to None

        :type b: theano.tensor.TensorType
        :param b: Theano variable pointing to a set of biases values (for
                  hidden units) that should be shared belong dA and another
                  architecture; if dA should be standalone set this to None

        :type tie_weights: bool
        :param tie_weights: if True, constrain visible->hidden, hidden->visible
                     weights to satisfy W' = W.transpose. This is a necessary
                     condition if the encoding, decoding steps are both linear;
                     i.e. if the activation functions using in both are the
                     identity.

        :type tie_biases: bool
        :param tie_biases: if True, constrain visible->hidden, hidden->visible
                           biases to satisfy b` = -bW`. The conditions
                             W` = W
                             b` = -bW`
                           are necessarily satisfied when the encoding, decoding
                           steps are both purely linear.
        :type initial_W: numpy.ndarray
        :param initial_W: initial value for W, or None

        :type initial_b: numpy.ndarray
        :param initial_b: initial value for b, or None

        :type initial_W_prime: numpy.ndarray
        :param initial_W_prime: initial value for W_prime, or None

        ;type initial_b_prime: numpy.ndarray
        :param initial_b_prime: initial value for b_prime, or None
        
        :type encoder_activation_fn: str
        :param encoder_activation_fn: Encoder activation function, applied to input.
                                      One of 'tanh', 'sigmoid', 'symmetric_sigmoid', 'relu', 'identity'

        :param decoder_activation_fn: Decoder activation function, applied to output of initial
                                      encoding step.
                                      One of 'tanh', 'sigmoid', 'symmetric_sigmoid', 'relu', 'identity'

        :type initialize_W_as_identity: bool
        :param initialize_W_as_identity: initialize W as identity matrix, helpful
                                         for quick convergence in purely linear case.

        :type initialize_W_prime_as_W_transpose: bool
        :param initialize_W_prime_as_W_transpose: Initialize W_prime as W.T
        
        :type add_noise_to_W: bool
        :param add_noise_to_W: add uniform noise to W, only valid for intiialize_W_as_identity=True

        :type noise_limit: float
        :param noise_limit: controls upper and lower bounds for uniform noise
                            corruption of initial W value, only valid for initialize_W_as_identity=True,
                            add_noise_t_W=True.
                                      

        """
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        
        self.tie_weights = tie_weights
        self.tie_biases = tie_biases
        self.initialize_W_as_identity = initialize_W_as_identity
        self.initialize_W_prime_as_W_transpose = initialize_W_prime_as_W_transpose        
        self.add_noise_to_W = add_noise_to_W
        self.noise_limit = noise_limit
        self.initial_W_prime = initial_W_prime
        self.initial_b_prime = initial_b_prime

        self.encoder_activation_fn = encoder_activation_fn
        self.decoder_activation_fn = decoder_activation_fn

        if not theano_rng:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))

        if not W:
            # theano.config.floatX so that the code is runable on GPU
            # initialize W with identity - usually for linear activation + deactivation
            if initial_W is None:
                assert n_visible == n_hidden
                if self.initialize_W_as_identity:
                    initial_W = numpy.eye( n_visible, dtype=theano.config.floatX )
                else:
                    initial_W = create_weight( n_visible, n_hidden, use_xavier=True)
                # Add uniform noise to initial W
                if self.add_noise_to_W:
                    assert n_visible == n_hidden
                    initial_W += numpy_rng.uniform( -1.*self.noise_limit,
                                                    self.noise_limit,
                                                    (n_visible, n_visible))
            W = theano.shared(value=initial_W, name='W', borrow=True)
        else:
            initial_W = W.get_value()

        if self.tie_weights and self.initial_W_prime:
            raise RuntimeError( 'cannot specify W_prime and also tie to W')
        if self.tie_weights and self.initialize_W_as_identity:
            raise RuntimeError( 'cannot specify W_prime and also initialize W_prime')        
        if self.tie_weights and self.initialize_W_prime_as_W_transpose:
            raise RuntimeError( 'cannot specify W_prime and also initialize W_prime')        

        if not self.tie_weights:                
            # Initialize W_prime to W`, don't constrain it
            if self.initialize_W_as_identity:
                initial_W_prime = numpy.eye( n_visible, dtype=theano.config.floatX)
                if self.add_noise_to_W:
                    initial_W_prime += numpy_rng.uniform(
                        -1.*self.noise_limit, self.noise_limit, (n_visible, n_visible))
            elif self.initialize_W_prime_as_W_transpose:
                initial_W_prime = W.get_value().T
            elif self.initial_W_prime is None:
                initial_W_prime = create_weight( n_hidden, n_visible, use_xavier=True)
            W_prime = theano.shared(value=initial_W_prime, name='W_prime', borrow=True)
        else:
            initial_W_prime = W.get_value().transpose()

        if not b:
            if initial_b is None:
                initial_b = create_bias( n_hidden, use_xavier=True, dim_input=n_visible)
            b = theano.shared(
                value=initial_b,
                name='b',
                borrow=True
            )
        else:
            initial_b = b.get_value()


        if not self.tie_biases:
            initial_b_prime = create_bias( n_visible, use_xavier=True, dim_input=n_visible)
            b_prime = theano.shared(
                value=initial_b_prime,
                name='b_prime',
                borrow=True
                )
        else:
            initial_b_prime = b_prime.get_value().copy()
            
        self.initial_W = initial_W.copy()
        self.initial_W_prime = initial_W_prime.copy()
        self.initial_b = initial_b.copy()
        self.initial_b_prime = initial_b_prime.copy()
        
        self.W = W
        self.b = b

        if self.tie_weights:
            self.W_prime = self.W.T
        else:
            self.W_prime = W_prime

        if self.tie_biases:
            self.b_prime = -T.dot( self.b, self.W_prime)        
        else:
            self.b_prime = b_prime
            
        self.theano_rng = theano_rng
        
        if input is None:
            self.x = T.dmatrix(name='input')
        else:
            self.x = input

        # Unsupervised, "bombe" encoding
        # see, e.g. http://en.wikipedia.org/wiki/Bombe#Structure
        self.y = self.x 

        weight_params = [self.W]
        
        if not self.tie_weights:
            print('Untied weights; adding W_prime to parameter set')
            weight_params += [self.W_prime]
        if self.tie_biases:
            b_params = [self.b]
        else:
            b_params = [self.b, self.b_prime]
        self.params = weight_params + b_params

    def _get_corrupted_input(self, input, corruption_level):
        '''This function keeps ``1-corruption_level`` entries of the inputs the
        same and zero-out randomly selected subset of size 'coruption_level'
        '''
        return self.theano_rng.binomial(size=input.shape, n=1,
                                        p=1 - corruption_level,
                                        dtype=theano.config.floatX) * input

    def output_from_input(self, input, corruption_level=0.):
        ''' Outputs from hidden values for a given input
        '''
        tilde_x = self._get_corrupted_input(input, corruption_level)
        y = self._get_hidden_values(tilde_x)
        return y

    def output(self, corruption_level=0.):
        ''' Outputs hidden values
        '''
        return self.output_from_input( self.x, corruption_level=corruption_level)

    def predict_from_input(self, input, corruption_level=0.):
        ''' Unsupervised problem
        '''
        tilde_input = self._get_corrupted_input( input, corruption_level)
        y = self._get_hidden_values(tilde_input)
        z = self._get_reconstructed_input(y)
        return z

    def predict( self, corruption_level=0.):
        ''' Prediction for the unsupervised problem
        '''
        return self.predict_from_input( self.x, corruption_level=corruption_level)
    
    def _get_hidden_values(self, input):
        if self.encoder_activation_fn == 'tanh':
            return T.tanh(T.dot(input, self.W) + self.b)
        elif self.encoder_activation_fn == 'sigmoid':
            return T.nnet.sigmoid(T.dot(input, self.W) + self.b)
        elif self.encoder_activation_fn == 'relu':
            return T.nnet.relu(T.dot(input, self.W) + self.b)
        elif self.encoder_activation_fn == 'symmetric_sigmoid':
            return -1 + 2 * T.nnet.sigmoid(T.dot(input, self.W) + self.b)
        elif self.encoder_activation_fn == 'softplus':
            return T.nnet.softplus
        elif self.encoder_activation_fn == 'softmax':
            return T.nnet.softmax
        elif self.encoder_activation_fn == 'identity':
            return T.dot(input, self.W) + self.b            
        else:
            raise RuntimeError('encoder_activation_fn %s not supported' %
                               (self.encoder_activation_fn,))

    def _get_reconstructed_input(self, hidden):
        if self.decoder_activation_fn == 'tanh':
            return T.tanh(T.dot(hidden, self.W_prime) + self.b_prime)
        elif self.decoder_activation_fn == 'sigmoid':
            return T.nnet.sigmoid(T.dot(hidden, self.W_prime) + self.b_prime)
        elif self.decoder_activation_fn == 'relu':
            return T.nnet.relu(T.dot(hidden, self.W_prime) + self.b_prime)
        elif self.decoder_activation_fn == 'symmetric_sigmoid':
            return -1 + 2 * T.nnet.sigmoid(T.dot(hidden, self.W_prime) + self.b_prime)
        elif self.decoder_activation_fn == 'softplus':
            return T.nnet.softplus(T.dot(hidden, self.W_prime) + self.b_prime)
        elif self.decoder_activation_fn == 'softmax':
            return T.nnet.softmax(T.dot(hidden, self.W_prime) + self.b_prime)
        elif self.decoder_activation_fn == 'identity':
            return T.dot(hidden, self.W_prime) + self.b_prime
        else:
            raise RuntimeError('decoder_activation_fn %s not supported' %
                               (self.decoder_activation_fn,))

    def describe(self, train_set_x, title=None):
        ''' Some basic information about the stacked denoising
            autoencoder
        '''

        buff = io.StringIO('')

        title = title+' ' if title else ''
        print(title, file=buff)

        y_hat = T.matrix('y_hat')
        y_hat = self.predict()
        
        # No updates for the following
        predict = theano.function( [], y_hat, givens={ self.x: train_set_x })
        mse = numpy.mean( numpy.sqrt( numpy.sum( ( train_set_x.get_value() - predict()) ** 2, axis=1)))

        mlp_layer_hdr1 = 'LAYERS'
        mlp_layer_hdr2 = '=' * len(mlp_layer_hdr1)
        print( '\n'.join( [ mlp_layer_hdr1, mlp_layer_hdr2 ] ), file=buff )
        print( 'cost: %2.6f' % (mse, ), file=buff)        
        print( '\n'.join( ['layer: %-2d W:%3d x %-3d    b: 1x%s |W-W0|_2: %-4.4f |b-b0|_2: %-4.4f' %
                           (i,(*layer.W.get_value().shape), (*layer.b.get_value().shape),
                            numpy.sqrt(numpy.sum( (layer.W.get_value() - layer.initial_W)**2) ),
                            numpy.sqrt(numpy.sum( (layer.b.get_value() - layer.initial_b)**2) ) )
                           for i,layer in enumerate([self])] ), file=buff )

        return buff.getvalue()
        
def _prop(a, n):
    return a if hasattr(a, '__iter__') and not isinstance( a, str) else [a] * n

def _get_activation(fn_name):
    if fn_name == 'tanh':
        return T.tanh
    elif fn_name == 'sigmoid':
        return T.nnet.sigmoid
    elif fn_name == 'relu':
        return T.nnet.relu
    elif fn_name == 'symmetric_sigmoid':
        return lambda x: -1 + 2 * T.nnet.sigmoid(x)
    elif fn_name == 'softplus':
        return T.nnet.softplus
    elif fn_name == 'softmax':
        return T.nnet.softmax
    elif fn_name == 'identity':
        return lambda x: x
    else:
        raise RuntimeError( 'activation function %s not supported' % (fn_name, ))

class SdA(object):
    ''' Stacked auto-encoder class, for unsupervised problem. It will produce an
        unsymmetric stack of denoising autoencoders with sizes:

              n_visible -> h[0] -> ... -> h[n-1]

        or a symmetric stack if symmetric=True, with sizes:
        
              n_visible -> h[0] -> ... -> h[n-2] -> h[n-1] -> h[n-2] -> ... -> h[0]

        where h[i] = dA_layers_sizes[i], the size of the ith hidden layer.

        During the pretraining step, each layer will be turned in on itself for the
        unsupervised learning task. The pre-trained weights and biases W,b are then
        used as starting points for the finetuning step, which turns the entire
        set in on itself.

        The computational net is a sequence of hidden layers with dimensions
        specified as above. A parallel layer or denoising autoencoders with the
        same dimensions is constructed, only for use during the pretraining step.
        The class interface exposes the hidden layer attributes.

        Unlike the single denoising autoencoder, we need to specify a solver type
        as it used in the pretraining phase. The finetuning (actual training of the
        end-to-end model) is handled by a class helper or decorator class.
        
        Solver types of 'rmsprop' (default) or 'gd', etc.,  can be specified for
        pretraining and finetuning.
        
    '''

    # Many of the following input parameters can be specified as an iterable,
    # to be applied to each layer in succession, or a single value to be broadcast
    # to all layers
    def __init__(self,
                 numpy_rng,
                 theano_rng=None,
                 input=None,
                 n_visible=784,
                 dA_layers_sizes=[392, 196],
                 symmetric=True,
                 global_decoder_activation_fn='tanh',
                 corruption_levels=None,                  # Iterable, or value broadcast to all layers
                 tie_weights=True,                        # Iterable, or value broadcast to ll layers 
                 tie_biases=False,                        # Iterable, or value broadcast to all layers
                 encoder_activation_fn='tanh',            # Iterable, or value broadcast to all layers
                 decoder_activation_fn='tanh',            # Iterable, or value broadcast to all layers
                 initial_W=None,                          # Iterable, or value broadcast to all layers
                 initial_W_prime=None,                    # Iterable, or value broadcast to all layers
                 initialize_W_as_identity=False,          # Iterable, or value broadcast to all layers
                 initialize_W_prime_as_W_transpose=False, # Iterable, or value broadcast to all layers
                 add_noise_to_W=True,                     # Iterable, or value broadcast to all layers
                 noise_limit=0.01,                        # Iterable, or value broadcast to all layers
                 solver_type='rmsprop',                   
                 predict_modifier_type=None,              
                 solver_kwargs={}                         # Iterable, or value broadcast to all layers
            ):

        n = len(dA_layers_sizes)
        self.symmetric = symmetric

        self.initialize_W_as_identity = initialize_W_as_identity
        self.initialize_W_prime_as_W_transpose = initialize_W_prime_as_W_transpose
        self.add_noise_to_W = add_noise_to_W
        self.noise_limit = noise_limit

        self.dA_encoder_tie_weights = _prop(tie_weights, n)
        self.dA_encoder_tie_biases = _prop(tie_biases, n)
        self.dA_encoder_activation_fn = _prop(encoder_activation_fn, n)
        self.dA_encoder_initialize_W_as_identity = _prop(initialize_W_as_identity, n)
        self.dA_encoder_initialize_W_prime_as_W_transpose = _prop(initialize_W_prime_as_W_transpose, n)
        self.dA_encoder_add_noise_to_W = _prop(add_noise_to_W, n)
        self.dA_encoder_noise_limit = _prop(noise_limit, n)

        if symmetric:
            self.dA_decoder_tie_weights = self.dA_encoder_tie_weights[::-1]
            self.dA_decoder_tie_biases = self.dA_encoder_tie_biases[::-1]
            self.dA_decoder_activation_fn = _prop(decoder_activation_fn, n)        
            self.dA_decoder_initialize_W_as_identity = self.dA_encoder_initialize_W_as_identity[::-1]
            self.dA_decoder_initialize_W_prime_as_transpose = self.dA_encoder_initialize_W_prime_as_W_transpose
            self.dA_decoder_add_noise_to_W = self.dA_encoder_add_noise_to_W[::-1]
            self.dA_decoder_noise_limit = self.dA_encoder_noise_limit[::-1]
        else:
            self.dA_decoder_tie_weights = _prop(self.dA_encoder_tie_weights[-1], 1)
            self.dA_decoder_tie_biases = _prop(self.dA_encoder_tie_biases[-1], 1)
            self.dA_decoder_activation_fn = _prop(decoder_activation_fn, n)
            self.dA_decoder_initialize_W_as_identity = _prop(self.dA_encoder_initialize_W_as_identity[-1], 1)
            self.dA_decoder_add_noise_to_W = _prop(self.dA_encoder_add_noise_to_W[0], 1)
            self.dA_decoder_noise_limit = _prop(self.dA_encoder_noise_limit[-1], 1)
            
        self.global_decoder_activation_fn = global_decoder_activation_fn
        
        # self.params will contain only the parameters for the mlp layers.
        # This is because we will pretrain the dA layers by turning them
        # in on themselves in an unsupervised manner. Once the pretraining
        # is done, with a solver for each pretrained layer, we use a global
        # solver, consuming self.params, for the mlp layer pipeline. dA layers
        # are not referred to after that.
        
        self.dA_layers = []
        self.mlp_layers = []
        self.dA_params = []
        self.mlp_params = []
        self.params = []
        
        self.n_encoder_layers = len(dA_layers_sizes)

        if symmetric:
            self.n_decoder_layers = self.n_encoder_layers
        else:
            self.n_decoder_layers = 0
        self.n_layers = self.n_encoder_layers + self.n_decoder_layers

        self.n_visible = n_visible
        self.n_hidden = dA_layers_sizes[-1]
        
        self.dA_encoder_layers_sizes = dA_layers_sizes
        if symmetric:
            self.dA_decoder_layers_sizes = self.dA_encoder_layers_sizes[::-1]
        else:
            # no decoder layers
            self.dA_decoder_layers_sizes = [0]            

        self.encoder_corruption_levels = _prop(corruption_levels, n) if corruption_levels else _prop(0., n)
        if symmetric:
            self.decoder_corruption_levels = self.encoder_corruption_levels[::-1]
        else:
            self.decoder_corruption_levels = []
        self.corruption_levels = self.encoder_corruption_levels + self.decoder_corruption_levels

        assert self.n_layers > 0
        assert len(self.encoder_corruption_levels) == self.n_encoder_layers
        assert len(self.corruption_levels) == self.n_layers

        if not theano_rng:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))

        if solver_type == 'gd':
            self.solver = theano_utils.gd_solver
        elif solver_type == 'rmsprop':
            self.solver = theano_utils.rmsprop_solver
        else:
            raise RuntimeError('Solver type %s not supported' & (solver_type,))
        self.solver_kwargs = solver_kwargs

        self.predict_modifier_type = predict_modifier_type
            
        if input is None:
            self.x = T.dmatrix(name='input')
        else:
            self.x = input

        # W_prime
        if initial_W_prime is None:
            if self.initialize_W_as_identity:
                initial_W_prime = numpy.eye( n_visible, dtype=theano.config.floatX)
                if self.add_noise_to_W:
                    initial_W_prime += numpy_rng.uniform(
                        -1.*self.noise_limit, self.nosie_limit, (n_visible, n_visible))
            elif self.initialize_W_prime_as_W_transpose:
                initial_W_prime = W.get_value().T
            else:
                initial_W_prime = create_weight( self.n_hidden, self.n_visible, use_xavier=True)
        self.W_prime = theano.shared(value=initial_W_prime, name='W_prime', borrow=True)
        
        self.initial_W = initial_W.copy() if initial_W else None
        self.initial_W_prime = initial_W_prime.copy()
        
        # b_prime
        initial_b_prime = create_bias( n_visible, use_xavier=True, dim_input=n_visible)
        self.b_prime = theano.shared(value=initial_b_prime, name='b_prime', borrow=True)

        self.params = [self.W_prime, self.b_prime]

        # Unsupervised "bombe" training
        self.y = self.x

        ########################
        # START ENCODER LAYERS #
        ########################
        for i in range(self.n_encoder_layers):
            if i == 0:
                input_size = n_visible
                mlp_layer_input = self.x
                dA_layer_input = self.x
            else:
                input_size = self.dA_encoder_layers_sizes[i-1]
                mlp_layer_input = self.mlp_layers[-1].output()
                dA_layer_input = self.dA_layers[-1].output()
            output_size = self.dA_encoder_layers_sizes[i]

            mlp_activation = _get_activation(self.dA_encoder_activation_fn[i])

            # a perceptron layer, and one parallel denoising autoencoder layer are set up
            # The dA layer is only used for the greedy pretraining stage, with weights
            # shared with the mlp layer.
            dA_layer = dA(numpy_rng=numpy_rng,
                          theano_rng=theano_rng,
                          input=dA_layer_input,
                          n_visible=input_size,
                          n_hidden=output_size,
                          tie_weights=self.dA_encoder_tie_weights[i],
                          tie_biases=self.dA_encoder_tie_biases[i],
                          encoder_activation_fn=self.dA_encoder_activation_fn[i],
                          decoder_activation_fn=self.dA_decoder_activation_fn[i],
                          initial_W=create_weight(input_size, output_size, use_xavier=True),
                          initialize_W_as_identity=self.dA_encoder_initialize_W_as_identity[i],
                          initialize_W_prime_as_W_transpose=self.dA_encoder_initialize_W_prime_as_W_transpose,
                          add_noise_to_W=self.dA_encoder_add_noise_to_W[i],
                          noise_limit=self.dA_encoder_noise_limit[i],
                          )

            mlp_layer = HiddenLayer(rng=numpy_rng,
                                    input=mlp_layer_input,
                                    n_in=input_size,
                                    n_out=output_size,
                                    activation=mlp_activation,
                                    W=dA_layer.W,
                                    b=dA_layer.b
                                    )            

            self.mlp_layers.append(mlp_layer)
            self.dA_layers.append(dA_layer)            
            
            self.mlp_params.extend(mlp_layer.params)
            self.dA_params.extend(dA_layer.params)

        #####################################################
        # Decoder layers: (hidden_layer X n_decoder_layers) #
        #####################################################
        for i in range(self.n_decoder_layers):
            if i == self.n_decoder_layers - 1:
                output_size = self.n_visible
            else:
                output_size = self.dA_decoder_layers_sizes[i+1]                
            input_size = self.dA_decoder_layers_sizes[i]    
            mlp_layer_input = self.mlp_layers[-1].output()
            dA_layer_input = self.dA_layers[-1].output()

            mlp_activation = _get_activation(self.dA_decoder_activation_fn[i])            

            dA_layer = dA(numpy_rng=numpy_rng,
                          theano_rng=theano_rng,
                          input=dA_layer_input,
                          n_visible=input_size,
                          n_hidden=output_size,
                          tie_weights=self.dA_decoder_tie_weights[i],
                          tie_biases=self.dA_decoder_tie_biases[i],
                          encoder_activation_fn=self.dA_encoder_activation_fn[i],
                          decoder_activation_fn=self.dA_decoder_activation_fn[i],
                          initial_W=create_weight(input_size, output_size, use_xavier=True),
                          initial_W_prime=self.initial_W_prime,
                          initialize_W_as_identity=self.dA_decoder_initialize_W_as_identity[i],
                          initialize_W_prime_as_W_transpose=self.dA_encoder_initialize_W_prime_as_W_transpose[i],
                          add_noise_to_W=self.dA_decoder_add_noise_to_W[i],
                          noise_limit=self.dA_decoder_noise_limit[i],
                          )

            mlp_layer = HiddenLayer(rng=numpy_rng,
                                    input=mlp_layer_input,
                                    n_in=input_size,
                                    n_out=output_size,
                                    activation=mlp_activation,
                                    W=dA_layer.W,
                                    b=dA_layer.b
                                    )                        

            self.mlp_layers.append(mlp_layer)
            self.dA_layers.append(dA_layer)

            self.mlp_params.extend(mlp_layer.params)
            self.dA_params.extend(dA_layer.params)

        self.params += self.mlp_params

    def pretraining_functions(self, train_set_x, batch_size, update=True):
        pretrain_fns = []
        
        index = T.lscalar('index')
        corruption_level = T.scalar('corrpution_level')

        for num, layer in enumerate(self.dA_layers):
            if self.predict_modifier_type == 'negative_feedback':
                solver = theano_utils.negative_feedback(self.solver(layer, **self.solver_kwargs))
            else:
                solver = self.solver(layer, **self.solver_kwargs)
            cost, updates = solver.compute_cost_updates()

            # The givens specification makes this hard to distribute as the
            # pretraining functions must be called in ascending order
            # Replace the initial input to the chain of layers with the
            # givens specification will propagate correct input to the current
            # layer
            if not update:
                updates = None
            pretrain_fn = theano.function(
                inputs=[index],
                outputs=cost,
                updates=updates,
                givens= { self.x: train_set_x[index*batch_size : (index+1)*batch_size] }
            )
            pretrain_fns.append(pretrain_fn)

        return pretrain_fns

    def output_from_input(self, input):
        output = input
        for layer in self.mlp_layers:
            output = layer.output_from_input(output)
        return output

    def output(self):
        return self.output_from_input(self.x)

    def _get_corrupted_input(self, input, corruption_level):
        '''This function keeps ``1-corruption_level`` entries of the inputs the
        same and zero-out randomly selected subset of size 'coruption_level'
        '''
        return self.theano_rng.binomial(size=input.shape, n=1,
                                        p=1 - corruption_level,
                                        dtype=theano.config.floatX) * input

    def predict(self):
        # We can't just turn the last dA in on itself, as the
        # reconstructed input should match the input of the first
        # dA.
        y = self.mlp_layers[-1].output()
        z = self._get_reconstructed_input(y)
        return z

    def _get_reconstructed_input(self, hidden):
        if self.global_decoder_activation_fn == 'tanh':
            return T.tanh(T.dot(hidden, self.W_prime) + self.b_prime)
        elif self.global_decoder_activation_fn == 'sigmoid':
            return T.nnet.sigmoid(T.dot(hidden, self.W_prime) + self.b_prime)
        elif self.global_decoder_activation_fn == 'relu':
            return T.nnet.relu(T.dot(hidden, self.W_prime) + self.b_prime)
        elif self.global_decoder_activation_fn == 'symmetric_sigmoid':
            return -1 + 2 * T.nnet.sigmoid(T.dot(hidden, self.W_prime) + self.b_prime)
        elif self.global_decoder_activation_fn == 'softplus':
            return T.nnet.softplus( T.dot(hidden, self.W_prime) + self.b_prime)
        elif self.global_decoder_activation_fn == 'softmax':
            return T.nnet.softmax( T.dot(hidden, self.W_prime) + self.b_prime)        
        elif self.global_decoder_activation_fn == 'identity':
            return T.dot(hidden, self.W_prime) + self.b_prime
        else:
            raise RuntimeError('global_decoder_activation_fn %s not supported' %
                               (self.global_decoder_activation_fn,))

    def mlp_layerOutputs(self, train_set_x):
        predict_fns = []
        
        for i in range(self.n_layers):
            y_hat = T.matrix(name='y_hat_%s' % i)
            y_hat = self.mlp_layers[i].output()
            output_fn = theano.function(
                [],
                y_hat,
                givens={ self.x: train_set_x } )
            predict_fns.append(output_fn)
        return predict_fns

    def dA_layerOutputs(self, train_set_x):
        predict_fns = []

        for i in range(self.n_layers):
            y_hat = T.matrix(name='y_hat_%s' % i)
            y_hat = self.dA_layers[i].output()
            output_fn = theano.function(
                [],
                y_hat,
                givens={ self.x: train_set_x } )
            predict_fns.append(output_fn)

        return predict_fns
            

    def describe(self, train_set_x, title=None):
        ''' Some basic information about the stacked denoising
            autoencoder
        '''

        buff = io.StringIO('')

        title = title+' ' if title else ''
        title = '\n'+title
        sym_str = 'Symmetric' if self.symmetric else 'Non-symmetric'
        print(title + '%s case' % (sym_str, ), file=buff)

        pretraining_fns = self.pretraining_functions(train_set_x=train_set_x,
                                                     batch_size=train_set_x.get_value().shape[0], update=False)
        pretraining_costs = [pretraining_fn(0) for pretraining_fn in pretraining_fns]

        y_hat = T.matrix('y_hat')
        y_hat = self.predict()
        predict = theano.function( [], y_hat, givens={ self.x: train_set_x })
        mse = numpy.mean( numpy.sqrt( numpy.sum( ( train_set_x.get_value() - predict()) ** 2, axis=1)))

        da_layer_hdr1 = 'DA LAYERS'
        da_layer_hdr2 = '=' * len(da_layer_hdr1)
        print( '\n'.join( [ da_layer_hdr1, da_layer_hdr2 ] ), file=buff )
        print( '\n'.join( ['layer: %-2d W:%3d x %-3d    b: 1x%s   pretrain cost: %-2.6f |W-W0|_2: %-4.4f |b-b0|_2: %-4.4f' %
                           (i,(*layer_info[0].W.get_value().shape), (*layer_info[0].b.get_value().shape), layer_info[1],
                            numpy.sqrt(numpy.sum( (layer_info[0].W.get_value() - layer_info[0].initial_W)**2 )),
                            numpy.sqrt(numpy.sum( (layer_info[0].b.get_value() - layer_info[0].initial_b)**2 )))
                           for i,layer_info in enumerate(zip(self.dA_layers, pretraining_costs))] ), file=buff )

        mlp_layer_hdr1 = 'MLP LAYERS'
        mlp_layer_hdr2 = '=' * len(mlp_layer_hdr1)
        print( '\n'.join( [ '', mlp_layer_hdr1, mlp_layer_hdr2 ] ), file=buff )
        print( 'finetune cost:  %2.6f' % (mse, ), file=buff)        
        print( '\n'.join( ['layer: %-2d W:%3d x %-3d    b: 1x%s |W-W0|_2: %-4.4f |b-b0|_2: %-4.4f' %
                           (i,(*layer.W.get_value().shape), (*layer.b.get_value().shape),
                            numpy.sqrt(numpy.sum( (layer.W.get_value() - layer.initial_W)**2 ) ),
                            numpy.sqrt(numpy.sum( (layer.b.get_value() - layer.initial_b)**2)) )
                           for i,layer in enumerate(self.mlp_layers)] ), file=buff )

        return buff.getvalue()
