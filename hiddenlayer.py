"""
Standard hidden layer

.. math::

    f(x) = G( b^{(2)} + W^{(2)}( s( b^{(1)} + W^{(1)} x))),

References:

    - textbooks: "Pattern Recognition and Machine Learning" -
                 Christopher M. Bishop, section 5

"""

from __future__ import print_function

__docformat__ = 'restructedtext en'


import os
import sys
import timeit

import numpy

import theano
import theano.tensor as T


from theano_utils import create_weight, create_bias


# start-snippet-1
class HiddenLayer(object):
    def __init__(self,
                 rng,
                 input,
                 n_in,
                 n_out,
                 output=None,
                 W=None,
                 b=None,
                 initial_W=None,
                 initial_b=None,
                 activation=T.tanh):
        """
        Typical hidden layer of a MLP: units are fully-connected and have
        sigmoidal activation function. Weight matrix W is of shape (n_in,n_out)
        and the bias vector b is of shape (n_out,).

        NOTE : The nonlinearity used here is tanh

        Hidden unit activation is given by: tanh(dot(input,W) + b)

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dmatrix
        :param input: a symbolic tensor of shape (n_examples, n_in)

        :type n_in: int
        :param n_in: dimensionality of input

        :type n_out: int
        :param n_out: number of hidden units

        :type activation: theano.Op or function
        :param activation: Non linearity to be applied in the hidden
                           layer
        """
        self.n_visible = n_in
        self.n_hidden = n_out

        if input is None:
            self.x = T.dmatrix('x')
        else:
            self.x = input

        if output is None:
            self.y = T.dmatrix('y')
        else:
            self.y = output

        # end-snippet-1

        if W is None:
            W_values = create_weight( n_in, n_out, use_xavier=True)
            W = theano.shared(value=W_values, name='W', borrow=True)
            initial_W = W_values
        else:
            initial_W = W.get_value()

        if b is None:
            b_values = create_bias( n_out, use_xavier=True, dim_input=n_in)
            b = theano.shared(value=b_values, name='b', borrow=True)

            initial_b = b_values
        else:
            initial_b = b.get_value()

        self.initial_W = initial_W
        self.initial_b = initial_b
        
        self.W = W
        self.b = b

        self.activation = activation

        # parameters of the model
        self.params = [self.W, self.b]

    def output_from_input(self, input):
        lin_output = T.dot(input, self.W) + self.b
        y = lin_output if self.activation is None else self.activation(lin_output)
        return y
    
    def output(self):
        return self.output_from_input(self.x)

    def predict(self):
        return self.output()
        
