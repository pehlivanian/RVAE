import numpy as np
import theano
import theano.tensor as T
import abc
import os
import gzip
import pickle
import numpy

def elu(x, alpha=1.0):
    return T.switch(x > 0, x, T.exp(x) - 1)

###########
# SOLVERS #
###########

class decorator(object):
    def __init__(self, decorated):
        self.decorated = decorated

class optimizer_decorator(decorator):
    def __init__(self, decorated, minimize=True, **k):

        if isinstance(decorated, decorator):
            raise RunTimeError('you must decorate a model object')

        super(optimizer_decorator, self).__init__(decorated)

        self.minimize = minimize
        self.mult = 1 if self.minimize else -1
        
        # required
        assert self.decorated.x
        assert self.decorated.params
        assert self.decorated.objective

    @abc.abstractmethod
    def compute_obj_updates(self, *a, **k):
        pass

    def eval_objective(self, *a, **k):
        z = self.decorated.objective(*a, **k)
        return z

class rmsprop_optimizer( optimizer_decorator ):
    ''' Rmsprop optimizer decorator.

        :type beta: theano.tensor.scalar
        :param beta: initial learning rate
        
        :type eta: theano.tensor.scalar
        :param eta: decay rate
        
        :type parameters: theano variable
        :param parameters: model parameters to update
        
        :type grads: theano variable
        :param grads: gradients of const wrt parameters
    '''
    
    def __init__(self,
                 decorated,
                 minimize=True,
                 eta=1.e-2,
                 beta=.75,
                 epsilon=1.e-6,
                 cost_fn='MSE',
                 *args,
                 **kwargs
                 ):

        super(rmsprop_optimizer, self).__init__(decorated, minimize=minimize)
        
        self.eta = eta
        self.beta = beta
        self.epsilon = epsilon
        self.cost_fn = cost_fn
    
    def compute_obj_updates(self, *a, **k):

        '''

        RMSProp update
        
        Parameters
        ----------

        Returns
        -------
        tuple
        returns cost and a list of updates for the required parameters
        
        
        '''

        z = self.eval_objective(*a, **k)
        
        grads = T.grad(z, self.decorated.params)
        
        one = T.constant(1.0)

        def _updates(param, cache, df, beta=0.9, eta=1.e-2, epsilon=1.e-6):
            cache_val = beta * cache + (one-beta) * df**2
            x = T.switch( T.abs_(cache_val) <
                          epsilon, cache_val, eta * df /
                          (T.sqrt(cache_val) + epsilon))
            updates = (param, param-self.mult*x), (cache, cache_val)

            return updates

        caches = [theano.shared(name='c_%s' % param,
                                value=param.get_value() * 0.,
                                broadcastable=param.broadcastable)
                  for param in self.decorated.params]

        updates = []

        for param, cache, grad in zip(self.decorated.params, caches, grads):
            param_updates, cache_updates = _updates(param,
                                                   cache,
                                                   grad,
                                                   beta=self.beta,
                                                   eta=self.eta,
                                                   epsilon=self.epsilon)
            updates.append(param_updates)
            updates.append(cache_updates)

        return z, updates

class adam_optimizer( optimizer_decorator ):
    ''' Adam solver decorator.

    '''
    def __init__(self,
                 decorated,
                 minimize=True,
                 learning_rate=0.001,
                 beta1=0.9,
                 beta2=0.999,
                 epsilon=1e-8,
                 cost_fn='CROSS-ENTROPY',
                 *args,
                 **kwargs
                 ):

        super( adam_optimizer, self).__init__(decorated, minimize=minimize)
        
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.cost_fn = cost_fn
    
    def compute_obj_updates(self, *a, **k):
        
        ''' Adam updates
        '''
        updates = list()
        
        epoch_pre = theano.shared(np.asarray(.0, dtype=theano.config.floatX))
        epoch_num = epoch_pre + 1
        gamma = self.learning_rate * T.sqrt(1 - self.beta2 ** epoch_num) / (1 - self.beta1 ** epoch_num)

        z = self.eval_objective(*a, **k)
        grads = T.grad(z, self.decorated.params)

        for param, grad in zip(self.decorated.params, grads):
            v = param.get_value(borrow=True)
            m_pre = theano.shared(np.zeros(v.shape, dtype=v.dtype), broadcastable=param.broadcastable)
            v_pre = theano.shared(np.zeros(v.shape, dtype=v.dtype), broadcastable=param.broadcastable)

            m_t = self.beta1 * m_pre + (1 - self.beta1) * grad
            v_t = self.beta2 * v_pre + (1 - self.beta2) * grad ** 2
            step = gamma * m_t / (T.sqrt(v_t) + self.epsilon)

            updates.append((m_pre, m_t))
            updates.append((v_pre, v_t))
            updates.append((param, param - self.mult * step))

        updates.append((epoch_pre, epoch_num))
        return z, updates

class gd_optimizer(optimizer_decorator):
    ''' Gradient descent solver decorator.

        :type learning_Rate: float
        :param parameters: learning rate for gradient descent
        
        :cost_fn: str
        :param cost_fn: one of 'mse', 'cross-entropy'
    '''

    def __init__(self,
                 decorated,
                 minimize=True,
                 learning_rate=0.01,
                 cost_fn='CROSS-ENTROPY',
                 *args,
                 **kwargs
                 ):

        super( gd_optimizer, self).__init__(decorated, minimize=minimize)
        
        self.learning_rate = learning_rate
        self.cost_fn = cost_fn

    def compute_obj_updates(self, *a, **k):

        '''

        Gradient descent update
        
        Parameters
        ----------

        Returns
        -------
        tuple
        returns objective value, cost, and a list of updates for the
        required parameters for the minimization problem
        
        
        '''

        z = self.eval_objective(*a, **k)

        grads = T.grad(z, self.decorated.params)
        updates = [ (param, param - self.mult * self.learning_rate * grad)
                    for param, grad in zip(self.decorated.params, grads) ]

        return z, updates


class solver_decorator(decorator):
    def __init__(self, decorated, *a, **k):

        if isinstance(decorated, decorator):
            raise RuntimeError('you must decorate a model object')

        super(solver_decorator, self).__init__(decorated)
        
        # required
        assert self.decorated.x
        assert self.decorated.y
        assert self.decorated.params
        assert self.decorated.predict

    @abc.abstractmethod
    def compute_cost_updates(self, *a, **k):
        pass

    def compute_cost(self, *a, **k):
        cost, _ = self.compute_cost_updates(*a, **k)
        return cost

    def compute_updates(self, *a, **k):
        _, updates = self.compute_cost_updates(*a, **k)
        return updates
        

    def predict(self, *a, **k):
        z = self.decorated.predict(*a,**k)
        return z

    def predict_from_input(self, *a, **k):
        if hasattr(self.decorated, 'predict_from_input'):
            z = self.decorated.predict_from_input(*a, **k)
            return z
        else:
            raise RuntimeError('model has no predict_from_input method')        

class predict_feedback_decorator(decorator):
    def __init__(self, decorated, *a, **k):
        super(predict_feedback_decorator, self).__init__(decorated)

        self.decorated.predict = self.predict
        self.compute_cost_updates = self.decorated.compute_cost_updates

        # Ensure we dispatch calls to solver object
        while isinstance(decorated, decorator):
            decorated = decorated.decorated

        self.decorated = decorated
    
class gd_solver( solver_decorator ):
    ''' Gradient descent solver decorator.

        :type learning_Rate: float
        :param parameters: learning rate for gradient descent
        
        :cost_fn: str
        :param cost_fn: one of 'mse', 'cross-entropy'
    '''

    def __init__(self,
                 decorated,
                 learning_rate=0.01,
                 cost_fn='MSE',
                 *args,
                 **kwargs
                 ):

        super( gd_solver, self).__init__(decorated)
        
        self.learning_rate = learning_rate
        self.cost_fn = cost_fn

    def compute_cost_updates(self, *a, **k):

        '''

        Gradient descent update
        
        Parameters
        ----------

        Returns
        -------
        tuple
        returns cost and a list of updates for the required parameters
        
        
        '''

        z = self.predict(*a, **k)
        if self.cost_fn.upper() == 'MSE':
            L = T.sqrt( T.sum( (self.decorated.y - z) ** 2, axis=1))
        elif self.cost_fn.upper() in [ 'CROSS-ENTROPY', 'CROSS_ENTROPY' ]:
            L = T.sum( self.decorated.y * T.log(z) + (1-self.decorated.y) * T.log(1-z), axis=1)
        cost = T.mean(L)

        grads = T.grad(cost, self.decorated.params)

        updates = [ (param, param - self.learning_rate * grad)
                    for param, grad in zip(self.decorated.params, grads) ]

        return (cost, updates)

class rmsprop_solver( solver_decorator ):
    ''' Rmsprop solver decorator.

        :type beta: theano.tensor.scalar
        :param beta: initial learning rate
        
        :type eta: theano.tensor.scalar
        :param eta: decay rate
        
        :type parameters: theano variable
        :param parameters: model parameters to update
        
        :type grads: theano variable
        :param grads: gradients of const wrt parameters
    '''
    
    def __init__(self,
                 decorated,
                 eta=1.e-2,
                 beta=.75,
                 epsilon=1.e-6,
                 cost_fn='MSE',
                 replace_input=False,
                 *args,
                 **kwargs
                 ):

        super(rmsprop_solver, self).__init__(decorated)
        
        self.eta = eta
        self.beta = beta
        self.epsilon = epsilon
        self.cost_fn = cost_fn
        self.replace_input = replace_input
    
    def compute_cost_updates(self, *a, **k):

        '''

        RMSProp update
        
        Parameters
        ----------

        Returns
        -------
        tuple
        returns cost and a list of updates for the required parameters
        
        
        '''
        
        z = self.predict(*a, **k)
        if self.cost_fn.upper() == 'MSE':
            L = T.sqrt( T.sum( (self.decorated.y - z) ** 2, axis=1 ))
        elif self.cost_fn.upper() in [ 'CROSS-ENTROPY', 'CROSS_ENTROPY' ]:
            L = -T.sum(self.decorated.y * T.log(z) + (1 - self.decorated.y) * T.log(1-z), axis=1)
        else:
            raise RunTimeError('Cost function not defined')

        cost = T.mean(L)

        grads = T.grad(cost, self.decorated.params)
        
        one = T.constant(1.0)

        def _updates(param, cache, df, beta=0.9, eta=1.e-2, epsilon=1.e-6):
            cache_val = beta * cache + (one-beta) * df**2
            x = T.switch( T.abs_(cache_val) <
                          epsilon, cache_val, eta * df /
                          (T.sqrt(cache_val) + epsilon))
            updates = (param, param-x), (cache, cache_val)

            return updates

        caches = [theano.shared(name='c_%s' % param,
                                value=param.get_value() * 0.,
                                broadcastable=param.broadcastable)
                  for param in self.decorated.params]

        updates = []

        for param, cache, grad in zip(self.decorated.params, caches, grads):
            param_updates, cache_updates = _updates(param,
                                                   cache,
                                                   grad,
                                                   beta=self.beta,
                                                   eta=self.eta,
                                                   epsilon=self.epsilon)
            updates.append(param_updates)
            updates.append(cache_updates)

        return (cost, updates)

class negative_feedback( predict_feedback_decorator ):
    ''' Negative feedback solver decorator.
    '''

    def __init__(self,
                 decorated,
                 *args,
                 **kwargs
                 ):

        super( negative_feedback, self).__init__(decorated)
        
    def predict(self, *a, **k):
        # Method I - works well for nearly linear activation functionsa
        # z = self.decorated.predict(*a, **k)
        # for ind in range(22):
        #     z = self.decorated.predict_from_input( z - self.decorated.y )
        # return z

        # Method II
        # (1 - z + z**2 - z**3 + ...)(z) = z - z**2 + z**3 - z**$ + ...
        z = self.decorated.predict(*a, **k)
        z_sum = 0
        for ind in range(0,3):
            z_sum = z_sum + ((-1)**ind) * z            
            z = self.decorated.predict_from_input( z)
        return z_sum
        
def load_data(dataset):
    ''' Loads the dataset

    :type dataset: string
    :param dataset: the path to the dataset (here MNIST)
    '''

    #############
    # LOAD DATA #
    #############

    # Download the MNIST dataset if it is not present
    data_dir, data_file = os.path.split(dataset)
    if data_dir == "" and not os.path.isfile(dataset):
        # Check if dataset is in the data directory.
        new_path = os.path.join(
            os.path.split(__file__)[0],
            "..",
            "data",
            dataset
        )
        if os.path.isfile(new_path) or data_file == 'mnist.pkl.gz':
            dataset = new_path

    if (not os.path.isfile(dataset)) and data_file == 'mnist.pkl.gz':
        from six.moves import urllib
        origin = (
            'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
        )
        print('Downloading data from %s' % origin)
        urllib.request.urlretrieve(origin, dataset)

    print('... loading data')

    # Load the dataset
    with gzip.open(dataset, 'rb') as f:
        try:
            train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
        except:
            train_set, valid_set, test_set = pickle.load(f)
    # train_set, valid_set, test_set format: tuple(input, target)
    # input is a numpy.ndarray of 2 dimensions (a matrix)
    # where each row corresponds to an example. target is a
    # numpy.ndarray of 1 dimension (vector) that has the same length as
    # the number of rows in the input. It should give the target
    # to the example with the same index in the input.

    print('...finished loading data')

    def shared_dataset(data_xy, borrow=True):
        """ Function that loads the dataset into shared variables

        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
        """
        data_x, data_y = data_xy
        shared_x = theano.shared(numpy.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(numpy.asarray(data_y,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        # When storing data on the GPU it has to be stored as floats
        # therefore we will store the labels as ``floatX`` as well
        # (``shared_y`` does exactly that). But during our computations
        # we need them as ints (we use labels as index, and if they are
        # floats it doesn't make sense) therefore instead of returning
        # ``shared_y`` we will have to cast it to int. This little hack
        # lets ous get around this issue
        return shared_x, T.cast(shared_y, 'int32')

    test_set_x, test_set_y = shared_dataset(test_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    train_set_x, train_set_y = shared_dataset(train_set)

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]
    return rval

def create_weight(dim_input, dim_output, sigma_init=0.01, use_xavier=False):
    rng = np.random.RandomState(47)
    if use_xavier:
        return rng.uniform(low=-1/np.sqrt(dim_input),
                           high=1/np.sqrt(dim_input),
                           size=(dim_input,
                                 dim_output)).astype(theano.config.floatX)
    return rng.normal(0,
                      sigma_init,
                      (dim_input,
                       dim_output)).astype(theano.config.floatX)
def create_bias(dim_output, sigma_init=0.01, use_xavier=False, dim_input=1):
    rng = np.random.RandomState(48)
    if use_xavier:
        return rng.uniform(low=-1/np.sqrt(dim_input),
                           high=1/np.sqrt(dim_input),
                           size=dim_output).astype(theano.config.floatX)
    return rng.normal(0,
                      sigma_init,
                      dim_output).astype(theano.config.floatX)
