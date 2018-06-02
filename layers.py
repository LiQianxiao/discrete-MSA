"""Layer classes for PMP and MSA based methods.
Mostly wrappers over standard Tensorflow functions,
but incorporates MSA based training methods.
"""

import numpy as np
import tensorflow as tf
import utils


class AbstractLayer(object):
    """
    Abstract layer class
    """

    def __init__(self):
        self.vars = []
        self.is_trainable = False
        self.is_discrete = None

    def set_vars(self, x, layer_id):
        """Set layer_id and variables

        Arguments:
            x {tf tensor} -- Input to layer
            layer_id {int} -- Id of layer
        """
        self.layer_id = layer_id

    def regularizer(self):
        """Regularizer

        Returns:
            tf tensor (rank 0) or float -- Regularization loss
        """
        return 0.0

    def forward(self, x):
        """Forward propagation of MSA state eqn

        Arguments:
            x {tf tensor} -- x_{t}

        Raises:
            NotImplementedError

        Returns:
            tf tensor -- x_{t+1}
        """
        raise NotImplementedError
        return x

    def backward(self, x, p):
        """Backward propagation of MSA co-state eqn

        Arguments:
            x {tf tensor} -- state at t
            p {tf tensor} -- co-state at t+1

        Returns:
            tf tensor -- costate p at t
        """
        x = tf.stop_gradient(x)
        p = tf.stop_gradient(p)
        H = tf.reduce_sum(p * self.forward(x))
        p_next = tf.gradients(H, x)[0]
        return p_next

    def _H_and_grad(self, x, p):
        """Compuate Hamiltonian (without regularization)
        and its gradient wrt to parameters. The gradient
        actually computes M=(x^T p), but this is more
        efficient for conv nets than raw matrix
        multiplication and reshaping

        Arguments:
            x {tf tensor} -- x_{t}
            p {tf tensor} -- p_{t+1}

        Returns:
            tuple of tf tensors -- Hamiltonian and grad
        """
        dHdp = self.forward(x)
        H = tf.reduce_sum(p * dHdp)
        dHdw = tf.gradients(H, self.vars)
        return H, dHdw


class ReluLayer(AbstractLayer):
    """ReLU layer class
    """
    def __init__(self):
        super().__init__()
        self.name = 'ReLU'

    def forward(self, x):
        """Forward propagation

        Arguments:
            x {tf tensor} -- x_{t}

        Returns:
            x {tf tensor} -- x_{t+1}
        """
        return tf.nn.relu(x)


class TanhLayer(AbstractLayer):
    """Tanh layer class
    """
    def __init__(self):
        super().__init__()
        self.name = 'Tanh'

    def forward(self, x):
        """Forward propagation

        Arguments:
            x {tf tensor} -- x_{t}

        Returns:
            x {tf tensor} -- x_{t+1}
        """
        return tf.nn.tanh(x)


class SoftPlusLayer(AbstractLayer):
    """SoftPlus layer class
    """
    def __init__(self):
        super().__init__()
        self.name = 'SoftPlus'

    def forward(self, x):
        """Forward propagation

        Arguments:
            x {tf tensor} -- x_{t}

        Returns:
            x {tf tensor} -- x_{t+1}
        """
        return tf.nn.softplus(x)


class MaxPoolingLayer(AbstractLayer):
    """Max-pooling layer class
    """
    def __init__(self, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                 padding='SAME'):
        """Initializer

        Keyword Arguments:
            ksize {list (4)} -- kernel size (default: {[1,2,2,1]})
            strides {list (4)} -- strides (default: {[1,2,2,1]})
            padding {'SAME','VALID','FULL'} -- padding (default: {'SAME'})
        """
        super().__init__()
        self.ksize = ksize
        self.strides = strides
        self.padding = padding
        self.name = 'Max-pool'

    def forward(self, x):
        """Forward propagation

        Arguments:
            x {tf tensor} -- x_{t}

        Returns:
            x {tf tensor} -- x_{t+1}
        """
        return tf.nn.max_pool(x, self.ksize, self.strides, self.padding)


class DropOutLayer(AbstractLayer):
    """Drop-out layer class
    """
    def __init__(self, keep_prob=0.5, dtype=tf.float32):
        """Initalizer

        Keyword Arguments:
            keep_prob {float} -- Keep prob (default: {0.5})
            dtype {tf dtype} -- data type (default: {tf.float32})
        """

        super().__init__()
        self.keep_prob = tf.placeholder(tf.float32, shape=[])
        self.keep_prob_train = keep_prob
        self.name = 'Drop-out'

    def forward(self, x):
        """Forward propagation

        Arguments:
            x {tf tensor} -- x_{t}

        Returns:
            x {tf tensor} -- x_{t+1}
        """
        return tf.nn.dropout(x, self.keep_prob)


class BinaryFullyConnectedLayer(AbstractLayer):
    """Fully connected layer class
    """
    def __init__(self, out_dim, ema_decay_rate=0.5,
                 rho=1e-1, dtype=tf.float32):
        """Initalizer

        Arguments:
            out_dim {int} -- dimension of output x_{t+1}

        Keyword Arguments:
            ema_decay_rate {float} -- decay rate (alpha) (default: {0.5})
            rho {float} -- regularization parameter (default: {1e-1})
            dtype {tf dtype} -- data type (default: {tf.float32})
        """
        super().__init__()
        self.out_dim = out_dim
        self.dtype = dtype
        self.is_trainable = True
        self.is_discrete = True
        self.init_edr = ema_decay_rate
        self.init_rho = rho
        self.name = 'Binary-fully-connected'
        self.init_func = lambda shape: \
            np.random.choice([-1.0, 1.0], size=shape)

    def forward(self, x):
        """Forward propagation

        Arguments:
            x {tf tensor} -- x_{t}

        Returns:
            x {tf tensor} -- x_{t+1}
        """
        x_in_dim = np.prod(x.get_shape().as_list()[1:])
        w_in_dim = self.vars[0].get_shape().as_list()[0]
        assert x_in_dim == w_in_dim, 'Input/vars shape inconsistent.'
        x_next = tf.contrib.layers.flatten(x) @ tf.to_float(self.vars[0])
        return x_next

    def set_vars(self, x, layer_id):
        """Set layer_id and variables

        Arguments:
            x {tf tensor} -- Input to layer
            layer_id {int} -- Id of layer
        """
        self.layer_id = layer_id
        in_dim = np.prod(x.get_shape().as_list()[1:])
        with tf.variable_scope('layer'+str(self.layer_id)):
            wshape = [in_dim, self.out_dim]
            winit = self.init_func(wshape)
            weights = tf.get_variable(
                'weights', wshape, self.dtype,
                initializer=tf.constant_initializer(winit))
            ema_dH = tf.get_variable(
                'ema_dH', wshape, self.dtype,
                trainable=False,
                initializer=tf.random_uniform_initializer(
                    minval=-1e-6, maxval=1e-6))
            self.ema_decay_rate = tf.get_variable(
                'ema_decay_rate', [], self.dtype,
                trainable=False,
                initializer=tf.constant_initializer(self.init_edr))
            self.rho = tf.get_variable(
                'rho', [], self.dtype,
                trainable=False,
                initializer=tf.constant_initializer(self.init_rho))
        self.vars = [weights]
        self.dH_ema = [ema_dH]

    def _sign_func(self, x):
        """Modified sign function
        randomize if 0, else sign

        Arguments:
            x {tf tensor} -- input tensor

        Returns:
            tf tensor -- sign(x)
        """
        cond = x == tf.zeros_like(x)
        return tf.where(cond, tf.ones_like(x), tf.sign(x))

    def _ema_update_op(self):
        """Update op of ema average, x^T p
        """
        self.ema_update_op = [
            tf.assign_sub(ema, (1.0-self.ema_decay_rate)*(ema-g))
            for g, ema in zip(self.dHdw, self.dH_ema)]

    def _edr_setter_op(self):
        """Op to set decay rate
        """
        self.edr_ph = utils.get_ph(self.ema_decay_rate)
        self.set_edr_op = tf.assign(self.ema_decay_rate, self.edr_ph)

    def _rho_setter_op(self):
        """Op to set regularizer rho
        """
        self.rho_ph = utils.get_ph(self.rho)
        self.set_rho_op = tf.assign(self.rho, self.rho_ph)

    def _train_op(self):
        """Set all train ops
        """
        with tf.control_dependencies(self.ema_update_op):
            current_signs = self.vars[0]
            correct_signs = self._sign_func(self.dH_ema[0])
            wrong_signs = tf.not_equal(current_signs, correct_signs)
            magnitudes = tf.abs(self.dH_ema[0])
            max_magnitudes = tf.reduce_max(
                tf.boolean_mask(magnitudes, wrong_signs))
            condition = magnitudes > self.rho * max_magnitudes * \
                tf.ones_like(magnitudes)
            new_var = tf.where(condition, correct_signs, current_signs)
            self.train_op = [tf.assign(self.vars[0], new_var)]

    def set_ops(self, x, p):
        """Set all ops

        Arguments:
            x {tf tensor} -- x_{t}
            p {tf tensor} -- p_{t+1}
        """
        self.x_ph = utils.get_ph(x)
        self.p_ph = utils.get_ph(p)
        self.H, self.dHdw = self._H_and_grad(self.x_ph, self.p_ph)

        self._ema_update_op()
        self._edr_setter_op()
        self._rho_setter_op()
        self._train_op()

    def train(self, session, feeds):
        """Train operation for layer

        Arguments:
            session {tf session} -- Session
            feeds {python dict} -- Feed-dict, for x_{t} and p_{t+1}
        """

        assert len(feeds) == 2, \
            'Requires feeds=(x_n, p_{n+1})'
        x_value, p_value = feeds
        feed_dict = {self.x_ph: x_value, self.p_ph: p_value}
        session.run(self.train_op, feed_dict)

    def set_ema_decay_rate(self, session, new_rate):
        """Op to set decay

        Arguments:
            session {tf session} -- Session
            new_rate {np float} -- New decay rates
        """
        session.run(self.set_edr_op, {self.edr_ph: new_rate})

    def set_rho(self, session, new_rate):
        """Op to set rho

        Arguments:
            session {tf session} -- Session
            new_rate {np float} -- new rhos
        """
        session.run(self.set_rho_op, {self.rho_ph: new_rate})


class BinaryConvolutionLayer(BinaryFullyConnectedLayer):
    """Binary Convolutional layer class
    """
    def __init__(self, out_dim,
                 filter_size,
                 strides=[1, 1, 1, 1],
                 padding='SAME',
                 ema_decay_rate=0.5,
                 rho=1e-1,
                 dtype=tf.float32):
        """Initializer

        Arguments:
            out_dim {int} -- Output dimension
            filter_size {int} -- filter size dimension (only square for now)

        Keyword Arguments:
            strides {list (4)} -- Strides (default: {[1, 1, 1, 1]})
            padding {'SAME', 'VALID', 'FULL'} -- padding (default: {'SAME'})
            ema_decay_rate {float} -- decay rate (default: {0.5})
            rho {float} -- regularization parameter (default: {1e-1})
            dtype {tf dtype} -- data type (default: {tf.float32})
        """
        super(BinaryConvolutionLayer, self).__init__(
            out_dim, ema_decay_rate, rho, dtype)
        self.filter_size = filter_size
        self.strides = strides
        self.padding = padding
        self.name = 'Binary-Conv2D'

    def set_vars(self, x, layer_id):
        """Set layer_id and variables

        Arguments:
            x {tf tensor} -- Input to layer
            layer_id {int} -- Id of layer
        """
        self.layer_id = layer_id
        in_dim = x.get_shape().as_list()[-1]
        with tf.variable_scope('layer'+str(self.layer_id)):
            wshape = [self.filter_size, self.filter_size,
                      in_dim, self.out_dim]
            winit = self.init_func(wshape)
            weights = tf.get_variable(
                'weights', wshape, self.dtype,
                initializer=tf.constant_initializer(winit))
            ema_dH = tf.get_variable(
                'ema_dH', wshape, self.dtype,
                trainable=False,
                initializer=tf.random_uniform_initializer(
                    minval=-1e-6, maxval=1e-6))
            self.ema_decay_rate = tf.get_variable(
                'ema_decay_rate', [], self.dtype,
                trainable=False,
                initializer=tf.constant_initializer(self.init_edr))
            self.rho = tf.get_variable(
                'rho', [], self.dtype,
                trainable=False,
                initializer=tf.constant_initializer(self.init_rho))
        self.vars = [weights]
        self.dH_ema = [ema_dH]

    def forward(self, x):
        """Forward propagation

        Arguments:
            x {tf tensor} -- x_{t}

        Returns:
            x {tf tensor} -- x_{t+1}
        """
        x_in_dim = x.get_shape().as_list()[-1]
        w_in_dim = self.vars[0].get_shape().as_list()[2]
        assert x_in_dim == w_in_dim, 'Input/vars shape inconsistent.'
        weights = tf.to_float(self.vars[0])
        x_next = tf.nn.conv2d(
            x, weights, strides=self.strides, padding=self.padding)
        return x_next


class BatchNormLayer(AbstractLayer):
    """
    Batch-normalization layer class
    """
    def __init__(self, learning_rate=1e-3, ema_decay_rate=0.9,
                 epsilon=1e-4, axes=[0], dtype=tf.float32):
        """Initializer

        Keyword Arguments:
            learning_rate {float} -- learning rate (default: {1e-3})
            ema_decay_rate {float} -- ema decay rate (default: {0.9})
            epsilon {float} -- prevent /var overflow (default: {1e-3})
            axes {list} -- dimensions to normalize over (default: {[0]})
        """
        super().__init__()
        self.epsilon = epsilon
        self.axes = axes
        self.dtype = dtype
        self.is_trainable = True
        self.init_lr = learning_rate
        self.init_edr = ema_decay_rate
        self.mean, self.variance = None, None
        self.name = 'Batch-norm'

    def forward(self, x):
        """Forward propagation

        Arguments:
            x {tf tensor} -- x_{t}

        Returns:
            x {tf tensor} -- x_{t+1}
        """
        x_in_dim = np.asarray(x.get_shape().as_list())
        x_in_dim[self.axes] = 1
        scale_dim = np.asarray(self.vars[0].get_shape().as_list())
        offset_dim = np.asarray(self.vars[1].get_shape().as_list())
        assert (x_in_dim == scale_dim).all() and \
            (x_in_dim == offset_dim).all(), \
            'Input/vars shape inconsistent.'
        mean, variance = tf.nn.moments(
            x, axes=self.axes, keep_dims=True)
        if self.mean is None and self.variance is None:
            self.mean = mean
            self.variance = variance
            self.ema_vars = [self.mean, self.variance]
        x_normalized = (x - mean) / tf.sqrt(
            self.epsilon + variance)
        x_next = self.vars[0]*x_normalized + self.vars[1]
        return x_next

    def set_vars(self, x, layer_id):
        """Set layer_id and variables

        Arguments:
            x {tf tensor} -- Input to layer
            layer_id {int} -- Id of layer
        """
        self.layer_id = layer_id
        in_shape = np.asarray(x.get_shape().as_list())
        in_shape[self.axes] = 1
        with tf.variable_scope('layer'+str(self.layer_id)):
            scale = tf.get_variable(
                'scale', in_shape, self.dtype,
                initializer=tf.truncated_normal_initializer())
            offset = tf.get_variable(
                'offset', in_shape, self.dtype,
                initializer=tf.truncated_normal_initializer())
            self.learning_rate = tf.get_variable(
                'learning_rate', [], self.dtype,
                trainable=False,
                initializer=tf.constant_initializer(self.init_lr))
            self.alpha = tf.get_variable(
                'alpha', [], self.dtype,
                trainable=False,
                initializer=tf.constant_initializer(self.init_edr))
        self.vars = [scale, offset]

    def _ema_update_op(self):
        """Op to update ema
        """
        with tf.variable_scope('layer'+str(self.layer_id)):
            mean, variance = tf.nn.moments(
                self.x_ph, axes=self.axes, keep_dims=True)
            ema_tensors = [mean, variance]
            self.mean_ema = tf.get_variable(
                'mean_ema', mean.get_shape(), self.dtype,
                trainable=False,
                initializer=tf.zeros_initializer())
            self.variance_ema = tf.get_variable(
                'variance_ema', variance.get_shape(), self.dtype,
                trainable=False,
                initializer=tf.constant_initializer(1.0))
            self.ema_shadow_vars = [self.mean_ema, self.variance_ema]
        self.ema_update_op = [
            tf.assign_sub(sv, (1.0-self.alpha)*(sv - v))
            for v, sv in zip(ema_tensors, self.ema_shadow_vars)]

    def _lr_setter_op(self):
        """Op to set lr
        """
        self.lr_ph = utils.get_ph(self.learning_rate)
        self.set_lr_op = tf.assign(self.learning_rate, self.lr_ph)

    def _train_op(self):
        """Train ops
        """
        with tf.control_dependencies(self.ema_update_op):
            minus_H = -self.H
            optimizer = tf.train.AdamOptimizer(
                learning_rate=self.learning_rate)
            self.train_op = optimizer.minimize(
                minus_H, var_list=self.vars)

    def set_ops(self, x, p):
        """Set all ops

        Arguments:
            x {tf tensor} -- x_{t}
            p {tf tensor} -- p_{t+1}
        """
        self.x_ph = utils.get_ph(x)
        self.p_ph = utils.get_ph(p)
        self.H, self.dHdw = self._H_and_grad(self.x_ph, self.p_ph)

        self._ema_update_op()
        self._lr_setter_op()
        self._train_op()

    def train(self, session, feeds):
        """Train operation for layer

        Arguments:
            session {tf session} -- Session
            feeds {python dict} -- Feed-dict, for x_{t} and p_{t+1}
        """
        assert len(feeds) == 2, \
            'Requires feeds=(x_n, p_{n+1})'
        x_value, p_value = feeds
        feed_dict = {self.x_ph: x_value, self.p_ph: p_value}
        session.run(self.train_op, feed_dict)

    def set_learning_rate(self, session, new_rate):
        session.run(self.set_lr_op, {self.lr_ph: new_rate})


class TernaryFullyConnectedLayer(BinaryFullyConnectedLayer):
    """Ternary Fully connected layer class
    with {+1, 0, -1} as possible values
    """
    def __init__(self, out_dim,
                 reg_factor=1e-7,
                 ema_decay_rate=0.5,
                 rho=1e-1,
                 dtype=tf.float32):
        """Initializer

        Arguments:
            out_dim {int} -- output dimension

        Keyword Arguments:
            reg_factor {float} -- regularization coef for sparsity
                                  (default: {1e-7})
            ema_decay_rate {float} -- decay rate (default: {0.5})
            rho {float} -- rho (default: {1e-1})
            dtype {tf dtype} -- data type (default: {tf.float32})
        """
        super(TernaryFullyConnectedLayer, self).__init__(
            out_dim, ema_decay_rate, rho, dtype)
        self.reg_factor = reg_factor
        self.init_func = lambda shape: \
            np.random.choice([-1.0, 0.0, 1.0], size=shape)
        self.name = 'Ternary-fully-connected'

    def regularizer(self):
        """Regularizer

        Returns:
            tf tensor (rank 0) or float -- Regularization loss
        """
        reg_loss = self.reg_factor*tf.add_n(
            [tf.reduce_sum(tf.abs(v)) for v in self.vars])
        return reg_loss

    def _train_op(self):
        """Train ops
        """
        with tf.control_dependencies(self.ema_update_op):
            # Calculate $rho_{k,t}$
            current_signs = self.vars[0]
            correct_signs = tf.sign(self.dH_ema[0])
            wrong_signs = tf.not_equal(current_signs, correct_signs)
            magnitudes = tf.abs(self.dH_ema[0])
            max_magnitudes = tf.reduce_max(
                tf.boolean_mask(magnitudes, wrong_signs))
            rho_k = self.rho * max_magnitudes

            # Update parameters
            condition_plus_1 = self.dH_ema[0] > \
                rho_k*(1.0-2.0*self.vars[0]) + self.reg_factor
            condition_minus_1 = self.dH_ema[0] < \
                -rho_k*(1.0+2.0*self.vars[0]) - self.reg_factor
            new_var = tf.zeros_like(self.vars[0])
            all_ones = tf.ones_like(self.vars[0])
            new_var = tf.where(
                condition_plus_1, all_ones, new_var)
            new_var = tf.where(
                condition_minus_1, -all_ones, new_var)
            self.train_op = [tf.assign(self.vars[0], new_var)]


class TernaryConvolutionLayer(BinaryConvolutionLayer):
    """Ternary Convolutional layer class
    """
    def __init__(self, out_dim,
                 filter_size,
                 strides=[1, 1, 1, 1],
                 padding='SAME',
                 reg_factor=1e-7,
                 ema_decay_rate=0.5,
                 rho=1e-1,
                 dtype=tf.float32):
        """Initializer

        Arguments:
            out_dim {int} -- output dimension
            filter_size {int} -- filter size (square only)

        Keyword Arguments:
            strides {list (4)} -- strides (default: {[1,1,1,1]})
            padding {'SAME','VALID','FULL'} -- padding (default: {'SAME'})
            reg_factor {float} -- regularization coef for sparsity
                                  (default: {1e-7})
            ema_decay_rate {float} -- decay rate (default: {0.5})
            rho {float} -- rho (default: {1e-1})
            dtype {tf dtype} -- data type (default: {tf.float32})
        """
        super(TernaryConvolutionLayer, self).__init__(
            out_dim, filter_size, strides, padding,
            ema_decay_rate, rho, dtype)
        self.reg_factor = reg_factor
        self.init_func = lambda shape: \
            np.random.choice([-1.0, 0.0, 1.0], size=shape)
        self.name = 'Ternary-Conv2D'

    def regularizer(self):
        """Regularizer

        Returns:
            tf tensor (rank 0) or float -- Regularization loss
        """
        reg_loss = self.reg_factor*tf.add_n(
            [tf.reduce_sum(tf.abs(v)) for v in self.vars])
        return reg_loss

    def _train_op(self):
        """Train op
        """
        with tf.control_dependencies(self.ema_update_op):
            # Calculate $rho_{k,t}$
            current_signs = self.vars[0]
            correct_signs = self._sign_func(self.dH_ema[0])
            wrong_signs = tf.not_equal(current_signs, correct_signs)
            magnitudes = tf.abs(self.dH_ema[0])
            max_magnitudes = tf.reduce_max(
                tf.boolean_mask(magnitudes, wrong_signs))
            rho_k = self.rho * max_magnitudes

            # Update parameters
            condition_plus_1 = self.dH_ema[0] > \
                rho_k*(1.0-2.0*self.vars[0]) + self.reg_factor
            condition_minus_1 = self.dH_ema[0] < \
                -rho_k*(1.0+2.0*self.vars[0]) - self.reg_factor
            new_var = tf.zeros_like(self.vars[0])
            all_ones = tf.ones_like(self.vars[0])
            new_var = tf.where(
                condition_plus_1, all_ones, new_var)
            new_var = tf.where(
                condition_minus_1, -all_ones, new_var)
            self.train_op = [tf.assign(self.vars[0], new_var)]
