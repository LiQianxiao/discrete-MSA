"""Discrete-weight Neural Networks
"""

import numpy as np
import tensorflow as tf
import utils


class NeuralNetwork(object):
    """Discrete-weight neural network class
    that wraps layers and pass into a train.Trainer
    """

    def __init__(self, in_size, n_out_classes,
                 loss_func=tf.losses.softmax_cross_entropy,
                 dtype=tf.float32):
        """Initializer

        Arguments:
            in_size {list} -- input dimensions
            n_out_classes {int} -- number of output classes
            loss_func {tf losses funcs} -- loss function
                (default: {tf.losses.softmax_cross_entropy})
            dtype {tf dtype} -- data type (default: {tf.float32})
        """
        self.in_size = in_size
        self.n_out_classes = n_out_classes
        self.loss_func = loss_func
        self.dtype = dtype
        self.reset_graph()

    def reset_graph(self):
        """Reset tensorflow graph
        """
        tf.reset_default_graph()
        self.x = tf.placeholder(
            self.dtype, self.in_size, 'input_placeholder')
        self.out = self.x
        self.layers = []
        self.xs = [self.out]
        self.regularization_loss = 0
        self.n_layers = 0
        self.finalized = False

    def add_layer(self, layer_object):
        """Add a layer to network

        Arguments:
            layer_object {layers.AbstractLayer} -- Layer object
        """
        assert not self.finalized, 'Graph already finalized.'
        layer_object.set_vars(self.out, self.n_layers)
        self.out = layer_object.forward(self.out)
        self.xs.append(self.out)
        self.regularization_loss += layer_object.regularizer()
        self.layers.append(layer_object)
        self.n_layers += 1

    def finalize(self):
        """Finalize graph for training
        """
        out_dim = self.xs[-1].get_shape().as_list()[-1]
        assert out_dim == self.n_out_classes, \
            'Final layer output dimension should be equal to n_out_classes.'
        assert not self.finalized, 'Graph already finalized.'

        # Loss function
        self.y = tf.placeholder(
            self.dtype, [None, self.n_out_classes], 'output_placeholder')
        self.loss = self.loss_func(self.y, self.out)
        correct_prediction = tf.equal(
            tf.argmax(self.out, 1), tf.argmax(self.y, 1))
        self.accuracy = tf.reduce_mean(
            tf.cast(correct_prediction, self.dtype))

        # Compute co-state
        pout = -tf.gradients(self.loss, self.out)[0]
        self.ps = [pout]
        for l in reversed(range(self.n_layers)):
            pout = self.layers[l].backward(self.xs[l], pout)
            self.ps.append(pout)
        self.ps.reverse()

        # Set training ops
        for l in range(self.n_layers):
            if self.layers[l].is_trainable:
                self.layers[l].set_ops(self.xs[l], self.ps[l+1])

        self.saver = tf.train.Saver(tf.global_variables())
        self.init = tf.global_variables_initializer()
        tf.Graph.finalize(tf.get_default_graph())
        self.finalized = True
