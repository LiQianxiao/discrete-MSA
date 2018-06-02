"""Trainer
"""

import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle
import random
import os


class Trainer(object):
    """
    MSA Optimizer
    """

    def __init__(self, network, data_train, sess=None):
        """Initializer

        Arguments:
            network {network.NeuralNetwork} -- NeuralNetwork object
            data_train {tuple of np float} -- (train_images, train_labels)

        Keyword Arguments:
            sess {tf session} -- Session (default: {None})
        """
        self.network = network
        assert self.network.finalized, 'Finalize network first.'
        self.x_train, self.y_train = data_train
        self.sess = sess or tf.Session()
        self.saver = self.network.saver
        self.reset()

    def reset(self):
        """Re-init everything
        """
        self.ptr = 0
        self.sess.run(self.network.init)

    def getbatch(self, batch_size):
        """Generate a batch of training data

        Arguments:
            batch_size {int} -- batch size

        Returns:
            tuple of np float -- (batch_images, batch_labels)
        """
        train_size = self.x_train.shape[0]
        start = self.ptr
        end = self.ptr+batch_size \
            if self.ptr+batch_size < train_size else train_size
        x_batch, y_batch = \
            self.x_train[start:end], \
            self.y_train[start:end]
        self.ptr = end if end < train_size else 0
        return x_batch, y_batch

    def get_learning_rates(self):
        """Get learning rates

        Returns:
            list -- learning rates (None means no lr)
        """
        learning_rates = [
            self.sess.run(obj.learning_rate) if hasattr(obj, 'learning_rate')
            else None for obj in self.network.layers]
        return learning_rates

    def set_learning_rates(self, value):
        """Set learning rates to value

        Arguments:
            value {float or list} -- New learning rates
                if scalar: set all to new value
                if list: set corresponding layers to new values
        """
        if not isinstance(value, list):
            value = [value]*self.network.n_layers
        for l in range(self.network.n_layers):
            if hasattr(self.network.layers[l], 'learning_rate'):
                self.network.layers[l].set_learning_rate(
                    session=self.sess, new_rate=value[l])

    def decay_learning_rates(self, decay_rate):
        """Decay learning rates

        Arguments:
            decay_rate {float} -- factor to decay
        """
        if not isinstance(decay_rate, list):
            decay_rate = [decay_rate]*self.network.n_layers
        assert len(decay_rate) == self.network.n_layers
        lrs = self.get_learning_rates()
        lrs = [lr*dr if lr else None
               for lr, dr in zip(lrs, decay_rate)]
        self.set_learning_rates(lrs)

    def get_ema_rates(self):
        """Get EMA rates

        Returns:
            list -- EMA rates (None means no ema rate)
        """
        ema_rates = [
            self.sess.run(obj.ema_decay_rate)
            if hasattr(obj, 'ema_decay_rate')
            else None for obj in self.network.layers]
        return ema_rates

    def set_ema_rates(self, values):
        """Set EMA rates to value

        Arguments:
            value {float or list} -- New EMA rates
                if scalar: set all to new value
                if list: set corresponding layers to new values
        """
        if not isinstance(values, list):
            values = [values]*self.network.n_layers
        for l in range(self.network.n_layers):
            if hasattr(self.network.layers[l], 'ema_decay_rate'):
                self.network.layers[l].set_ema_decay_rate(
                    session=self.sess, new_rate=values[l])

    def decay_ema_rates(self, decay_rate):
        """Decay learning rates
        new_rate = 1 - (1-old_rate)*decay_rate

        Arguments:
            decay_rate {float} -- factor to decay
        """
        if not isinstance(decay_rate, list):
            decay_rate = [decay_rate]*self.network.n_layers
        assert len(decay_rate) == self.network.n_layers
        edrs = self.get_ema_rates()
        edrs = [max(min(1.0 - (1.0-edr)*dr, 1.0), 0.0)
                if edr else None
                for edr, dr in zip(edrs, decay_rate)]
        self.set_ema_rates(edrs)

    def get_rho(self):
        """Get rho values

        Returns:
            list -- rho values (None means no value)
        """
        rho = [
            self.sess.run(obj.rho)
            if hasattr(obj, 'rho')
            else None for obj in self.network.layers]
        return rho

    def set_rho(self, values):
        """Set rho to value

        Arguments:
            value {float or list} -- New rho values
                if scalar: set all to new value
                if list: set corresponding layers to new values
        """
        if not isinstance(values, list):
            values = [values]*self.network.n_layers
        for l in range(self.network.n_layers):
            if hasattr(self.network.layers[l], 'rho'):
                self.network.layers[l].set_rho(
                    session=self.sess, new_rate=values[l])

    def train_step(self, data_batch, n_train):
        """Train step

        Arguments:
            data_batch {tuple of np float} -- (x_batch, y_batch)
            n_train {None or list of float} -- Layers to train
                None: train all layers
                Scalar: train randomly selected sublist of this length
                List: train layers whose id are found in the list
        """
        x_batch, y_batch = data_batch
        feed_batch = {self.network.x: x_batch,
                      self.network.y: y_batch}
        # Set drop-out layer feed_prob
        feed_dropout = dict(
            [(obj.keep_prob, obj.keep_prob_train)
             for obj in self.network.layers
             if hasattr(obj, 'keep_prob')])
        feed_batch.update(feed_dropout)

        # Get x,p values
        xs_val, ps_val = self.sess.run(
            [self.network.xs, self.network.ps], feed_batch)

        # Trainable layers
        trainable_layers = [obj for obj in self.network.layers
                            if obj.is_trainable]
        if n_train is not None:
            if isinstance(n_train, list):
                # Train selected layers
                trainable_layers = [obj for obj in trainable_layers
                                    if obj in n_train]
            else:
                # Train randomly selected layers
                n_train = np.minimum(n_train, len(trainable_layers))
                trainable_layers = random.sample(trainable_layers, n_train)
        # Otherwise train all layers
        for obj in trainable_layers:
            l = obj.layer_id
            obj.train(session=self.sess,
                      feeds=(xs_val[l], ps_val[l+1]))

    def train_epoch(self, batch_size, n_output=10, lr_decay=None,
                    ema_decay=None, n_train=None, shuffle_data=True,
                    verbose=True):
        """Train one epoch

        Arguments:
            batch_size {int} -- batch size

        Keyword Arguments:
            n_output {int} -- number of outputs (default: {10})
            lr_decay {float} -- learning rate decay (default: {None})
            ema_decay {float} -- ema rate decay (default: {None})
            n_train {None or int or list} -- layers to train (default: {None})
                None: train all layers
                Scalar: train randomly selected sublist of this length
                List: train layers whose id are found in the list
            shuffle_data {boolean} -- whether to shuffle data at the start
                (default: {True})
            verbose {boolean} -- print n_output data (default: {True})

        Returns:
            tuple of list of floats -- computed batch losses
                and accuracies in epoch
        """
        self.ptr = 0
        if shuffle_data:
            self.x_train, self.y_train = \
                shuffle(self.x_train, self.y_train)
        n_batches = self.x_train.shape[0]//batch_size
        chunk_size = max(1, n_batches//n_output)
        if verbose:
            print('========= Begin epoch =========')
            print('batch_size = %d' % batch_size)
            print('EMA rates:')
            print([a for a in self.get_ema_rates() if a])
            print('rho:')
            print([a for a in self.get_rho() if a])
        epoch_losses, epoch_accs = [], []

        for t in range(n_batches):
            data_batch = self.getbatch(batch_size)
            self.train_step(data_batch, n_train=n_train)
            current_loss, current_acc = self.loss_and_accuracy(
                data=data_batch)
            epoch_losses.append(current_loss)
            epoch_accs.append(current_acc)
            if t % chunk_size == 0 and verbose:
                print('Iter: %d of %d || Estimated train loss/acc: %f, %.2f' %
                      (t, n_batches,
                       current_loss, current_acc))

        if lr_decay:
            self.decay_learning_rates(lr_decay)
        if ema_decay:
            self.decay_ema_rates(ema_decay)

        return epoch_losses, epoch_accs

    def loss_and_accuracy(self, data, inference=False, max_batch=10000):
        """Compute losses and accuracies

        Arguments:
            data {tuple of np float} -- (images, labels)

        Keyword Arguments:
            inference {boolean} -- use EMA average values in batch norm layers
                (default: {False})
            max_batch {int} -- Maximum batch size to feed (for GPU memory)
                (default: {10000})

        Returns:
            tuple of float -- loss, accuracy
        """
        num_samples = data[0].shape[0]
        loss_total, acc_total = 0.0, 0.0
        ptr = 0

        while ptr < num_samples:
            end_ptr = np.minimum(num_samples, ptr + max_batch)
            x_data, y_data = data[0][ptr:end_ptr], data[1][ptr:end_ptr]
            feed = {self.network.x: x_data,
                    self.network.y: y_data}
            # Set drop-out layer feed_prob = 1
            feed_dropout = dict(
                [(obj.keep_prob, 1.0) for obj in self.network.layers
                 if hasattr(obj, 'keep_prob')])
            feed.update(feed_dropout)
            if inference:
                # Use ema shadow vars for Batch-norm layers
                bn_layers = [obj for obj in self.network.layers
                             if 'Batch-norm' == obj.name]
                keys = [v for obj in bn_layers for v in obj.ema_vars]
                values = self.sess.run(
                    [v for obj in bn_layers for v in obj.ema_shadow_vars])
                feed_inference = dict(zip(keys, values))
                feed.update(feed_inference)
            lossval, accval = self.sess.run(
                [self.network.loss, self.network.accuracy], feed)
            loss_total += lossval*(end_ptr - ptr)
            acc_total += accval*(end_ptr - ptr)
            ptr = end_ptr

        loss_total /= num_samples
        acc_total /= num_samples
        return loss_total, acc_total

    def save(self, directory, filename):
        """Save trained model

        Arguments:
            directory {string} -- directory
            filename {string} -- saved file name
        """
        if not os.path.exists(directory):
            os.makedirs(directory)
        self.saver.save(self.sess, directory+filename)

    def restore(self, directory, filename):
        """Restore trained model

        Arguments:
            directory {string} -- directory
            filename {string} -- saved file name
        """
        self.saver.restore(self.sess, directory+filename)
