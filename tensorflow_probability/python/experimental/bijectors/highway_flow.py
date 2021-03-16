import numpy as np
import tensorflow as tf

import tensorflow_probability as tfp

tfd = tfp.distributions
tfb = tfp.bijectors
tfe = tf.experimental

'''
This initial implementation is based on the PyTorch implementation at
https://github.com/LucaAmbrogioni/CascadingFlow/blob/main/modules/networks.py
'''


class TriResNet(tfb.Bijector):

    # set activation=False for last block
    def __init__(self, width, activation=True, validate_args=False, name='tri_res_net'):
        super(TriResNet, self).__init__(
            validate_args=validate_args,
            forward_min_event_ndims=0,
            name=name)

        self.width = width
        self.W = tf.Variable(np.random.normal(0, 0.01, (self.width, self.width)), trainable=True, dtype=tf.float32)
        self.b = tf.Variable(np.random.normal(0, 0.01, (self.width,)), trainable=True, dtype=tf.float32)  # bias

        # positive diagonal for U
        self.d = tf.Variable(np.random.normal(0, 0.01, (self.width,)), trainable=True, dtype=tf.float32)

        # lambda before sigmoid
        self.pre_l = tf.Variable(np.random.normal(0, 0.1, (1,)), trainable=True, dtype=tf.float32)

        self.activation = activation

        self.masku = tfe.numpy.tril(tf.ones((self.width, self.width)), -1)
        self.maskl = tfe.numpy.triu(tf.ones((self.width, self.width)), 1)

    def get_l(self):
        return tf.nn.sigmoid(self.pre_l)

    def get_L(self):
        return self.W * self.maskl + tf.eye(self.width)

    def get_U(self):
        return self.W * self.masku + tf.linalg.diag(tf.math.softplus(self.d))

    def df(self, x):
        # derivative of activation
        l = self.get_l()
        return l + (1 - l) * tf.math.sigmoid(x) * (1 - tf.math.sigmoid(x))

    def cu(self, M):
        # convex update
        # same as doing as in the paper, but I guess easier to invert
        I = tf.eye(self.width)
        l = self.get_l()
        return l * I + (1 - l) * M

    def inv_f(self, y, N=20):
        # inverse with Newton iteration
        x = tf.ones(y.shape)
        for _ in range(N):
            x = x - (self.get_l() * x + (1 - self.get_l()) * tf.math.softplus(x) - y) / (self.df(x))
        return x

    def _forward(self, x):
        x = tf.linalg.matmul(x, self.cu(self.get_L()))
        x = tf.linalg.matmul(x, self.cu(self.get_U())) + self.b  # in the implementation there was only one bias
        if self.activation:
            x = self.get_l() * x + (1 - self.get_l()) * tf.math.softplus(x)
        return x

    def _inverse(self, y):
        if self.activation:
            y = self.inv_f(y)

        y = tf.linalg.matmul(y - self.b, tf.linalg.inv(self.cu(self.get_U())))
        y = tf.linalg.matmul(y, tf.linalg.inv(self.cu(self.get_L())))
        return y

    def _inverse_log_det_jacobian(self, y):
        return self._forward_log_det_jacobian(y)  # TODO: is this correct?

    def _forward_log_det_jacobian(self, x):
        J = tf.reduce_sum(tf.math.log(self.get_l() + (1 - self.get_l()) * tf.math.softplus(self.d)))  # Ju
        # Jl is 0
        if self.activation:  # else Ja is 0
            J += tf.reduce_sum(tf.math.log(self.df(x)))  # Ja
        return J
