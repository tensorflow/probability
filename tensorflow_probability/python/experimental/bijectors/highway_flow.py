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


    def __init__(self, width, residual_fraction_initial_value=0.5, activation_fn=None, validate_args=False, name='tri_res_net'):
        super(TriResNet, self).__init__(
            validate_args=validate_args,
            forward_min_event_ndims=1,
            name=name)

        self.width = width
        self.W = tf.Variable(np.random.normal(0, 0.01, (self.width, self.width)), trainable=True, dtype=tf.float32)
        self.b = tf.Variable(np.random.normal(0, 0.01, (self.width,)), trainable=True, dtype=tf.float32)  # bias

        # positive diagonal for U
        self.d = tf.Variable(np.random.normal(0, 0.01, (self.width,)), trainable=True, dtype=tf.float32)

        # TODO: add control that residual_fraction_initial_value is between 0 and 1
        self.residual_fraction = tfp.util.TransformedVariable(
            initial_value=residual_fraction_initial_value,
            bijector=tfb.Sigmoid())

        if activation_fn:
            self.activation_fn = activation_fn

        self.masku = tfe.numpy.tril(tf.ones((self.width, self.width)), -1)
        self.maskl = tfe.numpy.triu(tf.ones((self.width, self.width)), 1)

    def get_L(self):
        return self.W * self.maskl + tf.eye(self.width)

    def get_U(self):
        return self.W * self.masku + tf.linalg.diag(tf.math.softplus(self.d))

    def df(self, x):
        # derivative of activation
        return self.residual_fraction + (1 - self.residual_fraction) * tf.math.sigmoid(x) * (1 - tf.math.sigmoid(x))

    def cu(self, M):
        # convex update
        # same as doing as in the paper, but I guess easier to invert
        I = tf.eye(self.width)
        return self.residual_fraction * I + (1 - self.residual_fraction) * M

    def inv_f(self, y, N=20):
        # inverse with Newton iteration
        x = tf.ones(y.shape)
        for _ in range(N):
            x = x - (self.residual_fraction * x + (1 - self.residual_fraction) * tf.math.softplus(x) - y) / (self.df(x))
        return x

    def _forward(self, x):
        x = tf.linalg.matvec(x, self.cu(self.get_L()))
        x = tf.linalg.matvec(x, self.cu(self.get_U())) + self.b  # in the implementation there was only one bias
        if self.activation:
            x = self.residual_fraction * x + (1 - self.residual_fraction) * tf.math.softplus(x)
        return x

    def _inverse(self, y):
        if self.activation:
            y = self.inv_f(y)

        y = tf.linalg.matvec(y - self.b, tf.linalg.inv(self.cu(self.get_U())))
        y = tf.linalg.matvec(y, tf.linalg.inv(self.cu(self.get_L())))
        return y

    def _forward_log_det_jacobian(self, x):
        J = tf.reduce_sum(
            tf.math.log(self.residual_fraction + (1 - self.residual_fraction) * tf.math.softplus(self.d)))  # Ju
        # Jl is 0
        if self.activation:  # else Ja is 0
            J += tf.reduce_sum(tf.math.log(self.df(x)))  # Ja
        return J