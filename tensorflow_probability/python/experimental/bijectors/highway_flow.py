import numpy as np
import tensorflow as tf

import tensorflow_probability as tfp

tfd = tfp.distributions
tfb = tfp.bijectors
tfe = tf.experimental


def build_highway_flow_layer(width, residual_fraction_initial_value=0.5, activation_fn=None):
    # FIXME: should everything be in float32 or float64?
    # TODO: add control that residual_fraction_initial_value is between 0 and 1
    return HighwayFlow(
        width=width,
        residual_fraction=tfp.util.TransformedVariable(
            initial_value=np.asarray(residual_fraction_initial_value, dtype='float32'),
            bijector=tfb.Sigmoid()),
        activation_fn=activation_fn,
        bias=tf.Variable(np.random.normal(0, 0.01, (width,)), dtype=tf.float32),
        upper_diagonal_weights_matrix=tfp.util.TransformedVariable(
            initial_value=np.random.uniform(0., 1., (width, width)).astype('float32'),
            bijector=tfb.FillScaleTriL(diag_bijector=tfb.Softplus(), diag_shift=None)),
        lower_diagonal_weights_matrix=None
    )


class HighwayFlow(tfb.Bijector):

    def __init__(self, width, residual_fraction, activation_fn, bias, upper_diagonal_weights_matrix,
                 lower_diagonal_weights_matrix, validate_args=False,
                 name='highway_flow'):
        super(HighwayFlow, self).__init__(
            validate_args=validate_args,
            forward_min_event_ndims=1,  # FIXME: should this also be an argument of HighwayFlow __init__?
            name=name)

        self.width = width

        ###########################################
        # once we find a way to define L matrix as TransformedVariable, these two lines should be removed
        self.weights = tf.Variable(np.random.normal(0, 0.01, (width, width)),
                                   dtype=tf.float32)
        self.maskl = tfe.numpy.triu(tf.ones((self.width, self.width)), 1)
        ###########################################

        self.bias = bias

        self.residual_fraction = residual_fraction

        # still lower triangular, transposed is done in matvec.
        self.upper_diagonal_weights_matrix = upper_diagonal_weights_matrix

        # TODO: Not implemented yet
        self.lower_diagonal_weights_matrix = lower_diagonal_weights_matrix

        self.activation_fn = activation_fn

    def get_L(self):
        return self.weights * self.maskl + tf.eye(self.width)

    def df(self, x):
        # derivative of activation
        return self.residual_fraction + (1 - self.residual_fraction) * tf.math.sigmoid(x) * (1 - tf.math.sigmoid(x))

    def _convex_update(self, weights_matrix):
        # convex update
        # same as in the paper, but probably easier to invert
        identity_matrix = tf.eye(self.width)
        return self.residual_fraction * identity_matrix + (1 - self.residual_fraction) * weights_matrix

    def inv_f(self, y, N=20):
        # inverse with Newton iteration
        x = tf.ones(y.shape)
        for _ in range(N):
            x = x - (self.residual_fraction * x + (1 - self.residual_fraction) * tf.math.softplus(x) - y) / (self.df(x))
        return x

    def _forward(self, x):
        x = tf.linalg.matvec(self._convex_update(self.get_L()), x)
        x = tf.linalg.matvec(self._convex_update(self.upper_diagonal_weights_matrix), x,
                             transpose_a=True) + self.bias  # in the implementation there was only one bias
        if self.activation_fn:
            x = self.residual_fraction * x + (1 - self.residual_fraction) * self.activation_fn(x)
        return x

    def _inverse(self, y):
        if self.activation_fn:
            y = self.inv_f(y)

        # this works with y having shape [BATCH x WIDTH], don't know how well it generalizes
        y = tf.linalg.triangular_solve(tf.transpose(self._convex_update(self.upper_diagonal_weights_matrix)),
                                       tf.linalg.matrix_transpose(y - self.bias), lower=False)
        y = tf.linalg.triangular_solve(self._convex_update(self.get_L()), y)
        return tf.linalg.matrix_transpose(y)

    def _forward_log_det_jacobian(self, x):
        jacobian = tf.reduce_sum(
            tf.math.log(self.residual_fraction + (1 - self.residual_fraction) * tf.linalg.diag_part(
                self.upper_diagonal_weights_matrix)))  # jacobian from upper matrix
        # jacobian from lower matrix is 0
        # FIXME: need to compute jacobian depending on selected activation
        if self.activation_fn:  # else activation jacobian is 0
            jacobian += tf.reduce_sum(tf.math.log(self.df(x)))  # Ja
        return jacobian
