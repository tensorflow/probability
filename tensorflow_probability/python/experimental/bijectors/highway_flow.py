import numpy as np
import tensorflow.compat.v2 as tf

from tensorflow_probability.python import bijectors as tfb
from tensorflow_probability.python import util
from tensorflow_probability.python.experimental.bijectors import scalar_function_with_inferred_inverse

def build_highway_flow_layer(width, residual_fraction_initial_value=0.5, activation_fn=None):
    # TODO: add control that residual_fraction_initial_value is between 0 and 1
    residual_fraction_initial_value = tf.convert_to_tensor(residual_fraction_initial_value,
                                                           dtype_hint=tf.float32,
                                                           name='residual_fraction_initial_value')
    dtype = residual_fraction_initial_value.dtype
    return HighwayFlow(
        width=width,
        residual_fraction=util.TransformedVariable(
            initial_value=np.asarray(residual_fraction_initial_value),
            bijector=tfb.Sigmoid(),
            dtype=dtype),
        activation_fn=activation_fn,
        bias=tf.Variable(np.random.normal(0., 0.01, (width,)), dtype=dtype),
        upper_diagonal_weights_matrix=util.TransformedVariable(
            initial_value=np.tril(np.random.normal(0., 1., (width, width)), -1) + np.diag(
                np.random.uniform(size=width)),
            bijector=tfb.FillScaleTriL(diag_bijector=tfb.Softplus(), diag_shift=None),
            dtype=dtype),
        lower_diagonal_weights_matrix=util.TransformedVariable(
            initial_value=np.random.normal(0., 1., (width, width)),
            bijector=tfb.Chain([tfb.TransformDiagonal(diag_bijector=tfb.Shift(1.)),
                                tfb.Pad(paddings=[(1, 0), (0, 1)]),
                                tfb.FillTriangular()]),
            dtype=dtype)
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

        self.bias = bias

        self.residual_fraction = residual_fraction

        # still lower triangular, transposed is done in matvec.
        self.upper_diagonal_weights_matrix = upper_diagonal_weights_matrix

        self.lower_diagonal_weights_matrix = lower_diagonal_weights_matrix

        self.activation_fn = activation_fn

    def df(self, x):
        # derivative of activation
        return self.residual_fraction + (1. - self.residual_fraction) * tf.math.sigmoid(x) * (1. - tf.math.sigmoid(x))

    def _convex_update(self, weights_matrix):
        # convex update
        # same as in the paper, but probably easier to invert
        identity_matrix = tf.eye(self.width)
        return self.residual_fraction * identity_matrix + (1. - self.residual_fraction) * weights_matrix

    def inv_f(self, y, N=20):
        # inverse with Newton iteration
        x = tf.ones(y.shape)
        for _ in range(N):
            x = x - (self.residual_fraction * x + (1. - self.residual_fraction) * tf.math.softplus(x) - y) / (
                self.df(x))
        return x

    def _augmented_forward(self, x):

        # upper mmatrix jacobian
        fldj = tf.reduce_sum(
            tf.math.log(self.residual_fraction + (1. - self.residual_fraction) * tf.linalg.diag_part(
                self.upper_diagonal_weights_matrix)))  # jacobian from upper matrix
        # jacobian from lower matrix is 0

        x = tf.linalg.matvec(self._convex_update(self.lower_diagonal_weights_matrix), x)
        x = tf.linalg.matvec(self._convex_update(self.upper_diagonal_weights_matrix), x,
                             transpose_a=True) + self.bias  # in the implementation there was only one bias
        if self.activation_fn:
            activation_layer = scalar_function_with_inferred_inverse.ScalarFunctionWithInferredInverse(
                lambda x: self.residual_fraction * x + (1. - self.residual_fraction) * self.activation_fn(x))
            fldj += activation_layer.forward_log_det_jacobian(x, self.forward_min_event_ndims)
            x = activation_layer.forward(x)
        return x, {'ildj': -fldj, 'fldj': fldj}

    def _forward(self, x):
        y, _ = self._augmented_forward(x)
        return y

    def _inverse(self, y):
        if self.activation_fn:
            y = self.inv_f(y)

            # FIXME: this way of using inverse activation does not seem to work because residual_fraction is a variable
            # activation_layer = scalar_function_with_inferred_inverse.ScalarFunctionWithInferredInverse(
            # lambda x: self.residual_fraction * x + (1. - self.residual_fraction) * self.activation_fn(x))
            # y = activation_layer.inverse(y)

        # this works with y having shape [BATCH x WIDTH], don't know how well it generalizes
        y = tf.linalg.triangular_solve(tf.transpose(self._convex_update(self.upper_diagonal_weights_matrix)),
                                       tf.linalg.matrix_transpose(y - self.bias), lower=False)
        y = tf.linalg.triangular_solve(self._convex_update(self.lower_diagonal_weights_matrix), y)
        return tf.linalg.matrix_transpose(y)

    def _forward_log_det_jacobian(self, x):
        cached = self._cache.forward_attributes(x)
        # If LDJ isn't in the cache, call forward once.
        if 'fldj' not in cached:
            _, attrs = self._augmented_forward(x)
            cached.update(attrs)
        return cached['fldj']
