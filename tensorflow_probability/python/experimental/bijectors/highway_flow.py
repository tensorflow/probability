import tensorflow.compat.v2 as tf

from tensorflow_probability.python import bijectors as tfb
from tensorflow_probability.python import util
from tensorflow_probability.python.internal import cache_util
from tensorflow_probability.python.experimental.bijectors import scalar_function_with_inferred_inverse
from tensorflow_probability.python.internal import samplers


def build_highway_flow_layer(width, residual_fraction_initial_value=0.5, activation_fn=None, seed=None):
    # TODO: add control that residual_fraction_initial_value is between 0 and 1
    residual_fraction_initial_value = tf.convert_to_tensor(residual_fraction_initial_value,
                                                           dtype_hint=tf.float32,
                                                           name='residual_fraction_initial_value')
    dtype = residual_fraction_initial_value.dtype

    bias_seed, upper_seed, lower_seed, diagonal_seed = samplers.split_seed(seed, n=4)

    return HighwayFlow(
        width=width,
        residual_fraction=util.TransformedVariable(
            initial_value=tf.convert_to_tensor(residual_fraction_initial_value),
            bijector=tfb.Sigmoid(),
            dtype=dtype),
        activation_fn=activation_fn,
        bias=tf.Variable(samplers.normal((width,), 0., 0.01, seed=bias_seed), dtype=dtype, name='bias'),
        upper_diagonal_weights_matrix=util.TransformedVariable(
            initial_value=tf.experimental.numpy.tril(samplers.normal((width, width), 0., 1., seed=upper_seed),
                                                     -1) + tf.linalg.diag(
                samplers.uniform((width,), minval=0., maxval=1., seed=diagonal_seed)),
            bijector=tfb.FillScaleTriL(diag_bijector=tfb.Softplus(), diag_shift=None),
            dtype=dtype),
        lower_diagonal_weights_matrix=util.TransformedVariable(
            initial_value=samplers.normal((width, width), 0., 1., seed=lower_seed),
            bijector=tfb.Chain([tfb.TransformDiagonal(diag_bijector=tfb.Shift(1.)),
                                tfb.Pad(paddings=[(1, 0), (0, 1)]),
                                tfb.FillTriangular()]),
            dtype=dtype)
    )


class HighwayFlow(tfb.Bijector):

    # todo: update comments?
    # HighWay Flow simultaneously computes `forward` and `fldj` (and `inverse`/`ildj`),
    # so we override the bijector cache to update the LDJ entries of attrs on
    # forward/inverse inverse calls (instead of updating them only when the LDJ
    # methods themselves are called).
    _cache = cache_util.BijectorCacheWithGreedyAttrs(
        forward_name='_augmented_forward',
        inverse_name='_augmented_inverse')

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
        # TODO: transpose directly here or in the definition of TransformedVariable?
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

        # todo: see how to optimize _convex_update
        x = tf.linalg.matvec(self._convex_update(self.lower_diagonal_weights_matrix), x)
        x = tf.linalg.matvec(self._convex_update(tf.transpose(self.upper_diagonal_weights_matrix)), x) + self.bias  # in the implementation there was only one bias
        if self.activation_fn:
            #fldj += tf.reduce_sum(tf.math.log(self.df(x)))
            #x = self.residual_fraction * x + (1. - self.residual_fraction) * self.activation_fn(x)
            activation_layer = scalar_function_with_inferred_inverse.ScalarFunctionWithInferredInverse(
                lambda x: self.residual_fraction * x + (1. - self.residual_fraction) * self.activation_fn(x))
            fldj += activation_layer.forward_log_det_jacobian(x, self.forward_min_event_ndims)
            x = activation_layer.forward(x)
        return x, {'ildj': -fldj, 'fldj': fldj}

    def _augmented_inverse(self, y):
        ildj = tf.reduce_sum(
            tf.math.log(self.residual_fraction + (1. - self.residual_fraction) * tf.linalg.diag_part(
                self.upper_diagonal_weights_matrix)))
        if self.activation_fn:
            # y = self.inv_f(y)
            # ildj += tf.reduce_sum(tf.math.log(self.df(y)))
            residual_fraction = tf.identity(self.residual_fraction)
            # FIXME: this way of using inverse activation does not seem to work because residual_fraction is a variable
            activation_layer = scalar_function_with_inferred_inverse.ScalarFunctionWithInferredInverse(
                lambda x: residual_fraction * x + (1. - residual_fraction) * self.activation_fn(x))
            y = activation_layer.inverse(y)
            ildj += activation_layer.inverse_log_det_jacobian(y, self.forward_min_event_ndims)

        # this works with y having shape [BATCH x WIDTH], don't know how well it generalizes
        y = tf.linalg.triangular_solve(self._convex_update(tf.transpose(self.upper_diagonal_weights_matrix)),
                                       tf.linalg.matrix_transpose(y - self.bias), lower=False)
        y = tf.linalg.triangular_solve(self._convex_update(self.lower_diagonal_weights_matrix), y)
        return tf.linalg.matrix_transpose(y), {'ildj': ildj, 'fldj': -ildj}

    def _forward(self, x):
        y, _ = self._augmented_forward(x)
        return y

    def _inverse(self, y):
        x, _ = self._augmented_inverse(y)
        return x

    def _forward_log_det_jacobian(self, x):
        cached = self._cache.forward_attributes(x)
        # If LDJ isn't in the cache, call forward once.
        if 'fldj' not in cached:
            _, attrs = self._augmented_forward(x)
            cached.update(attrs)
        return cached['fldj']

    def _inverse_log_det_jacobian(self, y):
        cached = self._cache.inverse_attributes(y)
        # If LDJ isn't in the cache, call inverse once.
        if 'ildj' not in cached:
            _, attrs = self._augmented_inverse(y)
            cached.update(attrs)
        return cached['ildj']