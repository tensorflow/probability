# Copyright 2021 The TensorFlow Probability Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

"""Highway Flow bijector."""

import tensorflow.compat.v2 as tf

from tensorflow_probability.python import bijectors as tfb
from tensorflow_probability.python import util
from tensorflow_probability.python.internal import cache_util
from tensorflow_probability.python.internal import samplers


def build_highway_flow_layer(width,
                             residual_fraction_initial_value=0.5,
                             activation_fn=False,
                             gate_first_n=-1,
                             seed=None):
  """Builds HighwayFlow making sure that all the requirements are satisfied.

  Args:
    width: Input dimension of the bijector.
    residual_fraction_initial_value: Initial value for gating parameter, must be
      between 0 and 1.
    activation_fn: Whether or not use SoftPlus activation function.
    gate_first_n: Decides which part of the input should be gated (useful for
    example when using auxiliary variables).
    seed: Seed for random initialization of the weights.

  Returns:
    The initialized bijector with the following elements:
      `residual_fraction` is bounded between 0 and 1.
      `upper_diagonal_weights_matrix` is a randomly initialized (lower) diagonal
      matrix with positive diagonal of size `width x width`.
      `lower_diagonal_weights_matrix` is a randomly initialized lower diagonal
      matrix with ones on the diagonal of size `width x width`;
      `bias` is a randomly initialized vector of size `width`
  """

  if gate_first_n == -1:
    gate_first_n = width
  # TODO: add control that residual_fraction_initial_value is between 0 and 1
  residual_fraction_initial_value = tf.convert_to_tensor(
    residual_fraction_initial_value,
    dtype_hint=tf.float32,
    name='residual_fraction_initial_value')
  dtype = residual_fraction_initial_value.dtype

  bias_seed, upper_seed, lower_seed = samplers.split_seed(
    seed, n=3)
  lower_bijector = tfb.Chain(
    [tfb.TransformDiagonal(diag_bijector=tfb.Shift(1.)),
     tfb.Pad(paddings=[(1, 0), (0, 1)]),
     tfb.FillTriangular()])
  unconstrained_lower_initial_values = samplers.normal(
    shape=lower_bijector.inverse_event_shape([width, width]),
    mean=0.,
    stddev=.01,
    seed=lower_seed)
  upper_bijector = tfb.FillScaleTriL(diag_bijector=tfb.Softplus(),
                                     diag_shift=None)
  unconstrained_upper_initial_values = samplers.normal(
    shape=upper_bijector.inverse_event_shape([width, width]),
    mean=0.,
    stddev=.01,
    seed=upper_seed)

  return HighwayFlow(
    residual_fraction=util.TransformedVariable(
      initial_value=residual_fraction_initial_value,
      bijector=tfb.Sigmoid(),
      dtype=dtype),
    activation_fn=activation_fn,
    bias=tf.Variable(
      samplers.normal((width,), mean=0., stddev=0.01, seed=bias_seed),
      dtype=dtype),
    upper_diagonal_weights_matrix=util.TransformedVariable(
      initial_value=upper_bijector.forward(unconstrained_upper_initial_values),
      bijector=upper_bijector,
      dtype=dtype),
    lower_diagonal_weights_matrix=util.TransformedVariable(
      initial_value=lower_bijector.forward(unconstrained_lower_initial_values),
      bijector=lower_bijector,
      dtype=dtype),
    gate_first_n=gate_first_n
  )


class HighwayFlow(tfb.Bijector):
  """Implements an Highway Flow bijector [1].

  HighwayFlow interpolates the input `X` with the transformations at each step
  of the bjiector. The Highway Flow can be used as building block for a
  Cascading flow [1] or as a generic normalizing flow.

  The transformation consists of a convex update between the input `X` and a
  linear transformation of `X` followed by activation with the form `g(A @
  X + b)`, where `g(.)` is a differentiable non-decreasing activation
  function, and `A` and `b` are trainable weights.

  The convex update is regulated by a trainable residual fraction `l`
  constrained between 0 and 1, and can be
  formalized as:
  `Y = l * X + (1 - l) * g(A @ X + b)`.

  To make this transformation invertible, the bijector is split in three
  convex updates:
   - `Y1 = l * X + (1 - l) * L @ X`, with `L` lower diagonal matrix with ones
   on the diagonal;
   - `Y2 = l * Y1 + (1 - l) * (U @ Y1 + b)`, with `U` upper diagonal matrix
   with positive diagonal;
   - `Y = l * Y2 + (1 - l) * g(Y2)`

  The function `build_highway_flow_layer` helps initializing the bijector
  with the variables respecting the various constraints.

  For more details on Highway Flow and Cascading Flows see [1].

  #### Usage example
  ```python
  tfd = tfp.distributions
  tfb = tfp.bijectors

  dim = 4 # last input dimension

  bijector = build_highway_flow_layer(dim, activation_fn=True)
  y = bijector.forward(x)  # forward mapping
  x = bijector.inverse(y)  # inverse mapping
  base = tfd.MultivariateNormalDiag(loc=tf.zeros(dim)) # Base distribution
  transformed_distribution = tfd.TransformedDistribution(base, bijector)
  ```

  #### References

  [1]: Ambrogioni, Luca, Gianluigi Silvestri, and Marcel van Gerven.
  "Automatic variational inference with cascading flows." arXiv preprint
  arXiv:2102.04801 (2021).
  """

  # HighWay Flow simultaneously computes `forward` and `fldj`
  # (and `inverse`/`ildj`), so we override the bijector cache to update the
  # LDJ entries of attrs on forward/inverse inverse calls (instead of
  # updating them only when the LDJ methods themselves are called).

  _cache = cache_util.BijectorCacheWithGreedyAttrs(
    forward_name='_augmented_forward',
    inverse_name='_augmented_inverse')

  def __init__(self, residual_fraction, activation_fn, bias,
               upper_diagonal_weights_matrix,
               lower_diagonal_weights_matrix,
               gate_first_n,
               validate_args=False,
               name=None):
    """Initializes the HighwayFlow.
    Args:
      residual_fraction: Scalar `Tensor` used for the convex update, must be
        between 0 and 1.
      activation_fn: Boolean to decide whether to use SoftPlus (True) activation
        or no activation (False).
      bias: Bias vector.
      upper_diagonal_weights_matrix: Lower diagional matrix of size
        (width, width) with positive diagonal (is transposed to Upper diagonal
        within the bijector).
      lower_diagonal_weights_matrix: Lower diagonal matrix with ones on the main
        diagional.
      gate_first_n: Integer that decides what part of the input is gated.
    """
    parameters = dict(locals())
    name = name or 'highway_flow'
    with tf.name_scope(name) as name:
      self._width = tf.shape(bias)[-1]
      self._bias = bias
      self._residual_fraction = residual_fraction
      # The upper matrix is still lower triangular, transpose is done in
      # _inverse and _forwars metowds, within matvec.
      self._upper_diagonal_weights_matrix = upper_diagonal_weights_matrix
      self._lower_diagonal_weights_matrix = lower_diagonal_weights_matrix
      self._activation_fn = activation_fn
      self._gate_first_n = gate_first_n

      super(HighwayFlow, self).__init__(
        validate_args=validate_args,
        forward_min_event_ndims=1,
        parameters=parameters,
        name=name)

  @property
  def bias(self):
    return self._bias

  @property
  def width(self):
    return self._width

  @property
  def residual_fraction(self):
    return self._residual_fraction

  @property
  def upper_diagonal_weights_matrix(self):
    return self._upper_diagonal_weights_matrix

  @property
  def lower_diagonal_weights_matrix(self):
    return self._lower_diagonal_weights_matrix

  @property
  def activation_fn(self):
    return self._activation_fn

  @property
  def gate_first_n(self):
    return self._gate_first_n

  def _derivative_of_softplus(self, x):
    return tf.concat([(self.residual_fraction) * tf.ones(
      self.gate_first_n), tf.zeros(self.width - self.gate_first_n)],
                     axis=0) + (
             tf.concat([(1. - self.residual_fraction) * tf.ones(
               self.gate_first_n), tf.ones(self.width - self.gate_first_n)],
                       axis=0)) * tf.math.sigmoid(x)

  def _convex_update(self, weights_matrix):
    return tf.concat(
      [self.residual_fraction * tf.eye(num_rows=self.gate_first_n,
                                       num_columns=self.width),
       tf.zeros([self.width - self.gate_first_n, self.width])],
      axis=0) + tf.concat([(
                               1. - self.residual_fraction) * tf.ones(
      self.gate_first_n), tf.ones(self.width - self.gate_first_n)],
                          axis=0) * weights_matrix

  def _inverse_of_softplus(self, y, n=20):
    """Inverse of the activation layer with softplus using Newton iteration."""
    x = tf.ones(y.shape)
    for _ in range(n):
      x = x - (tf.concat([(self.residual_fraction) * tf.ones(
        self.gate_first_n), tf.zeros(self.width - self.gate_first_n)],
                         axis=0) * x + tf.concat(
        [(1. - self.residual_fraction) * tf.ones(
          self.gate_first_n), tf.ones(self.width - self.gate_first_n)],
        axis=0) * tf.math.softplus(
        x) - y) / (
            self._derivative_of_softplus(x))
    return x

  def _augmented_forward(self, x):
    """Computes forward and forward_log_det_jacobian transformations.

    Args:
      x: Input of the bijector.

    Returns:
      x after forward flow and a dict containing forward and inverse log
      determinant of the jacobian.
    """

    # Log determinant term from the upper matrix. Note that the log determinant
    # of the lower matrix is zero.

    fldj = tf.zeros(x.shape[:-1]) + tf.reduce_sum(
      tf.math.log(tf.concat([(self.residual_fraction) * tf.ones(
        self.gate_first_n), tf.zeros(self.width - self.gate_first_n)],
                            axis=0) + (
                    tf.concat([(1. - self.residual_fraction) * tf.ones(
                      self.gate_first_n),
                               tf.ones(self.width - self.gate_first_n)],
                              axis=0)) * tf.linalg.diag_part(
        self.upper_diagonal_weights_matrix)))
    x = x[tf.newaxis, ...]
    x = tf.linalg.matvec(
      self._convex_update(self.lower_diagonal_weights_matrix), x)
    x = tf.linalg.matvec(self._convex_update(self.upper_diagonal_weights_matrix),
      x, transpose_a=True)
    x += (tf.concat([(1. - self.residual_fraction) * tf.ones(
      self.gate_first_n), tf.ones(self.width - self.gate_first_n)],
                   axis=0) * self.bias)[tf.newaxis, ...]

    if self.activation_fn:
      fldj += tf.reduce_sum(tf.math.log(self._derivative_of_softplus(x[0])),
                            -1)
      x = tf.concat([(self.residual_fraction) * tf.ones(
        self.gate_first_n), tf.zeros(self.width - self.gate_first_n)],
                    axis=0) * x + tf.concat(
        [(1. - self.residual_fraction) * tf.ones(
          self.gate_first_n), tf.ones(self.width - self.gate_first_n)],
        axis=0) * tf.nn.softplus(x)

    return tf.squeeze(x, 0), {'ildj': -fldj, 'fldj': fldj}

  def _augmented_inverse(self, y):
    """Computes inverse and inverse_log_det_jacobian transformations.

    Args:
      y: input of the (inverse) bijectorr.

    Returns:
      y after inverse flow and a dict containing inverse and forward log
      determinant of the jacobian.
    """

    ildj = tf.zeros(y.shape[:-1]) - tf.reduce_sum(
      tf.math.log(tf.concat([(self.residual_fraction) * tf.ones(
        self.gate_first_n), tf.zeros(self.width - self.gate_first_n)],
                            axis=0) + tf.concat(
        [(1. - self.residual_fraction) * tf.ones(
          self.gate_first_n), tf.ones(self.width - self.gate_first_n)],
        axis=0) * tf.linalg.diag_part(
        self.upper_diagonal_weights_matrix)))


    if self.activation_fn:
      y = self._inverse_of_softplus(y)
      ildj -= tf.reduce_sum(tf.math.log(self._derivative_of_softplus(y)),
                            -1)

    y = y[..., tf.newaxis]

    y = y - (tf.concat([(1. - self.residual_fraction) * tf.ones(
      self.gate_first_n), tf.ones(self.width - self.gate_first_n)],
                      axis=0) * self.bias)[..., tf.newaxis]
    y = tf.linalg.triangular_solve(
      self._convex_update(self.upper_diagonal_weights_matrix), y,
      lower=True, adjoint=True)
    y = tf.linalg.triangular_solve(
      self._convex_update(self.lower_diagonal_weights_matrix), y)

    return tf.squeeze(y, -1), {'ildj': ildj, 'fldj': -ildj}

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
