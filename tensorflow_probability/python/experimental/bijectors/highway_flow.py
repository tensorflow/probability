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
from tensorflow_probability.python.experimental.bijectors import scalar_function_with_inferred_inverse
from tensorflow_probability.python.internal import cache_util
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import prefer_static as ps
from tensorflow_probability.python.internal import samplers
from tensorflow_probability.python.internal import tensor_util

__all__ = [
    'build_trainable_highway_flow',
    'HighwayFlow'
]


def build_trainable_highway_flow(width,
                                 residual_fraction_initial_value=0.5,
                                 activation_fn=None,
                                 gate_first_n=None,
                                 seed=None,
                                 validate_args=False):
  """Builds a HighwayFlow parameterized by trainable variables.

  The variables are transformed to enforce the following parameter constraints:

  - `residual_fraction` is bounded between 0 and 1.
  - `upper_diagonal_weights_matrix` is a randomly initialized (lower) diagonal
     matrix with positive diagonal of size `width x width`.
  - `lower_diagonal_weights_matrix` is a randomly initialized lower diagonal
     matrix with ones on the diagonal of size `width x width`;
  - `bias` is a randomly initialized vector of size `width`.

  Args:
    width: Input dimension of the bijector.
    residual_fraction_initial_value: Initial value for gating parameter, must be
      between 0 and 1.
    activation_fn: Callable invertible activation function
      (e.g., `tf.nn.softplus`), or `None`.
    gate_first_n: Decides which part of the input should be gated (useful for
      example when using auxiliary variables).
    seed: Seed for random initialization of the weights.
    validate_args: Python `bool`. Whether to validate input with runtime
        assertions.
        Default value: `False`.

  Returns:
    trainable_highway_flow: The initialized bijector.
  """

  residual_fraction_initial_value = tf.convert_to_tensor(
      residual_fraction_initial_value,
      dtype_hint=tf.float32,
      name='residual_fraction_initial_value')
  dtype = residual_fraction_initial_value.dtype

  bias_seed, upper_seed, lower_seed = samplers.split_seed(seed, n=3)
  lower_bijector = tfb.Chain([
      tfb.TransformDiagonal(diag_bijector=tfb.Shift(1.)),
      tfb.Pad(paddings=[(1, 0), (0, 1)]),
      tfb.FillTriangular()
  ])
  unconstrained_lower_initial_values = samplers.normal(
      shape=lower_bijector.inverse_event_shape([width, width]),
      mean=0.,
      stddev=.01,
      seed=lower_seed)
  upper_bijector = tfb.FillScaleTriL(
      diag_bijector=tfb.Softplus(), diag_shift=None)
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
          initial_value=upper_bijector.forward(
              unconstrained_upper_initial_values),
          bijector=upper_bijector,
          dtype=dtype),
      lower_diagonal_weights_matrix=util.TransformedVariable(
          initial_value=lower_bijector.forward(
              unconstrained_lower_initial_values),
          bijector=lower_bijector,
          dtype=dtype),
      gate_first_n=gate_first_n,
      validate_args=validate_args)


# TODO(b/188814119): Decorate as auto composite tensor and add test.
class HighwayFlow(tfb.Bijector):
  """Implements an Highway Flow bijector [1].

  HighwayFlow interpolates the vector-valued input `X` with the transformations
  at each step of the bjiector. The Highway Flow can be used as building block
  for a Cascading flow [1] or as a generic normalizing flow.

  The transformation consists of a convex update between the input `X` and a
  linear transformation of `X` followed by activation with the form `g(A @
  X + b)`, where `g(.)` is a differentiable non-decreasing activation
  function, and `A` and `b` are weights.

  The convex update is regulated by a residual fraction `lam`
  constrained between 0 and 1. Conceptually, we'd like to represent the
  function:
  `Y = lam * X + (1 - lam) * g(A @ X + b)`.

  To make this transformation invertible, the bijector is split in three
  convex updates:
   - `Y1 = lam * X + (1 - lam) * L @ X`, with `L` lower diagonal matrix with
     ones on the diagonal;
   - `Y2 = lam * Y1 + (1 - lam) * (U @ Y1 + b)`, with `U` upper diagonal matrix
     with positive diagonal;
   - `Y = lam * Y2 + (1 - lam) * g(Y2)`.
  where the identity function is mixed in at each step to ensure invertibility.
  While this is not exactly equivalent to the original expression, it is
  'morally similar' in that it similarly specializes to the
  identity function when `lam = 1`.

  The function `build_trainable_highway_flow` helps initializing the bijector
  with the variables respecting the various constraints.

  For more details on Highway Flow and Cascading Flows see [1].

  #### Usage example
  ```python
  tfd = tfp.distributions
  tfb = tfp.bijectors

  dim = 4 # last input dimension

  bijector = build_trainable_highway_flow(dim, activation_fn=tf.nn.softplus)
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
      forward_name='_augmented_forward', inverse_name='_augmented_inverse')

  def __init__(self,
               residual_fraction,
               activation_fn,
               bias,
               upper_diagonal_weights_matrix,
               lower_diagonal_weights_matrix,
               gate_first_n,
               validate_args=False,
               name=None):
    """Initializes the HighwayFlow.

    Args:
      residual_fraction: Scalar `Tensor` used for the convex update, must be
        between 0 and 1.
      activation_fn: Callable invertible activation function
      (e.g., `tf.nn.softplus`), or `None`.
      bias: Bias vector.
      upper_diagonal_weights_matrix: Lower diagional matrix of size (width,
        width) with positive diagonal (is transposed to Upper diagonal within
        the bijector).
      lower_diagonal_weights_matrix: Lower diagonal matrix with ones on the main
        diagional.
      gate_first_n: Integer number of initial dimensions to gate using
        `residual_fraction`. A value of `None` defaults to gating all dimensions
        (`gate_first_n == width`). Other values specify that it is only
        necessary to be able to represent the identity function over some
        prefix of the transformed dimensions.
        Default value: `None`.
      validate_args: Python `bool`. Whether to validate input with runtime
        assertions.
        Default value: `False`.
      name: Python `str` name for ops created by this object.
    """
    parameters = dict(locals())
    name = name or 'highway_flow'
    dtype = dtype_util.common_dtype([
        residual_fraction, bias, upper_diagonal_weights_matrix,
        lower_diagonal_weights_matrix
    ], dtype_hint=tf.float32)
    with tf.name_scope(name) as name:
      self._width = ps.shape(bias)[-1]
      self._bias = tensor_util.convert_nonref_to_tensor(
          bias, dtype=dtype, name='bias')
      self._residual_fraction = tensor_util.convert_nonref_to_tensor(
          residual_fraction, dtype=dtype, name='residual_fraction')
      # The upper matrix is still lower triangular. The transpose is done in
      # the _inverse and _forward methods, within matvec.
      self._upper_diagonal_weights_matrix = (
          tensor_util.convert_nonref_to_tensor(
              upper_diagonal_weights_matrix,
              dtype=dtype,
              name='upper_diagonal_weights_matrix'))
      self._lower_diagonal_weights_matrix = (
          tensor_util.convert_nonref_to_tensor(
              lower_diagonal_weights_matrix,
              dtype=dtype,
              name='lower_diagonal_weights_matrix'))
      self._activation_fn = activation_fn
      self._gate_first_n = self.width if gate_first_n is None else gate_first_n
      self._num_ungated = self.width - self.gate_first_n

      super(HighwayFlow, self).__init__(
          validate_args=validate_args,
          forward_min_event_ndims=1,
          parameters=parameters,
          dtype=dtype,
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

  @property
  def num_ungated(self):
    return self._num_ungated

  def _gated_residual_fraction(self):
    """Returns a vector of residual fractions that encodes gated dimensions."""
    return tf.concat(
        [
            self.residual_fraction * tf.ones([self.gate_first_n],
                                             dtype=self.dtype),
            tf.zeros([self.num_ungated], dtype=self.dtype)
        ], axis=0)

  def _activation_bijector(self, gated_residual_fraction):
    return (scalar_function_with_inferred_inverse.
            ScalarFunctionWithInferredInverse(
                lambda x, lam: lam * x + (1 - lam) * self.activation_fn(x),
                additional_scalar_parameters_requiring_gradients=[
                    gated_residual_fraction]))

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
    gated_residual_fraction = self._gated_residual_fraction()
    fldj = tf.reduce_sum(
        tf.math.log(gated_residual_fraction +
                    (1 - gated_residual_fraction) * tf.linalg.diag_part(
                        self.upper_diagonal_weights_matrix)),
        axis=-1)
    x = (gated_residual_fraction * x +
         (1 - gated_residual_fraction) * tf.linalg.matvec(
             self.lower_diagonal_weights_matrix, x))
    x = (gated_residual_fraction * x +
         (1 - gated_residual_fraction) * tf.linalg.matvec(
             self.upper_diagonal_weights_matrix, x, transpose_a=True))
    x = x + (1 - gated_residual_fraction) * self.bias

    if self.activation_fn:
      bij = self._activation_bijector(gated_residual_fraction)
      fldj += bij.forward_log_det_jacobian(x, event_ndims=1)
      x = bij.forward(x)

    return x, {'ildj': -fldj, 'fldj': fldj}

  def _augmented_inverse(self, y):
    """Computes inverse and inverse_log_det_jacobian transformations.

    Args:
      y: input of the (inverse) bijectorr.

    Returns:
      y after inverse flow and a dict containing inverse and forward log
      determinant of the jacobian.
    """

    gated_residual_fraction = self._gated_residual_fraction()
    ildj = -tf.reduce_sum(
        tf.math.log(
            gated_residual_fraction +
            (1 - gated_residual_fraction) * tf.linalg.diag_part(
                self.upper_diagonal_weights_matrix)),
        axis=-1)

    if self.activation_fn:
      bij = self._activation_bijector(gated_residual_fraction)
      ildj += bij.inverse_log_det_jacobian(y, event_ndims=1)
      y = bij.inverse(y)

    y = y - (1 - gated_residual_fraction) * self.bias

    y = y[..., tf.newaxis]  # Triangular solve requires matrix input.
    y = tf.linalg.triangular_solve(
        # Apply gating over columns (not rows) since this is transposed.
        (gated_residual_fraction[..., tf.newaxis, :] *
         tf.eye(self.width, dtype=self.dtype) +
         ((1 - gated_residual_fraction)[..., tf.newaxis, :] *
          self.upper_diagonal_weights_matrix)),
        y,
        lower=True,
        adjoint=True)
    y = tf.linalg.triangular_solve(
        (gated_residual_fraction[..., tf.newaxis] *
         tf.eye(self.width, dtype=self.dtype) +
         ((1 - gated_residual_fraction)[..., tf.newaxis] *
          self.lower_diagonal_weights_matrix)),
        y)
    y = y[..., 0]

    return y, {'ildj': ildj, 'fldj': -ildj}

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
