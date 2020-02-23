# Copyright 2020 The TensorFlow Probability Authors.
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
"""Johnson's SU distribution class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
import tensorflow.compat.v2 as tf

from tensorflow_probability.python import math
from tensorflow_probability.python.bijectors import inline as inline_bijector
from tensorflow_probability.python.bijectors import invert as invert_bijector
from tensorflow_probability.python.bijectors import scale as scale_bijector
from tensorflow_probability.python.bijectors import shift as shift_bijector
from tensorflow_probability.python.distributions import normal
from tensorflow_probability.python.distributions import transformed_distribution
from tensorflow_probability.python.internal import assert_util
from tensorflow_probability.python.internal import distribution_util
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import tensor_util


__all__ = [
    'JohnsonSU',
]


class JohnsonSU(transformed_distribution.TransformedDistribution):
  """Johnson's SU-distribution.

  This distribution has parameters: shape parameters `gamma` and `delta`,
  location `loc`, and `scale`.

  #### Mathematical details

  The probability density function (pdf) is,

  ```none
  pdf(x; gamma, delta, xi, sigma) = exp(-0.5 (gamma + delta arcsinh(y))**2) / Z
  where,
  y = (x - xi) / sigma
  Z = sigma sqrt(2 pi) sqrt(1 + y**2) / delta
  ```

  where:
  * `loc = xi`,
  * `scale = sigma`, and,
  * `Z` is the normalization constant.

  The JohnsonSU distribution is a member of the [location-scale family](
  https://en.wikipedia.org/wiki/Location-scale_family), i.e., it can be
  constructed as,

  ```none
  X ~ JohnsonSU(gamma, delta, loc=0, scale=1)
  Y = loc + scale * X
  ```

  #### Examples

  Examples of initialization of one or a batch of distributions.

  ```python
  import tensorflow_probability as tfp
  tfd = tfp.distributions

  # Define a single scalar Johnson's SU-distribution.
  single_dist = tfd.JohnsonSU(gamma=-2., delta=2., loc=1.1, scale=1.5)

  # Evaluate the pdf at 1, returning a scalar Tensor.
  single_dist.prob(1.)

  # Define a batch of two scalar valued Johnson SU's.
  # The first has shape parameters 1 and 2, mean 3, and scale 11.
  # The second 4, 5, 6 and 22.
  multi_dist = tfd.JohnsonSU(gamma=[1, 4], delta=[2, 5],
                             loc=[3, 6], scale=[11, 22.])

  # Evaluate the pdf of the first distribution on 0, and the second on 1.5,
  # returning a length two tensor.
  multi_dist.prob([0, 1.5])

  # Get 3 samples, returning a 3 x 2 tensor.
  multi_dist.sample(3)
  ```

  Arguments are broadcast when possible.

  ```python
  # Define a batch of two Johnson's SU distributions.
  # Both have gamma 2, delta 3 and mean 1, but different scales.
  dist = tfd.JohnsonSU(gamma=2, delta=3, loc=1, scale=[11, 22.])

  # Evaluate the pdf of both distributions on the same point, 3.0,
  # returning a length 2 tensor.
  dist.prob(3.0)
  ```

  Compute the gradients of samples w.r.t. the parameters:

  ```python
  gamma = tf.Variable(2.0)
  delta = tf.Variable(3.0)
  loc = tf.Variable(2.0)
  scale = tf.Variable(11.0)
  dist = tfd.JohnsonSU(gamma=gamma, delta=delta, loc=loc, scale=scale)
  with tf.GradientTape() as tape:
    samples = dist.sample(5)  # Shape [5]
    loss = tf.reduce_mean(tf.square(samples))  # Arbitrary loss function
  # Unbiased stochastic gradients of the loss function
  grads = tape.gradient(loss, dist.variables)
  ```

  """

  def __init__(self,
               gamma,
               delta,
               loc,
               scale,
               validate_args=False,
               allow_nan_stats=True,
               name=None):
    """Construct Johnson's SU distributions.

    The distributions have shape parameteres `delta` and `gamma`, mean `loc`,
    and scale `scale`.

    The parameters `delta`, `gamma`, `loc`, and `scale` must be shaped in a way
    that supports broadcasting (e.g. `gamma + delta + loc + scale` is a valid
    operation).

    Args:
      gamma: Floating-point `Tensor`. The shape parameter(s) of the
        distribution(s).
      delta: Floating-point `Tensor`. The shape parameter(s) of the
        distribution(s). `delta` must contain only positive values.
      loc: Floating-point `Tensor`. The mean(s) of the distribution(s).
      scale: Floating-point `Tensor`. The scaling factor(s) for the
        distribution(s). Note that `scale` is not technically the standard
        deviation of this distribution but has semantics more similar to
        standard deviation than variance.
      validate_args: Python `bool`, default `False`. When `True` distribution
        parameters are checked for validity despite possibly degrading runtime
        performance. When `False` invalid inputs may silently render incorrect
        outputs.
      allow_nan_stats: Python `bool`, default `True`. When `True`,
        statistics (e.g., mean, mode, variance) use the value '`NaN`' to
        indicate the result is undefined. When `False`, an exception is raised
        if one or more of the statistic's batch members are undefined.
      name: Python `str` name prefixed to Ops created by this class.

    Raises:
      TypeError: if any of gamma, delta, loc and scale are different dtypes.
    """
    parameters = dict(locals())
    with tf.name_scope(name or 'JohnsonSU') as name:
      dtype = dtype_util.common_dtype([gamma, delta, loc, scale], tf.float32)
      self._gamma = tensor_util.convert_nonref_to_tensor(
          gamma, name='gamma', dtype=dtype)
      self._delta = tensor_util.convert_nonref_to_tensor(
          delta, name='delta', dtype=dtype)
      self._loc = tensor_util.convert_nonref_to_tensor(
          loc, name='loc', dtype=dtype)
      self._scale = tensor_util.convert_nonref_to_tensor(
          scale, name='scale', dtype=dtype)
      dtype_util.assert_same_float_dtype((self._gamma, self._delta,
                                          self._loc, self._scale))

      norm_affine = invert_bijector.Invert(
          shift_bijector.Shift(
              shift=self._gamma,
              validate_args=validate_args)(
                  scale_bijector.Scale(
                      scale=self._delta,
                      validate_args=validate_args
                  )
              )
      )

      sinh = inline_bijector.Inline(
          forward_fn=tf.sinh,
          inverse_fn=tf.asinh,
          forward_log_det_jacobian_fn=lambda x: tf.math.log(tf.math.cosh(x)),
          inverse_log_det_jacobian_fn=lambda y: -0.5 * math.log1psquare(y),
          forward_min_event_ndims=0,
          is_increasing=lambda: True,
          name='sinh'
      )

      affine = shift_bijector.Shift(
          shift=self._loc,
          validate_args=validate_args)(
              scale_bijector.Scale(
                  scale=self._scale,
                  validate_args=validate_args
              )
          )

      batch_shape = distribution_util.get_broadcast_shape(
          self._gamma, self._delta, self._loc, self._scale)

      super(JohnsonSU, self).__init__(
          distribution=normal.Normal(
              loc=tf.zeros([], dtype=dtype),
              scale=tf.ones([], dtype=dtype),
              validate_args=validate_args,
              allow_nan_stats=allow_nan_stats),
          bijector=affine(sinh(norm_affine)),
          batch_shape=batch_shape,
          validate_args=validate_args,
          parameters=parameters,
          name=name)

  @staticmethod
  def _param_shapes(sample_shape):
    return dict(
        zip(('gamma', 'delta', 'loc', 'scale'),
            ([tf.convert_to_tensor(sample_shape, dtype=tf.int32)] * 4)))

  @classmethod
  def _params_event_ndims(cls):
    return dict(gamma=0, delta=0, loc=0, scale=0)

  @property
  def gamma(self):
    """Gamma shape parameters in these Johnson's SU distribution(s)."""
    return self._gamma

  @property
  def delta(self):
    """Delta shape parameters in these Johnson's SU distribution(s)."""
    return self._delta

  @property
  def loc(self):
    """Locations of these Johnson's SU distribution(s)."""
    return self._loc

  @property
  def scale(self):
    """Scaling factors of these Johnson's SU distribution(s)."""
    return self._scale

  def _mean(self):
    batch_shape_tensor = self.batch_shape_tensor()
    gamma = tf.broadcast_to(self.gamma, batch_shape_tensor)
    delta = tf.broadcast_to(self.delta, batch_shape_tensor)
    scale = tf.broadcast_to(self.scale, batch_shape_tensor)
    loc = tf.broadcast_to(self.loc, batch_shape_tensor)

    return loc - scale * tf.math.exp(0.5 / tf.math.square(delta)) * \
      tf.math.sinh(gamma / delta)

  def _variance(self):
    batch_shape_tensor = self.batch_shape_tensor()
    gamma = tf.broadcast_to(self.gamma, batch_shape_tensor)
    delta = tf.broadcast_to(self.delta, batch_shape_tensor)
    scale = tf.broadcast_to(self.scale, batch_shape_tensor)

    return 0.5 * scale**2 * tf.math.expm1(1./delta**2) * \
        (tf.math.exp(1/delta**2) * tf.math.cosh(2. * gamma / delta) + 1.)

  def _parameter_control_dependencies(self, is_init):
    assertions = super(JohnsonSU, self)._parameter_control_dependencies(is_init)

    if is_init != tensor_util.is_ref(self.delta):
      assertions.append(assert_util.assert_positive(
          self.delta, message='Argument `delta` must be positive.'))
    if is_init != tensor_util.is_ref(self.scale):
      assertions.append(assert_util.assert_positive(
          self.scale, message='Argument `scale` must be positive.'))

    return assertions
