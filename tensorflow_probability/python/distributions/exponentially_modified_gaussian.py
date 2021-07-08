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
"""The exponentially modified Gaussian distribution class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports

import tensorflow.compat.v2 as tf
from tensorflow_probability.python.bijectors import identity as identity_bijector
from tensorflow_probability.python.bijectors import softplus as softplus_bijector
from tensorflow_probability.python.distributions import distribution
from tensorflow_probability.python.distributions import exponential as exponential_lib
from tensorflow_probability.python.distributions import normal as normal_lib
from tensorflow_probability.python.internal import assert_util
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import parameter_properties
from tensorflow_probability.python.internal import reparameterization
from tensorflow_probability.python.internal import samplers
from tensorflow_probability.python.internal import special_math
from tensorflow_probability.python.internal import tensor_util
from tensorflow_probability.python.math import generic as tfp_math

__all__ = [
    'ExponentiallyModifiedGaussian',
]


class ExponentiallyModifiedGaussian(
    distribution.AutoCompositeTensorDistribution):
  """Exponentially modified Gaussian distribution.

  #### Mathematical details

  The exponentially modified Gaussian distribution is the sum of a normal
  distribution and an exponential distribution.
  ```none
  X ~ Normal(loc, scale)
  Y ~ Exponential(rate)
  Z = X + Y
  ```
  is equivalent to
  ```none
  Z ~ ExponentiallyModifiedGaussian(loc, scale, rate)
  ```

  #### Examples
  ```python
  tfd = tfp.distributions

  # Define a single scalar ExponentiallyModifiedGaussian distribution
  dist = tfd.ExponentiallyModifiedGaussian(loc=0., scale=1., rate=3.)

  # Evaluate the pdf at 1, returing a scalar.
  dist.prob(1.)
  ```


  """

  def __init__(self,
               loc,
               scale,
               rate,
               validate_args=False,
               allow_nan_stats=True,
               name='ExponentiallyModifiedGaussian'):
    """Construct an exponentially-modified Gaussian distribution.

    The Gaussian distribution has mean `loc` and stddev `scale`,
    and Exponential distribution has rate parameter `rate`.

    The parameters `loc`, `scale`, and `rate` must be shaped in a way that
    supports broadcasting (e.g. `loc + scale + rate` is a valid operation).
    Args:
      loc: Floating-point `Tensor`; the means of the distribution(s).
      scale: Floating-point `Tensor`; the stddevs of the distribution(s). Must
        contain only positive values.
      rate: Floating-point `Tensor`; the rate parameter for the exponential
        distribution.
      validate_args: Python `bool`, default `False`. When `True` distribution
        parameters are checked for validity despite possibly degrading runtime
        performance. When `False` invalid inputs may silently render incorrect
        outputs.
      allow_nan_stats: Python `bool`, default `True`. When `True`, statistics
        (e.g., mean, mode, variance) use the value "`NaN`" to indicate the
        result is undefined. When `False`, an exception is raised if one or more
        of the statistic's batch members are undefined.
      name: Python `str` name prefixed to Ops created by this class.

    Raises:
      TypeError: if `loc`, `scale`, and `rate` are not all the same `dtype`.
    """
    parameters = dict(locals())
    with tf.name_scope(name) as name:
      dtype = dtype_util.common_dtype([loc, scale, rate], dtype_hint=tf.float32)
      self._loc = tensor_util.convert_nonref_to_tensor(
          loc, dtype=dtype, name='loc')
      self._scale = tensor_util.convert_nonref_to_tensor(
          scale, dtype=dtype, name='scale')
      self._rate = tensor_util.convert_nonref_to_tensor(
          rate, dtype=dtype, name='rate')
      super(ExponentiallyModifiedGaussian, self).__init__(
          dtype=dtype,
          reparameterization_type=reparameterization.FULLY_REPARAMETERIZED,
          validate_args=validate_args,
          allow_nan_stats=allow_nan_stats,
          parameters=parameters,
          name=name)

  @staticmethod
  def _param_shapes(sample_shape):
    return dict(
        zip(('loc', 'scale', 'rate'),
            ([tf.convert_to_tensor(sample_shape, dtype=tf.int32)] * 3)))

  @classmethod
  def _parameter_properties(cls, dtype, num_classes=None):
    return dict(
        loc=parameter_properties.ParameterProperties(),
        scale=parameter_properties.ParameterProperties(
            default_constraining_bijector_fn=(
                lambda: softplus_bijector.Softplus(low=dtype_util.eps(dtype)))),
        rate=parameter_properties.ParameterProperties(
            default_constraining_bijector_fn=(
                lambda: softplus_bijector.Softplus(low=dtype_util.eps(dtype)))))

  @property
  def loc(self):
    """Distribution parameter for the mean of the normal distribution."""
    return self._loc

  @property
  def scale(self):
    """Distribution parameter for standard deviation of the normal distribution."""
    return self._scale

  @property
  def rate(self):
    """Distribution parameter for rate parameter of exponential distribution."""
    return self._rate

  def _event_shape_tensor(self):
    return tf.constant([], dtype=tf.int32)

  def _event_shape(self):
    return tf.TensorShape([])

  def _sample_n(self, n, seed=None):
    normal_seed, exp_seed = samplers.split_seed(seed, salt='emg_sample')
    # need to make sure component distributions are broadcast appropriately
    # for correct generation of samples
    loc = tf.convert_to_tensor(self.loc)
    rate = tf.convert_to_tensor(self.rate)
    scale = tf.convert_to_tensor(self.scale)
    batch_shape = self._batch_shape_tensor(loc=loc, scale=scale, rate=rate)
    loc_broadcast = tf.broadcast_to(loc, batch_shape)
    rate_broadcast = tf.broadcast_to(rate, batch_shape)
    normal_dist = normal_lib.Normal(loc=loc_broadcast, scale=scale)
    exp_dist = exponential_lib.Exponential(rate_broadcast)
    x = normal_dist.sample(n, normal_seed)
    y = exp_dist.sample(n, exp_seed)
    return x + y

  def _log_prob(self, x):
    loc = tf.convert_to_tensor(self.loc)
    rate = tf.convert_to_tensor(self.rate)
    scale = tf.convert_to_tensor(self.scale)
    two = dtype_util.as_numpy_dtype(x.dtype)(2.)
    z = (x - loc) / scale
    w = rate * scale
    return (tf.math.log(rate) + w / two * (w - 2 * z) +
            special_math.log_ndtr(z - w))

  def _log_cdf(self, x):
    rate = tf.convert_to_tensor(self.rate)
    scale = tf.convert_to_tensor(self.scale)
    x_centralized = x - self.loc
    u = rate * x_centralized
    v = rate * scale
    vsquared = tf.square(v)
    return tfp_math.log_sub_exp(
        special_math.log_ndtr(x_centralized / scale),
        -u + vsquared / 2. + special_math.log_ndtr((u - vsquared) / v))

  def _mean(self):
    return tf.broadcast_to(
        self.loc + 1 / self.rate, self._batch_shape_tensor())

  def _variance(self):
    return tf.broadcast_to(
        tf.square(self.scale) + 1 / tf.square(self.rate),
        self._batch_shape_tensor())

  def _parameter_control_dependencies(self, is_init):
    assertions = []

    if is_init:
      try:
        self._batch_shape()
      except ValueError:
        raise ValueError(
            'Arguments `loc`, `scale`, and `rate` must have compatible shapes; '
            'loc.shape={}, scale.shape={}, rate.shape={}.'.format(
                self.loc.shape, self.scale.shape, self.rate.shape))
      # We don't bother checking the shapes in the dynamic case because
      # all member functions access both arguments anyway.

    if is_init != tensor_util.is_ref(self.scale):
      assertions.append(assert_util.assert_positive(
          self.scale, message='Argument `scale` must be positive.'))

    if is_init != tensor_util.is_ref(self.rate):
      assertions.append(assert_util.assert_positive(
          self.rate, message='Argument `rate` must be positive.'))

    return assertions

  def _default_event_space_bijector(self):
    return identity_bijector.Identity(validate_args=self.validate_args)
