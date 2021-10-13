# Copyright 2018 The TensorFlow Probability Authors.
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
"""The Logistic distribution class."""

# Dependency imports
import numpy as np
import tensorflow.compat.v2 as tf

from tensorflow_probability.python.bijectors import identity as identity_bijector
from tensorflow_probability.python.bijectors import softplus as softplus_bijector
from tensorflow_probability.python.distributions import distribution
from tensorflow_probability.python.internal import assert_util
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import parameter_properties
from tensorflow_probability.python.internal import prefer_static as ps
from tensorflow_probability.python.internal import reparameterization
from tensorflow_probability.python.internal import samplers
from tensorflow_probability.python.internal import tensor_util


class Logistic(distribution.AutoCompositeTensorDistribution):
  """The Logistic distribution with location `loc` and `scale` parameters.

  #### Mathematical details

  The cumulative density function of this distribution is:

  ```none
  cdf(x; mu, sigma) = 1 / (1 + exp(-(x - mu) / sigma))
  ```

  where `loc = mu` and `scale = sigma`.

  The Logistic distribution is a member of the [location-scale family](
  https://en.wikipedia.org/wiki/Location-scale_family), i.e., it can be
  constructed as,

  ```none
  X ~ Logistic(loc=0, scale=1)
  Y = loc + scale * X
  ```

  #### Examples

  Examples of initialization of one or a batch of distributions.

  ```python
  tfd = tfp.distributions

  # Define a single scalar Logistic distribution.
  dist = tfd.Logistic(loc=0., scale=3.)

  # Evaluate the cdf at 1, returning a scalar.
  dist.cdf(1.)

  # Define a batch of two scalar valued Logistics.
  # The first has mean 1 and scale 11, the second 2 and 22.
  dist = tfd.Logistic(loc=[1, 2.], scale=[11, 22.])

  # Evaluate the pdf of the first distribution on 0, and the second on 1.5,
  # returning a length two tensor.
  dist.prob([0, 1.5])

  # Get 3 samples, returning a 3 x 2 tensor.
  dist.sample([3])

  # Arguments are broadcast when possible.
  # Define a batch of two scalar valued Logistics.
  # Both have mean 1, but different scales.
  dist = tfd.Logistic(loc=1., scale=[11, 22.])

  # Evaluate the pdf of both distributions on the same point, 3.0,
  # returning a length 2 tensor.
  dist.prob(3.0)
  ```

  """

  def __init__(self,
               loc,
               scale,
               validate_args=False,
               allow_nan_stats=True,
               name='Logistic'):
    """Construct Logistic distributions with mean and scale `loc` and `scale`.

    The parameters `loc` and `scale` must be shaped in a way that supports
    broadcasting (e.g. `loc + scale` is a valid operation).

    Args:
      loc: Floating point tensor, the means of the distribution(s).
      scale: Floating point tensor, the scales of the distribution(s). Must
        contain only positive values.
      validate_args: Python `bool`, default `False`. When `True` distribution
        parameters are checked for validity despite possibly degrading runtime
        performance. When `False` invalid inputs may silently render incorrect
        outputs.
      allow_nan_stats: Python `bool`, default `True`. When `True`, statistics
        (e.g., mean, mode, variance) use the value '`NaN`' to indicate the
        result is undefined. When `False`, an exception is raised if one or
        more of the statistic's batch members are undefined.
      name: The name to give Ops created by the initializer.

    Raises:
      TypeError: if loc and scale are different dtypes.
    """
    parameters = dict(locals())
    with tf.name_scope(name) as name:
      dtype = dtype_util.common_dtype([loc, scale], dtype_hint=tf.float32)
      self._loc = tensor_util.convert_nonref_to_tensor(
          loc, name='loc', dtype=dtype)
      self._scale = tensor_util.convert_nonref_to_tensor(
          scale, name='scale', dtype=dtype)
      super(Logistic, self).__init__(
          dtype=self._scale.dtype,
          reparameterization_type=reparameterization.FULLY_REPARAMETERIZED,
          validate_args=validate_args,
          allow_nan_stats=allow_nan_stats,
          parameters=parameters,
          name=name)

  @classmethod
  def _parameter_properties(cls, dtype, num_classes=None):
    # pylint: disable=g-long-lambda
    return dict(
        loc=parameter_properties.ParameterProperties(),
        scale=parameter_properties.ParameterProperties(
            default_constraining_bijector_fn=(
                lambda: softplus_bijector.Softplus(low=dtype_util.eps(dtype)))))
    # pylint: enable=g-long-lambda

  @property
  def loc(self):
    """Distribution parameter for the location."""
    return self._loc

  @property
  def scale(self):
    """Distribution parameter for scale."""
    return self._scale

  def _event_shape_tensor(self):
    return tf.constant([], dtype=tf.int32)

  def _event_shape(self):
    return tf.TensorShape([])

  def _sample_n(self, n, seed=None):
    loc = tf.convert_to_tensor(self.loc)
    scale = tf.convert_to_tensor(self.scale)
    shape = ps.concat([[n], self._batch_shape_tensor(loc=loc, scale=scale)], 0)
    # Uniform variates must be sampled from the open-interval `(0, 1)` rather
    # than `[0, 1)`. To do so, we use
    # `np.finfo(dtype_util.as_numpy_dtype(self.dtype)).tiny` because it is the
    # smallest, positive, 'normal' number. A 'normal' number is such that the
    # mantissa has an implicit leading 1. Normal, positive numbers x, y have the
    # reasonable property that, `x + y >= max(x, y)`. In this case, a subnormal
    # number (i.e., np.nextafter) can cause us to sample 0.
    uniform = samplers.uniform(
        shape=shape,
        minval=np.finfo(dtype_util.as_numpy_dtype(self.dtype)).tiny,
        maxval=1.,
        dtype=self.dtype,
        seed=seed)
    sampled = tf.math.log(uniform) - tf.math.log1p(-uniform)
    return sampled * scale + loc

  def _log_prob(self, x):
    loc = tf.convert_to_tensor(self.loc)
    scale = tf.convert_to_tensor(self.scale)
    z = (x - loc) / scale
    return -z - 2. * tf.math.softplus(-z) - tf.math.log(scale)

  def _log_cdf(self, x):
    return -tf.math.softplus(-self._z(x))

  def _cdf(self, x):
    return tf.sigmoid(self._z(x))

  def _log_survival_function(self, x):
    return -tf.math.softplus(self._z(x))

  def _survival_function(self, x):
    return tf.sigmoid(-self._z(x))

  def _entropy(self):
    scale = tf.convert_to_tensor(self.scale)
    return tf.broadcast_to(2. + tf.math.log(scale),
                           self._batch_shape_tensor(scale=scale))

  def _mean(self):
    loc = tf.convert_to_tensor(self.loc)
    return tf.broadcast_to(loc, self._batch_shape_tensor(loc=loc))

  def _stddev(self):
    scale = tf.convert_to_tensor(self.scale)
    return tf.broadcast_to(
        scale * tf.constant(np.pi / np.sqrt(3), dtype=scale.dtype),
        self._batch_shape_tensor(scale=scale))

  def _mode(self):
    return self._mean()

  def _z(self, x):
    """Standardize input `x` to a unit logistic."""
    with tf.name_scope('standardize'):
      return (x - self.loc) / self.scale

  def _quantile(self, x):
    return self.loc + self.scale * (tf.math.log(x) - tf.math.log1p(-x))

  def _default_event_space_bijector(self):
    return identity_bijector.Identity(validate_args=self.validate_args)

  def _parameter_control_dependencies(self, is_init):
    if is_init:
      dtype_util.assert_same_float_dtype([self.loc, self.scale])
    if not self.validate_args:
      return []
    assertions = []
    if is_init != tensor_util.is_ref(self._scale):
      assertions.append(assert_util.assert_positive(
          self._scale, message='Argument `scale` must be positive.'))
    return assertions
