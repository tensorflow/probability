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
"""The Normal (Gaussian) distribution class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
import numpy as np

import tensorflow.compat.v2 as tf

from tensorflow_probability.python.distributions import distribution
from tensorflow_probability.python.distributions import kullback_leibler
from tensorflow_probability.python.internal import assert_util
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import reparameterization
from tensorflow_probability.python.internal import special_math
from tensorflow_probability.python.internal import tensor_util


__all__ = [
    "Normal",
]


class Normal(distribution.Distribution):
  """The Normal distribution with location `loc` and `scale` parameters.

  #### Mathematical details

  The probability density function (pdf) is,

  ```none
  pdf(x; mu, sigma) = exp(-0.5 (x - mu)**2 / sigma**2) / Z
  Z = (2 pi sigma**2)**0.5
  ```

  where `loc = mu` is the mean, `scale = sigma` is the std. deviation, and, `Z`
  is the normalization constant.

  The Normal distribution is a member of the [location-scale family](
  https://en.wikipedia.org/wiki/Location-scale_family), i.e., it can be
  constructed as,

  ```none
  X ~ Normal(loc=0, scale=1)
  Y = loc + scale * X
  ```

  #### Examples

  Examples of initialization of one or a batch of distributions.

  ```python
  import tensorflow_probability as tfp
  tfd = tfp.distributions

  # Define a single scalar Normal distribution.
  dist = tfd.Normal(loc=0., scale=3.)

  # Evaluate the cdf at 1, returning a scalar.
  dist.cdf(1.)

  # Define a batch of two scalar valued Normals.
  # The first has mean 1 and standard deviation 11, the second 2 and 22.
  dist = tfd.Normal(loc=[1, 2.], scale=[11, 22.])

  # Evaluate the pdf of the first distribution on 0, and the second on 1.5,
  # returning a length two tensor.
  dist.prob([0, 1.5])

  # Get 3 samples, returning a 3 x 2 tensor.
  dist.sample([3])
  ```

  Arguments are broadcast when possible.

  ```python
  # Define a batch of two scalar valued Normals.
  # Both have mean 1, but different standard deviations.
  dist = tfd.Normal(loc=1., scale=[11, 22.])

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
               name="Normal"):
    """Construct Normal distributions with mean and stddev `loc` and `scale`.

    The parameters `loc` and `scale` must be shaped in a way that supports
    broadcasting (e.g. `loc + scale` is a valid operation).

    Args:
      loc: Floating point tensor; the means of the distribution(s).
      scale: Floating point tensor; the stddevs of the distribution(s).
        Must contain only positive values.
      validate_args: Python `bool`, default `False`. When `True` distribution
        parameters are checked for validity despite possibly degrading runtime
        performance. When `False` invalid inputs may silently render incorrect
        outputs.
      allow_nan_stats: Python `bool`, default `True`. When `True`,
        statistics (e.g., mean, mode, variance) use the value "`NaN`" to
        indicate the result is undefined. When `False`, an exception is raised
        if one or more of the statistic's batch members are undefined.
      name: Python `str` name prefixed to Ops created by this class.

    Raises:
      TypeError: if `loc` and `scale` have different `dtype`.
    """
    parameters = dict(locals())
    with tf.name_scope(name) as name:
      dtype = dtype_util.common_dtype([loc, scale], dtype_hint=tf.float32)
      self._loc = tensor_util.convert_immutable_to_tensor(
          loc, dtype=dtype, name="loc")
      self._scale = tensor_util.convert_immutable_to_tensor(
          scale, dtype=dtype, name="scale")
      super(Normal, self).__init__(
          dtype=dtype,
          reparameterization_type=reparameterization.FULLY_REPARAMETERIZED,
          validate_args=validate_args,
          allow_nan_stats=allow_nan_stats,
          parameters=parameters,
          name=name)

  @staticmethod
  def _param_shapes(sample_shape):
    return dict(
        zip(("loc", "scale"),
            ([tf.convert_to_tensor(sample_shape, dtype=tf.int32)] * 2)))

  @classmethod
  def _params_event_ndims(cls):
    return dict(loc=0, scale=0)

  @property
  def loc(self):
    """Distribution parameter for the mean."""
    return self._loc

  @property
  def scale(self):
    """Distribution parameter for standard deviation."""
    return self._scale

  def _batch_shape_tensor(self, loc=None, scale=None):
    return tf.broadcast_dynamic_shape(
        tf.shape(self.loc if loc is None else loc),
        tf.shape(self.scale if scale is None else scale))

  def _batch_shape(self):
    return tf.broadcast_static_shape(
        self.loc.shape,
        self.scale.shape)

  def _event_shape_tensor(self):
    return tf.constant([], dtype=tf.int32)

  def _event_shape(self):
    return tf.TensorShape([])

  def _sample_n(self, n, seed=None):
    loc = tf.convert_to_tensor(self.loc)
    scale = tf.convert_to_tensor(self.scale)
    shape = tf.concat([[n], self._batch_shape_tensor(loc=loc, scale=scale)], 0)
    sampled = tf.random.normal(
        shape=shape, mean=0., stddev=1., dtype=self.dtype, seed=seed)
    return sampled * scale + loc

  def _log_prob(self, x):
    scale = tf.convert_to_tensor(self.scale)
    log_unnormalized = -0.5 * tf.square(self._z(x, scale=scale))
    log_normalization = (0.5 * np.log(2. * np.pi)) + tf.math.log(scale)
    return log_unnormalized - log_normalization

  def _log_cdf(self, x):
    return special_math.log_ndtr(self._z(x))

  def _cdf(self, x):
    return special_math.ndtr(self._z(x))

  def _log_survival_function(self, x):
    return special_math.log_ndtr(-self._z(x))

  def _survival_function(self, x):
    return special_math.ndtr(-self._z(x))

  def _entropy(self):
    # Use broadcasting rules to calculate the full broadcast scale.
    scale = self.scale * tf.ones_like(self.loc)
    return 0.5 * np.log(2. * np.pi * np.e) + tf.math.log(scale)

  def _mean(self):
    return self.loc * tf.ones_like(self.scale)

  def _quantile(self, p):
    return special_math.ndtri(p) * self.scale + self.loc

  def _stddev(self):
    return self.scale * tf.ones_like(self.loc)

  def _mode(self):
    return self._mean()

  def _z(self, x, scale=None):
    """Standardize input `x` to a unit normal."""
    with tf.name_scope("standardize"):
      return (x - self.loc) / (self.scale if scale is None else scale)

  def _parameter_control_dependencies(self, is_init):
    if not self.validate_args:
      return []
    assertions = []
    if is_init != tensor_util.is_mutable(self.scale):
      assertions.append(assert_util.assert_positive(
          self.scale,
          message="Argument `scale` must be positive."))
    return assertions


@kullback_leibler.RegisterKL(Normal, Normal)
def _kl_normal_normal(n_a, n_b, name=None):
  """Calculate the batched KL divergence KL(n_a || n_b) with n_a and n_b Normal.

  Args:
    n_a: instance of a Normal distribution object.
    n_b: instance of a Normal distribution object.
    name: (optional) Name to use for created operations.
      default is "kl_normal_normal".

  Returns:
    Batchwise KL(n_a || n_b)
  """
  with tf.name_scope(name or "kl_normal_normal"):
    one = tf.constant(1, dtype=n_a.dtype)
    two = tf.constant(2, dtype=n_a.dtype)
    half = tf.constant(0.5, dtype=n_a.dtype)
    s_a_squared = tf.square(n_a.scale)
    s_b_squared = tf.square(n_b.scale)
    ratio = s_a_squared / s_b_squared
    return (tf.square(n_a.loc - n_b.loc) / (two * s_b_squared) + half *
            (ratio - one - tf.math.log(ratio)))
