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
"""The Gumbel distribution class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
# Dependency imports
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import tensor_shape


class _Gumbel(tf.distributions.Distribution):
  """The scalar Gumbel distribution with location `loc` and `scale` parameters.

  #### Mathematical details

  The probability density function (pdf) of this distribution is,

  ```none
  pdf(x; mu, sigma) = exp(-(x - mu) / sigma - exp(-(x - mu) / sigma))
  ```

  where `loc = mu` and `scale = sigma`.

  The cumulative density function of this distribution is,

  ```cdf(x; mu, sigma) = exp(-exp(-(x - mu) / sigma))```

  The Gumbel distribution is a member of the [location-scale family](
  https://en.wikipedia.org/wiki/Location-scale_family), i.e., it can be
  constructed as,

  ```none
  X ~ Gumbel(loc=0, scale=1)
  Y = loc + scale * X
  ```

  #### Examples

  Examples of initialization of one or a batch of distributions.

  ```python
  tfd = tfp.distributions

  # Define a single scalar Gumbel distribution.
  dist = tfd.Gumbel(loc=0., scale=3.)

  # Evaluate the cdf at 1, returning a scalar.
  dist.cdf(1.)

  # Define a batch of two scalar valued Gumbels.
  # The first has mean 1 and scale 11, the second 2 and 22.
  dist = tfd.Gumbel(loc=[1, 2.], scale=[11, 22.])

  # Evaluate the pdf of the first distribution on 0, and the second on 1.5,
  # returning a length two tensor.
  dist.prob([0, 1.5])

  # Get 3 samples, returning a 3 x 2 tensor.
  dist.sample([3])
  ```

  Arguments are broadcast when possible.

  ```python
  # Define a batch of two scalar valued Logistics.
  # Both have mean 1, but different scales.
  dist = tfd.Gumbel(loc=1., scale=[11, 22.])

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
               name="Gumbel"):
    """Construct Gumbel distributions with location and scale `loc` and `scale`.

    The parameters `loc` and `scale` must be shaped in a way that supports
    broadcasting (e.g. `loc + scale` is a valid operation).

    Args:
      loc: Floating point tensor, the means of the distribution(s).
      scale: Floating point tensor, the scales of the distribution(s).
        scale must contain only positive values.
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
      TypeError: if loc and scale are different dtypes.
    """
    parameters = dict(locals())
    with tf.name_scope(name, values=[loc, scale]) as name:
      with tf.control_dependencies([tf.assert_positive(scale)]
                                   if validate_args else []):
        self._loc = tf.identity(loc, name="loc")
        self._scale = tf.identity(scale, name="scale")
        tf.assert_same_float_dtype([self._loc, self._scale])
    super(_Gumbel, self).__init__(
        dtype=self._scale.dtype,
        reparameterization_type=tf.distributions.FULLY_REPARAMETERIZED,
        validate_args=validate_args,
        allow_nan_stats=allow_nan_stats,
        parameters=parameters,
        graph_parents=[self._loc, self._scale],
        name=name)

  @staticmethod
  def _param_shapes(sample_shape):
    return dict(
        zip(("loc", "scale"),
            ([tf.convert_to_tensor(sample_shape, dtype=tf.int32)] * 2)))

  @property
  def loc(self):
    """Distribution parameter for the location."""
    return self._loc

  @property
  def scale(self):
    """Distribution parameter for scale."""
    return self._scale

  def _batch_shape_tensor(self):
    return tf.broadcast_dynamic_shape(tf.shape(self.loc), tf.shape(self.scale))

  def _batch_shape(self):
    return tf.broadcast_static_shape(self.loc.get_shape(),
                                     self.scale.get_shape())

  def _event_shape_tensor(self):
    return tf.constant([], dtype=tf.int32)

  def _event_shape(self):
    return tensor_shape.scalar()

  def _sample_n(self, n, seed=None):
    # Uniform variates must be sampled from the open-interval `(0, 1)` rather
    # than `[0, 1)`. To do so, we use `np.finfo(self.dtype.as_numpy_dtype).tiny`
    # because it is the smallest, positive, "normal" number. A "normal" number
    # is such that the mantissa has an implicit leading 1. Normal, positive
    # numbers x, y have the reasonable property that, `x + y >= max(x, y)`. In
    # this case, a subnormal number (i.e., np.nextafter) can cause us to sample
    # 0.
    uniform = tf.random_uniform(
        shape=tf.concat([[n], self.batch_shape_tensor()], 0),
        minval=np.finfo(self.dtype.as_numpy_dtype).tiny,
        maxval=1.,
        dtype=self.dtype,
        seed=seed)
    sampled = -tf.log(-tf.log(uniform))
    return sampled * self.scale + self.loc

  def _log_prob(self, x):
    return self._log_unnormalized_prob(x) - self._log_normalization()

  def _log_cdf(self, x):
    return -tf.exp(-self._z(x))

  def _cdf(self, x):
    return tf.exp(-tf.exp(-self._z(x)))

  def _log_unnormalized_prob(self, x):
    z = self._z(x)
    return -z - tf.exp(-z)

  def _log_normalization(self):
    return tf.log(self.scale)

  def _entropy(self):
    # Use broadcasting rules to calculate the full broadcast sigma.
    scale = self.scale * tf.ones_like(self.loc)
    return 1 + tf.log(scale) + np.euler_gamma

  def _mean(self):
    return self.loc + self.scale * np.euler_gamma

  def _stddev(self):
    return self.scale * tf.ones_like(self.loc) * math.pi / math.sqrt(6)

  def _mode(self):
    return self.loc * tf.ones_like(self.scale)

  def _z(self, x):
    """Standardize input `x` to a unit logistic."""
    with tf.name_scope("standardize", values=[x]):
      return (x - self.loc) / self.scale
