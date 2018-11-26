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

# Dependency imports
import numpy as np
import tensorflow as tf

from tensorflow_probability.python.bijectors import gumbel as gumbel_bijector
from tensorflow_probability.python.bijectors import invert as invert_bijector
from tensorflow_probability.python.distributions import transformed_distribution
from tensorflow_probability.python.distributions import uniform
from tensorflow_probability.python.internal import distribution_util
from tensorflow_probability.python.internal import dtype_util


class Gumbel(transformed_distribution.TransformedDistribution):
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
        Default value: `False`.
      allow_nan_stats: Python `bool`, default `True`. When `True`,
        statistics (e.g., mean, mode, variance) use the value "`NaN`" to
        indicate the result is undefined. When `False`, an exception is raised
        if one or more of the statistic's batch members are undefined.
        Default value: `True`.
      name: Python `str` name prefixed to Ops created by this class.
        Default value: `'Gumbel'`.

    Raises:
      TypeError: if loc and scale are different dtypes.
    """
    with tf.name_scope(name, values=[loc, scale]) as name:
      dtype = dtype_util.common_dtype([loc, scale], preferred_dtype=tf.float32)
      loc = tf.convert_to_tensor(loc, name="loc", dtype=dtype)
      scale = tf.convert_to_tensor(scale, name="scale", dtype=dtype)
      with tf.control_dependencies([tf.assert_positive(scale)]
                                   if validate_args else []):
        loc = tf.identity(loc, name="loc")
        scale = tf.identity(scale, name="scale")
        tf.assert_same_float_dtype([loc, scale])
        self._gumbel_bijector = gumbel_bijector.Gumbel(
            loc=loc, scale=scale, validate_args=validate_args)

      super(Gumbel, self).__init__(
          distribution=uniform.Uniform(
              low=tf.zeros([], dtype=loc.dtype),
              high=tf.ones([], dtype=loc.dtype),
              allow_nan_stats=allow_nan_stats),
          # The Gumbel bijector encodes the quantile
          # function as the forward, and hence needs to
          # be inverted.
          bijector=invert_bijector.Invert(self._gumbel_bijector),
          batch_shape=distribution_util.get_broadcast_shape(loc, scale),
          name=name)

  @staticmethod
  def _param_shapes(sample_shape):
    return dict(
        zip(("loc", "scale"),
            ([tf.convert_to_tensor(sample_shape, dtype=tf.int32)] * 2)))

  @property
  def loc(self):
    """Distribution parameter for the location."""
    return self._gumbel_bijector.loc

  @property
  def scale(self):
    """Distribution parameter for scale."""
    return self._gumbel_bijector.scale

  def _entropy(self):
    # Use broadcasting rules to calculate the full broadcast sigma.
    scale = self.scale * tf.ones_like(self.loc)
    return 1. + tf.log(scale) + np.euler_gamma

  def _mean(self):
    return self.loc + self.scale * np.euler_gamma

  def _stddev(self):
    return self.scale * tf.ones_like(self.loc) * np.pi / np.sqrt(6)

  def _mode(self):
    return self.loc * tf.ones_like(self.scale)
