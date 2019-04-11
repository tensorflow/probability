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
"""The Half Normal distribution class."""

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


__all__ = [
    "HalfNormal",
]


class HalfNormal(distribution.Distribution):
  """The Half Normal distribution with scale `scale`.

  #### Mathematical details

  The half normal is a transformation of a centered normal distribution.
  If some random variable `X` has normal distribution,
  ```none
  X ~ Normal(0.0, scale)
  Y = |X|
  ```
  Then `Y` will have half normal distribution. The probability density
  function (pdf) is:

  ```none
  pdf(x; scale, x > 0) = sqrt(2) / (scale * sqrt(pi)) *
    exp(- 1/2 * (x / scale) ** 2)
  )
  ```
  Where `scale = sigma` is the standard deviation of the underlying normal
  distribution.

  #### Examples

  Examples of initialization of one or a batch of distributions.

  ```python
  # Define a single scalar HalfNormal distribution.
  dist = tfp.distributions.HalfNormal(scale=3.0)

  # Evaluate the cdf at 1, returning a scalar.
  dist.cdf(1.)

  # Define a batch of two scalar valued HalfNormals.
  # The first has scale 11.0, the second 22.0
  dist = tfp.distributions.HalfNormal(scale=[11.0, 22.0])

  # Evaluate the pdf of the first distribution on 1.0, and the second on 1.5,
  # returning a length two tensor.
  dist.prob([1.0, 1.5])

  # Get 3 samples, returning a 3 x 2 tensor.
  dist.sample([3])
  ```

  """

  def __init__(self,
               scale,
               validate_args=False,
               allow_nan_stats=True,
               name="HalfNormal"):
    """Construct HalfNormals with scale `scale`.

    Args:
      scale: Floating point tensor; the scales of the distribution(s).
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
    """
    parameters = dict(locals())
    with tf.name_scope(name) as name:
      scale = tf.convert_to_tensor(
          value=scale,
          name="scale",
          dtype=dtype_util.common_dtype([scale], preferred_dtype=tf.float32))
      with tf.control_dependencies(
          [assert_util.assert_positive(scale)] if validate_args else []):
        self._scale = tf.identity(scale, name="scale")
    super(HalfNormal, self).__init__(
        dtype=self._scale.dtype,
        reparameterization_type=reparameterization.FULLY_REPARAMETERIZED,
        validate_args=validate_args,
        allow_nan_stats=allow_nan_stats,
        parameters=parameters,
        graph_parents=[self._scale],
        name=name)

  @staticmethod
  def _param_shapes(sample_shape):
    return {"scale": tf.convert_to_tensor(value=sample_shape, dtype=tf.int32)}

  @classmethod
  def _params_event_ndims(cls):
    return dict(scale=0)

  @property
  def scale(self):
    """Distribution parameter for the scale."""
    return self._scale

  def _batch_shape_tensor(self):
    return tf.shape(input=self.scale)

  def _batch_shape(self):
    return self.scale.shape

  def _event_shape_tensor(self):
    return tf.constant([], dtype=tf.int32)

  def _event_shape(self):
    return tf.TensorShape([])

  def _sample_n(self, n, seed=None):
    shape = tf.concat([[n], self.batch_shape_tensor()], 0)
    sampled = tf.random.normal(
        shape=shape, mean=0., stddev=1., dtype=self.dtype, seed=seed)
    return tf.abs(sampled * self.scale)

  def _prob(self, x):
    coeff = np.sqrt(2) / self.scale / np.sqrt(np.pi)
    pdf = coeff * tf.exp(-0.5 * (x / self.scale)**2)
    return pdf * tf.cast(x >= 0, self.dtype)

  def _cdf(self, x):
    truncated_x = tf.nn.relu(x)
    return tf.math.erf(truncated_x / self.scale / np.sqrt(2.0))

  def _entropy(self):
    return 0.5 * tf.math.log(np.pi * self.scale**2.0 / 2.0) + 0.5

  def _mean(self):
    return self.scale * np.sqrt(2.0) / np.sqrt(np.pi)

  def _quantile(self, p):
    return np.sqrt(2.0) * self.scale * special_math.erfinv(p)

  def _mode(self):
    return tf.zeros(self.batch_shape_tensor())

  def _variance(self):
    return self.scale ** 2.0 * (1.0 - 2.0 / np.pi)


@kullback_leibler.RegisterKL(HalfNormal, HalfNormal)
def _kl_half_normal_half_normal(a, b, name=None):
  """Calculate the batched KL divergence KL(a || b) with a and b `HalfNormal`.

  Args:
    a: Instance of a `HalfNormal` distribution object.
    b: Instance of a `HalfNormal` distribution object.
    name: (optional) Name to use for created operations.
      default is "kl_half_normal_half_normal".

  Returns:
    Batchwise KL(a || b)
  """
  with tf.name_scope(name or "kl_half_normal_half_normal"):
    # Consistent with
    # http://www.mast.queensu.ca/~communications/Papers/gil-msc11.pdf, page 119
    return (tf.math.log(b.scale) - tf.math.log(a.scale) +
            (a.scale**2 - b.scale**2) / (2 * b.scale**2))
