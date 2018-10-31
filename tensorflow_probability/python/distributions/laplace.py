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
"""The Laplace distribution class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

# Dependency imports
import numpy as np
import tensorflow as tf

from tensorflow_probability.python.distributions import distribution
from tensorflow_probability.python.distributions import kullback_leibler
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import reparameterization
from tensorflow_probability.python.internal import special_math
from tensorflow.python.framework import tensor_shape

__all__ = [
    "Laplace",
]


class Laplace(distribution.Distribution):
  """The Laplace distribution with location `loc` and `scale` parameters.

  #### Mathematical details

  The probability density function (pdf) of this distribution is,

  ```none
  pdf(x; mu, sigma) = exp(-|x - mu| / sigma) / Z
  Z = 2 sigma
  ```

  where `loc = mu`, `scale = sigma`, and `Z` is the normalization constant.

  Note that the Laplace distribution can be thought of two exponential
  distributions spliced together "back-to-back."

  The Lpalce distribution is a member of the [location-scale family](
  https://en.wikipedia.org/wiki/Location-scale_family), i.e., it can be
  constructed as,

  ```none
  X ~ Laplace(loc=0, scale=1)
  Y = loc + scale * X
  ```

  """

  def __init__(self,
               loc,
               scale,
               validate_args=False,
               allow_nan_stats=True,
               name="Laplace"):
    """Construct Laplace distribution with parameters `loc` and `scale`.

    The parameters `loc` and `scale` must be shaped in a way that supports
    broadcasting (e.g., `loc / scale` is a valid operation).

    Args:
      loc: Floating point tensor which characterizes the location (center)
        of the distribution.
      scale: Positive floating point tensor which characterizes the spread of
        the distribution.
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
      TypeError: if `loc` and `scale` are of different dtype.
    """
    parameters = dict(locals())
    with tf.name_scope(name, values=[loc, scale]) as name:
      dtype = dtype_util.common_dtype([loc, scale], tf.float32)
      loc = tf.convert_to_tensor(loc, name="loc", dtype=dtype)
      scale = tf.convert_to_tensor(scale, name="scale", dtype=dtype)
      with tf.control_dependencies([tf.assert_positive(scale)] if
                                   validate_args else []):
        self._loc = tf.identity(loc)
        self._scale = tf.identity(scale)
        tf.assert_same_float_dtype([self._loc, self._scale])
      super(Laplace, self).__init__(
          dtype=dtype,
          reparameterization_type=reparameterization.FULLY_REPARAMETERIZED,
          validate_args=validate_args,
          allow_nan_stats=allow_nan_stats,
          parameters=parameters,
          graph_parents=[self._loc, self._scale],
          name=name)

  @staticmethod
  def _param_shapes(sample_shape):
    return dict(
        zip(("loc", "scale"), ([tf.convert_to_tensor(
            sample_shape, dtype=tf.int32)] * 2)))

  @property
  def loc(self):
    """Distribution parameter for the location."""
    return self._loc

  @property
  def scale(self):
    """Distribution parameter for scale."""
    return self._scale

  def _batch_shape_tensor(self):
    return tf.broadcast_dynamic_shape(
        tf.shape(self.loc), tf.shape(self.scale))

  def _batch_shape(self):
    return tf.broadcast_static_shape(
        self.loc.shape, self.scale.shape)

  def _event_shape_tensor(self):
    return tf.constant([], dtype=tf.int32)

  def _event_shape(self):
    return tensor_shape.scalar()

  def _sample_n(self, n, seed=None):
    shape = tf.concat([[n], self.batch_shape_tensor()], 0)
    # Uniform variates must be sampled from the open-interval `(-1, 1)` rather
    # than `[-1, 1)`. In the case of `(0, 1)` we'd use
    # `np.finfo(self.dtype.as_numpy_dtype).tiny` because it is the smallest,
    # positive, "normal" number. However, the concept of subnormality exists
    # only at zero; here we need the smallest usable number larger than -1,
    # i.e., `-1 + eps/2`.
    uniform_samples = tf.random_uniform(
        shape=shape,
        minval=np.nextafter(self.dtype.as_numpy_dtype(-1.),
                            self.dtype.as_numpy_dtype(0.)),
        maxval=1.,
        dtype=self.dtype,
        seed=seed)
    return (self.loc - self.scale * tf.sign(uniform_samples) *
            tf.log1p(-tf.abs(uniform_samples)))

  def _log_prob(self, x):
    return self._log_unnormalized_prob(x) - self._log_normalization()

  def _prob(self, x):
    return tf.exp(self._log_prob(x))

  def _log_cdf(self, x):
    return special_math.log_cdf_laplace(self._z(x))

  def _log_survival_function(self, x):
    return special_math.log_cdf_laplace(-self._z(x))

  def _cdf(self, x):
    z = self._z(x)
    return (0.5 + 0.5 * tf.sign(z) *
            (1. - tf.exp(-tf.abs(z))))

  def _log_unnormalized_prob(self, x):
    return -tf.abs(self._z(x))

  def _log_normalization(self):
    return math.log(2.) + tf.log(self.scale)

  def _entropy(self):
    # Use broadcasting rules to calculate the full broadcast scale.
    scale = self.scale + tf.zeros_like(self.loc)
    return math.log(2.) + 1. + tf.log(scale)

  def _mean(self):
    return self.loc + tf.zeros_like(self.scale)

  def _stddev(self):
    return math.sqrt(2.) * self.scale + tf.zeros_like(self.loc)

  def _median(self):
    return self._mean()

  def _mode(self):
    return self._mean()

  def _z(self, x):
    return (x - self.loc) / self.scale


@kullback_leibler.RegisterKL(Laplace, tf.distributions.Laplace)
@kullback_leibler.RegisterKL(tf.distributions.Laplace, Laplace)
@kullback_leibler.RegisterKL(Laplace, Laplace)
def _kl_laplace_laplace(a, b, name=None):
  """Calculate the batched KL divergence KL(a || b) with a and b Laplace.

  Args:
    a: instance of a Laplace distribution object.
    b: instance of a Laplace distribution object.
    name: (optional) Name to use for created operations.
      default is "kl_laplace_laplace".

  Returns:
    Batchwise KL(a || b)
  """
  with tf.name_scope(name, "kl_laplace_laplace",
                     [a.loc, b.loc, a.scale, b.scale]):
    # Consistent with
    # http://www.mast.queensu.ca/~communications/Papers/gil-msc11.pdf, page 38
    distance = tf.abs(a.loc - b.loc)
    ratio = a.scale / b.scale

    return (-tf.log(ratio) - 1 + distance / b.scale +
            ratio * tf.exp(-distance / a.scale))
