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

# Dependency imports
import numpy as np
import tensorflow.compat.v2 as tf

from tensorflow_probability.python.distributions import distribution
from tensorflow_probability.python.distributions import kullback_leibler
from tensorflow_probability.python.internal import assert_util
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import prefer_static
from tensorflow_probability.python.internal import reparameterization
from tensorflow_probability.python.internal import special_math
from tensorflow_probability.python.internal import tensor_util


__all__ = [
    'Laplace',
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
  distributions spliced together 'back-to-back.'

  The Laplace distribution is a member of the [location-scale family](
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
               name='Laplace'):
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
        statistics (e.g., mean, mode, variance) use the value '`NaN`' to
        indicate the result is undefined. When `False`, an exception is raised
        if one or more of the statistic's batch members are undefined.
      name: Python `str` name prefixed to Ops created by this class.

    Raises:
      TypeError: if `loc` and `scale` are of different dtype.
    """
    parameters = dict(locals())
    with tf.name_scope(name) as name:
      dtype = dtype_util.common_dtype([loc, scale], tf.float32)
      self._loc = tensor_util.convert_immutable_to_tensor(
          loc, name='loc', dtype=dtype)
      self._scale = tensor_util.convert_immutable_to_tensor(
          scale, name='scale', dtype=dtype)
      dtype_util.assert_same_float_dtype([self._loc, self._scale])
      super(Laplace, self).__init__(
          dtype=dtype,
          reparameterization_type=reparameterization.FULLY_REPARAMETERIZED,
          validate_args=validate_args,
          allow_nan_stats=allow_nan_stats,
          parameters=parameters,
          name=name)

  @staticmethod
  def _param_shapes(sample_shape):
    return dict(
        zip(('loc', 'scale'),
            ([tf.convert_to_tensor(sample_shape, dtype=tf.int32)] * 2)))

  @classmethod
  def _params_event_ndims(cls):
    return dict(loc=0, scale=0)

  @property
  def loc(self):
    """Distribution parameter for the location."""
    return self._loc

  @property
  def scale(self):
    """Distribution parameter for scale."""
    return self._scale

  def _batch_shape_tensor(self, loc=None, scale=None):
    return prefer_static.broadcast_shape(
        prefer_static.shape(self.loc if loc is None else loc),
        prefer_static.shape(self.scale if scale is None else scale))

  def _batch_shape(self):
    return tf.broadcast_static_shape(
        self.loc.shape, self.scale.shape)

  def _event_shape_tensor(self):
    return tf.constant([], dtype=tf.int32)

  def _event_shape(self):
    return tf.TensorShape([])

  def _sample_n(self, n, seed=None):
    loc = tf.convert_to_tensor(self.loc)
    scale = tf.convert_to_tensor(self.scale)
    shape = tf.concat([[n], self._batch_shape_tensor(loc=loc, scale=scale)], 0)
    # Uniform variates must be sampled from the open-interval `(-1, 1)` rather
    # than `[-1, 1)`. In the case of `(0, 1)` we'd use
    # `np.finfo(dtype_util.as_numpy_dtype(self.dtype)).tiny` because it is the
    # smallest, positive, 'normal' number. However, the concept of subnormality
    # exists only at zero; here we need the smallest usable number larger than
    # -1, i.e., `-1 + eps/2`.
    dt = dtype_util.as_numpy_dtype(self.dtype)
    uniform_samples = tf.random.uniform(
        shape=shape,
        minval=np.nextafter(dt(-1.), dt(1.)),
        maxval=1.,
        dtype=self.dtype,
        seed=seed)
    return (loc - scale * tf.sign(uniform_samples) *
            tf.math.log1p(-tf.abs(uniform_samples)))

  def _log_prob(self, x):
    loc = tf.convert_to_tensor(self.loc)
    scale = tf.convert_to_tensor(self.scale)
    z = (x - loc) / scale
    return -tf.abs(z) - np.log(2.) - tf.math.log(scale)

  def _log_cdf(self, x):
    return special_math.log_cdf_laplace(self._z(x))

  def _log_survival_function(self, x):
    return special_math.log_cdf_laplace(-self._z(x))

  def _cdf(self, x):
    z = self._z(x)
    return 0.5 - 0.5 * tf.sign(z) * tf.math.expm1(-tf.abs(z))

  def _entropy(self):
    scale = tf.convert_to_tensor(self.scale)
    return tf.broadcast_to(np.log(2.) + 1 + tf.math.log(scale),
                           self._batch_shape_tensor(scale=scale))

  def _mean(self):
    loc = tf.convert_to_tensor(self.loc)
    return tf.broadcast_to(loc, self._batch_shape_tensor(loc=loc))

  def _stddev(self):
    scale = tf.convert_to_tensor(self.scale)
    return tf.broadcast_to(np.sqrt(2.) * scale,
                           self._batch_shape_tensor(scale=scale))

  def _median(self):
    return self._mean()

  def _mode(self):
    return self._mean()

  def _z(self, x):
    return (x - self.loc) / self.scale

  def _parameter_control_dependencies(self, is_init):
    if not self.validate_args:
      return []
    assertions = []
    if is_init != tensor_util.is_mutable(self._scale):
      assertions.append(assert_util.assert_positive(
          self._scale, message='Argument `scale` must be positive.'))
    return assertions


@kullback_leibler.RegisterKL(Laplace, Laplace)
def _kl_laplace_laplace(a, b, name=None):
  """Calculate the batched KL divergence KL(a || b) with a and b Laplace.

  Args:
    a: instance of a Laplace distribution object.
    b: instance of a Laplace distribution object.
    name: Python `str` name to use for created operations.
      Default value: `None` (i.e., `'kl_laplace_laplace'`).

  Returns:
    kl_div: Batchwise KL(a || b)
  """
  with tf.name_scope(name or 'kl_laplace_laplace'):
    # Consistent with
    # http://www.mast.queensu.ca/~communications/Papers/gil-msc11.pdf, page 38
    distance = tf.abs(a.loc - b.loc)
    a_scale = tf.convert_to_tensor(a.scale)
    b_scale = tf.convert_to_tensor(b.scale)
    delta_log_scale = tf.math.log(a_scale) - tf.math.log(b_scale)
    return (-delta_log_scale +
            distance / b_scale - 1. +
            tf.exp(-distance / a_scale + delta_log_scale))
