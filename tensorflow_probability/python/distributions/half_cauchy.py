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
"""Half-Cauchy Distribution Class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
import numpy as np
import tensorflow.compat.v2 as tf

from tensorflow_probability.python.bijectors import chain as chain_bijector
from tensorflow_probability.python.bijectors import exp as exp_bijector
from tensorflow_probability.python.bijectors import shift as shift_bijector
from tensorflow_probability.python.distributions import distribution
from tensorflow_probability.python.internal import assert_util
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import reparameterization
from tensorflow_probability.python.internal import samplers
from tensorflow_probability.python.internal import tensor_util

__all__ = [
    'HalfCauchy',
]


class HalfCauchy(distribution.Distribution):
  """Half-Cauchy distribution.

  The half-Cauchy distribution is parameterized by a `loc` and a
  `scale` parameter. It represents the right half of the two symmetric halves in
  a [Cauchy distribution](https://en.wikipedia.org/wiki/Cauchy_distribution).

  #### Mathematical Details
  The probability density function (pdf) for the half-Cauchy distribution
  is given by

  ```none
  pdf(x; loc, scale) = 2 / (pi scale (1 + z**2))
  z = (x - loc) / scale
  ```

  where `loc` is a scalar in `R` and `scale` is a positive scalar in `R`.

  The support of the distribution is given by the interval `[loc, infinity)`.

  """

  def __init__(self,
               loc,
               scale,
               validate_args=False,
               allow_nan_stats=True,
               name='HalfCauchy'):
    """Construct a half-Cauchy distribution with `loc` and `scale`.

    Args:
      loc: Floating-point `Tensor`; the location(s) of the distribution(s).
      scale: Floating-point `Tensor`; the scale(s) of the distribution(s).
        Must contain only positive values.
      validate_args: Python `bool`, default `False`. When `True` distribution
        parameters are checked for validity despite possibly degrading runtime
        performance. When `False` invalid inputs may silently render incorrect
        outputs. Default value: `False` (i.e. do not validate args).
      allow_nan_stats: Python `bool`, default `True`. When `True`, statistics
        (e.g., mean, mode, variance) use the value "`NaN`" to indicate the
        result is undefined. When `False`, an exception is raised if one or
        more of the statistic's batch members are undefined.
        Default value: `True`.
      name: Python `str` name prefixed to Ops created by this class.
        Default value: 'HalfCauchy'.

    Raises:
      TypeError: if `loc` and `scale` have different `dtype`.
    """
    parameters = dict(locals())
    with tf.name_scope(name) as name:
      dtype = dtype_util.common_dtype([loc, scale], dtype_hint=tf.float32)
      self._loc = tensor_util.convert_nonref_to_tensor(
          loc, name='loc', dtype=dtype)
      self._scale = tensor_util.convert_nonref_to_tensor(
          scale, name='scale', dtype=dtype)
      super(HalfCauchy, self).__init__(
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
    """Distribution parameter for the scale."""
    return self._scale

  def _batch_shape_tensor(self, loc=None, scale=None):
    return tf.broadcast_dynamic_shape(
        tf.shape(self.loc if loc is None else loc),
        tf.shape(self.scale if scale is None else scale))

  def _batch_shape(self):
    return tf.broadcast_static_shape(self.loc.shape, self.scale.shape)

  def _event_shape_tensor(self):
    return tf.constant([], dtype=tf.int32)

  def _event_shape(self):
    return tf.TensorShape([])

  def _sample_n(self, n, seed=None):
    loc = tf.convert_to_tensor(self.loc)
    scale = tf.convert_to_tensor(self.scale)
    shape = tf.concat([[n], self._batch_shape_tensor(
        loc=loc, scale=scale)], 0)
    probs = samplers.uniform(
        shape, minval=0., maxval=1., dtype=self.dtype, seed=seed)
    # Quantile function.
    return loc + scale * tf.tan((np.pi / 2) * probs)

  def _log_prob(self, x):
    loc = tf.convert_to_tensor(self.loc)
    scale = tf.convert_to_tensor(self.scale)
    safe_x = self._get_safe_input(x, loc=loc, scale=scale)
    log_prob = (np.log(2 / np.pi) - tf.math.log(scale) - tf.math.log1p(
        ((safe_x - loc) / scale)**2))
    return tf.where(x < loc, dtype_util.as_numpy_dtype(
        self.dtype)(-np.inf), log_prob)

  def _log_cdf(self, x):
    loc = tf.convert_to_tensor(self.loc)
    scale = tf.convert_to_tensor(self.scale)
    safe_x = self._get_safe_input(x, loc=loc, scale=scale)
    log_cdf = np.log(2 / np.pi) + tf.math.log(tf.atan((safe_x - loc) / scale))
    return tf.where(x < loc, dtype_util.as_numpy_dtype(
        self.dtype)(-np.inf), log_cdf)

  def _entropy(self):
    h = np.log(2 * np.pi) + tf.math.log(self.scale)
    return h * tf.ones_like(self.loc)

  def _quantile(self, p):
    return self.loc + self.scale * tf.tan((np.pi / 2) * p)

  def _mode(self):
    return self.loc * tf.ones_like(self.scale)

  def _mean(self):
    if self.allow_nan_stats:
      return tf.fill(self.batch_shape_tensor(),
                     dtype_util.as_numpy_dtype(self.dtype)(np.nan))
    raise ValueError('`mean` is undefined for the half-Cauchy distribution.')

  def _stddev(self):
    if self.allow_nan_stats:
      return tf.fill(self.batch_shape_tensor(),
                     dtype_util.as_numpy_dtype(self.dtype)(np.nan))
    raise ValueError('`stddev` is undefined for the half-Cauchy distribution.')

  def _variance(self):
    if self.allow_nan_stats:
      return tf.fill(self.batch_shape_tensor(),
                     dtype_util.as_numpy_dtype(self.dtype)(np.nan))
    raise ValueError(
        '`variance` is undefined for the half-Cauchy distribution.')

  def _get_safe_input(self, x, loc, scale):
    safe_value = 0.5 * scale + loc
    return tf.where(x < loc, safe_value, x)

  def _default_event_space_bijector(self):
    return chain_bijector.Chain([
        shift_bijector.Shift(
            shift=self.loc, validate_args=self.validate_args),
        exp_bijector.Exp(validate_args=self.validate_args)
    ], validate_args=self.validate_args)

  def _sample_control_dependencies(self, x):
    """Checks the validity of a sample."""
    assertions = []
    if not self.validate_args:
      return assertions
    loc = tf.convert_to_tensor(self.loc)
    assertions.append(assert_util.assert_greater_equal(
        x, loc, message='Sample must be greater than or equal to `loc`.'))
    return assertions

  def _parameter_control_dependencies(self, is_init):
    if not self.validate_args:
      return []
    assertions = []
    if is_init != tensor_util.is_ref(self.scale):
      assertions.append(assert_util.assert_positive(
          self.scale,
          message='Argument `scale` must be positive.'))
    return assertions

