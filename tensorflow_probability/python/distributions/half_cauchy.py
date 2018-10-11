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

import functools
# Dependency imports
import numpy as np
import tensorflow as tf

from tensorflow_probability.python.distributions import distribution
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import reparameterization
from tensorflow.python.framework import tensor_shape

__all__ = [
    "HalfCauchy",
]


def check_arg_in_support(f):
  """Decorator function for argument bounds checking.

  This decorator is meant to be used with methods that require the first
  argument to be in the support of the distribution. If `validate_args` is
  `True`, the method is wrapped with an assertion that the first argument is
  greater than or equal to `loc`, since the support of the half-Cauchy
  distribution is given by `[loc, infinity)`.


  Args:
    f: method to be decorated.

  Returns:
    Returns a decorated method that, when `validate_args` attribute of the class
    is `True`, will assert that all elements in the first argument are within
    the support of the distribution before executing the original method.
  """
  @functools.wraps(f)
  def _check_arg_and_apply_f(*args, **kwargs):
    dist = args[0]
    x = args[1]
    with tf.control_dependencies([
        tf.assert_greater_equal(
            x, dist.loc,
            message="x is not in the support of the distribution"
        )] if dist.validate_args else []):
      return f(*args, **kwargs)
  return _check_arg_and_apply_f


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
               name="HalfCauchy"):
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
    with tf.name_scope(name, values=[loc, scale]) as name:
      dtype = dtype_util.common_dtype([loc, scale], preferred_dtype=tf.float32)
      loc = tf.convert_to_tensor(loc, name="loc", dtype=dtype)
      scale = tf.convert_to_tensor(scale, name="scale", dtype=dtype)
      with tf.control_dependencies([tf.assert_positive(scale)]
                                   if validate_args else []):
        self._loc = tf.identity(loc, name="loc")
        self._scale = tf.identity(scale, name="loc")
      tf.assert_same_float_dtype([self._loc, self._scale])
    super(HalfCauchy, self).__init__(
        dtype=self._scale.dtype,
        reparameterization_type=reparameterization.FULLY_REPARAMETERIZED,
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
    """Distribution parameter for the scale."""
    return self._scale

  def _batch_shape_tensor(self):
    return tf.broadcast_dynamic_shape(tf.shape(self.loc), tf.shape(self.scale))

  def _batch_shape(self):
    return tf.broadcast_static_shape(self.loc.shape, self.scale.shape)

  def _event_shape_tensor(self):
    return tf.constant([], dtype=tf.int32)

  def _event_shape(self):
    return tensor_shape.scalar()

  def _sample_n(self, n, seed=None):
    shape = tf.concat([[n], self._batch_shape_tensor()], 0)
    probs = tf.random_uniform(
        shape, minval=0., maxval=1., dtype=self.dtype, seed=seed)
    return self._quantile(probs)

  @check_arg_in_support
  def _log_prob(self, x):
    def log_prob_on_support(x):
      return (np.log(2 / np.pi) - tf.log(self.scale) -
              tf.log1p(self._z(x) ** 2))
    return self._extend_support_with_default_value(
        x, log_prob_on_support, default_value=-np.inf)

  @check_arg_in_support
  def _log_cdf(self, x):
    return self._extend_support_with_default_value(
        x,
        lambda x: np.log(2 / np.pi) + tf.log(tf.atan(self._z(x))),
        default_value=-np.inf)

  def _z(self, x):
    """Standardize input `x`."""
    with tf.name_scope("standardize", values=[x]):
      return (x - self.loc) / self.scale

  def _inv_z(self, z):
    """Reconstruct input `x` from a its normalized version."""
    with tf.name_scope("reconstruct", values=[z]):
      return z * self.scale + self.loc

  def _entropy(self):
    h = np.log(2 * np.pi) + tf.log(self.scale)
    return h * tf.ones_like(self.loc)

  def _quantile(self, p):
    return self.loc + self.scale * tf.tan((np.pi / 2) * p)

  def _mode(self):
    return self.loc * tf.ones_like(self.scale)

  def _mean(self):
    if self.allow_nan_stats:
      return tf.fill(self.batch_shape_tensor(),
                     self.dtype.as_numpy_dtype(np.nan))
    raise ValueError("`mean` is undefined for the half-Cauchy distribution.")

  def _stddev(self):
    if self.allow_nan_stats:
      return tf.fill(self.batch_shape_tensor(),
                     self.dtype.as_numpy_dtype(np.nan))
    raise ValueError("`stddev` is undefined for the half-Cauchy distribution.")

  def _variance(self):
    if self.allow_nan_stats:
      return tf.fill(self.batch_shape_tensor(),
                     self.dtype.as_numpy_dtype(np.nan))
    raise ValueError(
        "`variance` is undefined for the half-Cauchy distribution.")

  def _extend_support_with_default_value(self, x, f, default_value):
    """Returns `f(x)` if x is in the support, and `default_value` otherwise.

    Given `f` which is defined on the support of this distribution
    (`x >= loc`), extend the function definition to the real line
    by defining `f(x) = default_value` for `x < loc`.

    Args:
      x: Floating-point `Tensor` to evaluate `f` at.
      f: Callable that takes in a `Tensor` and returns a `Tensor`. This
        represents the function whose domain of definition we want to extend.
      default_value: Python or numpy literal representing the value to use for
        extending the domain.
    Returns:
      `Tensor` representing an extension of `f(x)`.
    """
    with tf.name_scope(name="extend_support_with_default_value", values=[x]):
      x = tf.convert_to_tensor(x, dtype=self.dtype, name="x")
      loc = self.loc + tf.zeros_like(self.scale) + tf.zeros_like(x)
      x = x + tf.zeros_like(loc)
      # Substitute out-of-support values in x with values that are in the
      # support of the distribution before applying f.
      y = f(tf.where(x < loc, self._inv_z(0.5), x))
      if default_value == 0.:
        default_value = tf.zeros_like(y)
      elif default_value == 1.:
        default_value = tf.ones_like(y)
      else:
        default_value = tf.fill(
            dims=tf.shape(y),
            value=np.array(default_value, dtype=self.dtype.as_numpy_dtype))
      return tf.where(x < loc, default_value, y)
