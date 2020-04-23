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
"""GEV bijector."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v2 as tf

from tensorflow_probability.python.bijectors import bijector
from tensorflow_probability.python.internal import assert_util
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import tensor_util


__all__ = [
    'GEVCDF',
]


class GEVCDF(bijector.Bijector):
  """Compute `Y = g(X) = exp(-t(X))`, the GEV CDF,
  where `t(x) = (1 + shape * (x - loc) / scale) ) ^ (-1 / shape)`.

  This bijector maps inputs from `[-inf, inf]` to `[0, 1]`. The inverse of the
  bijector applied to a uniform random variable `X ~ U(0, 1)` gives back a
  random variable with the [Generalized extreme value distribution](
  https://https://en.wikipedia.org/wiki/Generalized_extreme_value_distribution):

  ```none
  Y ~ GEVCDF(loc, scale, shape)
  t(y; loc, scale, shape) = (1 + shape * (y - loc) / scale) ) ^ (-1 / shape)
  pdf(y; loc, scale, shape) = t(y; loc, scale, shape) ^ (1 + shape) * exp(
  - t(y; loc, scale, shape) ) / scale
  ```
  """

  def __init__(self,
               loc=0.,
               scale=1.,
               shape=0.5,
               validate_args=False,
               name='gev_cdf'):
    """Instantiates the `GEVCDF` bijector.

    Args:
      loc: Float-like `Tensor` that is the same dtype and is
        broadcastable with `scale` and `shape`.
        This is `loc` in `Y = exp(-t(X))`
        where `t(x) = (1 + shape * (x - loc) / scale) ) ^ (-1 / shape)`.
      scale: Positive Float-like `Tensor` that is the same dtype and is
        broadcastable with `loc` and `shape`.
        This is `scale` in `Y = exp(-t(X))`
        where `t(x) = (1 + shape * (x - loc) / scale) ) ^ (-1 / shape)`.
      shape: Nonzero float-like `Tensor` that is the same dtype and is
        broadcastable with `loc` and `scale`.
        This is `shape` in `Y = exp(-t(X))`
        where `t(x) = (1 + shape * (x - loc) / scale) ) ^ (-1 / shape)`.
      validate_args: Python `bool` indicating whether arguments should be
        checked for correctness.
      name: Python `str` name given to ops managed by this object.
    """
    parameters = dict(locals())
    with tf.name_scope(name) as name:
      dtype = dtype_util.common_dtype(
          [loc, scale, shape], dtype_hint=tf.float32)
      self._loc = tensor_util.convert_nonref_to_tensor(
          loc, dtype=dtype, name='loc')
      self._scale = tensor_util.convert_nonref_to_tensor(
          scale, dtype=dtype, name='scale')
      self._shape = tensor_util.convert_nonref_to_tensor(
          shape, dtype=dtype, name='shape')
      super(GEVCDF, self).__init__(
          validate_args=validate_args,
          forward_min_event_ndims=0,
          parameters=parameters,
          name=name)

  @property
  def loc(self):
    """
    The `loc` in `Y = exp(-t(X))`
    where `t(x) = (1 + shape * (x - loc) / scale) ) ^ (-1 / shape)`.
    """
    return self._loc

  @property
  def scale(self):
    """
    The `scale` in `Y = exp(-t(X))`
    where `t(x) = (1 + shape * (x - loc) / scale) ) ^ (-1 / shape)`.
    """
    return self._scale

  @property
  def shape(self):
    """
    The `shape` in `Y = exp(-t(X))`
    where `t(x) = (1 + shape * (x - loc) / scale) ) ^ (-1 / shape)`.
    """
    return self._shape

  @classmethod
  def _is_increasing(cls):
    return True

  def _forward(self, x):
    with tf.control_dependencies(self._maybe_assert_valid_x(x)):
      z = (x - self.loc) / self.scale
      t = (tf.ones_like(z) + self.shape * z) ** (-1 / self.shape)
      return tf.exp(-t)

  def _inverse(self, y):
    with tf.control_dependencies(self._maybe_assert_valid_y(y)):
      t = - tf.math.log(y)
      z = tf.ones_like(y) - t ** (-self.shape)
      return self.loc - self.scale * z / self.shape

  def _inverse_log_det_jacobian(self, y):
    with tf.control_dependencies(self._maybe_assert_valid_y(y)):
      dz = - self.scale / self.shape
      t = - tf.math.log(y)
      dt = dz * self.shape * (t ** (-self.shape - 1))
      return tf.math.log(-dt / y)

  def _forward_log_det_jacobian(self, x):
    with tf.control_dependencies(self._maybe_assert_valid_x(x)):
      scale = tf.convert_to_tensor(self.scale)
      z = (x - self.loc) / self.scale
      t = (tf.ones_like(z) + self.shape * z) ** (-1 / self.shape)
      return (self.shape + 1) * tf.math.log(t) - t - tf.math.log(scale)

  def _maybe_assert_valid_x(self, x):
    if not self.validate_args:
      return []
    return [assert_util.assert_less_equal(
        self.shape * self.loc - self.scale,
        x * self.shape,
        message='Forward transformation input must be inside domain.')]

  def _maybe_assert_valid_y(self, y):
    if not self.validate_args:
      return []
    is_positive = assert_util.assert_non_negative(
        y, message='Inverse transformation input must be greater than 0.')
    less_than_one = assert_util.assert_less_equal(
        y,
        tf.constant(1., y.dtype),
        message='Inverse transformation input must be less than or equal to 1.')
    return [is_positive, less_than_one]

  def _parameter_control_dependencies(self, is_init):
    if not self.validate_args:
      return []
    assertions = []
    if is_init != tensor_util.is_ref(self.scale):
      assertions.append(assert_util.assert_positive(
          self.scale,
          message='Argument `scale` must be positive.'))
    if is_init != tensor_util.is_ref(self.shape):
      assertions.append(assert_util.assert_none_equal(
          self.shape,
          tf.constant(0, self.shape.dtype),
          message='Argument `shape` must be nonzero. Use Gumbel_cdf instead.'))
    return assertions
