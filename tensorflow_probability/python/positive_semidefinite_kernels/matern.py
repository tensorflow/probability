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
"""Matern kernels."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow_probability.python.positive_semidefinite_kernels import positive_semidefinite_kernel as psd_kernel
from tensorflow_probability.python.positive_semidefinite_kernels import util

__all__ = [
    "MaternOneHalf",
    "MaternThreeHalves",
    "MaternFiveHalves",
]


def _validate_arg_if_not_none(arg, assertion, validate_args):
  if arg is None:
    return arg
  with tf.control_dependencies([assertion(arg)] if validate_args else []):
    result = tf.identity(arg)
  return result


class MaternOneHalf(psd_kernel.PositiveSemidefiniteKernel):
  """Matern Kernel with parameter 1/2.

  This kernel is part of the Matern family of kernels, with parameter 1/2.
  Also known as the Exponential or Laplacian Kernel, a Gaussian process
  parameterized by this kernel, is also known as an Ornstein-Uhlenbeck process
  (https://en.wikipedia.org/wiki/Ornstein%E2%80%93Uhlenbeck_process).

  The kernel has the following form:

  ```none
    k(x, y) = amplitude**2  * exp(-||x - y|| / length_scale)
  ```

  where the double-bars represent vector length (i.e. Euclidean, or L2 Norm).
  This kernel, acts over the space `S = R^(D1 x D2 .. x Dd)`.
  """

  def __init__(self,
               amplitude=None,
               length_scale=None,
               feature_ndims=1,
               validate_args=False,
               name="MaternOneHalf"):
    """Construct a MaternOneHalf kernel instance.

    Args:
      amplitude: Positive floating point `Tensor` that controls the maximum
        value of the kernel. Must be broadcastable with `length_scale` and
        inputs to `apply` and `matrix` methods. A value of `None` is treated
        like 1.
      length_scale: Positive floating point `Tensor` that controls how sharp
        or wide the kernel shape is. This provides a characteristic "unit" of
        length against which `||x - y||` can be compared for scale. Must be
        broadcastable with `amplitude` and inputs to `apply` and `matrix`
        methods. A value of `None` is treated like 1.
      feature_ndims: Python `int` number of rightmost dims to include in the
        squared difference norm in the exponential.
      validate_args: If `True`, parameters are checked for validity despite
        possibly degrading runtime performance
      name: Python `str` name prefixed to Ops created by this class.
    """
    with tf.name_scope(name, values=[amplitude, length_scale]) as name:
      self._amplitude = _validate_arg_if_not_none(amplitude, tf.assert_positive,
                                                  validate_args)
      self._length_scale = _validate_arg_if_not_none(
          length_scale, tf.assert_positive, validate_args)
      tf.assert_same_float_dtype([self._amplitude, self._length_scale])
    super(MaternOneHalf, self).__init__(feature_ndims, name)

  @property
  def amplitude(self):
    """Amplitude parameter."""
    return self._amplitude

  @property
  def length_scale(self):
    """Length scale parameter."""
    return self._length_scale

  def _apply(self, x1, x2, param_expansion_ndims=0):
    norm = tf.sqrt(
        util.sum_rightmost_ndims_preserving_shape(
            tf.squared_difference(x1, x2), self.feature_ndims))
    if self.length_scale is not None:
      length_scale = util.pad_shape_right_with_ones(
          self.length_scale, ndims=param_expansion_ndims)
      norm /= length_scale
    result = tf.exp(-norm)

    if self.amplitude is not None:
      amplitude = util.pad_shape_right_with_ones(
          self.amplitude, ndims=param_expansion_ndims)
      result *= amplitude**2
    return result

  def _batch_shape(self):
    return tf.broadcast_static_shape(
        self.amplitude.shape, self.length_scale.shape)

  def _batch_shape_tensor(self):
    with self._name_scope("batch_shape_tensor"):
      return tf.broadcast_dynamic_shape(
          tf.shape(self.amplitude), tf.shape(self.length_scale))


class MaternThreeHalves(psd_kernel.PositiveSemidefiniteKernel):
  """Matern Kernel with parameter 3/2.

  This kernel is part of the Matern family of kernels, with parameter 3/2.

  ```none
    z = sqrt(3) * ||x - y|| / length_scale
    k(x, y) = (1 + z) * exp(-z)
  ```

  where the double-bars represent vector length (i.e. Euclidean, or L2 Norm).
  This kernel, acts over the space `S = R^(D1 x D2 .. x Dd)`.
  """

  def __init__(self,
               amplitude=None,
               length_scale=None,
               feature_ndims=1,
               validate_args=False,
               name="MaternThreeHalves"):
    """Construct a MaternThreeHalves kernel instance.

    Args:
      amplitude: Positive floating point `Tensor` that controls the maximum
        value of the kernel. Must be broadcastable with `length_scale` and
        inputs to `apply` and `matrix` methods. A value of `None` is treated
        like 1.
      length_scale: Positive floating point `Tensor` that controls how sharp
        or wide the kernel shape is. This provides a characteristic "unit" of
        length against which `||x - y||` can be compared for scale. Must be
        broadcastable with `amplitude` and inputs to `apply` and `matrix`
        methods. A value of `None` is treated like 1.
      feature_ndims: Python `int` number of rightmost dims to include in the
        squared difference norm in the exponential.
      validate_args: If `True`, parameters are checked for validity despite
        possibly degrading runtime performance
      name: Python `str` name prefixed to Ops created by this class.
    """
    with tf.name_scope(name, values=[amplitude, length_scale]) as name:
      self._amplitude = _validate_arg_if_not_none(amplitude, tf.assert_positive,
                                                  validate_args)
      self._length_scale = _validate_arg_if_not_none(
          length_scale, tf.assert_positive, validate_args)
      tf.assert_same_float_dtype([self._amplitude, self._length_scale])
    super(MaternThreeHalves, self).__init__(feature_ndims, name)

  @property
  def amplitude(self):
    """Amplitude parameter."""
    return self._amplitude

  @property
  def length_scale(self):
    """Length scale parameter."""
    return self._length_scale

  def _apply(self, x1, x2, param_expansion_ndims=0):
    norm = tf.sqrt(
        util.sum_rightmost_ndims_preserving_shape(
            tf.squared_difference(x1, x2), self.feature_ndims))
    if self.length_scale is not None:
      length_scale = util.pad_shape_right_with_ones(
          self.length_scale, ndims=param_expansion_ndims)
      norm /= length_scale
    series_term = np.sqrt(3) * norm
    result = (1. + series_term) * tf.exp(-series_term)

    if self.amplitude is not None:
      amplitude = util.pad_shape_right_with_ones(self.amplitude,
                                                 param_expansion_ndims)
      result *= amplitude**2
    return result

  def _batch_shape(self):
    return tf.broadcast_static_shape(
        self.amplitude.shape, self.length_scale.shape)

  def _batch_shape_tensor(self):
    with self._name_scope("batch_shape_tensor"):
      return tf.broadcast_dynamic_shape(
          tf.shape(self.amplitude), tf.shape(self.length_scale))


class MaternFiveHalves(psd_kernel.PositiveSemidefiniteKernel):
  """Matern 5/2 Kernel.

  This kernel is part of the Matern family of kernels, with parameter 5/2.

  ```none
    z = sqrt(5) * ||x - y|| / length_scale
    k(x, y) = (1 + z + (z ** 2) / 3) * exp(-z)
  ```

  where the double-bars represent vector length (i.e. Euclidean, or L2 Norm).
  This kernel, acts over the space `S = R^(D1 x D2 .. x Dd)`.
  """

  def __init__(self,
               amplitude=None,
               length_scale=None,
               feature_ndims=1,
               validate_args=False,
               name="MaternFiveHalves"):
    """Construct a MaternFiveHalves kernel instance.

    Args:
      amplitude: Positive floating point `Tensor` that controls the maximum
        value of the kernel. Must be broadcastable with `length_scale` and
        inputs to `apply` and `matrix` methods. A value of `None` is treated
        like 1.
      length_scale: Positive floating point `Tensor` that controls how sharp
        or wide the kernel shape is. This provides a characteristic "unit" of
        length against which `||x - y||` can be compared for scale. Must be
        broadcastable with `amplitude`, and inputs to `apply` and `matrix`
        methods. A value of `None` is treated like 1.
      feature_ndims: Python `int` number of rightmost dims to include in the
        squared difference norm in the exponential.
      validate_args: If `True`, parameters are checked for validity despite
        possibly degrading runtime performance
      name: Python `str` name prefixed to Ops created by this class.
    """
    with tf.name_scope(name, values=[amplitude, length_scale]) as name:
      self._amplitude = _validate_arg_if_not_none(amplitude, tf.assert_positive,
                                                  validate_args)
      self._length_scale = _validate_arg_if_not_none(
          length_scale, tf.assert_positive, validate_args)
      tf.assert_same_float_dtype([self._amplitude, self._length_scale])
    super(MaternFiveHalves, self).__init__(feature_ndims, name)

  @property
  def amplitude(self):
    """Amplitude parameter."""
    return self._amplitude

  @property
  def length_scale(self):
    """Length scale parameter."""
    return self._length_scale

  def _apply(self, x1, x2, param_expansion_ndims=0):
    norm = tf.sqrt(
        util.sum_rightmost_ndims_preserving_shape(
            tf.squared_difference(x1, x2), self.feature_ndims))
    if self.length_scale is not None:
      length_scale = util.pad_shape_right_with_ones(
          self.length_scale, ndims=param_expansion_ndims)
      norm /= length_scale
    series_term = np.sqrt(5) * norm
    result = (1. + series_term + series_term**2 / 3.) * tf.exp(-series_term)

    if self.amplitude is not None:
      amplitude = util.pad_shape_right_with_ones(self.amplitude,
                                                 param_expansion_ndims)
      result *= amplitude**2
    return result

  def _batch_shape(self):
    return tf.broadcast_static_shape(
        self.amplitude.shape, self.length_scale.shape)

  def _batch_shape_tensor(self):
    with self._name_scope("batch_shape_tensor"):
      return tf.broadcast_dynamic_shape(
          tf.shape(self.amplitude), tf.shape(self.length_scale))
