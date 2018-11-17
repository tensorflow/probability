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
"""ExpSinSquared kernel."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.positive_semidefinite_kernels import positive_semidefinite_kernel as psd_kernel
from tensorflow_probability.python.positive_semidefinite_kernels.internal import util

__all__ = [
    'ExpSinSquared',
]


def _validate_arg_if_not_none(arg, assertion, validate_args):
  if arg is None:
    return arg
  with tf.control_dependencies([assertion(arg)] if validate_args else []):
    result = tf.identity(arg)
  return result


class ExpSinSquared(psd_kernel.PositiveSemidefiniteKernel):
  """Exponentiated Sine Squared Kernel.

    Also known as the "Periodic" kernel, this kernel, when
    parameterizing a Gaussian Process, results in random functions that are
    periodic almost everywhere (with the same period for every dimension).

    ```none
    k(x, y) = amplitude**2 * exp(
      -2  / length_scale ** 2 * sum_k sin(pi * |x_k - y_k| / period) ** 2)
    ```

    This kernel acts over the space `S = R^(D1 x D2 x D3 ... Dd)`.
  """

  def __init__(self,
               amplitude=None,
               length_scale=None,
               period=None,
               feature_ndims=1,
               validate_args=False,
               name='ExpSinSquared'):
    """Construct a ExpSinSquared kernel instance.

    Args:
      amplitude: Positive floating point `Tensor` that controls the maximum
        value of the kernel. Must be broadcastable with `period`, `length_scale`
        and inputs to `apply` and `matrix` methods. A value of `None` is treated
        like 1.
      length_scale: Positive floating point `Tensor` that controls how sharp or
        wide the kernel shape is. This provides a characteristic "unit" of
        length against which `|x - y|` can be compared for scale. Must be
        broadcastable with `amplitude`, `period`  and inputs to `apply` and
        `matrix` methods. A value of `None` is treated like 1.
      period: Positive floating point `Tensor` that controls the period of the
        kernel. Must be broadcastable with `amplitude`, `length_scale` and
        inputs to `apply` and `matrix` methods.  A value of `None` is treated
        like 1.
      feature_ndims: Python `int` number of rightmost dims to include in kernel
        computation.
      validate_args: If `True`, parameters are checked for validity despite
        possibly degrading runtime performance
      name: Python `str` name prefixed to Ops created by this class.
    """
    with tf.name_scope(name, values=[amplitude, period, length_scale]) as name:
      dtype = dtype_util.common_dtype([amplitude, period, length_scale],
                                      tf.float32)
      if amplitude is not None:
        amplitude = tf.convert_to_tensor(
            amplitude, name='amplitude', dtype=dtype)
      self._amplitude = _validate_arg_if_not_none(
          amplitude, tf.assert_positive, validate_args)
      if period is not None:
        period = tf.convert_to_tensor(period, name='period', dtype=dtype)
      self._period = _validate_arg_if_not_none(
          period, tf.assert_positive, validate_args)
      if length_scale is not None:
        length_scale = tf.convert_to_tensor(
            length_scale, name='length_scale', dtype=dtype)
      self._length_scale = _validate_arg_if_not_none(
          length_scale, tf.assert_positive, validate_args)
      tf.assert_same_float_dtype(
          [self._amplitude, self._length_scale, self._period])
    super(ExpSinSquared, self).__init__(feature_ndims, dtype=dtype, name=name)

  def _apply(self, x1, x2, param_expansion_ndims=0):
    difference = np.pi * tf.abs(x1 - x2)

    if self.period is not None:
      # period acts as a batch of periods, and hence we must additionally
      # pad the shape with self.feature_ndims number of ones.
      period = util.pad_shape_right_with_ones(
          self.period, ndims=(param_expansion_ndims + self.feature_ndims))
      difference /= period
    log_kernel = util.sum_rightmost_ndims_preserving_shape(
        -2 * tf.sin(difference) ** 2, ndims=self.feature_ndims)

    if self.length_scale is not None:
      length_scale = util.pad_shape_right_with_ones(
          self.length_scale, ndims=param_expansion_ndims)
      log_kernel /= length_scale ** 2

    if self.amplitude is not None:
      amplitude = util.pad_shape_right_with_ones(
          self.amplitude, ndims=param_expansion_ndims)
      log_kernel += 2. * tf.log(amplitude)
    return tf.exp(log_kernel)

  @property
  def amplitude(self):
    """Amplitude parameter."""
    return self._amplitude

  @property
  def length_scale(self):
    """Length scale parameter."""
    return self._length_scale

  @property
  def period(self):
    """Period parameter."""
    return self._period

  def _batch_shape(self):
    scalar_shape = tf.TensorShape([])
    return tf.broadcast_static_shape(
        tf.broadcast_static_shape(
            scalar_shape if self.amplitude is None else self.amplitude.shape,
            scalar_shape if self.period is None else self.period.shape),
        scalar_shape if self.length_scale is None else self.length_scale.shape)

  def _batch_shape_tensor(self):
    return tf.broadcast_dynamic_shape(
        tf.broadcast_dynamic_shape(
            [] if self.amplitude is None else tf.shape(self.amplitude),
            [] if self.length_scale is None else tf.shape(self.length_scale)),
        [] if self.period is None else tf.shape(self.period))
