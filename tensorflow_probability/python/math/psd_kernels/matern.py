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
import tensorflow.compat.v2 as tf

from tensorflow_probability.python.internal import assert_util
from tensorflow_probability.python.internal import tensor_util
from tensorflow_probability.python.math.psd_kernels.internal import util
from tensorflow_probability.python.math.psd_kernels.positive_semidefinite_kernel import PositiveSemidefiniteKernel

__all__ = [
    'MaternOneHalf',
    'MaternThreeHalves',
    'MaternFiveHalves',
]


class _AmplitudeLengthScaleMixin(object):
  """Shared logic for amplitude/length_scale parameterized kernels."""

  def _init_params(self, amplitude, length_scale):
    """Shared init logic for `amplitude` and `length_scale` params.

    Args:
      amplitude: `Tensor` (or convertible) or `None` to convert, validate.
      length_scale: `Tensor` (or convertible) or `None` to convert, validate.

    Returns:
      dtype: The common `DType` of the parameters.
    """
    dtype = util.maybe_get_common_dtype(
        [amplitude, length_scale])
    self._amplitude = tensor_util.convert_nonref_to_tensor(
        amplitude, name='amplitude', dtype=dtype)
    self._length_scale = tensor_util.convert_nonref_to_tensor(
        length_scale, name='length_scale', dtype=dtype)
    return dtype

  @property
  def amplitude(self):
    """Amplitude parameter."""
    return self._amplitude

  @property
  def length_scale(self):
    """Length scale parameter."""
    return self._length_scale

  def _batch_shape(self):
    scalar_shape = tf.TensorShape([])
    return tf.broadcast_static_shape(
        scalar_shape if self.amplitude is None else self.amplitude.shape,
        scalar_shape if self.length_scale is None else self.length_scale.shape)

  def _batch_shape_tensor(self):
    return tf.broadcast_dynamic_shape(
        [] if self.amplitude is None else tf.shape(self.amplitude),
        [] if self.length_scale is None else tf.shape(self.length_scale))

  def _parameter_control_dependencies(self, is_init):
    """Control dependencies for parameters."""
    if not self.validate_args:
      return []
    assertions = []
    for arg_name, arg in dict(amplitude=self.amplitude,
                              length_scale=self.length_scale).items():
      if arg is not None and is_init != tensor_util.is_ref(arg):
        assertions.append(assert_util.assert_positive(
            arg,
            message='{} must be positive.'.format(arg_name)))
    return assertions


class MaternOneHalf(_AmplitudeLengthScaleMixin, PositiveSemidefiniteKernel):
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
               name='MaternOneHalf'):
    """Construct a MaternOneHalf kernel instance.

    Args:
      amplitude: Positive floating point `Tensor` that controls the maximum
        value of the kernel. Must be broadcastable with `length_scale` and
        inputs to `apply` and `matrix` methods. A value of `None` is treated
        like 1.
      length_scale: Positive floating point `Tensor` that controls how sharp or
        wide the kernel shape is. This provides a characteristic "unit" of
        length against which `||x - y||` can be compared for scale. Must be
        broadcastable with `amplitude` and inputs to `apply` and `matrix`
        methods. A value of `None` is treated like 1.
      feature_ndims: Python `int` number of rightmost dims to include in the
        squared difference norm in the exponential.
      validate_args: If `True`, parameters are checked for validity despite
        possibly degrading runtime performance
      name: Python `str` name prefixed to Ops created by this class.
    """
    parameters = dict(locals())
    with tf.name_scope(name) as name:
      dtype = super(MaternOneHalf, self)._init_params(amplitude, length_scale)
      super(MaternOneHalf, self).__init__(
          feature_ndims,
          dtype=dtype,
          name=name,
          validate_args=validate_args,
          parameters=parameters)

  def _apply_with_distance(
      self, x1, x2, pairwise_square_distance, example_ndims=0):
    # Use util.sqrt_with_finite_grads to avoid NaN gradients when `x1 == x2`.
    norm = util.sqrt_with_finite_grads(pairwise_square_distance)
    if self.length_scale is not None:
      length_scale = tf.convert_to_tensor(self.length_scale)
      length_scale = util.pad_shape_with_ones(
          length_scale, ndims=example_ndims)
      norm /= length_scale
    log_result = -norm

    if self.amplitude is not None:
      amplitude = tf.convert_to_tensor(self.amplitude)
      amplitude = util.pad_shape_with_ones(
          amplitude, ndims=example_ndims)
      log_result += 2. * tf.math.log(amplitude)
    return tf.exp(log_result)

  def _apply(self, x1, x2, example_ndims=0):
    pairwise_square_distance = util.sum_rightmost_ndims_preserving_shape(
        tf.math.squared_difference(x1, x2), ndims=self.feature_ndims)
    return self._apply_with_distance(
        x1, x2, pairwise_square_distance, example_ndims=example_ndims)

  def _matrix(self, x1, x2):
    pairwise_square_distance = util.pairwise_square_distance_matrix(
        x1, x2, self.feature_ndims)
    return self._apply_with_distance(
        x1, x2, pairwise_square_distance, example_ndims=2)

  def _tensor(self, x1, x2, x1_example_ndims, x2_example_ndims):
    pairwise_square_distance = util.pairwise_square_distance_tensor(
        x1, x2, self.feature_ndims, x1_example_ndims, x2_example_ndims)
    return self._apply_with_distance(
        x1, x2, pairwise_square_distance,
        example_ndims=(x1_example_ndims + x2_example_ndims))


class MaternThreeHalves(_AmplitudeLengthScaleMixin, PositiveSemidefiniteKernel):
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
               name='MaternThreeHalves'):
    """Construct a MaternThreeHalves kernel instance.

    Args:
      amplitude: Positive floating point `Tensor` that controls the maximum
        value of the kernel. Must be broadcastable with `length_scale` and
        inputs to `apply` and `matrix` methods. A value of `None` is treated
        like 1.
      length_scale: Positive floating point `Tensor` that controls how sharp or
        wide the kernel shape is. This provides a characteristic "unit" of
        length against which `||x - y||` can be compared for scale. Must be
        broadcastable with `amplitude` and inputs to `apply` and `matrix`
        methods. A value of `None` is treated like 1.
      feature_ndims: Python `int` number of rightmost dims to include in the
        squared difference norm in the exponential.
      validate_args: If `True`, parameters are checked for validity despite
        possibly degrading runtime performance
      name: Python `str` name prefixed to Ops created by this class.
    """
    parameters = dict(locals())
    with tf.name_scope(name) as name:
      dtype = super(MaternThreeHalves, self)._init_params(
          amplitude, length_scale)
      super(MaternThreeHalves, self).__init__(
          feature_ndims,
          dtype=dtype,
          name=name,
          validate_args=validate_args,
          parameters=parameters)

  def _apply_with_distance(
      self, x1, x2, pairwise_square_distance, example_ndims=0):
    # Use util.sqrt_with_finite_grads to avoid NaN gradients when `x1 == x2`.
    norm = util.sqrt_with_finite_grads(pairwise_square_distance)
    if self.length_scale is not None:
      length_scale = tf.convert_to_tensor(self.length_scale)
      length_scale = util.pad_shape_with_ones(
          length_scale, ndims=example_ndims)
      norm = norm / length_scale
    series_term = np.sqrt(3) * norm
    log_result = tf.math.log1p(series_term) - series_term

    if self.amplitude is not None:
      amplitude = tf.convert_to_tensor(self.amplitude)
      amplitude = util.pad_shape_with_ones(amplitude, example_ndims)
      log_result += 2. * tf.math.log(amplitude)
    return tf.exp(log_result)

  def _apply(self, x1, x2, example_ndims=0):
    pairwise_square_distance = util.sum_rightmost_ndims_preserving_shape(
        tf.math.squared_difference(x1, x2), ndims=self.feature_ndims)
    return self._apply_with_distance(
        x1, x2, pairwise_square_distance, example_ndims=example_ndims)

  def _matrix(self, x1, x2):
    pairwise_square_distance = util.pairwise_square_distance_matrix(
        x1, x2, self.feature_ndims)
    return self._apply_with_distance(
        x1, x2, pairwise_square_distance, example_ndims=2)

  def _tensor(self, x1, x2, x1_example_ndims, x2_example_ndims):
    pairwise_square_distance = util.pairwise_square_distance_tensor(
        x1, x2, self.feature_ndims, x1_example_ndims, x2_example_ndims)
    return self._apply_with_distance(
        x1, x2, pairwise_square_distance,
        example_ndims=(x1_example_ndims + x2_example_ndims))


class MaternFiveHalves(_AmplitudeLengthScaleMixin, PositiveSemidefiniteKernel):
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
               name='MaternFiveHalves'):
    """Construct a MaternFiveHalves kernel instance.

    Args:
      amplitude: Positive floating point `Tensor` that controls the maximum
        value of the kernel. Must be broadcastable with `length_scale` and
        inputs to `apply` and `matrix` methods. A value of `None` is treated
        like 1.
      length_scale: Positive floating point `Tensor` that controls how sharp or
        wide the kernel shape is. This provides a characteristic "unit" of
        length against which `||x - y||` can be compared for scale. Must be
        broadcastable with `amplitude`, and inputs to `apply` and `matrix`
        methods. A value of `None` is treated like 1.
      feature_ndims: Python `int` number of rightmost dims to include in the
        squared difference norm in the exponential.
      validate_args: If `True`, parameters are checked for validity despite
        possibly degrading runtime performance
      name: Python `str` name prefixed to Ops created by this class.
    """
    parameters = dict(locals())
    with tf.name_scope(name) as name:
      dtype = super(MaternFiveHalves, self)._init_params(
          amplitude, length_scale)
      super(MaternFiveHalves, self).__init__(
          feature_ndims,
          dtype=dtype,
          name=name,
          validate_args=validate_args,
          parameters=parameters)

  def _apply_with_distance(
      self, x1, x2, pairwise_square_distance, example_ndims=0):
    # Use util.sqrt_with_finite_grads to avoid NaN gradients when `x1 == x2`.
    norm = util.sqrt_with_finite_grads(pairwise_square_distance)
    if self.length_scale is not None:
      length_scale = tf.convert_to_tensor(self.length_scale)
      length_scale = util.pad_shape_with_ones(
          length_scale, ndims=example_ndims)
      norm = norm / length_scale
    series_term = np.sqrt(5) * norm
    log_result = tf.math.log1p(series_term + series_term**2 / 3.) - series_term

    if self.amplitude is not None:
      amplitude = tf.convert_to_tensor(self.amplitude)
      amplitude = util.pad_shape_with_ones(amplitude, example_ndims)
      log_result += 2. * tf.math.log(amplitude)
    return tf.exp(log_result)

  def _apply(self, x1, x2, example_ndims=0):
    pairwise_square_distance = util.sum_rightmost_ndims_preserving_shape(
        tf.math.squared_difference(x1, x2), ndims=self.feature_ndims)
    return self._apply_with_distance(
        x1, x2, pairwise_square_distance, example_ndims=example_ndims)

  def _matrix(self, x1, x2):
    pairwise_square_distance = util.pairwise_square_distance_matrix(
        x1, x2, self.feature_ndims)
    return self._apply_with_distance(
        x1, x2, pairwise_square_distance, example_ndims=2)

  def _tensor(self, x1, x2, x1_example_ndims, x2_example_ndims):
    pairwise_square_distance = util.pairwise_square_distance_tensor(
        x1, x2, self.feature_ndims, x1_example_ndims, x2_example_ndims)
    return self._apply_with_distance(
        x1, x2, pairwise_square_distance,
        example_ndims=(x1_example_ndims + x2_example_ndims))
