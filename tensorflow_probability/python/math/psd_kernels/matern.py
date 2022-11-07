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

import numpy as np
import tensorflow.compat.v2 as tf

from tensorflow_probability.python.internal import assert_util
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import parameter_properties
from tensorflow_probability.python.internal import tensor_util
from tensorflow_probability.python.math import bessel
from tensorflow_probability.python.math.psd_kernels import positive_semidefinite_kernel as psd_kernel
from tensorflow_probability.python.math.psd_kernels.internal import util

__all__ = [
    'GeneralizedMatern',
    'MaternOneHalf',
    'MaternThreeHalves',
    'MaternFiveHalves',
]


class _AmplitudeLengthScaleMixin(object):
  """Shared logic for amplitude/length_scale parameterized kernels."""

  def _init_params(self, amplitude, length_scale, inverse_length_scale):
    """Shared init logic for `amplitude` and `length_scale` params.

    Args:
      amplitude: `Tensor` (or convertible) or `None` to convert, validate.
      length_scale: `Tensor` (or convertible) or `None` to convert, validate.
      inverse_length_scale: `Tensor` (or convertible) or `None` to convert,
        validate.

    Returns:
      dtype: The common `DType` of the parameters.
    """
    if (length_scale is not None) and (inverse_length_scale is not None):
      raise ValueError('Must specify at most one of `length_scale` and '
                       '`inverse_length_scale`.')
    dtype = util.maybe_get_common_dtype(
        [amplitude, length_scale, inverse_length_scale])
    self._amplitude = tensor_util.convert_nonref_to_tensor(
        amplitude, name='amplitude', dtype=dtype)
    self._length_scale = tensor_util.convert_nonref_to_tensor(
        length_scale, name='length_scale', dtype=dtype)
    self._inverse_length_scale = tensor_util.convert_nonref_to_tensor(
        inverse_length_scale, name='inverse_length_scale', dtype=dtype)
    return dtype

  @property
  def amplitude(self):
    """Amplitude parameter."""
    return self._amplitude

  @property
  def length_scale(self):
    """Length scale parameter."""
    return self._length_scale

  @property
  def inverse_length_scale(self):
    """Inverse length scale parameter."""
    return self._inverse_length_scale

  def _inverse_length_scale_parameter(self):
    if self.inverse_length_scale is None:
      if self.length_scale is not None:
        return tf.math.reciprocal(self.length_scale)
      return None
    return tf.convert_to_tensor(self.inverse_length_scale)

  @classmethod
  def _parameter_properties(cls, dtype):
    from tensorflow_probability.python.bijectors import softplus  # pylint:disable=g-import-not-at-top
    return dict(
        amplitude=parameter_properties.ParameterProperties(
            default_constraining_bijector_fn=(
                lambda: softplus.Softplus(low=dtype_util.eps(dtype)))),
        length_scale=parameter_properties.ParameterProperties(
            default_constraining_bijector_fn=(
                lambda: softplus.Softplus(low=dtype_util.eps(dtype)))),
        inverse_length_scale=parameter_properties.ParameterProperties(
            default_constraining_bijector_fn=softplus.Softplus))

  def _parameter_control_dependencies(self, is_init):
    """Control dependencies for parameters."""
    if not self.validate_args:
      return []
    assertions = []
    if (self._inverse_length_scale is not None and
        is_init != tensor_util.is_ref(self._inverse_length_scale)):
      assertions.append(assert_util.assert_non_negative(
          self._inverse_length_scale,
          message='`inverse_scale_diag` must be non-negative.'))
    for arg_name, arg in dict(amplitude=self.amplitude,
                              length_scale=self._length_scale).items():
      if arg is not None and is_init != tensor_util.is_ref(arg):
        assertions.append(assert_util.assert_positive(
            arg, message=f'{arg_name} must be positive.'))
    return assertions

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


class GeneralizedMatern(_AmplitudeLengthScaleMixin,
                        psd_kernel.AutoCompositeTensorPsdKernel):
  """Generalized Matern Kernel.

  This kernel parameterizes the Matern family of kernels.

  The kernel has the following form:

  ```none
    k(x, y) = amplitude ** 2 * 2 ** (1 - v) / Gamma(v) * s ** v * K(v, s)
  ```
  where
    * `s` is `sqrt(2 * v) * d / l`.
    * `v` is the degrees of freedom parameter `df`.
    * `d` is the Euclidean distance `||x - y||**2`.
    * `l` is the `length_scale`.
    * `K(v, s)` is the Modified Bessel function of the second kind.

  where the double-bars represent vector length (i.e. Euclidean, or L2 Norm).
  This kernel, acts over the space `S = R^(D1 x D2 .. x Dd)`.

  Warning: Gradients are not available with respect to the `df` parameter.
  Warning: It is recommended that for `df = 0.5, 1.5, 2.5` to use the
  specialized kernels `MaternOneHalf`, `MaternThreeHalves` and
  `MaternFiveHalves` instead.
  """

  def __init__(self,
               df,
               amplitude=None,
               length_scale=None,
               inverse_length_scale=None,
               feature_ndims=1,
               validate_args=False,
               name='GeneralizedMatern'):
    parameters = dict(locals())
    with tf.name_scope(name) as name:
      super(GeneralizedMatern, self)._init_params(
          amplitude, length_scale, inverse_length_scale)
      dtype = util.maybe_get_common_dtype([self._amplitude, df])
      self._df = tensor_util.convert_nonref_to_tensor(
          df, name='df', dtype=dtype)
      super(GeneralizedMatern, self).__init__(
          feature_ndims,
          dtype=dtype,
          name=name,
          validate_args=validate_args,
          parameters=parameters)

  @property
  def df(self):
    """Degree of Freedom parameter."""
    return self._df

  @classmethod
  def _parameter_properties(cls, dtype):
    from tensorflow_probability.python.bijectors import softplus  # pylint:disable=g-import-not-at-top
    return dict(
        amplitude=parameter_properties.ParameterProperties(
            default_constraining_bijector_fn=(
                lambda: softplus.Softplus(low=dtype_util.eps(dtype)))),
        df=parameter_properties.ParameterProperties(
            default_constraining_bijector_fn=(
                lambda: softplus.Softplus(low=dtype_util.eps(dtype)))),
        length_scale=parameter_properties.ParameterProperties(
            default_constraining_bijector_fn=(
                lambda: softplus.Softplus(low=dtype_util.eps(dtype)))),
        inverse_length_scale=parameter_properties.ParameterProperties(
            default_constraining_bijector_fn=softplus.Softplus))

  def _apply_with_distance(
      self, x1, x2, pairwise_square_distance, example_ndims=0):
    norm = tf.math.sqrt(pairwise_square_distance)
    inverse_length_scale = self._inverse_length_scale_parameter()
    if inverse_length_scale is not None:
      inverse_length_scale = util.pad_shape_with_ones(
          inverse_length_scale, ndims=example_ndims)
      norm = norm * inverse_length_scale
    df = tf.convert_to_tensor(self.df)
    df = tf.stop_gradient(df)
    df = util.pad_shape_with_ones(df, ndims=example_ndims)
    norm = tf.math.sqrt(2 * df) * norm

    # When norm -> 0, the expression should tend to zero (along
    # with the gradient).
    safe_norm = tf.where(
        tf.math.equal(norm, 0.),
        dtype_util.as_numpy_dtype(self.dtype)(1.),
        norm)
    log_result = tf.where(
        tf.math.equal(norm, 0.),
        dtype_util.as_numpy_dtype(self.dtype)(0.),
        df * tf.math.log(safe_norm) +
        bessel.log_bessel_kve(df, safe_norm) - safe_norm -
        tf.math.lgamma(df) + (1. - df) * np.log(2.))

    if self.amplitude is not None:
      amplitude = tf.convert_to_tensor(self.amplitude)
      amplitude = util.pad_shape_with_ones(
          amplitude, ndims=example_ndims)
      log_result = log_result + 2. * tf.math.log(amplitude)
    return tf.exp(log_result)

  def _parameter_control_dependencies(self, is_init):
    """Control dependencies for parameters."""
    if not self.validate_args:
      return []
    assertions = []
    if (self._inverse_length_scale is not None and
        is_init != tensor_util.is_ref(self._inverse_length_scale)):
      assertions.append(assert_util.assert_non_negative(
          self._inverse_length_scale,
          message='`inverse_scale_diag` must be non-negative.'))
    for arg_name, arg in dict(
        df=self.df,
        amplitude=self.amplitude,
        length_scale=self._length_scale).items():
      if arg is not None and is_init != tensor_util.is_ref(arg):
        assertions.append(assert_util.assert_positive(
            arg,
            message=f'{arg_name} must be positive.'))
    return assertions


class MaternOneHalf(_AmplitudeLengthScaleMixin,
                    psd_kernel.AutoCompositeTensorPsdKernel):
  """Matern Kernel with parameter 1/2.

  This kernel is part of the Matern family of kernels, with parameter 1/2.
  Also known as the Exponential or Laplacian Kernel, a Gaussian process
  parameterized by this kernel, is also known as an Ornstein-Uhlenbeck process
  (https://en.wikipedia.org/wiki/Ornstein%E2%80%93Uhlenbeck_process).

  The kernel has the following form:

  ```none
    k(x, y) = amplitude ** 2  * exp(-||x - y|| / length_scale)
  ```

  where the double-bars represent vector length (i.e. Euclidean, or L2 Norm).
  This kernel, acts over the space `S = R^(D1 x D2 .. x Dd)`.
  """

  def __init__(self,
               amplitude=None,
               length_scale=None,
               inverse_length_scale=None,
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
      inverse_length_scale: Non-negative floating point `Tensor` that is
        treated as `1 / length_scale`. Only one of `length_scale` or
        `inverse_length_scale` should be provided.
        Default value: None
      feature_ndims: Python `int` number of rightmost dims to include in the
        squared difference norm in the exponential.
      validate_args: If `True`, parameters are checked for validity despite
        possibly degrading runtime performance
      name: Python `str` name prefixed to Ops created by this class.
    """
    parameters = dict(locals())
    with tf.name_scope(name) as name:
      dtype = super(MaternOneHalf, self)._init_params(
          amplitude, length_scale, inverse_length_scale)
      super(MaternOneHalf, self).__init__(
          feature_ndims,
          dtype=dtype,
          name=name,
          validate_args=validate_args,
          parameters=parameters)

  def _apply_with_distance(
      self, x1, x2, pairwise_square_distance, example_ndims=0):
    # Where pairwise square distance is 0, gradients with respect to each of the
    # inputs should be 0 as well. Set square distances to be statically 0 to
    # ensure that gradients are 0 (and not infinity/NaN when the square root is
    # taken).
    pairwise_sq = tf.where(
        tf.equal(pairwise_square_distance, 0.),
        tf.zeros([], dtype=pairwise_square_distance.dtype),
        pairwise_square_distance)
    norm = tf.math.sqrt(pairwise_sq)
    inverse_length_scale = self._inverse_length_scale_parameter()
    if inverse_length_scale is not None:
      inverse_length_scale = util.pad_shape_with_ones(
          inverse_length_scale, ndims=example_ndims)
      norm = norm * inverse_length_scale
    log_result = -norm

    if self.amplitude is not None:
      amplitude = tf.convert_to_tensor(self.amplitude)
      amplitude = util.pad_shape_with_ones(
          amplitude, ndims=example_ndims)
      log_result = log_result + 2. * tf.math.log(amplitude)
    return tf.exp(log_result)


class MaternThreeHalves(_AmplitudeLengthScaleMixin,
                        psd_kernel.AutoCompositeTensorPsdKernel):
  """Matern Kernel with parameter 3/2.

  This kernel is part of the Matern family of kernels, with parameter 3/2.

  ```none
    z = sqrt(3) * ||x - y|| / length_scale
    k(x, y) = amplitude ** 2 * (1 + z) * exp(-z)
  ```

  where the double-bars represent vector length (i.e. Euclidean, or L2 Norm).
  This kernel, acts over the space `S = R^(D1 x D2 .. x Dd)`.
  """

  def __init__(self,
               amplitude=None,
               length_scale=None,
               inverse_length_scale=None,
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
        broadcastable with `amplitude`, and inputs to `apply` and `matrix`
        methods. A value of `None` is treated like 1.
      inverse_length_scale: Non-negative floating point `Tensor` that is
        treated as `1 / length_scale`. Only one of `length_scale` or
        `inverse_length_scale` should be provided.
        Default value: None
      feature_ndims: Python `int` number of rightmost dims to include in the
        squared difference norm in the exponential.
      validate_args: If `True`, parameters are checked for validity despite
        possibly degrading runtime performance
      name: Python `str` name prefixed to Ops created by this class.
    """
    parameters = dict(locals())
    with tf.name_scope(name) as name:
      dtype = super(MaternThreeHalves, self)._init_params(
          amplitude, length_scale, inverse_length_scale)
      super(MaternThreeHalves, self).__init__(
          feature_ndims,
          dtype=dtype,
          name=name,
          validate_args=validate_args,
          parameters=parameters)

  def _apply_with_distance(
      self, x1, x2, pairwise_square_distance, example_ndims=0):
    # Where pairwise square distance is 0, gradients with respect to each of the
    # inputs should be 0 as well. Set square distances to be statically 0 to
    # ensure that gradients are 0 (and not infinity/NaN when the square root is
    # taken).
    pairwise_sq = tf.where(
        tf.equal(pairwise_square_distance, 0.),
        tf.zeros([], dtype=pairwise_square_distance.dtype),
        pairwise_square_distance)
    norm = tf.math.sqrt(pairwise_sq)
    np_dtype = dtype_util.as_numpy_dtype(norm.dtype)
    inverse_length_scale = self._inverse_length_scale_parameter()
    if inverse_length_scale is not None:
      inverse_length_scale = util.pad_shape_with_ones(
          inverse_length_scale, ndims=example_ndims)
      norm = norm * inverse_length_scale
    series_term = np.sqrt(3).astype(np_dtype) * norm
    log_result = tf.math.log1p(series_term) - series_term

    if self.amplitude is not None:
      amplitude = tf.convert_to_tensor(self.amplitude)
      amplitude = util.pad_shape_with_ones(amplitude, example_ndims)
      log_result = log_result + 2. * tf.math.log(amplitude)
    return tf.exp(log_result)


class MaternFiveHalves(_AmplitudeLengthScaleMixin,
                       psd_kernel.AutoCompositeTensorPsdKernel):
  """Matern 5/2 Kernel.

  This kernel is part of the Matern family of kernels, with parameter 5/2.

  ```none
    z = sqrt(5) * ||x - y|| / length_scale
    k(x, y) = amplitude ** 2 * (1 + z + (z ** 2) / 3) * exp(-z)
  ```

  where the double-bars represent vector length (i.e. Euclidean, or L2 Norm).
  This kernel, acts over the space `S = R^(D1 x D2 .. x Dd)`.
  """

  def __init__(self,
               amplitude=None,
               length_scale=None,
               inverse_length_scale=None,
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
      inverse_length_scale: Non-negative floating point `Tensor` that is
        treated as `1 / length_scale`. Only one of `length_scale` or
        `inverse_length_scale` should be provided.
        Default value: None
      feature_ndims: Python `int` number of rightmost dims to include in the
        squared difference norm in the exponential.
      validate_args: If `True`, parameters are checked for validity despite
        possibly degrading runtime performance
      name: Python `str` name prefixed to Ops created by this class.
    """
    parameters = dict(locals())
    with tf.name_scope(name) as name:
      dtype = super(MaternFiveHalves, self)._init_params(
          amplitude, length_scale, inverse_length_scale)
      super(MaternFiveHalves, self).__init__(
          feature_ndims,
          dtype=dtype,
          name=name,
          validate_args=validate_args,
          parameters=parameters)

  def _apply_with_distance(
      self, x1, x2, pairwise_square_distance, example_ndims=0):
    # Where pairwise square distance is 0, gradients with respect to each of the
    # inputs should be 0 as well. Set square distances to be statically 0 to
    # ensure that gradients are 0 (and not infinity/NaN when the square root is
    # taken).
    pairwise_sq = tf.where(
        tf.equal(pairwise_square_distance, 0.),
        tf.zeros([], dtype=pairwise_square_distance.dtype),
        pairwise_square_distance)
    norm = tf.math.sqrt(pairwise_sq)
    inverse_length_scale = self._inverse_length_scale_parameter()
    if inverse_length_scale is not None:
      inverse_length_scale = util.pad_shape_with_ones(
          inverse_length_scale, ndims=example_ndims)
      norm = norm * inverse_length_scale
    series_term = tf.math.sqrt(tf.constant(5., dtype=norm.dtype)) * norm
    log_result = tf.math.log1p(series_term + series_term**2 / 3.) - series_term

    if self.amplitude is not None:
      amplitude = tf.convert_to_tensor(self.amplitude)
      amplitude = util.pad_shape_with_ones(amplitude, example_ndims)
      log_result = log_result + 2. * tf.math.log(amplitude)
    return tf.exp(log_result)
