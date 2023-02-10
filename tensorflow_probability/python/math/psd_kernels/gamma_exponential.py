# Copyright 2023 The TensorFlow Probability Authors.
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
"""The GammaExponential kernel."""

import tensorflow.compat.v2 as tf

from tensorflow_probability.python.internal import assert_util
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import parameter_properties
from tensorflow_probability.python.internal import tensor_util
from tensorflow_probability.python.math.psd_kernels import positive_semidefinite_kernel as psd_kernel
from tensorflow_probability.python.math.psd_kernels.internal import util


__all__ = ['GammaExponential']


class GammaExponential(psd_kernel.AutoCompositeTensorPsdKernel):
  """The GammaExponential kernel.

  A generalization of the `ExponentiatedQuadratic` ("squared exponential",
  "Gaussian" or "radial basis function") kernel, this kernel takes the form

    ```none
    k(x, y) = amplitude**2 * exp(
        -||x - y||**(2 * power) / (2 * length_scale**2))
    ```

  where the double-bars represent vector length (ie, Euclidean, or L2 norm).

  The `power` parameter is sometimes referred to as gamma, or as gamma / 2.
  This parameter controls the shape of the kernel.
  """

  def __init__(self,
               amplitude=None,
               length_scale=None,
               inverse_length_scale=None,
               power=None,
               feature_ndims=1,
               validate_args=False,
               name='GammaExponential'):
    """Construct an GammaExponential kernel instance.

    Args:
      amplitude: floating point `Tensor` that controls the maximum value
        of the kernel. Must be broadcastable with `length_scale`, `power`, and
        inputs to `apply` and `matrix` methods. Must be greater than zero. A
        value of `None` is treated like 1.
        Default value: None
      length_scale: floating point `Tensor` that controls how sharp or wide the
        kernel shape is. This provides a characteristic "unit" of length against
        which `||x - y||` can be compared for scale. Must be broadcastable with
        `amplitude` and inputs to `apply` and `matrix` methods. A value of
        `None` is treated like 1. Only one of `length_scale` or
        `inverse_length_scale` should be provided.
        Default value: None
      inverse_length_scale: Non-negative floating point `Tensor` that is
        treated as `1 / length_scale`. Only one of `length_scale` or
        `inverse_length_scale` should be provided.
        Default value: None
      power: Non-negative floating point `Tensor`. Must be in `(0, 1]`.
        Sometimes referred to as gamma (or, gamma / 2). `None` is treated as
        `1`, equivalent to an `ExponentiatedQuadratic` kernel.
        Default value: None
      feature_ndims: Python `int` number of rightmost dims to include in the
        squared difference norm in the exponential.
      validate_args: If `True`, parameters are checked for validity despite
        possibly degrading runtime performance
      name: Python `str` name prefixed to Ops created by this class.
    """
    parameters = dict(locals())
    if (length_scale is not None) and (inverse_length_scale is not None):
      raise ValueError('Must specify at most one of `length_scale` and '
                       '`inverse_length_scale`.')
    with tf.name_scope(name):
      dtype = util.maybe_get_common_dtype(
          [amplitude, length_scale, inverse_length_scale, power])
      self._amplitude = tensor_util.convert_nonref_to_tensor(
          amplitude, name='amplitude', dtype=dtype)
      self._length_scale = tensor_util.convert_nonref_to_tensor(
          length_scale, name='length_scale', dtype=dtype)
      self._inverse_length_scale = tensor_util.convert_nonref_to_tensor(
          inverse_length_scale, name='inverse_length_scale', dtype=dtype)
      self._power = tensor_util.convert_nonref_to_tensor(
          power, name='power', dtype=dtype)
      super().__init__(
          feature_ndims,
          dtype=dtype,
          name=name,
          validate_args=validate_args,
          parameters=parameters)

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

  @property
  def power(self):
    """Power (gamma) parameter."""
    return self._power

  def _inverse_length_scale_parameter(self):
    if self.inverse_length_scale is None:
      if self.length_scale is not None:
        return tf.math.reciprocal(self.length_scale)
      return None
    return tf.convert_to_tensor(self.inverse_length_scale)

  @classmethod
  def _parameter_properties(cls, dtype):
    from tensorflow_probability.python.bijectors import sigmoid  # pylint:disable=g-import-not-at-top
    from tensorflow_probability.python.bijectors import softplus  # pylint:disable=g-import-not-at-top
    return dict(
        amplitude=parameter_properties.ParameterProperties(
            default_constraining_bijector_fn=(
                lambda: softplus.Softplus(low=dtype_util.eps(dtype)))),
        length_scale=parameter_properties.ParameterProperties(
            default_constraining_bijector_fn=(
                lambda: softplus.Softplus(low=dtype_util.eps(dtype)))),
        inverse_length_scale=parameter_properties.ParameterProperties(
            default_constraining_bijector_fn=softplus.Softplus),
        power=parameter_properties.ParameterProperties(
            default_constraining_bijector_fn=(
                lambda: sigmoid.Sigmoid(low=dtype_util.eps(dtype), high=1.))))

  def _apply_with_distance(
      self, x1, x2, pairwise_square_distance, example_ndims=0):

    if self.power is not None:
      power = tf.convert_to_tensor(self.power)
      power = util.pad_shape_with_ones(power, example_ndims)
      pairwise_pow_distance = pairwise_square_distance ** power
    else:
      pairwise_pow_distance = pairwise_square_distance

    exponent = -0.5 * pairwise_pow_distance
    inverse_length_scale = self._inverse_length_scale_parameter()
    if inverse_length_scale is not None:
      inverse_length_scale = util.pad_shape_with_ones(
          inverse_length_scale, example_ndims)
      exponent = exponent * tf.math.square(inverse_length_scale)

    if self.amplitude is not None:
      amplitude = tf.convert_to_tensor(self.amplitude)
      amplitude = util.pad_shape_with_ones(amplitude, example_ndims)
      exponent = exponent + 2. * tf.math.log(amplitude)

    return tf.exp(exponent)

  def _apply(self, x1, x2, example_ndims=0):
    pairwise_square_distance = util.sum_rightmost_ndims_preserving_shape(
        tf.math.squared_difference(x1, x2), self.feature_ndims)
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

  def _parameter_control_dependencies(self, is_init):
    if not self.validate_args:
      return []
    assertions = []
    if (self._inverse_length_scale is not None and
        is_init != tensor_util.is_ref(self._inverse_length_scale)):
      assertions.append(assert_util.assert_non_negative(
          self._inverse_length_scale,
          message='`inverse_length_scale` must be non-negative.'))
    if (self._power is not None and
        is_init != tensor_util.is_ref(self._power)):
      power = tf.convert_to_tensor(self._power)
      assertions.append(assert_util.assert_positive(
          power, message='`power` must be positive.'))
      assertions.append(assert_util.assert_less_equal(
          power, tf.ones([], dtype=self.dtype),
          message='`power` must be <= 1.'))
    for arg_name, arg in dict(amplitude=self.amplitude,
                              length_scale=self._length_scale).items():
      if arg is not None and is_init != tensor_util.is_ref(arg):
        assertions.append(assert_util.assert_positive(
            arg, message=f'{arg_name} must be positive.'))
    return assertions
