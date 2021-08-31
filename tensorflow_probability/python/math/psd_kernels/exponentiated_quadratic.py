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
"""The ExponentiatedQuadratic kernel."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v2 as tf

from tensorflow_probability.python.internal import assert_util
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import parameter_properties
from tensorflow_probability.python.internal import tensor_util
from tensorflow_probability.python.math.psd_kernels import positive_semidefinite_kernel as psd_kernel
from tensorflow_probability.python.math.psd_kernels.internal import util


__all__ = ['ExponentiatedQuadratic']


class ExponentiatedQuadratic(psd_kernel.AutoCompositeTensorPsdKernel):
  """The ExponentiatedQuadratic kernel.

  Sometimes called the "squared exponential", "Gaussian" or "radial basis
  function", this kernel function has the form

    ```none
    k(x, y) = amplitude**2 * exp(-||x - y||**2 / (2 * length_scale**2))
    ```

  where the double-bars represent vector length (ie, Euclidean, or L2 norm).
  """

  def __init__(self,
               amplitude=None,
               length_scale=None,
               feature_ndims=1,
               validate_args=False,
               name='ExponentiatedQuadratic'):
    """Construct an ExponentiatedQuadratic kernel instance.

    Args:
      amplitude: floating point `Tensor` that controls the maximum value
        of the kernel. Must be broadcastable with `length_scale` and inputs to
        `apply` and `matrix` methods. Must be greater than zero. A value of
        `None` is treated like 1.
        Default value: None
      length_scale: floating point `Tensor` that controls how sharp or wide the
        kernel shape is. This provides a characteristic "unit" of length against
        which `||x - y||` can be compared for scale. Must be broadcastable with
        `amplitude` and inputs to `apply` and `matrix` methods. A value of
        `None` is treated like 1.
        Default value: None
      feature_ndims: Python `int` number of rightmost dims to include in the
        squared difference norm in the exponential.
      validate_args: If `True`, parameters are checked for validity despite
        possibly degrading runtime performance
      name: Python `str` name prefixed to Ops created by this class.
    """
    parameters = dict(locals())
    with tf.name_scope(name):
      dtype = util.maybe_get_common_dtype(
          [amplitude, length_scale])
      self._amplitude = tensor_util.convert_nonref_to_tensor(
          amplitude, name='amplitude', dtype=dtype)
      self._length_scale = tensor_util.convert_nonref_to_tensor(
          length_scale, name='length_scale', dtype=dtype)
      super(ExponentiatedQuadratic, self).__init__(
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

  @classmethod
  def _parameter_properties(cls, dtype):
    from tensorflow_probability.python.bijectors import softplus  # pylint:disable=g-import-not-at-top
    return dict(
        amplitude=parameter_properties.ParameterProperties(
            default_constraining_bijector_fn=(
                lambda: softplus.Softplus(low=dtype_util.eps(dtype)))),
        length_scale=parameter_properties.ParameterProperties(
            default_constraining_bijector_fn=(
                lambda: softplus.Softplus(low=dtype_util.eps(dtype)))))

  def _apply_with_distance(
      self, x1, x2, pairwise_square_distance, example_ndims=0):
    exponent = -0.5 * pairwise_square_distance
    if self.length_scale is not None:
      length_scale = tf.convert_to_tensor(self.length_scale)
      length_scale = util.pad_shape_with_ones(
          length_scale, example_ndims)
      exponent = exponent / length_scale**2

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
    for arg_name, arg in dict(amplitude=self.amplitude,
                              length_scale=self.length_scale).items():
      if arg is not None and is_init != tensor_util.is_ref(arg):
        assertions.append(assert_util.assert_positive(
            arg,
            message='{} must be positive.'.format(arg_name)))
    return assertions
