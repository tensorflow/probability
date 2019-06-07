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

import tensorflow as tf
from tensorflow_probability.python.positive_semidefinite_kernels import positive_semidefinite_kernel as psd_kernel
from tensorflow_probability.python.positive_semidefinite_kernels.internal import util


__all__ = [
    'ExponentiatedQuadratic',
]


def _validate_arg_if_not_none(arg, assertion, validate_args):
  if arg is None:
    return arg
  with tf.control_dependencies([assertion(arg)] if validate_args else []):
    result = tf.identity(arg)
  return result


class ExponentiatedQuadratic(psd_kernel.PositiveSemidefiniteKernel):
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
        `apply` and `matrix` methods. Must be greater than zero.
      length_scale: floating point `Tensor` that controls how sharp or wide the
        kernel shape is. This provides a characteristic "unit" of length against
        which `||x - y||` can be compared for scale. Must be broadcastable with
        `amplitude` and inputs to `apply` and `matrix` methods.
      feature_ndims: Python `int` number of rightmost dims to include in the
        squared difference norm in the exponential.
      validate_args: If `True`, parameters are checked for validity despite
        possibly degrading runtime performance
      name: Python `str` name prefixed to Ops created by this class.
    """
    with tf.compat.v1.name_scope(
        name, values=[amplitude, length_scale]) as name:
      dtype = util.maybe_get_common_dtype(
          [amplitude, length_scale])
      if amplitude is not None:
        amplitude = tf.convert_to_tensor(
            value=amplitude, name='amplitude', dtype=dtype)
      self._amplitude = _validate_arg_if_not_none(
          amplitude, tf.compat.v1.assert_positive, validate_args)
      if length_scale is not None:
        length_scale = tf.convert_to_tensor(
            value=length_scale, name='length_scale', dtype=dtype)
      self._length_scale = _validate_arg_if_not_none(
          length_scale, tf.compat.v1.assert_positive, validate_args)
    super(ExponentiatedQuadratic, self).__init__(
        feature_ndims, dtype=dtype, name=name)

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
        [] if self.amplitude is None else tf.shape(input=self.amplitude),
        [] if self.length_scale is None else tf.shape(input=self.length_scale))

  def _apply(self, x1, x2, example_ndims=0):
    exponent = -0.5 * util.sum_rightmost_ndims_preserving_shape(
        tf.math.squared_difference(x1, x2), self.feature_ndims)
    if self.length_scale is not None:
      length_scale = util.pad_shape_with_ones(
          self.length_scale, example_ndims)
      exponent /= length_scale**2

    if self.amplitude is not None:
      amplitude = util.pad_shape_with_ones(
          self.amplitude, example_ndims)
      exponent += 2. * tf.math.log(amplitude)

    return tf.exp(exponent)
