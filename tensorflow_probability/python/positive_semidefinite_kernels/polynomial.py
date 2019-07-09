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
"""Polynomial and Linear kernel."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import functools


import tensorflow as tf
from tensorflow_probability.python.positive_semidefinite_kernels import positive_semidefinite_kernel as psd_kernel
from tensorflow_probability.python.positive_semidefinite_kernels.internal import util

__all__ = [
    'Linear',
    'Polynomial',
]


def _validate_arg_if_not_none(arg, assertion, validate_args):
  if arg is None:
    return arg
  with tf.control_dependencies([assertion(arg)] if validate_args else []):
    result = tf.identity(arg)
  return result


def _maybe_shape_static(tensor):
  if tensor is None:
    return tf.TensorShape([])
  return tensor.shape


def _maybe_shape_dynamic(tensor):
  if tensor is None:
    return []
  return tf.shape(input=tensor)


class Polynomial(psd_kernel.PositiveSemidefiniteKernel):
  """Polynomial Kernel.

    Is based on the dot product covariance function and can be obtained
    from polynomial regression. This kernel, when parameterizing a
    Gaussian Process, results in random polynomial functions.
    A linear kernel can be created from this by setting the exponent to 1
    or None.

    ```none
    k(x, y) = bias_variance**2 + slope_variance**2 *
              ((x - shift) dot (y - shift))**exponent
    ```

    #### References

    [1]: Carl Edward Rasmussen and Christopher K. I. Williams. Gaussian
         Processes for Machine Learning. Section 4.4.2. 2006.
         http://www.gaussianprocess.org/gpml/chapters/RW4.pdf
    [2]: David Duvenaud. The Kernel Cookbook.
         https://www.cs.toronto.edu/~duvenaud/cookbook/

  """

  def __init__(self,
               bias_variance=None,
               slope_variance=None,
               shift=None,
               exponent=None,
               feature_ndims=1,
               validate_args=False,
               name='Polynomial'):
    """Construct a Polynomial kernel instance.

    Args:
      bias_variance: Positive floating point `Tensor` that controls the
        variance from the origin. If bias = 0, there is no variance and the
        fitted function goes through the origin.  Must be broadcastable with
        `slope_variance`, `shift`, `exponent`, and inputs to `apply` and
        `matrix` methods. A value of `None` is treated like 0.
        Default Value: `None`
      slope_variance: Positive floating point `Tensor` that controls the
        variance of the regression line slope that is the basis for the
        polynomial. Must be broadcastable with `bias_variance`, `shift`,
        `exponent`, and inputs to `apply` and `matrix` methods. A value of
        `None` is treated like 1.
        Default Value: `None`
      shift: Floating point `Tensor` that contols the intercept with the
        x-axis of the linear function to be exponentiated to get this
        polynomial. Must be broadcastable with `bias_variance`,
        `slope_variance`, `exponent` and inputs to `apply` and `matrix`
        methods. A value of `None` is treated like 0, which results in having
        the intercept at the origin.
        Default Value: `None`
      exponent: Positive floating point `Tensor` that controls the exponent
        (also known as the degree) of the polynomial function. Must be
        broadcastable with `bias_variance`, `slope_variance`, `shift`,
        and inputs to `apply` and `matrix` methods. A value of `None` is
        treated like 1, which results in a linear kernel.
        Default Value: `None`
      feature_ndims: Python `int` number of rightmost dims to include in
        kernel computation.
        Default Value: 1
      validate_args: If `True`, parameters are checked for validity despite
        possibly degrading runtime performance.
        Default Value: `False`
      name: Python `str` name prefixed to Ops created by this class.
        Default Value: `'Polynomial'`
    """
    with tf.compat.v1.name_scope(
        name, values=[bias_variance, slope_variance, shift, exponent]):
      dtype = util.maybe_get_common_dtype(
          [bias_variance, slope_variance, shift, exponent])
      if bias_variance is not None:
        bias_variance = tf.convert_to_tensor(
            value=bias_variance, name='bias_variance', dtype=dtype)
      self._bias_variance = _validate_arg_if_not_none(
          bias_variance, tf.compat.v1.assert_positive, validate_args)
      if slope_variance is not None:
        slope_variance = tf.convert_to_tensor(
            value=slope_variance, name='slope_variance', dtype=dtype)
      self._slope_variance = _validate_arg_if_not_none(
          slope_variance, tf.compat.v1.assert_positive, validate_args)
      if shift is not None:
        shift = tf.convert_to_tensor(value=shift, name='shift', dtype=dtype)
      self._shift = shift
      if exponent is not None:
        exponent = tf.convert_to_tensor(
            value=exponent, name='exponent', dtype=dtype)
      self._exponent = _validate_arg_if_not_none(
          exponent, tf.compat.v1.assert_positive, validate_args)
    super(Polynomial, self).__init__(
        feature_ndims, dtype=dtype, name=name)

  @property
  def bias_variance(self):
    """Variance on bias parameter."""
    return self._bias_variance

  @property
  def slope_variance(self):
    """Variance on slope parameter."""
    return self._slope_variance

  @property
  def shift(self):
    """Shift of linear function that is exponentiated."""
    return self._shift

  @property
  def exponent(self):
    """Exponent of the polynomial term."""
    return self._exponent

  def _batch_shape(self):
    return functools.reduce(
        tf.broadcast_static_shape,
        map(_maybe_shape_static, [self.slope_variance, self.bias_variance,
                                  self.shift, self.exponent]))

  def _batch_shape_tensor(self):
    return functools.reduce(
        tf.broadcast_dynamic_shape,
        map(_maybe_shape_dynamic, [self.slope_variance, self.bias_variance,
                                   self.shift, self.exponent]))

  def _apply(self, x1, x2, example_ndims=0):
    if self.shift is None:
      dot_prod = util.sum_rightmost_ndims_preserving_shape(
          x1 * x2, ndims=self.feature_ndims)
    else:
      dot_prod = util.sum_rightmost_ndims_preserving_shape(
          (x1 - self.shift) * (x2 - self.shift),
          ndims=self.feature_ndims)

    if self.exponent is not None:
      exponent = util.pad_shape_with_ones(
          self.exponent, example_ndims)
      dot_prod **= exponent

    if self.slope_variance is not None:
      slope_variance = util.pad_shape_with_ones(
          self.slope_variance, example_ndims)
      dot_prod *= slope_variance ** 2.

    if self.bias_variance is not None:
      bias_variance = util.pad_shape_with_ones(
          self.bias_variance, example_ndims)
      dot_prod += bias_variance ** 2.

    return dot_prod


class Linear(Polynomial):
  """Linear Kernel.

    Is based on the dot product covariance function and can be obtained
    from linear regression. This kernel, when parameterizing a
    Gaussian Process, results in random linear functions.
    The Linear kernel is based on the Polynomial kernel without the
    exponent.

    ```none
    k(x, y) = bias_variance**2 + slope_variance**2 *
              ((x - shift) dot (y - shift))
    ```
  """

  def __init__(self,
               bias_variance=None,
               slope_variance=None,
               shift=None,
               feature_ndims=1,
               validate_args=False,
               name='Linear'):
    """Construct a Linear kernel instance.

    Args:
      bias_variance: Positive floating point `Tensor` that controls the
        variance from the origin. If bias = 0, there is no variance and the
        fitted function goes through the origin (also known as the homogeneous
        linear kernel). Must be broadcastable with `slope_variance`,
        `shift` and inputs to `apply` and `matrix` methods. A value of
        `None` is treated like 0.
        Default Value: `None`
      slope_variance: Positive floating point `Tensor` that controls the
        variance of the regression line slope. Must be broadcastable with
        `bias_variance`, `shift`, and inputs to `apply` and `matrix`
        methods. A value of `None` is treated like 1.
        Default Value: `None`
      shift: Floating point `Tensor` that controls the intercept with the
        x-axis of the linear interpolation. Must be broadcastable with
        `bias_variance`, `slope_variance`, and inputs to `apply` and `matrix`
        methods. A value of `None` is treated like 0, which results in having
        the intercept at the origin.
      feature_ndims: Python `int` number of rightmost dims to include in
        kernel computation.
        Default Value: 1
      validate_args: If `True`, parameters are checked for validity despite
        possibly degrading runtime performance.
        Default Value: `False`
      name: Python `str` name prefixed to Ops created by this class.
        Default Value: `'Linear'`
    """
    super(Linear, self).__init__(
        bias_variance=bias_variance,
        slope_variance=slope_variance,
        shift=shift,
        exponent=None,
        feature_ndims=feature_ndims,
        validate_args=validate_args,
        name=name)
