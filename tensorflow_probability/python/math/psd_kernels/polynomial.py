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

import tensorflow.compat.v2 as tf

from tensorflow_probability.python.internal import assert_util
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import parameter_properties
from tensorflow_probability.python.internal import tensor_util
from tensorflow_probability.python.math.psd_kernels import positive_semidefinite_kernel as psd_kernel
from tensorflow_probability.python.math.psd_kernels.internal import util

__all__ = [
    'Constant',
    'Linear',
    'Polynomial',
]


class Polynomial(psd_kernel.AutoCompositeTensorPsdKernel):
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
               parameters=None,
               name='Polynomial'):
    """Construct a Polynomial kernel instance.

    Args:
      bias_variance: Non-negative floating point `Tensor` that controls the
        variance from the origin. If bias = 0, there is no variance and the
        fitted function goes through the origin.  Must be broadcastable with
        `slope_variance`, `shift`, `exponent`, and inputs to `apply` and
        `matrix` methods. A value of `None` is treated like 0.
        Default Value: `None`
      slope_variance: Non-negative floating point `Tensor` that controls the
        variance of the regression line slope that is the basis for the
        polynomial. Must be broadcastable with `bias_variance`, `shift`,
        `exponent`, and inputs to `apply` and `matrix` methods. A value of
        `None` is treated like 1.
        Default Value: `None`
      shift: Floating point `Tensor` that contols the intercept with the x-axis
        of the linear function to be exponentiated to get this polynomial. Must
        be broadcastable with `bias_variance`, `slope_variance`, `exponent` and
        inputs to `apply` and `matrix` methods. A value of `None` is treated
        like 0, which results in having the intercept at the origin.
        Default Value: `None`
      exponent: Positive floating point `Tensor` that controls the exponent
        (also known as the degree) of the polynomial function, and must be an
        integer.
        Must be broadcastable with `bias_variance`, `slope_variance`, `shift`,
        and inputs to `apply` and `matrix` methods. A value of `None` is treated
        like 1, which results in a linear kernel.
        Default Value: `None`
      feature_ndims: Python `int` number of rightmost dims to include in kernel
        computation.
        Default Value: 1
      validate_args: If `True`, parameters are checked for validity despite
        possibly degrading runtime performance.
        Default Value: `False`
      parameters: For subclasses, a dict of constructor arguments.
      name: Python `str` name prefixed to Ops created by this class.
        Default Value: `'Polynomial'`
    """
    parameters = dict(locals()) if parameters is None else parameters
    with tf.name_scope(name):
      dtype = util.maybe_get_common_dtype(
          [bias_variance, slope_variance, shift, exponent])
      self._bias_variance = tensor_util.convert_nonref_to_tensor(
          bias_variance, name='bias_variance', dtype=dtype)
      self._slope_variance = tensor_util.convert_nonref_to_tensor(
          slope_variance, name='slope_variance', dtype=dtype)
      self._shift = tensor_util.convert_nonref_to_tensor(
          shift, name='shift', dtype=dtype)
      self._exponent = tensor_util.convert_nonref_to_tensor(
          exponent, name='exponent', dtype=dtype)
      super(Polynomial, self).__init__(
          feature_ndims,
          dtype=dtype,
          name=name,
          validate_args=validate_args,
          parameters=parameters)

  @classmethod
  def _parameter_properties(cls, dtype):
    from tensorflow_probability.python.bijectors import softplus  # pylint:disable=g-import-not-at-top
    return dict(
        bias_variance=parameter_properties.ParameterProperties(
            default_constraining_bijector_fn=(
                lambda: softplus.Softplus(low=dtype_util.eps(dtype)))),
        exponent=parameter_properties.ParameterProperties(
            default_constraining_bijector_fn=(
                lambda: softplus.Softplus(low=dtype_util.eps(dtype)))),
        slope_variance=parameter_properties.ParameterProperties(
            default_constraining_bijector_fn=(
                lambda: softplus.Softplus(low=dtype_util.eps(dtype)))),
        shift=parameter_properties.ParameterProperties())

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

  def _apply(self, x1, x2, example_ndims=0):
    if self.shift is None:
      dot_prod = util.sum_rightmost_ndims_preserving_shape(
          x1 * x2, ndims=self.feature_ndims)
    else:
      shift = tf.convert_to_tensor(self.shift)
      shift = util.pad_shape_with_ones(shift,
                                       example_ndims + self.feature_ndims)
      dot_prod = util.sum_rightmost_ndims_preserving_shape(
          (x1 - shift) * (x2 - shift), ndims=self.feature_ndims)

    if self.exponent is not None:
      exponent = tf.convert_to_tensor(self.exponent)
      exponent = util.pad_shape_with_ones(exponent, example_ndims)
      dot_prod **= exponent

    if self.slope_variance is not None:
      slope_variance = tf.convert_to_tensor(self.slope_variance)
      slope_variance = util.pad_shape_with_ones(slope_variance, example_ndims)
      dot_prod *= slope_variance**2.

    if self.bias_variance is not None:
      bias_variance = tf.convert_to_tensor(self.bias_variance)
      bias_variance = util.pad_shape_with_ones(bias_variance, example_ndims)
      dot_prod += bias_variance**2.

    return dot_prod

  def _parameter_control_dependencies(self, is_init):
    if not self.validate_args:
      return []
    assertions = []
    ok_to_check = lambda x: (  # pylint:disable=g-long-lambda
        x is not None) and (is_init != tensor_util.is_ref(x))

    bias_variance = self.bias_variance
    slope_variance = self.slope_variance

    if ok_to_check(self.exponent):
      exponent = tf.convert_to_tensor(self.exponent)
      assertions.append(
          assert_util.assert_positive(
              exponent, message='`exponent` must be positive.'))
      from tensorflow_probability.python.internal import distribution_util  # pylint: disable=g-import-not-at-top
      assertions.append(
          distribution_util.assert_integer_form(
              exponent, message='`exponent` must be an integer.'))
    if ok_to_check(self.bias_variance):
      bias_variance = tf.convert_to_tensor(self.bias_variance)
      assertions.append(
          assert_util.assert_non_negative(
              bias_variance, message='`bias_variance` must be non-negative.'))
    if ok_to_check(self.slope_variance):
      slope_variance = tf.convert_to_tensor(self.slope_variance)
      assertions.append(
          assert_util.assert_non_negative(
              slope_variance,
              message='`slope_variance` must be non-negative.'))

    if (ok_to_check(self.bias_variance) and ok_to_check(self.slope_variance)):
      assertions.append(
          assert_util.assert_positive(
              tf.math.abs(slope_variance) + tf.math.abs(bias_variance),
              message=('`slope_variance` and `bias_variance` '
                       'can not both be zero.')))

    return assertions


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
               parameters=None,
               name='Linear'):
    """Construct a Linear kernel instance.

    Args:
      bias_variance: Positive floating point `Tensor` that controls the variance
        from the origin. If bias = 0, there is no variance and the fitted
        function goes through the origin (also known as the homogeneous linear
        kernel). Must be broadcastable with `slope_variance`, `shift` and inputs
        to `apply` and `matrix` methods. A value of `None` is treated like 0.
        Default Value: `None`
      slope_variance: Positive floating point `Tensor` that controls the
        variance of the regression line slope. Must be broadcastable with
        `bias_variance`, `shift`, and inputs to `apply` and `matrix` methods. A
        value of `None` is treated like 1.
        Default Value: `None`
      shift: Floating point `Tensor` that controls the intercept with the x-axis
        of the linear interpolation. Must be broadcastable with `bias_variance`,
        `slope_variance`, and inputs to `apply` and `matrix` methods. A value of
        `None` is treated like 0, which results in having the intercept at the
        origin.
      feature_ndims: Python `int` number of rightmost dims to include in kernel
        computation.
        Default Value: 1
      validate_args: If `True`, parameters are checked for validity despite
        possibly degrading runtime performance.
        Default Value: `False`
      parameters: For subclasses, a dict of constructor arguments.
      name: Python `str` name prefixed to Ops created by this class.
        Default Value: `'Linear'`
    """
    parameters = dict(locals()) if parameters is None else parameters
    super(Linear, self).__init__(
        bias_variance=bias_variance,
        slope_variance=slope_variance,
        shift=shift,
        exponent=None,
        feature_ndims=feature_ndims,
        validate_args=validate_args,
        parameters=parameters,
        name=name)

  @classmethod
  def _parameter_properties(cls, dtype):
    from tensorflow_probability.python.bijectors import softplus  # pylint:disable=g-import-not-at-top
    return dict(
        bias_variance=parameter_properties.ParameterProperties(
            default_constraining_bijector_fn=(
                lambda: softplus.Softplus(low=dtype_util.eps(dtype)))),
        slope_variance=parameter_properties.ParameterProperties(
            default_constraining_bijector_fn=(
                lambda: softplus.Softplus(low=dtype_util.eps(dtype)))),
        shift=parameter_properties.ParameterProperties())


class Constant(Linear):
  """Kernel that just outputs positive constant values.

  Useful class for multiplying / adding constants with other kernels.

  Warning: This can potentially lead to poorly conditioned matrices, since
  if the constant is large, adding it to each entry of another matrix will
  make it closer to an ill-conditioned matrix. If using a `Constant` kernel
  as a summand for a composite kernel in a `GaussianProcess`, it's instead
  recommended to use a trainable mean instead.
  """

  def __init__(self,
               constant,
               feature_ndims=1,
               validate_args=False,
               name='Constant'):
    """Construct a constant kernel instance.

    Args:
      constant: Positive floating point `Tensor` (or convertible) that is used
        for all kernel entries.
      feature_ndims: Python `int` number of rightmost dims to include in kernel
        computation.
      validate_args: If `True`, parameters are checked for validity despite
        possibly degrading runtime performance
      name: Python `str` name prefixed to Ops created by this class.
    """
    parameters = dict(locals())
    with tf.name_scope(name):
      self._constant = tensor_util.convert_nonref_to_tensor(
          constant, name='constant')
      from tensorflow_probability.python import util as tfp_util  # pylint:disable=g-import-not-at-top
      super(Constant, self).__init__(
          bias_variance=tfp_util.DeferredTensor(self._constant, tf.math.sqrt),
          slope_variance=0.0,
          shift=None,
          feature_ndims=feature_ndims,
          validate_args=validate_args,
          parameters=parameters,
          name=name)

  @property
  def constant(self):
    return self._constant

  @classmethod
  def _parameter_properties(cls, dtype):
    from tensorflow_probability.python.bijectors import softplus  # pylint:disable=g-import-not-at-top
    return dict(
        constant=parameter_properties.ParameterProperties(
            default_constraining_bijector_fn=(
                lambda: softplus.Softplus(low=dtype_util.eps(dtype)))))
