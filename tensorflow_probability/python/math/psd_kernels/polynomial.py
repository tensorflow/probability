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
from tensorflow_probability.python.internal import prefer_static as ps
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
    k(x, y) = bias_amplitude**2 + slope_amplitude**2 *
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
               bias_amplitude=None,
               slope_amplitude=None,
               shift=None,
               exponent=None,
               feature_ndims=1,
               validate_args=False,
               parameters=None,
               name='Polynomial'):
    """Construct a Polynomial kernel instance.

    Args:
      bias_amplitude: Non-negative floating point `Tensor` that controls the
        stddev from the origin. If bias = 0, there is no stddev and the
        fitted function goes through the origin.  Must be broadcastable with
        `slope_amplitude`, `shift`, `exponent`, and inputs to `apply` and
        `matrix` methods. A value of `None` is treated like 0.
        Default Value: `None`
      slope_amplitude: Non-negative floating point `Tensor` that controls the
        stddev of the regression line slope that is the basis for the
        polynomial. Must be broadcastable with `bias_amplitude`, `shift`,
        `exponent`, and inputs to `apply` and `matrix` methods. A value of
        `None` is treated like 1.
        Default Value: `None`
      shift: Floating point `Tensor` that contols the intercept with the x-axis
        of the linear function to be exponentiated to get this polynomial. Must
        be broadcastable with `bias_amplitude`, `slope_amplitude`, `exponent`
        and inputs to `apply` and `matrix` methods. A value of `None` is treated
        like 0, which results in having the intercept at the origin.
        Default Value: `None`
      exponent: Positive floating point `Tensor` that controls the exponent
        (also known as the degree) of the polynomial function, and must be an
        integer.
        Must be broadcastable with `bias_amplitude`, `slope_amplitude`, `shift`,
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
          [bias_amplitude,
           slope_amplitude,
           shift,
           exponent])
      self._bias_amplitude = tensor_util.convert_nonref_to_tensor(
          bias_amplitude, name='bias_amplitude', dtype=dtype)
      self._slope_amplitude = tensor_util.convert_nonref_to_tensor(
          slope_amplitude, name='slope_amplitude', dtype=dtype)
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
        bias_amplitude=parameter_properties.ParameterProperties(
            default_constraining_bijector_fn=(
                lambda: softplus.Softplus(low=dtype_util.eps(dtype)))),
        exponent=parameter_properties.ParameterProperties(
            default_constraining_bijector_fn=(
                lambda: softplus.Softplus(low=dtype_util.eps(dtype)))),
        slope_amplitude=parameter_properties.ParameterProperties(
            default_constraining_bijector_fn=(
                lambda: softplus.Softplus(low=dtype_util.eps(dtype)))),
        shift=parameter_properties.ParameterProperties())

  @property
  def bias_amplitude(self):
    """Stddev on bias parameter."""
    return self._bias_amplitude

  @property
  def slope_amplitude(self):
    """Amplitude on slope parameter."""
    return self._slope_amplitude

  @property
  def shift(self):
    """Shift of linear function that is exponentiated."""
    return self._shift

  @property
  def exponent(self):
    """Exponent of the polynomial term."""
    return self._exponent

  def _matrix(self, x1, x2):
    if self.feature_ndims > 1:
      x1 = tf.reshape(x1, ps.concat(
          [ps.shape(x1)[:-self.feature_ndims], [-1]], axis=-1))
      x2 = tf.reshape(x2, ps.concat(
          [ps.shape(x2)[:-self.feature_ndims], [-1]], axis=-1))
    if self.shift is not None:
      if self.feature_ndims > 0:
        shift = self.shift[..., tf.newaxis, tf.newaxis]
      else:
        shift = self.shift[..., tf.newaxis]
      x1 = x1 - shift
      x2 = x2 - shift
    if self.feature_ndims > 0:
      dot_prod = tf.linalg.matmul(x1, x2, transpose_b=True)
    else:
      dot_prod = tf.linalg.einsum('...i,...j->...ij', x1, x2)

    if self.exponent is not None:
      exponent = tf.convert_to_tensor(self.exponent)[
          ..., tf.newaxis, tf.newaxis]
      dot_prod = dot_prod ** exponent

    if self.slope_amplitude is not None:
      slope_amplitude = tf.convert_to_tensor(self.slope_amplitude)[
          ..., tf.newaxis, tf.newaxis]
      dot_prod = dot_prod * slope_amplitude**2.

    if self.bias_amplitude is not None:
      bias_amplitude = tf.convert_to_tensor(self.bias_amplitude)[
          ..., tf.newaxis, tf.newaxis]
      dot_prod = dot_prod + bias_amplitude**2.

    return dot_prod

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
      dot_prod = dot_prod ** exponent

    slope_amplitude = self.slope_amplitude
    if slope_amplitude is not None:
      slope_amplitude = tf.convert_to_tensor(slope_amplitude)
      slope_amplitude = util.pad_shape_with_ones(slope_amplitude, example_ndims)
      dot_prod = dot_prod * slope_amplitude**2.

    bias_amplitude = self.bias_amplitude
    if bias_amplitude is not None:
      bias_amplitude = tf.convert_to_tensor(bias_amplitude)
      bias_amplitude = util.pad_shape_with_ones(bias_amplitude, example_ndims)
      dot_prod = dot_prod + bias_amplitude**2.

    return dot_prod

  def _parameter_control_dependencies(self, is_init):
    if not self.validate_args:
      return []
    assertions = []
    ok_to_check = lambda x: (  # pylint:disable=g-long-lambda
        x is not None) and (is_init != tensor_util.is_ref(x))

    bias_amplitude = self.bias_amplitude
    slope_amplitude = self.slope_amplitude

    if ok_to_check(self.exponent):
      exponent = tf.convert_to_tensor(self.exponent)
      assertions.append(
          assert_util.assert_positive(
              exponent, message='`exponent` must be positive.'))
      from tensorflow_probability.python.internal import distribution_util  # pylint: disable=g-import-not-at-top
      assertions.append(
          distribution_util.assert_integer_form(
              exponent, message='`exponent` must be an integer.'))
    if ok_to_check(bias_amplitude):
      bias_amplitude = tf.convert_to_tensor(bias_amplitude)
      assertions.append(
          assert_util.assert_non_negative(
              bias_amplitude, message='`bias_amplitude` must be non-negative.'))
    if ok_to_check(slope_amplitude):
      slope_amplitude = tf.convert_to_tensor(slope_amplitude)
      assertions.append(
          assert_util.assert_non_negative(
              slope_amplitude,
              message='`slope_amplitude` must be non-negative.'))

    if (ok_to_check(self.bias_amplitude) and ok_to_check(self.slope_amplitude)):
      assertions.append(
          assert_util.assert_positive(
              tf.math.abs(slope_amplitude) + tf.math.abs(bias_amplitude),
              message=('`slope_amplitude` and `bias_amplitude` '
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
    k(x, y) = bias_amplitude**2 + slope_amplitude**2 *
              ((x - shift) dot (y - shift))
    ```
  """

  def __init__(self,
               bias_amplitude=None,
               slope_amplitude=None,
               shift=None,
               feature_ndims=1,
               validate_args=False,
               parameters=None,
               name='Linear'):
    """Construct a Linear kernel instance.

    Args:
      bias_amplitude: Non-negative floating point `Tensor` that controls the
        stddev from the origin. If bias = 0, there is no stddev and the
        fitted function goes through the origin.  Must be broadcastable with
        `slope_amplitude`, `shift`, `exponent`, and inputs to `apply` and
        `matrix` methods. A value of `None` is treated like 0.
        Default Value: `None`
      slope_amplitude: Non-negative floating point `Tensor` that controls the
        stddev of the regression line slope that is the basis for the
        polynomial. Must be broadcastable with `bias_amplitude`, `shift`,
        `exponent`, and inputs to `apply` and `matrix` methods. A value of
        `None` is treated like 1.
        Default Value: `None`
      shift: Floating point `Tensor` that controls the intercept with the x-axis
        of the linear interpolation. Must be broadcastable with
        `bias_amplitude`, `slope_amplitude`, and inputs to `apply` and `matrix`
        methods. A value of `None` is treated like 0, which results in having
        the intercept at the origin.
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
        bias_amplitude=bias_amplitude,
        slope_amplitude=slope_amplitude,
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
        bias_amplitude=parameter_properties.ParameterProperties(
            default_constraining_bijector_fn=(
                lambda: softplus.Softplus(low=dtype_util.eps(dtype)))),
        slope_amplitude=parameter_properties.ParameterProperties(
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
          bias_amplitude=tfp_util.DeferredTensor(self._constant, tf.math.sqrt),
          slope_amplitude=0.0,
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
