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
"""Rational Quadratic kernel."""

import tensorflow.compat.v2 as tf

from tensorflow_probability.python.internal import assert_util
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import parameter_properties
from tensorflow_probability.python.internal import tensor_util
from tensorflow_probability.python.math.psd_kernels import positive_semidefinite_kernel as psd_kernel
from tensorflow_probability.python.math.psd_kernels.internal import util

__all__ = ['RationalQuadratic']


class RationalQuadratic(psd_kernel.AutoCompositeTensorPsdKernel):
  """RationalQuadratic Kernel.

  This kernel function has the form:

  ```none
  k(x, y) = amplitude**2 * (1. + ||x - y|| ** 2 / (
    2 * scale_mixture_rate * length_scale**2)) ** -scale_mixture_rate
  ```

  where the double-bars represent vector length (i.e. Euclidean, or L2 Norm).
  This kernel acts over the space `S = R^(D1 x D2 .. Dd)`. When
  `scale_mixture_rate` tends towards infinity, this kernel acts like an
  ExponentiatedQuadratic kernel.

  The name `scale_mixture_rate` comes from the interpretation that that this
  kernel is an Inverse-Gamma weighted mixture of ExponentiatedQuadratic Kernels
  with different length scales.

  More formally, if `r = ||x - y|||`, then:

  ```none
  integral from 0 to infinity (k_EQ(r | sqrt(t)) p(t | a, a * l ** 2)) dt =
    (1 + r**2 / (2 * a * l ** 2)) ** -a = k(r)
  ```

  where:
    * `a = scale_mixture_rate`
    * `l = length_scale`
    * `p(t | a, b)` is the Inverse-Gamma density.
      https://en.wikipedia.org/wiki/Inverse-gamma_distribution
    * `k_EQ(r | l) = exp(-r ** 2 / (2 * l ** 2)` is the ExponentiatedQuadratic
      positive semidefinite kernel.

  #### References

  [1]: Filip Tronarp, Toni Karvonen, and Simo Saarka. Mixture
       representation of the Matern class with applications in state space
       approximations and Bayesian quadrature. In _28th IEEE International
       Workshop on Machine Learning for Signal Processing_, 2018.
       https://acris.aalto.fi/ws/portalfiles/portal/30548539/ELEC_Tronarp_etal_Mixture_Presentation_of_the_Matern_MLSP2018.pdf
  """

  def __init__(
      self,
      amplitude=None,
      length_scale=None,
      inverse_length_scale=None,
      scale_mixture_rate=None,
      feature_ndims=1,
      validate_args=False,
      name='RationalQuadratic'):
    """Construct a RationalQuadratic kernel instance.

    Args:
      amplitude: Positive floating point `Tensor` that controls the maximum
        value of the kernel. Must be broadcastable with `length_scale` and
        `scale_mixture_rate` and inputs to `apply` and `matrix` methods. A
        value of `None` is treated like 1.
        Default value: None
      length_scale: Positive floating point `Tensor` that controls how sharp or
        wide the kernel shape is. This provides a characteristic "unit" of
        length against which `||x - y||` can be compared for scale. Must be
        broadcastable with `amplitude`, `scale_mixture_rate`  and inputs to
        `apply` and `matrix` methods. A value of `None` is treated like 1.
        Default value: None
      inverse_length_scale: Non-negative floating point `Tensor` that is
        treated as `1 / length_scale`. Only one of `length_scale` or
        `inverse_length_scale` should be provided.
        Default value: None
      scale_mixture_rate: Positive floating point `Tensor` that controls how the
        ExponentiatedQuadratic kernels are mixed.  Must be broadcastable with
        `amplitude`, `length_scale` and inputs to `apply` and `matrix` methods.
        A value of `None` is treated like 1.
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
    with tf.name_scope(name) as name:
      dtype = util.maybe_get_common_dtype(
          [amplitude, scale_mixture_rate, inverse_length_scale, length_scale])

      self._amplitude = tensor_util.convert_nonref_to_tensor(
          amplitude, name='amplitude', dtype=dtype)
      self._scale_mixture_rate = tensor_util.convert_nonref_to_tensor(
          scale_mixture_rate, name='scale_mixture_rate', dtype=dtype)
      self._length_scale = tensor_util.convert_nonref_to_tensor(
          length_scale, name='length_scale', dtype=dtype)
      self._inverse_length_scale = tensor_util.convert_nonref_to_tensor(
          inverse_length_scale, name='inverse_length_scale', dtype=dtype)

      super(RationalQuadratic, self).__init__(
          feature_ndims,
          dtype=dtype,
          name=name,
          validate_args=validate_args,
          parameters=parameters)

  def _apply_with_distance(
      self, x1, x2, pairwise_square_distance, example_ndims=0):

    pairwise_square_distance /= 2
    inverse_length_scale = self._inverse_length_scale_parameter()
    if inverse_length_scale is not None:
      inverse_length_scale = util.pad_shape_with_ones(
          inverse_length_scale, example_ndims)
      pairwise_square_distance = pairwise_square_distance * tf.math.square(
          inverse_length_scale)

    if self.scale_mixture_rate is None:
      power = 1.
    else:
      scale_mixture_rate = tf.convert_to_tensor(self.scale_mixture_rate)
      power = util.pad_shape_with_ones(
          scale_mixture_rate, ndims=example_ndims)
      pairwise_square_distance = pairwise_square_distance / power

    log_result = -power * tf.math.log1p(pairwise_square_distance)

    if self.amplitude is not None:
      amplitude = tf.convert_to_tensor(self.amplitude)
      amplitude = util.pad_shape_with_ones(amplitude, ndims=example_ndims)
      log_result = log_result + 2 * tf.math.log(amplitude)
    return tf.math.exp(log_result)

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
  def scale_mixture_rate(self):
    """scale_mixture_rate parameter."""
    return self._scale_mixture_rate

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
            default_constraining_bijector_fn=softplus.Softplus),
        scale_mixture_rate=parameter_properties.ParameterProperties(
            default_constraining_bijector_fn=(
                lambda: softplus.Softplus(low=dtype_util.eps(dtype)))))

  def _parameter_control_dependencies(self, is_init):
    if not self.validate_args:
      return []
    assertions = []
    if (self._inverse_length_scale is not None and
        is_init != tensor_util.is_ref(self._inverse_length_scale)):
      assertions.append(assert_util.assert_non_negative(
          self._inverse_length_scale,
          message='`inverse_length_scale` must be non-negative.'))
    for arg_name, arg in dict(
        amplitude=self.amplitude,
        length_scale=self.length_scale,
        scale_mixture_rate=self.scale_mixture_rate).items():
      if arg is not None and is_init != tensor_util.is_ref(arg):
        assertions.append(assert_util.assert_positive(
            arg, message=f'{arg_name} must be positive.'))
    return assertions
