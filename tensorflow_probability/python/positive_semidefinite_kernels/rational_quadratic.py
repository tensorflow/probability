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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

import tensorflow as tf
from tensorflow_probability.python.positive_semidefinite_kernels import positive_semidefinite_kernel as psd_kernel
from tensorflow_probability.python.positive_semidefinite_kernels.internal import util

__all__ = [
    "RationalQuadratic",
]


def _validate_arg_if_not_none(arg, assertion, validate_args):
  if arg is None:
    return arg
  with tf.control_dependencies([assertion(arg)] if validate_args else []):
    result = tf.identity(arg)
  return result


class RationalQuadratic(psd_kernel.PositiveSemidefiniteKernel):
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
       https://users.aalto.fi/~karvont2/pdf/TronarpKarvonenSarkka2018-MLSP.pdf
  """

  def __init__(
      self,
      amplitude=None,
      length_scale=None,
      scale_mixture_rate=None,
      feature_ndims=1,
      validate_args=False,
      name="RationalQuadratic"):
    """Construct a RationalQuadratic kernel instance.

    Args:
      amplitude: Positive floating point `Tensor` that controls the maximum
        value of the kernel. Must be broadcastable with `length_scale` and
        `scale_mixture_rate` and inputs to `apply` and `matrix` methods.
      length_scale: Positive floating point `Tensor` that controls how sharp or
        wide the kernel shape is. This provides a characteristic "unit" of
        length against which `||x - y||` can be compared for scale. Must be
        broadcastable with `amplitude`, `scale_mixture_rate`  and inputs to
        `apply` and `matrix` methods.
      scale_mixture_rate: Positive floating point `Tensor` that controls how the
        ExponentiatedQuadratic kernels are mixed.  Must be broadcastable with
        `amplitude`, `length_scale` and inputs to `apply` and `matrix` methods.
      feature_ndims: Python `int` number of rightmost dims to include in the
        squared difference norm in the exponential.
      validate_args: If `True`, parameters are checked for validity despite
        possibly degrading runtime performance
      name: Python `str` name prefixed to Ops created by this class.
    """
    with tf.compat.v1.name_scope(
        name, values=[amplitude, scale_mixture_rate, length_scale]) as name:
      dtype = util.maybe_get_common_dtype(
          [amplitude, scale_mixture_rate, length_scale])

      if amplitude is not None:
        amplitude = tf.convert_to_tensor(
            value=amplitude, name="amplitude", dtype=dtype)
      self._amplitude = _validate_arg_if_not_none(
          amplitude, tf.compat.v1.assert_positive, validate_args)

      if scale_mixture_rate is not None:
        scale_mixture_rate = tf.convert_to_tensor(
            value=scale_mixture_rate, name="scale_mixture_rate", dtype=dtype)
      self._scale_mixture_rate = _validate_arg_if_not_none(
          scale_mixture_rate, tf.compat.v1.assert_positive, validate_args)

      if length_scale is not None:
        length_scale = tf.convert_to_tensor(
            value=length_scale, name="length_scale", dtype=dtype)
      self._length_scale = _validate_arg_if_not_none(
          length_scale, tf.compat.v1.assert_positive, validate_args)

    super(RationalQuadratic, self).__init__(
        feature_ndims, dtype=dtype, name=name)

  def _apply(self, x1, x2, example_ndims=0):
    difference = util.sum_rightmost_ndims_preserving_shape(
        tf.math.squared_difference(x1, x2), ndims=self.feature_ndims)
    difference /= 2

    if self.length_scale is not None:
      length_scale = util.pad_shape_with_ones(
          self.length_scale, ndims=example_ndims)
      difference /= length_scale ** 2

    scale_mixture_rate = 1.
    if self.scale_mixture_rate is not None:
      scale_mixture_rate = util.pad_shape_with_ones(
          self.scale_mixture_rate, ndims=example_ndims)
      difference /= scale_mixture_rate

    result = (1. + difference) ** -scale_mixture_rate

    if self.amplitude is not None:
      amplitude = util.pad_shape_with_ones(
          self.amplitude, ndims=example_ndims)
      result *= amplitude ** 2
    return result

  @property
  def amplitude(self):
    """Amplitude parameter."""
    return self._amplitude

  @property
  def length_scale(self):
    """Length scale parameter."""
    return self._length_scale

  @property
  def scale_mixture_rate(self):
    """scale_mixture_rate parameter."""
    return self._scale_mixture_rate

  def _batch_shape(self):
    shape_list = [
        x.shape for x in [  # pylint: disable=g-complex-comprehension
            self.amplitude,
            self.scale_mixture_rate,
            self.length_scale
        ] if x is not None
    ]
    if not shape_list:
      return tf.TensorShape([])
    return functools.reduce(tf.broadcast_static_shape, shape_list)

  def _batch_shape_tensor(self):
    shape_list = [
        tf.shape(input=x)
        for x in [self.amplitude, self.scale_mixture_rate, self.length_scale]
        if x is not None
    ]
    if not shape_list:
      return tf.constant([], dtype=tf.int32)
    return functools.reduce(tf.broadcast_dynamic_shape, shape_list)
