# Copyright 2022 The TensorFlow Probability Authors.
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
"""ExponentialCurve kernel."""

import tensorflow.compat.v2 as tf

from tensorflow_probability.python.internal import assert_util
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import parameter_properties
from tensorflow_probability.python.internal import tensor_util
from tensorflow_probability.python.math.psd_kernels import positive_semidefinite_kernel as psd_kernel
from tensorflow_probability.python.math.psd_kernels.internal import util


__all__ = ['ExponentialCurve']


class ExponentialCurve(psd_kernel.AutoCompositeTensorPsdKernel):
  """Exponential Curve Kernel.

  This kernel is defined as over positive real numbers as:

  ```none
  k(x, y) = int_0^inf exp(-s * x) exp(-s * y) GammaPDF(c, r)
  ```
  where:
    * `GammaPDF` is the pdf of the Gamma distribution.
    * `c` is the concentration parameter of the Gamma distribution.
    * `r` is the rate parameter of the Gamma distribution.

  See equation 6 in https://arxiv.org/pdf/1406.3896.pdf

  When parameterizing a 1-dimensional Gaussian Process, this kernel produces
  samples that strongly favor exponential functions.

  When called on multiple feature dimensions, all feature dimensions are added
  together before the kernel function is applied.
  """

  def __init__(
      self,
      concentration,
      rate,
      feature_ndims=1,
      validate_args=False,
      name='ExponentialCurve'):
    # These are the concentration and rate parameters of the Gamma distribution.
    # The kernel is defined as the integral of exp(-s (t + t')) over s drawn
    # from a gamma distribution.
    parameters = dict(locals())
    dtype = util.maybe_get_common_dtype([concentration, rate])
    self._concentration = tensor_util.convert_nonref_to_tensor(
        concentration, name='concentration', dtype=dtype)
    self._rate = tensor_util.convert_nonref_to_tensor(
        rate, name='rate', dtype=dtype)
    super(ExponentialCurve, self).__init__(
        feature_ndims=feature_ndims,
        dtype=dtype,
        name=name,
        validate_args=validate_args,
        parameters=parameters)

  @property
  def concentration(self):
    """Concentration parameter."""
    return self._concentration

  @property
  def rate(self):
    """Rate parameter."""
    return self._rate

  @classmethod
  def _parameter_properties(cls, dtype, num_classes=None):
    from tensorflow_probability.python.bijectors import softplus as softplus_bijector  # pylint:disable=g-import-not-at-top
    return dict(
        concentration=parameter_properties.ParameterProperties(
            default_constraining_bijector_fn=(
                softplus_bijector.Softplus(low=dtype_util.eps(dtype)))),
        rate=parameter_properties.ParameterProperties(
            default_constraining_bijector_fn=(
                softplus_bijector.Softplus(low=dtype_util.eps(dtype)))))

  def _apply(self, x1, x2, example_ndims=0):
    a = tf.convert_to_tensor(self._concentration)
    b = tf.convert_to_tensor(self._rate)
    a = util.pad_shape_with_ones(a, ndims=example_ndims)
    b = util.pad_shape_with_ones(b, ndims=example_ndims)
    # The kernel is defined for scalars where t >= 0.
    # TODO(jburnim,srvasude): Raise or return NaN when `any(x1 < 0 | x2 < 0)`?
    sum_x1_x2 = util.sum_rightmost_ndims_preserving_shape(
        x1 + x2, ndims=self.feature_ndims)
    log_result = tf.math.xlogy(a, b) - tf.math.xlogy(a, sum_x1_x2 + b)
    return tf.math.exp(log_result)

  def _parameter_control_dependencies(self, is_init):
    """Control dependencies for parameters."""
    if not self.validate_args:
      return []
    assertions = []
    for arg_name, arg in dict(concentration=self.concentration,
                              rate=self.rate).items():
      if arg is not None and is_init != tensor_util.is_ref(arg):
        assertions.append(assert_util.assert_positive(
            arg,
            message='{} must be positive.'.format(arg_name)))
    return assertions
