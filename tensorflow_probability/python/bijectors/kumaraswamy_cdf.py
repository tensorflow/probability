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
"""KumaraswamyCDF bijector."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v2 as tf

from tensorflow_probability.python.bijectors import bijector
from tensorflow_probability.python.bijectors import softplus as softplus_bijector
from tensorflow_probability.python.internal import assert_util
from tensorflow_probability.python.internal import auto_composite_tensor
from tensorflow_probability.python.internal import distribution_util
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import parameter_properties
from tensorflow_probability.python.internal import tensor_util


__all__ = [
    'KumaraswamyCDF',
]


@auto_composite_tensor.auto_composite_tensor(
    omit_kwargs=('name',), module_name='tfp.bijectors')
class KumaraswamyCDF(bijector.AutoCompositeTensorBijector):
  """Compute `Y = g(X) = (1 - X**a)**b, X in [0, 1]`.

  This bijector maps inputs from `[0, 1]` to `[0, 1]`. The inverse of the
  bijector applied to a uniform random variable `X ~ U(0, 1)` gives back a
  random variable with the [Kumaraswamy distribution](
  https://en.wikipedia.org/wiki/Kumaraswamy_distribution):

  ```none
  Y ~ Kumaraswamy(a, b)
  pdf(y; a, b, 0 <= y <= 1) = a * b * y ** (a - 1) * (1 - y**a) ** (b - 1)
  ```
  """

  def __init__(self,
               concentration1=1.,
               concentration0=1.,
               validate_args=False,
               name='kumaraswamy_cdf'):
    """Instantiates the `KumaraswamyCDF` bijector.

    Args:
      concentration1: Python `float` scalar indicating the transform power,
        i.e., `Y = g(X) = (1 - (1 - X)**(1 / b))**(1 / a)` where `a` is
        `concentration1`.
      concentration0: Python `float` scalar indicating the transform power,
        i.e., `Y = g(X) = (1 - (1 - X)**(1 / b))**(1 / a)` where `b` is
        `concentration0`.
      validate_args: Python `bool` indicating whether arguments should be
        checked for correctness.
      name: Python `str` name given to ops managed by this object.
    """
    parameters = dict(locals())
    with tf.name_scope(name) as name:
      dtype = dtype_util.common_dtype([concentration0, concentration1],
                                      dtype_hint=tf.float32)
      self._concentration0 = tensor_util.convert_nonref_to_tensor(
          concentration0, dtype=dtype, name='concentration0')
      self._concentration1 = tensor_util.convert_nonref_to_tensor(
          concentration1, dtype=dtype, name='concentration1')
      super(KumaraswamyCDF, self).__init__(
          forward_min_event_ndims=0,
          validate_args=validate_args,
          parameters=parameters,
          name=name)

  @classmethod
  def _parameter_properties(cls, dtype):
    return dict(
        concentration0=parameter_properties.ParameterProperties(
            default_constraining_bijector_fn=(
                lambda: softplus_bijector.Softplus(low=dtype_util.eps(dtype)))),
        concentration1=parameter_properties.ParameterProperties(
            default_constraining_bijector_fn=(
                lambda: softplus_bijector.Softplus(low=dtype_util.eps(dtype)))))

  @property
  def concentration1(self):
    """The `a` in: `Y = g(X) = (1 - (1 - X)**(1 / b))**(1 / a)`."""
    return self._concentration1

  @property
  def concentration0(self):
    """The `b` in: `Y = g(X) = (1 - (1 - X)**(1 / b))**(1 / a)`."""
    return self._concentration0

  @classmethod
  def _is_increasing(cls):
    return True

  def _inverse(self, y):
    y = self._maybe_assert_valid(y)
    return tf.exp(
        tf.math.log1p(-tf.exp(tf.math.log1p(-y) / self.concentration0)) /
        self.concentration1)

  def _forward(self, x):
    x = self._maybe_assert_valid(x)
    return -tf.math.expm1(self.concentration0 * tf.math.log1p(
        -(x ** self.concentration1)))

  def _forward_log_det_jacobian(self, x):
    x = self._maybe_assert_valid(x)
    concentration1 = tf.convert_to_tensor(self.concentration1)
    concentration0 = tf.convert_to_tensor(self.concentration0)
    return (tf.math.log(concentration1) +
            tf.math.log(concentration0) +
            tf.math.xlogy(concentration1 - 1, x) +
            tf.math.xlog1py(concentration0 - 1, -x**concentration1))

  def _maybe_assert_valid(self, x):
    if not self.validate_args:
      return x
    return distribution_util.with_dependencies([
        assert_util.assert_non_negative(
            x, message='Sample must be non-negative.'),
        assert_util.assert_less_equal(
            x,
            tf.ones([], self.concentration0.dtype),
            message='Sample must be less than or equal to `1`.'),
    ], x)

  def _parameter_control_dependencies(self, is_init):
    if not self.validate_args:
      return []
    assertions = []
    if is_init != tensor_util.is_ref(self.concentration0):
      assertions.append(assert_util.assert_positive(
          self.concentration0,
          message='Argument `concentration0` must be positive.'))
    if is_init != tensor_util.is_ref(self.concentration1):
      assertions.append(assert_util.assert_positive(
          self.concentration1,
          message='Argument `concentration1` must be positive.'))
    return assertions
