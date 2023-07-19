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
"""SinhArcsinh bijector."""

import tensorflow.compat.v2 as tf

from tensorflow_probability.python.bijectors import bijector
from tensorflow_probability.python.bijectors import softplus as softplus_bijector
from tensorflow_probability.python.internal import assert_util
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import parameter_properties
from tensorflow_probability.python.internal import tensor_util
from tensorflow_probability.python.math import generic
from tensorflow_probability.python.math import numeric

__all__ = [
    'SinhArcsinh',
]


class SinhArcsinh(
    bijector.CoordinatewiseBijectorMixin,
    bijector.AutoCompositeTensorBijector):
  """`Y = g(X) = Sinh( (Arcsinh(X) + skewness) * tailweight ) * multiplier`.

  For `skewness in (-inf, inf)` and `tailweight in (0, inf)`, this
  transformation is a
  diffeomorphism of the real line `(-inf, inf)`.  The inverse transform is
  `X = g^{-1}(Y) = Sinh( ArcSinh(Y) / tailweight - skewness )`.

  The `SinhArcsinh` transformation of the Normal is described in
  [Sinh-arcsinh distributions](https://www.jstor.org/stable/27798865)
  This Bijector allows a similar transformation of any distribution supported on
  `(-inf, inf)`.

  #### Meaning of the parameters

  * If `skewness = 0` and `tailweight = 1`, this transform is the identity.
  * Positive (negative) `skewness` leads to positive (negative) skew.
    * positive skew means, for unimodal `X` centered at zero, the mode of `Y` is
      "tilted" to the right.
    * positive skew means positive values of `Y` become more likely, and
      negative values become less likely.
  * Larger (smaller) `tailweight` leads to fatter (thinner) tails.
    * Fatter tails mean larger values of `|Y|` become more likely.
    * If `X` is a unit Normal, `tailweight < 1` leads to a distribution that is
      "flat" around `Y = 0`, and a very steep drop-off in the tails.
    * If `X` is a unit Normal, `tailweight > 1` leads to a distribution more
      peaked at the mode with heavier tails.
  * The `multiplier` term is equal to 2 / F_0(2) where F_0 is
    the bijector with `skewness = 0`. This is important
    for CDF values of distributions.SinhArcsinh.

  To see the argument about the tails, note that for `|X| >> 1` and
  `|X| >> (|skewness| * tailweight)**tailweight`, we have
  `Y approx 0.5 X**tailweight e**(sign(X) skewness * tailweight)`.
  """

  def __init__(self,
               skewness=None,
               tailweight=None,
               validate_args=False,
               name='sinh_arcsinh'):
    """Instantiates the `SinhArcsinh` bijector.

    Args:
      skewness:  Skewness parameter.  Float-type `Tensor`.  Default is `0`
        of type `float32`.
      tailweight:  Tailweight parameter.  Positive `Tensor` of same `dtype` as
        `skewness` and broadcastable `shape`.  Default is `1` of type `float32`.
      validate_args: Python `bool` indicating whether arguments should be
        checked for correctness.
      name: Python `str` name given to ops managed by this object.
    """
    parameters = dict(locals())
    with tf.name_scope(name) as name:
      tailweight = 1. if tailweight is None else tailweight
      skewness = 0. if skewness is None else skewness
      dtype = dtype_util.common_dtype([tailweight, skewness],
                                      dtype_hint=tf.float32)
      self._skewness = tensor_util.convert_nonref_to_tensor(
          skewness, dtype=dtype, name='skewness')
      self._tailweight = tensor_util.convert_nonref_to_tensor(
          tailweight, dtype=dtype, name='tailweight')
      self._scale_number = tf.convert_to_tensor(2., dtype=dtype)
      super(SinhArcsinh, self).__init__(
          forward_min_event_ndims=0,
          validate_args=validate_args,
          parameters=parameters,
          name=name)

  @classmethod
  def _parameter_properties(cls, dtype):
    return dict(
        skewness=parameter_properties.ParameterProperties(),
        tailweight=parameter_properties.ParameterProperties(
            default_constraining_bijector_fn=(
                lambda: softplus_bijector.Softplus(low=dtype_util.eps(dtype)))))

  @property
  def skewness(self):
    """The `skewness` in: `Y  = Sinh((Arcsinh(X) + skewness) * tailweight)`."""
    return self._skewness

  @property
  def tailweight(self):
    """The `tailweight` in: `Y = Sinh((Arcsinh(X) + skewness) * tailweight)`."""
    return self._tailweight

  def _output_multiplier(self, tailweight):
    return self._scale_number / tf.sinh(
        tf.asinh(self._scale_number) * tailweight)

  @classmethod
  def _is_increasing(cls):
    return True

  def _forward(self, x):
    tailweight = tf.convert_to_tensor(self.tailweight)
    multiplier = self._output_multiplier(tailweight)
    bijector_output = tf.sinh((tf.asinh(x) + self.skewness) * tailweight)
    return bijector_output * multiplier

  def _inverse(self, y):
    tailweight = tf.convert_to_tensor(self.tailweight)
    multiplier = self._output_multiplier(tailweight)
    return tf.sinh(tf.asinh(y / multiplier) / tailweight - self.skewness)

  def _inverse_log_det_jacobian(self, y):
    # x = sinh(arcsinh(y / multiplier) / tailweight - skewness)
    # Using sinh' = cosh, arcsinh'(y) = 1 / sqrt(y**2 + 1),
    # dx/dy
    # = cosh(arcsinh(y / multiplier) / tailweight - skewness)
    #     / (tailweight * sqrt((y / multiplier)**2 + 1)) / multiplier

    tailweight = tf.convert_to_tensor(self.tailweight)
    multiplier = self._output_multiplier(tailweight)
    y = y / multiplier

    return (generic.log_cosh(tf.asinh(y) / tailweight - self.skewness) -
            0.5 * numeric.log1psquare(y) -
            tf.math.log(tailweight) - tf.math.log(multiplier))

  def _forward_log_det_jacobian(self, x):
    # y = sinh((arcsinh(x) + skewness) * tailweight) * multiplier
    # Using sinh' = cosh, arcsinh'(x) = 1 / sqrt(x**2 + 1),
    # dy/dx
    # = cosh((arcsinh(x) + skewness) * tailweight) * tailweight / sqrt(x**2 + 1)
    # * multiplier

    tailweight = tf.convert_to_tensor(self.tailweight)

    return (generic.log_cosh((tf.asinh(x) + self.skewness) * tailweight) -
            0.5 * numeric.log1psquare(x) +
            tf.math.log(tailweight) +
            tf.math.log(self._output_multiplier(tailweight)))

  def _parameter_control_dependencies(self, is_init):
    if not self.validate_args:
      return []
    assertions = []
    if is_init != tensor_util.is_ref(self.tailweight):
      assertions.append(assert_util.assert_positive(
          self.tailweight,
          message='Argument `tailweight` must be positive.'))
    return assertions
