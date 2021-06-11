# Copyright 2019 The TensorFlow Probability Authors.
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
"""The GeneralizedPareto bijector."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v2 as tf
from tensorflow_probability.python.bijectors import bijector as bijector_lib
from tensorflow_probability.python.bijectors import chain as chain_bijector
from tensorflow_probability.python.bijectors import shift as shift_bijector
from tensorflow_probability.python.bijectors import sigmoid as sigmoid_bijector
from tensorflow_probability.python.bijectors import softplus as softplus_bijector
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import parameter_properties
from tensorflow_probability.python.internal import tensor_util


__all__ = [
    'GeneralizedPareto',
]


class GeneralizedPareto(bijector_lib.AutoCompositeTensorBijector):
  """Bijector mapping R**n to non-negative reals.

  Forward computation maps R**n to the support of the `GeneralizedPareto`
  distribution with parameters `loc`, `scale`, and `concentration`.

  #### Mathematical Details

  The forward computation from `y` in R**n to `x` constrains `x` as follows:

  `x >= loc`                                             if `concentration >= 0`
  `x >= loc` and `x <= loc + scale / abs(concentration)` if `concentration < 0`

  This bijector is used as the `experimental_default_event_space_bijector` of
  the `GeneralizedPareto` distribution.

  """

  def __init__(self,
               loc,
               scale,
               concentration,
               validate_args=False,
               name='generalized_pareto'):
    parameters = dict(locals())
    with tf.name_scope(name) as name:
      dtype = dtype_util.common_dtype(
          [loc, scale, concentration], dtype_hint=tf.float32)

      self._loc = tensor_util.convert_nonref_to_tensor(loc)
      self._scale = tensor_util.convert_nonref_to_tensor(scale)
      self._concentration = tensor_util.convert_nonref_to_tensor(concentration)
      self._non_negative_concentration_bijector = chain_bijector.Chain([
          shift_bijector.Shift(shift=self._loc, validate_args=validate_args),
          softplus_bijector.Softplus(validate_args=validate_args)
      ], validate_args=validate_args)
      super(GeneralizedPareto, self).__init__(
          validate_args=validate_args,
          forward_min_event_ndims=0,
          dtype=dtype,
          parameters=parameters,
          name=name)

  @classmethod
  def _parameter_properties(cls, dtype):
    return dict(
        loc=parameter_properties.ParameterProperties(),
        scale=parameter_properties.ParameterProperties(
            default_constraining_bijector_fn=(
                lambda: softplus_bijector.Softplus(low=dtype_util.eps(dtype)))),
        concentration=parameter_properties.ParameterProperties(
            default_constraining_bijector_fn=(
                lambda: softplus_bijector.Softplus(low=dtype_util.eps(dtype)))))

  def _is_increasing(self):
    return True

  @property
  def loc(self):
    return self._loc

  @property
  def scale(self):
    return self._scale

  @property
  def concentration(self):
    return self._concentration

  def _negative_concentration_bijector(self):
    # Constructed dynamically so that `loc + scale / concentration` is
    # tape-safe.
    loc = tf.convert_to_tensor(self.loc)
    high = loc + tf.math.abs(self.scale / self.concentration)
    return sigmoid_bijector.Sigmoid(
        low=loc, high=high, validate_args=self.validate_args)

  def _forward(self, x):
    return tf.where(self._concentration < 0.,
                    self._negative_concentration_bijector().forward(x),
                    self._non_negative_concentration_bijector.forward(x))

  def _inverse(self, y):
    return tf.where(self._concentration < 0.,
                    self._negative_concentration_bijector().inverse(y),
                    self._non_negative_concentration_bijector.inverse(y))

  def _forward_log_det_jacobian(self, x):
    event_ndims = self.forward_min_event_ndims
    return tf.where(
        self._concentration < 0.,
        self._negative_concentration_bijector().forward_log_det_jacobian(
            x, event_ndims=event_ndims),
        self._non_negative_concentration_bijector.forward_log_det_jacobian(
            x, event_ndims=event_ndims))

  def _inverse_log_det_jacobian(self, y):
    event_ndims = self.inverse_min_event_ndims
    return tf.where(
        self._concentration < 0.,
        self._negative_concentration_bijector().inverse_log_det_jacobian(
            y, event_ndims=event_ndims),
        self._non_negative_concentration_bijector.inverse_log_det_jacobian(
            y, event_ndims=event_ndims))
