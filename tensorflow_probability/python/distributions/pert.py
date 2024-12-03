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
"""The PERT distribution."""

import tensorflow.compat.v2 as tf

from tensorflow_probability.python.bijectors import chain as chain_bijector
from tensorflow_probability.python.bijectors import scale as scale_bijector
from tensorflow_probability.python.bijectors import shift as shift_bijector
from tensorflow_probability.python.bijectors import sigmoid as sigmoid_bijector
from tensorflow_probability.python.bijectors import softplus as softplus_bijector
from tensorflow_probability.python.distributions import beta
from tensorflow_probability.python.distributions import distribution
from tensorflow_probability.python.distributions import transformed_distribution
from tensorflow_probability.python.internal import assert_util
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import parameter_properties
from tensorflow_probability.python.internal import prefer_static as ps
from tensorflow_probability.python.internal import reparameterization
from tensorflow_probability.python.internal import tensor_util

__all__ = [
    'PERT',
]


class PERT(distribution.AutoCompositeTensorDistribution):
  """Modified PERT distribution for modeling expert predictions.

  The PERT distribution is a loc-scale family of Beta distributions
  fit onto a real interval between `low` and `high` values set by the user,
  along with a `peak` to indicate the expert's most frequent prediction.
  [1](https://en.wikipedia.org/wiki/PERT_distribution), and `temperature` to
  control how sharp the peak is.

  The distribution is similar to a [Triangular distribution]
  (https://en.wikipedia.org/wiki/Triangular_distribution)
  (i.e. `tfd.Triangular`) but with a smooth peak.

  #### Mathematical Details

  In terms of a Beta distribution, PERT can be expressed as

  ```none
  PERT ~ loc + scale * Beta(concentration1, concentration0)
  ```
  where

  ```none
  loc = low
  scale = high - low
                                      peak - low
  concentration1 = 1 + temperature * ------------
                                      high - low

                                      high - peak
  concentration0 = 1 + temperature * ------------
                                      high - low
  temperature > 0
  ```
  The support is `[low, high]`.  The `peak` must fit in that interval:
  `low < peak < high`.  The `temperature` is a positive parameter that
  controls the shape of the distribution. Higher values yield a sharper peak.

  The standard PERT distribution is obtained when `temperature = 4`.

  #### Examples
  ```python
  import tensorflow_probability as tfp
  tfd = tfp.distributions

  # Single PERT distribution
  dist = tfd.PERT(low=1., peak=7., high=11., temperature=4.)
  dist.sample(10)
  dist.prob(7.)
  dist.prob(0.) # Returns nan when the input is outside the support.

  # Multiple PERT distributions with varying temperature (broadcasted)
  dist = tfd.PERT(low=1., peak=7., high=11., temperature=[1., 2., 3., 4.])
  dist.sample(10)
  dist.prob(7.)
  dist.prob([[7.],[5.]])

  # Multiple PERT distributions with varying peak
  dist = tfd.PERT(low=1., peak=[2., 5., 8.], high=11., temperature=4.)
  dist.sample(10)
  dist.sample([10, 5])
  ```
  """

  def __init__(self,
               low,
               peak,
               high,
               temperature=4.,
               validate_args=False,
               allow_nan_stats=False,
               name='PERT'):
    parameters = dict(locals())
    with tf.name_scope(name) as name:
      dtype = dtype_util.common_dtype([low, peak, high, temperature],
                                      tf.float32)
      self._low = tensor_util.convert_nonref_to_tensor(
          low, name='low', dtype=dtype)
      self._peak = tensor_util.convert_nonref_to_tensor(
          peak, name='peak', dtype=dtype)
      self._high = tensor_util.convert_nonref_to_tensor(
          high, name='high', dtype=dtype)
      self._temperature = tensor_util.convert_nonref_to_tensor(
          temperature, name='temperature', dtype=dtype)

      super(PERT, self).__init__(
          validate_args=validate_args,
          allow_nan_stats=allow_nan_stats,
          reparameterization_type=reparameterization.FULLY_REPARAMETERIZED,
          parameters=parameters,
          dtype=dtype,
          name=name)

  def _transformed_beta(self, low=None, peak=None, high=None, temperature=None):
    low = tf.convert_to_tensor(self.low) if low is None else low
    peak = tf.convert_to_tensor(self.peak) if peak is None else peak
    high = tf.convert_to_tensor(self.high) if high is None else high
    temperature = (
        tf.convert_to_tensor(self.temperature)
        if temperature is None else temperature)
    scale = high - low
    concentration1 = (1. + temperature * (peak - low) / scale)
    concentration0 = (1. + temperature * (high - peak) / scale)
    return transformed_distribution.TransformedDistribution(
        distribution=beta.Beta(
            concentration1=concentration1,
            concentration0=concentration0,
            allow_nan_stats=self.allow_nan_stats),
        bijector=chain_bijector.Chain([
            shift_bijector.Shift(shift=low),
            # Broadcasting scale on affine bijector to match batch dimension.
            # This prevents dimension mismatch for operations like cdf.
            # Note that `concentration1` incorporates the broadcast of all four
            # parameters.
            scale_bijector.Scale(
                scale=tf.broadcast_to(
                    scale, ps.shape(concentration1)))]))

  @classmethod
  def _parameter_properties(cls, dtype, num_classes=None):
    # pylint: disable=g-long-lambda
    return dict(
        low=parameter_properties.ParameterProperties(),
        # TODO(b/169874884): Support decoupled parameterization.
        high=parameter_properties.ParameterProperties(
            default_constraining_bijector_fn=parameter_properties
            .BIJECTOR_NOT_IMPLEMENTED,),
        # TODO(b/169874884): Support decoupled parameterization.
        peak=parameter_properties.ParameterProperties(
            default_constraining_bijector_fn=parameter_properties
            .BIJECTOR_NOT_IMPLEMENTED,),
        temperature=parameter_properties.ParameterProperties(
            default_constraining_bijector_fn=(
                lambda: softplus_bijector.Softplus(low=dtype_util.eps(dtype)))))
    # pylint: enable=g-long-lambda

  # Distribution properties
  @property
  def low(self):
    return self._low

  @property
  def peak(self):
    return self._peak

  @property
  def high(self):
    return self._high

  @property
  def temperature(self):
    return self._temperature

  def _event_shape(self):
    return ()

  def _sample_n(self, n, seed=None):
    return self._transformed_beta().sample(n, seed=seed)

  def _log_prob(self, x):
    return self._transformed_beta().log_prob(x)

  def _log_cdf(self, x):
    return self._transformed_beta().log_cdf(x)

  def _entropy(self):
    return self._transformed_beta().entropy()

  def _mean(self):
    return self._transformed_beta().mean()

  def _mode(self):
    return tf.broadcast_to(tf.convert_to_tensor(self.peak),
                           self._batch_shape_tensor())

  def _variance(self):
    low = tf.convert_to_tensor(self.low)
    high = tf.convert_to_tensor(self.high)
    temperature = tf.convert_to_tensor(self.temperature)
    # We capture the tensors here to avoid multiple conversions.
    mean = self._transformed_beta(
        low=low, high=high, temperature=temperature).mean()
    return (mean - low) * (high - mean) / (temperature + 3.)

  def _quantile(self, value):
    return self._transformed_beta().quantile(value)

  def _default_event_space_bijector(self):
    return sigmoid_bijector.Sigmoid(
        low=self.low, high=self.high, validate_args=self.validate_args)

  def _parameter_control_dependencies(self, is_init):
    if not self.validate_args:
      return []
    assertions = []
    peak = None
    if is_init != tensor_util.is_ref(self.low):
      peak = tf.convert_to_tensor(self.peak)
      assertions.append(
          assert_util.assert_greater(
              peak, self.low, message='`peak` must be greater than `low`.'))
    if is_init != tensor_util.is_ref(self.high):
      peak = tf.convert_to_tensor(self.peak) if peak is None else peak
      assertions.append(
          assert_util.assert_greater(
              self.high, peak, message='`high` must be greater than `peak`.'))
    if is_init != tensor_util.is_ref(self.temperature):
      assertions.append(
          assert_util.assert_positive(
              self.temperature, message='`temperature` must be positive.'))
    return assertions

  def _sample_control_dependencies(self, x):
    assertions = []
    if not self.validate_args:
      return assertions
    assertions.append(assert_util.assert_greater_equal(
        x, self.low,
        message='Sample must be greater than or equal to `low`.'))
    assertions.append(assert_util.assert_less_equal(
        x, self.high,
        message='Sample must be less than or equal to `high`.'))
    return assertions
