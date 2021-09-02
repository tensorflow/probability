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
"""The Exponential distribution class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
import numpy as np
import tensorflow.compat.v2 as tf

from tensorflow_probability.python.bijectors import softplus as softplus_bijector
from tensorflow_probability.python.distributions import gamma
from tensorflow_probability.python.internal import distribution_util
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import parameter_properties
from tensorflow_probability.python.internal import prefer_static as ps
from tensorflow_probability.python.internal import samplers
from tensorflow_probability.python.internal import tensor_util


__all__ = [
    'Exponential',
]


class Exponential(gamma.Gamma):
  """Exponential distribution.

  The Exponential distribution is parameterized by an event `rate` parameter.

  #### Mathematical Details

  The probability density function (pdf) is,

  ```none
  pdf(x; lambda, x > 0) = exp(-lambda x) / Z
  Z = 1 / lambda
  ```

  where `rate = lambda` and `Z` is the normalizaing constant.

  The Exponential distribution is a special case of the Gamma distribution,
  i.e.,

  ```python
  Exponential(rate) = Gamma(concentration=1., rate)
  ```

  The Exponential distribution uses a `rate` parameter, or "inverse scale",
  which can be intuited as,

  ```none
  X ~ Exponential(rate=1)
  Y = X / rate
  ```

  """

  def __init__(self,
               rate,
               force_probs_to_zero_outside_support=False,
               validate_args=False,
               allow_nan_stats=True,
               name='Exponential'):
    """Construct Exponential distribution with parameter `rate`.

    Args:
      rate: Floating point tensor, equivalent to `1 / mean`. Must contain only
        positive values.
      force_probs_to_zero_outside_support: Python `bool`. When `True`, negative
        and non-integer values are evaluated "strictly": `cdf` returns
        `0`, `sf` returns `1`, and `log_cdf` and `log_sf` correspond.  When
        `False`, the implementation is free to save computation (and TF graph
        size) by evaluating something that matches the Exponential cdf at
        non-negative values `x` but produces an unrestricted result on
        other inputs. In the case of Exponential distribution, the `cdf`
        formula in this case happens to be the continuous function
        `1 - exp(rate * value)`.
        Note that this function is not itself a cdf function.
        Default value: `False`.
      validate_args: Python `bool`, default `False`. When `True` distribution
        parameters are checked for validity despite possibly degrading runtime
        performance. When `False` invalid inputs may silently render incorrect
        outputs.
      allow_nan_stats: Python `bool`, default `True`. When `True`, statistics
        (e.g., mean, mode, variance) use the value "`NaN`" to indicate the
        result is undefined. When `False`, an exception is raised if one or
        more of the statistic's batch members are undefined.
      name: Python `str` name prefixed to Ops created by this class.
    """
    parameters = dict(locals())
    # Even though all statistics of are defined for valid inputs, this is not
    # true in the parent class "Gamma."  Therefore, passing
    # allow_nan_stats=True
    # through to the parent class results in unnecessary asserts.
    with tf.name_scope(name) as name:
      self._rate = tensor_util.convert_nonref_to_tensor(
          rate,
          name='rate',
          dtype=dtype_util.common_dtype([rate], dtype_hint=tf.float32))
      super(Exponential, self).__init__(
          concentration=1.,
          rate=self._rate,
          allow_nan_stats=allow_nan_stats,
          validate_args=validate_args,
          force_probs_to_zero_outside_support=(
              force_probs_to_zero_outside_support),
          name=name)
      self._parameters = parameters

  @classmethod
  def _parameter_properties(cls, dtype, num_classes=None):
    # pylint: disable=g-long-lambda
    return dict(
        rate=parameter_properties.ParameterProperties(
            default_constraining_bijector_fn=(
                lambda: softplus_bijector.Softplus(low=dtype_util.eps(dtype)))))
    # pylint: enable=g-long-lambda

  @property
  def rate(self):
    return self._rate

  def _cdf(self, value):
    cdf = -tf.math.expm1(-self.rate * value)
    # Set cdf = 0 when value is less than 0.
    return distribution_util.extend_cdf_outside_support(value, cdf, low=0.)

  def _log_survival_function(self, value):
    rate = tf.convert_to_tensor(self._rate)
    log_sf = self._log_prob(value, rate=rate) - tf.math.log(rate)

    if self.force_probs_to_zero_outside_support:
      # Set log_survival_function = 0 when value is less than 0.
      log_sf = tf.where(value < 0., tf.zeros_like(log_sf), log_sf)

    return log_sf

  def _sample_n(self, n, seed=None):
    rate = tf.convert_to_tensor(self.rate)
    shape = ps.concat([[n], ps.shape(rate)], 0)
    # Uniform variates must be sampled from the open-interval `(0, 1)` rather
    # than `[0, 1)`. To do so, we use
    # `np.finfo(dtype_util.as_numpy_dtype(self.dtype)).tiny`
    # because it is the smallest, positive, "normal" number. A "normal" number
    # is such that the mantissa has an implicit leading 1. Normal, positive
    # numbers x, y have the reasonable property that, `x + y >= max(x, y)`. In
    # this case, a subnormal number (i.e., np.nextafter) can cause us to sample
    # 0.
    sampled = samplers.uniform(
        shape,
        minval=np.finfo(dtype_util.as_numpy_dtype(self.dtype)).tiny,
        maxval=1.,
        seed=seed,
        dtype=self.dtype)
    return -tf.math.log(sampled) / rate

  def _quantile(self, value):
    return -tf.math.log1p(-value) / self.rate

  def _default_event_space_bijector(self):
    return softplus_bijector.Softplus(validate_args=self.validate_args)

  @classmethod
  def _maximum_likelihood_parameters(cls, value):
    return {'rate': 1. / tf.reduce_mean(value, axis=0)}
