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
import tensorflow as tf

from tensorflow_probability.python.distributions import gamma
from tensorflow_probability.python.internal import dtype_util


__all__ = [
    "Exponential",
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
               validate_args=False,
               allow_nan_stats=True,
               name="Exponential"):
    """Construct Exponential distribution with parameter `rate`.

    Args:
      rate: Floating point tensor, equivalent to `1 / mean`. Must contain only
        positive values.
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
    with tf.name_scope(name, values=[rate]) as name:
      self._rate = tf.convert_to_tensor(
          rate,
          name="rate",
          dtype=dtype_util.common_dtype([rate], preferred_dtype=tf.float32))
    super(Exponential, self).__init__(
        concentration=1.,
        rate=self._rate,
        allow_nan_stats=allow_nan_stats,
        validate_args=validate_args,
        name=name)
    self._parameters = parameters
    self._graph_parents += [self._rate]

  @staticmethod
  def _param_shapes(sample_shape):
    return {"rate": tf.convert_to_tensor(sample_shape, dtype=tf.int32)}

  @property
  def rate(self):
    return self._rate

  def _log_survival_function(self, value):
    return self._log_prob(value) - tf.log(self._rate)

  def _sample_n(self, n, seed=None):
    shape = tf.concat([[n], tf.shape(self._rate)], 0)
    # Uniform variates must be sampled from the open-interval `(0, 1)` rather
    # than `[0, 1)`. To do so, we use `np.finfo(self.dtype.as_numpy_dtype).tiny`
    # because it is the smallest, positive, "normal" number. A "normal" number
    # is such that the mantissa has an implicit leading 1. Normal, positive
    # numbers x, y have the reasonable property that, `x + y >= max(x, y)`. In
    # this case, a subnormal number (i.e., np.nextafter) can cause us to sample
    # 0.
    sampled = tf.random_uniform(
        shape,
        minval=np.finfo(self.dtype.as_numpy_dtype).tiny,
        maxval=1.,
        seed=seed,
        dtype=self.dtype)
    return -tf.log(sampled) / self._rate
