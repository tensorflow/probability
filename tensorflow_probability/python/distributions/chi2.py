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
"""The Chi2 distribution class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v2 as tf
from tensorflow_probability.python.distributions import gamma
from tensorflow_probability.python.internal import assert_util
from tensorflow_probability.python.internal import dtype_util
from tensorflow.python.util import deprecation  # pylint: disable=g-direct-tensorflow-import


__all__ = [
    "Chi2",
    "Chi2WithAbsDf",
]


class Chi2(gamma.Gamma):
  """Chi2 distribution.

  The Chi2 distribution is defined over positive real numbers using a degrees of
  freedom ("df") parameter.

  #### Mathematical Details

  The probability density function (pdf) is,

  ```none
  pdf(x; df, x > 0) = x**(0.5 df - 1) exp(-0.5 x) / Z
  Z = 2**(0.5 df) Gamma(0.5 df)
  ```

  where:

  * `df` denotes the degrees of freedom,
  * `Z` is the normalization constant, and,
  * `Gamma` is the [gamma function](
    https://en.wikipedia.org/wiki/Gamma_function).

  The Chi2 distribution is a special case of the Gamma distribution, i.e.,

  ```python
  Chi2(df) = Gamma(concentration=0.5 * df, rate=0.5)
  ```

  """

  def __init__(self,
               df,
               validate_args=False,
               allow_nan_stats=True,
               name="Chi2"):
    """Construct Chi2 distributions with parameter `df`.

    Args:
      df: Floating point tensor, the degrees of freedom of the
        distribution(s). `df` must contain only positive values.
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
    # Even though all stats of chi2 are defined for valid parameters, this is
    # not true in the parent class "gamma."  therefore, passing
    # allow_nan_stats=True
    # through to the parent class results in unnecessary asserts.
    with tf.name_scope(name) as name:
      df = tf.convert_to_tensor(
          df,
          name="df",
          dtype=dtype_util.common_dtype([df], dtype_hint=tf.float32))
      with tf.control_dependencies([
          assert_util.assert_positive(df),
      ] if validate_args else []):
        self._df = tf.identity(df, name="df")

      super(Chi2, self).__init__(
          concentration=0.5 * self._df,
          rate=0.5,
          validate_args=validate_args,
          allow_nan_stats=allow_nan_stats,
          name=name)
    self._parameters = parameters

  @staticmethod
  def _param_shapes(sample_shape):
    return {"df": tf.convert_to_tensor(sample_shape, dtype=tf.int32)}

  @classmethod
  def _params_event_ndims(cls):
    return dict(df=0)

  @property
  def df(self):
    return self._df


class Chi2WithAbsDf(Chi2):
  """Chi2 with parameter transform `df = floor(abs(df))`."""

  @deprecation.deprecated(
      "2019-06-05",
      "Chi2WithAbsDf is deprecated, use "
      "Chi2(df=tf.floor(tf.abs(df))) instead.",
      warn_once=True)
  def __init__(self,
               df,
               validate_args=False,
               allow_nan_stats=True,
               name="Chi2WithAbsDf"):
    parameters = dict(locals())
    with tf.name_scope(name) as name:
      super(Chi2WithAbsDf, self).__init__(
          df=tf.floor(tf.abs(df, name="abs_df"), name="floor_abs_df"),
          validate_args=validate_args,
          allow_nan_stats=allow_nan_stats,
          name=name)
    self._parameters = parameters
