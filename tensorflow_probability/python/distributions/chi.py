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
"""The Chi distribution class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow.compat.v2 as tf

from tensorflow_probability.python.bijectors import invert as invert_bijector
from tensorflow_probability.python.bijectors import square as square_bijector
from tensorflow_probability.python.distributions import chi2
from tensorflow_probability.python.distributions import kullback_leibler
from tensorflow_probability.python.distributions import transformed_distribution
from tensorflow_probability.python.internal import assert_util
from tensorflow_probability.python.internal import dtype_util


class Chi(transformed_distribution.TransformedDistribution):
  """Chi distribution.

  The Chi distribution is defined over nonnegative real numbers and uses a
  degrees of freedom ("df") parameter.

  #### Mathematical Details

  The probability density function (pdf) is,

  ```none
  pdf(x; df, x >= 0) = x**(df - 1) exp(-0.5 x**2) / Z
  Z = 2**(0.5 df - 1) Gamma(0.5 df)
  ```

  where:

  * `df` denotes the degrees of freedom,
  * `Z` is the normalization constant, and,
  * `Gamma` is the [gamma function](
    https://en.wikipedia.org/wiki/Gamma_function).

  The Chi distribution is a transformation of the Chi2 distribution; it is the
  distribution of the positive square root of a variable obeying a Chi2
  distribution.

  """

  def __init__(self,
               df,
               validate_args=False,
               allow_nan_stats=True,
               name="Chi"):
    """Construct Chi distributions with parameter `df`.

    Args:
      df: Floating point tensor, the degrees of freedom of the
        distribution(s). `df` must contain only positive values.
      validate_args: Python `bool`, default `False`. When `True` distribution
        parameters are checked for validity despite possibly degrading runtime
        performance. When `False` invalid inputs may silently render incorrect
        outputs.
      allow_nan_stats: Python `bool`, default `True`. When `True`, statistics
        (e.g., mean, mode, variance) use the value `NaN` to indicate the result
        is undefined. When `False`, an exception is raised if one or more of the
        statistic's batch members are undefined.
      name: Python `str` name prefixed to Ops created by this class.
        Default value: `'Chi'`.
    """
    parameters = dict(locals())
    with tf.name_scope(name) as name:
      df = tf.convert_to_tensor(
          value=df,
          name="df",
          dtype=dtype_util.common_dtype([df], preferred_dtype=tf.float32))
      validation_assertions = (
          [assert_util.assert_positive(df)] if validate_args else [])
      with tf.control_dependencies(validation_assertions):
        self._df = tf.identity(df, name="df")

      super(Chi, self).__init__(
          distribution=chi2.Chi2(df=self._df,
                                 validate_args=validate_args,
                                 allow_nan_stats=allow_nan_stats),
          bijector=invert_bijector.Invert(square_bijector.Square()),
          parameters=parameters,
          name=name)

  @classmethod
  def _params_event_ndims(cls):
    return dict(df=0)

  @property
  def df(self):
    """Distribution parameter for degrees of freedom."""
    return self._df

  def _mean(self):
    return np.sqrt(2) * tf.exp(
        tf.math.lgamma(0.5 * (self.df + 1)) - tf.math.lgamma(0.5 * self.df))

  def _variance(self):
    return self.df - tf.square(self._mean())

  def _entropy(self):
    return (tf.math.lgamma(self.df / 2) + 0.5 *
            (self.df - np.log(2) -
             (self.df - 1) * tf.math.digamma(0.5 * self.df)))


@kullback_leibler.RegisterKL(Chi, Chi)
def _kl_chi_chi(a, b, name=None):
  """Calculate the batched KL divergence KL(a || b) with a and b Chi.

  Args:
    a: instance of a Chi distribution object.
    b: instance of a Chi distribution object.
    name: (optional) Name to use for created operations.
      default is "kl_chi_chi".

  Returns:
    Batchwise KL(a || b)
  """
  with tf.name_scope(name or "kl_chi_chi"):
    # Consistent with
    # https://mast.queensu.ca/~communications/Papers/gil-msc11.pdf, page 118
    # The paper introduces an additional scaling parameter; setting that
    # parameter to 1 and simplifying yields the expression we use here.
    return (0.5 * tf.math.digamma(0.5 * a.df) * (a.df - b.df) +
            tf.math.lgamma(0.5 * b.df) - tf.math.lgamma(0.5 * a.df))
