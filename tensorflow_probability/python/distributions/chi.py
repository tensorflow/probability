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

from tensorflow_probability.python import math as tfp_math
from tensorflow_probability.python.bijectors import invert as invert_bijector
from tensorflow_probability.python.bijectors import softplus as softplus_bijector
from tensorflow_probability.python.bijectors import square as square_bijector
from tensorflow_probability.python.distributions import chi2
from tensorflow_probability.python.distributions import distribution
from tensorflow_probability.python.distributions import kullback_leibler
from tensorflow_probability.python.distributions import transformed_distribution
from tensorflow_probability.python.internal import assert_util
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import parameter_properties
from tensorflow_probability.python.internal import tensor_util


# TODO(b/182603117): Remove `AutoCompositeTensor` subclass when
# `TransformedDistribution` is converted to `CompositeTensor`.
class Chi(transformed_distribution.TransformedDistribution,
          distribution.AutoCompositeTensorDistribution):
  """Chi distribution.

  The Chi distribution is defined over nonnegative real numbers and uses a
  degrees of freedom ('df') parameter.

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
               name='Chi'):
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
      dtype = dtype_util.common_dtype([df], dtype_hint=tf.float32)
      self._df = tensor_util.convert_nonref_to_tensor(
          df, name='df', dtype=dtype)
      super(Chi, self).__init__(
          distribution=chi2.Chi2(df=self._df,
                                 validate_args=validate_args,
                                 allow_nan_stats=allow_nan_stats),
          bijector=invert_bijector.Invert(
              square_bijector.Square(validate_args=validate_args)),
          validate_args=validate_args,
          parameters=parameters,
          name=name)

  @classmethod
  def _parameter_properties(cls, dtype, num_classes=None):
    # pylint: disable=g-long-lambda
    return dict(
        df=parameter_properties.ParameterProperties(
            default_constraining_bijector_fn=(
                lambda: softplus_bijector.Softplus(low=dtype_util.eps(dtype)))))
    # pylint: enable=g-long-lambda

  @property
  def df(self):
    """Distribution parameter for degrees of freedom."""
    return self._df

  experimental_is_sharded = False

  def _mean(self, df=None):
    df = tf.convert_to_tensor(self.df if df is None else df)
    return np.sqrt(2.) * tf.exp(
        -tfp_math.log_gamma_difference(0.5, 0.5 * df))

  def _variance(self):
    df = tf.convert_to_tensor(self.df)
    return df - self._mean(df) ** 2

  def _entropy(self):
    df = tf.convert_to_tensor(self.df)
    return (tf.math.lgamma(0.5 * df) +
            0.5 * (df - np.log(2.) -
                   (df - 1.) * tf.math.digamma(0.5 * df)))

  def _default_event_space_bijector(self):
    return softplus_bijector.Softplus(validate_args=self.validate_args)

  def _parameter_control_dependencies(self, is_init):
    if not self.validate_args:
      return []
    assertions = []
    if is_init != tensor_util.is_ref(self._df):
      assertions.append(assert_util.assert_positive(
          self._df, message='Argument `df` must be positive.'))
    return assertions

  def _sample_control_dependencies(self, x):
    assertions = []
    if not self.validate_args:
      return assertions
    assertions.append(assert_util.assert_non_negative(
        x, message='Sample must be non-negative.'))
    return assertions


@kullback_leibler.RegisterKL(Chi, Chi)
def _kl_chi_chi(a, b, name=None):
  """Calculate the batched KL divergence KL(a || b) with a and b Chi.

  Args:
    a: instance of a Chi distribution object.
    b: instance of a Chi distribution object.
    name: (optional) Name to use for created operations.
      default is 'kl_chi_chi'.

  Returns:
    Batchwise KL(a || b)
  """
  with tf.name_scope(name or 'kl_chi_chi'):
    a_df = tf.convert_to_tensor(a.df)
    b_df = tf.convert_to_tensor(b.df)
    # Consistent with
    # https://mast.queensu.ca/~communications/Papers/gil-msc11.pdf, page 118
    # The paper introduces an additional scaling parameter; setting that
    # parameter to 1 and simplifying yields the expression we use here.
    return (0.5 * tf.math.digamma(0.5 * a_df) * (a_df - b_df) +
            tf.math.lgamma(0.5 * b_df) - tf.math.lgamma(0.5 * a_df))
