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

import numpy as np
import tensorflow.compat.v2 as tf

from tensorflow_probability.python.bijectors import softplus as softplus_bijector
from tensorflow_probability.python.distributions import distribution
from tensorflow_probability.python.distributions import gamma as gamma_lib
from tensorflow_probability.python.distributions import kullback_leibler
from tensorflow_probability.python.internal import assert_util
from tensorflow_probability.python.internal import distribution_util
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import parameter_properties
from tensorflow_probability.python.internal import prefer_static as ps
from tensorflow_probability.python.internal import reparameterization
from tensorflow_probability.python.internal import tensor_util


__all__ = [
    'Chi2',
]


class Chi2(distribution.Distribution):
  """Chi2 distribution.

  The Chi2 distribution is defined over positive real numbers using a degrees of
  freedom ('df') parameter.

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

  The Chi2 distribution is a special case of the Gamma distribution -- i.e.,
  `Chi2(df)` represents the same distribution as
  `Gamma(concentration=0.5 * df, rate=0.5)`.
  """

  def __init__(self,
               df,
               validate_args=False,
               allow_nan_stats=True,
               name='Chi2'):
    """Construct Chi2 distributions with parameter `df`.

    Args:
      df: Floating point tensor, the degrees of freedom of the
        distribution(s). `df` must contain only positive values.
      validate_args: Python `bool`, default `False`. When `True` distribution
        parameters are checked for validity despite possibly degrading runtime
        performance. When `False` invalid inputs may silently render incorrect
        outputs.
      allow_nan_stats: Python `bool`, default `True`. When `True`, statistics
        (e.g., mean, mode, variance) use the value '`NaN`' to indicate the
        result is undefined. When `False`, an exception is raised if one or
        more of the statistic's batch members are undefined.
      name: Python `str` name prefixed to Ops created by this class.
    """
    parameters = dict(locals())
    with tf.name_scope(name) as name:
      dtype = dtype_util.common_dtype([df], dtype_hint=tf.float32)
      self._df = tensor_util.convert_nonref_to_tensor(
          df, name='df', dtype=dtype)
      super(Chi2, self).__init__(
          dtype=dtype,
          validate_args=validate_args,
          allow_nan_stats=allow_nan_stats,
          reparameterization_type=reparameterization.FULLY_REPARAMETERIZED,
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
    return self._df

  def _batch_shape_tensor(self):
    return ps.shape(self.df)

  def _batch_shape(self):
    return self.df.shape

  def _event_shape_tensor(self):
    return tf.constant([], dtype=tf.int32)

  def _event_shape(self):
    return tf.TensorShape([])

  @distribution_util.AppendDocstring(
      """Note: See `tf.random.gamma` docstring for sampling details and
      caveats.""")
  def _sample_n(self, n, seed=None):
    return gamma_lib.random_gamma(
        shape=[n],
        concentration=0.5 * self.df,
        rate=tf.convert_to_tensor(0.5, dtype=self.dtype),
        seed=seed)

  def _log_prob(self, x):
    concentration = 0.5 * self.df
    rate = tf.convert_to_tensor(0.5, dtype=self.dtype)
    log_unnormalized_prob = tf.math.xlogy(concentration - 1., x) - rate * x
    log_normalization = (tf.math.lgamma(concentration) -
                         concentration * tf.math.log(rate))
    return log_unnormalized_prob - log_normalization

  def _cdf(self, x):
    return tf.math.igamma(0.5 * self.df, 0.5 * x)

  def _entropy(self):
    concentration = 0.5 * self.df
    rate = tf.convert_to_tensor(0.5, dtype=self.dtype)
    return (concentration - tf.math.log(rate) +
            tf.math.lgamma(concentration) +
            ((1. - concentration) * tf.math.digamma(concentration)))

  def _mean(self):
    return tf.identity(self.df)

  def _variance(self):
    return 2. * self.df

  @distribution_util.AppendDocstring(
      """The mode of a Chi2 distribution is `df - 2` when `df > 2`, and `NaN`
      otherwise. If `self.allow_nan_stats` is `False`, an exception will be
      raised rather than returning `NaN`.""")
  def _mode(self):
    df = tf.convert_to_tensor(self.df)
    mode = df - 2.
    if self.allow_nan_stats:
      assertions = []
    else:
      assertions = [assert_util.assert_less(
          2. * tf.ones([], self.dtype), df,
          message='Mode not defined when df <= 2.')]
    with tf.control_dependencies(assertions):
      return tf.where(
          df > 2.,
          mode,
          dtype_util.as_numpy_dtype(self.dtype)(np.nan))

  def _default_event_space_bijector(self):
    return softplus_bijector.Softplus(validate_args=self.validate_args)

  def _sample_control_dependencies(self, x):
    assertions = []
    if not self.validate_args:
      return assertions
    assertions.append(assert_util.assert_non_negative(
        x, message='Sample must be non-negative.'))
    return assertions

  def _parameter_control_dependencies(self, is_init):
    if not self.validate_args:
      return []
    assertions = []
    if is_init != tensor_util.is_ref(self.df):
      assertions.append(assert_util.assert_positive(
          self.df, message='Argument `df` must be positive.'))
    return assertions


@kullback_leibler.RegisterKL(Chi2, Chi2)
def _kl_chi2_chi2(c0, c1, name=None):
  """Calculate the batched KL divergence KL(c0 || c1) with c0 and c1 Chi2."""
  return _kl_gamma_gamma(concentration0=0.5 * c0.df, rate0=0.5,
                         concentration1=0.5 * c1.df, rate1=0.5,
                         name=name or 'kl_chi2_chi2')


@kullback_leibler.RegisterKL(Chi2, gamma_lib.Gamma)
def _kl_chi2_gamma(c0, g1, name=None):
  """Calculate batched KL divergence KL(c0 || g1) with c0 Chi2 and g1 Gamma."""
  return _kl_gamma_gamma(concentration0=0.5 * c0.df, rate0=0.5,
                         concentration1=g1.concentration, rate1=g1.rate,
                         name=name or 'kl_chi2_gamma')


@kullback_leibler.RegisterKL(gamma_lib.Gamma, Chi2)
def _kl_gamma_chi2(g0, c1, name=None):
  """Calculate batched KL divergence KL(g0 || c1) with g0 Gamma and c1 Chi2."""
  return _kl_gamma_gamma(concentration0=g0.concentration, rate0=g0.rate,
                         concentration1=0.5 * c1.df, rate1=0.5,
                         name=name or 'kl_gamma_chi2')


def _kl_gamma_gamma(concentration0, rate0, concentration1, rate1, name=None):
  """Calculate batched KL divergence KL(g0 || g1) with given Gamma parameters.

  Args:
    concentration0: Concentration of first Gamma distribution (g0).
    rate0: Rate of first Gamma distirbution (g0).
    concentration1: Concentration of second Gamma distribution (g1).
    rate1: Rate of second Gamma distirbution (g1).
    name: Python `str` name to use for created operations.
      Default value: `None` (i.e., `'kl_gamma_gamma'`).

  Returns:
    kl_gamma_gamma: `Tensor`. The batchwise KL(g0 || g1).
  """
  with tf.name_scope(name or 'kl_gamma_gamma'):
    # Result from:
    #   http://www.fil.ion.ucl.ac.uk/~wpenny/publications/densities.ps
    # For derivation see:
    #   http://stats.stackexchange.com/questions/11646/kullback-leibler-divergence-between-two-gamma-distributions   pylint: disable=line-too-long
    dtype = dtype_util.common_dtype(
        [concentration0, rate0, concentration1, rate1], dtype_hint=tf.float32)
    g0_concentration = tf.convert_to_tensor(concentration0, dtype=dtype)
    g0_rate = tf.convert_to_tensor(rate0, dtype=dtype)
    g1_concentration = tf.convert_to_tensor(concentration1, dtype=dtype)
    g1_rate = tf.convert_to_tensor(rate1, dtype=dtype)
    return (((g0_concentration - g1_concentration) *
             tf.math.digamma(g0_concentration)) +
            tf.math.lgamma(g1_concentration) -
            tf.math.lgamma(g0_concentration) +
            g1_concentration * tf.math.log(g0_rate) -
            g1_concentration * tf.math.log(g1_rate) + g0_concentration *
            (g1_rate / g0_rate - 1.))
