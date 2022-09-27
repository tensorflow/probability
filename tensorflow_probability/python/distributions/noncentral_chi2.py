# Copyright 2022 The TensorFlow Probability Authors.
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
"""The noncentral Chi2 distribution class."""

import numpy as np
import tensorflow.compat.v2 as tf

from tensorflow_probability.python.bijectors import softplus as softplus_bijector
from tensorflow_probability.python.distributions import chi2 as chi2_lib
from tensorflow_probability.python.distributions import distribution
from tensorflow_probability.python.distributions import gamma as gamma_lib
from tensorflow_probability.python.distributions import poisson as poisson_lib
from tensorflow_probability.python.internal import assert_util
from tensorflow_probability.python.internal import distribution_util
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import parameter_properties
from tensorflow_probability.python.internal import prefer_static as ps
from tensorflow_probability.python.internal import reparameterization
from tensorflow_probability.python.internal import samplers
from tensorflow_probability.python.internal import special_math
from tensorflow_probability.python.internal import tensor_util
from tensorflow_probability.python.math import bessel
import tensorflow_probability.python.random as tfp_random


__all__ = ['NoncentralChi2']


class NoncentralChi2(distribution.AutoCompositeTensorDistribution):
  """Noncentral Chi2 distribution.

  The Noncentral Chi2 distribution is defined over positive real numbers using a
  degrees of freedom ('df') and a noncentrality parameter.

  The Noncentral Chi2 arises as the distribution of the squared Euclidean norm
  of a 'df'-dimensional multivariate Gaussian distribution with mean vector
  'mu' and identity covariance. The noncentrality parameter is given by the
  squared Euclidean norm of the mean vector 'mu'.

  #### Mathematical Details

  The probability density function (pdf) is

  ```none
  pdf(x; df, nc, x > 0)
    = 0.5 * (x/nc)**(df/4 - 1/2) * exp(-(x + nc)/2) * I_{k/2 - 1}(sqrt(nc * x))
  ```

  where:

  * `df` denotes the degrees of freedom,
  * `nc` denotes the noncentrality
  * `I_{a}` is the modified Bessel function of the first kind of order `a`.
  """

  def __init__(self,
               df,
               noncentrality,
               validate_args=False,
               allow_nan_stats=True,
               name='NoncentralChi2'):
    """Construct noncentral Chi2 distributions with parameter `df` and `noncentrality`.

    Args:
      df: Floating point tensor, the degrees of freedom of the distribution(s).
        `df` must contain only positive values.
      noncentrality: Floating point tensor, the noncentrality of the
        distribution(s). `noncentrality` must contain only positive values.
      validate_args: Python `bool`, default `False`. When `True` distribution
        parameters are checked for validity despite possibly degrading runtime
        performance. When `False` invalid inputs may silently render incorrect
        outputs.
      allow_nan_stats: Python `bool`, default `True`. When `True`, statistics
        (e.g., mean, mode, variance) use the value '`NaN`' to indicate the
        result is undefined. When `False`, an exception is raised if one or more
        of the statistic's batch members are undefined.
      name: Python `str` name prefixed to Ops created by this class.
    """
    parameters = dict(locals())

    with tf.name_scope(name) as name:
      dtype = dtype_util.common_dtype([df, noncentrality],
                                      dtype_hint=tf.float32)
      self._df = tensor_util.convert_nonref_to_tensor(
          df, name='df', dtype=dtype)
      self._noncentrality = tensor_util.convert_nonref_to_tensor(
          noncentrality, name='noncentrality', dtype=dtype)
      super().__init__(
          dtype=dtype,
          reparameterization_type=reparameterization.NOT_REPARAMETERIZED,
          validate_args=validate_args,
          allow_nan_stats=allow_nan_stats,
          parameters=parameters,
          name=name)

  @classmethod
  def _parameter_properties(cls, dtype, num_classes=None):

    return dict(
        df=parameter_properties.ParameterProperties(
            default_constraining_bijector_fn=(
                lambda: softplus_bijector.Softplus(low=dtype_util.eps(dtype)))),
        noncentrality=parameter_properties.ParameterProperties(
            default_constraining_bijector_fn=(
                lambda: softplus_bijector.Softplus(low=dtype_util.eps(dtype)))))

  @property
  def df(self):
    """Returns the degrees of freedom of the distribution."""
    return self._df

  @property
  def noncentrality(self):
    """Returns the noncentrality of the distribution."""
    return self._noncentrality

  def _event_shape_tensor(self):
    return tf.constant([], dtype=tf.int32)

  def _event_shape(self):
    return tf.TensorShape([])

  @distribution_util.AppendDocstring(
      """Note: See `tf.random.gamma` docstring for sampling details and
      caveats.

      Note: For dimensions where `df` is greater than 1, the sampler is actually
      fully reparameterized.
      """)
  def _sample_n(self, n, seed=None):

    shape = ps.concat([[n], self._batch_shape_tensor(
        df=self.df, noncentrality=self.noncentrality)], axis=0)

    df = tf.broadcast_to(self.df, shape)
    noncentrality = tf.broadcast_to(self.noncentrality, shape)

    chi2_seed, norm_seed, pois_seed, nc_chi2_seed = tfp_random.split_seed(
        seed, n=4)

    # Ensures that the df parameter passed to `random_gamma` is valid and
    # can be short-circuited fast.
    high_df_mask = df > 1.
    safe_df = tf.where(high_df_mask, df, 2.)

    # Generate the samples for the case when df > 1
    numpy_dtype = dtype_util.as_numpy_dtype(self.dtype)
    half = numpy_dtype(0.5)
    chi2_samps = gamma_lib.random_gamma(
        shape=[],
        concentration=0.5 * (safe_df - 1.),
        rate=half,
        seed=chi2_seed)

    normal_samps = samplers.normal(
        shape=shape,
        mean=tf.math.sqrt(noncentrality),
        dtype=self.dtype,
        seed=norm_seed)

    nc_chi2_samps_high_df = chi2_samps + tf.math.square(normal_samps)

    # Generate the samples for the case when df < 1
    poisson_samps = poisson_lib.random_poisson(
        shape=[],
        rates=0.5 * noncentrality,
        output_dtype=self.dtype,
        seed=pois_seed)[0]

    nc_chi2_samps_low_df = gamma_lib.random_gamma(
        shape=[],
        concentration=0.5 * (df + 2. * poisson_samps),
        rate=half,
        seed=nc_chi2_seed,
    )

    return tf.where(high_df_mask, nc_chi2_samps_high_df, nc_chi2_samps_low_df)

  def _log_prob(self, x):
    numpy_dtype = dtype_util.as_numpy_dtype(self.dtype)
    log2 = numpy_dtype(np.log(2.))

    df = tf.convert_to_tensor(self.df, dtype=self.dtype)
    noncentrality = tf.convert_to_tensor(self.noncentrality, dtype=self.dtype)
    safe_noncentrality_mask = noncentrality > 0.
    safe_noncentrality = tf.where(safe_noncentrality_mask, noncentrality, 1.)

    sqrt_x = tf.math.sqrt(x)
    sqrt_nc = tf.math.sqrt(safe_noncentrality)

    df_factor = 0.25 * df - 0.5

    log_prob = -log2 - 0.5 * tf.math.square(sqrt_x - sqrt_nc)
    log_prob = log_prob + tf.math.xlogy(df_factor, x)
    log_prob = log_prob - tf.math.xlogy(df_factor, safe_noncentrality)
    log_prob = log_prob + bessel.log_bessel_ive(2. * df_factor,
                                                sqrt_nc * sqrt_x)

    return tf.where(safe_noncentrality_mask, log_prob,
                    chi2_lib.Chi2(df).log_prob(x))

  def _cdf(self, x):
    return self._nc2_cdf_abramowitz_and_stegun(x)

  def _nc2_cdf_abramowitz_and_stegun(self, x):
    """Computes the CDF of a noncentral chi2 random variable.

    Computation is performed according to Eq (26.4.25) in
    Abramowitz, M., Stegun, I. A., & Romer, R. H. (1988).
    Handbook of mathematical functions with formulas, graphs,
    and mathematical tables.

    Args:
      x: point at which CDF is to be evaluated.

    Returns:
      `tf.Tensor` of the CDF evaluated at x.
    """
    df = tf.convert_to_tensor(self.df, dtype=self.dtype)
    noncentrality = tf.convert_to_tensor(self.noncentrality, dtype=self.dtype)

    param_shape = self._batch_shape_tensor(df=df, noncentrality=noncentrality)
    df = tf.broadcast_to(df, param_shape)
    noncentrality = tf.broadcast_to(noncentrality, param_shape)

    numpy_dtype = dtype_util.as_numpy_dtype(self.dtype)
    log_eps_abs = tf.math.log(numpy_dtype(np.finfo(numpy_dtype).eps))

    log_eps_p = numpy_dtype(-15. * np.log(10.))
    log_eps_s = log_eps_p - 10.

    below_precision_mask = log_eps_abs > self._log_cdf_chernoff_bound(
        x, df, noncentrality)

    # At the end, we will just set the dimensions with very low
    # noncentrality using the CDF of the standard chi2
    small_noncentrality_mask = noncentrality < 1e-10

    nc_over_2 = noncentrality / 2.
    x_over_2 = x / 2.

    # Index of the term with the highest weight in the sum from Eq (26.4.25)
    central_index = tf.math.floor(nc_over_2)

    # Bump 0 indices to 1
    central_index = central_index + tf.where(
        tf.math.equal(central_index, 0.), numpy_dtype(1.), numpy_dtype(0.))

    # Calculate the "central" term in the series
    log_central_weight = tf.math.xlogy(central_index, nc_over_2)
    log_central_weight = log_central_weight - nc_over_2
    log_central_weight = log_central_weight - tf.math.lgamma(central_index + 1.)

    central_weight = tf.math.exp(log_central_weight)

    central_df_over_2 = 0.5 * df + central_index

    central_index = central_index * tf.ones_like(x)
    central_weight = central_weight * tf.ones_like(x)

    # Wilson-Hilferty transform
    # For large values of `df`, the Chi2 CDF of x is very well approximated
    # by the CDF of a Gaussian with mean 1 - 2 / (9 df) and variance 2 / (9 df)
    # evaluated at(x / df)^{1/3}.
    #
    # The rationale of using it here is that for `df`s larger than 1e3 it is
    # more numerically stable and accurate to use the Gaussian WH transform than
    # to evaluate the Chi2 CDF directly
    central_df_var_term = 1. / central_df_over_2 / 9.

    wilson_hilferty_x = (x_over_2 / central_df_over_2)**(1. / 3.)
    wilson_hilferty_x = wilson_hilferty_x - (1. - central_df_var_term)
    wilson_hilferty_x = wilson_hilferty_x / tf.math.sqrt(central_df_var_term)

    # Compute the central chi2 probability
    central_chi2_prob = tf.where(central_df_over_2 < 1e3,
                                 tf.math.igamma(central_df_over_2, x_over_2),
                                 special_math.ndtr(wilson_hilferty_x))

    # Prob will contain the return value of the function
    prob = central_weight * central_chi2_prob

    # Calculate adjustment terms to the chi2 probability above.
    # This means that `tf.math.igamma` only needs to be called once, the rest
    # of the CDFs can be calculated using the adjustment.
    log_adjustment_term = tf.math.xlogy(central_df_over_2, x_over_2)
    log_adjustment_term = log_adjustment_term - x_over_2 - tf.math.lgamma(
        central_df_over_2 + 1.)
    central_adjustment_term = tf.math.exp(log_adjustment_term)

    # Now, sum the series "backwards" from the central term
    def backwards_summation_loop_body(index, weight, adj_term, total_adj, prob):

      update_mask = ~below_precision_mask & (index > 0.) & (
          tf.math.log(weight) + tf.math.log(adj_term) > log_eps_s)

      df_over_2 = 0.5 * df + index

      adj_term = adj_term * tf.where(update_mask, df_over_2 / x_over_2, 1.)
      total_adj = total_adj + tf.where(update_mask, adj_term, 0.)

      adjusted_prob = central_chi2_prob + total_adj
      weight = weight * tf.where(update_mask, index / nc_over_2, 1.)

      prob_term = weight * adjusted_prob
      prob = prob + tf.where(update_mask, prob_term, 0.)

      index = index - tf.cast(tf.where(update_mask, 1., 0.), self.dtype)

      return index, weight, adj_term, total_adj, prob

    def backwards_summation_loop_cond(index, weight, adj_term, total_adj, prob):
      return tf.reduce_any(
          ~below_precision_mask &
          (index > 0.) &
          (tf.math.log(weight) + tf.math.log(adj_term) > log_eps_s))

    _, _, _, _, prob = tf.while_loop(
        cond=backwards_summation_loop_cond,
        body=backwards_summation_loop_body,
        loop_vars=(central_index, central_weight, central_adjustment_term,
                   tf.zeros_like(central_adjustment_term), prob))

    # Now, sum the series "forwards" from the central term
    def forwards_summation_loop_body(index, weight, adj_term, total_adj, prob):

      update_mask = ~below_precision_mask & (
          tf.math.log(weight) + tf.math.log(adj_term) > log_eps_s)

      weight = weight * tf.where(update_mask, nc_over_2 / (index + 1.), 1.)
      adjusted_prob = central_chi2_prob - total_adj

      prob_term = weight * adjusted_prob
      prob = prob + tf.where(update_mask, prob_term, 0.)

      index = tf.where(update_mask, index + 1., index)
      df_over_2 = 0.5 * df + index
      adj_term = adj_term * tf.where(update_mask, x_over_2 / df_over_2, 1.)
      total_adj = total_adj + tf.where(update_mask, adj_term, 0.)

      return index, weight, adj_term, total_adj, prob

    def forwards_summation_loop_cond(index, weight, adj_term, total_adj, prob):
      return tf.reduce_any(
          ~below_precision_mask &
          (tf.math.log(weight) + tf.math.log(adj_term) > log_eps_s))

    _, _, _, _, prob = tf.while_loop(
        cond=forwards_summation_loop_cond,
        body=forwards_summation_loop_body,
        loop_vars=(central_index, central_weight, central_adjustment_term,
                   central_adjustment_term, prob))

    prob = tf.where(small_noncentrality_mask,
                    tf.math.igamma(0.5 * df, 0.5 * x), prob)

    x_above_mean_mask = x > self.mean()

    prob = tf.where(below_precision_mask & x_above_mean_mask, tf.ones_like(x),
                    prob)

    prob = tf.where(below_precision_mask & ~x_above_mean_mask, tf.zeros_like(x),
                    prob)

    return prob

  def _log_cdf_chernoff_bound(self, x, df, noncentrality):
    # Implements Eqs (22) and (23) in Shnidman, D. A. The Calculation of the
    # Probability of Detection and the Generalized Marcum Q-Function (1989)
    # The substitutions used are self.df = 2N, self.nc = 2NX and x = 2Y

    df_over_2x = tf.math.xdivy(df, 2. * x)

    # Quantity in Eq (22)
    chernoff_coef = 1. - df_over_2x - tf.math.sqrt(
        tf.math.square(df_over_2x) + tf.math.xdivy(noncentrality, x))

    # Exponent in Eq (23)
    bound = -chernoff_coef * x + noncentrality * chernoff_coef / (
        1. - chernoff_coef) - tf.math.xlog1py(df, -chernoff_coef)
    bound = 0.5 * bound

    return bound

  def _mean(self):
    return self.df + self.noncentrality

  def _variance(self):
    return 2. * self.df + 4. * self.noncentrality

  def _default_event_space_bijector(self):
    return softplus_bijector.Softplus(validate_args=self.validate_args)

  def _sample_control_dependencies(self, x):
    assertions = []
    if not self.validate_args:
      return assertions

    assertions.append(
        assert_util.assert_non_negative(
            x, message='Sample must be non-negative.'))

    return assertions

  def _parameter_control_dependencies(self, is_init):
    if not self.validate_args:
      return []

    assertions = []

    if is_init != tensor_util.is_ref(self.df):
      assertions.append(
          assert_util.assert_positive(
              self.df, message='Argument `df` must be positive.'))

    if is_init != tensor_util.is_ref(self.noncentrality):
      assertions.append(
          assert_util.assert_non_negative(
              self.noncentrality,
              message='Argument `noncentrality` must be non-negative.'))

    return assertions
