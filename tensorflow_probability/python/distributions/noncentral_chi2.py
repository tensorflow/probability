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
from tensorflow_probability.python.internal import custom_gradient as tfp_custom_gradient
from tensorflow_probability.python.internal import distribution_util
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import parameter_properties
from tensorflow_probability.python.internal import prefer_static as ps
from tensorflow_probability.python.internal import reparameterization
from tensorflow_probability.python.internal import samplers
from tensorflow_probability.python.internal import special_math
from tensorflow_probability.python.internal import tensor_util
from tensorflow_probability.python.math import bessel
from tensorflow_probability.python.math import generic
from tensorflow_probability.python.math import root_search
import tensorflow_probability.python.random as tfp_random


__all__ = ['NoncentralChi2']


def _nc2_log_prob(x, df, noncentrality):
  """Computes the log probability of a noncentral Chi2 random variable.

  Note: the (pseudo) log probability can be evaluate for nonpositive degrees of
  freedom as well, but the results might be wrong as `log_bessel_ive` does not
  return the sign of its result.

  Args:
    x: `tf.Tensor` of locations at which to compute the log probability.
    df: `tf.Tensor` of the degrees of freedom of the NC2 variate.
    noncentrality: `tf.Tensor` of the noncentrality of the NC2 variate.

  Returns:
    `tf.Tensor` of log probabilities.
  """
  numpy_dtype = dtype_util.as_numpy_dtype(df.dtype)
  log2 = numpy_dtype(np.log(2.))

  safe_noncentrality_mask = noncentrality > 0.
  safe_noncentrality = tf.where(safe_noncentrality_mask, noncentrality, 1.)

  sqrt_x = tf.math.sqrt(x)
  sqrt_nc = tf.math.sqrt(safe_noncentrality)

  df_factor = 0.25 * df - 0.5

  log_prob = -log2 - 0.5 * tf.math.square(sqrt_x - sqrt_nc)
  log_prob = log_prob + tf.math.xlogy(df_factor, x)
  log_prob = log_prob - tf.math.xlogy(df_factor, safe_noncentrality)
  log_prob = log_prob + bessel.log_bessel_ive(2. * df_factor, sqrt_nc * sqrt_x)

  log_prob = tf.where(safe_noncentrality_mask, log_prob,
                      chi2_lib.Chi2(df).log_prob(x))

  return log_prob


def _nc2_cdf_abramowitz_and_stegun(x, df, noncentrality):
  """Computes the CDF of a noncentral chi2 random variable.

  Computation is performed according to Eq (26.4.25) in
  Abramowitz, M., Stegun, I. A., & Romer, R. H. (1988).
  Handbook of mathematical functions with formulas, graphs,
  and mathematical tables.

  Args:
    x: point at which CDF is to be evaluated.
    df: `tf.Tensor` of the degrees of freedom of the distribution.
    noncentrality: `tf.Tensor` of the noncentrality parameter of the
      distribution.

  Returns:
    `tf.Tensor` of the CDF evaluated at x.
  """

  dtype = df.dtype
  numpy_dtype = dtype_util.as_numpy_dtype(dtype)
  log_eps_abs = tf.math.log(numpy_dtype(np.finfo(numpy_dtype).eps))

  # This can be set to something a bit larger (e.g. twice the type's machine
  # epsilon) if we want to speed up the calculation in the tails,
  # at the cost of potentially quite wrong answers in the far tails.
  # SciPy uses log(1e-300) = -690 on 64 bit precision.
  log_eps_s = -80.

  below_precision_mask = log_eps_abs > _log_cdf_chernoff_bound(
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

    index = index - tf.cast(tf.where(update_mask, 1., 0.), dtype)

    return index, weight, adj_term, total_adj, prob

  def backwards_summation_loop_cond(index, weight, adj_term, total_adj, prob):
    return tf.reduce_any(~below_precision_mask & (index > 0.) & (
        tf.math.log(weight) + tf.math.log(adj_term) > log_eps_s))

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
    return tf.reduce_any(~below_precision_mask & (
        tf.math.log(weight) + tf.math.log(adj_term) > log_eps_s))

  _, _, _, _, prob = tf.while_loop(
      cond=forwards_summation_loop_cond,
      body=forwards_summation_loop_body,
      loop_vars=(central_index, central_weight, central_adjustment_term,
                 central_adjustment_term, prob))

  prob = tf.where(small_noncentrality_mask, tf.math.igamma(0.5 * df, 0.5 * x),
                  prob)

  x_above_mean_mask = x > (df + noncentrality)

  prob = tf.where(below_precision_mask & x_above_mean_mask, tf.ones_like(x),
                  prob)

  prob = tf.where(below_precision_mask & ~x_above_mean_mask, tf.zeros_like(x),
                  prob)

  return prob


def _log_cdf_chernoff_bound(x, df, noncentrality):
  """Implements a Chernoff bound on in the log domain on the CDF of a noncentral Chi2 random variable.

  Implements Eqs (22) and (23) in Shnidman, D. A. The Calculation of the
  Probability of Detection and the Generalized Marcum Q-Function (1989)
  The substitutions used are self.df = 2N, self.nc = 2NX and x = 2Y

  Args:
    x: `tf.Tensor` of points at which to evaluate the Chernoff bound.
    df: `tf.Tensor` of the degrees of freedom of the NC2 variate.
    noncentrality: `tf.Tensor` of the noncentrality of the NC2 variate.

  Returns:
    `tf.Tensor` of Chernoff bounds on the log CDF.
  """

  df_over_2x = tf.math.xdivy(df, 2. * x)

  # Quantity in Eq (22)
  chernoff_coef = 1. - df_over_2x - tf.math.sqrt(
      tf.math.square(df_over_2x) + tf.math.xdivy(noncentrality, x))

  # Exponent in Eq (23)
  bound = -chernoff_coef * x + noncentrality * chernoff_coef / (
      1. - chernoff_coef) - tf.math.xlog1py(df, -chernoff_coef)
  bound = 0.5 * bound

  return bound


def _nc2_cdf_fwd(x, df, noncentrality):
  output = _nc2_cdf_abramowitz_and_stegun(x, df, noncentrality)
  return output, (output, x, df, noncentrality)


def _nc2_cdf_bwd(aux, g):
  """Reverse mode implementation for `_nc2_cdf_abramowitz_and_stegun`."""
  # The gradients for the `noncentrality` parameter can be derived by
  # considering the infinite scale mixture of central Chi2s representation of
  # the noncentral Chi2 and noticing that after differentiating wrt the
  # noncentrality the result simplifies to a difference of two NC2 CDFs.
  nc2_cdf_1, x, df, noncentrality = aux

  # Compute gradient for `x`
  grad_x = tf.math.exp(_nc2_log_prob(x, df, noncentrality)) * g

  # Compute gradient for `noncentrality`
  nc2_cdf_2 = _nc2_cdf_naive(x, df + 2., noncentrality)

  grad_nc = 0.5 * (nc2_cdf_2 - nc2_cdf_1) * g

  grad_x, grad_nc = generic.fix_gradient_for_broadcasting([x, noncentrality],
                                                          [grad_x, grad_nc])

  return grad_x, None, grad_nc


def _nc2_cdf_jvp(primals, tangents):
  """Computes the JVP for `_nc2_cdf_abramowitz_and_stegun` (supports JAX custom derivative)."""
  x, df, noncentrality = primals
  d_x, _, d_noncentrality = tangents

  output = _nc2_cdf_naive(x, df, noncentrality)

  shape = ps.broadcast_shape(ps.shape(d_x), ps.shape(d_noncentrality))
  d_x = tf.broadcast_to(d_x, shape)
  d_noncentrality = tf.broadcast_to(d_noncentrality, shape)

  # Compute tangets for `x`
  d_x = tf.math.exp(_nc2_log_prob(x, df, noncentrality)) * d_x

  # Compute tangents for `noncentrality`
  nc2_cdf_1 = output
  nc2_cdf_2 = _nc2_cdf_naive(x, df + 2., noncentrality)

  d_noncentrality = 0.5 * (nc2_cdf_2 - nc2_cdf_1) * d_noncentrality

  tangents = d_x + d_noncentrality

  return output, tangents


@tfp_custom_gradient.custom_gradient(
    vjp_fwd=_nc2_cdf_fwd, vjp_bwd=_nc2_cdf_bwd, jvp_fn=_nc2_cdf_jvp)
def _nc2_cdf_naive(x, df, noncentrality):
  return _nc2_cdf_abramowitz_and_stegun(x, df, noncentrality)


def _nc2_quantile_numeric(p, df, noncentrality, tol):
  """Numerically inverts the CDF of the NC2 distribution.

  Uses the noncentral Wilson-Hilferty approximation from [1] (in the paper
  referred to as 'first approx') to guess the initial location for the
  root search.

  References
  [1] Abdel-Aty, S. H. (1954). Approximate formulae for the percentage points
      and the probability integral of the non-central χ2 distribution.
      Biometrika, 41(3/4), 538-540.

  Args:
    p: `tf.Tensor` of locations at which to invert the CDF.
    df: `tf.Tensor` of the degrees of freedom of the NC2 variate.
    noncentrality: `tf.Tensor` of the noncentrality of the NC2 variate.
    tol: `float` tolerance for the root search algorithm

  Returns:
  `tf.Tensor` of quantiles corresponding to the CDF locations.
  """
  # Ensure that pathological values don't affect the numerical root search

  numpy_dtype = dtype_util.as_numpy_dtype(p.dtype)

  p_zero_mask = tf.math.equal(p, 0.)
  p_one_mask = tf.math.equal(p, 1.)
  p_invalid_mask = tf.math.less(p, 0.) & tf.math.greater(p, 1.)

  safe_p = tf.where(
      p_zero_mask | p_one_mask | p_invalid_mask, 0.5 * tf.ones_like(p), p)

  df_plus_nc = df + noncentrality
  wilson_hilferty_coef = 2. / 9. * (df_plus_nc + noncentrality)
  wilson_hilferty_coef = wilson_hilferty_coef / tf.math.square(df_plus_nc)

  quant_approx = tf.math.ndtri(safe_p) * tf.math.sqrt(wilson_hilferty_coef)
  quant_approx = quant_approx + (1. - wilson_hilferty_coef)
  quant_approx = df_plus_nc * quant_approx**3.
  quant_approx = tf.nn.relu(quant_approx) + 1e-2  # Ensure approx is positive

  def objective_fn(ux):
    return _nc2_cdf_naive(tf.nn.softplus(ux), df, noncentrality) - safe_p

  unconstrained_numeric_root_search_results = root_search.find_root_secant(
      objective_fn=objective_fn,
      initial_position=generic.softplus_inverse(quant_approx),
      position_tolerance=tol,
      value_tolerance=tol)

  numeric_root = tf.nn.softplus(
      unconstrained_numeric_root_search_results.estimated_root)

  numeric_root = tf.where(p_zero_mask, tf.zeros_like(numeric_root),
                          numeric_root)
  numeric_root = tf.where(p_one_mask, numpy_dtype(np.inf), numeric_root)
  numeric_root = tf.where(p_invalid_mask, numpy_dtype(np.nan), numeric_root)

  return numeric_root


def _nc2_quantile_fwd(p, df, noncentrality, tol):
  output = _nc2_quantile_numeric(p, df, noncentrality, tol)
  return output, (output, p, df, noncentrality)


def _nc2_quantile_bwd(aux, g):
  output, p, df, noncentrality = aux

  # Compute the gradient for `p`
  grad_p = tf.math.exp(-_nc2_log_prob(output, df, noncentrality)) * g
  grad_p = generic.fix_gradient_for_broadcasting([p], [grad_p])

  return grad_p, None, None, None


def _nc2_quantile_jvp(primals, tangents):
  p, df, noncentrality, tol = primals
  d_p, _, _, _ = tangents

  output = _nc2_quantile_naive(p, df, noncentrality, tol)
  d_p = tf.math.exp(-_nc2_log_prob(output, df, noncentrality)) * d_p

  return output, d_p


@tfp_custom_gradient.custom_gradient(
    vjp_fwd=_nc2_quantile_fwd,
    vjp_bwd=_nc2_quantile_bwd,
    jvp_fn=_nc2_quantile_jvp)
def _nc2_quantile_naive(p, df, noncentrality, tol):
  return _nc2_quantile_numeric(p, df, noncentrality, tol)


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
    df = tf.convert_to_tensor(self.df, dtype=self.dtype)
    noncentrality = tf.convert_to_tensor(self.noncentrality, dtype=self.dtype)

    return _nc2_log_prob(x, df, noncentrality)

  def _cdf(self, x):
    df = tf.convert_to_tensor(self.df, dtype=self.dtype)
    noncentrality = tf.convert_to_tensor(self.noncentrality, dtype=self.dtype)

    param_shape = self._batch_shape_tensor(df=df, noncentrality=noncentrality)
    df = tf.broadcast_to(df, param_shape)
    noncentrality = tf.broadcast_to(noncentrality, param_shape)

    cdf = _nc2_cdf_naive(x, df, noncentrality)

    return distribution_util.extend_cdf_outside_support(x, cdf, low=0.)

  def _mean(self):
    return self.df + self.noncentrality

  def _variance(self):
    return 2. * self.df + 4. * self.noncentrality

  def _default_event_space_bijector(self):
    return softplus_bijector.Softplus(validate_args=self.validate_args)

  def _quantile(self, value):
    return self.quantile_approx(value)

  def quantile_approx(self, value, tol=None):
    """Approximates the quantile function of a noncentral X2 random variable by numerically inverting its CDF.

    Args:
      value: `tf.Tensor` of locations at which to invert the CDF.
      tol: Optional `tf.Tensor` that specifies the error tolerance of the root
      finding algorithm. Note, that the shape of `tol` must broadcast with
      `value`. If `None` is given, then the tolerance is set to the machine
      epsilon of the distribution's dtype. Defaults to `None`.

    Returns:
      `tf.Tensor` of approximate quantile values.
    """
    df = tf.convert_to_tensor(self.df, dtype=self.dtype)
    noncentrality = tf.convert_to_tensor(self.noncentrality, dtype=self.dtype)

    shape = ps.broadcast_shape(
        ps.shape(value),
        ps.broadcast_shape(ps.shape(df), ps.shape(noncentrality)))

    value = tf.broadcast_to(value, shape)
    df = tf.broadcast_to(df, shape)
    noncentrality = tf.broadcast_to(noncentrality, shape)

    numpy_dtype = dtype_util.as_numpy_dtype(self.dtype)

    if tol is None:
      tol = numpy_dtype(np.finfo(numpy_dtype).eps)
    else:
      tol = numpy_dtype(tol)
      tol = tf.broadcast_to(tol, shape)

    return _nc2_quantile_naive(value, df, noncentrality, tol)

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
