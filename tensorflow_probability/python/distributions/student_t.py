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
"""Student's t distribution class."""

import functools

import numpy as np
import tensorflow.compat.v2 as tf

from tensorflow_probability.python.bijectors import identity as identity_bijector
from tensorflow_probability.python.bijectors import softplus as softplus_bijector
from tensorflow_probability.python.distributions import distribution
from tensorflow_probability.python.distributions import gamma as gamma_lib
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
from tensorflow_probability.python.math import generic
from tensorflow_probability.python.math import gradient
from tensorflow_probability.python.math import numeric
from tensorflow_probability.python.math import special


__all__ = [
    'StudentT',
]


def sample_n(n, df, loc, scale, batch_shape, dtype, seed):
  """Draw n samples from a Student T distribution.

  Note that `scale` can be negative or zero.
  The sampling method comes from the fact that if:
    X ~ Normal(0, 1)
    Z ~ Chi2(df)
    Y = X / sqrt(Z / df)
  then:
    Y ~ StudentT(df)

  Args:
    n: int, number of samples
    df: Floating-point `Tensor`. The degrees of freedom of the
      distribution(s). `df` must contain only positive values.
    loc: Floating-point `Tensor`; the location(s) of the distribution(s).
    scale: Floating-point `Tensor`; the scale(s) of the distribution(s). Must
      contain only positive values.
    batch_shape: Callable to compute batch shape
    dtype: Return dtype.
    seed: PRNG seed; see `tfp.random.sanitize_seed` for details.

  Returns:
    samples: a `Tensor` with prepended dimensions `n`.
  """
  normal_seed, gamma_seed = samplers.split_seed(seed, salt='student_t')
  shape = ps.concat([[n], batch_shape], 0)

  normal_sample = samplers.normal(shape, dtype=dtype, seed=normal_seed)
  df = df * tf.ones(batch_shape, dtype=dtype)
  gamma_sample = gamma_lib.random_gamma(
      [n], concentration=0.5 * df, rate=0.5, seed=gamma_seed)
  samples = normal_sample * tf.math.rsqrt(gamma_sample / df)
  return samples * scale + loc


def log_prob(x, df, loc, scale):
  """Compute log probability of Student T distribution.

  Note that scale can be negative.

  Args:
    x: Floating-point `Tensor`. Where to compute the log probabilities.
    df: Floating-point `Tensor`. The degrees of freedom of the
      distribution(s). `df` must contain only positive values.
    loc: Floating-point `Tensor`; the location(s) of the distribution(s).
    scale: Floating-point `Tensor`; the scale(s) of the distribution(s).

  Returns:
    A `Tensor` with shape broadcast according to the arguments.
  """
  dtype = dtype_util.common_dtype([x, df, loc, scale], tf.float32)
  numpy_dtype = dtype_util.as_numpy_dtype(dtype)
  half = numpy_dtype(0.5)

  x, df, loc, scale = [
      tf.convert_to_tensor(param, dtype=dtype)
      for param in (x, df, loc, scale)]

  # Writing `y` this way reduces XLA mem copies.
  y = (x - loc) * (tf.math.rsqrt(df) / scale)
  log_unnormalized_prob = (-half * (df + numpy_dtype(1.)) *
                           numeric.log1psquare(y))
  log_normalization = (
      tf.math.log(tf.abs(scale)) + half * tf.math.log(df) +
      special.lbeta(half, half * df))
  return log_unnormalized_prob - log_normalization


def entropy(df, scale, batch_shape, dtype):
  """Compute entropy of the StudentT distribution.

  Args:
    df: Floating-point `Tensor`. The degrees of freedom of the
      distribution(s). `df` must contain only positive values.
    scale: Floating-point `Tensor`; the scale(s) of the distribution(s). Must
      contain only positive values.
    batch_shape: Floating-point `Tensor` of the batch shape
    dtype: Return dtype.

  Returns:
    A `Tensor` of the entropy for a Student's T with these parameters.
  """
  v = tf.ones(batch_shape, dtype=dtype)
  u = v * df
  return (tf.math.log(tf.abs(scale)) + 0.5 * tf.math.log(df) +
          special.lbeta(u / 2., v / 2.) + 0.5 * (df + 1.) *
          (tf.math.digamma(0.5 * (df + 1.)) - tf.math.digamma(0.5 * df)))


# The implementations of the Student's t-distribution cumulative distribution
# function and its inverse, respectively stdtr(df, t) and stdtrit(df, p), are
# based on ideas and equations available in the following references:
# [1] Geoffrey W. Hill
#     Algorithm 395: Student's t-distribution
#     Communications of the ACM, v. 13, n. 10, p. 617-619, 1970
#     https://doi.org/10.1145/355598.362775
# [2] Geoffrey W. Hill
#     Remark on "Algorithm 395: Student's t-Distribution [S14]"
#     ACM Transactions on Mathematical Software, v. 7, n. 2, p. 247-249, 1981
#     https://doi.org/10.1145/355945.355955
# [3] William Press, Saul Teukolsky, William Vetterling and Brian Flannery
#     Numerical Recipes: The Art of Scientific Computing
#     Cambridge University Press, 2007 (Third Edition)
#     http://numerical.recipes/book/book.html
# [4] Geoffrey W. Hill
#     Algorithm 396: Student's t-quantiles
#     Communications of the ACM, v. 13, n. 10, p. 619-620, 1970
#     https://doi.org/10.1145/355598.355600
# [5] Geoffrey W. Hill
#     Remark on "Algorithm 396: Student's t-Quantiles [S14]"
#     ACM Transactions on Mathematical Software, v. 7, n. 2, p. 250-251, 1981
#     https://doi.org/10.1145/355945.355956
# [6] R Core Team, R Foundation and Ross Ihaka
#     Mathlib: A C Library of Special Functions
#     https://svn.r-project.org/R/tags/R-4-2-1/src/nmath/qt.c


def _stdtr_asymptotic_expansion(df, t, numpy_dtype):
  """Compute `stdtr(df, t)` using asymptotic expansion."""
  # This function provides a fast approximation of stdtr(df, t) for large value
  # of df. It is based on an asymptotic normalizing expansion of Cornish-Fisher
  # type [1, 2].
  one = numpy_dtype(1.)
  two = numpy_dtype(2.)

  coeffs1 = [
      1.00000000000000000000E+0, 3.00000000000000000000E+0]

  coeffs2 = [
      4.00000000000000022204E-1, 3.29999999999999982236E+0,
      2.40000000000000000000E+1, 8.55000000000000000000E+1]

  coeffs3 = [
      3.04761904761904789396E-1, 3.75238095238095237249E+0,
      4.66714285714285708195E+1, 4.27500000000000000000E+2,
      2.58750000000000000000E+3, 8.51850000000000000000E+3]

  coeffs4 = [
      2.74285714285714299354E-1, 4.49904761904761940627E+0,
      7.84514285714285648510E+1, 1.11871071428571417528E+3,
      1.23876000000000003638E+4, 1.01024550000000002910E+5,
      5.59494000000000000000E+5, 1.76495962500000000000E+6]

  coeffs5 = [
      2.65974025974025973795E-1, 5.44969696969696926203E+0,
      1.22202943722943729199E+2, 2.35472987012987005073E+3,
      3.76250090259740245529E+4, 4.86996139285714307334E+5,
      4.96087065000000037253E+6, 3.79785955499999970198E+7,
      2.01505390875000000000E+8, 6.22437908625000000000E+8]

  terms_coeffs = [
      [numpy_dtype(c) for c in coeffs]
      for coeffs in (coeffs1, coeffs2, coeffs3, coeffs4, coeffs5)]

  df_minus_half = df - numpy_dtype(0.5)
  squared_z = df_minus_half * numeric.log1psquare(t * tf.math.rsqrt(df))
  z = tf.math.sqrt(squared_z)
  # To avoid overflow when df is huge, we manipulate b and the denominator of
  # each term of the expansion in the logarithmic space.
  log_b = tf.math.log(numpy_dtype(48.)) + tf.math.xlogy(two, df_minus_half)

  term_sign = one
  log_term_denominator = numpy_dtype(0.)
  # We initialize the series with its first term.
  series_sum = z
  last_index = len(terms_coeffs) - 1

  # We evaluate the next five terms using a procedure based on Horner's method.
  for index, coeffs in enumerate(terms_coeffs):
    if index < last_index:
      log_term_denominator = log_term_denominator + log_b
    else:
      log_term_denominator = log_term_denominator + tf.math.log(
          (numpy_dtype(0.43595) * squared_z + two) * squared_z +
          tf.math.exp(log_b) + numpy_dtype(537.))

    term_numerator = coeffs[0]
    for c in coeffs[1:]:
      term_numerator = c + term_numerator * squared_z

    term_numerator = term_numerator * z
    series_sum = series_sum + term_sign * tf.math.exp(
        tf.math.log(term_numerator) - log_term_denominator)
    term_sign = -one * term_sign

  return special_math.ndtr(tf.math.sign(t) * series_sum)


def _stdtr_computation(df, t):
  """Compute cumulative distribution function of Student T distribution."""
  dtype = dtype_util.common_dtype([df, t], tf.float32)
  numpy_dtype = dtype_util.as_numpy_dtype(dtype)
  half = numpy_dtype(0.5)
  one = numpy_dtype(1.)

  df, t = [tf.convert_to_tensor(param, dtype=dtype) for param in (df, t)]
  broadcast_shape = functools.reduce(
      ps.broadcast_shape, [ps.shape(df), ps.shape(t)])
  df, t = [tf.broadcast_to(param, broadcast_shape) for param in (df, t)]

  # For moderate df and relatively small t**2, or in case of large df, we use
  # asymptotic expansion [1, 2] to compute stdtr(df, t). The condition to use
  # it was specified by experimentation for np.float32 and was taken from [2,
  # page 249] for np.float64.

  if numpy_dtype == np.float32:
    use_asymptotic_expansion = (
        (df >= 10.) & (tf.math.square(t) < (16. * df - 5.))) | (df > 30.)
  else:
    use_asymptotic_expansion = (
        (df >= 100.) & (tf.math.square(t) < (0.1 * df - 5.))) | (df > 1000.)

  result = _stdtr_asymptotic_expansion(df, t, numpy_dtype)

  # Otherwise, we evaluate stdtr(df, t) using the regularized incomplete beta
  # function [3, page 323, equation 6.14.10]:
  #   stdtr(df, t) =
  #       0.5 * betainc(0.5 * df, 0.5, df / (df + t**2)), when t < 0
  #       1. - 0.5 * betainc(0.5 * df, 0.5, df / (df + t**2)), when t >= 0
  # To avoid rounding error when t**2 is small compared to df, we compute the
  # ratio df / (df + t**2) in the logarithmic space. If ratio > 0.99, we then
  # use the following symmetry relation:
  #   betainc(a, b, x) = 1 - betainc(b, a, 1 - x) .

  raw_ratio = t * tf.math.rsqrt(df)
  ratio = tf.math.exp(-numeric.log1psquare(raw_ratio))
  one_minus_ratio = tf.math.exp(
      -numeric.log1psquare(tf.math.reciprocal(raw_ratio)))

  # The maximum value for the ratio was set by experimentation.
  use_symmetry_relation = (ratio > 0.99)
  half_df = half * df
  a = tf.where(use_symmetry_relation, half, half_df)
  b = tf.where(use_symmetry_relation, half_df, half)
  x = tf.where(use_symmetry_relation, one_minus_ratio, ratio)

  y = special.betainc(a, b, x)
  result_betainc = half * tf.where(use_symmetry_relation, one - y, y)
  # Handle the case (t >= 0).
  result_betainc = tf.where(t >= 0., one - result_betainc, result_betainc)

  result = tf.where(use_asymptotic_expansion, result, result_betainc)

  # Determine if df is out of range (should return NaN output).
  result = tf.where(df <= 0., numpy_dtype(np.nan), result)

  return result


def _stdtr_partials(df, t):
  """Return the partial derivatives of `stdtr(df, t)`."""
  dtype = dtype_util.common_dtype([df, t], tf.float32)
  numpy_dtype = dtype_util.as_numpy_dtype(dtype)
  tiny = np.finfo(numpy_dtype).tiny
  eps = np.finfo(numpy_dtype).eps
  half = numpy_dtype(0.5)
  one = numpy_dtype(1.)

  df, t = [tf.convert_to_tensor(param, dtype=dtype) for param in (df, t)]
  broadcast_shape = functools.reduce(
      ps.broadcast_shape, [ps.shape(df), ps.shape(t)])
  df, t = [tf.broadcast_to(param, broadcast_shape) for param in (df, t)]

  # The gradient with respect to t can be computed more accurately and stably
  # using Student's t-distribution probability density function.

  stdtr_grad_t = tf.math.exp(log_prob(t, df, numpy_dtype(0.), one))

  # For moderate df and relatively small t**2, or in case of large df, we use
  # automatic differentiation of the procedure _stdtr_asymptotic_expansion to
  # compute the gradient with respect to df.

  if numpy_dtype == np.float32:
    use_asymptotic_expansion = (
        (df >= 10.) & (tf.math.square(t) < (16. * df - 5.))) | (df > 30.)
  else:
    use_asymptotic_expansion = (
        (df >= 100.) & (tf.math.square(t) < (0.1 * df - 5.))) | (df > 1000.)

  abs_t = tf.math.abs(t)
  min_abs_t = half * df * tf.math.pow(tiny, numpy_dtype(0.25))
  t_is_tiny = (abs_t < min_abs_t)
  df_asymptotic_expansion = tf.where(use_asymptotic_expansion, df, one)
  t_asymptotic_expansion = tf.where(
      use_asymptotic_expansion,
      # Mask out tiny t so the gradient correctly propagates. When t is tiny,
      # second derivative of log1psquare(t * tf.math.rsqrt(df)) can be NaN.
      tf.where(t_is_tiny, tf.where(t < 0., -min_abs_t, min_abs_t), t),
      one)

  stdtr_grad_df = gradient.value_and_gradient(
      lambda z: _stdtr_asymptotic_expansion(  # pylint: disable=g-long-lambda
          z, t_asymptotic_expansion, numpy_dtype),
      df_asymptotic_expansion)[1]
  # Handle the case (abs_t < min_abs_t): we use a rough linear approximation.
  stdtr_grad_df = tf.where(t_is_tiny, stdtr_grad_df * abs_t, stdtr_grad_df)

  # Otherwise, the gradient with respect to df is evaluated using the partial
  # derivatives of betainc(a, b, x). For t < 0, we have:
  #   stdtr_grad_df = 0.5 * (betainc_grad_a * a_grad_df +
  #                          betainc_grad_x * x_grad_df)
  #                 = 0.5 * (betainc_grad_a * a_grad_df +
  #                          2. * stdtr_grad_t / x_grad_t * x_grad_df)
  #                 = 0.5 * (betainc_grad_a * a_grad_df -
  #                          stdtr_grad_t * t / df) ,
  # where a = 0.5 * df and x = df / (df + t**2). In above equation, the second
  # equality follows from the fact that:
  #   stdtr_grad_t = 0.5 * betainc_grad_x * x_grad_t , for t < 0.
  # To avoid rounding error when t**2 is small compared to df, we compute the
  # ratio df / (df + t**2) in the logarithmic space. If ratio > 0.99, we then
  # use the following symmetry relation:
  #   betainc(a, b, x) = 1 - betainc(b, a, 1 - x) .

  min_abs_t_betainc = tf.math.sqrt(df * eps * tf.math.reciprocal(one - eps))
  t_is_small = (abs_t < min_abs_t_betainc)
  # Mask out small t so the gradient correctly propagates. When t is small,
  # ratio == 1 and one_minus_ratio < eps.
  abs_t_betainc = tf.where(t_is_small, min_abs_t_betainc, abs_t)

  raw_ratio = abs_t_betainc * tf.math.rsqrt(df)
  ratio = tf.math.exp(-numeric.log1psquare(raw_ratio))
  one_minus_ratio = tf.math.exp(
      -numeric.log1psquare(tf.math.reciprocal(raw_ratio)))

  # The maximum value for the ratio was set by experimentation.
  use_symmetry_relation = (ratio > 0.99)
  half_df = half * df
  a = tf.where(use_symmetry_relation, half, half_df)
  b = tf.where(use_symmetry_relation, half_df, half)
  x = tf.where(use_symmetry_relation, one_minus_ratio, ratio)

  # Prepare betainc inputs to make the evaluation of its gradients easier.
  use_betainc = ~use_asymptotic_expansion
  a = tf.where(use_betainc, a, half)
  b = tf.where(use_betainc, b, half)
  x = tf.where(use_betainc, x, half)

  betainc_grad_a, betainc_grad_b = gradient.value_and_gradient(
      lambda y, z: special.betainc(y, z, x), [a, b])[1]
  betainc_grad_a = tf.where(
      use_symmetry_relation, -betainc_grad_b, betainc_grad_a)

  stdtr_grad_df_betainc = half * (
      betainc_grad_a * half + stdtr_grad_t * abs_t / df)
  # Handle the case (t >= 0).
  stdtr_grad_df_betainc = tf.where(
      t >= 0., -stdtr_grad_df_betainc, stdtr_grad_df_betainc)
  # Handle the case (abs_t < min_abs_t_betainc): we use again a rough linear
  # approximation.
  stdtr_grad_df_betainc = tf.where(
      t_is_small, stdtr_grad_df_betainc * abs_t, stdtr_grad_df_betainc)

  stdtr_grad_df = tf.where(
      use_asymptotic_expansion, stdtr_grad_df, stdtr_grad_df_betainc)

  # Determine if df is out of range (should return NaN output).
  stdtr_grad_df, stdtr_grad_t = [
      tf.where(df <= 0., numpy_dtype(np.nan), grad)
      for grad in [stdtr_grad_df, stdtr_grad_t]]

  return stdtr_grad_df, stdtr_grad_t


def _stdtr_fwd(df, t):
  """Compute output, aux (it collaborates with _stdtr_bwd)."""
  output = _stdtr_computation(df, t)
  return output, (df, t)


def _stdtr_bwd(aux, g):
  """Reverse mode implementation for stdtr."""
  df, t = aux
  partial_df, partial_t = _stdtr_partials(df, t)
  return generic.fix_gradient_for_broadcasting(
      [df, t], [partial_df * g, partial_t * g])


def _stdtr_jvp(primals, tangents):
  """Compute JVP for stdtr (it supports JAX custom derivative)."""
  df, t = primals
  ddf, dt = tangents

  p = _stdtr_custom_gradient(df, t)
  partial_df, partial_t = _stdtr_partials(df, t)
  return (p, partial_df * ddf + partial_t * dt)


@tfp_custom_gradient.custom_gradient(
    vjp_fwd=_stdtr_fwd,
    vjp_bwd=_stdtr_bwd,
    jvp_fn=_stdtr_jvp)
def _stdtr_custom_gradient(df, t):
  """Compute `stdtr(df, t)` with correct custom gradient."""
  return _stdtr_computation(df, t)


def stdtr(df, t, name=None):
  """Compute cumulative distribution function of Student T distribution.

  This function returns the integral from minus infinity to `t` of Student T
  distribution with `df > 0` degrees of freedom.

  Args:
    df: Floating-point `Tensor`. The degrees of freedom of the
      distribution(s). `df` must contain only positive values.
    t: Floating-point `Tensor`. Where to compute the cumulative
      distribution function.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` with shape broadcast according to the arguments.

  Raises:
    TypeError: if `df` is not one of the following types: `float32`, `float64`.
  """
  with tf.name_scope(name or 'stdtr'):
    dtype = dtype_util.common_dtype([df, t], tf.float32)
    df = tf.convert_to_tensor(df, dtype=dtype)
    t = tf.convert_to_tensor(t, dtype=dtype)

    if dtype_util.as_numpy_dtype(dtype) not in [np.float32, np.float64]:
      raise TypeError(f'df.dtype={dtype} is not handled. '
                      'See docstring for supported types.')

    return _stdtr_custom_gradient(df, t)


def cdf(x, df, loc, scale):
  """Compute cumulative density function of Student T distribution.

  Note that scale can be negative.

  Args:
    x: Floating-point `Tensor`. Where to compute the cumulative
      distribution function.
    df: Floating-point `Tensor`. The degrees of freedom of the
      distribution(s). `df` must contain only positive values.
    loc: Floating-point `Tensor`; the location(s) of the distribution(s).
    scale: Floating-point `Tensor`; the scale(s) of the distribution(s).

  Returns:
    A `Tensor` with shape broadcast according to the arguments.
  """
  dtype = dtype_util.common_dtype([x, df, loc, scale], tf.float32)

  x, df, loc, scale = [
      tf.convert_to_tensor(param, dtype=dtype)
      for param in (x, df, loc, scale)]

  t = (x - loc) / tf.math.abs(scale)
  return stdtr(df, t)


def _stdtrit_betaincinv(df, p, numpy_dtype, use_betaincinv):
  """Compute `stdtrit(df, p)` using special.betaincinv."""
  # This function inverts the procedure that computes stdtr(df, t) using the
  # regularized incomplete beta function. For details on this procedure, see
  # the function _stdtr_computation.
  # We assume here that condition (p <= 0.5) is always true.
  half = numpy_dtype(0.5)
  one = numpy_dtype(1.)

  half_df = half * df
  two_p = numpy_dtype(2.) * p

  use_symmetry_relation = (
      p > (half * special.betainc(half_df, half, numpy_dtype(0.99))))
  a = tf.where(use_symmetry_relation, half, half_df)
  b = tf.where(use_symmetry_relation, half_df, half)
  y = tf.where(use_symmetry_relation, one - two_p, two_p)

  # Prepare betaincinv inputs to make its evaluation easier.
  a = tf.where(use_betaincinv, a, half)
  b = tf.where(use_betaincinv, b, half)
  y = tf.where(use_betaincinv, y, numpy_dtype(0.))

  x = special.betaincinv(a, b, y)

  log_abs_t = half * (
      tf.math.log(df) + tf.where(use_symmetry_relation, one, -one) * (
          tf.math.log(x) - tf.math.log1p(-x)))

  return -tf.math.exp(log_abs_t)


def _stdtrit_series_expansion(df, p, numpy_dtype):
  """Compute `stdtrit(df, p)` using series expansion."""
  # This function provides a fast approximation of stdtrit(df, p) for df >= 1.
  # It is based on an asymptotic inverse expansion of Cornish-Fisher type about
  # normal deviates. But for small p, where t**2 / df is large, a second series
  # expansion is used to achieve sufficient accuracy. Both approximations were
  # proposed in [4].
  # We assume here that condition (p <= 0.5) is always true.
  half, one, two, three, four, five, six, seven = [
      numpy_dtype(n) for n in (0.5,) + tuple(range(1, 8))]

  a = tf.math.reciprocal(df - half)
  b = numpy_dtype(48.) / tf.math.square(a)
  c = numpy_dtype(96.36) + a * (
      (numpy_dtype(20700.) * a / b - numpy_dtype(98.)) * a - numpy_dtype(16.))
  d = df * tf.math.sqrt(a * half * numpy_dtype(np.pi)) * (
      (numpy_dtype(94.5) / (b + c) - three) / b + one)

  # First series expansion: asymptotic inverse expansion about normal deviates.
  z = tf.math.ndtri(p)
  squared_z = tf.math.square(z)
  c = b + c + z * (
      ((numpy_dtype(0.05) * d * z - five) * z - seven) * z - two)
  c_correction = numpy_dtype(0.3) * (
      df - numpy_dtype(4.5)) * (z + numpy_dtype(0.6))
  c = tf.where(df >= 5., c, c + c_correction)

  squared_t_over_df = numpy_dtype(0.4) * squared_z + numpy_dtype(6.3)
  squared_t_over_df = squared_t_over_df * squared_z + numpy_dtype(36.)
  squared_t_over_df = squared_t_over_df * squared_z + numpy_dtype(94.5)
  squared_t_over_df = z * (
      (squared_t_over_df / c - squared_z - three) / b + one)
  squared_t_over_df = tf.math.expm1(a * tf.math.square(squared_t_over_df))

  # Second series expansion.
  y = tf.math.exp(two / df * (
      tf.math.log(d) + tf.math.log(two) + tf.math.log(p)))

  df_plus_2 = df + two
  large_squared_t_over_df = (df + six) / (df * y) - numpy_dtype(0.089) * d
  large_squared_t_over_df = df_plus_2 * three * (
      large_squared_t_over_df - numpy_dtype(0.822))
  large_squared_t_over_df = large_squared_t_over_df + half / (df + four)
  large_squared_t_over_df = y / large_squared_t_over_df - one
  large_squared_t_over_df = tf.math.reciprocal(y) + large_squared_t_over_df * (
      (df + one) / df_plus_2)

  p_is_not_small = (y >= (numpy_dtype(0.05) + a))
  # The condition to use the first series expansion was improved in [6].
  use_first_series_expansion = p_is_not_small | ((df < 2.1) & (p > 0.25))
  squared_t_over_df = tf.where(
      use_first_series_expansion, squared_t_over_df, large_squared_t_over_df)

  return -tf.math.sqrt(df * squared_t_over_df)


def _stdtrit_computation(df, p):
  """Return the inverse of `stdtr(df, t)` with respect to `t`."""
  # This function increases the accuracy of an initial estimate for t by using
  # Taylor series expansion iterations as proposed in [5].
  dtype = dtype_util.common_dtype([df, p], tf.float32)
  numpy_dtype = dtype_util.as_numpy_dtype(dtype)
  zero = numpy_dtype(0.)
  eps = np.finfo(numpy_dtype).eps
  half = numpy_dtype(0.5)
  one = numpy_dtype(1.)

  df, p = [tf.convert_to_tensor(param, dtype=dtype) for param in (df, p)]
  broadcast_shape = functools.reduce(
      ps.broadcast_shape, [ps.shape(df), ps.shape(p)])
  df, p = [tf.broadcast_to(param, broadcast_shape) for param in (df, p)]

  # max_iterations, use_betaincinv, and tolerance were set by experimentation.
  max_iterations = 3
  if numpy_dtype == np.float32:
    use_betaincinv = (df < 2.)
    tolerance = numpy_dtype(8.) * eps
  else:
    use_betaincinv = (df < 1.)
    tolerance = numpy_dtype(4096.) * eps

  adjusted_p = tf.where(p < 0.5, p, one - p)
  initial_candidate = tf.where(
      use_betaincinv,
      # Since _stdtrit_betaincinv is expensive, we pass use_betaincinv to it
      # to save computation.
      _stdtrit_betaincinv(df, adjusted_p, numpy_dtype, use_betaincinv),
      _stdtrit_series_expansion(df, adjusted_p, numpy_dtype))

  def taylor_expansion_improvement(should_stop, candidate):
    stdtr_grad_t = tf.math.exp(log_prob(candidate, df, zero, one))
    should_stop = should_stop | tf.math.equal(stdtr_grad_t, zero)

    first_order_correction = (adjusted_p - stdtr(df, candidate)) / stdtr_grad_t

    candidate_is_zero = tf.math.equal(candidate, zero)
    safe_inv_candidate = tf.where(
        candidate_is_zero, one, tf.math.reciprocal(candidate))
    second_order_correction = half * (df + one) * tf.math.square(
        first_order_correction) * safe_inv_candidate * tf.math.reciprocal(
            one + (df * safe_inv_candidate) * safe_inv_candidate)
    second_order_correction = tf.where(
        candidate_is_zero, zero, second_order_correction)

    correction = first_order_correction + second_order_correction
    new_candidate = tf.where(should_stop, candidate, candidate + correction)

    adjusted_tolerance = tf.math.abs(tolerance * new_candidate)
    should_stop = should_stop | (tf.math.abs(correction) <= adjusted_tolerance)

    return should_stop, new_candidate

  (_, result) = tf.while_loop(
      cond=lambda stop, _: tf.reduce_any(~stop),
      body=taylor_expansion_improvement,
      loop_vars=(
          ~tf.math.is_finite(initial_candidate),
          initial_candidate),
      maximum_iterations=max_iterations)

  # Handle the case (p >= 0.5).
  result = tf.math.sign(half - p) * result

  # Determine if the inputs are out of range (should return NaN output).
  result = tf.where((p <= zero) | (p >= one) | (df <= zero),
                    numpy_dtype(np.nan), result)

  return result


def _stdtrit_partials(df, p, return_value=False):
  """Return the partial derivatives of `stdtrit(df, p)`."""
  dtype = dtype_util.common_dtype([df, p], tf.float32)
  numpy_dtype = dtype_util.as_numpy_dtype(dtype)

  df, p = [tf.convert_to_tensor(param, dtype=dtype) for param in (df, p)]
  broadcast_shape = functools.reduce(
      ps.broadcast_shape, [ps.shape(df), ps.shape(p)])
  df, p = [tf.broadcast_to(param, broadcast_shape) for param in (df, p)]

  # We use the fact that stdtr and stdtrit are inverses of each other to
  # compute the gradients.
  t = _stdtrit_custom_gradient(df, p)
  stdtr_partial_df, stdtr_partial_t = _stdtr_partials(df, t)

  partial_df = -stdtr_partial_df / stdtr_partial_t
  partial_p = tf.math.reciprocal(stdtr_partial_t)

  if return_value:
    results = [partial_df, partial_p, t]
  else:
    results = [partial_df, partial_p]

  # Determine if the inputs are out of range (should return NaN output).
  results = [
      tf.where((p <= 0.) | (p >= 1.) | (df <= 0.), numpy_dtype(np.nan), result)
      for result in results]

  return results


def _stdtrit_fwd(df, p):
  """Compute output, aux (it collaborates with _stdtrit_bwd)."""
  output = _stdtrit_computation(df, p)
  return output, (df, p)


def _stdtrit_bwd(aux, g):
  """Reverse mode implementation for stdtrit."""
  df, p = aux
  partial_df, partial_p = _stdtrit_partials(df, p)
  return generic.fix_gradient_for_broadcasting(
      [df, p], [partial_df * g, partial_p * g])


def _stdtrit_jvp(primals, tangents):
  """Compute JVP for stdtrit (it supports JAX custom derivative)."""
  df, p = primals
  ddf, dp = tangents

  partial_df, partial_p, t = _stdtrit_partials(df, p, return_value=True)
  return (t, partial_df * ddf + partial_p * dp)


@tfp_custom_gradient.custom_gradient(
    vjp_fwd=_stdtrit_fwd,
    vjp_bwd=_stdtrit_bwd,
    jvp_fn=_stdtrit_jvp)
def _stdtrit_custom_gradient(df, p):
  """Compute `stdtrit(df, p)` with correct custom gradient."""
  return _stdtrit_computation(df, p)


def stdtrit(df, p, name=None):
  """Compute the inverse of `stdtr` with respect to `t`.

  This function returns a value `t` such that `p = stdtr(df, t)`.

  Args:
    df: Floating-point `Tensor`. The degrees of freedom of the
      distribution(s). `df` must contain only positive values.
    p: Floating-point `Tensor`. Probabilities from 0 to 1.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` with shape broadcast according to the arguments.

  Raises:
    TypeError: if `df` is not one of the following types: `float32`, `float64`.
  """
  with tf.name_scope(name or 'stdtrit'):
    dtype = dtype_util.common_dtype([df, p], tf.float32)
    df = tf.convert_to_tensor(df, dtype=dtype)
    p = tf.convert_to_tensor(p, dtype=dtype)

    if dtype_util.as_numpy_dtype(dtype) not in [np.float32, np.float64]:
      raise TypeError(f'df.dtype={dtype} is not handled. '
                      'See docstring for supported types.')

    return _stdtrit_custom_gradient(df, p)


def quantile(p, df, loc, scale):
  """Compute quantile function of Student T distribution.

  Note that scale can be negative.

  Args:
    p: Floating-point `Tensor`. Probabilities from 0 to 1.
    df: Floating-point `Tensor`. The degrees of freedom of the
      distribution(s). `df` must contain only positive values.
    loc: Floating-point `Tensor`; the location(s) of the distribution(s).
    scale: Floating-point `Tensor`; the scale(s) of the distribution(s).

  Returns:
    A `Tensor` with shape broadcast according to the arguments.
  """
  dtype = dtype_util.common_dtype([p, df, loc, scale], tf.float32)

  p, df, loc, scale = [
      tf.convert_to_tensor(param, dtype=dtype)
      for param in (p, df, loc, scale)]

  return loc + tf.math.abs(scale) * stdtrit(df, p)


class StudentT(distribution.AutoCompositeTensorDistribution):
  """Student's t-distribution.

  This distribution has parameters: degree of freedom `df`, location `loc`,
  and `scale`.

  #### Mathematical details

  The probability density function (pdf) is,

  ```none
  pdf(x; df, mu, sigma) = (1 + y**2 / df)**(-0.5 (df + 1)) / Z
  where,
  y = (x - mu) / sigma
  Z = abs(sigma) sqrt(df pi) Gamma(0.5 df) / Gamma(0.5 (df + 1))
  ```

  where:
  * `loc = mu`,
  * `scale = sigma`, and,
  * `Z` is the normalization constant, and,
  * `Gamma` is the [gamma function](
    https://en.wikipedia.org/wiki/Gamma_function).

  The StudentT distribution is a member of the [location-scale family](
  https://en.wikipedia.org/wiki/Location-scale_family), i.e., it can be
  constructed as,

  ```none
  X ~ StudentT(df, loc=0, scale=1)
  Y = loc + scale * X
  ```

  Notice that `scale` has semantics more similar to standard deviation than
  variance. However it is not actually the std. deviation; the Student's
  t-distribution std. dev. is `scale sqrt(df / (df - 2))` when `df > 2`.

  Samples of this distribution are reparameterized (pathwise differentiable).
  The derivatives are computed using the approach described in the paper

  [Michael Figurnov, Shakir Mohamed, Andriy Mnih.
  Implicit Reparameterization Gradients, 2018](https://arxiv.org/abs/1805.08498)

  #### Examples

  Examples of initialization of one or a batch of distributions.

  ```python
  import tensorflow_probability as tfp
  tfd = tfp.distributions

  # Define a single scalar Student t distribution.
  single_dist = tfd.StudentT(df=3)

  # Evaluate the pdf at 1, returning a scalar Tensor.
  single_dist.prob(1.)

  # Define a batch of two scalar valued Student t's.
  # The first has degrees of freedom 2, mean 1, and scale 11.
  # The second 3, 2 and 22.
  multi_dist = tfd.StudentT(df=[2, 3], loc=[1, 2.], scale=[11, 22.])

  # Evaluate the pdf of the first distribution on 0, and the second on 1.5,
  # returning a length two tensor.
  multi_dist.prob([0, 1.5])

  # Get 3 samples, returning a 3 x 2 tensor.
  multi_dist.sample(3)
  ```

  Arguments are broadcast when possible.

  ```python
  # Define a batch of two Student's t distributions.
  # Both have df 2 and mean 1, but different scales.
  dist = tfd.StudentT(df=2, loc=1, scale=[11, 22.])

  # Evaluate the pdf of both distributions on the same point, 3.0,
  # returning a length 2 tensor.
  dist.prob(3.0)
  ```

  Compute the gradients of samples w.r.t. the parameters:

  ```python
  df = tf.constant(2.0)
  loc = tf.constant(2.0)
  scale = tf.constant(11.0)
  dist = tfd.StudentT(df=df, loc=loc, scale=scale)
  samples = dist.sample(5)  # Shape [5]
  loss = tf.reduce_mean(tf.square(samples))  # Arbitrary loss function
  # Unbiased stochastic gradients of the loss function
  grads = tf.gradients(loss, [df, loc, scale])
  ```

  """

  def __init__(self,
               df,
               loc,
               scale,
               validate_args=False,
               allow_nan_stats=True,
               name='StudentT'):
    """Construct Student's t distributions.

    The distributions have degree of freedom `df`, mean `loc`, and scale
    `scale`.

    The parameters `df`, `loc`, and `scale` must be shaped in a way that
    supports broadcasting (e.g. `df + loc + scale` is a valid operation).

    Args:
      df: Floating-point `Tensor`. The degrees of freedom of the
        distribution(s). `df` must contain only positive values.
      loc: Floating-point `Tensor`. The mean(s) of the distribution(s).
      scale: Floating-point `Tensor`. The scaling factor(s) for the
        distribution(s). Note that `scale` is not technically the standard
        deviation of this distribution but has semantics more similar to
        standard deviation than variance.
      validate_args: Python `bool`, default `False`. When `True` distribution
        parameters are checked for validity despite possibly degrading runtime
        performance. When `False` invalid inputs may silently render incorrect
        outputs.
      allow_nan_stats: Python `bool`, default `True`. When `True`,
        statistics (e.g., mean, mode, variance) use the value '`NaN`' to
        indicate the result is undefined. When `False`, an exception is raised
        if one or more of the statistic's batch members are undefined.
      name: Python `str` name prefixed to Ops created by this class.

    Raises:
      TypeError: if loc and scale are different dtypes.
    """
    parameters = dict(locals())
    with tf.name_scope(name) as name:
      dtype = dtype_util.common_dtype([df, loc, scale], tf.float32)
      self._df = tensor_util.convert_nonref_to_tensor(
          df, name='df', dtype=dtype)
      self._loc = tensor_util.convert_nonref_to_tensor(
          loc, name='loc', dtype=dtype)
      self._scale = tensor_util.convert_nonref_to_tensor(
          scale, name='scale', dtype=dtype)
      dtype_util.assert_same_float_dtype((self._df, self._loc, self._scale))
      super(StudentT, self).__init__(
          dtype=dtype,
          reparameterization_type=reparameterization.FULLY_REPARAMETERIZED,
          validate_args=validate_args,
          allow_nan_stats=allow_nan_stats,
          parameters=parameters,
          name=name)

  @classmethod
  def _parameter_properties(cls, dtype, num_classes=None):
    # pylint: disable=g-long-lambda
    return dict(
        df=parameter_properties.ParameterProperties(
            default_constraining_bijector_fn=(
                lambda: softplus_bijector.Softplus(low=dtype_util.eps(dtype)))),
        loc=parameter_properties.ParameterProperties(),
        scale=parameter_properties.ParameterProperties(
            default_constraining_bijector_fn=(
                lambda: softplus_bijector.Softplus(low=dtype_util.eps(dtype)))))
    # pylint: enable=g-long-lambda

  @property
  def df(self):
    """Degrees of freedom in these Student's t distribution(s)."""
    return self._df

  @property
  def loc(self):
    """Locations of these Student's t distribution(s)."""
    return self._loc

  @property
  def scale(self):
    """Scaling factors of these Student's t distribution(s)."""
    return self._scale

  def _event_shape_tensor(self):
    return tf.constant([], dtype=tf.int32)

  def _event_shape(self):
    return tf.TensorShape([])

  def _sample_n(self, n, seed=None):
    df = tf.convert_to_tensor(self.df)
    loc = tf.convert_to_tensor(self.loc)
    scale = tf.convert_to_tensor(self.scale)
    batch_shape = self._batch_shape_tensor(df=df, loc=loc, scale=scale)
    return sample_n(
        n,
        df=df,
        loc=loc,
        scale=scale,
        batch_shape=batch_shape,
        dtype=self.dtype,
        seed=seed)

  def _log_prob(self, value):
    df = tf.convert_to_tensor(self.df)
    loc = tf.convert_to_tensor(self.loc)
    scale = tf.convert_to_tensor(self.scale)
    return log_prob(value, df, loc, scale)

  def _cdf(self, value):
    df = tf.convert_to_tensor(self.df)
    return cdf(value, df, self.loc, self.scale)

  def _survival_function(self, value):
    df = tf.convert_to_tensor(self.df)
    return cdf(-value, df, -self.loc, self.scale)

  def _quantile(self, value):
    df = tf.convert_to_tensor(self.df)
    return quantile(value, df, self.loc, self.scale)

  def _entropy(self):
    df = tf.convert_to_tensor(self.df)
    scale = tf.convert_to_tensor(self.scale)
    batch_shape = self._batch_shape_tensor(df=df, scale=scale)
    return entropy(df, scale, batch_shape, self.dtype)

  @distribution_util.AppendDocstring(
      """The mean of Student's T equals `loc` if `df > 1`, otherwise it is
      `NaN`. If `self.allow_nan_stats=False`, then an exception will be raised
      rather than returning `NaN`.""")
  def _mean(self):
    df = tf.convert_to_tensor(self.df)
    loc = tf.convert_to_tensor(self.loc)
    mean = loc * tf.ones(self._batch_shape_tensor(loc=loc),
                         dtype=self.dtype)
    if self.allow_nan_stats:
      return tf.where(
          df > 1.,
          mean,
          dtype_util.as_numpy_dtype(self.dtype)(np.nan))
    else:
      return distribution_util.with_dependencies([
          assert_util.assert_less(
              tf.ones([], dtype=self.dtype),
              df,
              message='mean not defined for components of df <= 1'),
      ], mean)

  @distribution_util.AppendDocstring("""
      The variance for Student's T equals

      ```
      df / (df - 2), when df > 2
      infinity, when 1 < df <= 2
      NaN, when df <= 1
      ```
      """)
  def _variance(self):
    df = tf.convert_to_tensor(self.df)
    scale = tf.convert_to_tensor(self.scale)
    # We need to put the tf.where inside the outer tf.where to ensure we never
    # hit a NaN in the gradient.
    denom = tf.where(df > 2., df - 2., tf.ones_like(df))
    # Abs(scale) superfluous.
    var = (tf.ones(self._batch_shape_tensor(df=df, scale=scale),
                   dtype=self.dtype)
           * tf.square(scale) * df / denom)
    # When 1 < df <= 2, variance is infinite.
    result_where_defined = tf.where(
        df > 2.,
        var,
        dtype_util.as_numpy_dtype(self.dtype)(np.inf))

    if self.allow_nan_stats:
      return tf.where(
          df > 1.,
          result_where_defined,
          dtype_util.as_numpy_dtype(self.dtype)(np.nan))
    else:
      return distribution_util.with_dependencies([
          assert_util.assert_less(
              tf.ones([], dtype=self.dtype),
              df,
              message='variance not defined for components of df <= 1'),
      ], result_where_defined)

  def _mode(self):
    loc = tf.convert_to_tensor(self.loc)
    return tf.broadcast_to(loc, self._batch_shape_tensor(loc=loc))

  def _parameter_control_dependencies(self, is_init):
    if not self.validate_args:
      return []
    assertions = []
    if is_init != tensor_util.is_ref(self._df):
      assertions.append(assert_util.assert_positive(
          self._df, message='Argument `df` must be positive.'))
    return assertions

  def _default_event_space_bijector(self):
    # TODO(b/145620027) Finalize choice of bijector (consider one that
    # transforms away the heavy tails).
    return identity_bijector.Identity(validate_args=self.validate_args)
