# Copyright 2020 The TensorFlow Probability Authors.
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
"""Implements special functions in TensorFlow."""

from __future__ import absolute_import
from __future__ import division
# [internal] enable type annotations
from __future__ import print_function

# Dependency imports
import numpy as np
import tensorflow.compat.v2 as tf
from tensorflow_probability.python.internal import custom_gradient as tfp_custom_gradient
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import prefer_static
from tensorflow_probability.python.internal import tensorshape_util


__all__ = [
    'atan_difference',
    'dawsn',
    'erfcinv',
    'erfcx',
    'igammainv',
    'igammacinv',
    'round_exponential_bump_function',
    'lambertw',
    'lambertw_winitzki_approx',
    'logerfc',
    'logerfcx',
    'log_gamma_correction',
    'log_gamma_difference',
    'lbeta',
    'owens_t',
]


def atan_difference(x, y, name=None):
  """Difference of arctan(x) and arctan(y).

  Computes arctan(x) - arctan(y) avoiding catastrophic cancellation. This is
  by resorting to the identity:

  ```none
  arctan(x) - arctan(y) = arctan((x - y) / (1 + x * y)) +
                          pi * sign(x) * 1_{x * y < -1)
  ```

  where `1_A` is the indicator function on the set `A`.

  For a derivation of this fact, see [1].


  #### References
  [1] De Stefano, Sum of Arctangents
      https://sites.google.com/site/micdestefano/mathematics/trigonometry/sum-of-arctangents

  Args:
    x: Floating-point Tensor. Should be broadcastable with `y`.
    y: Floating-point Tensor. Should be broadcastable with `x`.
    name: Optional Python `str` naming the operation.

  Returns:
    z: Tensor of same shape and dtype as `x` and `y`.
  """
  with tf.name_scope(name or 'atan_difference'):
    dtype = dtype_util.common_dtype([x, y], tf.float32)
    x = tf.convert_to_tensor(x, dtype=dtype)
    y = tf.convert_to_tensor(y, dtype=dtype)

    difference = tf.math.atan((x - y) / (1 + x * y))
    difference = difference + tf.where(
        x * y < - 1., np.pi * tf.math.sign(x), 0.)
    difference = tf.where(
        tf.math.equal(x * y, -1.), np.pi * tf.math.sign(x) / 2., difference)

    return difference


def _dawsn_naive(x):
  """Returns the Dawson Integral computed at x elementwise."""
  dtype = dtype_util.common_dtype([x], tf.float32)
  numpy_dtype = dtype_util.as_numpy_dtype(dtype)
  x = tf.convert_to_tensor(x, dtype=dtype)

  n1 = [
      1.13681498971755972054E-11,
      8.49262267667473811108E-10,
      1.94434204175553054283E-8,
      9.53151741254484363489E-7,
      3.07828309874913200438E-6,
      3.52513368520288738649E-4,
      -8.50149846724410912031E-4,
      4.22618223005546594270E-2,
      -9.17480371773452345351E-2,
      9.99999999999999994612E-1]

  d1 = [
      2.40372073066762605484E-11,
      1.48864681368493396752E-9,
      5.21265281010541664570E-8,
      1.27258478273186970203E-6,
      2.32490249820789513991E-5,
      3.25524741826057911661E-4,
      3.48805814657162590916E-3,
      2.79448531198828973716E-2,
      1.58874241960120565368E-1,
      5.74918629489320327824E-1,
      1.00000000000000000539E0]

  n2 = [
      5.08955156417900903354E-1,
      -2.44754418142697847934E-1,
      9.41512335303534411857E-2,
      -2.18711255142039025206E-2,
      3.66207612329569181322E-3,
      -4.23209114460388756528E-4,
      3.59641304793896631888E-5,
      -2.14640351719968974225E-6,
      9.10010780076391431042E-8,
      -2.40274520828250956942E-9,
      3.59233385440928410398E-11]

  d2 = [
      1.00000000000000000000E0,
      -6.31839869873368190192E-1,
      2.36706788228248691528E-1,
      -5.31806367003223277662E-2,
      8.48041718586295374409E-3,
      -9.47996768486665330168E-4,
      7.81025592944552338085E-5,
      -4.55875153252442634831E-6,
      1.89100358111421846170E-7,
      -4.91324691331920606875E-9,
      7.18466403235734541950E-11]

  n3 = [
      -5.90592860534773254987E-1,
      6.29235242724368800674E-1,
      -1.72858975380388136411E-1,
      1.64837047825189632310E-2,
      -4.86827613020462700845E-4]

  d3 = [
      1.00000000000000000000E0,
      -2.69820057197544900361E0,
      1.73270799045947845857E0,
      -3.93708582281939493482E-1,
      3.44278924041233391079E-2,
      -9.73655226040941223894E-4]

  n1, d1, n2, d2, n3, d3 = [
      [numpy_dtype(c) for c in lst] for lst in (n1, d1, n2, d2, n3, d3)]

  abs_x = tf.math.abs(x)

  result_small = abs_x * tf.math.polyval(
      n1, tf.math.square(x)) / tf.math.polyval(d1, tf.math.square(x))
  result_small = tf.math.sign(x) * result_small

  inv_xsq = tf.math.reciprocal(tf.math.square(x))
  result_medium = tf.math.reciprocal(abs_x) + inv_xsq * (
      tf.math.polyval(n2, inv_xsq) / (abs_x * tf.math.polyval(d2, inv_xsq)))
  result_medium = 0.5 * tf.math.sign(x) * result_medium

  result_very_large = 0.5 * tf.math.sign(x) * tf.math.reciprocal(abs_x)

  result_large = tf.math.reciprocal(abs_x) + inv_xsq * (
      tf.math.polyval(n3, inv_xsq) / (abs_x * tf.math.polyval(d3, inv_xsq)))
  result_large = 0.5 * tf.math.sign(x) * result_large

  return tf.where(
      abs_x < 3.25,
      result_small,
      tf.where(
          abs_x < 6.25,
          result_medium,
          tf.where(
              abs_x > 1e9,
              result_very_large,
              result_large)))


def _dawsn_fwd(x):
  """Compute output, aux (collaborates with _dawsn_bwd)."""
  output = _dawsn_naive(x)
  return output, (x,)


def _dawsn_bwd(aux, g):
  """Reverse mode impl for dawsn."""
  x, = aux
  y = _dawsn_custom_gradient(x)
  return g * (1. - 2 * x * y)


def _dawsn_jvp(primals, tangents):
  """Computes JVP for dawsn (supports JAX custom derivative)."""
  x, = primals
  dx, = tangents

  y = _dawsn_custom_gradient(x)
  return y, dx * (1. - 2 * x * y)


@tfp_custom_gradient.custom_gradient(
    vjp_fwd=_dawsn_fwd,
    vjp_bwd=_dawsn_bwd,
    jvp_fn=_dawsn_jvp)
def _dawsn_custom_gradient(x):
  return _dawsn_naive(x)


def dawsn(x, name=None):
  """Computes Dawson's integral element-wise.

  Dawson's integral is defined as `exp(-x**2) * int_0^x exp(t**2)`
  with the domain of definition all real numbers.

  This implementation is based on the Cephes math library.

  Args:
    x: A Tensor with type `float32` or `float64`.
    name: A name for the operation (optional).

  Returns:
    dawsn: dawsn evaluated at `x`. A Tensor with the same shape and same
      dtype as `x`.
  """
  with tf.name_scope(name or 'dawsn'):
    return _dawsn_custom_gradient(x)


def erfcinv(z, name=None):
  """Computes the inverse of `tf.math.erfc` of `z` element-wise.

  NOTE: This is mathematically equivalent to computing `erfinv(1 - x)`
  however is more numerically stable.

  Args:
    z: A Tensor with type `float32` or `float64`.
    name: A name for the operation (optional).

  Returns:
    erfcinv: erfcinv evaluated at `z`. A Tensor with the same shape and same
      dtype as `z`.
  """
  with tf.name_scope(name or 'erfcinv'):
    z = tf.convert_to_tensor(z)
    return -tf.math.ndtri(0.5 * z) * np.sqrt(0.5)


def _erfcx_naive(x):
  """Compute erfcx using a Chebyshev expansion."""
  # The implementation is based on
  # [1] M. Shepherd and J. Laframboise,
  #     Chebyshev approximation of (1 + 2 * x) * exp(x**2) * erfc(x)
  #     https://www.ams.org/journals/mcom/1981-36-153/S0025-5718-1981-0595058-X/

  dtype = dtype_util.common_dtype([x], tf.float32)
  numpy_dtype = dtype_util.as_numpy_dtype(dtype)
  x = tf.convert_to_tensor(x, dtype=dtype)
  x_abs = tf.math.abs(x)
  # TODO(b/180390310): The approximation quality can be made better by sweeping
  # the shift parameter '3.75'.
  y = (x_abs - 3.75) / (x_abs + 3.75)

  # The list of coefficients is taken from [1].
  coeff = [
      3e-21,
      9.7e-20,
      2.7e-20,
      -2.187e-18,
      -2.237e-18,
      5.0681e-17,
      7.4182e-17,
      -1.250795e-15,
      -1.864563e-15,
      3.33478119e-14,
      3.2525481e-14,
      -9.65469675e-13,
      1.94558685e-13,
      2.8687950109e-11,
      -6.3180883409e-11,
      -7.75440020883e-10,
      4.521959811218e-09,
      1.0764999465671e-08,
      -2.18864010492344e-07,
      7.74038306619849e-07,
      4.139027986073010e-06,
      -6.9169733025012064e-05,
      4.90775836525808632e-04,
      -2.413163540417608191e-03,
      9.074997670705265094e-03,
      -2.6658668435305752277e-02,
      5.9209939998191890498e-02,
      -8.4249133366517915584e-02,
      -4.590054580646477331e-03,
      1.177578934567401754080,
  ]

  result = -4e-21
  previous_result = 0.
  for i in range(len(coeff) - 1):
    result, previous_result = (
        2 * y * result - previous_result + coeff[i], result)
  result = y * result - previous_result + coeff[len(coeff) - 1]

  result = result / (1. + 2. * x_abs)

  # The approximation is only valid for positive x, so flip the integral.
  # TODO(b/180390310): Improve this approximation for negative values.
  result = tf.where(
      x < 0., 2. * tf.math.exp(tf.math.square(x)) - result, result)
  result = tf.where(tf.math.equal(x, np.inf), numpy_dtype(1.), result)
  return result


def _erfcx_fwd(x):
  """Compute output, aux (collaborates with _erfcx_bwd)."""
  output = _erfcx_naive(x)
  return output, (x,)


def _erfcx_bwd(aux, g):
  x, = aux
  y = _erfcx_custom_gradient(x)
  numpy_dtype = dtype_util.as_numpy_dtype(
      dtype_util.common_dtype([x], tf.float32))
  px = 2. * x * y - numpy_dtype(2. / np.sqrt(np.pi))
  return [px * g]


def _erfcx_jvp(primals, tangents):
  """Computes JVP for erfcx (supports JAX custom derivative)."""
  x, = primals
  dx, = tangents

  y = _erfcx_custom_gradient(x)
  numpy_dtype = dtype_util.as_numpy_dtype(
      dtype_util.common_dtype([x], tf.float32))
  px = 2. * x * y - numpy_dtype(2. / np.sqrt(np.pi))
  return y, px * dx


@tfp_custom_gradient.custom_gradient(
    vjp_fwd=_erfcx_fwd,
    vjp_bwd=_erfcx_bwd,
    jvp_fn=_erfcx_jvp)
def _erfcx_custom_gradient(x):
  """Computes Erfcx(x) with correct custom gradient."""
  return _erfcx_naive(x)


def erfcx(x, name=None):
  """Computes the scaled complementary error function exp(x**) * erfc(x).

  # References
  [1] M. Shepherd and J. Laframboise,
      Chebyshev approximation of (1 + 2 * x) * exp(x**2) * erfc(x)
      https://www.ams.org/journals/mcom/1981-36-153/S0025-5718-1981-0595058-X/

  Args:
    x: A Tensor with type `float32` or `float64`.
    name: A name for the operation (optional).

  Returns:
    erfcx: erfcx(x) evaluated at `x`. A Tensor with the same shape and same
      dtype as `x`.
  """
  with tf.name_scope(name or 'logerfc'):
    dtype = dtype_util.common_dtype([x], tf.float32)
    x = tf.convert_to_tensor(x, dtype=dtype)
    return _erfcx_custom_gradient(x)


def logerfc(x, name=None):
  """Computes the logarithm of `tf.math.erfc` of `x` element-wise.

  NOTE: This is mathematically equivalent to computing `log(erfc(x))`
  however is more numerically stable.

  Args:
    x: A Tensor with type `float32` or `float64`.
    name: A name for the operation (optional).

  Returns:
    logerfc: log(erfc(x)) evaluated at `x`. A Tensor with the same shape and
      same dtype as `x`.
  """
  with tf.name_scope(name or 'logerfc'):
    dtype = dtype_util.common_dtype([x], tf.float32)
    x = tf.convert_to_tensor(x, dtype=dtype)
    safe_positive_x = tf.where(x > 0., x, 1.)
    safe_negative_x = tf.where(x < 0., x, -1.)
    return tf.where(
        x < 0.,
        tf.math.log(tf.math.erfc(safe_negative_x)),
        # erfcx saturates to zero much slower than erfc.
        tf.math.log(erfcx(safe_positive_x)) - tf.math.square(safe_positive_x))


def logerfcx(x, name=None):
  """Computes the logarithm of `tfp.math.erfcx` of `x` element-wise.

  NOTE: This is mathematically equivalent to computing `log(erfcx(x))`
  however is more numerically stable.

  Args:
    x: A Tensor with type `float32` or `float64`.
    name: A name for the operation (optional).

  Returns:
    logerfcx: log(erfcx(x)) evaluated at `x`. A Tensor with the same shape and
      same dtype as `x`.
  """
  with tf.name_scope(name or 'logerfc'):
    dtype = dtype_util.common_dtype([x], tf.float32)
    x = tf.convert_to_tensor(x, dtype=dtype)
    safe_positive_x = tf.where(x > 0., x, 1.)
    safe_negative_x = tf.where(x < 0., x, -1.)
    return tf.where(
        x < 0.,
        # erfcx goes to infinity fast in the left tail.
        tf.math.log(
            tf.math.erfc(safe_negative_x)) + tf.math.square(safe_negative_x),
        tf.math.log(erfcx(safe_positive_x)))


# Implementation of Inverse Incomplete Gamma based on
# A. Didonato and A. Morris,
# Computation of the Incomplete Gamma Function Ratios and their Inverse
# https://dl.acm.org/doi/10.1145/22721.23109


def _didonato_eq_twenty_three(log_b, v, a):
  return -log_b + tf.math.xlogy(a - 1., v) - tf.math.log1p((1. - a) / (1. + v))


def _didonato_eq_thirty_two(p, q):
  """Compute Equation 32 from Didonato's paper."""
  dtype = dtype_util.common_dtype([p, q], tf.float32)
  numpy_dtype = dtype_util.as_numpy_dtype(dtype)
  numerator_coeffs = [
      0.213623493715853, 4.28342155967104, 11.6616720288968, 3.31125922108741]
  numerator_coeffs = [numpy_dtype(c) for c in numerator_coeffs]
  denominator_coeffs = [
      0.36117081018842e-1, 1.27364489782223, 6.40691597760039,
      6.61053765625462, 1.]
  denominator_coeffs = [numpy_dtype(c) for c in denominator_coeffs]
  t = tf.where(
      p < 0.5,
      tf.math.sqrt(-2 * tf.math.log(p)),
      tf.math.sqrt(-2. * tf.math.log(q)))
  result = (t - tf.math.polyval(numerator_coeffs, t) / tf.math.polyval(
      denominator_coeffs, t))
  return tf.where(p < 0.5, -result, result)


def _didonato_eq_thirty_four(a, x):
  """Compute Equation 34 from Didonato's paper."""
  # This function computes `S_n` in equation thirty four.
  dtype = dtype_util.common_dtype([a, x], tf.float32)

  # TODO(b/178793508): Change this tolerance to be dtype dependent.
  tolerance = 1e-4

  def _taylor_series(should_stop, index, partial, series_sum):
    partial = partial * x / (a + index)
    series_sum = tf.where(should_stop, series_sum, series_sum + partial)
    # TODO(b/178793508): Change the number of iterations to be dtype dependent.
    should_stop = (partial < tolerance) | (index > 100)
    return should_stop, index + 1, partial, series_sum

  _, _, _, series_sum = tf.while_loop(
      cond=lambda stop, *_: tf.reduce_any(~stop),
      body=_taylor_series,
      loop_vars=(
          tf.zeros_like(a + x, dtype=tf.bool),
          tf.cast(1., dtype=dtype),
          tf.ones_like(a + x, dtype=dtype),
          tf.ones_like(a + x, dtype=dtype)))
  return series_sum


def _didonato_eq_twenty_five(a, y):
  """Compute Equation 25 from Didonato's paper."""
  c1 = tf.math.xlogy(a - 1., y)
  c1_sq = tf.math.square(c1)
  c1_cub = c1_sq * c1
  c1_fourth = tf.math.square(c1_sq)
  a_sq = tf.math.square(a)
  a_cub = a_sq * a
  c2 = (a - 1.) * (1. + c1)
  c3 = (a - 1.) * ((3. * a - 5.) / 2. + c1 * (a - 2. - c1 / 2.))
  c4 = (a - 1.) * (
      (c1_cub / 3.) - (3. * a - 5.) * c1_sq / 2. +
      (a_sq - 6. * a + 7.) * c1 + (11. * a_sq - 46. * a + 47.) / 6.)
  c5 = ((a - 1.) * (-c1_fourth / 4. +
                    (11. * a - 17.) * c1_cub / 6 +
                    (-3. * a_sq + 13. * a - 13.) * c1_sq +
                    (2. * a_cub - 25. * a_sq + 72. * a - 61.) * c1 / 2. +
                    (25. * a_cub - 195. * a_sq + 477 * a - 379) / 12.))
  return y + c1 + (((c5 / y + c4) / y + c3 / y) + c2) / y


def _inverse_igamma_initial_approx(a, p, q, use_p_for_logq=True):
  """Compute an initial guess for `igammainv(a, p)`.

  Compute an initial estimate of `igammainv(a, p)`. This will be further
  refined by Newton-Halley iterations.

  Args:
    a: A positive `float` `Tensor`. Must be broadcastable with `p`.
    p: A `float` `Tensor` whose entries lie in `[0, 1]`.
       Must be broadcastable with `a`. This is `1 - q`.
    q: A `float` `Tensor` whose entries lie in `[0, 1]`.
       Must be broadcastable with `a`. This is `1 - p`.
    use_p_for_logq: `bool` describing whether to compute
      `log(q)` by using `log(1 - p)` or `log(q)`.
      Default value: `True`.

  Returns:
    igamma_approx: Approximation to `igammainv(a, p)`.
  """

  dtype = dtype_util.common_dtype([a, p, q], tf.float32)
  numpy_dtype = dtype_util.as_numpy_dtype(dtype)
  a = tf.convert_to_tensor(a, dtype=dtype)
  p = tf.convert_to_tensor(p, dtype=dtype)
  q = tf.convert_to_tensor(q, dtype=dtype)

  lgamma_a = tf.math.lgamma(a)

  # This ensures that computing log(1 - p) avoids roundoff errors. This is
  # needed since igammacinv and igammainv both use this codepath,
  if use_p_for_logq:
    log_q = tf.math.log1p(-p)
  else:
    log_q = tf.math.log(q)

  log_b = log_q + lgamma_a

  result = _didonato_eq_twenty_five(a, -log_b)

  # The code below is for when a < 1.

  v = -log_b - (1. - a) * tf.math.log(-log_b)
  v_sq = tf.math.square(v)

  # This is Equation 24.
  result = tf.where(
      log_b > np.log(0.01),
      -log_b - (1. - a) * tf.math.log(v) - tf.math.log(
          (v_sq + 2. * (3. - a) * v + (2. - a) * (3 - a)) /
          (v_sq + (5. - a) * v + 2.)),
      result)

  result = tf.where(
      log_b >= np.log(0.15),
      _didonato_eq_twenty_three(log_b, v, a),
      result)

  t = tf.math.exp(-np.euler_gamma - tf.math.exp(log_b))
  u = t * tf.math.exp(t)
  result = tf.where(
      (a < 0.3) & (log_b >= np.log(0.35)),
      t * tf.math.exp(u),
      result)

  # These are hand tuned constants to compute (p * Gamma(a + 1)) ** (1 / a)
  # TODO(b/178793508): Change these bounds / computation to be dtype dependent.
  # This is Equation 21.
  u = tf.where((tf.math.exp(log_b) * q > 1e-8) & (q > 1e-5),
               tf.math.pow(p * tf.math.exp(lgamma_a) * a,
                           tf.math.reciprocal(a)),
               # When (1 - p) * Gamma(a) or (1 - p) is small,
               # we can taylor expand Gamma(a + 1) ** 1 / a to get
               # exp(-euler_gamma for the zeroth order term.
               # Also p ** 1 / a = exp(log(p) / a) = exp(log(1 - q) / a)
               # ~= exp(-q / a) resulting in the following expression.
               tf.math.exp((-q / a) - np.euler_gamma))

  result = tf.where(
      (log_b > np.log(0.6)) | ((log_b >= np.log(0.45)) & (a >= 0.3)),
      u / (1. - (u / (a + 1.))),
      result)

  # The code below is for when a < 1.

  sqrt_a = tf.math.sqrt(a)
  s = _didonato_eq_thirty_two(p, q)
  s_sq = tf.math.square(s)
  s_cub = s_sq * s
  s_fourth = tf.math.square(s_sq)
  s_fifth = s_fourth * s

  # This is the Cornish-Fisher 6 term expansion for x (by viewing igammainv as
  # the quantile function for the Gamma distribution). This is equation (31).
  w = a + s * sqrt_a + (s_sq - 1.) / 3.
  w = w + (s_cub - 7. * s) / (36. * sqrt_a)
  w = w - (3. * s_fourth + 7. * s_sq - 16.) / (810 * a)
  w = w + (9. * s_fifth + 256. * s_cub - 433. * s) / (38880 * a * sqrt_a)

  # The code below is for when a > 1. and p > 0.5.
  d = tf.math.maximum(numpy_dtype(2.), a * (a - 1.))
  result_a_large_p_large = tf.where(
      log_b <= -d * np.log(10.),
      _didonato_eq_twenty_five(a, -log_b),
      _didonato_eq_twenty_three(
          log_b, _didonato_eq_twenty_three(log_b, w, a), a))
  result_a_large_p_large = tf.where(w < 3. * a, w, result_a_large_p_large)
  # TODO(b/178793508): Change these bounds / computation to be dtype dependent.
  result_a_large_p_large = tf.where(
      (a >= 500.) & (tf.math.abs(1. - w / a) < 1e-6),
      w, result_a_large_p_large)

  # The code below is for when a > 1. and p <= 0.5.
  z = w
  v = tf.math.log(p) + tf.math.lgamma(a + 1.)

  # The code below follows Equation 35 which involves multiple evaluations of
  # F_i.
  modified_z = tf.math.exp((v + w) / a)
  for _ in range(2):
    s = tf.math.log1p(
        modified_z / (a + 1.) * (
            1. + modified_z / (a + 2.)))
    modified_z = tf.math.exp(
        (v + modified_z - s) / a)

  s = tf.math.log1p(
      modified_z / (a + 1.) * (1. + modified_z / (a + 2.) * (
          1. + modified_z / (a + 3.))))
  modified_z = tf.math.exp((v + modified_z - s) / a)
  z = tf.where(w <= 0.15 * (a + 1.), modified_z, z)

  ls = tf.math.log(_didonato_eq_thirty_four(a, z))
  medium_z = tf.math.exp((v + z - ls) / a)
  result_a_large_p_small = tf.where(
      (z <= 0.01 * (a + 1.)) | (z > 0.7 * (a + 1.)),
      z,
      medium_z * (
          1. - (
              a * tf.math.log(medium_z) - medium_z - v + ls) / (a - medium_z)))

  result_a_large = tf.where(
      p <= 0.5, result_a_large_p_small, result_a_large_p_large)
  result = tf.where(a < 1., result, result_a_large)

  # This ensures that computing log(1 - p) avoids roundoff errors. This is
  # needed since igammacinv and igammainv both use this codepath,
  # switching p and q.
  result = tf.where(tf.math.equal(a, 1.), -log_q, result)
  return result


def _shared_igammainv_computation(a, p, is_igammainv=True):
  """Shared computation for the igammainv/igammacinv."""

  dtype = dtype_util.common_dtype([a, p], tf.float32)
  numpy_dtype = dtype_util.as_numpy_dtype(dtype)

  if is_igammainv:
    q = 1. - p
  else:
    q = p
    p = 1. - q

  x = _inverse_igamma_initial_approx(a, p, q, use_p_for_logq=is_igammainv)

  # Run 3 steps of Newton-Halley method.
  for _ in range(3):
    factorial = tf.math.exp(a * tf.math.log(x) - x - tf.math.lgamma(a))

    f_over_der = tf.where(
        ((p <= 0.9) & is_igammainv) | ((q > 0.9) & (not is_igammainv)),
        (tf.math.igamma(a, x) - p) * x / factorial,
        -(tf.math.igammac(a, x) - q) * x / factorial)
    second_der_over_der = -1. + (a - 1.) / x
    modified_x = tf.where(
        tf.math.is_inf(second_der_over_der),
        # Use Newton's method if the second derivative is not available.
        x - f_over_der,
        # Use Halley's method otherwise. Halley's method is:
        # x_{n+1} = x_n - f(x_n) / f'(x_n) * (
        #    1 - f(x_n) / f'(x_n) * 0.5 f''(x_n) / f'(x_n))
        x - f_over_der / (1. - 0.5 * f_over_der * second_der_over_der))
    x = tf.where(tf.math.equal(factorial, 0.), x, modified_x)
  x = tf.where((a < 0.) | (p < 0.) | (p > 1.), numpy_dtype(np.nan), x)
  x = tf.where(tf.math.equal(p, 0.), numpy_dtype(0.), x)
  x = tf.where(tf.math.equal(p, 1.), numpy_dtype(np.inf), x)

  return x


def _igammainv_fwd(a, p):
  """Compute output, aux (collaborates with _igammainv_bwd)."""
  output = _shared_igammainv_computation(a, p, is_igammainv=True)
  return output, (a, p)


def _igammainv_partials(a, x):
  """Compute partial derivatives of `igammainv(a, x)`."""
  # Partials for igamma.

  # This function does not have gradients in TF, and thus using
  # `stop_gradient` does not change behavior in TF.
  # Ideally, it would be nice to throw an exception when taking gradients of
  # this function in JAX mode, but this is not possible at the moment with
  # `custom_jvp`. See https://github.com/google/jax/issues/5913 for details.
  # TODO(https://github.com/google/jax/issues/5913): remove stop_gradients.
  igamma_partial_a = tf.raw_ops.IgammaGradA(
      a=tf.stop_gradient(a), x=tf.stop_gradient(x))
  igamma_partial_x = tf.math.exp(
      -x + tf.math.xlogy(a - 1., x) - tf.math.lgamma(a))

  # Use the fact that igamma and igammainv are inverses of each other to compute
  # the gradients.
  igammainv_partial_a = -igamma_partial_a / igamma_partial_x
  igammainv_partial_x = tf.math.reciprocal(igamma_partial_x)
  return igammainv_partial_a, igammainv_partial_x


def _igammainv_bwd(aux, g):
  """Reverse mode impl for igammainv."""
  a, p = aux
  x = _igammainv_custom_gradient(a, p)
  # Use the fact that igamma and igammainv are inverses to compute the gradient.
  pa, pp = _igammainv_partials(a, x)
  return _fix_gradient_for_broadcasting(a, p, pa * g, pp * g)


def _igammainv_jvp(primals, tangents):
  """Computes JVP for igammainv (supports JAX custom derivative)."""
  a, p = primals
  da, dp = tangents
  # TODO(https://github.com/google/jax/issues/3768): eliminate broadcast_to?
  bc_shp = prefer_static.broadcast_shape(prefer_static.shape(da),
                                         prefer_static.shape(dp))
  da = tf.broadcast_to(da, bc_shp)
  dp = tf.broadcast_to(dp, bc_shp)

  x = _igammainv_custom_gradient(a, p)
  pa, pp = _igammainv_partials(a, x)

  return x, pa * da + pp * dp


@tfp_custom_gradient.custom_gradient(
    vjp_fwd=_igammainv_fwd,
    vjp_bwd=_igammainv_bwd,
    jvp_fn=_igammainv_jvp)
def _igammainv_custom_gradient(a, p):
  return _shared_igammainv_computation(a, p, is_igammainv=True)


def igammainv(a, p, name=None):
  """Computes the inverse to `tf.math.igamma` with respect to `p`.

  This function is defined as the solution `x` to the equation
  `p = tf.math.igamma(a, x)`.

  # References
  [1] A. Didonato and A. Morris,
      Computation of the Incomplete Gamma Function Ratios and their Inverse
      https://dl.acm.org/doi/10.1145/22721.23109

  Args:
    a: A positive `float` `Tensor`. Must be broadcastable with `p`.
    p: A `float` `Tensor` whose entries lie in `[0, 1]`.
       Must be broadcastable with `a`.
    name: Optional Python `str` naming the operation.

  Returns:
    igammainv: igammainv(a, p). Has same type as `a`.
  """
  with tf.name_scope(name or 'igammainv'):
    dtype = dtype_util.common_dtype([a, p], tf.float32)
    a = tf.convert_to_tensor(a, dtype=dtype)
    p = tf.convert_to_tensor(p, dtype=dtype)
    return _igammainv_custom_gradient(a, p)


def _igammacinv_fwd(a, p):
  """Compute output, aux (collaborates with _igammacinv_bwd)."""
  output = _shared_igammainv_computation(a, p, is_igammainv=False)
  return output, (a, p)


def _igammacinv_bwd(aux, g):
  """Reverse mode impl for igammacinv."""
  a, p = aux
  x = _igammacinv_custom_gradient(a, p)
  pa, pp = _igammainv_partials(a, x)
  pp = -pp
  return _fix_gradient_for_broadcasting(a, p, pa * g, pp * g)


def _igammacinv_jvp(primals, tangents):
  """Computes JVP for igammacinv (supports JAX custom derivative)."""
  a, p = primals
  da, dp = tangents
  # TODO(https://github.com/google/jax/issues/3768): eliminate broadcast_to?
  bc_shp = prefer_static.broadcast_shape(prefer_static.shape(da),
                                         prefer_static.shape(dp))
  da = tf.broadcast_to(da, bc_shp)
  dp = tf.broadcast_to(dp, bc_shp)

  x = _igammacinv_custom_gradient(a, p)
  pa, pp = _igammainv_partials(a, x)
  pp = -pp

  return x, pa * da + pp * dp


@tfp_custom_gradient.custom_gradient(
    vjp_fwd=_igammacinv_fwd,
    vjp_bwd=_igammacinv_bwd,
    jvp_fn=_igammacinv_jvp)
def _igammacinv_custom_gradient(a, p):
  return _shared_igammainv_computation(a, p, is_igammainv=False)


def igammacinv(a, p, name=None):
  """Computes the inverse to `tf.math.igammac` with respect to `p`.

  This function is defined as the solution `x` to the equation
  `p = tf.math.igammac(a, x)`.

  # References
  [1] A. Didonato and A. Morris,
      Computation of the Incomplete Gamma Function Ratios and their Inverse
      https://dl.acm.org/doi/10.1145/22721.23109

  Args:
    a: A positive `float` `Tensor`. Must be broadcastable with `p`.
    p: A `float` `Tensor` whose entries lie in `[0, 1]`.
       Must be broadcastable with `a`.
    name: Optional Python `str` naming the operation.

  Returns:
    igammacinv: igammacinv(a, p). Has same type as `a`.
  """

  with tf.name_scope(name or 'igammacinv'):
    dtype = dtype_util.common_dtype([a, p], tf.float32)
    a = tf.convert_to_tensor(a, dtype=dtype)
    p = tf.convert_to_tensor(p, dtype=dtype)
    return _igammacinv_custom_gradient(a, p)


def round_exponential_bump_function(x, name=None):
  r"""Function supported on [-1, 1], smooth on the real line, with a round top.

  Define

  ```
  f(x) := exp(-1 / (1 - x**2)) * exp(1), for x in (-1, 1)
  f(x) := 0, for |x| >= 1.
  ```

  One can show that f(x)...

  * is C^\infty on the real line.
  * is supported on [-1, 1].
  * is equal to 1 at x = 0.
  * is strictly increasing on (-1, 0).
  * is strictly decreasing on (0, 1).
  * has gradient = 0 at 0.

  See [Bump Function](https://en.wikipedia.org/wiki/Bump_function)

  Args:
    x: Floating-point Tensor.
    name: Optional Python `str` naming the operation.

  Returns:
    y: Tensor of same shape and dtype as `x`.
  """
  with tf.name_scope(name or 'round_exponential_bump_function'):
    x = tf.convert_to_tensor(x, name='x')
    one_m_x2 = 1 - x**2
    y = tf.math.exp(1. - tf.math.reciprocal_no_nan(one_m_x2))
    return tf.where(one_m_x2 > 0., y, 0.)


def lambertw_winitzki_approx(z, name=None):
  """Computes Winitzki approximation to Lambert W function at z >= -1/exp(1).

  The approximation for z >= -1/exp(1) will be used as a starting point in the
  iterative algorithm to compute W(z). See _lambertw_principal_branch() below.
  See
  https://www.researchgate.net/post/Is_there_approximation_to_the_LambertWx_function
  and in particular (38) in
  https://pdfs.semanticscholar.org/e934/24f33e2742016ef18c36a80788400d2f17b4.pdf

  Args:
    z: value for which W(z) should be computed. Expected z >= -1/exp(1). If not
     then function will fail due to log(<0).
    name: optionally pass name for output.

  Returns:
    lambertw_winitzki_approx: Approximation for W(z) for z >= -1/exp(1).
  """
  with tf.name_scope(name or 'lambertw_winitzki_approx'):
    z = tf.convert_to_tensor(z)
    # See eq (38) here:
    # https://pdfs.semanticscholar.org/e934/24f33e2742016ef18c36a80788400d2f17b4.pdf
    # or (10) here:
    # https://hal.archives-ouvertes.fr/hal-01586546/document
    log1pz = tf.math.log1p(z)
    return log1pz * (1. - tf.math.log1p(log1pz) / (2. + log1pz))


def _fritsch_iteration(unused_should_stop, z, w, tol):
  """Root finding iteration for W(z) using Fritsch iteration."""
  # See Section 2.3 in https://arxiv.org/pdf/1209.0735.pdf
  # Approximate W(z) by viewing iterative algorithm as multiplicative factor
  #
  #  W(n+1) = W(n) * (1 + error)
  #
  # where error can be expressed as a function of z and W(n). See paper for
  # details.
  z = tf.convert_to_tensor(z)
  w = tf.convert_to_tensor(w)
  zn = tf.math.log(tf.abs(z)) - tf.math.log(tf.abs(w)) - w
  wp1 = w + 1.0
  q = 2. * wp1 * (wp1 + 2. / 3. * zn)
  q_minus_2zn = q - 2. * zn
  error = zn / wp1 * (1. + zn / q_minus_2zn)
  # Check absolute tolerance (not relative).  Here the iteration error is
  # for relative tolerance, as W(n+1) = W(n) * (1 + error).  Use
  # W(n+1) - W(n) = W(n) * error to get absolute tolerance.
  converged = abs(error * w) <= tol
  should_stop_next = tf.reduce_all(converged)
  return should_stop_next, w * (1. + error), z, tol


def _halley_iteration(unused_should_stop, w, z, tol, iteration_count):
  """Halley's method on root finding of w for the equation w * exp(w) = z."""
  w = tf.convert_to_tensor(w)
  z = tf.convert_to_tensor(z)
  f = w - z * tf.math.exp(-w)
  delta = f / (w + 1. - 0.5 * (w + 2.) * f / (w + 1.))
  w_next = w - delta
  converged = tf.math.abs(delta) <= tol * tf.math.abs(w_next)
  # We bound the number of iterations to be at most a 100.

  # When x is close to the branch point, the derivatives tend to very large
  # values, which causes the iteration to be slow. For x <= 0., 100 iterations
  # seems to be enough to guarantee a relative error of at most 1e-6.

  # The Winitzki approximation has a relative error of at most
  # 0.01. When x >= 0., the first through third derivatives are bounded such
  # that coupled with the initial approximation, we are in the realm of cubic
  # convergence.
  should_stop_next = tf.reduce_all(converged) | (iteration_count >= 100)
  return should_stop_next, w_next, z, tol, iteration_count + 1


def _lambertw_principal_branch(z, name=None):
  """Computes Lambert W of `z` element-wise at the principal (k = 0) branch.

  The Lambert W function is the inverse of `z = y * tf.exp(y)` and is a
  many-valued function. Here `y = W_0(z)`, where `W_0` is the Lambert W function
  evaluated at the 0-th branch (aka principal branch).

  Args:
    z: A Tensor with type `float32` or `float64`.
    name: A name for the operation (optional).
      Default value: `None` (i.e., 'lambertw_principal_branch').

  Returns:
    lambertw_principal_branch: A Tensor with same shape and same dtype as `z`.
  """
  with tf.name_scope(name or 'lambertw_principal_branch'):
    z = tf.convert_to_tensor(z)
    np_finfo = np.finfo(dtype_util.as_numpy_dtype(z.dtype))
    tolerance = tf.convert_to_tensor(2. * np_finfo.resolution, dtype=z.dtype)
    # Start while loop with the initial value at the approximate Lambert W
    # solution, instead of 'z' (for z > -1 / exp(1)).  Using 'z' has bad
    # convergence properties especially for large z (z > 5).
    z0 = tf.where(z > -np.exp(-1.), lambertw_winitzki_approx(z), z)
    z0 = tf.while_loop(cond=lambda stop, *_: ~stop,
                       body=_halley_iteration,
                       loop_vars=(False, z0, z, tolerance, 0))[1]
    return tf.cast(z0, dtype=z.dtype)


def _lambert_fwd(z):
  """Compute output, aux (collaborates with _lambert_bwd)."""
  wz = _lambertw_principal_branch(z)
  return wz, (z,)


def _lambert_bwd(aux, g):
  """Reverse mode impl for lambert."""
  z, = aux
  wz = _lambert_custom_gradient(z)
  # At z = 0 the analytic expressions for the gradient results in a 0/0
  # expression.  However, the continuous expansion (l'Hospital rule) gives a
  # derivative of 1.0 at z = 0.  This case has to be handled separately with
  # a where clause.
  return g * tf.where(
      tf.equal(z, 0.), tf.ones([], wz.dtype), wz / (z * (1. + wz)))


def _lambert_jvp(primals, tangents):
  """Computes JVP for lambert (supports JAX custom derivative)."""
  z, = primals
  dz, = tangents
  wz = _lambert_custom_gradient(z)

  # At z = 0 the analytic expressions for the gradient results in a 0/0
  # expression.  However, the continuous expansion (l'Hospital rule) gives a
  # derivative of 1.0 at z = 0.  This case has to be handled separately with
  # a where clause.
  pz = tf.where(tf.equal(z, 0.), tf.ones([], wz.dtype), wz / (z * (1. + wz)))
  return wz, pz * dz


@tfp_custom_gradient.custom_gradient(
    vjp_fwd=_lambert_fwd,
    vjp_bwd=_lambert_bwd,
    jvp_fn=_lambert_jvp)
def _lambert_custom_gradient(z):
  return _lambertw_principal_branch(z)


def lambertw(z, name=None):
  """Computes Lambert W of `z` element-wise.

  The Lambert W function is the inverse of `z = u * exp(u)`, i. e., it is the
  function that satisfies `u = W(z) * exp(W(z))`.  The solution cannot be
  expressed as a composition of elementary functions and is thus part of the
  *special* functions in mathematics.  See
  https://en.wikipedia.org/wiki/Lambert_W_function.

  In general it is a complex-valued function with multiple branches. The `k=0`
  branch is known as the *principal branch* of the Lambert W function and is
  implemented here. See also `scipy.special.lambertw`.

  This code returns only the real part of the image of the Lambert W function.

  # References

  Corless, R.M., Gonnet, G.H., Hare, D.E.G. et al. On the LambertW function.
  Adv Comput Math 5, 329-359 (1996) doi:10.1007/BF02124750

  Args:
    z: A Tensor with type `float32` or `float64`.
    name: A name for the operation (optional).

  Returns:
    lambertw: The Lambert W function evaluated at `z`. A Tensor with same shape
      and same dtype as `z`.
  """
  with tf.name_scope(name or 'lambertw'):
    z = tf.convert_to_tensor(z)
    return _lambert_custom_gradient(z)


def log_gamma_correction(x, name=None):
  """Returns the error of the Stirling approximation to lgamma(x) for x >= 8.

  This is useful for accurately evaluating ratios between Gamma functions, as
  happens when trying to compute Beta functions.

  Specifically,
  ```
  lgamma(x) approx (x - 0.5) * log(x) - x + 0.5 log (2 pi)
                   + log_gamma_correction(x)
  ```
  for x >= 8.

  This is the function called Delta in [1], eq (30).  We implement it with
  the rational minimax approximation given in [1], eq (32).

  References:

  [1] DiDonato and Morris, "Significant Digit Computation of the Incomplete Beta
      Function Ratios", 1988.  Technical report NSWC TR 88-365, Naval Surface
      Warfare Center (K33), Dahlgren, VA 22448-5000.  Section IV, Auxiliary
      Functions.  https://apps.dtic.mil/dtic/tr/fulltext/u2/a210118.pdf

  Args:
    x: Floating-point Tensor at which to evaluate the log gamma correction
      elementwise.  The approximation is accurate when x >= 8.
    name: Optional Python `str` naming the operation.

  Returns:
    lgamma_corr: Tensor of elementwise log gamma corrections.
  """
  with tf.name_scope(name or 'log_gamma_correction'):
    dtype = dtype_util.common_dtype([x], tf.float32)
    x = tf.convert_to_tensor(x, dtype=dtype)

    minimax_coeff = tf.constant([
        0.833333333333333e-01,
        -0.277777777760991e-02,
        0.793650666825390e-03,
        -0.595202931351870e-03,
        0.837308034031215e-03,
        -0.165322962780713e-02,
    ], dtype=dtype)

    inverse_x = 1 / x
    inverse_x_squared = inverse_x * inverse_x
    accum = minimax_coeff[5]
    for i in reversed(range(5)):
      accum = accum * inverse_x_squared + minimax_coeff[i]
    return accum * inverse_x


def _fix_gradient_for_broadcasting(a, b, grad_a, grad_b):
  """Reduces broadcast dimensions for a custom gradient."""
  if (tensorshape_util.is_fully_defined(a.shape) and
      tensorshape_util.is_fully_defined(b.shape) and
      a.shape == b.shape):
    return [grad_a, grad_b]
  a_shape = tf.shape(a)
  b_shape = tf.shape(b)
  ra, rb = tf.raw_ops.BroadcastGradientArgs(s0=a_shape, s1=b_shape)
  grad_a = tf.reshape(tf.reduce_sum(grad_a, axis=ra), a_shape)
  grad_b = tf.reshape(tf.reduce_sum(grad_b, axis=rb), b_shape)
  return [grad_a, grad_b]


def _log_gamma_difference_big_y(x, y):
  """Returns lgamma(y) - lgamma(x + y), accurately if 0 <= x <= y and y >= 8.

  This is more accurate than subtracting lgammas directly because lgamma grows
  as `x log(x) - x + o(x)`, and thus subtracting the value of lgamma for two
  close, large arguments incurs catastrophic cancellation.

  The method is to partition lgamma into the Striling approximation and the
  correction `log_gamma_correction`, symbolically cancel the former, and compute
  and subtract the latter.

  Args:
    x: Floating-point Tensor.  `x` should be non-negative, and elementwise no
      more than `y`.
    y: Floating-point Tensor.  `y` should be elementwise no less than 8.

  Returns:
    lgamma_diff: Floating-point Tensor, the difference lgamma(y) - lgamma(x+y),
      computed elementwise.
  """
  cancelled_stirling = (-1 * (x + y - 0.5) * tf.math.log1p(x / y)
                        - x * tf.math.log(y) + x)
  correction = log_gamma_correction(y) - log_gamma_correction(x + y)
  return correction + cancelled_stirling


def _log_gamma_difference_naive_gradient(x, y):
  big_y = _log_gamma_difference_big_y(x, y)
  small_y = tf.math.lgamma(y) - tf.math.lgamma(x + y)
  return tf.where(y >= 8, big_y, small_y)


def _log_gamma_difference_fwd(x, y):
  """Compute output, aux (collaborates with _log_gamma_difference_bwd)."""
  return _log_gamma_difference_naive_gradient(x, y), (x, y)


def _log_gamma_difference_bwd(aux, g):
  """Reverse mode impl for log-gamma-diff."""
  x, y = aux
  # Computing the gradient naively as the difference of digammas because
  # (i) digamma grows slower than gamma, so gets into bad cancellations
  # later, and (ii) doing better is work.  This matches what the gradient
  # would be if the forward pass were computed naively as the difference
  # of lgammas.
  #
  # Note: This gradient assumes x and y are the same shape; this needs to
  # be arranged by pre-broadcasting before calling
  # `_log_gamma_difference`.
  px = -tf.math.digamma(x + y)
  py = tf.math.digamma(y) + px
  return _fix_gradient_for_broadcasting(x, y, px * g, py * g)


def _log_gamma_difference_jvp(primals, tangents):
  """Computes JVP for log-gamma-difference (supports JAX custom derivative)."""
  x, y = primals
  dx, dy = tangents
  # TODO(https://github.com/google/jax/issues/3768): eliminate broadcast_to?
  bc_shp = prefer_static.broadcast_shape(prefer_static.shape(dx),
                                         prefer_static.shape(dy))
  dx = tf.broadcast_to(dx, bc_shp)
  dy = tf.broadcast_to(dy, bc_shp)
  # See note above in _log_gamma_difference_bwd.
  px = -tf.math.digamma(x + y)
  py = tf.math.digamma(y) + px
  return _log_gamma_difference_naive_gradient(x, y), px * dx + py * dy


@tfp_custom_gradient.custom_gradient(
    vjp_fwd=_log_gamma_difference_fwd,
    vjp_bwd=_log_gamma_difference_bwd,
    jvp_fn=_log_gamma_difference_jvp)
def _log_gamma_difference_custom_gradient(x, y):
  return _log_gamma_difference_naive_gradient(x, y)


def log_gamma_difference(x, y, name=None):
  """Returns lgamma(y) - lgamma(x + y), accurately.

  This is more accurate than subtracting lgammas directly because lgamma grows
  as `x log(x) - x + o(x)`, and thus subtracting the value of lgamma for two
  close, large arguments incurs catastrophic cancellation.

  The method is to partition lgamma into the Striling approximation and the
  correction `log_gamma_correction`, symbolically cancel the former, and compute
  and subtract the latter.

  Args:
    x: Floating-point Tensor.  `x` should be non-negative, and elementwise no
      more than `y`.
    y: Floating-point Tensor.  `y` should be elementwise no less than 8.
    name: Optional Python `str` naming the operation.

  Returns:
    lgamma_diff: Floating-point Tensor, the difference lgamma(y) - lgamma(x+y),
      computed elementwise.
  """
  with tf.name_scope(name or 'log_gamma_difference'):
    dtype = dtype_util.common_dtype([x, y], tf.float32)
    x = tf.convert_to_tensor(x, dtype=dtype)
    y = tf.convert_to_tensor(y, dtype=dtype)
    return _log_gamma_difference_custom_gradient(x, y)


def _lbeta_naive_gradient(x, y):
  """Computes log(Beta(x, y)) with autodiff gradients only."""
  # Flip args if needed so y >= x.  Beta is mathematically symmetric but our
  # method for computing it is not.
  x, y = tf.minimum(x, y), tf.maximum(x, y)

  log2pi = tf.constant(np.log(2 * np.pi), dtype=x.dtype)
  # Two large arguments case: y >= x >= 8.
  log_beta_two_large = (0.5 * log2pi
                        - 0.5 * tf.math.log(y)
                        + log_gamma_correction(x)
                        + log_gamma_correction(y)
                        - log_gamma_correction(x + y)
                        + (x - 0.5) * tf.math.log(x / (x + y))
                        - y * tf.math.log1p(x / y))

  # One large argument case: x < 8, y >= 8.
  log_beta_one_large = tf.math.lgamma(x) + _log_gamma_difference_big_y(x, y)

  # Small arguments case: x <= y < 8.
  log_beta_small = tf.math.lgamma(x) + tf.math.lgamma(y) - tf.math.lgamma(x + y)

  # Reference [1] has two more arms, for cases where x or y falls into the
  # interval (2, 8).  In these cases, reference [1] recommends iteratively
  # reducing the arguments using the identity
  #   B(x, y) = B(x - 1, y) * (x - 1) / (x + y - 1)
  # so they fall in the interval [1, 2].  We choose not to do that here to avoid
  # a TensorFlow while loop, and hope that subtracting lgammas will be accurate
  # enough for the user's purposes.

  return tf.where(x >= 8,
                  log_beta_two_large,
                  tf.where(y >= 8,
                           log_beta_one_large,
                           log_beta_small))


def _lbeta_fwd(x, y):
  """Compute output, aux (collaborates with _lbeta_bwd)."""
  return _lbeta_naive_gradient(x, y), (x, y)


def _lbeta_bwd(aux, g):
  x, y = aux
  total_digamma = tf.math.digamma(x + y)
  px = tf.math.digamma(x) - total_digamma
  py = tf.math.digamma(y) - total_digamma
  return _fix_gradient_for_broadcasting(x, y, px * g, py * g)


def _lbeta_jvp(primals, tangents):
  """Computes JVP for log-beta (supports JAX custom derivative)."""
  x, y = primals
  dx, dy = tangents
  # TODO(https://github.com/google/jax/issues/3768): eliminate broadcast_to?
  bc_shp = prefer_static.broadcast_shape(prefer_static.shape(dx),
                                         prefer_static.shape(dy))
  dx = tf.broadcast_to(dx, bc_shp)
  dy = tf.broadcast_to(dy, bc_shp)
  total_digamma = tf.math.digamma(x + y)
  px = tf.math.digamma(x) - total_digamma
  py = tf.math.digamma(y) - total_digamma
  return _lbeta_naive_gradient(x, y), px * dx + py * dy


@tfp_custom_gradient.custom_gradient(
    vjp_fwd=_lbeta_fwd,
    vjp_bwd=_lbeta_bwd,
    jvp_fn=_lbeta_jvp)
def _lbeta_custom_gradient(x, y):
  """Computes log(Beta(x, y)) with correct custom gradient."""
  return _lbeta_naive_gradient(x, y)


@tf.function(autograph=False)
def lbeta(x, y, name=None):
  """Returns log(Beta(x, y)).

  This is semantically equal to
    lgamma(x) + lgamma(y) - lgamma(x + y)
  but the method is more accurate for arguments above 8.

  The reason for accuracy loss in the naive computation is catastrophic
  cancellation between the lgammas.  This method avoids the numeric cancellation
  by explicitly decomposing lgamma into the Stirling approximation and an
  explicit `log_gamma_correction`, and cancelling the large terms from the
  Striling analytically.

  The computed gradients are the same as for the naive forward computation,
  because (i) digamma grows much slower than lgamma, so cancellations aren't as
  bad, and (ii) it's simpler and faster than trying to be more accurate.

  References:

  [1] DiDonato and Morris, "Significant Digit Computation of the Incomplete Beta
      Function Ratios", 1988.  Technical report NSWC TR 88-365, Naval Surface
      Warfare Center (K33), Dahlgren, VA 22448-5000.  Section IV, Auxiliary
      Functions.  https://apps.dtic.mil/dtic/tr/fulltext/u2/a210118.pdf

  Args:
    x: Floating-point Tensor.
    y: Floating-point Tensor.
    name: Optional Python `str` naming the operation.

  Returns:
    lbeta: Tensor of elementwise log beta(x, y).
  """
  with tf.name_scope(name or 'tfp_lbeta'):
    dtype = dtype_util.common_dtype([x, y], tf.float32)
    x = tf.convert_to_tensor(x, dtype=dtype)
    y = tf.convert_to_tensor(y, dtype=dtype)
    return _lbeta_custom_gradient(x, y)


# The Owen's T implementation below is based on
# [1] Patefield M., Tandy D., Fast and Accurate Calcuation of Owen's T-Function
#     Journal of Statistical Software http://www.jstatsoft.org/v05/i05/paper


def _owens_t_method1(h, a, m):
  """OwensT Method T1 using series expansions."""
  # Method T1, which is evaluation of a particular series expansion of OwensT.

  dtype = dtype_util.common_dtype([h, a], tf.float32)
  numpy_dtype = dtype_util.as_numpy_dtype(dtype)

  neg_half_h_squared = -0.5 * tf.math.square(h)
  a_squared = tf.math.square(a)

  def series_evaluation(
      should_stop,
      index,
      ai,
      di,
      gi,
      series_sum):

    new_ai = a_squared * ai
    new_di = gi - di
    new_gi = neg_half_h_squared / index * gi
    new_series_sum = tf.where(
        should_stop, series_sum,
        series_sum + new_di * new_ai / (2. * index - 1.))
    should_stop = index >= m
    return should_stop, index + 1., new_ai, new_di, new_gi, new_series_sum

  broadcast_shape = prefer_static.broadcast_shape(
      prefer_static.shape(h), prefer_static.shape(a))
  initial_ai = a / numpy_dtype(2 * np.pi)
  initial_di = tf.math.expm1(neg_half_h_squared)
  initial_gi = neg_half_h_squared * tf.math.exp(neg_half_h_squared)
  initial_sum = (
      tf.math.atan(a) / numpy_dtype(2 * np.pi) + initial_ai * initial_di)

  (_, _, _, _, _, series_sum) = tf.while_loop(
      cond=lambda stop, *_: tf.reduce_any(~stop),
      body=series_evaluation,
      loop_vars=(
          tf.zeros(broadcast_shape, dtype=tf.bool),
          tf.cast(2., dtype=dtype),
          initial_ai,
          initial_di,
          initial_gi,
          initial_sum))
  return series_sum


def _owens_t_method2(h, a, m):
  """OwensT Method T2 using Power series."""
  # Method T2, which is evaluation approximating the (1 + x^2)^-1 term in the
  # denominator of the OwensT integrand via power series, and integrating this
  # term by term to get a series expansion.
  dtype = dtype_util.common_dtype([h, a], tf.float32)
  numpy_dtype = dtype_util.as_numpy_dtype(dtype)
  h_squared = tf.math.square(h)
  nega_squared = -tf.math.square(a)
  num_iterations = 2 * m + 1.
  y = tf.math.reciprocal(h_squared)

  def series_evaluation(
      should_stop,
      index,
      summand,
      term,
      series_sum):
    new_summand = y * (term - index * summand)
    new_term = nega_squared * term
    new_series_sum = tf.where(should_stop, series_sum, series_sum + new_summand)
    should_stop = index >= num_iterations
    return should_stop, index + 2., new_summand, new_term, new_series_sum

  broadcast_shape = prefer_static.broadcast_shape(
      prefer_static.shape(h), prefer_static.shape(a))
  initial_summand = -0.5 * tf.math.erf(a * h) / h
  initial_sum = initial_summand
  initial_term = a * tf.math.exp(
      -0.5 * tf.math.square(a * h)) / numpy_dtype(np.sqrt(2 * np.pi))

  (_, _, _, _, series_sum) = tf.while_loop(
      cond=lambda stop, *_: tf.reduce_any(~stop),
      body=series_evaluation,
      loop_vars=(
          tf.zeros(broadcast_shape, dtype=tf.bool),
          tf.cast(1., dtype=dtype),
          initial_summand,
          initial_term,
          initial_sum))
  return (series_sum * tf.math.exp(-0.5 * h_squared) /
          numpy_dtype(np.sqrt(2 * np.pi)))


def _owens_t_method3(h, a):
  """OwensT Method T3, using Chebyshev series."""
  # Method T3, which is evaluation approximating the (1 + x^2)^-1 term in the
  # denominator of the OwensT integrand via chebyshev series, and integrating
  # this term by term to get a series expansion.
  coefficients = np.array([
      0.99999999999999999999999729978162447266851932041876728736094298092,
      -0.9999999999999999999946705637967839181062653325188532341679987487,
      0.99999999999999999824849349313270659391127814689133077036298754586,
      -0.9999999999999997703859616213643405880166422891953033591551179153,
      0.99999999999998394883415238173334565554173013941245103172035286759,
      -0.9999999999993063616095509371081203145247992197457263066869044528,
      0.99999999997973363404094644295992298705901604112382452758559037676,
      -0.9999999995749584120690466801190516397534123780375655213594441702,
      0.99999999332262341933753249439201609471582390767861031080974566177,
      -0.9999999188923242461073033481053037468263536806742737922476636768,
      0.99999921951434836744028537835494208830551296800829326291600811289,
      -0.9999939351372067128309979219133169714722271997418573865750972505,
      0.99996135597690552745362392866517133091672395614263398912807169603,
      -0.9997955636651394602640678896963029382098775775864121129307978458,
      0.99909278962961710015348625142385059005136666194734431542322608252,
      -0.9965938374119182021193086204326146003381573358628885806714509388,
      0.98910017138386127038463510314625339359073956513420458166238478926,
      -0.9700785580406933145213319822037627715121601685824945133478464073,
      0.92911438683263187495758525500033707204091967947532160289872782771,
      -0.8542058695956156057286980736842905011429254735181323743367879525,
      0.73796526033030091233118357742803709382964420335559408722681794195,
      -0.5852346988283739457012859900378515414416468058761587864517163279,
      0.41599777614567630616566166358186846050387420534301419658012217494,
      -0.2588210875241943574388730510317252236407805082485246378222935376,
      0.13755358251638926485046469515002655850557890194106175657270903465,
      -0.0607952766325955730493900985022020434830339794955745989150270485,
      0.02163376832998715280598364838403905142754886795307972945570602292,
      -0.0059340569345518672987699581418120390055001422042884348392721826,
      0.00117434148183329465104745761827392105533338601068118659634858706,
      -1.4891556133503689340734532606898813301663424844055299815106940E-4,
      9.07235432079435758771092950798881466945428151426884488484154734E-6])

  a_squared = tf.math.square(a)
  h_squared = tf.math.square(h)
  y = tf.math.reciprocal(h_squared)
  vi = a * tf.math.exp(-0.5 * tf.math.square(a * h)) / np.sqrt(2 * np.pi)
  zi = 0.5 * tf.math.erf(a * h / np.sqrt(2.)) / h
  result = 0.

  for i in range(31):
    result = result + zi * coefficients[i]
    zi = y * ((2 * i + 1.) * zi - vi)
    vi = a_squared * vi
  return result * tf.math.exp(-0.5 * h_squared) / np.sqrt(2 * np.pi)


def _owens_t_method4(h, a, m):
  """OwensT Method T4, which is a reordered evaluation of method T2."""
  dtype = dtype_util.common_dtype([h, a], tf.float32)
  h_squared = tf.math.square(h)
  nega_squared = -tf.math.square(a)
  num_iterations = 2 * m + 1.

  def series_evaluation(
      should_stop,
      index,
      term,
      coeff,
      series_sum):
    new_coeff = (1. - h_squared * coeff) / index
    new_term = nega_squared * term
    new_series_sum = tf.where(
        should_stop, series_sum, series_sum + new_coeff * new_term)
    should_stop = index >= num_iterations
    return should_stop, index + 2., new_term, new_coeff, new_series_sum

  broadcast_shape = prefer_static.broadcast_shape(
      prefer_static.shape(h), prefer_static.shape(a))
  initial_term = a * tf.math.exp(
      -0.5 * h_squared * (1 - nega_squared)) / (2 * np.pi)
  initial_sum = initial_term

  (_, _, _, _, series_sum) = tf.while_loop(
      cond=lambda stop, *_: tf.reduce_any(~stop),
      body=series_evaluation,
      loop_vars=(
          tf.zeros(broadcast_shape, dtype=tf.bool),
          tf.cast(3., dtype=dtype),
          initial_term,
          tf.ones(broadcast_shape, dtype=dtype),
          initial_sum))
  return series_sum


def _owens_t_method5(h, a):
  """OwensT Method T5 which uses Gaussian Quadrature."""
  # Method T5, which is a gaussian quadrature approximation of the integral.

  # These are shifted and squared.
  quadrature_points = np.array([
      0.35082039676451715489E-02, 0.31279042338030753740E-01,
      0.85266826283219451090E-01, 0.16245071730812277011E+00,
      0.25851196049125434828E+00, 0.36807553840697533536E+00,
      0.48501092905604697475E+00, 0.60277514152618576821E+00,
      0.71477884217753226516E+00, 0.81475510988760098605E+00,
      0.89711029755948965867E+00, 0.95723808085944261843E+00,
      0.99178832974629703586E+00])
  quadrature_weights = np.array([
      0.18831438115323502887E-01, 0.18567086243977649478E-01,
      0.18042093461223385584E-01, 0.17263829606398753364E-01,
      0.16243219975989856730E-01, 0.14994592034116704829E-01,
      0.13535474469662088392E-01, 0.11886351605820165233E-01,
      0.10070377242777431897E-01, 0.81130545742299586629E-02,
      0.60419009528470238773E-02, 0.38862217010742057883E-02,
      0.16793031084546090448E-02])
  r = tf.math.square(a[..., tf.newaxis]) * quadrature_points
  log_integrand = -0.5 * tf.math.square(
      h[..., tf.newaxis]) * (1. + r) - tf.math.log1p(r)
  return tf.math.exp(tf.math.log(a) + tf.math.reduce_logsumexp(
      log_integrand + np.log(quadrature_weights), axis=-1))


def _owens_t_method6(h, a):
  # Method T6, which is a special case for when a is near 1.
  r = tf.math.atan2(1. - a, 1. + a)
  # When a = 1, T(h, 1) = 0.5 * ndtr(h) * (1 - ndtr(h)).
  # Thus, when a is close to 1, we add a correction term.
  normh = 0.5 * tf.math.erfc(h / np.sqrt(2.))
  result = 0.5 * normh * (1 - normh)
  return tf.where(
      tf.math.equal(r, 0.),
      result,
      result - r * tf.math.exp(
          -(1. - a) * tf.math.square(h) / (2 * r)) / (2 * np.pi))


def _owens_t_regions(h, a):
  """Returns a list of Tensors describing the region of computation."""
  # We assume h >= 0, 0 <= a <= 1
  # Regions 1-7 that use T1.
  regions = []

  is_in_region1 = (h <= 0.06) & (a <= 0.025)
  is_in_region1 = is_in_region1 | (h <= 0.02) & (a <= 0.09)
  regions.append(is_in_region1)

  is_in_region2 = (h <= 0.02) & (a >= 0.09)
  is_in_region2 = (is_in_region2 |
                   (h >= 0.02) & (h <= 0.06) & (a >= 0.025) & (a <= 0.36))
  is_in_region2 = is_in_region2 | (h >= 0.06) & (h <= 0.09) & (a <= 0.09)
  regions.append(is_in_region2)

  is_in_region3 = (h >= 0.02) & (h <= 0.06) & (a >= 0.36)
  is_in_region3 = (is_in_region3 |
                   (h >= 0.06) & (h <= 0.09) & (a >= 0.09) & (a <= 0.5))
  is_in_region3 = (is_in_region3 |
                   (h >= 0.09) & (h <= 0.26) & (a >= 0.025) & (a <= 0.15))
  regions.append(is_in_region3)

  is_in_region4 = (h >= 0.06) & (h <= 0.125) & (a >= 0.9)
  regions.append(is_in_region4)

  is_in_region5 = (h >= 0.06) & (h <= 0.26) & (a >= 0.5) & (a <= 0.9)
  is_in_region5 = (is_in_region5 |
                   (h >= 0.09) & (h <= 0.26) & (a >= 0.15) & (a <= 0.5))
  is_in_region5 = (is_in_region5 |
                   (h >= 0.26) & (h <= 0.6) & (a >= 0.025) & (a <= 0.36))
  regions.append(is_in_region5)

  is_in_region6 = (h >= 0.26) & (h <= 0.6) & (a >= 0.36) & (a <= 0.9)
  is_in_region6 = is_in_region6 | (h >= 0.125) & (h <= 0.4) & (a >= 0.9)
  regions.append(is_in_region6)

  is_in_region7 = (h >= 0.6) & (h <= 1.7) & (a >= 0.15) & (a <= 0.36)
  regions.append(is_in_region7)

  is_in_region8 = (h >= 0.6) & (h <= 1.7) & (a >= 0.36) & (a <= 0.9)
  is_in_region8 = (is_in_region8 |
                   (h >= 0.4) & (h <= 1.6) & (a >= 0.9) & (a <= 0.99999))
  regions.append(is_in_region8)

  is_in_region9 = (h >= 4.8) & (a <= 0.09)
  regions.append(is_in_region9)

  is_in_region10 = (h >= 4.8) & (a >= 0.09) & (a <= 0.36)
  regions.append(is_in_region10)

  is_in_region11 = (h >= 4.8) & (a >= 0.36) & (a <= 0.5)
  regions.append(is_in_region11)

  is_in_region12 = (h >= 3.4) & (a >= 0.9)
  is_in_region12 = is_in_region12 | (h >= 3.36) & (a >= 0.36) & (a <= 0.9)
  is_in_region12 = is_in_region12 & ~is_in_region11
  regions.append(is_in_region12)

  is_in_region13 = (h >= 0.09) & (h <= 2.4) & (a <= 0.025)
  regions.append(is_in_region13)

  is_in_region14 = (h >= 0.6) & (h <= 1.7) & (a >= 0.025) & (a <= 0.09)
  regions.append(is_in_region14)

  is_in_region15 = (h >= 0.6) & (h <= 2.4) & (a >= 0.025) & (a <= 0.15)
  is_in_region15 = is_in_region15 & ~is_in_region14
  regions.append(is_in_region15)

  is_in_region16 = (h >= 1.7) & (h <= 2.4) & (a >= 0.15) & (a <= 0.36)
  is_in_region16 = is_in_region16 | (h >= 2.4) & (h <= 4.8) & (a <= 0.36)
  regions.append(is_in_region16)

  is_in_region17 = (h >= 1.6) & (h <= 3.4) & (a >= 0.9) & (a <= 0.99999)
  is_in_region17 = (is_in_region17 |
                    (h >= 1.7) & (h <= 3.4) & (a >= 0.36) & (a <= 0.9))
  regions.append(is_in_region17)

  # Near the line a = 1.
  is_in_region18 = (h >= 0.4) & (h <= 2.33) & (a >= 0.99999)
  regions.append(is_in_region18)

  return regions


def _owens_t_naive_gradient(h, a):
  """Computes OwensT(h, a) with autodiff gradients only."""
  dtype = dtype_util.common_dtype([h, a], tf.float32)
  numpy_dtype = dtype_util.as_numpy_dtype(dtype)

  # OwensT(-h, a) = OwensT(h, a)
  h = tf.math.abs(h)
  abs_a = tf.math.abs(a)

  # Remap arguments such that 0 <= a <= 1.
  modified_a = tf.where(
      abs_a <= 1.,
      abs_a,
      tf.math.reciprocal(abs_a))

  modified_h = tf.where(abs_a <= 1., h, abs_a * h)

  # For regions 1 - 8, we use method1 with different orders.

  regions = _owens_t_regions(modified_h, modified_a)

  # Short-circuit if we are not in the first 8 regions.
  order = numpy_dtype(1.)
  order = tf.where(regions[0], numpy_dtype(2.), order)
  order = tf.where(regions[1], numpy_dtype(3.), order)
  order = tf.where(regions[2], numpy_dtype(4.), order)
  order = tf.where(regions[3], numpy_dtype(5.), order)
  order = tf.where(regions[4], numpy_dtype(7.), order)
  order = tf.where(regions[5], numpy_dtype(10.), order)
  order = tf.where(regions[6], numpy_dtype(12.), order)
  order = tf.where(regions[7], numpy_dtype(18.), order)
  result = _owens_t_method1(modified_h, modified_a, order)

  # For regions 9, 10 and 11 we use method2 with different orders.
  order = numpy_dtype(1.)
  order = tf.where(regions[8], numpy_dtype(10.), order)
  order = tf.where(regions[9], numpy_dtype(20.), order)
  order = tf.where(regions[10], numpy_dtype(30.), order)
  result = tf.where(
      regions[8] | regions[9] | regions[10],
      _owens_t_method2(modified_h, modified_a, order),
      result)

  # For region 12 we use method3.
  result = tf.where(
      regions[11], _owens_t_method3(modified_h, modified_a), result)

  # For regions 13, 14, 15 and 16 we use method4 with different orders.
  order = numpy_dtype(1.)
  order = tf.where(regions[12], numpy_dtype(4.), order)
  order = tf.where(regions[13], numpy_dtype(7.), order)
  order = tf.where(regions[14], numpy_dtype(8.), order)
  order = tf.where(regions[15], numpy_dtype(20.), order)
  result = tf.where(
      regions[12] | regions[13] | regions[14] | regions[15],
      _owens_t_method4(modified_h, modified_a, order),
      result)

  # For region 17 we use method5.
  result = tf.where(
      regions[16], _owens_t_method5(modified_h, modified_a), result)

  # For region 18, we use method6.
  result = tf.where(
      regions[17], _owens_t_method6(modified_h, modified_a), result)

  result = tf.where(
      tf.math.equal(modified_h, 0.),
      tf.math.atan(modified_a) / (2 * np.pi), result)

  # When a = 1, OwensT(h, 1) = ndtr(h) * (1 - ndtr(h))
  result = tf.where(
      tf.math.equal(modified_a, 1.),
      (0.125 * tf.math.erfc(-modified_h / np.sqrt(2.)) *
       tf.math.erfc(modified_h / np.sqrt(2.))), result)

  # When a = 0, we should return 0.
  result = tf.where(tf.math.equal(modified_a, 0.), numpy_dtype(0.), result)

  normh = tf.math.erfc(h / np.sqrt(2.))
  normah = tf.math.erfc(abs_a * h / np.sqrt(2.))
  # Compensate for when |a| > 1.
  result = tf.where(
      abs_a > 1.,
      tf.where(
          abs_a * h <= 0.67,
          0.25 - 0.25 * tf.math.erf(
              h / np.sqrt(2.)) * tf.math.erf(abs_a * h / np.sqrt(2.)) - result,
          0.25 * (normh + normah - normh * normah) - result),
      result)

  result = tf.math.sign(a) * result

  result = tf.where(tf.math.is_nan(a) | tf.math.is_nan(h),
                    numpy_dtype(np.nan),
                    result)
  return result


def _owens_t_fwd(h, a):
  """Compute output, aux (collaborates with _owens_t_bwd)."""
  return _owens_t_naive_gradient(h, a), (h, a)


def _owens_t_bwd(aux, g):
  h, a = aux
  ph = (-tf.math.exp(-0.5 * tf.math.square(h)) *
        tf.math.erf(a * h / np.sqrt(2)) / (2 * np.sqrt(2 * np.pi)))
  pa = (tf.math.exp(-0.5 * (tf.math.square(a) + 1) * tf.math.square(h)) /
        (2 * np.pi * (tf.math.square(a) + 1.)))
  return _fix_gradient_for_broadcasting(h, a, ph * g, pa * g)


def _owens_t_jvp(primals, tangents):
  """Computes JVP for log-beta (supports JAX custom derivative)."""
  h, a = primals
  dh, da = tangents
  # TODO(https://github.com/google/jax/issues/3768): eliminate broadcast_to?
  bc_shp = prefer_static.broadcast_shape(prefer_static.shape(dh),
                                         prefer_static.shape(da))
  dh = tf.broadcast_to(dh, bc_shp)
  da = tf.broadcast_to(da, bc_shp)
  ph = (-tf.math.exp(-0.5 * tf.math.square(h)) *
        tf.math.erf(a * h / np.sqrt(2)) / (2 * np.sqrt(2 * np.pi)))
  pa = (tf.math.exp(-0.5 * (tf.math.square(a) + 1.)* tf.math.square(h)) /
        (2 * np.pi * (tf.math.square(a) + 1.)))
  return _owens_t_naive_gradient(h, a), ph * dh + pa * da


@tfp_custom_gradient.custom_gradient(
    vjp_fwd=_owens_t_fwd,
    vjp_bwd=_owens_t_bwd,
    jvp_fn=_owens_t_jvp)
def _owens_t_custom_gradient(h, a):
  """Computes OwensT(h, a) with correct custom gradient."""
  return _owens_t_naive_gradient(h, a)


def owens_t(h, a, name=None):
  # pylint: disable=line-too-long
  """Computes Owen's T function of `h` and `a` element-wise.

  Owen's T function is defined as the combined probability of the event `X > h`
  and `0 < Y < a * X`, where `X` and `Y` are independent standard normal
  random variables.

  In integral form this is defined as `1 / (2 * pi)` times the integral of
  `exp(-0.5 * h ** 2 * (1 + x ** 2)) / (1 + x ** 2)` from `0` to `a`.
  `h` and `a` can be any real number

  The Owen's T implementation below is based on
  ([Patefield and Tandy, 2000][1]).

  The Owen's T function has several notable properties which
  we list here for convenience. ([Owen, 1980][2], page 414)

  - P2.1  `T( h, 0)   =  0`
  - P2.2  `T( 0, a)   =  arctan(a) / (2 pi)`
  - P2.3  `T( h, 1)   =  Phi(h) (1 - Phi(h)) / 2`
  - P2.4  `T( h, inf) =  (1 - Phi(|h|)) / 2`
  - P2.5  `T(-h, a)   =  T(h, a)`
  - P2.6  `T( h,-a)   = -T(h, a)`
  - P2.7  `T( h, a) + T(a h, 1 / a) = Phi(h)/2 + Phi(ah)/2 - Phi(h) Phi(ah) - [a<0]/2`
  - P2.8  `T( h, a)   =  arctan(a)/(2 pi) - 1/(2 pi) int_0^h int_0^{ax}` exp(-(x**2 + y**2)/2) dy dx`
  - P2.9  `T( h, a)   =  arctan(a)/(2 pi) - int_0**h phi(x) Phi(a x) dx + Phi(h)/2 - 1/4`

  `[a<0]` uses Iverson bracket notation, i.e., `[a<0] = {1 if a<0 and 0 otherwise`.

  Let us also define P2.10 as:
  - P2.10  `T(inf, a) = 0`
  - Proof

    Note that result #10,010.6 ([Owen, 1980][2], pg 403) states that:
    `int_0^inf phi(x) Phi(a+bx) dx = Phi(a/rho)/2 + T(a/rho,b) where rho = sqrt(1+b**2).`
    Using `a=0`, this result is:
    `int_0^inf phi(x) Phi(bx) dx = 1/4 + T(0,b) = 1/4 + arctan(b) / (2 pi)`
    Combining this with P2.9 implies
    ```none
    T(inf, a)
     =  arctan(a)/(2 pi) - [ 1/4 + arctan(a) / (2 pi)]  + Phi(inf)/2 - 1/4
     = -1/4 + 1/2 -1/4 = 0.
    ```
    QED

  Args:
    h: A `float` `Tensor` defined as in `P({X > h, 0 < Y < a X})`. Must be
       broadcastable with `a`.
    a: A `float` `Tensor` defined as in `P({X > h, 0 < Y < a X})`. Must be
       broadcastable with `h`.
    name: A name for the operation (optional).

  Returns:
    owens_t: A `Tensor` with the same type as `h` and `a`,

  #### References

  [1]: Patefield, Mike, and D. A. V. I. D. Tandy. "Fast and accurate calculation
       of Owen’s T function." Journal of Statistical Software 5.5 (2000): 1-25.
       http://www.jstatsoft.org/v05/i05/paper
  [2]: Owen, Donald Bruce. "A table of normal integrals: A table."
       Communications in Statistics-Simulation and Computation 9.4 (1980):
       389-419.
  """
  # pylint: enable=line-too-long
  with tf.name_scope(name or 'owens_t'):
    dtype = dtype_util.common_dtype([h, a], tf.float32)
    h = tf.convert_to_tensor(h, dtype=dtype, name='h')
    a = tf.convert_to_tensor(a, dtype=dtype, name='a')
    return _owens_t_custom_gradient(h, a)
