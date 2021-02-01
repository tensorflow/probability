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
    'erfcinv',
    'round_exponential_bump_function',
    'lambertw',
    'lambertw_winitzki_approx',
    'log_gamma_correction',
    'log_gamma_difference',
    'lbeta',
    'owens_t',
]


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


@tf.custom_gradient
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
    wz = _lambertw_principal_branch(z, name)

    def grad(dy):
      """Computes the derivative of Lambert W of `z` element-wise.

      The first derivative W'(z) can be computed from W(z) as it holds

        W'(z) = W(z) / (z * (1 + W(z)))

      Args:
        dy: A Tensor with type `float32` or `float64`.

      Returns:
        A Tensor with same shape and dtype as `z`.
      """
      # At z = 0 the analytic expressions for the gradient results in a 0/0
      # expression.  However, the continuous expansion (l'Hospital rule) gives a
      # derivative of 1.0 at z = 0.  This case has to be handled separately with
      # a where clause.
      grad_wz = (dy * tf.where(tf.equal(z, 0.0),
                               tf.ones_like(wz),
                               wz / (z * (1. + wz))))
      return grad_wz

    return wz, grad


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
  result = tf.where(tf.math.equal(modified_a, 0.), 0., result)

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

  result = tf.where(tf.math.is_nan(a) | tf.math.is_nan(h), np.nan, result)
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
  """Computes Owen's T function of `h` and `a` element-wise.

  Owen's T function is defined as the combined probability of the event `X > h`
  and `0 < Y < a * X`, where `X` and `Y` are independent standard normal
  random variables.

  In integral form this is defined as `1 / (2 * pi)` times the integral of
  `exp(-0.5 * h ** 2 * (1 + x ** 2)) / (1 + x ** 2)` from `0` to `a`.
  `h` and `a` can be any real number

  The Owen's T implementation below is based on [1]


  # References
  [1] Patefield M., Tandy D., Fast and Accurate Calcuation of Owen's T-Function
      Journal of Statistical Software http://www.jstatsoft.org/v05/i05/paper

  Args:
    h: A `float` `Tensor` defined as in `P({X > h, 0 < Y < a X})`. Must be
       broadcastable with `a`.
    a: A `float` `Tensor` defined as in `P({X > h, 0 < Y < a X})`. Must be
       broadcastable with `h`.
    name: A name for the operation (optional).
  Returns:
    owens_t: A `Tensor` with the same type as `h` and `a`,
  """
  with tf.name_scope(name or 'owens_t'):
    dtype = dtype_util.common_dtype([h, a], tf.float32)
    h = tf.convert_to_tensor(h, dtype=dtype)
    a = tf.convert_to_tensor(a, dtype=dtype)
    return _owens_t_custom_gradient(h, a)
