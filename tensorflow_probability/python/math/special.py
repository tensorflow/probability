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
