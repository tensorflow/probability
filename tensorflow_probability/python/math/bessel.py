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
"""Implements bessel functions in TensorFlow."""

from __future__ import absolute_import
from __future__ import division
# [internal] enable type annotations
from __future__ import print_function

# Dependency imports
import numpy as np
import tensorflow.compat.v2 as tf
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import tensorshape_util


__all__ = [
    'bessel_iv_ratio',
    'bessel_ive',
    'bessel_kve',
    'log_bessel_ive',
]


def _compute_general_continued_fraction(
    max_iterations,
    numerator_denominator_args_list,
    tolerance=None,
    partial_numerator_fn=None,
    partial_denominator_fn=None,
    dtype=tf.float32,
    name=None):
  """Compute a general continued fraction.

  Given at least one of `partial_numerator_fn` and `partial_denominator_fn`,
  compute the continued fraction associated with it via the forward recurrence.

  Let `a_i = partial_numerator_fn` and `b_i = partial_denominator_fn`. Then,
  this evaluates the infinite continued fraction:

  ```result = a_1 / (b_1 + a_2 / (b_2 + a_3 / (b_3 .....)```.

  If `partial_numerator_fn` or `partial_denominator_fn` are not given, then
  `a_i` (respectively `b_i`) are assumed to be 1. However one must be given.

  NOTE: Use this with caution. Forward recursion doesn't have numerical
  stability guarantees, compared to backward recursion.


  Args:
    max_iterations: Integer `Tensor` specifying the maximum number of terms to
      use.
    numerator_denominator_args_list: Arguments to pass in to
      `partial_numerator_fn` and `partial_denominator_fn`.
    tolerance: Float `Tensor` specifying the maximum acceptable tolerance
      between convergents. If unset, convergence is dictated by the number
      of iterations.
      Default value: `None`.
    partial_numerator_fn: Python callable that takes in as its first argument
      the current iteration count (an integer >= 1), and a list of *args, and
      returns a `Tensor`. These are used as partial numerators for the
      continued fraction.
      Default value: `None`.
    partial_denominator_fn: Python callable that takes in as its first argument
      the current iteration count (an integer >= 1), and a list of *args, and
      returns a `Tensor`. These are used as partial denominators for the
      continued fraction.
      Default value: `None`.
    dtype: The default dtype of the continued fraction. Default: `float32`.
    name: A name for the operation (optional).
      Default value: `None` (i.e., 'continued_fraction').

  Returns:
    Continued fraction computed to `max_iterations` iterations and/or
    up to absolute error `tolerance`.

  #### References
  [1]: Walter Gautschi and Josef Slavik. On the Computation of Modified
       Bessel Function Ratios. http://www.jstor.com/stable/2006491
  """
  with tf.name_scope(name or 'continued_fraction'):
    dtype = dtype_util.common_dtype(
        numerator_denominator_args_list, dtype)

    if (partial_numerator_fn is None) and (partial_denominator_fn is None):
      raise ValueError('Expect one of `partial_numerator_fn` and '
                       '`partial_denominator_fn` to be set.')

    def _continued_fraction_one_step(
        unused_should_stop,
        numerator,
        previous_numerator,
        denominator,
        previous_denominator,
        iteration_count):
      partial_denominator = 1.
      if partial_denominator_fn:
        partial_denominator = partial_denominator_fn(
            iteration_count, *numerator_denominator_args_list)
      new_numerator = partial_denominator * numerator
      new_denominator = partial_denominator * denominator

      partial_numerator = 1.
      if partial_numerator_fn:
        partial_numerator = partial_numerator_fn(
            iteration_count, *numerator_denominator_args_list)
      new_numerator = new_numerator + partial_numerator * previous_numerator
      new_denominator = (
          new_denominator + partial_numerator * previous_denominator)

      should_stop_next = iteration_count > max_iterations

      if tolerance is not None:
        # We can use a more efficient computation when the partial numerators
        # are 1.
        if partial_numerator_fn is None:
          # We now want to compute to relative error between the fraction at
          # this iteration, vs. the previous iteration.
          # Let h_i be the numerator and k_i the denominator, and a_i be the
          # i-th term.
          # h_i / k_i - h_{i-1} / k_{i-1} =
          # (h_i * k_{i - 1} - h_{i - 1} * k_i) / (k_i * k_{i - 1}) =
          # ((a_i h_{i - 1} + h_{i - 2}) * k_{i - 1} -
          # (a_i k_{i - 1} + k_{i - 2}) * h_{i - 1}) / (k_i * k_{i - 1}) =
          # -(h_{i - 1} * k_{i - 2} - h_{i - 2} * k_{i - 1}) / (k_i * k_{i - 1})
          # This suggests we should prove something about the numerator
          # inductively, and indeed
          # (h_i * k_{i - 1} - h_{i - 1} * k_i) = (-1)**i
          delta = tf.math.reciprocal(new_denominator * denominator)
        # We actually need to compute the difference of fractions.
        else:
          delta = new_numerator / new_denominator - numerator / denominator

        converged = tf.math.abs(delta) <= tolerance
        should_stop_next = tf.reduce_all(converged) | should_stop_next
      return (should_stop_next,
              new_numerator,
              numerator,
              new_denominator,
              denominator,
              iteration_count + 1.)

    # This is to infer the correct shape of tensors
    if partial_denominator_fn:
      term = partial_denominator_fn(1., *numerator_denominator_args_list)
    else:
      term = partial_numerator_fn(1., *numerator_denominator_args_list)

    zeroth_numerator = tf.ones_like(term, dtype=dtype)
    zeroth_denominator = tf.zeros_like(term, dtype=dtype)
    first_numerator = tf.zeros_like(term, dtype=dtype)
    first_denominator = tf.ones_like(term, dtype=dtype)

    results = tf.while_loop(
        cond=lambda stop, *_: ~stop,
        body=_continued_fraction_one_step,
        loop_vars=(
            False,
            first_numerator,
            zeroth_numerator,
            first_denominator,
            zeroth_denominator,
            tf.cast(1., dtype=dtype)))
    return results[1] / results[3]


@tf.custom_gradient
def bessel_iv_ratio(v, z, name=None):
  """Computes `I_{v} (z) / I_{v - 1} (z)` in a numerically stable way.

  Let I(v, z) be the modified bessel function of the first kind. This computes
  the ratio of I(v, z) / I(v - 1, z). This can be more numerically stable
  and faster than computing the ratio directly.

  This uses a continued fraction approximation attributed to Gauss for
  computing this quantity in the limit where z <= v, and a continued fraction
  approximation attributed to Perron for z > v.

  Args:
    v: value for which `I_{v}(z) / I_{v - 1}(z)` should be computed. Expect
      v > 0.
    z: value for which `I_{v}(z) / I_{v - 1}(z)` should be computed. Expect
      z > 0.
    name: A name for the operation (optional).
      Default value: `None` (i.e., 'bessel_iv_ratio').

  Returns:
    I(v, z) / I(v - 1, z).

  #### References
  [1]: Walter Gautschi and Josef Slavik. On the Computation of Modified
       Bessel Function Ratios. http://www.jstor.com/stable/2006491
  """
  with tf.name_scope(name or 'bessel_iv_ratio'):
    dtype = dtype_util.common_dtype([v, z], tf.float32)
    v = tf.convert_to_tensor(v, dtype=dtype)
    z = tf.convert_to_tensor(z, dtype=dtype)

    # I(v, z) == I(-v, z) when v is an integer.
    v_is_integer = tf.math.equal(tf.math.floor(v), v)
    v = tf.where((v < 0.) & v_is_integer, -v, v)

    np_finfo = np.finfo(dtype_util.as_numpy_dtype(dtype))
    tolerance = tf.cast(np_finfo.resolution, dtype=dtype)

    safe_to_use_perron = z > v

    def gauss_term_fn(iteration_count, v, z):
      """Terms for the Gauss continued fraction."""
      return tf.math.square(z) / 4. / (
          (v + iteration_count - 1) * (v + iteration_count))

    # The Gauss continued fraction converges faster for z < v.
    # For z > v, set z to something much less than v.
    safe_z_less_v = tf.where(safe_to_use_perron, v / 1000., z)

    # We use forward recurrence for the Gauss continued fraction.
    # This is so that we can do early termination.
    # There are a few reasons why this doesn't overflow:
    # * All partial numerators / denominators are positive.
    # * Partial numerators approach zero as 1 / n**2, where
    #   n is the iteration count.
    # * All partial numerators are less than 1.
    # Combined with the recurrence, this ensures no overflow.
    # as the number of iterations -> infinity.
    gauss_cf = _compute_general_continued_fraction(
        # Use a max of 200 steps. Almost always we will be much less
        # than this.
        200, [v, safe_z_less_v], tolerance=tolerance,
        partial_numerator_fn=gauss_term_fn)
    # Add the zeroth term for the Gauss continued fraction.
    gauss_cf = tf.math.reciprocal((1. + gauss_cf) * 2. * v / z)

    # For the Perron CF we use the backward recurrence. This is because
    # generally the backward recurrence is more numerically stable
    # than forward recurrence, especially with negative terms.
    # We use a flat 50 steps. Anecdotally, for z > v, convergence is
    # much faster than that.

    # The Perron continued fraction converges much faster for z >> v.
    # For z < v, set z to something much greater than v.
    safe_z_greater_v = tf.where(~safe_to_use_perron, 1000. * v, z)

    def perron_term_fn(iteration_count, v, z):
      """Terms for the Perron continued fraction."""
      return -0.5 * z * (v + iteration_count - 0.5) / (
          (v + z + (iteration_count - 1.) / 2.) *
          (v + z + iteration_count / 2.))

    total_perron_iteration_count = 50

    def _backward_cf_one_step(iteration_count, cf):
      cf = perron_term_fn(
          total_perron_iteration_count - iteration_count,
          v, safe_z_greater_v) / (1. + cf)
      return [iteration_count + 1., cf]

    # For the Perron CF, we omit the first numerator because it
    # has a different form.

    _, perron_cf = tf.while_loop(
        cond=lambda i, _: i < total_perron_iteration_count - 1,
        body=_backward_cf_one_step,
        # Use 50 iterations. Empirically, the Perron continued fraction
        # converges much faster than this.
        loop_vars=[tf.cast(0., dtype=dtype), tf.zeros_like(safe_z_greater_v)])
    first_term = -0.5 * z * (v + 0.5) / ((v + z / 2.) * (v + z + 0.5))

    perron_cf = first_term / (1. + perron_cf)

    # Add the zeroth term for the Perron continued fraction.
    perron_zeroth_term = (z + 2 * v) / z
    perron_cf = tf.math.reciprocal(perron_zeroth_term * (1. + perron_cf))
    result = tf.where(safe_to_use_perron, perron_cf, gauss_cf)

    def grad(dy):
      """Computes the derivative of the ratio elementwise with respect to z.

      For shorthand, let `I(v) = I(v, z)`, `R(v) = I(v, z) / I(v - 1, z)`

      ```
      R'(v) = (I'(v)I(v - 1) - I(v)I'(v - 1)) / I(v - 1) ** 2
             = 0.5 * ((I(v - 1) + I(v + 1))I(v - 1) - I(v)(
                  I(v) + I(v - 2))) / I(v - 1) ** 2
             = 0.5 * (1. + I(v + 1) / I(v - 1) - (I(v) / I(v - 1)) ** 2 - (
                  I(v) / I(v - 1)) * (I(v - 2) / I(v - 1)))
             = 0.5 * (1. + R(v + 1) * R(v) - R(v) ** 2 - R(v) / R(v - 1))
             = 0.5 * (1. + R(v) * (R(v + 1) - R(v) - 1. / R(v - 1)))
      ```
      To avoid computing R(v - 1) when v <= 1 (which is not valid),
      we can rewrite `I(v - 2) = 2 (v - 1) / z * I(v - 1) + I(v)`.
      Thus the last term becomes:
      ```
      -1. / R(v - 1) = -I(v - 2) / I(v - 1) = -2 (v - 1) / z - R(v)
      ```

      Args:
        dy: A Tensor with type `float32` or `float64`.

      Returns:
        A Tensor with same shape and dtype as `z`.
      """
      grad_z = 0.5 * (1. + result * (
          bessel_iv_ratio(v + 1., z) - 2. * result - 2. * (v - 1) / z)) * dy

      # We don't have an easily computable gradient with respect to v at the
      # moment, so ignore that for now.
      _, grad_z = _fix_gradient_for_broadcasting(
          v, z, tf.ones_like(grad_z), grad_z)
      return None, grad_z

    return result, grad


# Used for the polynomial coefficients parameterizing Olver's expansion.
_ASYMPTOTIC_OLVER_EXPANSION_COEFFICIENTS = [
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
     0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
     -0.20833333333333334, 0., 0.125, 0.],
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
     0., 0., 0., 0., 0., 0., 0., 0., 0., 0.3342013888888889, 0.,
     -0.40104166666666669, 0., 0.0703125, 0., 0.0],
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
     0., 0., 0., 0., 0., 0., -1.0258125964506173, 0., 1.8464626736111112,
     0., -0.89121093750000002, 0., 0.0732421875, 0., 0., 0.],
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
     0., 0., 0., 4.6695844234262474, 0., -11.207002616222995, 0.,
     8.78912353515625, 0., -2.3640869140624998, 0., 0.112152099609375,
     0., 0., 0., 0.],
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
     -28.212072558200244, 0., 84.636217674600744, 0., -91.818241543240035,
     0., 42.534998745388457, 0., -7.3687943594796312, 0., 0.22710800170898438,
     0., 0., 0., 0., 0.],
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 212.5701300392171, 0.,
     -765.25246814118157, 0., 1059.9904525279999, 0., -699.57962737613275,
     0., 218.19051174421159, 0., -26.491430486951554, 0., 0.57250142097473145,
     0., 0., 0., 0., 0., 0.],
    [0., 0., 0., 0., 0., 0., 0., 0., 0., -1919.4576623184068, 0.,
     8061.7221817373083, 0., -13586.550006434136, 0., 11655.393336864536,
     0., -5305.6469786134048, 0., 1200.9029132163525, 0.,
     -108.09091978839464, 0., 1.7277275025844574, 0., 0., 0., 0., 0., 0., 0.],
    [0., 0., 0., 0., 0., 0., 20204.291330966149, 0., -96980.598388637503, 0.,
     192547.0012325315, 0., -203400.17728041555, 0., 122200.46498301747,
     0., -41192.654968897557, 0., 7109.5143024893641, 0.,
     -493.915304773088, 0., 6.074042001273483, 0., 0., 0., 0., 0.,
     0., 0., 0.],
    [0., 0., 0., -242919.18790055133, 0., 1311763.6146629769, 0.,
     -2998015.9185381061, 0., 3763271.2976564039, 0., -2813563.2265865342, 0.,
     1268365.2733216248, 0., -331645.17248456361, 0., 45218.768981362737, 0.,
     -2499.8304818112092, 0., 24.380529699556064, 0., 0., 0., 0., 0.,
     0., 0., 0., 0.0],
    [3284469.8530720375, 0., -19706819.11843222, 0., 50952602.492664628,
     0., -74105148.211532637, 0., 66344512.274729028, 0., -37567176.660763353,
     0., 13288767.166421819, 0., -2785618.1280864552, 0., 308186.40461266245,
     0., -13886.089753717039, 0., 110.01714026924674, 0., 0., 0., 0., 0.,
     0., 0., 0., 0., 0.]
]


def _olver_asymptotic_uniform(v, z, name=None):
  """Use Olver's uniform asymptotic expansion for the Bessel function.

  Olver's uniform asymptotic expansion [1] is specified by

  `I_v(v, v * z) ~ f(a, v) * sum_k U_k(1 / sqrt(1 + z^2)) / v^k`
  `K_v(v, v * z) ~ f(a, v) * sum_k (-1) ** k * U_k(1 / sqrt(1 + z^2)) / v^k`
  where

  * `f(a, v) = `exp(v * a) / (sqrt(2 * pi * v) * (1 + z^2)^0.25)`
  * `U_k(z)` are polynomials that are given in [2]. We use the first
  10 polynomials.

  #### References
  [1]: Digital Library of Mathematical Functions: https://dlmf.nist.gov/10.41
  [2]: F. Olver, Tables for Bessel Functions of Moderate or Large Orders.
       National Physical Laboratory Mathematical Tables, Vol. 6.
       Department of Scientific and Industrial Research

  Args:
    v: value for which `I_{v}(z)` and `K_{v}(z) should be computed.
    z: value for which `I_{v}(z)` and `K_{v}(z) should be computed.
    name: A name for the operation (optional).
      Default value: `None` (i.e., 'olver_asymptotic_uniform').
  Returns:
    ive, kve: Asymptotic approximations to the modified bessel functions of the
      first and second kind.
  """
  with tf.name_scope(name or 'olver_asymptotic_uniform'):
    v_abs = tf.math.abs(v)
    w = z / v_abs
    t = tf.math.rsqrt(1. + tf.math.square(w))
    n_ufactors = len(_ASYMPTOTIC_OLVER_EXPANSION_COEFFICIENTS)

    divisor = v_abs
    ive_sum = 1.
    kve_sum = 1.

    # Note the polynomials have properties of oddness and evenness so that
    # could be taken advantage of when doing evaluation. For simplicity,
    # we naively sum using Horner's method.
    for i in range(n_ufactors):
      coeff = 0.
      for c in _ASYMPTOTIC_OLVER_EXPANSION_COEFFICIENTS[i]:
        coeff = coeff * t + c
      term = coeff / divisor
      ive_sum = ive_sum + term
      kve_sum = kve_sum + (term if i % 2 == 1 else -term)
      divisor = divisor * v_abs

    # This is modified from the original impl to be more numerically stable
    # since we are subtracting off x.
    shared_prefactor = 1. / (tf.math.sqrt(
        1 + tf.math.square(w)) + w) + tf.math.log(w / (1 + 1. / t))
    i_prefactor = tf.math.sqrt(
        t / (2 * np.pi * v_abs)) * tf.math.exp(v_abs * shared_prefactor)

    # Not the same here since they will have the same sign.
    k_prefactor = tf.math.sqrt(np.pi * t / (2 * v_abs)) * tf.math.exp(
        -v_abs * shared_prefactor)
    kve = k_prefactor * kve_sum

    ive = tf.where(
        v > 0.,
        i_prefactor * ive_sum,
        # This uses the reflection formulation for negative v, to
        # write this in terms of kve.
        i_prefactor * ive_sum + 2 / np.pi * tf.math.sin(
            np.pi * v_abs) * k_prefactor * kve_sum * tf.math.exp(-2. * z))
    return ive, kve


def _evaluate_temme_coeffs(v):
  """Numerically stable computation of difference of gammas."""
  # This function computes the following quantities:
  # coeff1 = (1 / Gamma(1 - v) - 1 / Gamma(1 + v)) / 2v
  # coeff2 = (1 / Gamma(1 - v) + 1 / Gamma(1 + v)) / 2
  # gamma1mv = 1 / Gamma(1 - v)
  # gamma1pv = 1 / Gamma(1 + v)
  # Naive computation of the above two coefficients leads to
  # catastrophic cancellations. The below function computes
  # Chebyshev expansions to `coeff1` and `coeff2`.

  # Stable evaluation of the coefficients for the Temme power series.
  # We refer to [1] for the numerical evaluation
  # [1] Numerical Recipes in C. The Art of Scientific Computing,
  #   2nd Edition, 1992

  # These are Chebyshev expansion coefficients defined in 6.7.18 in [1].
  coeff1_coeffs = [-1.142022680371168e0, 6.5165112670737e-3,
                   3.087090173086e-4, -3.4706269649e-6, 6.9437664e-9,
                   3.67795e-11, -1.356e-13]
  coeff2_coeffs = [1.843740587300905e0, -7.68528408447867e-2,
                   1.2719271366546e-3, -4.9717367042e-6, -3.31261198e-8,
                   2.423096e-10, -1.702e-13, -1.49e-15]
  w = 8 * tf.math.square(v) - 1.

  # Use Clenshaw's recurrence for evaluating the Chebyshev polynomials
  # associated to the coefficients.
  y = 2 * w

  prev = 0.
  current = 0.
  for i in reversed(range(1, len(coeff1_coeffs))):
    temp = current
    current = y * current - prev + coeff1_coeffs[i]
    prev = temp
  coeff1 = w * current - prev + 0.5 * coeff1_coeffs[0]

  prev = 0.
  current = 0.
  for i in reversed(range(1, len(coeff2_coeffs))):
    temp = current
    current = y * current - prev + coeff2_coeffs[i]
    prev = temp
  coeff2 = w * current - prev + 0.5 * coeff2_coeffs[0]
  gamma1pv = coeff2 - v * coeff1
  gamma1mv = coeff2 + v * coeff1
  return coeff1, coeff2, gamma1pv, gamma1mv


def _temme_series(v, z):
  """Computes K(v, z) and K(v + 1., z) via Power series expansion."""
  # This is based on:
  # [1] N. Temme, On the Numerical Evaluation of the Modified Bessel Function
  #   of the Third Kind. Journal of Computational Physics 19, 1975.
  # [2] Numerical Recipes in C. The Art of Scientific Computing,
  #   2nd Edition, 1992
  # We will assume |z| <= 2. and |v| < 0.5 for fast convergence.
  dtype = dtype_util.common_dtype([v, z], tf.float32)
  numpy_dtype = dtype_util.as_numpy_dtype(dtype)
  tol = tf.cast(np.finfo(numpy_dtype).eps, dtype=dtype)

  # The initial series term is defined by 6.7.39 in [2]. We compute
  # related coefficients and quantities.
  coeff1, coeff2, gamma1pv_inv, gamma1mv_inv = _evaluate_temme_coeffs(v)

  z_sq = tf.math.square(z)

  logzo2 = tf.math.log(z / 2.)
  mu = -v * logzo2
  sinc_v = tf.where(
      tf.math.equal(v, 0.),
      numpy_dtype(1.),
      tf.math.sin(np.pi * v) / (np.pi * v))
  sinhc_mu = tf.where(
      tf.math.equal(mu, 0.),
      numpy_dtype(1.),
      tf.math.sinh(mu) / mu)
  # These are defined in 6.7.17 in [2].
  initial_f = (coeff1 * tf.math.cosh(mu) +
               coeff2 * -logzo2 * sinhc_mu) / sinc_v
  initial_p = 0.5 * tf.math.exp(mu) / gamma1pv_inv
  initial_q = 0.5 * tf.math.exp(-mu) / gamma1mv_inv
  max_iterations = 1000

  def body_fn(should_stop, index, f, p, q, coeff, kv_sum, kvp1_sum):
    f = tf.where(
        should_stop,
        f,
        (index * f + p + q) / (tf.math.square(index) - tf.math.square(v)))
    p = tf.where(should_stop, p, p / (index - v))
    q = tf.where(should_stop, q, q / (index + v))
    h = p - index * f
    # c_k = (z ** 2 / 4) ** k / (k!)
    coeff = tf.where(should_stop, coeff, coeff * z_sq / (4 * index))
    kv_sum = tf.where(should_stop, kv_sum, kv_sum + coeff * f)
    kvp1_sum = tf.where(should_stop, kvp1_sum, kvp1_sum + coeff * h)
    index = index + 1
    should_stop = (
        tf.math.abs(coeff * f) < tf.math.abs(kv_sum) * tol) | (
            index > max_iterations)
    return should_stop, index, f, p, q, coeff, kv_sum, kvp1_sum

  _, _, _, _, _, _, kv_sum, kvp1_sum = tf.while_loop(
      cond=lambda stop, *_: tf.reduce_any(~stop),
      body=body_fn,
      loop_vars=(
          tf.zeros_like(initial_f, dtype=tf.bool),
          tf.cast(1., dtype),
          initial_f,
          initial_p,
          initial_q,
          tf.ones_like(initial_p),
          initial_f,
          initial_p))

  return kv_sum, 2 * kvp1_sum / z


def _continued_fraction_kv(v, z):
  """Compute Modified Bessels of Second Kind using Hypergeometric functions.

  First define `k_n(z) = (-1)**n U(v + n + 0.5, 2 * v + 1., 2 * z)` where
  `U(a, b, z)` is the confluent hypergeometric function.

  We can compute via [1] `K_v(z)` and `K_{v + 1}(z)` via the identities:

  `K_v(z) = sqrt(pi) * (2 * z) ** v * ezp(-z) * k_0(z)`,
  `K_{v + 1}(z) = K_v(z) * (v + z + 0.5 - k_1(z) / k_0(z)`,

  This function aims to compute the ratio `k_1(z) / k_0(z)` via
  a continued fraction, under the assumption |v| < 0.5, and finally
  `K_v(z)` and `K_{v + 1}(z)`.

  Args:
    v: Floating-point `Tensor` broadcastable with `z`.
    z: Floating-point `Tensor` broadcastable with `v`.
  Returns:
    kv_tuple: `K_v(z)` and `K_{v + 1}(z)`.

  #### References
  [1] N. Temme, On the Numerical Evaluation of the Modified Bessel Function
    of the Third Kind. Journal of Computational Physics 19, 1975.
  [2] J. Campbell. On Temme's Algorithm for the Modified Bessel Function
    of the Third Kind. https://dl.acm.org/doi/pdf/10.1145/355921.355928
  [3] Numerical Recipes in C. The Art of Scientific Computing,
    2nd Edition, 1992
  """
  dtype = dtype_util.common_dtype([v, z], tf.float32)
  tol = tf.cast(np.finfo(dtype_util.as_numpy_dtype(
      dtype)).eps, dtype=dtype)
  max_iterations = 1000

  # Use Steed's algorithm to evaluate the confluent hypergeometric
  # function continued fraction in a numerically stable manner.
  def steeds_algorithm(
      should_stop,
      index,
      partial_numerator,
      partial_denominator,
      denominator_ratio,
      convergent_difference,
      hypergeometric_ratio,
      # Terms for recurrence in 6.7.36 in [3].
      k_0,
      k_1,
      # Intermediate coefficient in 6.7.30 in [3].
      c,
      # Intermediate sum in 6.7.35 in [3].
      q,
      hypergeometric_sum):
    # The numerator is v**2 - (index - 0.5) ** 2
    partial_numerator = partial_numerator - 2. * (index - 1.)
    c = tf.where(should_stop, c, -c * partial_numerator / index)
    next_k = (k_0 - partial_denominator * k_1) / partial_numerator
    k_0 = tf.where(should_stop, k_0, k_1)
    k_1 = tf.where(should_stop, k_1, next_k)
    q = tf.where(should_stop, q, q + c * next_k)
    partial_denominator = partial_denominator + 2.
    denominator_ratio = 1. / (
        partial_denominator + partial_numerator * denominator_ratio)
    convergent_difference = tf.where(
        should_stop, convergent_difference,
        convergent_difference * (
            partial_denominator * denominator_ratio - 1.))
    hypergeometric_ratio = tf.where(
        should_stop,
        hypergeometric_ratio,
        hypergeometric_ratio + convergent_difference)
    hypergeometric_sum = tf.where(
        should_stop,
        hypergeometric_sum,
        hypergeometric_sum + q * convergent_difference)
    index = index + 1
    should_stop = (tf.math.abs(q * convergent_difference) <
                   tf.math.abs(hypergeometric_sum) * tol) | (
                       index > max_iterations)
    return (should_stop,
            index,
            partial_numerator,
            partial_denominator,
            denominator_ratio,
            convergent_difference,
            hypergeometric_ratio,
            k_0, k_1, c, q, hypergeometric_sum)

  initial_numerator = tf.math.square(v) - 0.25
  initial_denominator = 2 * (z + 1.)
  initial_ratio = 1. / initial_denominator + tf.zeros_like(v)
  initial_seq = -initial_numerator + tf.zeros_like(z)

  (_, _, _, _, _, _, hypergeometric_ratio,
   _, _, _, _, hypergeometric_sum) = tf.while_loop(
       cond=lambda stop, *_: tf.reduce_any(~stop),
       body=steeds_algorithm,
       loop_vars=(
           tf.zeros_like(v + z, dtype=tf.bool),
           tf.cast(2., dtype=dtype),
           initial_numerator,
           initial_denominator,
           initial_ratio,
           initial_ratio,
           initial_ratio,
           tf.zeros_like(v + z),
           tf.ones_like(v + z),
           initial_seq,
           initial_seq,
           1 - initial_numerator * initial_ratio))

  kv = tf.math.sqrt(np.pi / (2 * z))  * tf.math.exp(-z) / hypergeometric_sum
  kvp1 = kv * (0.5 + v + z + initial_numerator * hypergeometric_ratio) / z
  return kv, kvp1


def _temme_expansion(v, x):
  """Compute modified bessel functions using Temme's method."""
  # The implementation of this is based on [1].
  # [1] N. Temme, On the Numerical Evaluation of the Modified Bessel Function
  #   of the Third Kind. Journal of Computational Physics 19, 1975.
  dtype = dtype_util.common_dtype([v, x], tf.float32)
  numpy_dtype = dtype_util.as_numpy_dtype(dtype)
  v_less_than_zero = v < 0.
  v = tf.math.abs(v)
  n = tf.math.round(v)
  # Use this to compute Kv(u, x) and Kv(u + 1., x)
  u = v - n
  x_abs = tf.math.abs(x)

  small_x = tf.where(x_abs <= 2., x_abs, numpy_dtype(0.1))
  large_x = tf.where(x_abs > 2., x_abs, numpy_dtype(1000.))
  temme_ku, temme_kup1 = _temme_series(u, small_x)
  cf_ku, cf_kup1 = _continued_fraction_kv(u, large_x)

  ku = tf.where(x_abs <= 2., temme_ku, cf_ku)
  kup1 = tf.where(x_abs <= 2., temme_kup1, cf_kup1)

  # Now use the forward recurrence for modified bessel functions
  # to compute Kv(v, x). That is,
  # K_{v + 1}(z) - (2v / z) K_v(z) - K_{v - 1}(z) = 0.
  # This is known to be forward numerically stable.

  def bessel_recurrence(index, kv, kvp1):
    next_kvp1 = 2 * (u + index) * kvp1 / x_abs + kv
    kv = tf.where(index > n, kv, kvp1)
    kvp1 = tf.where(index > n, kvp1, next_kvp1)
    return index + 1., kv, kvp1

  _, kv, kvp1 = tf.while_loop(
      cond=lambda i, *_: tf.reduce_any(i <= n),
      body=bessel_recurrence,
      loop_vars=(tf.cast(1., dtype=dtype), ku, kup1))

  # Finally, it is known that the Wronskian
  # det(I_v * K'_v - K_v * I'_v) = - 1. / x. We can
  # use this to evaluate I_v by taking advantage of identities of Bessel
  # derivatives.

  iv = tf.math.reciprocal(
      x_abs * (kv * bessel_iv_ratio(v + 1., x) + kvp1))
  z = u + tf.math.mod(n, 2.)

  iv = tf.where(
      v_less_than_zero,
      iv + 2. / np.pi * tf.math.sin(np.pi * z) * kv, iv)
  iv = tf.where(
      tf.math.equal(x, 0.),
      tf.where(tf.math.equal(v, 0.), numpy_dtype(1.), numpy_dtype(0.)), iv)
  iv = tf.where(tf.math.equal(x, 0.) & v_less_than_zero,
                tf.where(
                    tf.math.equal(z, tf.math.floor(z)),
                    iv,
                    numpy_dtype(np.inf)),
                iv)
  kv = tf.where(tf.math.equal(x, 0.), numpy_dtype(np.inf), kv)
  iv = tf.where(x < 0., numpy_dtype(np.nan), iv)
  kv = tf.where(x < 0., numpy_dtype(np.nan), kv)
  return iv, kv


@tf.custom_gradient
def bessel_ive(v, z, name=None):
  """Computes exponentially scaled modified Bessel function of the first kind.

  This function computes `Ive`, which is an exponentially scaled version
  of the modified Bessel function of the first kind.

  `Ive(v, z) = Iv(v, z) * exp(-abs(z))`

  Warning: Gradients with respect to the first parameter `v` are currently not
  defined.

  Args:
    v: Floating-point `Tensor` broadcastable with `z` for which `Ive(v, z)`
      should be computed. `v` is expected to be non-negative.
    z: Floating-point `Tensor` broadcastable with `v` for which `Ive(v, z)`
      should be computed. If `z` is negative, `v` is expected to be an integer.
    name: A name for the operation (optional).
      Default value: `None` (i.e., 'bessel_ive').

  Returns:
    bessel_ive: Exponentially modified Bessel Function of the first kind.
  """
  with tf.name_scope(name or 'bessel_ive'):
    dtype = dtype_util.common_dtype([v, z], tf.float32)
    numpy_dtype = dtype_util.as_numpy_dtype(dtype)
    v = tf.convert_to_tensor(v, dtype=dtype)
    z = tf.convert_to_tensor(z, dtype=dtype)

    # I_{-v} == I_{v} for negative integers
    v_is_integer = tf.math.equal(tf.math.floor(v), v)
    v_abs = tf.where((v < 0.) & v_is_integer, -v, v)
    # Handle v = 0 specially with tf.math.special.i0e
    v_abs = tf.where(tf.math.equal(v_abs, 0.), numpy_dtype(1.), v_abs)

    z_abs = tf.math.abs(z)
    # Handle the zero case specially.
    z_abs = tf.where(tf.math.equal(z_abs, 0.), numpy_dtype(1.), z_abs)

    small_v = tf.where(tf.math.abs(v_abs) < 50., v_abs, numpy_dtype(0.1))
    large_v = tf.where(tf.math.abs(v_abs) >= 50., v_abs, numpy_dtype(1000.))

    olver_ive, _ = _olver_asymptotic_uniform(large_v, z_abs)
    temme_ive = _temme_expansion(small_v, z_abs)[0] * tf.math.exp(-z_abs)
    ive = tf.where(tf.math.abs(v) >= 50., olver_ive, temme_ive)

    # Handle when z is zero.
    ive = tf.where(
        tf.math.equal(z, 0.),
        tf.where(
            tf.math.equal(v, 0.),
            numpy_dtype(1.),
            tf.where(
                v_abs < 0.,
                numpy_dtype(np.inf),
                numpy_dtype(0.))), ive)

    # Handle when z < 0.
    ive = tf.where((z < 0.) & ~v_is_integer, numpy_dtype(np.nan), ive)
    # If v is an odd integer, we flip sign of the computation.
    ive = tf.where((z < 0.) & v_is_integer & tf.math.not_equal(
        2. * tf.math.floor(v / 2.), v), -ive, ive)
    # TODO(b/169357627): Investigate removing this and the above statement
    # to modify v. The function evaluation should be stable and fast enough
    # to handle the 0 case.
    ive = tf.where(tf.math.equal(v, 0.), tf.math.bessel_i0e(z), ive)

    def grad(dy):
      """Computes the derivative of Bessel Ive.

      Args:
        dy: A Tensor with type `float32` or `float64`.

      Returns:
        A Tensor with same shape and dtype as `z`.
      """
      grad_yz = dy * (bessel_ive(v + 1., z) + (v / z - tf.math.sign(z)) * ive)
      # We need to reduce over broadcasted indices.
      _, grad_yz = _fix_gradient_for_broadcasting(
          v, z, tf.ones_like(grad_yz), grad_yz)

      # No gradient for this at the moment. This is a complicated expression
      # The gradient with respect to the parameter doesn't have an easy closed
      # form. More work will need to be done to ensure good numerics for the
      # gradient.
      # TODO(b/169357627): Implement gradients of modified bessel functions with
      # respect to parameters.
      grad_yv = None

      return grad_yv, grad_yz

    return ive, grad


@tf.custom_gradient
def bessel_kve(v, z, name=None):
  """Computes exponentially scaled modified Bessel function of the 2nd kind.

  This function computes `Kve` which is an exponentially scaled version
  of the modified Bessel function of the first kind.

  `Kve(v, z) = Kv(v, z) * exp(abs(z))`

  Warning: Gradients with respect to the first parameter `v` are currently not
  defined.

  Args:
    v: Floating-point `Tensor` broadcastable with `z` for which `Kve(v, z)`
      should be computed. `v` is expected to be non-negative.
    z: Floating-point `Tensor` broadcastable with `v` for which `Kve(v, z)`
      should be computed. If `z` is negative, `v` is expected to be an integer.
    name: A name for the operation (optional).
      Default value: `None` (i.e., 'bessel_kve').

  Returns:
    bessel_kve: Exponentially modified Bessel Function of the 2nd kind.
  """
  with tf.name_scope(name or 'bessel_kve'):
    dtype = dtype_util.common_dtype([v, z], tf.float32)
    numpy_dtype = dtype_util.as_numpy_dtype(dtype)
    v = tf.convert_to_tensor(v, dtype=dtype)
    z = tf.convert_to_tensor(z, dtype=dtype)

    # K_{-v} == K_{v} for negative values.
    v = tf.math.abs(v)

    z_abs = tf.math.abs(z)
    # Handle the zero case specially.
    z_abs = tf.where(tf.math.equal(z_abs, 0.), numpy_dtype(1.), z_abs)

    small_v = tf.where(v < 50., v, numpy_dtype(0.1))
    large_v = tf.where(v >= 50., v, numpy_dtype(1000.))

    _, olver_kve = _olver_asymptotic_uniform(large_v, z_abs)
    temme_kve = _temme_expansion(small_v, z_abs)[1] * tf.math.exp(z_abs)
    kve = tf.where(v >= 50., olver_kve, temme_kve)

    # Handle when z is zero.
    kve = tf.where(tf.math.equal(z, 0.), numpy_dtype(np.inf), kve)
    kve = tf.where(z < 0., numpy_dtype(np.nan), kve)

    def grad(dy):
      """Computes the derivative of Bessel Kve.

      Args:
        dy: A Tensor with type `float32` or `float64`.

      Returns:
        A Tensor with same shape and dtype as `z`.
      """
      # Use bessel_kve(v - 1., z) This is because when z is small compared
      # to v, bessel_kve becomes large. Using bessel_kv(v + 1., z) would
      # exacerbate this.
      grad_yz = dy * ((z - v) / z * kve - bessel_kve(v - 1., z))
      _, grad_yz = _fix_gradient_for_broadcasting(
          v, z, tf.ones_like(grad_yz), grad_yz)

      # No gradient for this at the moment. This is a complicated expression
      # The gradient with respect to the parameter doesn't have an easy closed
      # form. More work will need to be done to ensure good numerics for the
      # gradient.
      # TODO(b/169357627): Implement gradients of modified bessel functions with
      # respect to parameters.
      grad_yv = None

      return grad_yv, grad_yz

    return kve, grad


@tf.custom_gradient
def log_bessel_ive(v, z, name=None):
  """Computes `log(tfp.math.bessel_ive(v, z))`.

  This function is the same as `log(tfp.math.bessel_ive(v, z))` except
  uses a more numerically stable gradient computation.

  Warning: Gradients with respect to the first parameter `v` are currently not
  defined.

  Args:
    v: Floating-point `Tensor` broadcastable with `z` for which `log(Ive(v, z))`
      should be computed. `v` is expected to be non-negative.
    z: Floating-point `Tensor` broadcastable with `v` for which `log(Ive(v, z))`
      should be computed. If `z` is negative, `v` is expected to be an integer.
    name: A name for the operation (optional).
      Default value: `None` (i.e., 'log_bessel_ive').

  Returns:
    log_bessel_ive: Log of Exponentially modified Bessel Function of the first
      kind.
  """
  with tf.name_scope(name or 'log_bessel_ive'):
    dtype = dtype_util.common_dtype([v, z], tf.float32)
    v = tf.convert_to_tensor(v, dtype=dtype)
    z = tf.convert_to_tensor(z, dtype=dtype)

    log_ive = tf.math.log(bessel_ive(v, z))

  def grad(dy):
    """Computes the derivative of log Bessel Ive.

    Args:
      dy: A Tensor with type `float32` or `float64`.

    Returns:
      A Tensor with same shape and dtype as `z`.
    """
    grad_yz = dy * (bessel_iv_ratio(v + 1., z) + (v / z - tf.math.sign(z)))
    # We need to reduce over broadcasted indices.j
    _, grad_yz = _fix_gradient_for_broadcasting(
        v, z, tf.ones_like(grad_yz), grad_yz)

    # No gradient for this at the moment. This is a complicated expression
    # The gradient with respect to the parameter doesn't have an easy closed
    # form. More work will need to be done to ensure good numerics for the
    # gradient.
    # TODO(b/169357627): Implement gradients of modified bessel functions with
    # respect to parameters.
    grad_yv = None

    return grad_yv, grad_yz

  return log_ive, grad


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
