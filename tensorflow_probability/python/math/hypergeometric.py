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
"""Implements hypergeometric functions in TensorFlow."""

import functools

# Dependency imports
import numpy as np
import tensorflow.compat.v2 as tf
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import prefer_static as ps
from tensorflow_probability.python.math import generic as tfp_math


__all__ = [
    'hyp2f1_small_argument',
]


def _hyp2f1_taylor_series(a, b, c, z):
  """Compute Hyp2F1(a, b, c, z) via the Taylor Series expansion."""
  with tf.name_scope('hyp2f1_taylor_series'):
    dtype = dtype_util.common_dtype([a, b, c, z], tf.float32)
    a = tf.convert_to_tensor(a, dtype=dtype)
    b = tf.convert_to_tensor(b, dtype=dtype)
    c = tf.convert_to_tensor(c, dtype=dtype)
    z = tf.convert_to_tensor(z, dtype=dtype)
    np_finfo = np.finfo(dtype_util.as_numpy_dtype(dtype))
    tolerance = tf.cast(np_finfo.resolution, dtype=dtype)

    broadcast_shape = functools.reduce(
        ps.broadcast_shape,
        [ps.shape(x) for x in [a, b, c, z]])

    def taylor_series(
        should_stop,
        index,
        term,
        taylor_sum,
        previous_term,
        previous_taylor_sum,
        two_before_taylor_sum):
      new_term = term * (a + index) * (b + index) * z / (
          (c + index) * (index + 1.))
      new_term = tf.where(should_stop, term, new_term)
      new_taylor_sum = tf.where(should_stop, taylor_sum, taylor_sum + new_term)

      # When a or be is near a negative integer n, it's possibly the term is
      # small because we are computing (a + n) * (b + n) in the numerator.
      # Checking that three consecutive terms are small compared their
      # corresponding sum will let us avoid this error.
      should_stop = (
          (tf.math.abs(new_term) < tolerance * tf.math.abs(taylor_sum)) &
          (tf.math.abs(term) < tolerance * tf.math.abs(previous_taylor_sum)) &
          (tf.math.abs(previous_term) < tolerance * tf.math.abs(
              two_before_taylor_sum)))
      return (
          tf.logical_or(should_stop, index > 2000.),
          index + 1.,
          new_term,
          new_taylor_sum,
          term,
          taylor_sum,
          previous_taylor_sum)

    (_, _, _, taylor_sum, _, _, _) = tf.while_loop(
        cond=lambda stop, *_: tf.reduce_any(~stop),
        body=taylor_series,
        loop_vars=(
            tf.zeros(broadcast_shape, dtype=tf.bool),
            tf.cast(0., dtype=dtype),
            # Only the previous term and taylor sum are used for computation.
            # The rest are used for checking convergence. We can safely set
            # these to zero.
            tf.ones(broadcast_shape, dtype=dtype),
            tf.ones(broadcast_shape, dtype=dtype),
            tf.zeros(broadcast_shape, dtype=dtype),
            tf.zeros(broadcast_shape, dtype=dtype),
            tf.zeros(broadcast_shape, dtype=dtype)))
    return taylor_sum


def _hyp2f1_fraction(a, b, c, z):
  """Compute 2F1(a, b, c, z) by using a running fraction."""
  with tf.name_scope('hyp2f1_fraction'):
    dtype = dtype_util.common_dtype([a, b, c, z], tf.float32)
    a = tf.convert_to_tensor(a, dtype=dtype)
    b = tf.convert_to_tensor(b, dtype=dtype)
    c = tf.convert_to_tensor(c, dtype=dtype)
    z = tf.convert_to_tensor(z, dtype=dtype)
    np_finfo = np.finfo(dtype_util.as_numpy_dtype(dtype))
    tolerance = tf.cast(np_finfo.resolution, dtype=dtype)

    broadcast_shape = functools.reduce(
        ps.broadcast_shape,
        [ps.shape(x) for x in [a, b, c, z]])

    def hypergeometric_fraction(
        should_stop,
        index,
        numerator_term0,
        numerator_term1,
        denominator,
        fraction,
        previous_fraction,
        two_before_fraction):
      new_numerator_term0 = (numerator_term0 + numerator_term1) * index
      new_numerator_term1 = (
          numerator_term1 * (a + index - 1.) * (b + index - 1.) * z) / (
              c + index - 1.)
      new_denominator = denominator * index

      largest_term = tf.math.maximum(
          tf.math.maximum(
              tf.math.abs(new_numerator_term0),
              tf.math.abs(new_numerator_term1)),
          tf.math.abs(new_denominator))

      should_rescale = largest_term > 100

      # Rescale to prevent overflow.
      new_numerator_term0 = tf.where(
          should_rescale,
          new_numerator_term0 / largest_term,
          new_numerator_term0)
      new_numerator_term1 = tf.where(
          should_rescale,
          new_numerator_term1 / largest_term,
          new_numerator_term1)
      new_denominator = tf.where(
          should_rescale,
          new_denominator / largest_term,
          new_denominator)

      new_fraction = (
          new_numerator_term0 + new_numerator_term1) / new_denominator
      new_fraction = tf.where(should_stop, fraction, new_fraction)

      # When a or be is near a negative integer n, it's possibly the term is
      # small because we are computing (a + n) * (b + n) in the numerator.
      # Checking that three consecutive terms are small compared their
      # corresponding sum will let us avoid this error.
      should_stop = (
          (tf.math.abs(new_fraction - fraction) <
           tolerance * tf.math.abs(fraction)) &
          (tf.math.abs(fraction - previous_fraction) <
           tolerance * tf.math.abs(previous_fraction)) &
          (tf.math.abs(previous_fraction - two_before_fraction) <
           tolerance * tf.math.abs(two_before_fraction)))
      return (
          tf.logical_or(should_stop, index > 1000.),
          index + 1.,
          new_numerator_term0,
          new_numerator_term1,
          new_denominator,
          new_fraction,
          fraction,
          previous_fraction)

    (_, _, _, _, _, fraction, _, _) = tf.while_loop(
        cond=lambda stop, *_: tf.reduce_any(~stop),
        body=hypergeometric_fraction,
        loop_vars=(
            tf.zeros(broadcast_shape, dtype=tf.bool),
            tf.cast(1., dtype=dtype),
            tf.zeros(broadcast_shape, dtype=dtype),
            tf.ones(broadcast_shape, dtype=dtype),
            tf.ones(broadcast_shape, dtype=dtype),
            # Only the previous term and taylor sum are used for computation.
            # The rest are used for checking convergence. We can safely set
            # these to zero.
            tf.ones(broadcast_shape, dtype=dtype),
            tf.zeros(broadcast_shape, dtype=dtype),
            tf.zeros(broadcast_shape, dtype=dtype)))
    return fraction


def _hyp2f1_small_parameters(a, b, c, z):
  """"Compute 2F1(a, b, c, z) when a, b, and c are small."""
  dtype = dtype_util.common_dtype([a, b, c, z], tf.float32)
  a = tf.convert_to_tensor(a, dtype=dtype)
  b = tf.convert_to_tensor(b, dtype=dtype)
  c = tf.convert_to_tensor(c, dtype=dtype)
  z = tf.convert_to_tensor(z, dtype=dtype)
  numpy_dtype = dtype_util.as_numpy_dtype(dtype)

  use_fraction_computation = (
      (tf.math.abs(c) < 1.) & (tf.math.abs(b) < 30.) & (tf.math.abs(c) < 30.))
  a_fraction = tf.where(use_fraction_computation, a, numpy_dtype(0.))
  b_fraction = tf.where(use_fraction_computation, b, numpy_dtype(0.))
  c_fraction = tf.where(use_fraction_computation, c, numpy_dtype(1.))
  result = _hyp2f1_fraction(a_fraction, b_fraction, c_fraction, z)

  a_taylor = tf.where(use_fraction_computation, numpy_dtype(0.), a)
  b_taylor = tf.where(use_fraction_computation, numpy_dtype(0.), b)
  c_taylor = tf.where(use_fraction_computation, numpy_dtype(1.), c)
  return tf.where(
      use_fraction_computation,
      result,
      _hyp2f1_taylor_series(a_taylor, b_taylor, c_taylor, z))


def _hyp2f1_large_negative_c(a, b, c, z):
  """Compute 2F1(a, b, c, z) when c < 0 and |c| large."""

  # The recurrences here are based on Gauss' continguous recurrence relations as
  # based on [1]
  # References
  # [1] M. Abramowitz, I. Stegun. Handbook of Mathematical Functions with
  #     Formulas, Graphs and Mathematical Tables.
  with tf.name_scope('hyp2f1_large_negative_c'):
    dtype = dtype_util.common_dtype([a, b, c, z], tf.float32)
    a = tf.convert_to_tensor(a, dtype=dtype)
    b = tf.convert_to_tensor(b, dtype=dtype)
    c = tf.convert_to_tensor(c, dtype=dtype)
    z = tf.convert_to_tensor(z, dtype=dtype)

    # We assume that c < 0 and a, b > 0.

    d = c - a - b
    integer_d = tf.math.floor(d)
    e = c + 2 - integer_d

    # If |a| >> |e|, use the recurrence for large a.
    recurrence_for_large_a = (
        (tf.math.abs(a) > tf.math.abs(e)) & (tf.math.abs(e - a) > 2))
    second_result = tf.where(
        recurrence_for_large_a,
        _hyp2f1_large_a(a, b, e, z),
        _hyp2f1_small_parameters(a, b, e, z))

    first_result = tf.where(
        recurrence_for_large_a,
        _hyp2f1_large_a(a, b, e + 1., z),
        _hyp2f1_small_parameters(a, b, e + 1., z))

    broadcast_shape = functools.reduce(
        ps.broadcast_shape,
        [ps.shape(x) for x in [a, b, c]])

    # We use recurrence 15.2.27 in [1]:
    # w * 2F1(a, b, c, z) + x * 2F1(a, b, c + 1, z) = y 2F1(a, b, c - 1, z)
    # Where, w, x and y are coefficients for the recurrence (as specified
    # in [1]).
    def hypergeometric_recurrence(
        should_stop,
        index,
        term,
        result,
        previous_result):
      c = term
      new_result = ((c * (c - 1 - (2 * c - a - b - 1.) * z) * result +
                     (c - a) * (c - b) * z * previous_result) / (
                         c * (c - 1) * (1. - z)))
      should_stop = index >= 2 - integer_d
      new_term = tf.where(should_stop, term, term - 1.)
      new_result = tf.where(should_stop, result, new_result)

      return should_stop, index + 1, new_term, new_result, result

    (_, _, _, result, _) = tf.while_loop(
        cond=lambda stop, *_: tf.reduce_any(~stop),
        body=hypergeometric_recurrence,
        loop_vars=(
            tf.zeros(broadcast_shape, dtype=tf.bool),
            tf.cast(0., dtype=dtype), e, second_result, first_result))
    return result


def _hyp2f1_large_a(a, b, c, z):
  """Compute 2F1(a, b, c, z) when |a| >> |c|."""
  # References
  # [1] M. Abramowitz, I. Stegun. Handbook of Mathematical Functions with
  #     Formulas, Graphs and Mathematical Tables.
  with tf.name_scope('hyp2f1_large_a'):
    dtype = dtype_util.common_dtype([a, b, c, z], tf.float32)
    a = tf.convert_to_tensor(a, dtype=dtype)
    b = tf.convert_to_tensor(b, dtype=dtype)
    c = tf.convert_to_tensor(c, dtype=dtype)
    z = tf.convert_to_tensor(z, dtype=dtype)
    numpy_dtype = dtype_util.as_numpy_dtype(dtype)

    # We assume that c < 0 and a, b > 0.

    # If a < c < 0 or a > c > 0, stop at round(a - c) steps
    num_iterations = tf.where(
        (tf.math.sign(a) == tf.math.sign(c)),
        tf.math.round(a - c),
        tf.math.round(a))

    broadcast_shape = functools.reduce(
        ps.broadcast_shape,
        [ps.shape(x) for x in [a, b, c]])

    # We use recurrence 15.2.10 in [1]:
    # w * 2F1(a, b, c, z) + x * 2F1(a + 1, b, c, z) = y 2F1(a - 1, b, c, z)
    # Where, w, x and y are coefficients for the recurrence (as specified
    # in [1]).
    t = a - num_iterations
    negative_recurrence_num_iterations = tf.where(
        num_iterations < 0, -num_iterations, numpy_dtype(0))
    first_result = _hyp2f1_small_parameters(t, b, c, z)
    second_result = _hyp2f1_small_parameters(
        t + tf.math.sign(num_iterations), b, c, z)
    def hypergeometric_negative_a_recurrence(
        should_stop,
        index,
        term,
        result,
        previous_result):
      t = term
      new_result = (-(2 * t - c + (b - t) * z) * result -
                    t * (z - 1.) * previous_result) / (c - t)
      should_stop = index >= negative_recurrence_num_iterations
      new_term = tf.where(should_stop, term, term - 1.)
      new_result = tf.where(should_stop, result, new_result)
      return should_stop, index + 1, new_term, new_result, result

    (_, _, _, negative_recurrence_result, _) = tf.while_loop(
        cond=lambda stop, *_: tf.reduce_any(~stop),
        body=hypergeometric_negative_a_recurrence,
        loop_vars=(
            tf.zeros(broadcast_shape, dtype=tf.bool),
            tf.cast(1., dtype=dtype),
            t - 1.,
            second_result,
            first_result))

    positive_recurrence_num_iterations = tf.where(
        num_iterations > 0, num_iterations, numpy_dtype(0))

    # We use recurrence 15.2.10 in [1]:
    # w * 2F1(a, b, c, z) + x * 2F1(a + 1, b, c, z) = y 2F1(a - 1, b, c, z)
    # Where, w, x and y are coefficients for the recurrence (as specified
    # in [1]).
    def hypergeometric_positive_a_recurrence(
        should_stop,
        index,
        term,
        result,
        previous_result):
      t = term
      new_result = -((2 * t - c + (b - t) * z) * result +
                     (c - t) * previous_result) / (t * (z - 1.))
      should_stop = index >= positive_recurrence_num_iterations
      new_term = tf.where(should_stop, term, term + 1.)
      new_result = tf.where(should_stop, result, new_result)
      return should_stop, index + 1, new_term, new_result, result

    (_, _, _, positive_recurrence_result, _) = tf.while_loop(
        cond=lambda stop, *_: tf.reduce_any(~stop),
        body=hypergeometric_positive_a_recurrence,
        loop_vars=(
            tf.zeros(broadcast_shape, dtype=tf.bool),
            tf.cast(1., dtype=dtype), t + 1., second_result, first_result))
    return tf.where(
        num_iterations > 0,
        positive_recurrence_result,
        negative_recurrence_result)


def _hyp2f1_internal(a, b, c, z):
  """Internal method for Hyp2F1 that decides what recurrence to use."""
  # Make safe for small parameters.
  dtype = dtype_util.common_dtype([a, b, c, z], tf.float32)
  numpy_dtype = dtype_util.as_numpy_dtype(dtype)
  a = tf.convert_to_tensor(a, dtype=dtype)
  b = tf.convert_to_tensor(b, dtype=dtype)
  c = tf.convert_to_tensor(c, dtype=dtype)
  z = tf.convert_to_tensor(z, dtype=dtype)

  a_larger = tf.where(tf.math.abs(a) > tf.math.abs(b), b, a)
  b = tf.where(tf.math.abs(a) > tf.math.abs(b), a, b)
  a = a_larger

  # If |a| >> |c|, use a recurrence to compute Hyp2f1(a, b, c, z) from
  # Hyp2f1(t, b, c, z), where |t| << |a|.
  recurrence_for_large_a = (
      (tf.math.abs(a) > tf.math.abs(c)) &
      (tf.math.abs(c - a) > 2) & (tf.math.abs(a) > 2))

  # If c is negative and large in magnitude, use a recurrence to
  # compute Hyp2f1(a, b, c, z) from Hyp2f1(a, b, t, z) where t > 0.
  recurrence_for_negative_c = (c < 0.) & (c < a + b) & (tf.math.abs(c) > 50.)

  unsafe_for_taylor_recurrence = recurrence_for_large_a | recurrence_for_negative_c

  small_a = tf.where(unsafe_for_taylor_recurrence, numpy_dtype(0.), a)
  small_b = tf.where(unsafe_for_taylor_recurrence, numpy_dtype(0.), b)
  small_c = tf.where(unsafe_for_taylor_recurrence, numpy_dtype(1.), c)
  result = _hyp2f1_small_parameters(small_a, small_b, small_c, z)

  # When c < 0, a, b > 0, use a recurrence to compute Hyp2f1(a, b, c, z). Note
  # that this will also reduce |a| as necessary (since it is likely |a| >> |t|).
  c_negative = tf.where(recurrence_for_negative_c, c, numpy_dtype(-1.))
  safe_a = tf.where(recurrence_for_negative_c, a, numpy_dtype(0.))
  safe_b = tf.where(recurrence_for_negative_c, b, numpy_dtype(0.))
  result = tf.where(
      recurrence_for_negative_c,
      _hyp2f1_large_negative_c(safe_a, safe_b, c_negative, z),
      result)

  # When |a| >> |c| use a recurrence to simplify.
  small_a = tf.where(recurrence_for_large_a, a, numpy_dtype(0.))
  result = tf.where(
      recurrence_for_large_a & ~recurrence_for_negative_c,
      _hyp2f1_large_a(small_a, b, c, z),
      result)

  return result


def _gamma_negative(z):
  """Returns whether Sign(Gamma(z)) == -1."""
  return (z < 0.) & tf.math.not_equal(
      tf.math.floormod(tf.math.floor(z), 2.), 0.)


def _hyp2f1_z_near_one(a, b, c, z):
  """"Compute 2F1(a, b, c, z) when z is near 1."""
  with tf.name_scope('hyp2f1_z_near_one'):
    dtype = dtype_util.common_dtype([a, b, c, z], tf.float32)
    a = tf.convert_to_tensor(a, dtype=dtype)
    b = tf.convert_to_tensor(b, dtype=dtype)
    c = tf.convert_to_tensor(c, dtype=dtype)
    z = tf.convert_to_tensor(z, dtype=dtype)

    # When z > 0.5, We can transform z to 1 - z and make use of a hypergeometric
    # identity.

    d = c - a - b

    # TODO(b/171982819): When tfp.math.log_gamma_difference and tfp.math.lbeta
    # support negative parameters, use them here for greater accuracy.
    log_first_coefficient = (tf.math.lgamma(c) + tf.math.lgamma(d) -
                             tf.math.lgamma(c - a) - tf.math.lgamma(c - b))

    sign_first_coefficient = (
        _gamma_negative(c) ^ _gamma_negative(d) ^
        _gamma_negative(c - a) ^ _gamma_negative(c - b))
    sign_first_coefficient = -2. * tf.cast(sign_first_coefficient, dtype) + 1.

    log_second_coefficient = (
        tf.math.xlog1py(d, -z) +
        tf.math.lgamma(c) + tf.math.lgamma(-d) -
        tf.math.lgamma(a) - tf.math.lgamma(b))

    sign_second_coefficient = (
        _gamma_negative(c) ^ _gamma_negative(a) ^ _gamma_negative(b) ^
        _gamma_negative(-d))
    sign_second_coefficient = -2. * tf.cast(sign_second_coefficient, dtype) + 1.

    first_term = _hyp2f1_internal(a, b, 1 - d, 1 - z)
    second_term = _hyp2f1_internal(c - a, c - b, d + 1., 1 - z)
    log_first_term = log_first_coefficient + tf.math.log(
        tf.math.abs(first_term))
    log_second_term = log_second_coefficient + tf.math.log(
        tf.math.abs(second_term))

    sign_first_term = sign_first_coefficient * tf.math.sign(first_term)
    sign_second_term = sign_second_coefficient * tf.math.sign(second_term)
    log_diff, sign_log_diff = tfp_math.log_sub_exp(
        log_first_term, log_second_term, return_sign=True)
    sign = tf.where(
        tf.math.equal(sign_first_term, sign_second_term),
        sign_first_term,
        sign_first_term * sign_log_diff)
    log_result = tf.where(
        tf.math.equal(sign_first_term, sign_second_term),
        tfp_math.log_add_exp(log_first_term, log_second_term),
        log_diff)
    return tf.math.exp(log_result) * sign


def _hyp2f1_z_near_negative_one(a, b, c, z):
  # 2F1(a, b, c, z) = (1 - z)**(-b) * 2F1(b, c - a, c, z / (z - 1))
  # When z < -0.5, we can transform z to z / (z - 1) and make use of a
  # hypergeometric identity.
  return tf.math.exp(
      tf.math.xlog1py(-b, -z)) * _hyp2f1_internal(b, c - a, c, z / (z - 1.))


def _is_negative_integer(x):
  return (x < 0.) & tf.math.equal(x, tf.math.floor(x))


def _mask_exceptional_arguments(a, b, c, z, dtype):
  """Masks parameters in exceptional cases for quick convergence."""
  # Ensure that exceptional cases are masked so that series computations
  # converge quickly.

  # Mask out z == 1., since this can result in a divergent seris for
  # certain values of a, b, c.
  safe_z = tf.where(tf.math.equal(z, 1.), dtype(0.), z)
  # When a == c (b == c), the series sums to (1 - z)**(-a) (-b respectively).
  safe_a = tf.where(tf.math.equal(a, c), dtype(0.), a)
  safe_b = tf.where(tf.math.equal(b, c), dtype(0.), b)

  # When c is a negative integer, the Taylor Series can blow up.
  # Ensure when c is a negative integer, that a or b are negative
  # integers that are larger to ensure convergence (since this will result in a
  # polynomial).
  safe_c = tf.where(
      _is_negative_integer(c),
      tf.where(
          ((_is_negative_integer(a) & (a > c)) |
           (_is_negative_integer(b) & (b > c))), c, a), c)

  return safe_a, safe_b, safe_c, safe_z


@tf.custom_gradient
def hyp2f1_small_argument(a, b, c, z, name=None):
  """Compute the Hypergeometric function 2f1(a, b, c, z) when |z| <= 1.

  Given `a, b, c` and `z`, compute Gauss' Hypergeometric Function, specified
  by the series:

  `1 + (a * b/c) * z + (a * (a + 1) * b * (b + 1) / ((c * (c + 1)) * z**2 / 2 +
  ... (a)_n * (b)_n / (c)_n * z ** n / n! + ....`


  NOTE: Gradients with only respect to `z` are available.
  NOTE: It is recommended that the arguments are `float64` due to the heavy
  loss of precision in float32.

  Args:
    a: Floating-point `Tensor`, broadcastable with `b, c, z`. Parameter for the
      numerator of the series fraction.
    b: Floating-point `Tensor`, broadcastable with `a, c, z`. Parameter for the
      numerator of the series fraction.
    c: Floating-point `Tensor`, broadcastable with `a, b, z`. Parameter for the
      denominator of the series fraction.
    z: Floating-point `Tensor`, broadcastable `a, b, c`. Value to compute
      `2F1(a, b, c, z)` at. Only values of `|z| < 1` are allowed.
    name: A name for the operation (optional).
      Default value: `None` (i.e., 'continued_fraction').

  Returns:
    hypergeo: `2F1(a, b, c, z)`


  #### References

  [1] F. Johansson. Computing hypergeometric functions rigorously.
     ACM Transactions on Mathematical Software, August 2019.
     https://arxiv.org/abs/1606.06977
  [2] J. Pearson, S. Olver, M. Porter. Numerical methods for the computation of
     the confluent and Gauss hypergeometric functions.
     Numerical Algorithms, August 2016.
  [3] M. Abramowitz, I. Stegun. Handbook of Mathematical Functions with
     Formulas, Graphs and Mathematical Tables.
  """
  with tf.name_scope(name or 'hyp2f1_small_argument'):
    dtype = dtype_util.common_dtype([a, b, c, z], tf.float32)
    numpy_dtype = dtype_util.as_numpy_dtype(dtype)
    a = tf.convert_to_tensor(a, dtype=dtype)
    b = tf.convert_to_tensor(b, dtype=dtype)
    c = tf.convert_to_tensor(c, dtype=dtype)
    z = tf.convert_to_tensor(z, dtype=dtype)

    # Mask out exceptional cases to ensure that the series transformations
    # terminate fast.
    safe_a, safe_b, safe_c, safe_z = _mask_exceptional_arguments(
        a, b, c, z, numpy_dtype)

    # TODO(b/128632717): Extend this by including transformations for:
    # * Large parameter ranges. Specifically use Hypergeometric recurrences
    # to decrease the parameter values. This should be done via backward
    # recurrences rather than forward recurrences since those are numerically
    # stable.
    # * Include |z| > 1. This can be done via Hypergeometric identities that
    # transform to |z| < 1.
    # * Handling exceptional cases where parameters are negative integers.

    # Assume that |b| > |a|. Swapping the two makes no effect on the
    # calculation.
    a_small = tf.where(
        tf.math.abs(safe_a) > tf.math.abs(safe_b), safe_b, safe_a)
    safe_b = tf.where(tf.math.abs(safe_a) > tf.math.abs(safe_b), safe_a, safe_b)
    safe_a = a_small

    d = safe_c - safe_a - safe_b

    # Use the identity
    # 2F1(a , b, c, z) = (1 - z) ** d * 2F1(c - a, c - b, c, z).
    # when the numerator coefficients become smaller.

    should_use_linear_transform = (
        (tf.math.abs(c - a) < tf.math.abs(a)) &
        (tf.math.abs(c - b) < tf.math.abs(b)))
    safe_a = tf.where(should_use_linear_transform, c - a, a)
    safe_b = tf.where(should_use_linear_transform, c - b, b)

    # When -0.5 < z < 0.9, use approximations to Taylor Series.
    safe_z_small = tf.where(
        (safe_z >= 0.9) | (safe_z <= -0.5), numpy_dtype(0.), safe_z)
    taylor_series = _hyp2f1_internal(safe_a, safe_b, safe_c, safe_z_small)

    # When z >= 0.9 or -0.5 > z, we use hypergeometric identities to ensure
    # that |z| is small.
    safe_positive_z_large = tf.where(safe_z >= 0.9, safe_z, numpy_dtype(1.))
    hyp2f1_z_near_one = _hyp2f1_z_near_one(
        safe_a, safe_b, safe_c, safe_positive_z_large)

    safe_negative_z_large = tf.where(safe_z <= -0.5, safe_z, numpy_dtype(-1.))
    hyp2f1_z_near_negative_one = _hyp2f1_z_near_negative_one(
        safe_a, safe_b, safe_c, safe_negative_z_large)

    result = tf.where(
        safe_z >= 0.9, hyp2f1_z_near_one,
        tf.where(safe_z <= -0.5, hyp2f1_z_near_negative_one, taylor_series))

    # Now if we applied the linear transformation identity, we need to
    # add a term (1 - z) ** (c - a - b)
    result = tf.where(
        should_use_linear_transform,
        tf.math.exp(d * tf.math.log1p(-safe_z)) * result,
        result)

    # Finally handle the exceptional cases.
    # First when z == 1., this expression diverges if c <= a + b, and otherwise
    # converges.
    hyp2f1_at_one = tf.math.exp(
        tf.math.lgamma(c) + tf.math.lgamma(c - a - b) -
        tf.math.lgamma(c - a) - tf.math.lgamma(c - b))
    sign_hyp2f1_at_one = (
        _gamma_negative(c) ^ _gamma_negative(c - a - b) ^
        _gamma_negative(c - a) ^ _gamma_negative(c - b))
    sign_hyp2f1_at_one = -2. * tf.cast(sign_hyp2f1_at_one, dtype) + 1.
    hyp2f1_at_one = hyp2f1_at_one * sign_hyp2f1_at_one

    result = tf.where(
        tf.math.equal(z, 1.),
        tf.where(c > a + b,
                 hyp2f1_at_one, numpy_dtype(np.nan)),
        result)

    # When a == c or b == c this reduces to (1 - z)**-b (-a respectively).
    result = tf.where(
        tf.math.equal(a, c),
        tf.math.exp(-b * tf.math.log1p(-z)),
        tf.where(
            tf.math.equal(b, c),
            tf.math.exp(-a * tf.math.log1p(-z)), result))

    # When c is a negative integer we can get a divergent series.
    result = tf.where(
        (_is_negative_integer(c) &
         ((a < c) | ~_is_negative_integer(a)) &
         ((b < c) | ~_is_negative_integer(b))),
        numpy_dtype(np.inf),
        result)

    def grad(dy):
      grad_z = a * b  * dy * hyp2f1_small_argument(
          a + 1., b + 1., c + 1., z) / c
      # We don't have an easily computable gradient with respect to parameters,
      # so ignore that for now.
      broadcast_shape = functools.reduce(
          ps.broadcast_shape,
          [ps.shape(x) for x in [a, b, c]])

      _, grad_z = tfp_math.fix_gradient_for_broadcasting(
          [tf.ones(broadcast_shape, dtype=z.dtype), z],
          [tf.ones_like(grad_z), grad_z])
      return None, None, None, grad_z

    return result, grad
