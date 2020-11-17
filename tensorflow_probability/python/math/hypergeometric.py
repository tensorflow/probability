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

from __future__ import absolute_import
from __future__ import division
# [internal] enable type annotations
from __future__ import print_function

import functools

# Dependency imports
import numpy as np
import tensorflow.compat.v2 as tf
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import prefer_static as ps
from tensorflow_probability.python.internal import tensorshape_util


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
          should_stop,
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

      # Rescale to prevent overflow.
      should_rescale = ((tf.math.abs(new_numerator_term0) > 10.) |
                        (tf.math.abs(new_numerator_term1) > 10.) |
                        (tf.math.abs(new_denominator) > 10.))
      new_numerator_term0 = tf.where(
          should_rescale, new_numerator_term0 / 10., new_numerator_term0)
      new_numerator_term1 = tf.where(
          should_rescale, new_numerator_term1 / 10., new_numerator_term1)
      new_denominator = tf.where(
          should_rescale, new_denominator / 10., new_denominator)

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
          should_stop | (index > 50.),
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
  safe_c = tf.where(tf.math.abs(c) < 1., c, 0.)
  safe_a = tf.where(tf.math.abs(c) < 1., a, 0.)
  safe_b = tf.where(tf.math.abs(c) < 1., b, 0.)
  result = _hyp2f1_fraction(safe_a, safe_b, safe_c, z)
  safe_c = tf.where(
      tf.math.abs(c) < 1., tf.math.abs(a) + tf.math.abs(b), c)
  result = tf.where(
      tf.math.abs(c) < 1.,
      result,
      _hyp2f1_taylor_series(a, b, safe_c, z))
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

    # When z > 0.9, We can transform z to 1 - z and make use of a hypergeometric
    # identity.

    # TODO(b/171982819): When tfp.math.log_gamma_difference and tfp.math.lbeta
    # support negative parameters, use them here for greater accuracy.
    log_first_coefficient = (tf.math.lgamma(c) + tf.math.lgamma(c - a - b) -
                             tf.math.lgamma(c - a) - tf.math.lgamma(c - b))

    sign_first_coefficient = (
        _gamma_negative(c) ^ _gamma_negative(c - a - b) ^
        _gamma_negative(c - a) ^ _gamma_negative(c - b))
    sign_first_coefficient = -2. * tf.cast(sign_first_coefficient, dtype) + 1.

    log_second_coefficient = (
        tf.math.xlog1py(c - a - b, -z) +
        tf.math.lgamma(c) + tf.math.lgamma(a + b - c) -
        tf.math.lgamma(a) - tf.math.lgamma(b))

    sign_second_coefficient = (
        _gamma_negative(c) ^ _gamma_negative(a) ^ _gamma_negative(b) ^
        _gamma_negative(a + b - c))
    sign_second_coefficient = -2. * tf.cast(sign_second_coefficient, dtype) + 1.

    safe_a = tf.where(c > 1., b - c + 1, a)
    safe_b = tf.where(c > 1., a - c + 1, b)
    first_term = _hyp2f1_small_parameters(safe_a, safe_b, a + b - c + 1., 1 - z)
    first_term = tf.where(
        c > 1.,
        tf.math.exp(tf.math.xlogy(1. - c, z)) * first_term,
        first_term)

    safe_a = tf.where(c > 1., 1. - b, c - a)
    safe_b = tf.where(c > 1., 1. - a, c - b)
    second_term = _hyp2f1_small_parameters(
        safe_a, safe_b, c - a - b + 1., 1 - z)
    second_term = tf.where(
        c > 1.,
        tf.math.exp(tf.math.xlogy(1. - c, z)) * second_term,
        second_term)

    result = (sign_first_coefficient * tf.math.exp(log_first_coefficient) *
              first_term +
              sign_second_coefficient * tf.math.exp(log_second_coefficient) *
              second_term)
    result = tf.where(
        c > 1.,
        tf.math.exp(tf.math.xlogy(1. - c, z)) * result,
        result)
    return result


def _hyp2f1_z_near_negative_one(a, b, c, z):
  # 2F1(a, b, c, z) = (1 - z)**(-b) * 2F1(b, c - a, c, z / (z - 1))
  # When z < -0.9, we can transform z to z / (z - 1) and make use of a
  # hypergeometric identity.
  return tf.math.exp(
      tf.math.xlog1py(-b, -z)) * _hyp2f1_small_parameters(
          b, c - a, c, z / (z - 1.))


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
  """
  with tf.name_scope(name or 'hyp2f1_small_argument'):
    dtype = dtype_util.common_dtype([a, b, c, z], tf.float32)
    numpy_dtype = dtype_util.as_numpy_dtype(dtype)
    a = tf.convert_to_tensor(a, dtype=dtype)
    b = tf.convert_to_tensor(b, dtype=dtype)
    c = tf.convert_to_tensor(c, dtype=dtype)
    z = tf.convert_to_tensor(z, dtype=dtype)

    # TODO(b/128632717): Extend this by including transformations for:
    # * Large parameter ranges. Specifically use Hypergeometric recurrences
    # to decrease the parameter values.
    # * Include |z| > 1. This can be done via Hypergeometric identities that
    # transform to |z| < 1.
    # * Handling exceptional cases where parameters are negative integers.

    # Assume that |b| > |a|. Swapping the two makes no effect on the
    # calculation.
    a_small = tf.where(tf.math.abs(a) > tf.math.abs(b), b, a)
    b = tf.where(tf.math.abs(a) > tf.math.abs(b), a, b)
    a = a_small

    safe_a = tf.where(c < a + b, c - a, a)
    safe_b = tf.where(c < a + b, c - b, b)

    # When |z| < 0.9, use approximations to Taylor Series.
    safe_z_small = tf.where(tf.math.abs(z) > 0.9, numpy_dtype(0.), z)
    taylor_series = _hyp2f1_small_parameters(safe_a, safe_b, c, safe_z_small)
    taylor_series = tf.where(
        c < a + b,
        tf.math.exp((c - a - b) * tf.math.log1p(-z)) * taylor_series,
        taylor_series)

    # When |z| >= 0.9, we use hypergeometric identities to ensure that |z| is
    # small.
    safe_positive_z_large = tf.where(z >= 0.9, z, numpy_dtype(1.))
    hyp2f1_z_near_one = _hyp2f1_z_near_one(a, b, c, safe_positive_z_large)

    safe_negative_z_large = tf.where(z <= -0.9, z, numpy_dtype(-1.))
    hyp2f1_z_near_negative_one = _hyp2f1_z_near_negative_one(
        a, b, c, safe_negative_z_large)

    result = tf.where(
        z >= 0.9, hyp2f1_z_near_one,
        tf.where(z <= -0.9, hyp2f1_z_near_negative_one, taylor_series))

    def grad(dy):
      grad_z = a * b  * dy * hyp2f1_small_argument(
          a + 1., b + 1., c + 1., z) / c
      # We don't have an easily computable gradient with respect to parameters,
      # so ignore that for now.
      broadcast_shape = functools.reduce(
          ps.broadcast_shape,
          [ps.shape(x) for x in [a, b, c]])

      _, grad_z = _fix_gradient_for_broadcasting(
          tf.ones(broadcast_shape, dtype=z.dtype),
          z, tf.ones_like(grad_z), grad_z)
      return None, None, None, grad_z

    return result, grad


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
