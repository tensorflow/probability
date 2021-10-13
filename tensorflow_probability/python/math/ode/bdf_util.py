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
"""Utilities for Backward differentiation formula (BDF) solver."""

import collections
import numpy as np
import tensorflow.compat.v1 as tf1
import tensorflow.compat.v2 as tf
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import prefer_static as ps
from tensorflow_probability.python.internal import tensorshape_util

MAX_ORDER = 5
ORDERS = np.arange(0, MAX_ORDER + 1)
RECIPROCAL_SUMS = np.concatenate([[np.nan], np.cumsum(1. / ORDERS[1:])])


def error_ratio(backward_difference, error_coefficient, tol):
  """Computes the ratio of the error in the computed state to the tolerance."""
  tol_cast = tf.cast(tol, backward_difference.dtype)
  error_ratio_ = tf.norm(error_coefficient * backward_difference / tol_cast)
  return tf.cast(error_ratio_, tf.abs(backward_difference).dtype)


def first_step_size(
    atol,
    first_order_error_coefficient,
    initial_state_vec,
    initial_time,
    ode_fn_vec,
    rtol,
    safety_factor,
    epsilon=1e-12,
    max_step_size=1.,
    min_step_size=1e-12,
):
  """Selects the first step size to use."""
  next_time = initial_time + epsilon
  first_derivative = ode_fn_vec(initial_time, initial_state_vec)
  state_dtype = initial_state_vec.dtype
  next_state_vec = initial_state_vec + first_derivative * epsilon
  second_derivative = (ode_fn_vec(next_time, next_state_vec) -
                       first_derivative) / epsilon
  tol = tf.cast(atol + rtol * tf.abs(initial_state_vec), state_dtype)
  # Local truncation error of an order one step is
  # `err(step_size) = first_order_error_coefficient * second_derivative *
  #                 * step_size**2`.
  # Choose the largest `step_size` such that `norm(err(step_size) / tol) <= 1`.
  norm = tf.norm(first_order_error_coefficient * second_derivative / tol)
  step_size = tf.cast(tf.math.rsqrt(norm), tf.abs(initial_state_vec).dtype)
  return tf.clip_by_value(safety_factor * step_size, min_step_size,
                          max_step_size)


def interpolate_backward_differences(backward_differences, order,
                                     step_size_ratio):
  """Updates backward differences when a change in the step size occurs."""
  state_dtype = backward_differences.dtype
  interpolation_matrix_ = interpolation_matrix(state_dtype, order,
                                               step_size_ratio)
  interpolation_matrix_unit_step_size_ratio = interpolation_matrix(
      state_dtype, order, 1.)
  interpolated_backward_differences_orders_one_to_five = tf.matmul(
      interpolation_matrix_unit_step_size_ratio,
      tf.matmul(interpolation_matrix_, backward_differences[1:MAX_ORDER + 1]))
  interpolated_backward_differences = tf.concat([
      tf.gather(backward_differences, [0]),
      interpolated_backward_differences_orders_one_to_five,
      ps.zeros(
          ps.stack([2, ps.shape(backward_differences)[1]]), dtype=state_dtype),
  ], 0)
  return interpolated_backward_differences


def interpolation_matrix(dtype, order, step_size_ratio):
  """Creates the matrix used to interpolate backward differences."""
  orders = tf.cast(tf.range(1, MAX_ORDER + 1), dtype=dtype)
  i = orders[:, tf.newaxis]
  j = orders[tf.newaxis, :]
  # Matrix whose (i, j)-th entry (`1 <= i, j <= order`) is
  # `1/j! (0 - i * step_size_ratio) * ... * ((j-1) - i * step_size_ratio)`.
  step_size_ratio_cast = tf.cast(step_size_ratio, dtype)
  full_interpolation_matrix = tf.math.cumprod(
      ((j - 1) - i * step_size_ratio_cast) / j, axis=1)
  zeros_matrix = tf.zeros_like(full_interpolation_matrix)
  interpolation_matrix_ = tf1.where(
      tf.range(1, MAX_ORDER + 1) <= order,
      tf.transpose(
          tf1.where(
              tf.range(1, MAX_ORDER + 1) <= order,
              tf.transpose(full_interpolation_matrix), zeros_matrix)),
      zeros_matrix)
  return interpolation_matrix_


def newton(backward_differences, max_num_iters, newton_coefficient, ode_fn_vec,
           order, step_size, time, tol, unitary, upper):
  """Runs Newton's method to solve the BDF equation."""
  initial_guess = tf.reduce_sum(
      tf1.where(
          tf.range(MAX_ORDER + 1) <= order,
          backward_differences[:MAX_ORDER + 1],
          tf.zeros_like(backward_differences)[:MAX_ORDER + 1]),
      axis=0)

  np_dtype = np_dtype = dtype_util.as_numpy_dtype(backward_differences.dtype)

  rhs_constant_term = newton_coefficient * tf.reduce_sum(
      tf1.where(
          tf.range(1, MAX_ORDER + 1) <= order,
          RECIPROCAL_SUMS[1:, np.newaxis].astype(np_dtype) *
          backward_differences[1:MAX_ORDER + 1],
          tf.zeros_like(backward_differences)[1:MAX_ORDER + 1]),
      axis=0)

  next_time = time + step_size
  step_size_cast = tf.cast(step_size, backward_differences.dtype)
  real_dtype = tf.abs(backward_differences).dtype

  def newton_body(iterand):
    """Performs one iteration of Newton's method."""
    next_backward_difference = iterand.next_backward_difference
    next_state_vec = iterand.next_state_vec

    rhs = newton_coefficient * step_size_cast * ode_fn_vec(
        next_time,
        next_state_vec) - rhs_constant_term - next_backward_difference
    delta = tf.squeeze(
        tf.linalg.triangular_solve(
            upper,
            tf.matmul(tf.transpose(unitary), rhs[:, tf.newaxis]),
            lower=False))
    num_iters = iterand.num_iters + 1

    next_backward_difference += delta
    next_state_vec += delta

    delta_norm = tf.cast(tf.norm(delta), real_dtype)
    lipschitz_const = delta_norm / iterand.prev_delta_norm

    # Stop if method has converged.
    approx_dist_to_sol = lipschitz_const / (1. - lipschitz_const) * delta_norm
    close_to_sol = approx_dist_to_sol < tol
    delta_norm_is_zero = tf.equal(delta_norm, tf.constant(0., dtype=real_dtype))
    converged = close_to_sol | delta_norm_is_zero
    finished = converged

    # Stop if any of the following conditions are met:
    # (A) We have hit the maximum number of iterations.
    # (B) The method is converging too slowly.
    # (C) The method is not expected to converge.
    too_slow = lipschitz_const > 1.
    finished = finished | too_slow
    if max_num_iters is not None:
      too_many_iters = tf.equal(num_iters, max_num_iters)
      num_iters_left = max_num_iters - num_iters
      num_iters_left_cast = tf.cast(num_iters_left, real_dtype)
      wont_converge = (
          approx_dist_to_sol * lipschitz_const**num_iters_left_cast > tol)
      finished = finished | too_many_iters | wont_converge

    return [
        _NewtonIterand(
            converged=converged,
            finished=finished,
            next_backward_difference=next_backward_difference,
            next_state_vec=next_state_vec,
            num_iters=num_iters,
            prev_delta_norm=delta_norm)
    ]

  iterand = _NewtonIterand(
      converged=False,
      finished=False,
      next_backward_difference=tf.zeros_like(initial_guess),
      next_state_vec=tf.identity(initial_guess),
      num_iters=0,
      prev_delta_norm=tf.constant(np.array(-0.), dtype=real_dtype))
  [iterand] = tf.while_loop(lambda iterand: tf.logical_not(iterand.finished),
                            newton_body, [iterand])
  return (iterand.converged, iterand.next_backward_difference,
          iterand.next_state_vec, iterand.num_iters)


_NewtonIterand = collections.namedtuple('NewtonIterand', [
    'converged',
    'finished',
    'next_backward_difference',
    'next_state_vec',
    'num_iters',
    'prev_delta_norm',
])


def newton_qr(jacobian_mat, newton_coefficient, step_size):
  """QR factorizes the matrix used in each iteration of Newton's method."""
  identity = tf.eye(ps.shape(jacobian_mat)[0], dtype=jacobian_mat.dtype)
  step_size_cast = tf.cast(step_size, jacobian_mat.dtype)
  newton_matrix = (
      identity - step_size_cast * newton_coefficient * jacobian_mat)
  factorization = tf.linalg.qr(newton_matrix)
  return factorization.q, factorization.r


def update_backward_differences(backward_differences, next_backward_difference,
                                next_state_vec, order):
  """Returns the backward differences for the next time."""
  backward_differences_array = tf.TensorArray(
      backward_differences.dtype,
      size=MAX_ORDER + 3,
      clear_after_read=False,
      element_shape=next_backward_difference.shape).unstack(
          backward_differences)
  new_backward_differences_array = tf.TensorArray(
      backward_differences.dtype,
      size=MAX_ORDER + 3,
      clear_after_read=False,
      element_shape=next_backward_difference.shape)
  new_backward_differences_array = new_backward_differences_array.write(
      order + 2,
      next_backward_difference - backward_differences_array.read(order + 1))
  new_backward_differences_array = new_backward_differences_array.write(
      order + 1, next_backward_difference)

  def body(k, new_backward_differences_array_):
    new_backward_differences_array_k = (
        new_backward_differences_array_.read(k + 1) +
        backward_differences_array.read(k))
    new_backward_differences_array_ = new_backward_differences_array_.write(
        k, new_backward_differences_array_k)
    return k - 1, new_backward_differences_array_

  _, new_backward_differences_array = tf.while_loop(
      lambda k, new_backward_differences_array: k > 0, body,
      [order, new_backward_differences_array])
  new_backward_differences_array = new_backward_differences_array.write(
      0, next_state_vec)
  new_backward_differences = new_backward_differences_array.stack()
  tensorshape_util.set_shape(new_backward_differences,
                             tf.TensorShape([MAX_ORDER + 3, None]))
  return new_backward_differences
