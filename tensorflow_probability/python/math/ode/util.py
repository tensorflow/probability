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
"""Utilities for TensorFlow Probability ODE solvers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v2 as tf


def assert_increasing(tensor, identifier):
  """Assert if a `Tensor` is strictly increasing."""
  return tf.Assert(
      tf.reduce_all(tensor[1:] > tensor[:-1]),
      ['`{}` must be strictly increasing.'.format(identifier)])


def assert_nonnegative(tensor, identifier):
  """Assert if a `Tensor` is nonnegative."""
  return tf.Assert(
      tf.reduce_all(tensor >= tf.zeros([], dtype=tensor.dtype)),
      ['`{}` must be nonnegative'.format(identifier)])


def assert_positive(tensor, identifier):
  """Assert if a `Tensor` is positive."""
  return tf.Assert(
      tf.reduce_all(tensor > tf.zeros([], dtype=tensor.dtype)),
      ['`{}` must be positive.'.format(identifier)])


def error_if_not_real_or_complex(tensor, identifier):
  """Raise a `TypeError` if the `Tensor` is neither real nor complex."""
  if not (tensor.dtype.is_floating or tensor.dtype.is_complex):
    raise TypeError(
        '`{}` must have a floating point or complex floating point dtype.'
        .format(identifier))


def error_if_not_vector(tensor, identifier):
  """Raise a `ValueError` if the `Tensor` is not 1-D."""
  if len(tensor.get_shape().as_list()) != 1:
    raise ValueError('`{}` must be a 1-D tensor.'.format(identifier))


def get_jacobian_fn_mat(jacobian_fn, ode_fn_vec, state_shape, use_pfor):
  """Returns a wrapper around the user-specified `jacobian_fn` argument.

  `jacobian_fn` is an optional argument that can either be a constant `Tensor`
  or a function of the form `jacobian_fn(time, state)`. This function returns a
  wrapper `jacobian_fn_mat(time, state_vec)` whose second argument and output
  are 1 and 2-D `Tensor`s, respectively, corresponding reshaped versions of
  `state` and `jacobian_fn(time, state)`.

  Args:
    jacobian_fn: User-specified `jacobian_fn` passed to `solve`.
    ode_fn_vec: User-specified `ode_fn` passed to `solve`.
    state_shape: The shape of the second argument and output of `ode_fn`.
    use_pfor: User-specified `use_pfor` passed to `solve`.

  Returns:
    The wrapper described above.
  """
  if jacobian_fn is None:

    def automatic_jacobian_fn_mat(time, state):
      with tf.GradientTape(
          watch_accessed_variables=False, persistent=not use_pfor) as tape:
        tape.watch(state)
        outputs = ode_fn_vec(time, state)
      jacobian = tape.jacobian(outputs, state, experimental_use_pfor=use_pfor)
      if jacobian is None:
        return tf.zeros([tf.size(state)] * 2, dtype=state.dtype)
      return jacobian

    return automatic_jacobian_fn_mat

  if not callable(jacobian_fn):
    jacobian_const = tf.convert_to_tensor(jacobian_fn)

    def constant_jacobian_fn_mat(_, state):
      return tf.reshape(jacobian_const, [-1, tf.size(state)])

    return constant_jacobian_fn_mat

  def jacobian_fn_mat(time, state):
    state = tf.reshape(state, state_shape)
    jacobian = tf.reshape(jacobian_fn(time, state), [-1, tf.size(state)])
    return jacobian

  return jacobian_fn_mat


def get_ode_fn_vec(ode_fn, state_shape):
  """Returns a wrapper around the user-specified `ode_fn` argument.

  The second argument and output of `ode_fn(time, state)` are N-D `Tensor`s.
  This function returns a wrapper `ode_fn_vec(time, state_vec)` whose
  second argument and output are 1-D `Tensor`s corresponding to reshaped
  versions of `state` and `ode_fn(time, state)`.

  Args:
    ode_fn: User-specified `ode_fn` passed to `solve`.
    state_shape: The shape of the second argument and output of `ode_fn`.

  Returns:
    The wrapper described above.
  """

  def ode_fn_vec(time, unrolled_state):
    return tf.reshape(
        ode_fn(time, tf.reshape(unrolled_state, state_shape)), [-1])

  return ode_fn_vec


def next_step_size(step_size, order, error_ratio, safety_factor,
                   min_step_size_factor, max_step_size_factor):
  """Computes the next step size to use.

  Computes the next step size by applying a multiplicative factor to the current
  step size. This factor is
  ```none
  factor_unclamped = error_ratio**(-1. / (order + 1)) * safety_factor
  factor = clamp(factor_unclamped, min_step_size_factor, max_step_size_factor)
  ```

  Args:
    step_size: Scalar float `Tensor` specifying the current step size.
    order: Scalar integer `Tensor` specifying the order of the method.
    error_ratio: Scalar float `Tensor` specifying the ratio of the error in the
      computed state and the tolerance.
    safety_factor: Scalar float `Tensor`.
    min_step_size_factor: Scalar float `Tensor` specifying a lower bound on the
      multiplicative factor.
    max_step_size_factor: Scalar float `Tensor` specifying an upper bound on the
      multiplicative factor.

  Returns:
    Scalar float `Tensor` specifying the next step size.
  """
  order_cast = tf.cast(order, error_ratio.dtype)
  factor = error_ratio**(-1. / (order_cast + 1.))
  return step_size * tf.clip_by_value(
      safety_factor * factor, min_step_size_factor, max_step_size_factor)
