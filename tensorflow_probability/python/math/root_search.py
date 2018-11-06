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
"""Methods for finding roots of functions of one variable."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

import tensorflow as tf

__all__ = [
    'secant_root',
]

RootSearchResults = collections.namedtuple(
    'RootSearchResults',
    [
        # A tensor containing the last position explored. If the search was
        # successful, this position is a root of the objective function.
        'estimated_root',
        # A tensor containing the value of the objective function at the last
        # position explored. If the search was successful, then this is close
        # to 0.
        'objective_at_estimated_root',
        # The number of iterations performed.
        'num_iterations',
    ])


def secant_root(objective_fn,
                initial_position,
                next_position=None,
                value_at_position=None,
                position_tolerance=1e-8,
                value_tolerance=1e-8,
                max_iterations=50,
                stopping_policy_fn=tf.reduce_all,
                validate_args=False,
                name=None):
  r"""Finds root(s) of a function of single variable using the secant method.

  The [secant method](https://en.wikipedia.org/wiki/Secant_method) is a
  root-finding algorithm that uses a succession of roots of secant lines to
  better approximate a root of a function. The secant method can be thought of
  as a finite-difference approximation of Newton's method.

  Args:
    objective_fn: Python callable for which roots are searched. It must be a
      callable of a single variable. `objective_fn` must return a `Tensor` of
      the same shape and dtype as `initial_position`.
    initial_position: `Tensor` or Python float representing the starting
      position. The callable will search for roots in the neighborhood of each
      point. The shape of `initial_position` should match that of the input to
      `objective_fn`.
    next_position: Optional `Tensor` representing the next position in the
      search. If specified, this argument must broadcast with the shape of
      `initial_position` and have the same dtype. It will be used to compute the
      first step to take when searching for roots. If not specified, a default
      value will be used instead.
      Default value: `initial_position * (1 + 1e-4) + sign(initial_position) *
        1e-4`.
    value_at_position: Optional `Tensor` or Pyhon float representing the value
      of `objective_fn` at `initial_position`. If specified, this argument must
      have the same shape and dtype as initial_position. If not specified, the
      value will be evaluated during the search.
      Default value: None.
    position_tolerance: Optional `Tensor` representing the tolerance for the
      estimated roots. If specified, this argument must broadcast with the shape
      of `initial_position` and have the same dtype.
      Default value: `1e-8`.
    value_tolerance: Optional `Tensor` representing the tolerance used to check
      for roots. If the absolute value of `objective_fn` is smaller than
      `value_tolerance` at a given position, then that position is considered a
      root for the function. If specified, this argument must broadcast with the
      shape of `initial_position` and have the same dtype.
      Default value: `1e-8`.
    max_iterations: Optional `Tensor` or Python integer specifying the maximum
      number of steps to perform for each initial position. Must broadcast with
      the shape of `initial_position`.
      Default value: `50`.
    stopping_policy_fn: Python `callable` controlling the algorithm termination.
      It must be a callable accepting a `Tensor` of booleans with the shape of
      `initial_position` (each denoting whether the search is finished for each
      starting point), and returning a scalar boolean `Tensor` (indicating
      whether the overall search should stop). Typical values are
      `tf.reduce_all` (which returns only when the search is finished for all
      points), and `tf.reduce_any` (which returns as soon as the search is
      finished for any point).
      Default value: `tf.reduce_all` (returns only when the search is finished
        for all points).
    validate_args: Python `bool` indicating whether to validate arguments such
      as `position_tolerance`, `value_tolerance`, and `max_iterations`.
      Default value: `False`.
    name: Python `str` name prefixed to ops created by this function.

  Returns:
    root_search_results: A Python `namedtuple` containing the following items:
      estimated_root: `Tensor` containing the last position explored. If the
        search was successful within the specified tolerance, this position is
        a root of the objective function.
      objective_at_estimated_root: `Tensor` containing the value of the
        objective function at `position`. If the search was successful within
        the specified tolerance, then this is close to 0.
      num_iterations: The number of iterations performed.

  Raises:
    ValueError: if a non-callable `stopping_policy` is passed.

  #### Examples

  ```python
  import tensorflow as tf
  import tensorflow_probability as tfp
  tf.enable_eager_execution()

  # Example 1: Roots of a single function from two different starting points.

  f = lambda x: (63 * x**5 - 70 * x**3 + 15 * x) / 8.
  x = tf.constant([-1, 10], dtype=tf.float64)

  tfp.math.secant_root(objective_fn=f, initial_position=x))
  # ==> RootSearchResults(
      estimated_root=array([-0.90617985, 0.90617985]),
      objective_at_estimated_root=array([ -4.81727769e-10, 7.44957651e-10]),
      num_iterations=array([ 7, 24], dtype=int32))

  tfp.math.secant_root(objective_fn=f,
                       initial_position=x,
                       stopping_policy_fn=tf.reduce_any)
  # ==> RootSearchResults(
      estimated_root=array([-0.90617985, 3.27379206]),
      objective_at_estimated_root=array([ -4.81727769e-10, 2.66058312e+03]),
      num_iterations=array([7, 8], dtype=int32))

  # Example 2: Roots of a multiplex function from a single starting point.

  def f(x):
    return tf.constant([0., 63. / 8], dtype=tf.float64) * x**5 \
        + tf.constant([5. / 2, -70. / 8], dtype=tf.float64) * x**3 \
        + tf.constant([-3. / 2, 15. / 8], dtype=tf.float64) * x

  x = tf.constant([-1, -1], dtype=tf.float64)

  tfp.math.secant_root(objective_fn=f, initial_position=x)
  # ==> RootSearchResults(
      estimated_root=array([-0.77459667, -0.90617985]),
      objective_at_estimated_root=array([ -7.81339438e-11, -4.81727769e-10]),
      num_iterations=array([7, 7], dtype=int32))

  # Example 3: Roots of a multiplex function from two starting points.

  def f(x):
    return tf.constant([0., 63. / 8], dtype=tf.float64) * x**5 \
        + tf.constant([5. / 2, -70. / 8], dtype=tf.float64) * x**3 \
        + tf.constant([-3. / 2, 15. / 8], dtype=tf.float64) * x

  x = tf.constant([[-1, -1], [10, 10]], dtype=tf.float64)

  tfp.math.secant_root(objective_fn=f, initial_position=x)
  # ==> RootSearchResults(
      estimated_root=array([
          [-0.77459667, -0.90617985],
          [ 0.77459667,  0.90617985]]),
      objective_at_estimated_root=array([
          [ -7.81339438e-11, -4.81727769e-10],
          [  6.66025013e-11,  7.44957651e-10]]),
      num_iterations=array([
          [ 7,  7],
          [16, 24]], dtype=int32))
  ```
  """
  if not callable(stopping_policy_fn):
    raise ValueError('stopping_policy must be callable')

  position = tf.convert_to_tensor(
      initial_position,
      name='position',
  )
  value_at_position = tf.convert_to_tensor(
      value_at_position or objective_fn(position),
      name='value_at_position',
      dtype=position.dtype.base_dtype)

  zero = tf.zeros_like(position)
  position_tolerance = tf.convert_to_tensor(
      position_tolerance, name='position_tolerance', dtype=position.dtype)
  value_tolerance = tf.convert_to_tensor(
      value_tolerance, name='value_tolerance', dtype=position.dtype)

  num_iterations = tf.zeros_like(position, dtype=tf.int32)
  max_iterations = tf.convert_to_tensor(max_iterations, dtype=tf.int32)
  max_iterations = tf.broadcast_to(
      max_iterations, name='max_iterations', shape=position.shape)

  # Compute the step from `next_position` if present. This covers the case where
  # a user has two starting points, which bound the root or has a specific step
  # size in mind.
  if next_position is None:
    epsilon = tf.constant(1e-4, dtype=position.dtype, shape=position.shape)
    step = position * epsilon + tf.sign(position) * epsilon
  else:
    step = next_position - initial_position

  finished = tf.constant(False, shape=position.shape)

  # Negate `stopping_condition` to determine if the search should continue.
  # This means, in particular, that tf.reduce_*all* will return only when the
  # search is finished for *all* starting points.
  def _should_continue(position, value_at_position, num_iterations, step,
                       finished):
    """Indicates whether the overall search should continue.

    Args:
      position: `Tensor` containing the current root estimates.
      value_at_position: `Tensor` containing the value of `objective_fn` at
        `position`.
      num_iterations: `Tensor` containing the current iteration index for each
        point.
      step: `Tensor` containing the size of the step to take for each point.
      finished: `Tensor` indicating for which points the search is finished.

    Returns:
      A boolean value indicating whether the overall search should continue.
    """
    del position, value_at_position, num_iterations, step  # Unused
    return ~tf.convert_to_tensor(
        stopping_policy_fn(finished), name='should_stop', dtype=tf.bool)

  # For each point in `position`, the search is stopped if either:
  # (1) A root has been found
  # (2) f(position) == f(position + step)
  # (3) The maximum number of iterations has been reached
  # In case (2), the search may be stopped both before the desired tolerance is
  # achieved (or even a root is found), and the maximum number of iterations is
  # reached.
  def _body(position, value_at_position, num_iterations, step, finished):
    """Performs one iteration of the secant root-finding algorithm.

    Args:
      position: `Tensor` containing the current root estimates.
      value_at_position: `Tensor` containing the value of `objective_fn` at
        `position`.
      num_iterations: `Tensor` containing the current iteration index for each
        point.
      step: `Tensor` containing the size of the step to take for each point.
      finished: `Tensor` indicating for which points the search is finished.

    Returns:
      The `Tensor`s to use for the next iteration of the algorithm.
    """

    # True if the search was already finished, or (1) or (3) just became true.
    was_finished = finished | (num_iterations >= max_iterations) | (
        tf.abs(step) < position_tolerance) | (
            tf.abs(value_at_position) < value_tolerance)

    # Compute the next position and the value at that point.
    next_position = tf.where(was_finished, position, position + step)
    value_at_next_position = tf.where(was_finished, value_at_position,
                                      objective_fn(next_position))

    # True if the search was already finished, or (2) just became true.
    is_finished = tf.equal(value_at_position, value_at_next_position)

    # Use the mid-point between the last two positions if (2) just became true.
    next_position = tf.where(is_finished & ~was_finished,
                             (position + next_position) * 0.5, next_position)

    # Once finished, stop updating the iteration index and set the step to zero.
    num_iterations = tf.where(is_finished, num_iterations, num_iterations + 1)
    next_step = tf.where(
        is_finished, zero, step * value_at_next_position /
        (value_at_position - value_at_next_position))

    return (next_position, value_at_next_position, num_iterations, next_step,
            is_finished)

  with tf.name_scope(
      name, 'secant_root',
      [position, next_position, value_at_position, max_iterations]):

    assertions = []
    if validate_args:
      assertions += [
          tf.Assert(
              tf.reduce_all(position_tolerance > zero), [position_tolerance]),
          tf.Assert(tf.reduce_all(value_tolerance > zero), [value_tolerance]),
          tf.Assert(
              tf.reduce_all(max_iterations >= num_iterations),
              [max_iterations]),
      ]

    with tf.control_dependencies(assertions):
      root, value_at_root, num_iterations, _, _ = tf.while_loop(
          _should_continue,
          _body,
          loop_vars=[
              position, value_at_position, num_iterations, step, finished
          ])

  return RootSearchResults(
      estimated_root=root,
      objective_at_estimated_root=value_at_root,
      num_iterations=num_iterations)
