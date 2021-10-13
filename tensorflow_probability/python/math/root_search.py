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

import collections

import numpy as np
import tensorflow.compat.v2 as tf

from tensorflow_probability.python.internal import assert_util
from tensorflow_probability.python.internal import callable_util
from tensorflow_probability.python.internal import distribution_util
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import prefer_static as ps
from tensorflow_probability.python.internal import tensorshape_util
from tensorflow.python.util import deprecation  # pylint: disable=g-direct-tensorflow-import

NUMPY_MODE = False

__all__ = [
    'bracket_root',
    'secant_root',
    'find_root_chandrupatla',
    'find_root_secant',
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


def find_root_secant(objective_fn,
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
      position. The function will search for roots in the neighborhood of each
      point. The shape of `initial_position` should match that of the input to
      `objective_fn`.
    next_position: Optional `Tensor` representing the next position in the
      search. If specified, this argument must broadcast with the shape of
      `initial_position` and have the same dtype. It will be used to compute the
      first step to take when searching for roots. If not specified, a default
      value will be used instead.
      Default value: `initial_position * (1 + 1e-4) + sign(initial_position) *
        1e-4`.
    value_at_position: Optional `Tensor` or Python float representing the value
      of `objective_fn` at `initial_position`. If specified, this argument must
      have the same shape and dtype as `initial_position`. If not specified, the
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
    ValueError: if a non-callable `stopping_policy_fn` is passed.

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
      objective_at_estimated_root=array([-4.81727769e-10, 7.44957651e-10]),
      num_iterations=array([ 7, 24], dtype=int32))

  tfp.math.secant_root(objective_fn=f,
                       initial_position=x,
                       stopping_policy_fn=tf.reduce_any)
  # ==> RootSearchResults(
      estimated_root=array([-0.90617985, 3.27379206]),
      objective_at_estimated_root=array([-4.81727769e-10, 2.66058312e+03]),
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
      objective_at_estimated_root=array([-7.81339438e-11, -4.81727769e-10]),
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
          [ 0.77459667, 0.90617985]]),
      objective_at_estimated_root=array([
          [-7.81339438e-11, -4.81727769e-10],
          [6.66025013e-11, 7.44957651e-10]]),
      num_iterations=array([
          [7, 7],
          [16, 24]], dtype=int32))
  ```
  """
  if not callable(stopping_policy_fn):
    raise ValueError('stopping_policy_fn must be callable')

  position = tf.convert_to_tensor(
      initial_position,
      name='position',
  )
  value_at_position = tf.convert_to_tensor(
      value_at_position or objective_fn(position),
      name='value_at_position',
      dtype=dtype_util.base_dtype(position.dtype))

  zero = tf.zeros_like(position)
  position_tolerance = tf.convert_to_tensor(
      position_tolerance, name='position_tolerance', dtype=position.dtype)
  value_tolerance = tf.convert_to_tensor(
      value_tolerance, name='value_tolerance', dtype=position.dtype)

  num_iterations = tf.zeros_like(position, dtype=tf.int32)
  max_iterations = tf.convert_to_tensor(max_iterations, dtype=tf.int32)
  max_iterations = tf.broadcast_to(
      max_iterations, name='max_iterations', shape=ps.shape(position))

  # Compute the step from `next_position` if present. This covers the case where
  # a user has two starting points, which bound the root or has a specific step
  # size in mind.
  if next_position is None:
    step = (position + tf.sign(position)) * 1e-4
  else:
    step = next_position - initial_position

  finished = tf.zeros(ps.shape(position), dtype=tf.bool)

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

  with tf.name_scope(name or 'find_root_secant'):

    assertions = []
    if validate_args:
      assertions += [
          tf.debugging.assert_greater(
              position_tolerance, zero,
              message='`position_tolerance` must be greater than 0.'),
          tf.debugging.assert_greater(
              value_tolerance, zero,
              message='`value_tolerance` must be greater than 0.'),
          tf.debugging.assert_greater_equal(
              max_iterations, num_iterations,
              message='`max_iterations` must be nonnegative.')
      ]

    with tf.control_dependencies(assertions):
      root, value_at_root, num_iterations, _, _ = tf.while_loop(
          cond=_should_continue,
          body=_body,
          loop_vars=(
              position, value_at_position, num_iterations, step, finished
          ))

  return RootSearchResults(
      estimated_root=root,
      objective_at_estimated_root=value_at_root,
      num_iterations=num_iterations)


secant_root = deprecation.deprecated_alias(
    'tfp.math.secant_root', 'tfp.math.find_root_secant', find_root_secant)


def _structure_broadcasting_where(c, x, y):
  """Selects elements from two structures using a shared condition `c`."""
  return tf.nest.map_structure(
      lambda xp, yp: tf.where(c, xp, yp), x, y)


def find_root_chandrupatla(objective_fn,
                           low=None,
                           high=None,
                           position_tolerance=1e-8,
                           value_tolerance=0.,
                           max_iterations=50,
                           stopping_policy_fn=tf.reduce_all,
                           validate_args=False,
                           name='find_root_chandrupatla'):
  r"""Finds root(s) of a scalar function using Chandrupatla's method.

  Chandrupatla's method [1, 2] is a root-finding algorithm that is guaranteed
  to converge if a root lies within the given bounds. It generalizes the
  [bisection method](https://en.wikipedia.org/wiki/Bisection_method); at each
  step it chooses to perform either bisection or inverse quadratic
  interpolation. This makes it similar in spirit to [Brent's method](
  https://en.wikipedia.org/wiki/Brent%27s_method), which also considers steps
  that use the secant method, but Chandrupatla's method is simpler and often
  converges at least as quickly [3].

  Args:
    objective_fn: Python callable for which roots are searched. It must be a
      callable of a single variable. `objective_fn` must return a `Tensor` with
      shape `batch_shape` and dtype matching `lower_bound` and `upper_bound`.
    low: Float `Tensor` of shape `batch_shape` representing a lower
      bound(s) on the value of a root(s). If either of `low` or `high` is not
      provided, both are ignored and `tfp.math.bracket_root` is used to attempt
      to infer bounds.
      Default value: `None`.
    high: Float `Tensor` of shape `batch_shape` representing an upper
      bound(s) on the value of a root(s). If either of `low` or `high` is not
      provided, both are ignored and `tfp.math.bracket_root` is used to attempt
      to infer bounds.
      Default value: `None`.
    position_tolerance: Optional `Tensor` representing the maximum absolute
      error in the positions of the estimated roots. Shape must broadcast with
      `batch_shape`.
      Default value: `1e-8`.
    value_tolerance: Optional `Tensor` representing the absolute error allowed
      in the value of the objective function. If the absolute value of
      `objective_fn` is smaller than
      `value_tolerance` at a given position, then that position is considered a
      root for the function. Shape must broadcast with `batch_shape`.
      Default value: `1e-8`.
    max_iterations: Optional `Tensor` or Python integer specifying the maximum
      number of steps to perform. Shape must broadcast with `batch_shape`.
      Default value: `50`.
    stopping_policy_fn: Python `callable` controlling the algorithm termination.
      It must be a callable accepting a `Tensor` of booleans with the same shape
      as `lower_bound` and `upper_bound` (denoting whether each search is
      finished), and returning a scalar boolean `Tensor` indicating
      whether the overall search should stop. Typical values are
      `tf.reduce_all` (which returns only when the search is finished for all
      points), and `tf.reduce_any` (which returns as soon as the search is
      finished for any point).
      Default value: `tf.reduce_all` (returns only when the search is finished
        for all points).
    validate_args: Python `bool` indicating whether to validate arguments.
      Default value: `False`.
    name: Python `str` name prefixed to ops created by this function.
      Default value: 'find_root_chandrupatla'.

  Returns:
    root_search_results: A Python `namedtuple` containing the following items:
      estimated_root: `Tensor` containing the last position explored. If the
        search was successful within the specified tolerance, this position is
        a root of the objective function.
      objective_at_estimated_root: `Tensor` containing the value of the
        objective function at `position`. If the search was successful within
        the specified tolerance, then this is close to 0.
      num_iterations: The number of iterations performed.

  #### References

  [1] Tirupathi R. Chandrupatla. A new hybrid quadratic/bisection algorithm for
      finding the zero of a nonlinear function without using derivatives.
      _Advances in Engineering Software_, 28.3:145-149, 1997.
  [2] Philipp OJ Scherer. Computational Physics. _Springer Berlin_,
      Heidelberg, 2010.
      Section 6.1.7.3 https://books.google.com/books?id=cC-8BAAAQBAJ&pg=PA95
  [3] Jason Sachs. Ten Little Algorithms, Part 5: Quadratic Extremum
      Interpolation and Chandrupatla's Method (2015).
      https://www.embeddedrelated.com/showarticle/855.php
  """

  ################################################
  # Loop variables used by Chandrupatla's method:
  #
  #  a: endpoint of an interval `[min(a, b), max(a, b)]` containing the
  #     root. There is no guarantee as to which of `a` and `b` is larger.
  #  b: endpoint of an interval `[min(a, b), max(a, b)]` containing the
  #       root. There is no guarantee as to which of `a` and `b` is larger.
  #  f_a: value of the objective at `a`.
  #  f_b: value of the objective at `b`.
  #  t: the next position to be evaluated as the coefficient of a convex
  #    combination of `a` and `b` (i.e., a value in the unit interval).
  #  num_iterations: integer number of steps taken so far.
  #  converged: boolean indicating whether each batch element has converged.
  #
  # All variables have the same shape `batch_shape`.

  def _should_continue(a, b, f_a, f_b, t, num_iterations, converged):
    del a, b, f_a, f_b, t  # Unused.
    all_converged = stopping_policy_fn(
        tf.logical_or(converged,
                      num_iterations >= max_iterations))
    return ~all_converged

  def _body(a, b, f_a, f_b, t, num_iterations, converged):
    """One step of Chandrupatla's method for root finding."""
    previous_loop_vars = (a, b, f_a, f_b, t, num_iterations, converged)
    finalized_elements = tf.logical_or(converged,
                                       num_iterations >= max_iterations)

    # Evaluate the new point.
    x_new = (1 - t) * a + t * b
    f_new = objective_fn(x_new)
    # If we've bisected (t==0.5) and the new float value for `a` is identical to
    # that from the previous iteration, then we'll keep bisecting (the
    # logic below will set t==0.5 for the next step), and nothing further will
    # change.
    at_fixed_point = tf.equal(x_new, a) & tf.equal(t, 0.5)
    # Otherwise, tighten the bounds.
    a, b, c, f_a, f_b, f_c = _structure_broadcasting_where(
        tf.equal(tf.math.sign(f_new), tf.math.sign(f_a)),
        (x_new, b, a, f_new, f_b, f_a),
        (x_new, a, b, f_new, f_a, f_b))

    # Check for convergence.
    f_best = tf.where(tf.abs(f_a) < tf.abs(f_b), f_a, f_b)
    interval_tolerance = position_tolerance / (tf.abs(b - c))
    converged = tf.logical_or(interval_tolerance > 0.5,
                              tf.logical_or(
                                  tf.math.abs(f_best) <= value_tolerance,
                                  at_fixed_point))

    # Propose next point to evaluate.
    xi = (a - b) / (c - b)
    phi = (f_a - f_b) / (f_c - f_b)
    t = tf.where(
        # Condition for inverse quadratic interpolation.
        tf.logical_and(1 - tf.math.sqrt(1 - xi) < phi,
                       tf.math.sqrt(xi) > phi),
        # Propose a point by inverse quadratic interpolation.
        (f_a / (f_b - f_a) * f_c / (f_b - f_c) +
         (c - a) / (b - a) * f_a / (f_c - f_a) * f_b / (f_c - f_b)),
        # Otherwise, just cut the interval in half (bisection).
        0.5)
    # Constrain the proposal to the current interval (0 < t < 1).
    t = tf.minimum(tf.maximum(t, interval_tolerance),
                   1 - interval_tolerance)

    # Update elements that haven't converged.
    return _structure_broadcasting_where(
        finalized_elements,
        previous_loop_vars,
        (a, b, f_a, f_b, t, num_iterations + 1, converged))

  with tf.name_scope(name):
    max_iterations = tf.convert_to_tensor(
        max_iterations, name='max_iterations', dtype_hint=tf.int32)
    dtype = dtype_util.common_dtype(
        [low, high, position_tolerance, value_tolerance], dtype_hint=tf.float32)
    position_tolerance = tf.convert_to_tensor(
        position_tolerance, name='position_tolerance', dtype=dtype)
    value_tolerance = tf.convert_to_tensor(
        value_tolerance, name='value_tolerance', dtype=dtype)

    if low is None or high is None:
      a, b = bracket_root(objective_fn, dtype=dtype)
    else:
      a = tf.convert_to_tensor(low, name='lower_bound', dtype=dtype)
      b = tf.convert_to_tensor(high, name='upper_bound', dtype=dtype)
    f_a, f_b = objective_fn(a), objective_fn(b)
    batch_shape = ps.broadcast_shape(ps.shape(f_a), ps.shape(f_b))

    assertions = []
    if validate_args:
      assertions += [
          assert_util.assert_none_equal(
              tf.math.sign(f_a), tf.math.sign(f_b),
              message='Bounds must be on different sides of a root.')]

    with tf.control_dependencies(assertions):
      initial_loop_vars = [
          a,
          b,
          f_a,
          f_b,
          tf.cast(0.5, dtype=f_a.dtype),
          tf.cast(0, dtype=max_iterations.dtype),
          False
      ]
      a, b, f_a, f_b, _, num_iterations, _ = tf.while_loop(
          _should_continue,
          _body,
          loop_vars=tf.nest.map_structure(
              lambda x: tf.broadcast_to(x, batch_shape),
              initial_loop_vars))

    x_best, f_best = _structure_broadcasting_where(
        tf.abs(f_a) < tf.abs(f_b),
        (a, f_a),
        (b, f_b))
  return RootSearchResults(
      estimated_root=x_best,
      objective_at_estimated_root=f_best,
      num_iterations=num_iterations)


def bracket_root(objective_fn,
                 dtype=tf.float32,
                 num_points=512,
                 name='bracket_root'):
  """Finds bounds that bracket a root of the objective function.

  This method attempts to return an interval bracketing a root of the objective
  function. It evaluates the objective in parallel at `num_points`
  locations, at exponentially increasing distance from the origin, and returns
  the first pair of adjacent points `[low, high]` such that the objective is
  finite and has a different sign at the two points. If no such pair was
  observed, it returns the trivial interval
  `[np.finfo(dtype).min, np.finfo(dtype).max]` containing all float values of
  the specified `dtype`. If the objective has multiple
  roots, the returned interval will contain at least one (but perhaps not all)
  of the roots.

  Args:
    objective_fn: Python callable for which roots are searched. It must be a
      continuous function that accepts a scalar `Tensor` of type `dtype` and
      returns a `Tensor` of shape `batch_shape`.
    dtype: Optional float `dtype` of inputs to `objective_fn`.
      Default value: `tf.float32`.
    num_points: Optional Python `int` number of points at which to evaluate
      the objective.
      Default value: `512`.
    name: Python `str` name given to ops created by this method.
  Returns:
    low: Float `Tensor` of shape `batch_shape` and dtype `dtype`. Lower bound
      on a root of `objective_fn`.
    high: Float `Tensor` of shape `batch_shape` and dtype `dtype`. Upper bound
      on a root of `objective_fn`.
  """
  with tf.name_scope(name):
    # Build a logarithmic sequence of `num_points` values from -inf to inf.
    dtype_info = np.finfo(dtype_util.as_numpy_dtype(dtype))
    xs_positive = tf.exp(tf.linspace(tf.cast(-10., dtype),
                                     tf.math.log(dtype_info.max),
                                     num_points // 2))
    xs = tf.concat([tf.reverse(-xs_positive, axis=[0]), xs_positive], axis=0)

    # Evaluate the objective at all points. The objective function may return
    # a batch of values (e.g., `objective(x) = x - batch_of_roots`).
    if NUMPY_MODE:
      objective_output_spec = objective_fn(tf.zeros([], dtype=dtype))
    else:
      objective_output_spec = callable_util.get_output_spec(
          objective_fn,
          tf.convert_to_tensor(0., dtype=dtype))
    batch_ndims = tensorshape_util.rank(objective_output_spec.shape)
    if batch_ndims is None:
      raise ValueError('Cannot infer tensor rank of objective values.')
    xs_pad_shape = ps.pad([num_points],
                          paddings=[[0, batch_ndims]],
                          constant_values=1)
    ys = objective_fn(tf.reshape(xs, xs_pad_shape))

    # Find the smallest point where the objective is finite.
    is_finite = tf.math.is_finite(ys)
    ys_transposed = distribution_util.move_dimension(  # For batch gather.
        ys, 0, -1)
    first_finite_value = tf.gather(
        ys_transposed,
        tf.argmax(is_finite, axis=0),  # Index of smallest finite point.
        batch_dims=batch_ndims,
        axis=-1)
    # Select the next point where the objective has a different sign.
    sign_change_idx = tf.argmax(
        tf.not_equal(tf.math.sign(ys),
                     tf.math.sign(first_finite_value)) & is_finite,
        axis=0)
    # If the sign never changes, we can't bracket a root.
    bracketing_failed = tf.equal(sign_change_idx, 0)
    # If the objective's sign is zero, we've found an actual root.
    root_found = tf.equal(tf.gather(tf.math.sign(ys_transposed),
                                    sign_change_idx,
                                    batch_dims=batch_ndims,
                                    axis=-1),
                          0.)
    return _structure_broadcasting_where(
        bracketing_failed,
        # If we didn't detect a sign change, fall back to the trivial interval.
        (dtype_info.min, dtype_info.max),
        # Otherwise, return the points around the sign change, unless we
        # actually evaluated a root, in which case, return the zero-width
        # bracket at that root.
        (tf.gather(xs, tf.where(bracketing_failed | root_found,
                                sign_change_idx,
                                sign_change_idx - 1)),
         tf.gather(xs, sign_change_idx)))
