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
"""Common functions for BFGS and L-BFGS algorithm."""

import collections
import numpy as np
import tensorflow.compat.v2 as tf

from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import prefer_static as ps
from tensorflow_probability.python.optimizer import linesearch

# A namedtuple to hold the point at which a line function is evaluated, the
# value of the function, directional derivative, and full gradient evaluated
# evaluated at that point. To be used with the linesearch method.
ValueAndGradient = collections.namedtuple('ValueAndGradient',
                                          ['x', 'f', 'df', 'full_gradient'])


def converged_any(converged, failed):
  """Condition to stop when any batch member converges, or all have failed."""
  return tf.reduce_any(converged) | tf.reduce_all(failed)


def converged_all(converged, failed):
  """Condition to stop when all batch members have converged or failed."""
  return tf.reduce_all(converged | failed)


def get_initial_state_args(value_and_gradients_function,
                           initial_position,
                           grad_tolerance,
                           control_inputs=None):
  """Returns a dictionary to populate the initial state of the search procedure.

  Performs an initial convergence check and the first evaluation of the
  objective function.

  Args:
    value_and_gradients_function: A Python callable that accepts a tensor and
      returns a tuple of two tensors: the objective function value and its
      derivative.
    initial_position: The starting point of the search procedure.
    grad_tolerance: The gradient tolerance for the procedure.
    control_inputs: Optional ops used to assert the validity of inputs, these
      are added as control dependencies to execute before the objective
      function is evaluated for the first time.

  Returns:
    An dictionary with values for the following keys:
      converged: True if the convergence check finds that the initial position
        is already an argmin of the objective function.
      failed: Initialized to False.
      num_objective_evaluations: Initialized to 1.
      position: Initialized to the initial position.
      objective_value: Initialized to the value of the objective function at
        the initial position.
      objective_gradient: Initialized to the gradient of the objective
        function at the initial position.
  """
  if control_inputs:
    with tf.control_dependencies(control_inputs):
      f0, df0 = value_and_gradients_function(initial_position)
  else:
    f0, df0 = value_and_gradients_function(initial_position)
  # This is a gradient-based convergence check.  We only do it for finite
  # objective values because we assume the gradient reported at a position with
  # a non-finite objective value is untrustworthy.  The main loop handles
  # non-finite objective values itself (see `terminate_if_not_finite`).
  init_converged = tf.math.is_finite(f0) & (norm(df0, dims=1) < grad_tolerance)
  return dict(
      converged=init_converged,
      failed=tf.zeros_like(init_converged),  # i.e. False.
      num_iterations=tf.convert_to_tensor(0),
      num_objective_evaluations=tf.convert_to_tensor(1),
      position=initial_position,
      objective_value=f0,
      objective_gradient=df0)


def terminate_if_not_finite(state, value=None, gradient=None):
  """Terminates optimization if the objective or gradient values are not finite.

  Specifically,

  - If the objective is -inf, stop with success, since this position is a global
    minimum.

  - Otherwise, if the objective or any component of the gradient is not finite,
    stop with failure.

  Why fail?

  - If the objective is nan, it could be a global minimum, but we can't know.

  - If the objective is +inf, we can't trust the gradient, so we can't know
    where to go next.  This should only ever happen on the first iteration,
    because the line search avoids returning points whose objective values are
    +inf.

  - If the gradient has any nonfinite values, we can't use it to move a finite
    amount.

  Args:
    state: A BfgsOptimizerResults or LbfgsOptimizerResults representing the
      current position and information about it.
    value: A Tensor giving the value of the objective function.
      `state.objective_value` if not supplied.
    gradient: A Tensor giving the gradient of the objective function.
      `state.objective_gradient` if not supplied.

  Returns:
    state: A namedputple of the same type with possibly updated `converged` and
      `failed` fields.
  """
  if value is None:
    value = state.objective_value
  if gradient is None:
    gradient = state.objective_gradient

  minus_inf_mask = _is_negative_inf(value)
  state = state._replace(converged=state.converged | minus_inf_mask)

  non_finite_mask = (
      ~tf.math.is_finite(value) |
      tf.reduce_any(~tf.math.is_finite(gradient), axis=-1))
  state = state._replace(failed=state.failed |
                         (~state.converged & non_finite_mask))

  return state


def _is_negative_inf(x):
  return x <= tf.constant(float('-inf'), dtype=x.dtype)


def line_search_step(state, value_and_gradients_function, search_direction,
                     grad_tolerance, f_relative_tolerance, x_tolerance,
                     stopping_condition, max_iterations, f_absolute_tolerance):
  """Performs the line search step of the BFGS search procedure.

  Uses hager_zhang line search procedure to compute a suitable step size
  to advance the current `state.position` along the given `search_direction`.
  Also, if the line search is successful, updates the `state.position` by
  taking the corresponding step.

  Args:
    state: A namedtuple instance holding values for the current state of the
      search procedure. The state must include the fields: `position`,
        `objective_value`, `objective_gradient`, `num_iterations`,
        `num_objective_evaluations`, `converged` and `failed`.
    value_and_gradients_function: A Python callable that accepts a point as a
      real `Tensor` of shape `[..., n]` and returns a tuple of two tensors of
      the same dtype: the objective function value, a real `Tensor` of shape
        `[...]`, and its derivative, another real `Tensor` of shape `[..., n]`.
    search_direction: A real `Tensor` of shape `[..., n]`. The direction along
      which to perform line search.
    grad_tolerance: Scalar `Tensor` of real dtype. Specifies the gradient
      tolerance for the procedure.
    f_relative_tolerance: Scalar `Tensor` of real dtype. Specifies the tolerance
      for the relative change in the objective value.
    x_tolerance: Scalar `Tensor` of real dtype. Specifies the tolerance for the
      change in the position.
    stopping_condition: A Python function that takes as input two Boolean
      tensors of shape `[...]`, and returns a Boolean scalar tensor. The input
      tensors are `converged` and `failed`, indicating the current status of
      each respective batch member; the return value states whether the
      algorithm should stop.
    max_iterations: A Python integer that is used as the maximum number of
      iterations of the hager_zhang line search algorithm
    f_absolute_tolerance: Scalar `Tensor` of real dtype. Specifies the tolerance
      for the absolute change in the objective value.

  Returns:
    A copy of the input state with the following fields updated:
      converged: a Boolean `Tensor` of shape `[...]` indicating whether the
        convergence criteria has been met.
      failed: a Boolean `Tensor` of shape `[...]` indicating whether the line
        search procedure failed to converge, or if either the updated gradient
        or objective function are no longer finite.
      num_iterations: Increased by 1.
      num_objective_evaluations: Increased by the number of times that the
        objective function got evaluated.
      position, objective_value, objective_gradient: If line search succeeded,
        updated by computing the new position and evaluating the objective
        function at that position.
  """
  line_search_value_grad_func = _restrict_along_direction(
      value_and_gradients_function, state.position, search_direction)
  derivative_at_start_pt = tf.reduce_sum(
      state.objective_gradient * search_direction, axis=-1)
  val_0 = ValueAndGradient(x=_broadcast(0, state.position),
                           f=state.objective_value,
                           df=derivative_at_start_pt,
                           full_gradient=state.objective_gradient)
  inactive = state.failed | state.converged
  ls_result = linesearch.hager_zhang(
      line_search_value_grad_func,
      initial_step_size=_broadcast(1, state.position),
      value_at_zero=val_0,
      converged=inactive,
      max_iterations=max_iterations)  # No search needed for these.

  state_after_ls = update_fields(
      state,
      failed=state.failed | (~state.converged & ~ls_result.converged),
      num_iterations=state.num_iterations + 1,
      num_objective_evaluations=(
          state.num_objective_evaluations + ls_result.func_evals))

  def _do_update_position():
    # For inactive batch members `left.x` is zero. However, their
    # `search_direction` might also be undefined, so we can't rely on
    # multiplication by zero to produce a `position_delta` of zero.
    position_delta = tf.where(
        inactive[..., tf.newaxis],
        dtype_util.as_numpy_dtype(search_direction.dtype)(0),
        search_direction * ls_result.left.x[..., tf.newaxis])
    return _update_position(
        state_after_ls,
        position_delta,
        ls_result.left.f,
        ls_result.left.full_gradient,
        grad_tolerance, f_relative_tolerance, x_tolerance,
        f_absolute_tolerance)  # pyformat: disable

  return ps.cond(
      stopping_condition(state.converged, state.failed),
      true_fn=lambda: state_after_ls,
      false_fn=_do_update_position)


def update_fields(state, **kwargs):
  """Copies the argument and overrides some of its fields.

  Args:
    state: A `collections.namedtuple` instance.
    **kwargs: Other named arguments represent fields in the tuple to override
      with new values.

  Returns:
    A namedtuple, of the same class as the input argument, with the updated
    fields.

  Raises:
    ValueError if the supplied kwargs contain fields not present in the
    input argument.
  """
  return state._replace(**kwargs)


def _restrict_along_direction(value_and_gradients_function,
                              position,
                              direction):
  """Restricts a function in n-dimensions to a given direction.

  Suppose f: R^n -> R. Then given a point x0 and a vector p0 in R^n, the
  restriction of the function along that direction is defined by:

  ```None
  g(t) = f(x0 + t * p0)
  ```

  This function performs this restriction on the given function. In addition, it
  also computes the gradient of the restricted function along the restriction
  direction. This is equivalent to computing `dg/dt` in the definition above.

  Args:
    value_and_gradients_function: Callable accepting a single real `Tensor`
      argument of shape `[..., n]` and returning a tuple of a real `Tensor` of
      shape `[...]` and a real `Tensor` of shape `[..., n]`. The multivariate
      function whose restriction is to be computed. The output values of the
      callable are the function value and the gradients at the input argument.
    position: `Tensor` of real dtype and shape consumable by
      `value_and_gradients_function`. Corresponds to `x0` in the definition
      above.
    direction: `Tensor` of the same dtype and shape as `position`. The direction
      along which to restrict the function. Note that the direction need not
      be a unit vector.

  Returns:
    restricted_value_and_gradients_func: A callable accepting a tensor of shape
      broadcastable to `[...]` and same dtype as `position` and returning a
      namedtuple of `Tensors`. The input tensor is the parameter along the
      direction labelled `t` above. The return value contains fields:
        x: A real `Tensor` of shape `[...]`. The input value `t` where the line
          function was evaluated, after any necessary broadcasting.
        f: A real `Tensor` of shape `[...]` containing the value of the
          function at the point `position + t * direction`.
        df: A real `Tensor` of shape `[...]` containing the derivative at
          `position + t * direction`.
        full_gradient: A real `Tensor` of shape `[..., n]`, the full gradient
          of the original `value_and_gradients_function`.
  """
  def _restricted_func(t):
    pt = position + t[..., tf.newaxis] * direction
    t = _broadcast(t, position)
    objective_value, gradient = value_and_gradients_function(pt)
    return ValueAndGradient(
        x=t,
        f=objective_value,
        df=tf.reduce_sum(gradient * direction, axis=-1),
        full_gradient=gradient)

  return _restricted_func


def _update_position(state,
                     position_delta,
                     next_objective,
                     next_gradient,
                     grad_tolerance,
                     f_relative_tolerance,
                     x_tolerance,
                     f_absolute_tolerance):  # pyformat: disable
  """Updates the state advancing its position by a given position_delta."""
  state = terminate_if_not_finite(state, next_objective, next_gradient)

  next_position = state.position + position_delta
  # pyformat: disable
  converged = ~state.failed & _check_convergence(state.position,
                                                 next_position,
                                                 state.objective_value,
                                                 next_objective,
                                                 next_gradient,
                                                 grad_tolerance,
                                                 f_relative_tolerance,
                                                 x_tolerance,
                                                 f_absolute_tolerance)
  # pyformat: enable
  return update_fields(
      state,
      converged=state.converged | converged,
      position=next_position,
      objective_value=next_objective,
      objective_gradient=next_gradient)


def norm(value, dims, order=None):
  """Compute the norm of the given (possibly batched) value.

  Args:
    value: A `Tensor` of real dtype.
    dims: An Python integer with the number of non-batching dimensions in the
      value, i.e. `dims=0` (scalars), `dims=1` (vectors), `dims=2` (matrices).
    order: Order of the norm, defaults to `np.inf`.
  """
  if dims == 0:
    return tf.math.abs(value)
  elif dims == 1:
    axis = -1
  elif dims == 2:
    axis = [-1, -2]
  else:
    ValueError(dims)
  if order is None:
    order = np.inf
  return tf.norm(tensor=value, axis=axis, ord=order)


def _check_convergence(current_position,
                       next_position,
                       current_objective,
                       next_objective,
                       next_gradient,
                       grad_tolerance,
                       f_relative_tolerance,
                       x_tolerance,
                       f_absolute_tolerance):  # pyformat: disable
  """Checks if the algorithm satisfies the convergence criteria."""
  grad_converged = norm(next_gradient, dims=1) <= grad_tolerance
  x_converged = norm(next_position - current_position, dims=1) <= x_tolerance
  f_relative_converged = (
      norm(next_objective - current_objective, dims=0) <=
      f_relative_tolerance * current_objective)
  f_absolute_converged = (
      norm(next_objective - current_objective, dims=0) <= f_absolute_tolerance)
  return (grad_converged | x_converged | f_relative_converged
          | f_absolute_converged)


def _broadcast(value, target):
  """Broadcast a value to match the batching dimensions of a target.

  If necessary the value is converted into a tensor. Both value and target
  should be of the same dtype.

  Args:
    value: A value to broadcast.
    target: A `Tensor` of shape [b1, ..., bn, d].

  Returns:
    A `Tensor` of shape [b1, ..., bn] and same dtype as the target.
  """
  return tf.broadcast_to(
      tf.convert_to_tensor(value, dtype=target.dtype),
      ps.shape(target)[:-1])
