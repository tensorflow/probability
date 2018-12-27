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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from tensorflow_probability.python.optimizer import linesearch


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
  return dict(
      converged=_check_within_tolerance(df0, grad_tolerance),
      failed=tf.convert_to_tensor(False),
      num_iterations=tf.convert_to_tensor(0),
      num_objective_evaluations=tf.convert_to_tensor(1),
      position=initial_position,
      objective_value=f0,
      objective_gradient=df0)


def line_search_step(state, value_and_gradients_function, search_direction,
                     grad_tolerance, f_relative_tolerance, x_tolerance):
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
      real `Tensor` and returns a tuple of two tensors of the same dtype: the
      objective function value, a real scalar `Tensor`, and its derivative, a
      `Tensor` with the same shape as the input to the function.
    search_direction: A real `Tensor` of the same shape as the `state.position`.
      The direction along which to perform line search.
    grad_tolerance: Scalar `Tensor` of real dtype. Specifies the gradient
      tolerance for the procedure.
    f_relative_tolerance: Scalar `Tensor` of real dtype. Specifies the
      tolerance for the relative change in the objective value.
    x_tolerance: Scalar `Tensor` of real dtype. Specifies the tolerance for the
      change in the position.

  Returns:
    A copy of the input state with the following fields updated:
      converged: True if the convergence criteria has been met.
      failed: True if the line search procedure failed to converge, or if
        either the updated gradient or objective function are no longer finite.
      num_iterations: Increased by 1.
      num_objective_evaluations: Increased by the number of times that the
        objective function got evaluated.
      position, objective_value, objective_gradient: If line search succeeded,
        updated by computing the new position and evaluating the objective
        function at that position.
  """
  dtype = state.position.dtype.base_dtype
  line_search_value_grad_func = _restrict_along_direction(
      value_and_gradients_function, state.position, search_direction)
  derivative_at_start_pt = tf.reduce_sum(state.objective_gradient *
                                         search_direction)
  ls_result = linesearch.hager_zhang(
      line_search_value_grad_func,
      initial_step_size=tf.convert_to_tensor(1, dtype=dtype),
      objective_at_zero=state.objective_value,
      grad_objective_at_zero=derivative_at_start_pt)

  state_after_ls = update_fields(
      state,
      failed=~ls_result.converged,  # Fail if line search failed to converge.
      num_iterations=state.num_iterations + 1,
      num_objective_evaluations=(
          state.num_objective_evaluations + ls_result.func_evals))

  def _do_update_position():
    return _update_position(
        value_and_gradients_function,
        state_after_ls,
        search_direction * ls_result.left_pt,
        grad_tolerance, f_relative_tolerance, x_tolerance)

  return tf.contrib.framework.smart_cond(
      state_after_ls.failed,
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
    value_and_gradients_function: Callable accepting a single `Tensor` argument
      of real dtype and returning a tuple of a scalar real `Tensor` and a
      `Tensor` of same shape as the input argument. The multivariate function
      whose restriction is to be computed. The output values of the callable are
      the function value and the gradients at the input argument.
    position: `Tensor` of real dtype and shape consumable by
      `value_and_gradients_function`. Corresponds to `x0` in the definition
      above.
    direction: `Tensor` of the same dtype and shape as `position`. The direction
      along which to restrict the function. Note that the direction need not
      be a unit vector.

  Returns:
    restricted_value_and_gradients_func: A callable accepting a scalar tensor
      of same dtype as `position` and returning a tuple of `Tensors`. The
      input tensor is the parameter along the direction labelled `t` above. The
      first element of the return tuple is a scalar `Tensor` containing
      the value of the function at the point `position + t * direction`. The
      second element is the derivative at `position + t * direction` which is
      also a scalar `Tensor` of the same dtype as `position`.
  """
  def _restricted_func(t):
    pt = position + t * direction
    objective_value, gradient = value_and_gradients_function(pt)
    return objective_value, tf.reduce_sum(gradient * direction)
  return _restricted_func


def _update_position(value_and_gradients_function,
                     state,
                     position_delta,
                     grad_tolerance,
                     f_relative_tolerance,
                     x_tolerance):
  """Updates the state advancing its position by a given position_delta."""
  next_position = state.position + position_delta
  next_objective, next_gradient = value_and_gradients_function(next_position)
  converged = _check_convergence(state.position,
                                 next_position,
                                 state.objective_value,
                                 next_objective,
                                 next_gradient,
                                 grad_tolerance,
                                 f_relative_tolerance,
                                 x_tolerance)
  return update_fields(
      state,
      converged=converged,
      failed=~_is_finite(next_objective, next_gradient),
      num_objective_evaluations=state.num_objective_evaluations + 1,
      position=next_position,
      objective_value=next_objective,
      objective_gradient=next_gradient)


def _check_within_tolerance(value, tolerance):
  """Checks whether the given value is below the supplied tolerance."""
  return tf.norm(value, ord=np.inf) <= tolerance


def _check_convergence(current_position,
                       next_position,
                       current_objective,
                       next_objective,
                       next_gradient,
                       grad_tolerance,
                       f_relative_tolerance,
                       x_tolerance):
  """Checks if the algorithm satisfies the convergence criteria."""
  grad_converged = _check_within_tolerance(next_gradient, grad_tolerance)
  x_converged = _check_within_tolerance(next_position - current_position,
                                        x_tolerance)
  f_converged = _check_within_tolerance(
      next_objective - current_objective,
      f_relative_tolerance * current_objective)
  return grad_converged | x_converged | f_converged


def _is_finite(arg1, *args):
  """Checks if the supplied tensors are finite.

  Args:
    arg1: A numeric `Tensor`.
    *args: (Optional) Other `Tensors` to check for finiteness.

  Returns:
    is_finite: Scalar boolean `Tensor` indicating whether all the supplied
      tensors are finite.
  """
  finite = tf.reduce_all(tf.is_finite(arg1))
  for arg in args:
    finite = finite & tf.reduce_all(tf.is_finite(arg))
  return finite
