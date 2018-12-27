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
"""The Limited-Memory BFGS minimization algorithm.

Limited-memory quasi-Newton methods are useful for solving large problems
whose Hessian matrices cannot be computed at a reasonable cost or are not
sparse. Instead of storing fully dense n x n approximations of Hessian
matrices, they only save a few vectors of length n that represent the
approximations implicitly.

This module implements the algorithm know as L-BFGS, which, as its name
suggests, is a limited-memory version of the BFGS algorithm.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

# Dependency imports
import tensorflow as tf

from tensorflow_probability.python.internal import distribution_util
from tensorflow_probability.python.optimizer import bfgs_utils


LBfgsOptimizerResults = collections.namedtuple(
    'LBfgsOptimizerResults', [
        'converged',  # Scalar boolean tensor indicating whether the minimum
                      # was found within tolerance.
        'failed',  # Scalar boolean tensor indicating whether a line search
                   # step failed to find a suitable step size satisfying Wolfe
                   # conditions. In the absence of any constraints on the
                   # number of objective evaluations permitted, this value will
                   # be the complement of `converged`. However, if there is
                   # a constraint and the search stopped due to available
                   # evaluations being exhausted, both `failed` and `converged`
                   # will be simultaneously False.
        'num_iterations',  # The number of iterations of the BFGS update.
        'num_objective_evaluations',  # The total number of objective
                                      # evaluations performed.
        'position',  # A tensor containing the last argument value found
                     # during the search. If the search converged, then
                     # this value is the argmin of the objective function.
        'objective_value',  # A tensor containing the value of the objective
                            # function at the `position`. If the search
                            # converged, then this is the (local) minimum of
                            # the objective function.
        'objective_gradient',  # A tensor containing the gradient of the
                               # objective function at the
                               # `final_position`. If the search converged
                               # the max-norm of this tensor should be
                               # below the tolerance.
        'position_deltas',  # A tensor encoding information about the latest
                            # changes in `position` during the algorithm
                            # execution. Its shape is of the form
                            # `(num_correction_pairs,) + position.shape` where
                            # `num_correction_pairs` is given as an argument to
                            # the minimize function.
        'gradient_deltas',  # A tensor encoding information about the latest
                            # changes in `objective_gradient` during the
                            # algorithm execution. Has the same shape as
                            # position_deltas.
    ])


def minimize(value_and_gradients_function,
             initial_position,
             num_correction_pairs=10,
             tolerance=1e-8,
             x_tolerance=0,
             f_relative_tolerance=0,
             initial_inverse_hessian_estimate=None,
             max_iterations=50,
             parallel_iterations=1,
             name=None):
  """Applies the L-BFGS algorithm to minimize a differentiable function.

  Performs unconstrained minimization of a differentiable function using the
  L-BFGS scheme. See [Nocedal and Wright(2006)][1] for details of the algorithm.

  ### Usage:

  The following example demonstrates the L-BFGS optimizer attempting to find the
  minimum for a simple high-dimensional quadratic objective function.

  ```python
    # A high-dimensional quadratic bowl.
    ndims = 60
    minimum = np.ones([ndims], dtype='float64')
    scales = np.arange(ndims, dtype='float64') + 1.0

    # The objective function and the gradient.
    def quadratic(x):
      value = tf.reduce_sum(scales * (x - minimum) ** 2)
      return value, tf.gradients(value, x)[0]

    start = np.arange(ndims, 0, -1, dtype='float64')
    optim_results = tfp.optimizer.lbfgs_minimize(
        quadratic, initial_position=start, num_correction_pairs=10,
        tolerance=1e-8)

    with tf.Session() as session:
      results = session.run(optim_results)
      # Check that the search converged
      assert(results.converged)
      # Check that the argmin is close to the actual value.
      np.testing.assert_allclose(results.position, minimum)
  ```

  ### References:

  [1] Jorge Nocedal, Stephen Wright. Numerical Optimization. Springer Series
      in Operations Research. pp 176-180. 2006

  http://pages.mtu.edu/~struther/Courses/OLD/Sp2013/5630/Jorge_Nocedal_Numerical_optimization_267490.pdf

  Args:
    value_and_gradients_function:  A Python callable that accepts a point as a
      real `Tensor` and returns a tuple of `Tensor`s of real dtype containing
      the value of the function and its gradient at that point. The function
      to be minimized. The first component of the return value should be a
      real scalar `Tensor`. The second component (the gradient) should have the
      same shape as the input value to the function.
    initial_position: `Tensor` of real dtype. The starting point of the search
      procedure. Should be a point at which the function value and the gradient
      norm are finite.
    num_correction_pairs: Positive integer. Specifies the maximum number of
      (position_delta, gradient_delta) correction pairs to keep as implicit
      approximation of the Hessian matrix.
    tolerance: Scalar `Tensor` of real dtype. Specifies the gradient tolerance
      for the procedure. If the supremum norm of the gradient vector is below
      this number, the algorithm is stopped.
    x_tolerance: Scalar `Tensor` of real dtype. If the absolute change in the
      position between one iteration and the next is smaller than this number,
      the algorithm is stopped.
    f_relative_tolerance: Scalar `Tensor` of real dtype. If the relative change
      in the objective value between one iteration and the next is smaller
      than this value, the algorithm is stopped.
    initial_inverse_hessian_estimate: None. Option currently not supported.
    max_iterations: Scalar positive int32 `Tensor`. The maximum number of
      iterations for BFGS updates.
    parallel_iterations: Positive integer. The number of iterations allowed to
      run in parallel.
    name: (Optional) Python str. The name prefixed to the ops created by this
      function. If not supplied, the default name 'minimize' is used.

  Returns:
    optimizer_results: A namedtuple containing the following items:
      converged: Scalar boolean tensor indicating whether the minimum was
        found within tolerance.
      failed:  Scalar boolean tensor indicating whether a line search
        step failed to find a suitable step size satisfying Wolfe
        conditions. In the absence of any constraints on the
        number of objective evaluations permitted, this value will
        be the complement of `converged`. However, if there is
        a constraint and the search stopped due to available
        evaluations being exhausted, both `failed` and `converged`
        will be simultaneously False.
      num_objective_evaluations: The total number of objective
        evaluations performed.
      position: A tensor containing the last argument value found
        during the search. If the search converged, then
        this value is the argmin of the objective function.
      objective_value: A tensor containing the value of the objective
        function at the `position`. If the search converged, then this is
        the (local) minimum of the objective function.
      objective_gradient: A tensor containing the gradient of the objective
        function at the `position`. If the search converged the
        max-norm of this tensor should be below the tolerance.
      position_deltas: A tensor encoding information about the latest
        changes in `position` during the algorithm execution.
      gradient_deltas: A tensor encoding information about the latest
        changes in `objective_gradient` during the algorithm execution.
  """
  if initial_inverse_hessian_estimate is not None:
    raise NotImplementedError(
        'Support of initial_inverse_hessian_estimate arg not yet implemented')

  with tf.name_scope(name, 'minimize', [initial_position,
                                        tolerance]):
    initial_position = tf.convert_to_tensor(initial_position,
                                            name='initial_position')
    dtype = initial_position.dtype.base_dtype
    tolerance = tf.convert_to_tensor(tolerance, dtype=dtype,
                                     name='grad_tolerance')
    f_relative_tolerance = tf.convert_to_tensor(f_relative_tolerance,
                                                dtype=dtype,
                                                name='f_relative_tolerance')
    x_tolerance = tf.convert_to_tensor(x_tolerance,
                                       dtype=dtype,
                                       name='x_tolerance')
    max_iterations = tf.convert_to_tensor(max_iterations, name='max_iterations')

    # The `state` here is a `LBfgsOptimizerResults` tuple with values for the
    # current state of the algorithm computation.
    def _cond(state):
      """Stopping condition for the algorithm."""
      should_stop = (state.converged | state.failed |
                     (state.num_iterations >= max_iterations))
      return ~should_stop

    def _body(current_state):
      """Main optimization loop."""

      search_direction = _get_search_direction(current_state)

      # TODO(b/120134934): Check if the derivative at the start point is not
      # negative, if so then reset position/gradient deltas and recompute
      # search direction.

      next_state = bfgs_utils.line_search_step(
          current_state,
          value_and_gradients_function, search_direction,
          tolerance, f_relative_tolerance, x_tolerance)

      def _update_inv_hessian():
        position_delta = next_state.position - current_state.position
        gradient_delta = (
            next_state.objective_gradient - current_state.objective_gradient)
        return bfgs_utils.update_fields(
            next_state,
            position_deltas=_stack_append(current_state.position_deltas,
                                          position_delta),
            gradient_deltas=_stack_append(current_state.gradient_deltas,
                                          gradient_delta))

      # If not failed or converged, update the Hessian estimate.
      state_after_inv_hessian_update = tf.contrib.framework.smart_cond(
          next_state.converged | next_state.failed,
          lambda: next_state,
          _update_inv_hessian)
      return [state_after_inv_hessian_update]

    initial_state = _get_initial_state(value_and_gradients_function,
                                       initial_position,
                                       num_correction_pairs,
                                       tolerance)
    return tf.while_loop(_cond, _body, [initial_state],
                         parallel_iterations=parallel_iterations)[0]


def _get_initial_state(value_and_gradients_function,
                       initial_position,
                       num_correction_pairs,
                       tolerance):
  """Create LBfgsOptimizerResults with initial state of search procedure."""
  init_args = bfgs_utils.get_initial_state_args(
      value_and_gradients_function,
      initial_position,
      tolerance)
  empty_stack = _make_empty_stack_like(initial_position, num_correction_pairs)
  init_args.update(position_deltas=empty_stack, gradient_deltas=empty_stack)
  return LBfgsOptimizerResults(**init_args)


def _get_search_direction(state):
  """Computes the search direction to follow at the current state.

  On the `k`-th iteration of the main L-BFGS algorithm, the state has collected
  the most recent `m` correction pairs in position_deltas and gradient_deltas,
  where `k = state.num_iterations` and `m = min(k, num_correction_pairs)`.

  Assuming these, the code below is an implementation of the L-BFGS two-loop
  recursion algorithm given by [Nocedal and Wright(2006)][1]:

  ```None
    q_direction = objective_gradient
    for i in reversed(range(m)):  # First loop.
      inv_rho[i] = gradient_deltas[i]^T * position_deltas[i]
      alpha[i] = position_deltas[i]^T * q_direction / inv_rho[i]
      q_direction = q_direction - alpha[i] * gradient_deltas[i]

    kth_inv_hessian_factor = (gradient_deltas[-1]^T * position_deltas[-1] /
                              gradient_deltas[-1]^T * gradient_deltas[-1])
    r_direction = kth_inv_hessian_factor * I * q_direction

    for i in range(m):  # Second loop.
      beta = gradient_deltas[i]^T * r_direction / inv_rho[i]
      r_direction = r_direction + position_deltas[i] * (alpha[i] - beta)

    return -r_direction  # Approximates - H_k * objective_gradient.
  ```

  Args:
    state: A `LBfgsOptimizerResults` tuple with the current state of the
      search procedure.

  Returns:
    A real `Tensor` of the same shape as the `state.position`. The direction
    along which to perform line search.
  """
  # The number of correction pairs that have been collected so far.
  num_elements = tf.minimum(
      state.num_iterations,
      distribution_util.prefer_static_shape(state.position_deltas)[0])

  def _two_loop_algorithm():
    """L-BFGS two-loop algorithm."""
    # Correction pairs are always appended to the end, so only the latest
    # `num_elements` vectors have valid position/gradient deltas.
    position_deltas = state.position_deltas[-num_elements:]
    gradient_deltas = state.gradient_deltas[-num_elements:]

    # Pre-compute all `inv_rho[i]`s.
    inv_rhos = tf.reduce_sum(gradient_deltas * position_deltas, axis=1)

    def first_loop(acc, args):
      _, q_direction = acc
      position_delta, gradient_delta, inv_rho = args
      alpha = tf.reduce_sum(position_delta * q_direction) / inv_rho
      return (alpha, q_direction - alpha * gradient_delta)

    # Run first loop body computing and collecting `alpha[i]`s, while also
    # computing the updated `q_direction` at each step.
    zero = tf.zeros_like(inv_rhos[0])
    alphas, q_directions = tf.scan(
        first_loop, [position_deltas, gradient_deltas, inv_rhos],
        initializer=(zero, state.objective_gradient), reverse=True)

    # We use `H^0_k = gamma_k * I` as an estimate for the initial inverse
    # hessian for the k-th iteration; then `r_direction = H^0_k * q_direction`.
    gamma_k = inv_rhos[-1] / tf.reduce_sum(
        gradient_deltas[-1] * gradient_deltas[-1])
    r_direction = gamma_k * q_directions[0]

    def second_loop(r_direction, args):
      alpha, position_delta, gradient_delta, inv_rho = args
      beta = tf.reduce_sum(gradient_delta * r_direction) / inv_rho
      return r_direction + (alpha - beta) * position_delta

    # Finally, run second loop body computing the updated `r_direction` at each
    # step.
    r_directions = tf.scan(
        second_loop, [alphas, position_deltas, gradient_deltas, inv_rhos],
        initializer=r_direction)
    return -r_directions[-1]

  return tf.contrib.framework.smart_cond(
      tf.equal(num_elements, 0),
      (lambda: -state.objective_gradient),
      _two_loop_algorithm)


def _make_empty_stack_like(element, k):
  """Creates a `tf.Tensor` suitable to hold k element-shaped vectors.

  For example:

  ```python
    element = tf.constant([1., 2., 3., 4., 5.])

    _make_empty_stack_like(element, 3)
    # => [[0., 0., 0., 0., 0.],
    #     [0., 0., 0., 0., 0.],
    #     [0., 0., 0., 0., 0.]]
  ```

  Args:
    element: A `tf.Tensor`, only its shape and dtype information are relevant.
    k: A positive scalar integer `tf.Tensor`.

  Returns:
    A zero-filed `Tensor` of shape `(k, ) + tf.shape(element)` and dtype same
    as `element`.
  """
  stack_shape = tf.concat(
      [[k], distribution_util.prefer_static_shape(element)], axis=0)
  return tf.zeros(stack_shape, dtype=element.dtype.base_dtype)


def _stack_append(stack, element):
  """Appends a new `element` at the top of the `stack` and drops the bottom one.

  For example:

  ```python
    stack = tf.constant([[0., 0., 0., 0., 0.],
                         [1., 2., 3., 4., 5.],
                         [5., 4., 3., 2., 1.]])

    element = tf.constant([1., 2., 3., 2., 1.])

    _stack_append(stack, element)
    # => [[1., 2., 3., 4., 5.],
    #     [5., 4., 3., 2., 1.],
    #     [1., 2., 3., 2., 1.]]
  ```

  Args:
    stack: A `tf.Tensor` of shape (k, d1, ..., dn).
    element: A `tf.Tensor` of shape (d1, ..., dn).

  Returns:
    A new `tf.Tensor`, of the same shape as the input `stack`.
  """
  return tf.concat([stack[1:], [element]], axis=0)
