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

This module implements the algorithm known as L-BFGS, which, as its name
suggests, is a limited-memory version of the BFGS algorithm.
"""
import collections

# Dependency imports
import tensorflow.compat.v2 as tf

from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import prefer_static as ps
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
             previous_optimizer_results=None,
             num_correction_pairs=10,
             tolerance=1e-8,
             x_tolerance=0,
             f_relative_tolerance=0,
             initial_inverse_hessian_estimate=None,
             max_iterations=50,
             parallel_iterations=1,
             stopping_condition=None,
             max_line_search_iterations=50,
             f_absolute_tolerance=0,
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
    def quadratic_loss_and_gradient(x):
      return tfp.math.value_and_gradient(
          lambda x: tf.reduce_sum(
              scales * tf.math.squared_difference(x, minimum), axis=-1),
          x)
    start = np.arange(ndims, 0, -1, dtype='float64')
    optim_results = tfp.optimizer.lbfgs_minimize(
        quadratic_loss_and_gradient,
        initial_position=start,
        num_correction_pairs=10,
        tolerance=1e-8)

    # Check that the search converged
    assert(optim_results.converged)
    # Check that the argmin is close to the actual value.
    np.testing.assert_allclose(optim_results.position, minimum)
  ```

  ### References:

  [1] Jorge Nocedal, Stephen Wright. Numerical Optimization. Springer Series
      in Operations Research. pp 176-180. 2006

  http://pages.mtu.edu/~struther/Courses/OLD/Sp2013/5630/Jorge_Nocedal_Numerical_optimization_267490.pdf

  Args:
    value_and_gradients_function:  A Python callable that accepts a point as a
      real `Tensor` and returns a tuple of `Tensor`s of real dtype containing
      the value of the function and its gradient at that point. The function
      to be minimized. The input is of shape `[..., n]`, where `n` is the size
      of the domain of input points, and all others are batching dimensions.
      The first component of the return value is a real `Tensor` of matching
      shape `[...]`. The second component (the gradient) is also of shape
      `[..., n]` like the input value to the function.
    initial_position: Real `Tensor` of shape `[..., n]`. The starting point, or
      points when using batching dimensions, of the search procedure. At these
      points the function value and the gradient norm should be finite.
      Exactly one of `initial_position` and `previous_optimizer_results` can be
      non-None.
    previous_optimizer_results: An `LBfgsOptimizerResults` namedtuple to
      intialize the optimizer state from, instead of an `initial_position`.
      This can be passed in from a previous return value to resume optimization
      with a different `stopping_condition`. Exactly one of `initial_position`
      and `previous_optimizer_results` can be non-None.
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
      iterations for L-BFGS updates.
    parallel_iterations: Positive integer. The number of iterations allowed to
      run in parallel.
    stopping_condition: (Optional) A Python function that takes as input two
      Boolean tensors of shape `[...]`, and returns a Boolean scalar tensor.
      The input tensors are `converged` and `failed`, indicating the current
      status of each respective batch member; the return value states whether
      the algorithm should stop. The default is tfp.optimizer.converged_all
      which only stops when all batch members have either converged or failed.
      An alternative is tfp.optimizer.converged_any which stops as soon as one
      batch member has converged, or when all have failed.
    max_line_search_iterations: Python int. The maximum number of iterations
      for the `hager_zhang` line search algorithm.
    f_absolute_tolerance: Scalar `Tensor` of real dtype. If the absolute change
      in the objective value between one iteration and the next is smaller
      than this value, the algorithm is stopped.
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

  if stopping_condition is None:
    stopping_condition = bfgs_utils.converged_all

  with tf.name_scope(name or 'minimize'):
    if (initial_position is None) == (previous_optimizer_results is None):
      raise ValueError(
          'Exactly one of `initial_position` or '
          '`previous_optimizer_results` may be specified.')

    if initial_position is not None:
      initial_position = tf.convert_to_tensor(
          initial_position, name='initial_position')
      dtype = dtype_util.base_dtype(initial_position.dtype)

    if previous_optimizer_results is not None:
      dtype = dtype_util.base_dtype(previous_optimizer_results.position.dtype)

    tolerance = tf.convert_to_tensor(
        tolerance, dtype=dtype, name='grad_tolerance')
    f_relative_tolerance = tf.convert_to_tensor(
        f_relative_tolerance, dtype=dtype, name='f_relative_tolerance')
    f_absolute_tolerance = tf.convert_to_tensor(
        f_absolute_tolerance, dtype=dtype, name='f_absolute_tolerance')
    x_tolerance = tf.convert_to_tensor(
        x_tolerance, dtype=dtype, name='x_tolerance')
    max_iterations = tf.convert_to_tensor(max_iterations, name='max_iterations')

    # The `state` here is a `LBfgsOptimizerResults` tuple with values for the
    # current state of the algorithm computation.
    def _cond(state):
      """Continue if iterations remain and stopping condition is not met."""
      return ((state.num_iterations < max_iterations) &
              tf.logical_not(stopping_condition(state.converged, state.failed)))

    def _body(current_state):
      """Main optimization loop."""
      current_state = bfgs_utils.terminate_if_not_finite(current_state)
      search_direction = _get_search_direction(current_state)

      # TODO(b/120134934): Check if the derivative at the start point is not
      # negative, if so then reset position/gradient deltas and recompute
      # search direction.

      next_state = bfgs_utils.line_search_step(
          current_state, value_and_gradients_function, search_direction,
          tolerance, f_relative_tolerance, x_tolerance, stopping_condition,
          max_line_search_iterations, f_absolute_tolerance)

      # If not failed or converged, update the Hessian estimate.
      should_update = ~(next_state.converged | next_state.failed)
      state_after_inv_hessian_update = bfgs_utils.update_fields(
          next_state,
          position_deltas=_queue_push(
              current_state.position_deltas, should_update,
              next_state.position - current_state.position),
          gradient_deltas=_queue_push(
              current_state.gradient_deltas, should_update,
              next_state.objective_gradient - current_state.objective_gradient))
      return [state_after_inv_hessian_update]

    if previous_optimizer_results is None:
      assert initial_position is not None
      initial_state = _get_initial_state(value_and_gradients_function,
                                         initial_position,
                                         num_correction_pairs,
                                         tolerance)
    else:
      initial_state = previous_optimizer_results

    return tf.while_loop(
        cond=_cond,
        body=_body,
        loop_vars=[initial_state],
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
  empty_queue = _make_empty_queue_for(num_correction_pairs, initial_position)
  init_args.update(position_deltas=empty_queue, gradient_deltas=empty_queue)
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
  num_elements = ps.minimum(
      state.num_iterations,  # TODO(b/162733947): Change loop state -> closure.
      ps.shape(state.position_deltas)[0])

  def _two_loop_algorithm():
    """L-BFGS two-loop algorithm."""
    # Correction pairs are always appended to the end, so only the latest
    # `num_elements` vectors have valid position/gradient deltas. Vectors
    # that haven't been computed yet are zero.
    position_deltas = state.position_deltas
    gradient_deltas = state.gradient_deltas

    # Pre-compute all `inv_rho[i]`s.
    inv_rhos = tf.reduce_sum(
        gradient_deltas * position_deltas, axis=-1)

    def first_loop(acc, args):
      _, q_direction = acc
      position_delta, gradient_delta, inv_rho = args
      alpha = tf.math.divide_no_nan(
          tf.reduce_sum(position_delta * q_direction, axis=-1), inv_rho)
      direction_delta = alpha[..., tf.newaxis] * gradient_delta
      return (alpha, q_direction - direction_delta)

    # Run first loop body computing and collecting `alpha[i]`s, while also
    # computing the updated `q_direction` at each step.
    zero = tf.zeros_like(inv_rhos[-num_elements])
    alphas, q_directions = tf.scan(
        first_loop, [position_deltas, gradient_deltas, inv_rhos],
        initializer=(zero, state.objective_gradient), reverse=True)

    # We use `H^0_k = gamma_k * I` as an estimate for the initial inverse
    # hessian for the k-th iteration; then `r_direction = H^0_k * q_direction`.
    gamma_k = inv_rhos[-1] / tf.reduce_sum(
        gradient_deltas[-1] * gradient_deltas[-1], axis=-1)
    r_direction = gamma_k[..., tf.newaxis] * q_directions[-num_elements]

    def second_loop(r_direction, args):
      alpha, position_delta, gradient_delta, inv_rho = args
      beta = tf.math.divide_no_nan(
          tf.reduce_sum(gradient_delta * r_direction, axis=-1), inv_rho)
      direction_delta = (alpha - beta)[..., tf.newaxis] * position_delta
      return r_direction + direction_delta

    # Finally, run second loop body computing the updated `r_direction` at each
    # step.
    r_directions = tf.scan(
        second_loop, [alphas, position_deltas, gradient_deltas, inv_rhos],
        initializer=r_direction)
    return -r_directions[-1]

  return ps.cond(ps.equal(num_elements, 0),
                 lambda: -state.objective_gradient,
                 _two_loop_algorithm)


def _make_empty_queue_for(k, element):
  """Creates a `tf.Tensor` suitable to hold `k` element-shaped tensors.

  For example:

  ```python
    element = tf.constant([[0., 1., 2., 3., 4.],
                           [5., 6., 7., 8., 9.]])

    # A queue capable of holding 3 elements.
    _make_empty_queue_for(3, element)
    # => [[[ 0.,  0.,  0.,  0.,  0.],
    #      [ 0.,  0.,  0.,  0.,  0.]],
    #
    #     [[ 0.,  0.,  0.,  0.,  0.],
    #      [ 0.,  0.,  0.,  0.,  0.]],
    #
    #     [[ 0.,  0.,  0.,  0.,  0.],
    #      [ 0.,  0.,  0.,  0.,  0.]]]
  ```

  Args:
    k: A positive scalar integer, number of elements that each queue will hold.
    element: A `tf.Tensor`, only its shape and dtype information are relevant.

  Returns:
    A zero-filed `tf.Tensor` of shape `(k,) + tf.shape(element)` and same dtype
    as `element`.
  """
  queue_shape = ps.concat([[k], ps.shape(element)], axis=0)
  return tf.zeros(queue_shape, dtype=dtype_util.base_dtype(element.dtype))


def _queue_push(queue, should_update, new_vecs):
  """Conditionally push new vectors into a batch of first-in-first-out queues.

  The `queue` of shape `[k, ..., n]` can be thought of as a batch of queues,
  each holding `k` n-D vectors; while `new_vecs` of shape `[..., n]` is a
  fresh new batch of n-D vectors. The `should_update` batch of Boolean scalars,
  i.e. shape `[...]`, indicates batch members whose corresponding n-D vector in
  `new_vecs` should be added at the back of its queue, pushing out the
  corresponding n-D vector from the front. Batch members in `new_vecs` for
  which `should_update` is False are ignored.

  Note: the choice of placing `k` at the dimension 0 of the queue is
  constrained by the L-BFGS two-loop algorithm above. The algorithm uses
  tf.scan to iterate over the `k` correction pairs simulatneously across all
  batches, and tf.scan itself can only iterate over dimension 0.

  For example:

  ```python
    k, b, n = (3, 2, 5)
    queue = tf.reshape(tf.range(30), (k, b, n))
    # => [[[ 0,  1,  2,  3,  4],
    #      [ 5,  6,  7,  8,  9]],
    #
    #     [[10, 11, 12, 13, 14],
    #      [15, 16, 17, 18, 19]],
    #
    #     [[20, 21, 22, 23, 24],
    #      [25, 26, 27, 28, 29]]]

    element = tf.reshape(tf.range(30, 40), (b, n))
    # => [[30, 31, 32, 33, 34],
          [35, 36, 37, 38, 39]]

    should_update = tf.constant([True, False])  # Shape: (b,)

    _queue_add(should_update, queue, element)
    # => [[[10, 11, 12, 13, 14],
    #      [ 5,  6,  7,  8,  9]],
    #
    #     [[20, 21, 22, 23, 24],
    #      [15, 16, 17, 18, 19]],
    #
    #     [[30, 31, 32, 33, 34],
    #      [25, 26, 27, 28, 29]]]
  ```

  Args:
    queue: A `tf.Tensor` of shape `[k, ..., n]`; a batch of queues each with
      `k` n-D vectors.
    should_update: A Boolean `tf.Tensor` of shape `[...]` indicating batch
      members where new vectors should be added to their queues.
    new_vecs: A `tf.Tensor` of shape `[..., n]`; a batch of n-D vectors to add
      at the end of their respective queues, pushing out the first element from
      each.

  Returns:
    A new `tf.Tensor` of shape `[k, ..., n]`.
  """
  new_queue = tf.concat([queue[1:], [new_vecs]], axis=0)
  return tf.where(
      should_update[tf.newaxis, ..., tf.newaxis], new_queue, queue)
