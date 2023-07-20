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
"""The Broyden-Fletcher-Goldfarb-Shanno minimization algorithm.

Quasi-Newton methods are a class of popular first-order optimization algorithm.
These methods use a positive-definite approximation to the exact
Hessian to find the search direction. The Broyden-Fletcher-Goldfarb-Shanno
algorithm (BFGS) is a specific implementation of this general idea.

This module provides an implementation of the BFGS scheme using the Hager Zhang
line search method.
"""

import collections

# Dependency imports
import tensorflow.compat.v2 as tf

from tensorflow_probability.python.internal import distribution_util
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import prefer_static as ps
from tensorflow_probability.python.optimizer import bfgs_utils


BfgsOptimizerResults = collections.namedtuple(
    'BfgsOptimizerResults', [
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
        'inverse_hessian_estimate'  # A tensor containing the inverse of the
                                    # estimated Hessian.
    ])


def minimize(value_and_gradients_function,
             initial_position,
             tolerance=1e-8,
             x_tolerance=0,
             f_relative_tolerance=0,
             initial_inverse_hessian_estimate=None,
             max_iterations=50,
             parallel_iterations=1,
             stopping_condition=None,
             validate_args=True,
             max_line_search_iterations=50,
             f_absolute_tolerance=0,
             name=None):
  """Applies the BFGS algorithm to minimize a differentiable function.

  Performs unconstrained minimization of a differentiable function using the
  BFGS scheme. For details of the algorithm, see [Nocedal and Wright(2006)][1].

  ### Usage:

  The following example demonstrates the BFGS optimizer attempting to find the
  minimum for a simple two dimensional quadratic objective function.

  ```python
    minimum = np.array([1.0, 1.0])  # The center of the quadratic bowl.
    scales = np.array([2.0, 3.0])  # The scales along the two axes.

    # The objective function and the gradient.
    def quadratic_loss_and_gradient(x):
      return tfp.math.value_and_gradient(
          lambda x: tf.reduce_sum(
              scales * tf.math.squared_difference(x, minimum), axis=-1),
          x)

    start = tf.constant([0.6, 0.8])  # Starting point for the search.
    optim_results = tfp.optimizer.bfgs_minimize(
        quadratic_loss_and_gradient, initial_position=start, tolerance=1e-8)

    # Check that the search converged
    assert(optim_results.converged)
    # Check that the argmin is close to the actual value.
    np.testing.assert_allclose(optim_results.position, minimum)
    # Print out the total number of function evaluations it took. Should be 5.
    print ("Function evaluations: %d" % optim_results.num_objective_evaluations)
  ```

  ### References:
  [1]: Jorge Nocedal, Stephen Wright. Numerical Optimization. Springer Series in
    Operations Research. pp 136-140. 2006
    http://pages.mtu.edu/~struther/Courses/OLD/Sp2013/5630/Jorge_Nocedal_Numerical_optimization_267490.pdf

  Args:
    value_and_gradients_function:  A Python callable that accepts a point as a
      real `Tensor` and returns a tuple of `Tensor`s of real dtype containing
      the value of the function and its gradient at that point. The function
      to be minimized. The input should be of shape `[..., n]`, where `n` is
      the size of the domain of input points, and all others are batching
      dimensions. The first component of the return value should be a real
      `Tensor` of matching shape `[...]`. The second component (the gradient)
      should also be of shape `[..., n]` like the input value to the function.
    initial_position: real `Tensor` of shape `[..., n]`. The starting point, or
      points when using batching dimensions, of the search procedure. At these
      points the function value and the gradient norm should be finite.
    tolerance: Scalar `Tensor` of real dtype. Specifies the gradient tolerance
      for the procedure. If the supremum norm of the gradient vector is below
      this number, the algorithm is stopped.
    x_tolerance: Scalar `Tensor` of real dtype. If the absolute change in the
      position between one iteration and the next is smaller than this number,
      the algorithm is stopped.
    f_relative_tolerance: Scalar `Tensor` of real dtype. If the relative change
      in the objective value between one iteration and the next is smaller
      than this value, the algorithm is stopped.
    initial_inverse_hessian_estimate: Optional `Tensor` of the same dtype
      as the components of the output of the `value_and_gradients_function`.
      If specified, the shape should broadcastable to shape `[..., n, n]`; e.g.
      if a single `[n, n]` matrix is provided, it will be automatically
      broadcasted to all batches. Alternatively, one can also specify a
      different hessian estimate for each batch member.
      For the correctness of the algorithm, it is required that this parameter
      be symmetric and positive definite. Specifies the starting estimate for
      the inverse of the Hessian at the initial point. If not specified,
      the identity matrix is used as the starting estimate for the
      inverse Hessian.
    max_iterations: Scalar positive int32 `Tensor`. The maximum number of
      iterations for BFGS updates.
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
    validate_args: Python `bool`, default `True`. When `True` optimizer
      parameters are checked for validity despite possibly degrading runtime
      performance. When `False` invalid inputs may silently render incorrect
      outputs.
    max_line_search_iterations: Python int. The maximum number of iterations
      for the `hager_zhang` line search algorithm.
    f_absolute_tolerance: Scalar `Tensor` of real dtype. If the absolute change
      in the objective value between one iteration and the next is smaller
      than this value, the algorithm is stopped.
    name: (Optional) Python str. The name prefixed to the ops created by this
      function. If not supplied, the default name 'minimize' is used.

  Returns:
    optimizer_results: A namedtuple containing the following items:
      converged: boolean tensor of shape `[...]` indicating for each batch
        member whether the minimum was found within tolerance.
      failed:  boolean tensor of shape `[...]` indicating for each batch
        member whether a line search step failed to find a suitable step size
        satisfying Wolfe conditions. In the absence of any constraints on the
        number of objective evaluations permitted, this value will
        be the complement of `converged`. However, if there is
        a constraint and the search stopped due to available
        evaluations being exhausted, both `failed` and `converged`
        will be simultaneously False.
      num_objective_evaluations: The total number of objective
        evaluations performed.
      position: A tensor of shape `[..., n]` containing the last argument value
        found during the search from each starting point. If the search
        converged, then this value is the argmin of the objective function.
      objective_value: A tensor of shape `[...]` with the value of the
        objective function at the `position`. If the search converged, then
        this is the (local) minimum of the objective function.
      objective_gradient: A tensor of shape `[..., n]` containing the gradient
        of the objective function at the `position`. If the search converged
        the max-norm of this tensor should be below the tolerance.
      inverse_hessian_estimate: A tensor of shape `[..., n, n]` containing the
        inverse of the estimated Hessian.
  """
  with tf.name_scope(name or 'minimize'):
    initial_position = tf.convert_to_tensor(
        initial_position, name='initial_position')
    dtype = dtype_util.base_dtype(initial_position.dtype)
    tolerance = tf.convert_to_tensor(
        tolerance, dtype=dtype, name='grad_tolerance')
    f_relative_tolerance = tf.convert_to_tensor(
        f_relative_tolerance, dtype=dtype, name='f_relative_tolerance')
    x_tolerance = tf.convert_to_tensor(
        x_tolerance, dtype=dtype, name='x_tolerance')
    max_iterations = tf.convert_to_tensor(max_iterations, name='max_iterations')

    input_shape = ps.shape(initial_position)
    batch_shape, domain_size = input_shape[:-1], input_shape[-1]

    if stopping_condition is None:
      stopping_condition = bfgs_utils.converged_all

    # Control inputs are an optional list of tensors to evaluate before
    # the start of the search procedure. These can be used to assert the
    # validity of inputs to the search procedure.
    control_inputs = None

    if initial_inverse_hessian_estimate is None:
      # Create a default initial inverse Hessian.
      initial_inv_hessian = tf.eye(domain_size,
                                   batch_shape=batch_shape,
                                   dtype=dtype,
                                   name='initial_inv_hessian')
    else:
      # If an initial inverse Hessian is supplied, compute some control inputs
      # to ensure that it is positive definite and symmetric.
      initial_inv_hessian = tf.convert_to_tensor(
          initial_inverse_hessian_estimate,
          dtype=dtype,
          name='initial_inv_hessian')
      if validate_args:
        control_inputs = _inv_hessian_control_inputs(initial_inv_hessian)
      else:
        control_inputs = None
      hessian_shape = ps.concat([batch_shape, [domain_size, domain_size]], 0)
      initial_inv_hessian = tf.broadcast_to(initial_inv_hessian, hessian_shape)

    # The `state` here is a `BfgsOptimizerResults` tuple with values for the
    # current state of the algorithm computation.
    def _cond(state):
      """Continue if iterations remain and stopping condition is not met."""
      return ((state.num_iterations < max_iterations) &
              tf.logical_not(stopping_condition(state.converged, state.failed)))

    def _body(state):
      """Main optimization loop."""
      state = bfgs_utils.terminate_if_not_finite(state)
      search_direction = _get_search_direction(state.inverse_hessian_estimate,
                                               state.objective_gradient)
      derivative_at_start_pt = tf.reduce_sum(
          state.objective_gradient * search_direction, axis=-1)

      # If the derivative at the start point is not negative, recompute the
      # search direction with the initial inverse Hessian.
      needs_reset = (~state.failed & ~state.converged &
                     (derivative_at_start_pt >= 0))

      search_direction_reset = _get_search_direction(
          initial_inv_hessian, state.objective_gradient)

      actual_search_direction = tf.where(
          needs_reset[..., tf.newaxis],
          search_direction_reset, search_direction)
      actual_inv_hessian = tf.where(
          needs_reset[..., tf.newaxis, tf.newaxis],
          initial_inv_hessian, state.inverse_hessian_estimate)

      # Replace the hessian estimate in the state, in case it had to be reset.
      current_state = bfgs_utils.update_fields(
          state, inverse_hessian_estimate=actual_inv_hessian)

      next_state = bfgs_utils.line_search_step(
          current_state, value_and_gradients_function, actual_search_direction,
          tolerance, f_relative_tolerance, x_tolerance, stopping_condition,
          max_line_search_iterations, f_absolute_tolerance)

      # Update the inverse Hessian if needed and continue.
      return [_update_inv_hessian(current_state, next_state)]

    kwargs = bfgs_utils.get_initial_state_args(
        value_and_gradients_function,
        initial_position,
        tolerance,
        control_inputs)
    kwargs['inverse_hessian_estimate'] = initial_inv_hessian
    initial_state = BfgsOptimizerResults(**kwargs)
    return tf.while_loop(
        cond=_cond,
        body=_body,
        loop_vars=[initial_state],
        parallel_iterations=parallel_iterations)[0]


def _inv_hessian_control_inputs(inv_hessian):
  """Computes control inputs to validate a provided inverse Hessian.

  These ensure that the provided inverse Hessian is positive definite and
  symmetric.

  Args:
    inv_hessian: The starting estimate for the inverse of the Hessian at the
      initial point.

  Returns:
    A list of tf.Assert ops suitable for use with tf.control_dependencies.
  """
  # The easiest way to validate if the inverse Hessian is positive definite is
  # to compute its Cholesky decomposition.
  try:
    is_positive_definite = tf.reduce_all(
        tf.math.is_finite(tf.linalg.cholesky(inv_hessian)))
  except tf.errors.InvalidArgumentError:
    is_positive_definite = False

  # Then check that the supplied inverse Hessian is symmetric.
  is_symmetric = tf.reduce_all(
      tf.equal(
          bfgs_utils.norm(inv_hessian - _batch_transpose(inv_hessian), dims=2),
          0))

  # Simply adding a control dependencies on these results is not enough to
  # trigger them, we need to add asserts on the results.
  return [
      tf.debugging.Assert(
          is_positive_definite,
          ['Initial inverse Hessian is not positive definite.', inv_hessian]),
      tf.debugging.Assert(
          is_symmetric,
          ['Initial inverse Hessian is not symmetric', inv_hessian])]


def _get_search_direction(inv_hessian_approx, gradient):
  """Computes the direction along which to perform line search."""
  return -tf.linalg.matvec(inv_hessian_approx, gradient)


def _update_inv_hessian(prev_state, next_state):
  """Update the BGFS state by computing the next inverse hessian estimate."""
  # Only update the inverse Hessian if not already failed or converged.
  should_update = ~next_state.converged & ~next_state.failed

  # Compute the normalization term (y^T . s), should not update if is singular.
  gradient_delta = next_state.objective_gradient - prev_state.objective_gradient
  position_delta = next_state.position - prev_state.position
  normalization_factor = tf.reduce_sum(gradient_delta * position_delta, axis=-1)
  should_update = should_update & ~tf.equal(normalization_factor, 0)

  # Rescale the initial hessian at the first step, as suggested
  # in Chapter 6 of Numerical Optimization, by Nocedal and Wright.
  scale_factor = tf.where(
      tf.math.equal(prev_state.num_iterations, 0),
      normalization_factor / tf.reduce_sum(
          tf.math.square(gradient_delta), axis=-1), 1.)

  inverse_hessian_estimate = scale_factor[
      ..., tf.newaxis, tf.newaxis] * prev_state.inverse_hessian_estimate

  def _do_update_inv_hessian():
    next_inv_hessian = _bfgs_inv_hessian_update(
        gradient_delta, position_delta, normalization_factor,
        inverse_hessian_estimate)
    return bfgs_utils.update_fields(
        next_state,
        inverse_hessian_estimate=tf.where(
            should_update[..., tf.newaxis, tf.newaxis],
            next_inv_hessian,
            inverse_hessian_estimate))

  return ps.cond(
      ps.reduce_any(should_update),
      _do_update_inv_hessian,
      lambda: next_state)


def _bfgs_inv_hessian_update(grad_delta, position_delta, normalization_factor,
                             inv_hessian_estimate):
  """Applies the BFGS update to the inverse Hessian estimate.

  The BFGS update rule is (note A^T denotes the transpose of a vector/matrix A).

  ```None
    rho = 1/(grad_delta^T * position_delta)
    U = (I - rho * position_delta * grad_delta^T)
    H_1 =  U * H_0 * U^T + rho * position_delta * position_delta^T
  ```

  Here, `H_0` is the inverse Hessian estimate at the previous iteration and
  `H_1` is the next estimate. Note that `*` should be interpreted as the
  matrix multiplication (with the understanding that matrix multiplication for
  scalars is usual multiplication and for matrix with vector is the action of
  the matrix on the vector.).

  The implementation below utilizes an expanded version of the above formula
  to avoid the matrix multiplications that would be needed otherwise. By
  expansion it is easy to see that one only needs matrix-vector or
  vector-vector operations. The expanded version is:

  ```None
    f = 1 + rho * (grad_delta^T * H_0 * grad_delta)
    H_1 - H_0 = - rho * [position_delta * (H_0 * grad_delta)^T +
                        (H_0 * grad_delta) * position_delta^T] +
                  rho * f * [position_delta * position_delta^T]
  ```

  All the terms in square brackets are matrices and are constructed using
  vector outer products. All the other terms on the right hand side are scalars.
  Also worth noting that the first and second lines are both rank 1 updates
  applied to the current inverse Hessian estimate.

  Args:
    grad_delta: Real `Tensor` of shape `[..., n]`. The difference between the
      gradient at the new position and the old position.
    position_delta: Real `Tensor` of shape `[..., n]`. The change in position
      from the previous iteration to the current one.
    normalization_factor: Real `Tensor` of shape `[...]`. Should be equal to
      `grad_delta^T * position_delta`, i.e. `1/rho` as defined above.
    inv_hessian_estimate: Real `Tensor` of shape `[..., n, n]`. The previous
      estimate of the inverse Hessian. Should be positive definite and
      symmetric.

  Returns:
    A tuple containing the following fields
      is_valid: A Boolean `Tensor` of shape `[...]` indicating batch members
        where the update succeeded. The update can fail if the position change
        becomes orthogonal to the gradient change.
      next_inv_hessian_estimate: A `Tensor` of shape `[..., n, n]`. The next
        Hessian estimate updated using the BFGS update scheme. If the
        `inv_hessian_estimate` is symmetric and positive definite, the
        `next_inv_hessian_estimate` is guaranteed to satisfy the same
        conditions.
  """
  # The quadratic form: y^T.H.y; where H is the inverse Hessian and y is the
  # gradient change.
  conditioned_grad_delta = tf.linalg.matvec(inv_hessian_estimate, grad_delta)
  conditioned_grad_delta_norm = tf.reduce_sum(
      conditioned_grad_delta * grad_delta, axis=-1)

  # The first rank 1 update term requires the outer product: s.y^T.
  cross_term = _tensor_product(position_delta, conditioned_grad_delta)

  def _expand_scalar(s):
    # Expand dimensions of a batch of scalars to multiply or divide a matrix.
    return s[..., tf.newaxis, tf.newaxis]

  # Symmetrize
  cross_term += _tensor_product(conditioned_grad_delta, position_delta)
  position_term = _tensor_product(position_delta, position_delta)
  with tf.control_dependencies([position_term]):
    position_term *= _expand_scalar(
        1 + conditioned_grad_delta_norm / normalization_factor)

  return (inv_hessian_estimate +
          (position_term - cross_term) / _expand_scalar(normalization_factor))


def _tensor_product(t1, t2):
  """Computes the outer product of two possibly batched vectors.

  Args:
    t1: A `tf.Tensor` of shape `[..., n]`.
    t2: A `tf.Tensor` of shape `[..., m]`.

  Returns:
    A tensor of shape `[..., n, m]` with matching batch dimensions, let's call
    it `r`, whose components are:

    ```None
      r[..., i, j] = t1[..., i] * t2[..., j]
    ```
  """
  return tf.linalg.matmul(t1[..., tf.newaxis], t2[..., tf.newaxis, :])


def _batch_transpose(mat):
  """Transpose a possibly batched matrix.

  Args:
    mat: A `tf.Tensor` of shape `[..., n, m]`.

  Returns:
    A tensor of shape `[..., m, n]` with matching batch dimensions.
  """
  n = distribution_util.prefer_static_rank(mat)
  perm = tf.range(n)
  perm = tf.concat([perm[:-2], [perm[-1], perm[-2]]], axis=0)
  return tf.transpose(a=mat, perm=perm)
