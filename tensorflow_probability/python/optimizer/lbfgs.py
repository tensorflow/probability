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
"""A constrained version of the Limited-Memory BFGS minimization algorithm.

Limited-memory quasi-Newton methods are useful for solving large problems
whose Hessian matrices cannot be computed at a reasonable cost or are not
sparse. Instead of storing fully dense n x n approximations of Hessian
matrices, they only save a few vectors of length n that represent the
approximations implicitly.

This module implements the algorithm known as L-BFGS-B, which, as its name
suggests, is a limited-memory version of the BFGS algorithm, with bounds.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

# Dependency imports
import tensorflow.compat.v2 as tf

from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import prefer_static as ps
from tensorflow_probability.python.optimizer import bfgs_utils
from tensorflow_probability.python.optimizer import lbfgs_minimize


LBfgsBOptimizerResults = collections.namedtuple(
  'LBfgsBOptimizerResults', [
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
    'lower_bounds',  # A tensor containing the lower bounds to the constrained
    # optimization, cast to the shape of `position`.
    'upper_bounds',  # A tensor containing the upper bounds to the constrained
    # optimization, cast to the shape of `position`.
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
    'history',  # How many gradient/position deltas should be considered.
  ])

_ConstrainedCauchyState = collections.namedtuple(
  '_CauchyMinimizationResult', [
    # `\theta` in [2]; n the Cauchy search, relates to the implicit Hessian
    'theta',
    # `B = \theta*I - WMW'` (`I` the identity, see [1,2] for details)
    # `M_k` matrix in [2]; part of the implicit representation of the Hessian,
    'm',
    # see the comment above
    'breakpoints',  # `t_i` in [Byrd et al.][2];
    # the breakpoints in the branch definition of the
    # projection of the gradients, batched
    'breakpoints_argsort',  # Range from 0...n-1 sorted by increasing breakpoints
    # Tensor of shape [batch]; the index into `breakpoints_argsort`
    'next_free_idx',
    # for the breakpoint in effect
    'steepest',  # `d` in [2]; steepest descent clamped to bounds
    'p',  # as in [2]
    # as in [2]; eventually made to equal `W'(cauchy_point - position)`
    'c',
    'df',  # `f'` in [2]
    'ddf',  # `f''` in [2]
    'dt',  # `\Delta t` in [2]
    'dt_min',  # `\Delta t_min` in [2]
    'tsum',  # Sum of all the considered breakpoints so far
    'breakpoint_min_old',  # t_old in [2]
    # `x^cp` in [2]; the actual cauchy point (we're looking for)
    'cauchy_point',
    'active',  # What batches are in active optimization
    'free_mask',  # Boolean tensor of what variables are actively constrained
  ])


def minimize(value_and_gradients_function,
       initial_position,
       bounds=None,
       previous_optimizer_results=None,
       num_correction_pairs=10,
       tolerance=1e-5,
       x_tolerance=0,
       f_relative_tolerance=0,
       initial_inverse_hessian_estimate=None,
       max_iterations=50,
       parallel_iterations=1,
       stopping_condition=None,
       max_line_search_iterations=50,
       name=None):
  """Applies the L-BFGS-B algorithm to minimize a differentiable function.

  Performs optionally constrained minimization of a differentiable function using the
  L-BFGS-B scheme. See [Nocedal and Wright(2006)][1] for details on the unconstrained
  version, and [Byrd et al.][2] for details on the constrained algorithm.

  ### Usage:

  The following example demonstrates the L-BFGS-B optimizer attempting to find the
  constrained minimum for a simple high-dimensional quadratic objective function.

  ```python
  ndims = 60
  minimum = tf.convert_to_tensor(
    np.ones([ndims]), dtype=tf.float32)
  lower_bounds = tf.convert_to_tensor(
    np.arange(ndims), dtype=tf.float32)
  upper_bounds = tf.convert_to_tensor(
    np.arange(100, 100-ndims, -1), dtype=tf.float32)
  scales = tf.convert_to_tensor(
    (np.random.rand(ndims) + 1.)*5. + 1., dtype=tf.float32)
  start = tf.constant(np.random.rand(2, ndims)*100, dtype=tf.float32)

  # The objective function and the gradient.
  def quadratic_loss_and_gradient(x):
    return tfp.math.value_and_gradient(
      lambda x: tf.reduce_sum(
        scales * tf.math.squared_difference(x, minimum), axis=-1),
      x)
  opt_results = tfp.optimizer.lbfgsb_minimize(
          quadratic_loss_and_gradient,
          initial_position=start,
          num_correction_pairs=10,
          tolerance=1e-10,
          bounds=[lower_bounds, upper_bounds])
  ```

  ### References:

  [1] Jorge Nocedal, Stephen Wright. Numerical Optimization. Springer Series
    in Operations Research. pp 176-180. 2006

  http://pages.mtu.edu/~struther/Courses/OLD/Sp2013/5630/Jorge_Nocedal_Numerical_optimization_267490.pdf

  [2] Richard H. Byrd, Peihuang Lu, Jorge Nocedal, & Ciyou Zhu (1995).
    A Limited Memory Algorithm for Bound Constrained Optimization
    SIAM Journal on Scientific Computing, 16(5), 1190–1208.

  https://doi.org/10.1137/0916069

  [3] Jose Luis Morales, Jorge Nocedal (2011).
    "Remark On Algorithm 788: L-BFGS-B: Fortran Subroutines for Large-Scale
      Bound Constrained Optimization"
    ACM Trans. Math. Softw. 38, 1, Article 7.

  https://dl.acm.org/doi/abs/10.1145/2049662.2049669

  Args:
    value_and_gradients_function:  A Python callable that accepts a point as a
    real `Tensor` and reporting arguments, and returns a tuple of `Tensor`s of
    real dtype containing the value of the function and its gradient at that
    point. The function to be minimized. The input is of shape `[..., n]`,
    where `n` is the size of the domain of input points, and all others are
    batching dimensions. The first component of the return value is a real
    `Tensor` of matching shape `[...]`. The second component (the gradient) is
    also of shape `[..., n]` like the input value to the function.
    The reporting arguments consist of a Boolean `Tensor` of shape `[...]`
    denoting which batches have terminated, and two real `Tensor` of shape
    `[..., n]`, denoting the last evaluated objective values and gradients
    (respectively). 
    initial_position: Real `Tensor` of shape `[..., n]`. The starting point, or
    points when using batching dimensions, of the search procedure. At these
    points the function value and the gradient norm should be finite.
    Exactly one of `initial_position` and `previous_optimizer_results` can be
    non-None.
    bounds: Tuple of two real `Tensor`s of shape `[..., n]`. The first element
    indicates the lower bounds in the constrained optimization, and the second
    element of the tuple indicates the upper bounds of the optimization. If
    `bounds` is `None`, the optimization is deferred to the unconstrained
    version (see also `lbfgs_minimize`). If one of the elements of the tuple
    is `None`, the optimization is assumed to be unconstrained (from above/below,
    respectively). 
    previous_optimizer_results: An `LBfgsBOptimizerResults` namedtuple to
    intialize the optimizer state from, instead of an `initial_position`.
    This can be passed in from a previous return value to resume optimization
    with a different `stopping_condition`. Exactly one of `initial_position`
    and `previous_optimizer_results` can be non-None.
    num_correction_pairs: Positive integer. Specifies the maximum number of
    (position_delta, gradient_delta) correction pairs to keep as implicit
    approximation of the Hessian matrix
    A real `Tensor` of the same shape as the `state.position`, of dtype `bool`,
    denoting a mask over the free variables.x.
    tolerance: Scalar `Tensor` of real dtype. Specifies the gradient tolerance
    for the procedure. If the supremum norm of the gradient vector is below
    this number, the algorithm is stopped.
    x_tolerance: Scalar `Tensor` of real dtype. If the absolute change in the
    position between one iteration and the next is smaller than this number,
    the algorithm is stopped.
    f_relative_tolerance: Scalar `Tensor` of real dtype. If the relative change
    in the objective value between one iteration and the next is smaller
    than this value, referenced to the current objective value, the previous
    objective value, or `1`, whichever is greatest, the algorithm is stopped.
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

  if len(bounds) != 2:
    raise ValueError(
      '`bounds` parameter has unexpected number of elements '
      '(expected 2).')

  lower_bounds, upper_bounds = bounds

  # Defer further conversion of the bounds to appropriate tensors
  # until the shape of the input is known

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
      # Force at least one batching dimension
      if len(ps.shape(initial_position)) == 1:
        initial_position = initial_position[tf.newaxis, :]
      position_shape = ps.shape(initial_position)
      dtype = dtype_util.base_dtype(initial_position.dtype)

    if previous_optimizer_results is not None:
      position_shape = ps.shape(previous_optimizer_results.position)
      dtype = dtype_util.base_dtype(
        previous_optimizer_results.position.dtype)

    # TODO: This isn't agnostic to the number of batch dimensions, it only
    #  supports one batch dimension, but I've found RaggedTensors to be far
    #  too finicky/undocumented to handle multiple batch dimensions in any
    #  sane way. (Even the way it's working so far is less than ideal.)
    if len(position_shape) > 2:
      raise NotImplementedError(
        "More than a batch dimension is not implemented. "
        "Consider flattening and then reshaping the results.")
    # NOTE: Broadcasting the batched dimensions breaks when there are no
    #  batched dimensions. Although this isn't handled like this in
    #  `lbfgs.py`, I'd rather force a batch dimension with a single
    #  element than do conditional checks later.
    if len(position_shape) == 1:
      position_shape = tf.concat([[1], position_shape], axis=0)
      initial_position = tf.broadcast_to(
        initial_position, position_shape)

    # NOTE: Could maybe use bfgs_utils._broadcast here, but would have to check
    #  that the non-batching dimensions also match; using `tf.broadcast_to` has
    #  the advantage that passing a (1,)-shaped tensor as bounds will correctly
    #  bound every variable at the single value.
    if lower_bounds is None:
      lower_bounds = tf.constant(
        [-float('inf')], shape=position_shape, dtype=dtype, name='lower_bounds')
    else:
      lower_bounds = tf.cast(
        tf.convert_to_tensor(lower_bounds), dtype=dtype)
      try:
        lower_bounds = tf.broadcast_to(
          lower_bounds, position_shape, name='lower_bounds')
      except tf.errors.InvalidArgumentError:
        raise ValueError(
          'Failed to broadcast lower bounds tensor to the shape of starting '
          'position. Are the lower bounds well formed?')
    if upper_bounds is None:
      upper_bounds = tf.constant(
        [float('inf')], shape=position_shape, dtype=dtype, name='upper_bounds')
    else:
      upper_bounds = tf.cast(
        tf.convert_to_tensor(upper_bounds), dtype=dtype)
      try:
        upper_bounds = tf.broadcast_to(
          upper_bounds, position_shape, name='upper_bounds')
      except tf.errors.InvalidArgumentError:
        raise ValueError(
          'Failed to broadcast upper bounds tensor to the shape of starting '
          'position. Are the lower bounds well formed?')

    # Clamp the starting position to the bounds, because the algorithm expects
    # the variables to be in range for the Hessian inverse estimation, but also
    # because that fast-tracks the first iteration of the Cauchy optimization.
    initial_position = tf.clip_by_value(
      initial_position, lower_bounds, upper_bounds)

    tolerance = tf.convert_to_tensor(
      tolerance, dtype=dtype, name='grad_tolerance')
    f_relative_tolerance = tf.convert_to_tensor(
      f_relative_tolerance, dtype=dtype, name='f_relative_tolerance')
    x_tolerance = tf.convert_to_tensor(
      x_tolerance, dtype=dtype, name='x_tolerance')
    max_iterations = tf.convert_to_tensor(
      max_iterations, name='max_iterations')

    # The `state` here is a `LBfgsBOptimizerResults` tuple with values for the
    # current state of the algorithm computation.
    def _cond(state):
      """Continue if iterations remain and stopping condition is not met."""
      return ((state.num_iterations < max_iterations) &
          tf.logical_not(stopping_condition(state.converged, state.failed)))

    def _body(current_state):
      """Main optimization loop."""
      current_state = bfgs_utils.terminate_if_not_finite(current_state)
      cauchy_state, current_state = _cauchy_minimization(
        current_state, num_correction_pairs, parallel_iterations)

      search_direction, current_state, clip_before, refreshed = (
        _find_search_direction(
          current_state, cauchy_state, num_correction_pairs))

      # If any batch needs a refresh, restart the whole thing, to reduce number
      # of function evaluations

      def _continue_minimization():
        """Proceeds with minimization iteration."""
        next_state = _constrained_line_search_step(
          current_state, value_and_gradients_function, search_direction,
          tolerance, f_relative_tolerance, x_tolerance, stopping_condition,
          max_line_search_iterations, clip_before)

        # If not failed or converged, update the Hessian estimate.
        # Only do this if the new pairs obey the s.y > eps.||y||
        position_delta = (next_state.position - current_state.position)
        gradient_delta = (next_state.objective_gradient -
                  current_state.objective_gradient)
        # Article is ambiguous; see lbfgs.f:863
        curvature_cond = (
          tf.reduce_sum(position_delta * gradient_delta, axis=-1) >=
          bfgs_utils.norm(current_state.objective_gradient, dims=1) *
          dtype_util.eps(position_delta.dtype))
        should_push = (~(next_state.converged | next_state.failed) &
                 curvature_cond & ~refreshed)
        # TODO: Track number of skipped pairs
        new_position_deltas = _queue_push(
          next_state.position_deltas, should_push, position_delta)
        new_gradient_deltas = _queue_push(
          next_state.gradient_deltas, should_push, gradient_delta)
        new_history = tf.where(
          should_push,
          tf.math.minimum(next_state.history + 1,
                  num_correction_pairs),
          next_state.history)

        if not tf.executing_eagerly():
          # Hint the compiler that the shape of the properties has not changed
          new_position_deltas = tf.ensure_shape(
            new_position_deltas, next_state.position_deltas.shape)
          new_gradient_deltas = tf.ensure_shape(
            new_gradient_deltas, next_state.gradient_deltas.shape)
          new_history = tf.ensure_shape(
            new_history, next_state.history.shape)

        next_state = bfgs_utils.update_fields(
          next_state,
          position_deltas=new_position_deltas,
          gradient_deltas=new_gradient_deltas,
          history=new_history)

        return [next_state]

      return tf.cond(
        pred=tf.reduce_any(refreshed),
        true_fn=lambda: [current_state],
        false_fn=_continue_minimization)

    if previous_optimizer_results is None:
      assert initial_position is not None
      initial_state = _get_initial_state(value_and_gradients_function,
                         initial_position,
                         lower_bounds,
                         upper_bounds,
                         num_correction_pairs,
                         tolerance)
    else:
      initial_state = previous_optimizer_results

    return tf.while_loop(
      cond=_cond,
      body=_body,
      loop_vars=[initial_state],
      parallel_iterations=parallel_iterations)[0]


def _cauchy_minimization(bfgs_state, num_correction_pairs, parallel_iterations):
  """Calculates the Cauchy point, bounding the gradient by the bounds.

  This function minimizes the quadratic approximation to the objective
  function at the current position, in the direction of steepest descent,
  but bounding the gradient by the corresponding bounds.

  See algorithm CP and associated discussion of [Byrd,Lu,Nocedal,Zhu][2]
  for details.

  This function may modify the given `bfgs_state`, in that it refreshes the
  memory for batches that are found to be in an invalid state.

  Args:
    bfgs_state: current `LBfgsBOptimizerResults` state
    num_correction_pairs: the (maximum) number of past steps to keep as
    history for the LBFGS algorithm
    parallel_iterations: argument of `tf.while` loops
  Returns:
    A `_CauchyMinimizationResult` containing the results of the Cauchy point
    computation.
    Updated `bfgs_state`
  """
  cauchy_state, bfgs_state = _get_initial_cauchy_state(
    bfgs_state, num_correction_pairs)
  n = ps.shape(bfgs_state.position)[-1]
  idx_range = tf.range(ps.shape(bfgs_state.position)[-1])[tf.newaxis, ...]
  # NOTE: See lbfgsb.f (l. 1524)
  ddf_org = -cauchy_state.theta * cauchy_state.df

  def _cond(state):
    """Test convergence to Cauchy point at current branch"""
    return tf.reduce_any(state.active)

  def _body(state):
    """Cauchy point iterative loop (While loop of CP algorithm [2])"""
    # Because of `where` statements, the indices for gathering must always
    # be valid, even if the result is not used afterwards. For batches that
    # are no longer active, the `next_free_idx` (which points to the index
    # of the current minimum breakpoint via `breakpoints_argsort`) may
    # exceed the size of `breakpoints_argsort` (if the batch isn't active
    # because there are no free variables left). So, instead, we take 0 as a
    # dummy value, which will later be discarded by the `where` statements.
    next_free_idx = tf.where(
      state.active,
      state.next_free_idx,
      0)
    breakpoint_min_idx = tf.where(
      state.active,
      tf.gather(
        state.breakpoints_argsort,
        next_free_idx,
        batch_dims=1),
      0)
    breakpoint_min = tf.where(
      state.active,
      tf.gather(
        state.breakpoints,
        breakpoint_min_idx,
        batch_dims=1),
      state.breakpoint_min_old)

    dt = (breakpoint_min - state.breakpoint_min_old)

    # NOTE: We immediately update active to simulate an early return
    # This value should be used below (instead of `state.active`)
    active = (state.active & (state.dt_min >= dt))

    # Set the considered variable as fixed
    tsum = tf.where(active, state.tsum + dt, state.tsum)
    breakpoint_min_idx_mask = (
      idx_range == breakpoint_min_idx[..., tf.newaxis])
    steepest = tf.where(
      active[..., tf.newaxis],
      tf.where(
        breakpoint_min_idx_mask,
        0.,
        state.steepest),
      state.steepest)
    free_mask = tf.where(
      active[..., tf.newaxis],
      (state.free_mask & ~breakpoint_min_idx_mask),
      state.free_mask)
    d_b = tf.gather(
      state.steepest,
      breakpoint_min_idx,
      batch_dims=1)
    x_cp_b = tf.gather(
      tf.where(
        (d_b > 0.)[..., tf.newaxis],
        bfgs_state.upper_bounds,
        tf.where(
            (d_b < 0.)[..., tf.newaxis],
          bfgs_state.lower_bounds,
          state.cauchy_point
        )),
      breakpoint_min_idx,
      batch_dims=1)
    cauchy_point = tf.where(
      active[..., tf.newaxis],
      tf.where(
        breakpoint_min_idx_mask,
        x_cp_b[..., tf.newaxis],
        state.cauchy_point),
      state.cauchy_point)

    # If we're out of free variables, set dt_min to dt and "return"
    next_free_idx = tf.where(active, next_free_idx + 1, next_free_idx)
    no_more_free = (next_free_idx >= n)
    dt_min = tf.where(no_more_free, dt, state.dt_min)
    active &= ~no_more_free

    # Update remaining properties
    # - Update `c`
    c = tf.where(
      active[..., tf.newaxis],
      state.c + dt[..., tf.newaxis] * state.p,
      state.c)
    # - Get the `b`th row of W (needed for f', f'')
    # The matrix M has shape
    #
    #  [[ 0  0   ]
    #   [ 0  M_h ]]
    #
    # where M_h is the M matrix considering the current history `h`.
    # Therefore, for W, we should consider that the last `h` columns
    #  are
    #     Y[k-h,...,k-1] theta*S[k-h,...k-1]
    #         (so that the first `2*(m-h)` columns are 0.
    # 1. Create the "full" W matrix row
    w_b = tf.concat(
      [tf.gather(
        bfgs_state.gradient_deltas,
        breakpoint_min_idx,
        axis=-1,
        batch_dims=1),
       (state.theta[..., tf.newaxis] *
        tf.gather(
         bfgs_state.position_deltas,
         breakpoint_min_idx,
         axis=-1,
         batch_dims=1))
       ],
      axis=-1)
    # 2. "Permute" the relevant items to the right
    idx = tf.concat(
      [
        tf.ragged.range(
          num_correction_pairs - bfgs_state.history),
        tf.ragged.range(
          num_correction_pairs,
          2*num_correction_pairs - bfgs_state.history),
        tf.ragged.range(
          num_correction_pairs - bfgs_state.history,
          num_correction_pairs),
        tf.ragged.range(
          2*num_correction_pairs - bfgs_state.history,
          2*num_correction_pairs)
      ],
      axis=-1).to_tensor()
    w_b = tf.gather(
      w_b,
      idx,
      batch_dims=1)

    # - Update f'
    x_b = tf.gather(
      bfgs_state.position,
      breakpoint_min_idx,
      batch_dims=1)
    # NOTE Use of d_b = -g_b
    df = tf.where(
      active,
      (state.df + dt * state.ddf +
       d_b**2 -
       state.theta * d_b * (x_cp_b - x_b) +
       d_b * tf.einsum(
         '...j,...jk,...k->...',
         w_b,
         state.m,
         c)),
      state.df)

    # - Update f''
    # NOTE use of d_b = -g_b
    ddf = tf.where(
      active,
      (state.ddf - state.theta * d_b**2 +
       2. * d_b * tf.einsum(
         "...i,...ij,...j->...",
         w_b,
         state.m,
         state.p) -
       d_b**2 * tf.einsum(
         "...i,...ij,...j->...",
         w_b,
         state.m,
         w_b)),
      state.ddf)
    # NOTE: See lbfgsb.f (l. 1649)
    ddf = tf.where(
      active,
      tf.math.maximum(ddf, dtype_util.eps(ddf.dtype)*ddf_org),
      state.ddf)

    # - Update p
    # NOTE use of d_b = -g_b
    p = tf.where(
      active[..., tf.newaxis],
      state.p - d_b[..., tf.newaxis] * w_b,
      state.p)

    # - Update dt_min
    dt_min = tf.where(
      active, -tf.math.divide_no_nan(df, ddf), state.dt_min)

    # Create the updated state

    # We need to hint the compiler that nothing changed shapes
    if not tf.executing_eagerly():
      steepest = tf.ensure_shape(steepest, state.steepest.shape)
      p = tf.ensure_shape(p, state.p.shape)
      c = tf.ensure_shape(c, state.c.shape)
      df = tf.ensure_shape(df, state.df.shape)
      ddf = tf.ensure_shape(ddf, state.ddf.shape)
      dt = tf.ensure_shape(dt, state.dt.shape)
      dt_min = tf.ensure_shape(dt_min, state.dt_min.shape)
      tsum = tf.ensure_shape(tsum, state.tsum.shape)
      breakpoint_min = tf.ensure_shape(
        breakpoint_min, state.breakpoint_min_old.shape)
      next_free_idx = tf.ensure_shape(
        next_free_idx, state.next_free_idx.shape)
      cauchy_point = tf.ensure_shape(
        cauchy_point, state.cauchy_point.shape)
      free_mask = tf.ensure_shape(free_mask, state.free_mask.shape)
      active = tf.ensure_shape(active, state.active.shape)

    new_state = bfgs_utils.update_fields(
      state, steepest=steepest, p=p, c=c, df=df, ddf=ddf, dt=dt,
      dt_min=dt_min, tsum=tsum, breakpoint_min_old=breakpoint_min,
      next_free_idx=next_free_idx, cauchy_point=cauchy_point,
      free_mask=free_mask, active=active)

    return [new_state]

  cauchy_loop = tf.while_loop(
    cond=_cond,
    body=_body,
    loop_vars=[cauchy_state],
    parallel_iterations=parallel_iterations)[0]

  # NOTE: See lbfgs.f lines 1584, 1606, 1667, 1682
  free_remaining = (cauchy_loop.next_free_idx < n)
  dt_min = tf.where(
    free_remaining,
    tf.math.maximum(cauchy_loop.dt_min, 0),
    cauchy_loop.dt_min)
  tsum = tf.where(
    free_remaining,
    cauchy_loop.tsum + dt_min,
    cauchy_loop.tsum)

  cauchy_point = tf.where(
    (bfgs_state.converged | bfgs_state.failed)[..., tf.newaxis],
    bfgs_state.position,
    tf.where(
      free_remaining[..., tf.newaxis],
      cauchy_loop.cauchy_point +
      tsum[..., tf.newaxis] * cauchy_loop.steepest,
      cauchy_loop.cauchy_point))

  c = cauchy_loop.c + dt_min[..., tf.newaxis]*cauchy_loop.p
  # NOTE: `c` is already permuted to match the subspace of `M`, because `w_b`
  #  was already permuted.
  # You can explicitly check this by comparing its value with W'.(x^c - x)
  #  at this point.

  # Set points where gradient is 0 as fixed
  # TODO: Does this cause problems with sadle points?
  free_mask = (cauchy_loop.free_mask & (bfgs_state.objective_gradient != 0))

  # Hint the compiler that shape of things will not change
  if not tf.executing_eagerly():
    dt_min = tf.ensure_shape(dt_min, cauchy_loop.dt_min.shape)
    tsum = tf.ensure_shape(tsum, cauchy_loop.tsum.shape)
    cauchy_point = tf.ensure_shape(
      cauchy_point, cauchy_loop.cauchy_point.shape)
    c = tf.ensure_shape(c, cauchy_loop.c.shape)
    free_mask = tf.ensure_shape(free_mask, cauchy_loop.free_mask.shape)
  # Do the actual updating
  final_cauchy_state = bfgs_utils.update_fields(
    cauchy_loop, dt_min=dt_min, tsum=tsum, cauchy_point=cauchy_point, c=c,
    free_mask=free_mask)

  return final_cauchy_state, bfgs_state


def _get_initial_cauchy_state(bfgs_state, num_correction_pairs):
  """Create `_ConstrainedCauchyState` with initial parameters.

  This will calculate the elements of `_ConstrainedCauchyState` based on the
  given `LBfgsBOptimizerResults` state object. Some of these properties may be
  incalculable, for which batches the state will be reset.

  Args:
    bfgs_state: `LBfgsBOptimizerResults` object representing the current state
    of the LBFGSB optimization
    num_correction_pairs: typically `m`; the (maximum) number of past steps to
    keep as history for the LBFGS algorithm

  Returns:
    Initialized `_ConstrainedCauchyState`
    Updated `bfgs_state`
  """
  cauchy_point = bfgs_state.position

  theta = tf.math.divide_no_nan(
    tf.reduce_sum(bfgs_state.gradient_deltas[..., -1, :]**2, axis=-1),
    (tf.reduce_sum(bfgs_state.gradient_deltas[..., -1, :] *
             bfgs_state.position_deltas[..., -1, :], axis=-1)))
  theta = tf.where(bfgs_state.history == 0, 1., theta)

  m, refresh = _cauchy_init_m(bfgs_state, theta, num_correction_pairs)

  # Erase the history where M isn't invertible
  bfgs_state = _erase_history(bfgs_state, refresh)
  theta = tf.where(refresh, 1., theta)

  breakpoints = _cauchy_init_breakpoints(bfgs_state)
  breakpoints_argsort = tf.argsort(breakpoints)

  steepest = tf.where((breakpoints > 0.), -bfgs_state.objective_gradient, 0.)

  # We need to account for the varying histories:
  # we assume that the first `2*(m-h)` rows of W'^T
  # are 0 (where `m` is the number of correction pairs
  # and `h` is the history), in concordance with the first
  # `2*(m-h)` rows of M being 0.
  # 1. Calculate all elements
  p = tf.concat(
    [
      tf.einsum(
        "...mi,...i->...m",
        bfgs_state.gradient_deltas,
        steepest),
      (theta[..., tf.newaxis] *
       tf.einsum(
        "...mi,...i->...m",
        bfgs_state.position_deltas,
        steepest))
    ],
    axis=-1)
  # 2. Assemble the rows in the correct order
  idx = tf.concat(
    [
      tf.ragged.range(
        num_correction_pairs - bfgs_state.history),
      tf.ragged.range(
        num_correction_pairs,
        2*num_correction_pairs - bfgs_state.history),
      tf.ragged.range(
        num_correction_pairs - bfgs_state.history,
        num_correction_pairs),
      tf.ragged.range(
        2*num_correction_pairs - bfgs_state.history,
        2*num_correction_pairs)
    ],
    axis=-1).to_tensor()
  p = tf.gather(
    p,
    idx,
    batch_dims=1)

  c = tf.zeros_like(p)
  df = -tf.reduce_sum(steepest**2, axis=-1)
  ddf = -theta*df - tf.einsum("...i,...ij,...j->...", p, m, p)
  dt_min = -tf.math.divide_no_nan(df, ddf)
  tsum = tf.zeros_like(dt_min)

  # NOTE: These are placeholder values.
  # All of these have shape [batch], which matches dt_min
  dt = tf.zeros_like(dt_min)
  breakpoint_min_old = tf.zeros_like(dt_min)

  next_free_idx = tf.reduce_sum(tf.where(breakpoints <= 0., 1, 0), axis=-1)
  free_mask = (breakpoints > 0.)

  # NOTE: _cauchy_update_active should NOT be accounted for here; the first
  # iteration should always run (if the batch is overall active)
  active = ~(bfgs_state.converged | bfgs_state.failed)

  cauchy_state = _ConstrainedCauchyState(
    theta=theta, m=m, breakpoints=breakpoints,
    breakpoints_argsort=breakpoints_argsort, next_free_idx=next_free_idx,
    steepest=steepest, p=p, c=c, df=df, ddf=ddf, dt=dt, dt_min=dt_min,
    tsum=tsum, breakpoint_min_old=breakpoint_min_old, cauchy_point=cauchy_point,
    active=active, free_mask=free_mask)

  return cauchy_state, bfgs_state


def _cauchy_init_breakpoints(state):
  """Calculate the breakpoints for a `_CauchyMinimizationResult` state."""
  breakpoints = (
    tf.where(
      state.objective_gradient < 0,
      tf.math.divide_no_nan(
        state.position - state.upper_bounds,
        state.objective_gradient),
      tf.where(
        state.objective_gradient > 0,
        tf.math.divide_no_nan(
          state.position - state.lower_bounds,
          state.objective_gradient),
        float('inf')))
  )

  return breakpoints


def _find_search_direction(bfgs_state, cauchy_state, num_correction_pairs):
  """Finds the search direction based on the direct primal method.

  This function corresponds to points 1-6 of the Direct Primal Method presented
  in [2, p. 1199] for subspace minimization, with the first modification
  suggested in [3].

  If an invalid condition is reached for a given batch, its history is reset.
  Therefore, this function also returns an updated `bfgs_state`. 

  Args:
    bfgs_state: the `LBfgsBOptimizerResults` object representing the current
      iteration.
    cauchy_state: the `_CauchyMinimizationResult` results of a cauchy search
      computation. Typically the output of `_cauchy_minimization`.
    num_correction_pairs: The (maximum) number of correction pairs stored in
      memory (`m`)
  Returns:
    Tensor of batched search directions,
    Updated `bfgs_state`,
    Tensor of Boolean dtype indicating whether the search direction should be
      clamped to bounds before the search is performed,
    Tensor of Boolean dtype indicating what batches have been refreshed.
  """
  def _find_constrained_minimizer():
    """Performs free subspace minimization based on the Direct Method."""
    # Let the reduced gradient be [2, eq. 5.4]
    #
    #     ρ = Z'r
    #     r = g + Θ(x^c - x) + (1/Θ).W.M.c
    #
    # and the search direction [2, eq. 5.7]
    #
    #     d = -B⁻¹ρ
    #
    # and [2, eq. 5.10]
    #
    #     B⁻¹ = 1/Θ [ I + 1/Θ Z'.W.N⁻¹.M.W'.Z ]
    #     N   = I - 1/Θ M.W'.Z.Z'.W
    #
    # Therefore,
    #
    #     d = Z' . (-1/Θ) . [ r + 1/Θ W.N⁻¹.M.W'.Z.Z'.r ]
    #
    # NOTE that the leading sign does not match that of [2, eq. 5.11]. This is
    # because the article conflates the definition of r in [2, eq. 5.4] and the
    # definition employed in the Fortran implementation, where
    #
    #    r = -Z'B(x^c - x) - Z'g
    #
    # From which follows
    #
    #    d = Z'  (1/Θ) . [ r + 1/Θ W.N⁻¹.M.W'.Z.Z'.r ]
    idx = (
      tf.concat([
        tf.ragged.range(
          num_correction_pairs - bfgs_state.history),
        tf.ragged.range(
          num_correction_pairs,
          2*num_correction_pairs - bfgs_state.history),
        tf.ragged.range(
          num_correction_pairs - bfgs_state.history,
          num_correction_pairs),
        tf.ragged.range(
          2*num_correction_pairs - bfgs_state.history,
          2*num_correction_pairs)
      ],
        axis=-1).to_tensor())

    w_transpose = (
      tf.gather(
        tf.concat(
          [bfgs_state.gradient_deltas,
           cauchy_state.theta[..., tf.newaxis, tf.newaxis] *
           bfgs_state.position_deltas],
          axis=-2),
        idx,
        batch_dims=1)
    )

    r = (
      cauchy_state.theta[..., tf.newaxis] *
      (bfgs_state.position - cauchy_state.cauchy_point) +
      tf.einsum(
        '...ji,...jk,...k->...i',
        w_transpose,
        cauchy_state.m,
        cauchy_state.c) -
      bfgs_state.objective_gradient)

    n = (
      tf.eye(
        num_rows=num_correction_pairs*2,
        batch_shape=ps.shape(bfgs_state.position)[:-1]) -
      (tf.einsum(
        '...ij,...jk,...lk->...il',
        cauchy_state.m,
        w_transpose,
        tf.where(
          cauchy_state.free_mask[..., tf.newaxis, :],
          w_transpose,
          0.)
      ) / cauchy_state.theta[..., tf.newaxis, tf.newaxis]))

    # NOTE: No need to "mask" the no-history subspace of N: because of I - (...)
    # we correctly get a block form. The extraneous identity block is then
    # zeroed when the product with M is taken
    refresh = (tf.linalg.det(n) == 0.)

    n = tf.linalg.inv(
      tf.where(
        refresh[..., tf.newaxis, tf.newaxis],
        tf.eye(
          num_rows=num_correction_pairs*2,
          batch_shape=ps.shape(bfgs_state.position)[:-1]),
        n))

    n = tf.where(
      refresh[..., tf.newaxis, tf.newaxis],
      tf.zeros_like(n),
      n)

    # d is composed in three parts
    d = tf.einsum('...ji,...jk,...kl,...lm,...m->...i',
            w_transpose,
            n,
            cauchy_state.m,
            tf.where(
              cauchy_state.free_mask[..., tf.newaxis, :],
              w_transpose,
              0.),
            r)

    d = r + d/cauchy_state.theta[..., tf.newaxis]
    d = d/cauchy_state.theta[..., tf.newaxis]

    d = tf.where(cauchy_state.free_mask, d, 0.)

    # Per [3]:
    # Project `(cauchy point) + d` into the bounds
    # NOTE: `d` is zeroed for constrained variables, and `movement_clip` is
    # at most 1.
    minimizer = tf.clip_by_value(
      cauchy_state.cauchy_point + d,
      bfgs_state.lower_bounds,
      bfgs_state.upper_bounds)

    # Per [3]: If the search direction obtained with this minimizer is not a
    # direction of strong descent, allow the minimizer to be oob, and clip the
    # direction (i.e. fall back to the original algorithm). The clipping is
    # handled outside this fn.
    fallback = (tf.reduce_sum((minimizer - bfgs_state.position) *
                  bfgs_state.objective_gradient, axis=-1) > 0)

    minimizer = tf.where(
      fallback[..., tf.newaxis],
      cauchy_state.cauchy_point + d,
      minimizer)

    active = (tf.reduce_any(cauchy_state.free_mask, axis=-1) &
          (bfgs_state.history > 0))
    minimizer = tf.where(
      active[..., tf.newaxis], minimizer, cauchy_state.cauchy_point)

    return minimizer, refresh, fallback

  # NOTE: we're abusing `bfgs_state.history.shape` again to get the batch
  # dimensions Also: the Cauchy point is a minimization along the (projected)
  # minus gradient direction; this is why we can skip subspace minimization if
  # there's no history (because the search direction would indeed have been the
  # minus gradient), but should run it otherwise (to make use of the BFGS
  # information).
  skip_subspace = (
    (~tf.reduce_any(cauchy_state.free_mask)) |
    tf.reduce_all(bfgs_state.history == 0))
  minimizer, refresh, clip_before = (
    tf.cond(
      pred=skip_subspace,
      true_fn=lambda: (cauchy_state.cauchy_point,
               tf.broadcast_to(
                 False, ps.shape(bfgs_state.history)),
               tf.broadcast_to(True, ps.shape(bfgs_state.history))),
      false_fn=_find_constrained_minimizer))

  search_direction = (minimizer - bfgs_state.position)

  # Reset if the search direction still isn't a direction of strong descent
  refresh |= (
    tf.reduce_sum(
      search_direction * bfgs_state.objective_gradient, axis=-1) > 0)

  # Refresh conditions only make sense if a batch had not already converged
  refresh &= ~ (bfgs_state.converged | bfgs_state.failed)

  # Apply refresh
  bfgs_state = _erase_history(bfgs_state, refresh)

  return search_direction, bfgs_state, clip_before, refresh


def _constrained_line_search_step(bfgs_state, value_and_gradients_function,
                  search_direction, grad_tolerance, f_relative_tolerance,
                  x_tolerance, stopping_condition, max_iterations, clip_before):
  """Performs a constrained line search clamped to bounds in given direction."""
  inactive = (bfgs_state.failed | bfgs_state.converged)

  def _do_line_search_step():
    """Do unconstrained line search."""
    nonlocal search_direction
    # Truncation bounds
    lower_term = tf.math.divide_no_nan(
      bfgs_state.lower_bounds - bfgs_state.position,
      search_direction)
    upper_term = tf.math.divide_no_nan(
      bfgs_state.upper_bounds - bfgs_state.position,
      search_direction)
    bounds_clip = (
      tf.reduce_min(
        tf.where(
          (search_direction > 0),
          upper_term,
          tf.where(
            (search_direction < 0),
            lower_term,
            float('inf'))),
        axis=-1)
    )

    search_direction *= tf.where(
      clip_before,
      tf.math.minimum(1., bounds_clip),
      1.)[..., tf.newaxis]

    def _fn_with_report(x):
      return value_and_gradients_function(
        x, inactive, bfgs_state.objective_value, bfgs_state.objective_gradient)

    ls_result = _hz_line_search(
      bfgs_state.position, bfgs_state.objective_value,
      bfgs_state.objective_gradient,
      _fn_with_report, search_direction,
      max_iterations, inactive)

    # Truncate to bounds after search
    step = (
      tf.math.minimum(
        bounds_clip,
        ls_result.left.x
      )
    )

    # For inactive batch members `left.x` is zero. However, their
    # `search_direction` might also be undefined, so we can't rely on
    # multiplication by zero to produce a `position_delta` of zero.
    next_position = tf.where(
      inactive[..., tf.newaxis],
      bfgs_state.position,
      bfgs_state.position + step[..., tf.newaxis] * search_direction)

    # If the movement isn't clipped, we can use the final results of the
    # line search.
    reevaluated = (tf.reduce_any(ls_result.left.x > bounds_clip))
    next_objective, next_gradient = (
      tf.cond(
        pred=reevaluated,
        true_fn=lambda: value_and_gradients_function(
          next_position, inactive, bfgs_state.objective_value,
          bfgs_state.objective_gradient),
        false_fn=lambda: (ls_result.left.f,
                  ls_result.left.full_gradient)
      )
    )

    new_failed = (bfgs_state.failed | (
      ~inactive & ~bfgs_state.converged & ~ls_result.converged))
    new_num_iterations = bfgs_state.num_iterations + 1
    new_num_objective_evaluations = tf.cond(
      pred=reevaluated,
      true_fn=lambda: (
        bfgs_state.num_objective_evaluations + ls_result.func_evals + 1),
      false_fn=lambda: (
        bfgs_state.num_objective_evaluations + ls_result.func_evals))

    # Hint the compiler that the properties' shape will not change
    if not tf.executing_eagerly():
      new_failed = tf.ensure_shape(new_failed, bfgs_state.failed.shape)
      new_num_iterations = tf.ensure_shape(
        new_num_iterations, bfgs_state.num_iterations.shape)
      new_num_objective_evaluations = tf.ensure_shape(
        new_num_objective_evaluations, bfgs_state.num_objective_evaluations.shape)

    state_after_ls = bfgs_utils.update_fields(
      state=bfgs_state,
      failed=new_failed,
      num_iterations=new_num_iterations,
      num_objective_evaluations=new_num_objective_evaluations)

    return state_after_ls, next_position, next_objective, next_gradient

  # NOTE: It's important that the default (false `pred`) step matches
  # the shape of true `pred` shape for graph purposes
  state_after_ls, next_position, next_objective, next_gradient = (
    tf.cond(
      pred=tf.math.logical_not(tf.reduce_all(inactive)),
      true_fn=_do_line_search_step,
      false_fn=lambda: (bfgs_state,
                bfgs_state.position,
                bfgs_state.objective_value,
                bfgs_state.objective_gradient)
    ))

  def _do_update_position():
    """Update the position"""
    return _update_position(
      state_after_ls,
      next_position,
      next_objective,
      next_gradient,
      grad_tolerance,
      f_relative_tolerance,
      x_tolerance,
      inactive)

  return ps.cond(
    (stopping_condition(bfgs_state.converged, bfgs_state.failed) |
     tf.reduce_all(inactive)),
    true_fn=lambda: state_after_ls,
    false_fn=_do_update_position)


def _hz_line_search(starting_position, starting_value, starting_gradient,
          value_and_gradients_function, search_direction, max_iterations,
          inactive):
  """Performs Hager Zhang line search via `bfgs_utils.linesearch.hager_zhang`."""
  line_search_value_grad_func = bfgs_utils._restrict_along_direction(
    value_and_gradients_function, starting_position, search_direction)
  derivative_at_start_pt = tf.reduce_sum(
    starting_gradient * search_direction, axis=-1)
  val_0 = bfgs_utils.ValueAndGradient(
    x=bfgs_utils._broadcast(0, starting_position),
    f=starting_value,
    df=derivative_at_start_pt,
    full_gradient=starting_gradient)
  return bfgs_utils.linesearch.hager_zhang(
    line_search_value_grad_func,
    initial_step_size=bfgs_utils._broadcast(1, starting_position),
    value_at_zero=val_0,
    converged=inactive,
    max_iterations=max_iterations)


def _update_position(state,
           next_position,
           next_objective,
           next_gradient,
           grad_tolerance,
           f_relative_tolerance,
           x_tolerance,
           inactive):
  """Updates the state advancing its position by a given position_delta."""
  state = bfgs_utils.terminate_if_not_finite(
    state, next_objective, next_gradient)

  converged = (~inactive & ~state.failed &
         _check_convergence_bounded(state.position,
                      next_position,
                      state.objective_value,
                      next_objective,
                      next_gradient,
                      grad_tolerance,
                      f_relative_tolerance,
                      x_tolerance,
                      state.lower_bounds,
                      state.upper_bounds))
  new_converged = (state.converged | converged)

  if not tf.executing_eagerly():
    # Hint the compiler that the properties have not changed shape
    new_converged = tf.ensure_shape(new_converged, state.converged.shape)
    next_position = tf.ensure_shape(next_position, state.position.shape)
    next_objective = tf.ensure_shape(
      next_objective, state.objective_value.shape)
    next_gradient = tf.ensure_shape(
      next_gradient, state.objective_gradient.shape)

  return bfgs_utils.update_fields(
    state,
    converged=new_converged,
    position=next_position,
    objective_value=next_objective,
    objective_gradient=next_gradient)


def _erase_history(bfgs_state, where_erase):
  """Erases the BFGS correction pairs for the specified batches.

  This function will zero `gradient_deltas`, `position_deltas`, and `history`.

  Args:
    `bfgs_state`: a `LBfgsBOptimizerResults` to modify
    `where_erase`: a Boolean tensor with shape matching the batch dimensions
          with `True` for the batches to erase the history of.
  Returns:
    Modified `bfgs_state`.
  """
  # Calculate new values
  new_gradient_deltas = (tf.where(
    where_erase[..., tf.newaxis, tf.newaxis],
    0.,
    bfgs_state.gradient_deltas))
  new_position_deltas = (tf.where(
    where_erase[..., tf.newaxis, tf.newaxis],
    0.,
    bfgs_state.position_deltas))
  new_history = tf.where(where_erase, 0, bfgs_state.history)
  # Assure the compiler that the shape of things has not changed
  if not tf.executing_eagerly():
    new_gradient_deltas = (
      tf.ensure_shape(
        new_gradient_deltas,
        bfgs_state.gradient_deltas.shape))
    new_position_deltas = (
      tf.ensure_shape(
        new_position_deltas,
        bfgs_state.position_deltas.shape))
    new_history = (
      tf.ensure_shape(
        new_history,
        bfgs_state.history.shape))
  # Update and return
  return bfgs_utils.update_fields(
    bfgs_state,
    gradient_deltas=new_gradient_deltas,
    position_deltas=new_position_deltas,
    history=new_history)


def _check_convergence_bounded(current_position,
                 next_position,
                 current_objective,
                 next_objective,
                 next_gradient,
                 grad_tolerance,
                 f_relative_tolerance,
                 x_tolerance,
                 lower_bounds,
                 upper_bounds):
  """Checks if the algorithm satisfies the convergence criteria."""
  # NOTE: The original algorithm (as described in [2]) only considers halting on
  # the projected gradient condition. However, `x_converged` and `f_converged`
  # do not seem to pose a problem when refreshing is correctly accounted for
  # (so that the optimization does not halt upon a refresh), and the default
  # values of `0` for `f_relative_tolerance` and `x_tolerance` further
  # strengthen these conditions.
  proj_grad_converged = bfgs_utils.norm(
    tf.clip_by_value(
      next_position - next_gradient,
      lower_bounds,
      upper_bounds) - next_position, dims=1) <= grad_tolerance
  x_converged = bfgs_utils.norm(
    next_position - current_position, dims=1) <= x_tolerance
  f_ref = tf.math.maximum(1., tf.math.maximum(
    tf.math.abs(next_objective),
    tf.math.abs(current_objective)))
  f_converged = (tf.math.abs(next_objective - current_objective)
           <= f_ref*f_relative_tolerance)
  return proj_grad_converged | x_converged | f_converged


def _get_initial_state(value_and_gradients_function,
             initial_position,
             lower_bounds,
             upper_bounds,
             num_correction_pairs,
             tolerance):
  """Create LBfgsBOptimizerResults with initial state of search procedure."""
  init_args = get_initial_state_args(value_and_gradients_function,
                     initial_position,
                     tolerance)
  empty_queue = _make_empty_queue_for(num_correction_pairs, initial_position)
  zero_history = tf.zeros(ps.shape(initial_position)[:-1], dtype=tf.int32)
  init_args.update(
    lower_bounds=lower_bounds,
    upper_bounds=upper_bounds,
    position_deltas=empty_queue,
    gradient_deltas=empty_queue,
    history=zero_history)
  return LBfgsBOptimizerResults(**init_args)


def get_initial_state_args(value_and_gradients_function,
               initial_position,
               grad_tolerance,
               control_inputs=None):
  none_finished = tf.broadcast_to(False, ps.shape(initial_position)[:-1])
  zero_values = bfgs_utils._broadcast(0., initial_position)
  zero_gradients = tf.zeros_like(initial_position)
  if control_inputs:
    with tf.control_dependencies(control_inputs):
      f0, df0 = value_and_gradients_function(
        initial_position, none_finished, zero_values, zero_gradients)
  else:
    f0, df0 = value_and_gradients_function(
      initial_position, none_finished, zero_values, zero_gradients)
  # This is a gradient-based convergence check.  We only do it for finite
  # objective values because we assume the gradient reported at a position with
  # a non-finite objective value is untrustworthy.  The main loop handles
  # non-finite objective values itself (see `terminate_if_not_finite`).
  init_converged = (tf.math.is_finite(f0) &
            (bfgs_utils.norm(df0, dims=1) < grad_tolerance))
  return dict(
    converged=init_converged,
    failed=tf.zeros_like(init_converged),  # i.e. False.
    num_iterations=tf.convert_to_tensor(0),
    num_objective_evaluations=tf.convert_to_tensor(1),
    position=initial_position,
    objective_value=f0,
    objective_gradient=df0)


def _cauchy_init_m(state, theta, num_correction_pairs):
  """Initialize the M matrix for a `_CauchyMinimizationResult` state."""
  def build_m():
    """Construct and invert the M block matrix."""
    # All of the below block matrices have dimensions [..., 2m, 2m]
    #  where `...` denotes the batch dimensions, and `m` the number
    #  of correction pairs.
    # New elements are pushed in "from the back", so we want to index
    #  position_deltas and gradient_deltas with negative indices.
    # Index 0 of `position_deltas` and `gradient_deltas` is oldest, and index -1
    #  is most recent, so the below respects the indexing of the article.

    # 1. calculate inner product (s_i.y_j) in shape [..., m, m]
    l = tf.einsum(
      "...mi,...ui->...mu",
      state.position_deltas,
      state.gradient_deltas)
    # 2. Zero out diagonal and upper triangular
    l_shape = ps.shape(l)
    l = tf.linalg.set_diag(
      tf.linalg.band_part(l, -1, 0),
      tf.zeros([l_shape[0], l_shape[-1]]))
    l_transpose = tf.linalg.matrix_transpose(l)
    s_t_s = tf.einsum(
      '...mi,...ni->...mn',
      state.position_deltas,
      state.position_deltas)
    d = tf.linalg.diag(
      tf.einsum(
        '...mi,...mi->...m',
        state.position_deltas,
        state.gradient_deltas))

    # Assemble into full matrix
    # shape [b, 2m, 2m]
    m_inv = tf.concat(
      [
        tf.concat([-d, l_transpose], axis=-1),
        tf.concat(
          [l, theta[..., tf.newaxis, tf.newaxis] * s_t_s], axis=-1)
      ], axis=-2)

    # Adjust for varying history:
    # Push columns indexed h,...,2m-h to the left (but to the right of 0...m-h)
    #  and same index rows to the bottom
    idx = tf.concat(
      [tf.ragged.range(num_correction_pairs-state.history),
       tf.ragged.range(num_correction_pairs, 2 *
              num_correction_pairs-state.history),
       tf.ragged.range(num_correction_pairs -
              state.history, num_correction_pairs),
       tf.ragged.range(
              2*num_correction_pairs-state.history, 2*num_correction_pairs)],
      axis=-1).to_tensor()
    m_inv = tf.gather(
      m_inv,
      idx,
      axis=-1,
      batch_dims=1)
    m_inv = tf.gather(
      m_inv,
      idx,
      axis=-2,
      batch_dims=1)

    # Insert an identity in the empty block
    identity_mask = (
      (tf.range(ps.shape(m_inv)[-1])[tf.newaxis, ...] <
       2*(num_correction_pairs - state.history[..., tf.newaxis]))[..., tf.newaxis])

    m_inv = tf.where(
      identity_mask,
      tf.eye(ps.shape(m_inv)[-1], batch_shape=ps.shape(m_inv)[:-2]),
      m_inv)

    # If M is not invertible, refresh the memory
    # TODO: Checking the determinant is likely overkill?
    refresh = (tf.linalg.det(m_inv) == 0)

    # Invert where invertible; 0s otherwise
    m = tf.where(
      refresh[..., tf.newaxis, tf.newaxis],
      tf.zeros_like(m_inv),
      tf.linalg.inv(
        tf.where(
          refresh[..., tf.newaxis, tf.newaxis],
          tf.eye(ps.shape(m_inv)[-1],
               batch_shape=ps.shape(m_inv)[:-2]),
          m_inv)))

    # Re-zero the introduced identity blocks
    m = tf.where(
      identity_mask,
      tf.zeros_like(m),
      m)

    return m, refresh

  # M is 0 for the first iterations
  # We abuse `state.history` to extract the batch shape
  m_shape = ps.concat([ps.shape(state.history),
             [num_correction_pairs*2, num_correction_pairs*2]], axis=0)
  return tf.cond(
    state.num_iterations < 1,
    lambda: (tf.zeros(m_shape),
         tf.broadcast_to(False, ps.shape(state.history))),
    build_m)


def _make_empty_queue_for(k, element):
  """Creates a `tf.Tensor` suitable to hold `k` element-shaped tensors.

  For example:

  ```python
    element = tf.constant([[0., 1., 2., 3., 4.],
               [5., 6., 7., 8., 9.]])

    # A queue capable of holding 3 elements.
    _make_empty_queue_for(3, element)
    # => [[[0., 0., 0., 0., 0.],
    #      [0., 0., 0., 0., 0.],
    #      [0., 0., 0., 0., 0.]],
    #     [[0., 0., 0., 0., 0.],
    #      [0., 0., 0., 0., 0.],
    #      [0., 0., 0., 0., 0.]]]
  ```

  Args:
    k: A positive scalar integer, number of elements that each queue will hold.
    element: A `tf.Tensor`, only its shape and dtype information are relevant.

  Returns:
    A zero-filed `tf.Tensor` of shape `(s[:-1], k, s[-1])`, where
    `s = tf.shape(element)`, and same dtype as `element`.
  """
  queue_shape = ps.concat(
    [ps.shape(element)[:-1], [k], ps.shape(element)[-1:]], axis=0)
  return tf.zeros(queue_shape, dtype=dtype_util.base_dtype(element.dtype))


def _queue_push(queue, should_update, new_vecs):
  """Conditionally push new vectors into a batch of first-in-first-out queues.

  The `queue` of shape `[..., k, n]` can be thought of as a batch of queues,
  each holding `k` n-D vectors; while `new_vecs` of shape `[..., n]` is a
  fresh new batch of n-D vectors. The `should_update` batch of Boolean scalars,
  i.e. shape `[...]`, indicates batch members whose corresponding n-D vector in
  `new_vecs` should be added at the back of its queue, pushing out the
  corresponding n-D vector from the front. Batch members in `new_vecs` for
  which `should_update` is False are ignored.

  Note: whereas `lbfgs.py` places the `k` at dimension 0 due to constraints
  of `tf.scan`, those do not apply here, and in fact it is more advantageous
  to have the batch dimensions before `k`.

  For example:

  ```python
    b, k, n = (2, 3, 5)
    queue = tf.reshape(tf.range(30), (b, k, n))
    # => [[[ 0,  1,  2,  3,  4],
    #      [ 5,  6,  7,  8,  9],
    #      [10, 11, 12, 13, 14]],
    #    [[15, 16, 17, 18, 19],
    #     [20, 21, 22, 23, 24],
    #     [25, 26, 27, 28, 29]]]

    element = tf.reshape(tf.range(30, 40), (b, n))
    # => [[30, 31, 32, 33, 34],
      [35, 36, 37, 38, 39]]

    should_update = tf.constant([True, False])  # Shape: (b,)

    _queue_push(queue, should_update, element)
    # => [[[ 5,  6,  7,  8,  9],
    #      [10, 11, 12, 13, 14],
    #      [30, 31, 32, 33, 34]],
    #     [[15, 16, 17, 18, 19],
    #      [20, 21, 22, 23, 24],
    #      [25, 26, 27, 28, 29]]]
  ```

  Args:
    queue: A `tf.Tensor` of shape `[..., k, n]`; a batch of queues each with
    `k` n-D vectors.
    should_update: A Boolean `tf.Tensor` of shape `[...]` indicating batch
    members where new vectors should be added to their queues.
    new_vecs: A `tf.Tensor` of shape `[..., n]`; a batch of n-D vectors to add
    at the end of their respective queues, pushing out the first element from
    each.

  Returns:
    A new `tf.Tensor` of shape `[..., k, n]`.
  """
  new_queue = tf.concat(
    [queue[..., 1:, :], new_vecs[..., tf.newaxis, :]], axis=-2)
  return tf.where(
    should_update[..., tf.newaxis, tf.newaxis], new_queue, queue)
