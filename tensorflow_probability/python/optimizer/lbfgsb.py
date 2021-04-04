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
from numpy.core.fromnumeric import argmin, clip

from tensorflow.python.ops.gen_array_ops import gather, lower_bound, where
from tensorflow_probability.python.internal.backend.numpy import dtype, numpy_math

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
        'lower_bounds', # A tensor containing the lower bounds to the constrained
                        # optimization, cast to the shape of `position`.
        'upper_bounds', # A tensor containing the upper bounds to the constrained
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
        'history', # How many gradient/position deltas should be considered.
    ])

_ConstrainedCauchyState = collections.namedtuple(
    '_ConstrainedCauchyResult', [
        'theta', # `\theta` in [2]; n the Cauchy search, relates to the implicit Hessian
                 # `B = \theta*I - WMW'` (`I` the identity, see [1,2] for details)
        'm', # `M_k` matrix in [2]; part of the implicit representation of the Hessian,
             # see the comment above
        'breakpoints', # `t_i` in [Byrd et al.][2];
                       # the breakpoints in the branch definition of the
                       # projection of the gradients, batched
        'steepest', # `d` in [2]; steepest descent clamped to bounds
        'free_vars_idx', # `\mathcal{F}` of [2]; the indices of (currently) free variables.
                          # Indices that are no longer free are marked with a negative value.
                          # This is used instead of a ragged tensor because the size of the
                          #  state object must remain constant between iterations of the
                          #  while loop.
        'free_mask', # Boolean mask of free variables
        'p', # as in [2]
        'c', # as in [2]
        'df', # `f'` in [2]; (corrected) gradient 2-norm
        'ddf', # `f''` in [2]; (corrected) laplacian 2-norm (?)
        'dt_min', # `\Delta t_min` in [2]; the minimizing parameter
                  # along the search direction
        'breakpoint_min', # `t` in [2]
        'breakpoint_min_idx', # `b` in [2]
        'dt', # `\Delta t` in [2]
        'breakpoint_min_old', # t_old in [2]
        'cauchy_point', # `x^cp` in [2]; the actual cauchy point (we're looking for)
        'active', # What batches are in active optimization
    ])

def minimize(value_and_gradients_function,
             initial_position,
             bounds=None,
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
             name=None):
  """Applies the L-BFGS-B algorithm to minimize a differentiable function.

  Performs optionally constrained minimization of a differentiable function using the
  L-BFGS-B scheme. See [Nocedal and Wright(2006)][1] for details on the unconstrained
  version, and [Byrd et al.][2] for details on the constrained algorithm.

  ### Usage:

  The following example demonstrates the L-BFGS-B optimizer attempting to find the
  constrained minimum for a simple high-dimensional quadratic objective function.

  ```python
    TODO
  ```

  ### References:

  [1] Jorge Nocedal, Stephen Wright. Numerical Optimization. Springer Series
      in Operations Research. pp 176-180. 2006

  http://pages.mtu.edu/~struther/Courses/OLD/Sp2013/5630/Jorge_Nocedal_Numerical_optimization_267490.pdf

  [2] Richard H. Byrd, Peihuang Lu, Jorge Nocedal, & Ciyou Zhu (1995).
      A Limited Memory Algorithm for Bound Constrained Optimization
      SIAM Journal on Scientific Computing, 16(5), 1190–1208.

  https://doi.org/10.1137/0916069

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
      approximation of the Hessian matri
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

  def _lbfgs_defer():
      return lbfgs_minimize(value_and_gradients_function,
             initial_position,
             previous_optimizer_results,
             num_correction_pairs,
             tolerance,
             x_tolerance,
             f_relative_tolerance,
             initial_inverse_hessian_estimate,
             max_iterations,
             parallel_iterations,
             stopping_condition,
             max_line_search_iterations,
             name)

  if bounds is None:
    return _lbfgs_defer()
    
  if len(bounds) != 2:
    raise ValueError(
      '`bounds` parameter has unexpected number of elements '
      '(expected 2).')

  lower_bounds, upper_bounds = bounds
  
  if lower_bounds is None and upper_bounds is None:
    return _lbfgs_defer()
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
      dtype = dtype_util.base_dtype(previous_optimizer_results.position.dtype)

    # TODO: This isn't agnostic to the number of batch dimensions, it only
    #  supports one batch dimension, but I've found RaggedTensors to be far
    #  too finicky/undocumented to handle multiple batch dimensions in any
    #  sane way. (Even the way it's working so far is less than ideal.) 
    if len(position_shape) > 2:
      raise NotImplementedError("More than a batch dimension is not implemented. "
                                "Consider flattening and then reshaping the results.") 
    # NOTE: Broadcasting the batched dimensions breaks when there are no
    #  batched dimensions. Although this isn't handled like this in
    #  `lbfgs.py`, I'd rather force a batch dimension with a single
    #  element than do conditional checks later.
    if len(position_shape) == 1:
      position_shape = tf.concat([[1], position_shape], axis=0)
      initial_position = tf.broadcast_to(initial_position, position_shape)

    # NOTE: Could maybe use bfgs_utils._broadcast here, but would have to check
    #  that the non-batching dimensions also match; using `tf.broadcast_to` has
    #  the advantage that passing a (1,)-shaped tensor as bounds will correctly
    #  bound every variable at the single value.
    if lower_bounds is None:
      lower_bounds = tf.constant(
        [-float('inf')], shape=position_shape, dtype=dtype, name='lower_bounds')
    else:
      lower_bounds = tf.cast(tf.convert_to_tensor(lower_bounds), dtype=dtype)
      try:
        lower_bounds = tf.broadcast_to(
          lower_bounds, position_shape, name='lower_bounds')
      except tf.errors.InvalidArgumentError:
        raise ValueError(
          'Failed to broadcast lower bounds tensor to the shape of starting position. '
          'Are the lower bounds well formed?')
    if upper_bounds is None:
      upper_bounds = tf.constant(
        [float('inf')], shape=position_shape, dtype=dtype, name='upper_bounds')
    else:
      upper_bounds = tf.cast(tf.convert_to_tensor(upper_bounds), dtype=dtype)
      try:
        upper_bounds = tf.broadcast_to(
          upper_bounds, position_shape, name='upper_bounds')
      except tf.errors.InvalidArgumentError:
        raise ValueError(
          'Failed to broadcast upper bounds tensor to the shape of starting position. '
          'Are the lower bounds well formed?')  

    # Clamp the starting position to the bounds, because the algorithm expects the
    # variables to be in range for the Hessian inverse estimation, but also because
    # that fast-tracks the first iteration of the Cauchy optimization.
    initial_position = tf.clip_by_value(initial_position, lower_bounds, upper_bounds)

    tolerance = tf.convert_to_tensor(
        tolerance, dtype=dtype, name='grad_tolerance')
    f_relative_tolerance = tf.convert_to_tensor(
        f_relative_tolerance, dtype=dtype, name='f_relative_tolerance')
    x_tolerance = tf.convert_to_tensor(
        x_tolerance, dtype=dtype, name='x_tolerance')
    max_iterations = tf.convert_to_tensor(max_iterations, name='max_iterations')

    # The `state` here is a `LBfgsBOptimizerResults` tuple with values for the
    # current state of the algorithm computation.
    def _cond(state):
      """Continue if iterations remain and stopping condition is not met."""
      return ((state.num_iterations < max_iterations) &
              tf.logical_not(stopping_condition(state.converged, state.failed)))

    def _body(current_state):
      """Main optimization loop."""
      current_state = bfgs_utils.terminate_if_not_finite(current_state)
  
      cauchy_point, free_mask = \
        _cauchy_minimization(current_state, num_correction_pairs, parallel_iterations)

      search_direction = _get_search_direction(current_state)

      # TODO(b/120134934): Check if the derivative at the start point is not
      # negative, if so then reset position/gradient deltas and recompute
      # search direction.
      # NOTE: Erasing is currently handled in `_bounded_line_search_step`
      search_direction = tf.where(
                          free_mask,
                          search_direction,
                          0.)
      bad_direction = \
        (tf.reduce_sum(search_direction * current_state.objective_gradient, axis=-1) > 0)

      cauchy_search = _cauchy_line_search_step(current_state,
          value_and_gradients_function, search_direction,
          tolerance, f_relative_tolerance, x_tolerance, stopping_condition,
          max_line_search_iterations, free_mask, cauchy_point)
      
      search_direction = cauchy_search.position - current_state.position
      next_state = _bounded_line_search_step(current_state,
          value_and_gradients_function, search_direction,
          tolerance, f_relative_tolerance, x_tolerance, stopping_condition,
          max_line_search_iterations, bad_direction)

      # If not failed or converged, update the Hessian estimate.
      # Only do this if the new pairs obey the s.y > 0
      position_delta = next_state.position - current_state.position
      gradient_delta = next_state.objective_gradient - current_state.objective_gradient
      positive_prod = (tf.math.reduce_sum(position_delta * gradient_delta, axis=-1) > \
                        1E-8*tf.reduce_sum(gradient_delta**2, axis=-1))
      should_push = ~(next_state.converged | next_state.failed) & positive_prod & ~bad_direction
      new_position_deltas = _queue_push(
              next_state.position_deltas, should_push, position_delta)
      new_gradient_deltas = _queue_push(
              next_state.gradient_deltas, should_push, gradient_delta)
      new_history = tf.where(
              should_push,
              tf.math.minimum(next_state.history + 1, num_correction_pairs),
              next_state.history)
      
      if not tf.executing_eagerly():
        # Hint the compiler that the shape of the properties has not changed
        new_position_deltas = tf.ensure_shape(
          new_position_deltas, next_state.position_deltas.shape)
        new_gradient_deltas = tf.ensure_shape(
          new_gradient_deltas, next_state.gradient_deltas.shape)
        new_history = tf.ensure_shape(
          new_history, next_state.history.shape)

      state_after_inv_hessian_update = bfgs_utils.update_fields(
          next_state,
          position_deltas=new_position_deltas,
          gradient_deltas=new_gradient_deltas,
          history=new_history)

      return [state_after_inv_hessian_update]

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
  """Calculates the Cauchy point (minimizes the quadratic approximation to the
  objective function at the current position, in the direction of steepest
  descent), but bounding the gradient by the corresponding bounds.

  See algorithm CP and associated discussion of [Byrd,Lu,Nocedal,Zhu][2]
  for details.

  Args:
    bfgs_state: A `_ConstrainedCauchyState` initialized to the starting point of the
    constrained minimization.
  Returns:
    A potentially modified `state`, the obtained `cauchy_point` and boolean
    `free_mask` indicating which variables are free (`True`) and which variables
    are under active constrain (`False`)
  """
  cauchy_state = _get_initial_cauchy_state(bfgs_state, num_correction_pairs)
  # NOTE: See lbfgsb.f (l. 1649)
  ddf_org = -cauchy_state.theta * cauchy_state.df

  def _cond(state):
    """Test convergence to Cauchy point at current branch"""
    return tf.math.reduce_any(state.active)

  def _body(state):
    """Cauchy point iterative loop
    
    (While loop of CP algorithm [2])"""
    # Remove b from the free indices
    free_vars_idx, free_mask = _cauchy_remove_breakpoint_min(
                                state.free_vars_idx,
                                state.breakpoint_min_idx,
                                state.free_mask,
                                state.active)

    # Shape: [b]
    d_b = tf.where(
            state.active,
            tf.gather(
              state.steepest,
              state.breakpoint_min_idx,
              batch_dims=1),
            0.)
    # Shape: [b]
    x_b = tf.where(
            state.active,
            tf.gather(
              bfgs_state.position,
              state.breakpoint_min_idx,
              batch_dims=1),
            0.)

    # Shape: [b]
    x_cp_b = tf.where(
              state.active,
              tf.where(
                d_b > 0.,
                tf.gather(
                  bfgs_state.upper_bounds,
                  state.breakpoint_min_idx,
                  batch_dims=1),
                tf.where(
                  d_b < 0.,
                  tf.gather(
                    bfgs_state.lower_bounds,
                    state.breakpoint_min_idx,
                    batch_dims=1),
                  x_b)),
              tf.gather(
                state.cauchy_point,
                state.breakpoint_min_idx,
                batch_dims=1))

    keep_idx = (tf.range(ps.shape(state.cauchy_point)[-1]) != \
                  state.breakpoint_min_idx[..., tf.newaxis])
    cauchy_point = tf.where(
                    state.active[..., tf.newaxis],
                    tf.where(
                      keep_idx,
                      state.cauchy_point,
                      x_cp_b[..., tf.newaxis]),
                    state.cauchy_point)

    z_b = tf.where(
            state.active,
            x_cp_b - x_b,
            0.)

    c = tf.where(
        state.active[..., tf.newaxis],
        state.c + state.dt[...,tf.newaxis] * state.p,
        state.c)
    
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
    # TODO: Transpose seems inevitable, because of batch dims?
    w_b = tf.concat(
              [
                tf.gather(
                  tf.transpose(
                    bfgs_state.gradient_deltas,
                    perm=[1,0,2]),
                  state.breakpoint_min_idx,
                  axis=-1,
                  batch_dims=1),
                state.theta[..., tf.newaxis] * \
                  tf.gather(
                    tf.transpose(
                      bfgs_state.position_deltas,
                      perm=[1,0,2]),
                    state.breakpoint_min_idx,
                    axis=-1,
                    batch_dims=1)
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

    # NOTE Use of d_b = -g_b
    df = tf.where(
          state.active,
          state.df + state.dt * state.ddf + \
            d_b**2 - \
            state.theta * d_b * z_b + \
            d_b * tf.einsum(
                    '...j,...jk,...k->...',
                    w_b,
                    state.m,
                    c),
          state.df)
          
    # NOTE use of d_b = -g_b
    ddf = tf.where(
            state.active,
            state.ddf - state.theta * d_b**2 + \
              2. * d_b * tf.einsum(
                          "...i,...ij,...j->...",
                          w_b,
                          state.m,
                          state.p) - \
              d_b**2 * tf.einsum(
                        "...i,...ij,...j->...",
                        w_b,
                        state.m,
                        w_b),
            state.ddf)
    # NOTE: See lbfgsb.f (l. 1649)
    # TODO: How to get machine epsilon?
    ddf = tf.math.maximum(ddf, 1E-8*ddf_org)

    # NOTE use of d_b = -g_b
    p = tf.where(
          state.active[..., tf.newaxis],
          state.p - d_b[..., tf.newaxis] * w_b,
          state.p)

    steepest_idx = tf.range(
        ps.shape(state.steepest)[-1],
        dtype=state.breakpoint_min_idx.dtype)[tf.newaxis, ...]
    steepest = tf.where(
      state.active[..., tf.newaxis],
      tf.where(
        steepest_idx == state.breakpoint_min_idx[..., tf.newaxis],
        0.,
        state.steepest),
      state.steepest)
    
    dt_min = tf.where(
              state.active,
              -tf.math.divide_no_nan(df, ddf),
              state.dt_min)

    breakpoint_min_old = tf.where(
                          state.active,
                          state.breakpoint_min,
                          state.breakpoint_min_old)
    
    # Find b
    breakpoint_min_idx, breakpoint_min = \
      _cauchy_get_breakpoint_min(
        state.breakpoints,
        free_vars_idx)
    breakpoint_min_idx = tf.where(
                          state.active,
                          breakpoint_min_idx,
                          state.breakpoint_min_idx)
    breakpoint_min = tf.where(
                      state.active,
                      breakpoint_min,
                      state.breakpoint_min)

    dt = tf.where(
          state.active,
          breakpoint_min - state.breakpoint_min,
          state.dt)
          
    active = tf.where(
              state.active,
              _cauchy_update_active(free_vars_idx, dt_min, dt),
              state.active)

    # We have to hint the "compiler" that the shapes of the new
    # values are the same as the old values.
    if not tf.executing_eagerly():
      steepest = tf.ensure_shape(steepest, state.steepest.shape)
      free_vars_idx = tf.ensure_shape(free_vars_idx, state.free_vars_idx.shape)
      free_mask = tf.ensure_shape(free_mask, state.free_mask.shape)
      p = tf.ensure_shape(p, state.p.shape)
      c = tf.ensure_shape(c, state.c.shape)
      df = tf.ensure_shape(df, state.df.shape)
      ddf = tf.ensure_shape(ddf, state.ddf.shape)
      dt_min = tf.ensure_shape(dt_min, state.dt_min.shape)
      breakpoint_min = tf.ensure_shape(breakpoint_min, state.breakpoint_min.shape)
      breakpoint_min_idx = tf.ensure_shape(breakpoint_min_idx, state.breakpoint_min_idx.shape)
      dt = tf.ensure_shape(dt, state.dt.shape)
      breakpoint_min_old = tf.ensure_shape(breakpoint_min_old, state.breakpoint_min_old.shape)
      cauchy_point = tf.ensure_shape(cauchy_point, state.cauchy_point.shape)
      active = tf.ensure_shape(active, state.active.shape)

    new_state = bfgs_utils.update_fields(
                  state, steepest=steepest, free_vars_idx=free_vars_idx,
                  free_mask=free_mask, p=p, c=c, df=df, ddf=ddf, dt_min=dt_min,
                  breakpoint_min=breakpoint_min, breakpoint_min_idx=breakpoint_min_idx,
                  dt=dt, breakpoint_min_old=breakpoint_min_old,
                  cauchy_point=cauchy_point, active=active)
    
    return [new_state]

  cauchy_loop = tf.while_loop(
        cond=_cond,
        body=_body,
        loop_vars=[cauchy_state],
        parallel_iterations=parallel_iterations)[0]

  # The loop broke, so the last identified `b` index never got
  # removed
  _free_vars_idx, free_mask = _cauchy_remove_breakpoint_min(
                              cauchy_loop.free_vars_idx,
                              cauchy_loop.breakpoint_min_idx,
                              cauchy_loop.free_mask,
                              cauchy_loop.active)

  dt_min = tf.math.maximum(cauchy_loop.dt_min, 0)
  t_old = cauchy_loop.breakpoint_min_old + dt_min
  
  # A breakpoint of -1 means that we ran out of free variables
  flagged_breakpoint_min = tf.where(
                              cauchy_loop.breakpoint_min < 0,
                              float('inf'),
                              cauchy_loop.breakpoint_min)
  cauchy_point = tf.where(
      ~(bfgs_state.converged | bfgs_state.failed)[..., tf.newaxis],
      tf.where(
        cauchy_loop.breakpoints >= flagged_breakpoint_min[..., tf.newaxis],
        bfgs_state.position + t_old[..., tf.newaxis] * cauchy_loop.steepest,
        cauchy_loop.cauchy_point),
      bfgs_state.position)

  # NOTE: We only return the cauchy point and the free mask, so there is no
  #  need to update the actual state, even though we could at this point update
  #  `free_vars_idx`, `free_mask`, and `cauchy_point`
  free_mask = free_mask & ~(cauchy_loop.breakpoints != cauchy_loop.breakpoint_min)

  return cauchy_point, free_mask


def _cauchy_update_active(free_vars_idx, dt_min, dt):
  return tf.where(
            tf.reduce_any(free_vars_idx >= 0, axis=-1) & (dt_min >= dt),
            True,
            False)


def _hz_line_search(state, value_and_gradients_function,
      search_direction, max_iterations, inactive):
  line_search_value_grad_func = bfgs_utils._restrict_along_direction(
      value_and_gradients_function, state.position, search_direction)
  derivative_at_start_pt = tf.reduce_sum(
      state.objective_gradient * search_direction, axis=-1)
  val_0 = bfgs_utils.ValueAndGradient(x=bfgs_utils._broadcast(0, state.position),
                           f=state.objective_value,
                           df=derivative_at_start_pt,
                           full_gradient=state.objective_gradient)
  return bfgs_utils.linesearch.hager_zhang(
      line_search_value_grad_func,
      initial_step_size=bfgs_utils._broadcast(1, state.position),
      value_at_zero=val_0,
      converged=inactive,
      max_iterations=max_iterations)  # No search needed for these.


def _cauchy_line_search_step(state, value_and_gradients_function, search_direction,
                     grad_tolerance, f_relative_tolerance, x_tolerance,
                     stopping_condition, max_iterations, free_mask, cauchy_point):
  """Performs the line search in given direction, backtracking in direction to the cauchy point,
  and clamping actively contrained variables to the cauchy point."""
  inactive = state.failed | state.converged
  ls_result = _hz_line_search(state, value_and_gradients_function,
                search_direction, max_iterations, inactive)
  
  state_after_ls = bfgs_utils.update_fields(
      state,
      failed=state.failed | (~state.converged & ~ls_result.converged & tf.reduce_any(free_mask, axis=-1)),
      num_iterations=state.num_iterations + 1,
      num_objective_evaluations=(
          state.num_objective_evaluations + ls_result.func_evals + 1))

  def _do_update_position():
    # For inactive batch members `left.x` is zero. However, their
    # `search_direction` might also be undefined, so we can't rely on
    # multiplication by zero to produce a `position_delta` of zero.
    alpha = ls_result.left.x[..., tf.newaxis]
    ideal_position = tf.where(
        inactive[..., tf.newaxis],
        state.position,
        tf.where(
          free_mask,
          state.position + search_direction * alpha,
          cauchy_point))

    # Backtrack from the ideal position in direction to the Cauchy point
    cauchy_to_ideal = ideal_position - cauchy_point
    clip_lower = tf.math.divide_no_nan(
                  state.lower_bounds - cauchy_point,
                  cauchy_to_ideal)
    clip_upper = tf.math.divide_no_nan(
                  state.upper_bounds - cauchy_point,
                  cauchy_to_ideal)
    clip = tf.math.reduce_min(
            tf.where(
              cauchy_to_ideal > 0,
              clip_upper,
              tf.where(
                cauchy_to_ideal < 0,
                clip_lower,
                float('inf'))),
            axis=-1)
    alpha = tf.minimum(1.0, clip)[..., tf.newaxis]
    
    next_position = tf.where(
        inactive[..., tf.newaxis],
        state.position,
        tf.where(
          free_mask,
          cauchy_point + alpha * cauchy_to_ideal,
          cauchy_point))
    
    # NOTE: one extra call to the function
    next_objective, next_gradient = \
      value_and_gradients_function(next_position)

    return _update_position(
        state_after_ls,
        next_position,
        next_objective,
        next_gradient,
        grad_tolerance,
        f_relative_tolerance,
        x_tolerance,
        tf.constant(False))

  return ps.cond(
      stopping_condition(state.converged, state.failed),
      true_fn=lambda: state_after_ls,
      false_fn=_do_update_position)


def _bounded_line_search_step(state, value_and_gradients_function, search_direction,
                     grad_tolerance, f_relative_tolerance, x_tolerance,
                     stopping_condition, max_iterations, bad_direction):
  """Performs a line search in given direction, clamping to the bounds, and fixing the actively
  constrained values to the given values."""
  inactive = state.failed | state.converged | bad_direction
  ls_result = _hz_line_search(state, value_and_gradients_function,
                search_direction, max_iterations, inactive)

  new_failed = state.failed | (~state.converged & ~ls_result.converged \
                              & tf.reduce_any(search_direction != 0, axis=-1)) \
                                & ~bad_direction
  new_num_iterations = state.num_iterations + 1
  new_num_objective_evaluations = (
          state.num_objective_evaluations + ls_result.func_evals + 1)

  if not tf.executing_eagerly():
    # Hint the compiler that the properties' shape will not change
    new_failed = tf.ensure_shape(
      new_failed, state.failed.shape)
    new_num_iterations = tf.ensure_shape(
      new_num_iterations, state.num_iterations.shape)
    new_num_objective_evaluations = tf.ensure_shape(
      new_num_objective_evaluations, state.num_objective_evaluations.shape)

  state_after_ls = bfgs_utils.update_fields(
      state,
      failed=new_failed,
      num_iterations=new_num_iterations,
      num_objective_evaluations=new_num_objective_evaluations)

  def _do_update_position():
    lower_term = tf.math.divide_no_nan(
                  state.lower_bounds - state.position,
                  search_direction)
    upper_term = tf.math.divide_no_nan(
                  state.upper_bounds - state.position,
                  search_direction)
    
    under_clip = tf.math.reduce_max(
                  tf.where(
                    (search_direction > 0),
                    lower_term,
                    tf.where(
                      (search_direction < 0),
                      upper_term,
                      -float('inf'))),
                  axis=-1)
    over_clip = tf.math.reduce_min(
                  tf.where(
                    (search_direction > 0),
                    upper_term,
                    tf.where(
                      (search_direction < 0),
                      lower_term,
                      float('inf'))),
                  axis=-1)

    alpha_clip = tf.clip_by_value(
                  ls_result.left.x,
                  under_clip,
                  over_clip)[..., tf.newaxis]

    # For inactive batch members `left.x` is zero. However, their
    # `search_direction` might also be undefined, so we can't rely on
    # multiplication by zero to produce a `position_delta` of zero.
    next_position = tf.where(
        inactive[..., tf.newaxis],
        state.position,
        state.position + search_direction * alpha_clip)
          
    # one extra call to the function, counted above
    next_objective, next_gradient = \
      value_and_gradients_function(next_position)

    return _update_position(
        state_after_ls,
        next_position,
        next_objective,
        next_gradient,
        grad_tolerance,
        f_relative_tolerance,
        x_tolerance,
        bad_direction)

  return ps.cond(
      stopping_condition(state.converged, state.failed),
      true_fn=lambda: state_after_ls,
      false_fn=_do_update_position)


def _update_position(state,
                     next_position,
                     next_objective,
                     next_gradient,
                     grad_tolerance,
                     f_relative_tolerance,
                     x_tolerance,
                     erase_memory):
  """Updates the state advancing its position by a given position_delta.
  Also erases the LBFGS memory if indicated."""
  state = bfgs_utils.terminate_if_not_finite(state, next_objective, next_gradient)

  converged = ~state.failed & \
                      _check_convergence_bounded(state.position,
                                                 next_position,
                                                 state.objective_value,
                                                 next_objective,
                                                 next_gradient,
                                                 grad_tolerance,
                                                 f_relative_tolerance,
                                                 x_tolerance,
                                                 state.lower_bounds,
                                                 state.upper_bounds)
  new_position_deltas = tf.where(
                      erase_memory[..., tf.newaxis],
                      tf.zeros_like(state.position_deltas),
                      state.position_deltas)
  new_gradient_deltas = tf.where(
                      erase_memory[..., tf.newaxis],
                      tf.zeros_like(state.gradient_deltas),
                      state.gradient_deltas)
  new_history = tf.where(
              erase_memory,
              tf.zeros_like(state.history),
              state.history)
  new_converged = (state.converged | converged)

  if not tf.executing_eagerly():
    # Hint the compiler that the properties have not changed shape
    new_converged = tf.ensure_shape(new_converged, state.converged.shape)
    next_position = tf.ensure_shape(next_position, state.position.shape)
    next_objective = tf.ensure_shape(next_objective, state.objective_value.shape)
    next_gradient = tf.ensure_shape(next_gradient, state.objective_gradient.shape)
    new_position_deltas = tf.ensure_shape(new_position_deltas, state.position_deltas.shape)
    new_gradient_deltas = tf.ensure_shape(new_gradient_deltas, state.gradient_deltas.shape)
    new_history = tf.ensure_shape(new_history, state.history.shape)

  return bfgs_utils.update_fields(
      state,
      converged=new_converged,
      position=next_position,
      objective_value=next_objective,
      objective_gradient=next_gradient,
      position_deltas=new_position_deltas,
      gradient_deltas=new_gradient_deltas,
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
  proj_grad_converged = bfgs_utils.norm(
                          tf.clip_by_value(
                            next_position - next_gradient,
                            lower_bounds,
                            upper_bounds) - next_position, dims=1) <= grad_tolerance
  x_converged = bfgs_utils.norm(next_position - current_position, dims=1) <= x_tolerance
  f_converged = bfgs_utils.norm(next_objective - current_objective, dims=0) <= \
                  f_relative_tolerance * current_objective
  return proj_grad_converged | x_converged | f_converged


def _get_initial_state(value_and_gradients_function,
                       initial_position,
                       lower_bounds,
                       upper_bounds,
                       num_correction_pairs,
                       tolerance):
  """Create LBfgsBOptimizerResults with initial state of search procedure."""
  init_args = bfgs_utils.get_initial_state_args(
      value_and_gradients_function,
      initial_position,
      tolerance)
  init_args.update(lower_bounds=lower_bounds, upper_bounds=upper_bounds)
  empty_queue = _make_empty_queue_for(num_correction_pairs, initial_position)
  init_args.update(
    position_deltas=empty_queue,
    gradient_deltas=empty_queue,
    history=tf.zeros(ps.shape(initial_position)[:-1], dtype=tf.int32))
  return LBfgsBOptimizerResults(**init_args)


def _get_initial_cauchy_state(state, num_correction_pairs):
  """Create _ConstrainedCauchyState with initial parameters"""
  
  theta = tf.math.divide_no_nan(
              tf.reduce_sum(state.gradient_deltas[-1, ...]**2, axis=-1),
              tf.reduce_sum(state.gradient_deltas[-1,...] * state.position_deltas[-1, ...], axis=-1))
  theta = tf.where(
            theta != 0,
            theta,
            1.0)

  m, refresh = _cauchy_init_m(
                  state,
                  ps.shape(state.position_deltas),
                  theta,
                  num_correction_pairs)
  # Erase the history where M isn't invertible
  state = \
    bfgs_utils.update_fields(
      state,
      gradient_deltas=tf.where(
                        refresh[..., tf.newaxis],
                        tf.zeros_like(state.gradient_deltas),
                        state.gradient_deltas),
      position_deltas=tf.where(
                        refresh[..., tf.newaxis],
                        tf.zeros_like(state.position_deltas),
                        state.position_deltas),
      history=tf.where(refresh, 0, state.history))
  theta = tf.where(refresh, 1.0, theta)

  breakpoints = _cauchy_init_breakpoints(state)

  steepest = tf.where(
              breakpoints != 0.,
              -state.objective_gradient,
              0.)

  free_mask = (breakpoints > 0)
  free_vars_idx = tf.where(
                    free_mask,
                    tf.broadcast_to(
                      tf.range(ps.shape(state.position)[-1], dtype=tf.int32),
                      ps.shape(state.position)),
                    -1)

  # We need to account for the varying histories:
  # we assume that the first `2*(m-h)` rows of W'^T
  # are 0 (where `m` is the number of correction pairs
  # and `h` is the history), in concordance with the first
  # `2*(m-h)` rows of M being 0.
  # 1. Calculate all elements
  p = tf.concat(
        [
          tf.einsum(
                  "m...i,...i->...m",
                  state.gradient_deltas,
                  steepest),
          theta[..., tf.newaxis] * \
                tf.einsum(
                  "m...i,...i->...m",
                  state.position_deltas,
                  steepest)
        ],
        axis=-1)
  # 2. Assemble the rows in the correct order
  idx = tf.concat(
          [
            tf.ragged.range(
              num_correction_pairs - state.history),
            tf.ragged.range(
              num_correction_pairs,
              2*num_correction_pairs - state.history),
            tf.ragged.range(
              num_correction_pairs - state.history,
              num_correction_pairs),
            tf.ragged.range(
              2*num_correction_pairs - state.history,
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

  breakpoint_min_idx, breakpoint_min = \
    _cauchy_get_breakpoint_min(breakpoints, free_vars_idx)

  dt = breakpoint_min

  breakpoint_min_old = tf.zeros_like(breakpoint_min)

  cauchy_point = state.position

  active = ~(state.converged | state.failed) & \
              _cauchy_update_active(free_vars_idx, dt_min, dt)

  return _ConstrainedCauchyState(
    theta, m, breakpoints, steepest, free_vars_idx, free_mask,
    p, c, df, ddf, dt_min, breakpoint_min, breakpoint_min_idx,
    dt, breakpoint_min_old, cauchy_point, active)


def _cauchy_init_m(state, deltas_shape, theta, num_correction_pairs):
  def build_m():
    # All of the below block matrices have dimensions [..., m, m]
    #  where `...` denotes the batch dimensions, and `m` the number
    #  of correction pairs (compare to `deltas_shape`, which is [m,...,n]).
    # New elements are pushed in "from the back", so we want to index
    #  position_deltas and gradient_deltas with negative indices.
    # Index 0 of `position_deltas` and `gradient_deltas` is oldest, and index -1
    #  is most recent, so the below respects the indexing of the article.

    # 1. calculate inner product (s_i.y_j) in shape [..., m, m]
    l = tf.einsum(
          "m...i,u...i->...mu",
          state.position_deltas,
          state.gradient_deltas)
    # 2. Zero out diagonal and upper triangular
    l_shape = ps.shape(l)
    l = tf.linalg.set_diag(
          tf.linalg.band_part(l, -1, 0),
          tf.zeros([l_shape[0], l_shape[-1]]))
    l_transpose = tf.linalg.matrix_transpose(l)
    s_t_s = tf.einsum(
              'm...i,n...i->...mn',
              state.position_deltas,
              state.position_deltas)
    d = tf.linalg.diag(
          tf.einsum(
          'm...i,m...i->...m',
          state.position_deltas,
          state.gradient_deltas))

    # Assemble into full matrix
    # TODO: Is there no better way to create a block matrix?
    block_d = tf.concat([-d, tf.zeros_like(d)], axis=-1)
    block_d = tf.concat([block_d, tf.zeros_like(block_d)], axis=-2)
    block_l_transpose = tf.concat([tf.zeros_like(l_transpose), l_transpose], axis=-1)
    block_l_transpose = tf.concat([block_l_transpose, tf.zeros_like(block_l_transpose)], axis=-2)
    block_l = tf.concat([l, tf.zeros_like(l)], axis=-1)
    block_l = tf.concat([tf.zeros_like(block_l), block_l], axis=-2)
    block_s_t_s = tf.concat([tf.zeros_like(s_t_s), s_t_s], axis=-1)
    block_s_t_s = tf.concat([tf.zeros_like(block_s_t_s), block_s_t_s], axis=-2)

    # shape [b, 2m, 2m]
    m_inv = block_d + block_l_transpose + block_l + \
              theta[..., tf.newaxis, tf.newaxis] * block_s_t_s
    
    # Adjust for varying history:
    # Push columns indexed h,...,2m-h to the left (but to the right of 0...m-h)
    #  and same index rows to the bottom
    idx = tf.concat(
            [tf.ragged.range(num_correction_pairs-state.history),
              tf.ragged.range(num_correction_pairs, 2*num_correction_pairs-state.history),
              tf.ragged.range(num_correction_pairs-state.history, num_correction_pairs),
              tf.ragged.range(2*num_correction_pairs-state.history, 2*num_correction_pairs)],
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
    identity_mask = \
      (tf.range(ps.shape(m_inv)[-1])[tf.newaxis, ...] < \
        2*(num_correction_pairs - state.history[..., tf.newaxis]))[..., tf.newaxis]
    
    m_inv = tf.where(
              identity_mask,
              tf.eye(deltas_shape[0]*2, batch_shape=[deltas_shape[1]]),
              m_inv)

    # If M is not invertible, refresh the memory
    refresh = (tf.linalg.det(m_inv) == 0)

    # Invert where invertible; 0s otherwise
    m = tf.where(
          refresh[..., tf.newaxis, tf.newaxis],
          tf.zeros_like(m_inv),
          tf.linalg.inv(
            tf.where(
              refresh[..., tf.newaxis, tf.newaxis],
              tf.eye(deltas_shape[0]*2, batch_shape=[deltas_shape[1]]),
              m_inv)))

    # Re-zero the introduced identity blocks
    m = tf.where(
          identity_mask,
          tf.zeros_like(m),
          m)

    return m, refresh
  
  # M is 0 for the first iterations
  return tf.cond(
          state.num_iterations < 1,
          lambda: (tf.zeros([deltas_shape[1], 2*deltas_shape[0], 2*deltas_shape[0]]),
                    tf.broadcast_to(False, ps.shape(state.history))),
          build_m)


def _cauchy_init_breakpoints(state):
  breakpoints = \
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

  return breakpoints


def _cauchy_remove_breakpoint_min(free_vars_idx,
                                  breakpoint_min_idx,
                                  free_mask,
                                  active):
  """Update the free variable indices to remove the minimum breakpoint index.

  Returns:
    Updated `free_vars_idx`, `free_mask`
  """

  # NOTE: In situations where none of the indices are free, breakpoint_min_idx
  #  will falsely report 0. However, this is fine, because in this situation,
  #  every element of free_vars_idx is -1, and so there is no match.
  matching = (free_vars_idx == breakpoint_min_idx[..., tf.newaxis])
  free_vars_idx = tf.where(
                    matching,
                    -1,
                    free_vars_idx)
  free_mask = tf.where(
                active[..., tf.newaxis],
                free_vars_idx >= 0,
                free_mask)
  
  return free_vars_idx, free_mask


def _cauchy_get_breakpoint_min(breakpoints, free_vars_idx):
  """Find the smallest breakpoint of free indices, returning the minimum breakpoint
  and the corresponding index.

  Returns:
    Tuple of `breakpoint_min_idx`, `breakpoint_min`
    where
      `breakpoint_min_idx` is the index that has min. breakpoint
      `breakpoint_min` is the corresponding breakpoint
  """
  # A tensor of shape [batch, dims] that has +infinity where free_vars_idx < 0,
  #  and has breakpoints[free_vars_idx] otherwise.
  flagged_breakpoints = tf.where(
                          free_vars_idx < 0,
                          float('inf'),
                          tf.gather(
                            breakpoints,
                            tf.where(
                              free_vars_idx < 0,
                              0,
                              free_vars_idx),
                            batch_dims=1))

  argmin_idx = tf.math.argmin(
                flagged_breakpoints,
                axis=-1,
                output_type=tf.int32)
  
  # NOTE: For situations where there are no more free indices
  #  (and therefore argmin_idx indexes into -1), we set
  #  breakpoint_min_idx to 0 and flag that there are no free
  #  indices by setting the breakpoint to -1 (this is an impossible
  #  value, as breakpoints are g.e. to 0).
  #  This is because in branching situations, indexing with
  #  breakpoint_min_idx can occur, and later be discarded, but all
  #  elements in breakpoint_min_idx must be a priori valid indices.
  no_free = tf.gather(
              free_vars_idx,
              argmin_idx,
              batch_dims=1) < 0
  breakpoint_min_idx = tf.where(
                        no_free,
                        0,
                        tf.gather(
                          free_vars_idx,
                          argmin_idx,
                          batch_dims=1))
  breakpoint_min = tf.where(
                    no_free,
                    -1.,
                    tf.gather(
                      breakpoints,
                      argmin_idx,
                      batch_dims=1))

  return breakpoint_min_idx, breakpoint_min


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
    state: A `LBfgsBOptimizerResults` tuple with the current state of the
      search procedure.

  Returns:
    A real `Tensor` of the same shape as the `state.position`. The direction
    along which to perform line search.
  """
  # The number of correction pairs that have been collected so far.
  #num_elements = ps.minimum(
  #    state.num_iterations,  # TODO(b/162733947): Change loop state -> closure.
  #    ps.shape(state.position_deltas)[0])

  def _two_loop_algorithm():
    """L-BFGS two-loop algorithm."""
    # Correction pairs are always appended to the end, so only the latest
    # `num_elements` vectors have valid position/gradient deltas. Vectors
    # that haven't been computed yet are zero.
    position_deltas = state.position_deltas
    gradient_deltas = state.gradient_deltas
    num_correction_pairs, num_batches, _point_dims = \
      ps.shape(gradient_deltas, out_type=tf.int32)

    # Pre-compute all `inv_rho[i]`s.
    inv_rhos = tf.reduce_sum(
        gradient_deltas * position_deltas, axis=-1)

    def first_loop(acc, args):
      _, q_direction, num_iter = acc
      position_delta, gradient_delta, inv_rho = args
      active = (num_iter < state.history)
      alpha = tf.math.divide_no_nan(
                tf.reduce_sum(
                  position_delta * q_direction,
                  axis=-1),
                inv_rho)
      direction_delta = alpha[..., tf.newaxis] * gradient_delta
      new_q_direction = tf.where(
                          active[..., tf.newaxis],
                          q_direction - direction_delta,
                          q_direction)

      return (alpha, new_q_direction, num_iter + 1)

    # Run first loop body computing and collecting `alpha[i]`s, while also
    # computing the updated `q_direction` at each step.
    zero = tf.zeros_like(inv_rhos[0])
    alphas, q_directions, _num_iters = tf.scan(
        first_loop, [position_deltas, gradient_deltas, inv_rhos],
        initializer=(zero, state.objective_gradient, 0), reverse=True)

    # We use `H^0_k = gamma_k * I` as an estimate for the initial inverse
    # hessian for the k-th iteration; then `r_direction = H^0_k * q_direction`.
    idx = tf.transpose(
            tf.stack(
              [tf.where(
                state.history > 0,
                num_correction_pairs - state.history,
                0),
              tf.range(num_batches)]))
    gamma_k = tf.math.divide_no_nan(
                tf.gather_nd(inv_rhos, idx),
                tf.reduce_sum(
                  tf.gather_nd(gradient_deltas, idx)**2,
                  axis=-1))
    gamma_k = tf.where(
                (state.history > 0),
                gamma_k,
                1.0)
    r_direction = gamma_k[..., tf.newaxis] * tf.gather_nd(q_directions, idx)

    def second_loop(acc, args):
      r_direction, iter_idx = acc
      alpha, position_delta, gradient_delta, inv_rho = args
      active = (iter_idx >= num_correction_pairs - state.history)
      beta = tf.math.divide_no_nan(
              tf.reduce_sum(
                gradient_delta * r_direction,
                axis=-1),
              inv_rho)
      direction_delta = (alpha - beta)[..., tf.newaxis] * position_delta
      new_r_direction = tf.where(
                          active[..., tf.newaxis],
                          r_direction + direction_delta,
                          r_direction)
      return (new_r_direction, iter_idx + 1)

    # Finally, run second loop body computing the updated `r_direction` at each
    # step.
    r_directions, _num_iters = tf.scan(
        second_loop, [alphas, position_deltas, gradient_deltas, inv_rhos],
        initializer=(r_direction, 0))

    return -r_directions[-1]

  return ps.cond(tf.reduce_any(state.history != 0),
                 _two_loop_algorithm,
                 lambda: -state.objective_gradient)


def _get_ragged_sizes(tensor, dtype=tf.int32):
  """Creates a tensor indicating the size of each component of
  a ragged dimension.

  For example:

  ```python
  element = tf.ragged.constant([[1,2], [3,4,5], [], [0]])
  _get_ragged_sizes(element)
  # => <tf.Tensor: shape=(4, 1), dtype=int32, numpy=
  #      array([[2],
  #             [3],
  #             [0],
  #             [1]], dtype=int32)>
  ```
  """
  return tf.reduce_sum(
            tf.ones_like(
              tensor,
              dtype=dtype),
            axis=-1)[..., tf.newaxis]


def _get_range_like_ragged(tensor, dtype=tf.int32):
  """Creates a batched range for the elements of the batched tensor.

  For example:

  ```python
  element = tf.ragged.constant([[1,2], [3,4,5], [], [0]])
  _get_range_like_ragged(element)
  # => <tf.RaggedTensor [[0, 1], [0, 1, 2], [], [0]]>

  Args:
    tensor: a RaggedTensor of shape `[n, None]`.

  Returns:
    A ragged tensor of shape `[n, None]` where the ragged dimensions
    match the ragged dimensions of `tensor`, and are a range from `0` to
    the size of the ragged dimension.
  ```
  """
  sizes = _get_ragged_sizes(tensor)
  flat_ranges = tf.ragged.range(
                  tf.reshape(
                    sizes,
                    [tf.reduce_prod(sizes.shape)]),
                  dtype=dtype)
  return tf.RaggedTensor.from_row_lengths(flat_ranges, sizes.shape[:-1])[0]


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
