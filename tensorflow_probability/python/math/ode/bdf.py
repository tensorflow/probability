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
"""Backward differentiation formula (BDF) solver."""

import collections
import functools

import numpy as np
import tensorflow.compat.v1 as tf1
import tensorflow.compat.v2 as tf

from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import prefer_static as ps
from tensorflow_probability.python.internal import tensorshape_util
from tensorflow_probability.python.math.ode import base
from tensorflow_probability.python.math.ode import bdf_util
from tensorflow_probability.python.math.ode import util
from tensorflow.python.util import deprecation  # pylint: disable=g-direct-tensorflow-import

__all__ = [
    'BDF',
]


class BDF(base.Solver):
  """Backward differentiation formula (BDF) solver for stiff ODEs.

  Implements the solver described in [Shampine and Reichelt (1997)][1], a
  variable step size, variable order (VSVO) BDF integrator with order varying
  between 1 and 5.

  #### Algorithm details

  Each step involves solving the following nonlinear equation by Newton's
  method:
  ```none
  0 = 1/1 * BDF(1, y)[n+1] + ... + 1/k * BDF(k, y)[n+1]
    - h ode_fn(t[n+1], y[n+1])
    - bdf_coefficients[k-1] * (1/1 + ... + 1/k) * (y[n+1] - y[n] - BDF(1, y)[n]
                                                          -  ... - BDF(k, y)[n])
  ```
  where `k >= 1` is the current order of the integrator, `h` is the current step
  size, `bdf_coefficients` is a list of numbers that parameterizes the method,
  and `BDF(m, y)` is the `m`-th order backward difference of the vector `y`. In
  particular, `BDF(0, y)[n] = y[n]` and
  `BDF(m + 1, y)[n] = BDF(m, y)[n] - BDF(m, y)[n - 1]` for `m >= 0`.

  Newton's method can fail because
  * the method has exceeded the maximum number of iterations,
  * the method is converging too slowly, or
  * the method is not expected to converge
  (the last two conditions are determined by approximating the Lipschitz
  constant associated with the iteration).

  When `evaluate_jacobian_lazily` is `True`, the solver avoids evaluating the
  Jacobian of the dynamics function as much as possible. In particular, Newton's
  method will try to use the Jacobian from a previous integration step. If
  Newton's method fails with an out-of-date Jacobian, the Jacobian is
  re-evaluated and Newton's method is restarted. If Newton's method fails and
  the Jacobian is already up-to-date, then the step size is decreased and
  Newton's method is restarted.

  Even if Newton's method converges, the solution it generates can still be
  rejected if it exceeds the specified error tolerance due to truncation error.
  In this case, the step size is decreased and Newton's method is restarted.

  #### References

  [1]: Lawrence F. Shampine and Mark W. Reichelt. The MATLAB ODE Suite. _SIAM
       Journal on Scientific Computing_ 18(1):1-22, 1997.
  """

  @deprecation.deprecated_args(
      '2021-11-01',
      'use_pfor_to_compute_jacobian is deprecated, and does nothing.',
      'use_pfor_to_compute_jacobian')
  def __init__(
      self,
      rtol=1e-3,
      atol=1e-6,
      first_step_size=None,
      safety_factor=0.9,
      min_step_size_factor=0.1,
      max_step_size_factor=10.,
      max_num_steps=None,
      max_order=bdf_util.MAX_ORDER,
      max_num_newton_iters=4,
      newton_tol_factor=0.1,
      newton_step_size_factor=0.5,
      bdf_coefficients=(-0.1850, -1. / 9., -0.0823, -0.0415, 0.),
      evaluate_jacobian_lazily=False,
      use_pfor_to_compute_jacobian=True,
      make_adjoint_solver_fn=None,
      validate_args=False,
      name='bdf',
  ):
    """Initializes the solver.

    Args:
      rtol: Optional float `Tensor` specifying an upper bound on relative error,
        per element of the dependent variable. The error tolerance for the next
        step is `tol = atol + rtol * abs(state)` where `state` is the computed
        state at the current step (see also `atol`). The next step is rejected
        if it incurs a local truncation error larger than `tol`.
        Default value: `1e-3`.
      atol: Optional float `Tensor` specifying an upper bound on absolute error,
        per element of the dependent variable (see also `rtol`).
        Default value: `1e-6`.
      first_step_size: Optional scalar float `Tensor` specifying the size of the
        first step. If unspecified, the size is chosen automatically.
        Default value: `None`.
      safety_factor: Scalar positive float `Tensor`. When Newton's method
        converges, the solver may choose to update the step size by applying a
        multiplicative factor to the current step size. This factor is `factor =
        clamp(factor_unclamped, min_step_size_factor, max_step_size_factor)`
        where `factor_unclamped = error_ratio**(-1. / (order + 1)) *
        safety_factor` (see also `min_step_size_factor` and
        `max_step_size_factor`). A small (respectively, large) value for the
        safety factor causes the solver to take smaller (respectively, larger)
        step sizes. A value larger than one, though not explicitly prohibited,
        is discouraged.
        Default value: `0.9`.
      min_step_size_factor: Scalar float `Tensor` (see `safety_factor`).
        Default value: `0.1`.
      max_step_size_factor: Scalar float `Tensor` (see `safety_factor`).
        Default value: `10.`.
      max_num_steps: Optional scalar integer `Tensor` specifying the maximum
        number of steps allowed (including rejected steps). If unspecified,
        there is no upper bound on the number of steps.
        Default value: `None`.
      max_order: Scalar integer `Tensor` taking values between 1 and 5
        (inclusive) specifying the maximum BDF order.
        Default value: `5`.
      max_num_newton_iters: Optional scalar integer `Tensor` specifying the
        maximum number of iterations per invocation of Newton's method. If
        unspecified, there is no upper bound on the number iterations.
        Default value: `4`.
      newton_tol_factor: Scalar float `Tensor` used to determine the stopping
        condition for Newton's method. In particular, Newton's method terminates
        when the distance to the root is estimated to be less than
        `newton_tol_factor * norm(atol + rtol * abs(state))` where `state` is
        the computed state at the current step.
        Default value: `0.1`.
      newton_step_size_factor: Scalar float `Tensor` specifying a multiplicative
        factor applied to the size of the integration step when Newton's method
        fails.
        Default value: `0.5`.
      bdf_coefficients: 1-D float `Tensor` with 5 entries that parameterize the
        solver. The default values are those proposed in [1].
        Default value: `(-0.1850, -1. / 9., -0.0823, -0.0415, 0.)`.
      evaluate_jacobian_lazily: Optional boolean specifying whether the Jacobian
        should be evaluated at each point in time or as needed (i.e., lazily).
        Default value: `True`.
      use_pfor_to_compute_jacobian: Boolean specifying whether or not to use
        parallel for in computing the Jacobian when `jacobian_fn` is not
        specified.
        Default value: `True`.
      make_adjoint_solver_fn: Callable that takes no arguments that constructs a
        `Solver` instance. The created solver is used in the adjoint senstivity
        analysis to compute gradients (if they are requested).
        Default value: A callable that returns this solver.
      validate_args: Whether to validate input with asserts. If `validate_args`
        is `False` and the inputs are invalid, correct behavior is not
        guaranteed.
        Default value: `False`.
      name: Python `str` name prefixed to Ops created by this function.
        Default value: `None` (i.e., 'bdf').
    """
    del use_pfor_to_compute_jacobian
    super(BDF, self).__init__(
        make_adjoint_solver_fn=make_adjoint_solver_fn,
        validate_args=validate_args,
        name=name,
    )
    # The default values of `rtol` and `atol` match `scipy.integrate.solve_ivp`.
    self._rtol = rtol
    self._atol = atol
    self._first_step_size = first_step_size
    self._safety_factor = safety_factor
    self._min_step_size_factor = min_step_size_factor
    self._max_step_size_factor = max_step_size_factor
    self._max_num_steps = max_num_steps
    self._max_order = max_order
    self._max_num_newton_iters = max_num_newton_iters
    self._newton_tol_factor = newton_tol_factor
    self._newton_step_size_factor = newton_step_size_factor
    self._bdf_coefficients = bdf_coefficients
    self._evaluate_jacobian_lazily = evaluate_jacobian_lazily

  def _solve(
      self,
      ode_fn,
      initial_time,
      initial_state,
      solution_times,
      jacobian_fn=None,
      jacobian_sparsity=None,
      batch_ndims=None,
      previous_solver_internal_state=None,
  ):
    # This function is comprised of the following sequential stages:
    # (1) Make static assertions.
    # (2) Initialize variables.
    # (3) Make non-static assertions.
    # (4) Solve up to final time.
    # (5) Return `Results` object.
    #
    # The stages can be found in the code by searching for (n) where n=1..5.
    #
    # By static vs. non-static assertions (see stages 1 and 3), we mean
    # assertions that can be made before the graph is run vs. those that can
    # only be made at run time. The latter are constructed as a list of
    # tf.Assert operations by the function `assert_ops` (see below).
    #
    # If `solution_times` is specified as a `Tensor`, stage 4 consists of three
    # nested loops, which can be conceptually understood as follows:
    # ```
    # current_time, current_state = initial_time, initial_state
    # order, step_size = 1, first_step_size
    # for solution_time in solution_times:
    #   while current_time < solution_time:
    #     while True:
    #       next_time = current_time + step_size
    #       next_state, error = (
    #           solve_nonlinear_equation_to_get_approximate_state_at_next_time(
    #           current_time, current_state, next_time, order))
    #       if error < tolerance:
    #         current_time, current_state = next_time, next_state
    #         order, step_size = (
    #           maybe_update_order_and_step_size(order, step_size))
    #         break
    #       else:
    #         step_size = decrease_step_size(step_size)
    # ```
    # The outermost loop advances the solver to the next `solution_time` (see
    # `advance_to_solution_time`). The middle loop advances the solver by a
    # small timestep (see `step`). The innermost loop determines the size of
    # that timestep (see `maybe_step`).
    #
    # If `solution_times` is specified as
    # `tfp.math.ode.ChosenBySolver(final_time)`, the outermost loop is skipped
    # and `solution_time` in the middle loop is replaced by `final_time`.

    def advance_to_solution_time(n, diagnostics, iterand, solver_internal_state,
                                 state_vec_array, time_array):
      """Takes multiple steps to advance time to `solution_times[n]`."""

      def step_cond(next_time, diagnostics, iterand, *_):
        return (iterand.time < next_time) & (tf.equal(diagnostics.status, 0))

      nth_solution_time = solution_time_array.read(n)
      [
          _, diagnostics, iterand, solver_internal_state, state_vec_array,
          time_array
      ] = tf.while_loop(step_cond, step, [
          nth_solution_time, diagnostics, iterand, solver_internal_state,
          state_vec_array, time_array
      ])
      state_vec_array = state_vec_array.write(
          n, solver_internal_state.backward_differences[0])
      time_array = time_array.write(n, nth_solution_time)
      return (n + 1, diagnostics, iterand, solver_internal_state,
              state_vec_array, time_array)

    def step(next_time, diagnostics, iterand, solver_internal_state,
             state_vec_array, time_array):
      """Takes a single step."""
      distance_to_next_time = next_time - iterand.time
      overstepped = iterand.new_step_size > distance_to_next_time
      iterand = iterand._replace(
          new_step_size=tf1.where(overstepped, distance_to_next_time,
                                  iterand.new_step_size),
          should_update_step_size=overstepped | iterand.should_update_step_size)

      if not self._evaluate_jacobian_lazily:
        diagnostics = diagnostics._replace(
            num_jacobian_evaluations=diagnostics.num_jacobian_evaluations + 1)
        iterand = iterand._replace(
            jacobian_mat=jacobian_fn_mat(
                iterand.time, solver_internal_state.backward_differences[0]),
            jacobian_is_up_to_date=True)

      def maybe_step_cond(accepted, diagnostics, *_):
        return tf.logical_not(accepted) & tf.equal(diagnostics.status, 0)

      _, diagnostics, iterand, solver_internal_state = tf.while_loop(
          maybe_step_cond, maybe_step,
          [False, diagnostics, iterand, solver_internal_state])

      if solution_times_chosen_by_solver:
        state_vec_array = state_vec_array.write(
            state_vec_array.size(),
            solver_internal_state.backward_differences[0])
        time_array = time_array.write(time_array.size(), iterand.time)

      return (next_time, diagnostics, iterand, solver_internal_state,
              state_vec_array, time_array)

    def maybe_step(accepted, diagnostics, iterand, solver_internal_state):
      """Takes a single step only if the outcome has a low enough error."""
      [
          num_jacobian_evaluations, num_matrix_factorizations,
          num_ode_fn_evaluations, status
      ] = diagnostics
      [
          jacobian_mat, jacobian_is_up_to_date, new_step_size, num_steps,
          num_steps_same_size, should_update_jacobian, should_update_step_size,
          time, unitary, upper
      ] = iterand
      [backward_differences, order, step_size] = solver_internal_state

      if max_num_steps is not None:
        status = tf1.where(tf.equal(num_steps, max_num_steps), -1, 0)

      backward_differences = tf1.where(
          should_update_step_size,
          bdf_util.interpolate_backward_differences(backward_differences, order,
                                                    new_step_size / step_size),
          backward_differences)
      step_size = tf1.where(should_update_step_size, new_step_size, step_size)
      should_update_factorization = should_update_step_size
      num_steps_same_size = tf1.where(should_update_step_size, 0,
                                      num_steps_same_size)

      def update_factorization():
        return bdf_util.newton_qr(jacobian_mat,
                                  newton_coefficients_array.read(order),
                                  step_size)

      if self._evaluate_jacobian_lazily:

        def update_jacobian_and_factorization():
          new_jacobian_mat = jacobian_fn_mat(time, backward_differences[0])
          new_unitary, new_upper = update_factorization()
          return [
              new_jacobian_mat, True, num_jacobian_evaluations + 1, new_unitary,
              new_upper
          ]

        def maybe_update_factorization():
          new_unitary, new_upper = tf.cond(
              should_update_factorization,
              update_factorization, lambda: [unitary, upper])
          return [
              jacobian_mat, jacobian_is_up_to_date, num_jacobian_evaluations,
              new_unitary, new_upper
          ]

        [
            jacobian_mat, jacobian_is_up_to_date, num_jacobian_evaluations,
            unitary, upper
        ] = tf.cond(should_update_jacobian, update_jacobian_and_factorization,
                    maybe_update_factorization)
      else:
        unitary, upper = update_factorization()
        num_matrix_factorizations += 1

      tol = p.atol + p.rtol * tf.abs(backward_differences[0])
      newton_tol = newton_tol_factor * tf.norm(tol)

      [
          newton_converged, next_backward_difference, next_state_vec,
          newton_num_iters
      ] = bdf_util.newton(backward_differences, max_num_newton_iters,
                          newton_coefficients_array.read(order), p.ode_fn_vec,
                          order, step_size, time, newton_tol, unitary, upper)
      num_steps += 1
      num_ode_fn_evaluations += newton_num_iters

      # If Newton's method failed and the Jacobian was up to date, decrease the
      # step size.
      newton_failed = tf.logical_not(newton_converged)
      should_update_step_size = newton_failed & jacobian_is_up_to_date
      new_step_size = step_size * tf1.where(should_update_step_size,
                                            newton_step_size_factor, 1.)

      # If Newton's method failed and the Jacobian was NOT up to date, update
      # the Jacobian.
      should_update_jacobian = newton_failed & tf.logical_not(
          jacobian_is_up_to_date)

      error_ratio = tf1.where(
          newton_converged,
          bdf_util.error_ratio(next_backward_difference,
                               error_coefficients_array.read(order), tol),
          np.nan)
      accepted = error_ratio < 1.
      converged_and_rejected = newton_converged & tf.logical_not(accepted)

      # If Newton's method converged but the solution was NOT accepted, decrease
      # the step size.
      new_step_size = tf1.where(
          converged_and_rejected,
          util.next_step_size(step_size, order, error_ratio, p.safety_factor,
                              min_step_size_factor, max_step_size_factor),
          new_step_size)
      should_update_step_size = should_update_step_size | converged_and_rejected

      # If Newton's method converged and the solution was accepted, update the
      # matrix of backward differences.
      time = tf1.where(accepted, time + step_size, time)
      backward_differences = tf1.where(
          accepted,
          bdf_util.update_backward_differences(backward_differences,
                                               next_backward_difference,
                                               next_state_vec, order),
          backward_differences)
      jacobian_is_up_to_date = jacobian_is_up_to_date & tf.logical_not(accepted)
      num_steps_same_size = tf1.where(accepted, num_steps_same_size + 1,
                                      num_steps_same_size)

      # Order and step size are only updated if we have taken strictly more than
      # order + 1 steps of the same size. This is to prevent the order from
      # being throttled.
      should_update_order_and_step_size = accepted & (
          num_steps_same_size > order + 1)

      backward_differences_array = tf.TensorArray(
          backward_differences.dtype,
          size=bdf_util.MAX_ORDER + 3,
          clear_after_read=False,
          element_shape=next_backward_difference.shape).unstack(
              backward_differences)
      new_order = order
      new_error_ratio = error_ratio
      for offset in [-1, +1]:
        proposed_order = tf.clip_by_value(order + offset, 1, max_order)
        proposed_error_ratio = bdf_util.error_ratio(
            backward_differences_array.read(proposed_order + 1),
            error_coefficients_array.read(proposed_order), tol)
        proposed_error_ratio_is_lower = proposed_error_ratio < new_error_ratio
        new_order = tf1.where(
            should_update_order_and_step_size & proposed_error_ratio_is_lower,
            proposed_order, new_order)
        new_error_ratio = tf1.where(
            should_update_order_and_step_size & proposed_error_ratio_is_lower,
            proposed_error_ratio, new_error_ratio)
      order = new_order
      error_ratio = new_error_ratio

      new_step_size = tf1.where(
          should_update_order_and_step_size,
          util.next_step_size(step_size, order, error_ratio, p.safety_factor,
                              min_step_size_factor, max_step_size_factor),
          new_step_size)
      should_update_step_size = (
          should_update_step_size | should_update_order_and_step_size)

      diagnostics = _BDFDiagnostics(num_jacobian_evaluations,
                                    num_matrix_factorizations,
                                    num_ode_fn_evaluations, status)
      iterand = _BDFIterand(jacobian_mat, jacobian_is_up_to_date, new_step_size,
                            num_steps, num_steps_same_size,
                            should_update_jacobian, should_update_step_size,
                            time, unitary, upper)
      solver_internal_state = _BDFSolverInternalState(backward_differences,
                                                      order, step_size)
      return accepted, diagnostics, iterand, solver_internal_state

    # (1) Make static assertions.
    # TODO(b/138304296): Support specifying Jacobian sparsity patterns.
    if jacobian_sparsity is not None:
      raise NotImplementedError('The BDF solver does not support specifying '
                                'Jacobian sparsity patterns.')
    if batch_ndims is not None and batch_ndims != 0:
      raise NotImplementedError('The BDF solver does not support batching.')
    solution_times_chosen_by_solver = (
        isinstance(solution_times, base.ChosenBySolver))

    with tf.name_scope(self._name):

      # (2) Convert to tensors.
      p = self._prepare_common_params(
          ode_fn=ode_fn,
          initial_state=initial_state,
          initial_time=initial_time,
      )

      if jacobian_fn is None and dtype_util.is_complex(p.common_state_dtype):
        raise NotImplementedError('The BDF solver does not support automatic '
                                  'Jacobian computations for complex dtypes.')

      # Convert everything to operate on a single, concatenated vector form.
      jacobian_fn_mat = util.get_jacobian_fn_mat(
          jacobian_fn,
          p.ode_fn_vec,
          p.state_shape,
          dtype=p.common_state_dtype,
      )

      num_solution_times = 0
      if solution_times_chosen_by_solver:
        final_time = tf.cast(solution_times.final_time, p.real_dtype)
      else:
        solution_times = tf.cast(solution_times, p.real_dtype)
        final_time = tf.reduce_max(solution_times)
        num_solution_times = ps.size(solution_times)
        solution_time_array = tf.TensorArray(
            solution_times.dtype, size=num_solution_times,
            element_shape=[]).unstack(solution_times)
        util.error_if_not_vector(solution_times, 'solution_times')
      min_step_size_factor = tf.convert_to_tensor(
          self._min_step_size_factor, dtype=p.real_dtype)
      max_step_size_factor = tf.convert_to_tensor(
          self._max_step_size_factor, dtype=p.real_dtype)
      max_num_steps = self._max_num_steps
      if max_num_steps is not None:
        max_num_steps = tf.convert_to_tensor(max_num_steps, dtype=tf.int32)
      max_order = tf.convert_to_tensor(self._max_order, dtype=tf.int32)
      max_num_newton_iters = self._max_num_newton_iters
      if max_num_newton_iters is not None:
        max_num_newton_iters = tf.convert_to_tensor(
            max_num_newton_iters, dtype=tf.int32)
      newton_tol_factor = tf.convert_to_tensor(
          self._newton_tol_factor, dtype=p.real_dtype)
      newton_step_size_factor = tf.convert_to_tensor(
          self._newton_step_size_factor, dtype=p.real_dtype)
      newton_coefficients, error_coefficients = self._prepare_coefficients(
          p.common_state_dtype)
      if self._validate_args:
        final_time = tf.ensure_shape(final_time, [])
        min_step_size_factor = tf.ensure_shape(min_step_size_factor, [])
        max_step_size_factor = tf.ensure_shape(max_step_size_factor, [])
        if max_num_steps is not None:
          max_num_steps = tf.ensure_shape(max_num_steps, [])
        max_order = tf.ensure_shape(max_order, [])
        if max_num_newton_iters is not None:
          max_num_newton_iters = tf.ensure_shape(max_num_newton_iters, [])
        newton_tol_factor = tf.ensure_shape(newton_tol_factor, [])
        newton_step_size_factor = tf.ensure_shape(newton_step_size_factor, [])
      newton_coefficients_array = tf.TensorArray(
          newton_coefficients.dtype,
          size=bdf_util.MAX_ORDER + 1,
          clear_after_read=False,
          element_shape=[]).unstack(newton_coefficients)
      error_coefficients_array = tf.TensorArray(
          error_coefficients.dtype,
          size=bdf_util.MAX_ORDER + 1,
          clear_after_read=False,
          element_shape=[]).unstack(error_coefficients)
      solver_internal_state = previous_solver_internal_state
      if solver_internal_state is None:
        solver_internal_state = self._initialize_solver_internal_state(
            ode_fn=ode_fn,
            initial_state=initial_state,
            initial_time=initial_time,
        )
      state_vec_array = tf.TensorArray(
          p.common_state_dtype,
          size=num_solution_times,
          dynamic_size=solution_times_chosen_by_solver,
          element_shape=p.initial_state_vec.shape)
      time_array = tf.TensorArray(
          p.real_dtype,
          size=num_solution_times,
          dynamic_size=solution_times_chosen_by_solver,
          element_shape=tf.TensorShape([]))
      diagnostics = _BDFDiagnostics(
          num_jacobian_evaluations=0,
          num_matrix_factorizations=0,
          num_ode_fn_evaluations=0,
          status=0)
      iterand = _BDFIterand(
          jacobian_mat=tf.zeros([p.num_odes, p.num_odes],
                                dtype=p.common_state_dtype),
          jacobian_is_up_to_date=False,
          new_step_size=solver_internal_state.step_size,
          num_steps=0,
          num_steps_same_size=0,
          should_update_jacobian=True,
          should_update_step_size=False,
          time=p.initial_time,
          unitary=tf.zeros([p.num_odes, p.num_odes],
                           dtype=p.common_state_dtype),
          upper=tf.zeros([p.num_odes, p.num_odes], dtype=p.common_state_dtype),
      )

      # (3) Make non-static assertions.
      with tf.control_dependencies(
          self._assert_ops(
              previous_solver_internal_state=previous_solver_internal_state,
              initial_state_vec=p.initial_state_vec,
              final_time=final_time,
              initial_time=p.initial_time,
              solution_times=solution_times,
              max_num_steps=max_num_steps,
              max_num_newton_iters=max_num_newton_iters,
              atol=p.atol,
              rtol=p.rtol,
              first_step_size=solver_internal_state.step_size,
              safety_factor=p.safety_factor,
              min_step_size_factor=min_step_size_factor,
              max_step_size_factor=max_step_size_factor,
              max_order=max_order,
              newton_tol_factor=newton_tol_factor,
              newton_step_size_factor=newton_step_size_factor,
              solution_times_chosen_by_solver=solution_times_chosen_by_solver,
          )):

        # (4) Solve up to final time.
        if solution_times_chosen_by_solver:

          def step_cond(next_time, diagnostics, iterand, *_):
            return (iterand.time < next_time) & (
                tf.equal(diagnostics.status, 0))

          [
              _, diagnostics, iterand, solver_internal_state, state_vec_array,
              time_array
          ] = tf.while_loop(step_cond, step, [
              final_time, diagnostics, iterand, solver_internal_state,
              state_vec_array, time_array
          ])

        else:

          def advance_to_solution_time_cond(n, diagnostics, *_):
            return (n < num_solution_times) & (tf.equal(diagnostics.status, 0))

          [
              _, diagnostics, iterand, solver_internal_state, state_vec_array,
              time_array
          ] = tf.while_loop(advance_to_solution_time_cond,
                            advance_to_solution_time, [
                                0, diagnostics, iterand, solver_internal_state,
                                state_vec_array, time_array
                            ])

        # (6) Return `Results` object.
        states = util.get_state_from_vec(state_vec_array.stack(), p.state_shape)
        times = time_array.stack()
        if not solution_times_chosen_by_solver:
          tensorshape_util.set_shape(times, solution_times.shape)
          tf.nest.map_structure(
              lambda s, ini_s: tensorshape_util.set_shape(  # pylint: disable=g-long-lambda
                  s,
                  tensorshape_util.concatenate(solution_times.shape, ini_s.shape
                                              )), states, p.initial_state)
        return base.Results(
            times=times,
            states=states,
            diagnostics=diagnostics,
            solver_internal_state=solver_internal_state)

  def _initialize_solver_internal_state(
      self,
      ode_fn,
      initial_time,
      initial_state,
  ):
    p = self._prepare_common_params(
        ode_fn=ode_fn,
        initial_state=initial_state,
        initial_time=initial_time,
    )

    first_step_size = self._first_step_size
    if first_step_size is None:
      _, error_coefficients = self._prepare_coefficients(p.common_state_dtype)
      first_step_size = bdf_util.first_step_size(
          p.atol, error_coefficients[1], p.initial_state_vec,
          p.initial_time, p.ode_fn_vec, p.rtol, p.safety_factor)
    first_step_size = tf.convert_to_tensor(first_step_size, dtype=p.real_dtype)
    if self._validate_args:
      first_step_size = tf.ensure_shape(first_step_size, [])

    first_order_backward_difference = p.ode_fn_vec(
        p.initial_time, p.initial_state_vec) * tf.cast(first_step_size,
                                                       p.common_state_dtype)
    backward_differences = tf.concat(
        [
            p.initial_state_vec[tf.newaxis, :],
            first_order_backward_difference[tf.newaxis, :],
            tf.zeros(
                ps.stack([bdf_util.MAX_ORDER + 1, p.num_odes]),
                dtype=p.common_state_dtype),
        ],
        axis=0,
    )
    return _BDFSolverInternalState(
        backward_differences=backward_differences,
        order=tf.ones([], tf.int32),
        step_size=first_step_size)

  def _prepare_coefficients(self, dtype):
    bdf_coefficients = tf.concat(
        [[0.], tf.cast(self._bdf_coefficients, dtype=dtype)], 0)
    if self._validate_args:
      bdf_coefficients = tf.ensure_shape(bdf_coefficients, [6])
    util.error_if_not_vector(bdf_coefficients, 'bdf_coefficients')
    np_dtype = dtype_util.as_numpy_dtype(bdf_coefficients.dtype)
    newton_coefficients = 1. / (
        (1. - bdf_coefficients) * bdf_util.RECIPROCAL_SUMS.astype(np_dtype))
    error_coefficients = (
        bdf_coefficients * bdf_util.RECIPROCAL_SUMS.astype(np_dtype) + 1. /
        (bdf_util.ORDERS.astype(np_dtype) + 1))

    return newton_coefficients, error_coefficients

  def _prepare_common_params(self, ode_fn, initial_state, initial_time):
    error_if_wrong_dtype = functools.partial(
        util.error_if_not_real_or_complex, identifier='initial_state')

    initial_state = tf.nest.map_structure(tf.convert_to_tensor, initial_state)
    tf.nest.map_structure(error_if_wrong_dtype, initial_state)

    state_shape = tf.nest.map_structure(ps.shape, initial_state)
    common_state_dtype = dtype_util.common_dtype(initial_state)
    real_dtype = dtype_util.real_dtype(common_state_dtype)
    # Use tf.cast instead of tf.convert_to_tensor for differentiable
    # parameters because the tf.custom_gradient decorator converts raw floats
    # into tf.float32, which cannot be converted to tf.float64.
    initial_time = tf.cast(initial_time, real_dtype)
    if self._validate_args:
      initial_time = tf.ensure_shape(initial_time, [])

    rtol = tf.convert_to_tensor(self._rtol, dtype=real_dtype)
    atol = tf.convert_to_tensor(self._atol, dtype=real_dtype)
    safety_factor = tf.convert_to_tensor(
        self._safety_factor, dtype=real_dtype)

    if self._validate_args:
      safety_factor = tf.ensure_shape(safety_factor, [])

    # Convert everything to operate on a single, concatenated vector form.
    initial_state_vec = util.get_state_vec(initial_state)
    ode_fn_vec = util.get_ode_fn_vec(ode_fn, state_shape)
    num_odes = ps.size(initial_state_vec)

    return util.Bunch(
        initial_state=initial_state,
        initial_time=initial_time,
        common_state_dtype=common_state_dtype,
        real_dtype=real_dtype,
        rtol=rtol,
        atol=atol,
        safety_factor=safety_factor,
        state_shape=state_shape,
        initial_state_vec=initial_state_vec,
        ode_fn_vec=ode_fn_vec,
        num_odes=num_odes,
    )

  def _assert_ops(
      self,
      previous_solver_internal_state,
      initial_state_vec,
      final_time,
      initial_time,
      solution_times,
      max_num_steps,
      max_num_newton_iters,
      atol,
      rtol,
      first_step_size,
      safety_factor,
      min_step_size_factor,
      max_step_size_factor,
      max_order,
      newton_tol_factor,
      newton_step_size_factor,
      solution_times_chosen_by_solver,
  ):
    """Creates a list of assert operations."""
    if not self._validate_args:
      return []
    assert_ops = []
    if previous_solver_internal_state is not None:
      assert_initial_state_matches_previous_solver_internal_state = (
          tf.debugging.assert_near(
              tf.norm(
                  initial_state_vec -
                  previous_solver_internal_state.backward_differences[0],
                  np.inf),
              0.,
              message='`previous_solver_internal_state` does not match '
              '`initial_state`.'))
      assert_ops.append(
          assert_initial_state_matches_previous_solver_internal_state)
    assert_ops.append(
        util.assert_positive(final_time - initial_time,
                             'final_time - initial_time'))
    if not solution_times_chosen_by_solver:
      assert_ops += [
          util.assert_increasing(solution_times, 'solution_times'),
          util.assert_nonnegative(solution_times[0] - initial_time,
                                  'solution_times[0] - initial_time'),
      ]
    if max_num_steps is not None:
      assert_ops.append(util.assert_positive(max_num_steps, 'max_num_steps'))
    if max_num_newton_iters is not None:
      assert_ops.append(
          util.assert_positive(max_num_newton_iters, 'max_num_newton_iters'))
    assert_ops += [
        util.assert_positive(rtol, 'rtol'),
        util.assert_positive(atol, 'atol'),
        util.assert_positive(first_step_size, 'first_step_size'),
        util.assert_positive(safety_factor, 'safety_factor'),
        util.assert_positive(min_step_size_factor, 'min_step_size_factor'),
        util.assert_positive(max_step_size_factor, 'max_step_size_factor'),
        tf.Assert((max_order >= 1) & (max_order <= bdf_util.MAX_ORDER), [
            '`max_order` must be between 1 and {}.'.format(bdf_util.MAX_ORDER)
        ]),
        util.assert_positive(newton_tol_factor, 'newton_tol_factor'),
        util.assert_positive(newton_step_size_factor,
                             'newton_step_size_factor'),
    ]
    return assert_ops


class _BDFDiagnostics(
    collections.namedtuple('_BDFDiagnostics', [
        'num_jacobian_evaluations',
        'num_matrix_factorizations',
        'num_ode_fn_evaluations',
        'status',
    ]), base.Diagnostics):
  """See `tfp.math.ode.Diagnostics`."""
  __slots__ = ()


_BDFIterand = collections.namedtuple('_BDFIterand', [
    'jacobian_mat',
    'jacobian_is_up_to_date',
    'new_step_size',
    'num_steps',
    'num_steps_same_size',
    'should_update_jacobian',
    'should_update_step_size',
    'time',
    'unitary',
    'upper',
])


class _BDFSolverInternalState(
    collections.namedtuple('_BDFSolverInternalState', [
        'backward_differences',
        'order',
        'step_size',
    ])):
  """Returned by the solver to warm start future invocations.

  Properties:
    backward_differences: 2-D `Tensor` corresponding to a matrix of backward
      differences upon termination. The `i`-th row contains the vector
      `BDF(i, y)` (see `BDF` for a definition). In particular, the `0`-th row
      contains `BDF(0, y) = y`, the state.
    order: Scalar integer `Tensor` containing the order used by the solver upon
      termination.
    step_size: Scalar float `Tensor` containing the step size used by the solver
      upon termination.
  """
  __slots__ = ()
