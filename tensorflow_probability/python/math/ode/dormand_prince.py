# Copyright 2019 The TensorFlow Probability Authors.
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
"""Dormand-Prince solver."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import functools

import tensorflow.compat.v2 as tf
from tensorflow_probability.python.internal import assert_util
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.math.ode import base
from tensorflow_probability.python.math.ode import runge_kutta_util as rk_util
from tensorflow_probability.python.math.ode import util

from tensorflow.python.util import nest  # pylint: disable=g-direct-tensorflow-import

__all__ = [
    'DormandPrince',
]


# Parameters from Shampine (1986), section 4.
# Satisfies consistency condition for Runge-Kutta coefficients:
# assert all(np.allclose(a, sum(b)) for a, b in zip(alpha, beta))
_TABLEAU = rk_util.ButcherTableau(
    a=[1 / 5, 3 / 10, 4 / 5, 8 / 9, 1., 1.],
    b=[
        [1 / 5],
        [3 / 40, 9 / 40],
        [44 / 45, -56 / 15, 32 / 9],
        [19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729],
        [9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656],
        [35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84],
    ],
    c_sol=[35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84, 0],
    c_mid=[
        6025192743 / 30085553152 / 2, 0, 51252292925 / 65400821598 / 2,
        -2691868925 / 45128329728 / 2, 187940372067 / 1594534317056 / 2,
        -1776094331 / 19743644256 / 2, 11237099 / 235043384 / 2
    ],
    c_error=[
        1951 / 21600 - 35 / 384,
        0,
        22642 / 50085 - 500 / 1113,
        451 / 720 - 125 / 192,
        -12231 / 42400 - -2187 / 6784,
        649 / 6300 - 11 / 84,
        1 / 60,
    ]
)


# TODO(b/141875653) Add base class for generic RungeKutta methods.
# TODO(dkochkov) Consider changing default behavior to fixed max_num_steps.
class DormandPrince(base.Solver):
  """Dormand-Prince explicit solver for non-stiff ODEs.

  Implements 5th order Runge-Kutta with adaptive step size control
  and dense output, using the Dormand-Prince method. Similar to the 'dopri5'
  method of `scipy.integrate.ode` and MATLAB's `ode45`. For details see [1].
  For solver API see `tfp.math.ode.Solver`.

  #### References

  [1]: Shampine, L. F. (1986). Some practical runge-kutta formulas.
      Mathematics of Computation, 46(173), 135-150, doi:10.2307/2008219
  """

  ORDER = 5  # Order of the DormandPrince integrator.
  ODE_FN_EVALS_PER_STEP = 6

  def __init__(
      self,
      rtol=1e-3,
      atol=1e-6,
      first_step_size=1e-3,
      safety_factor=0.9,
      min_step_size_factor=0.1,
      max_step_size_factor=10.,
      max_num_steps=None,
      make_adjoint_solver_fn=None,
      validate_args=False,
      name='dormand_prince',
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
      first_step_size: Scalar float `Tensor` specifying the size of the
        first step.
        Default value: `1e-3`.
      safety_factor: Scalar positive float `Tensor`. At the end of every Runge
        Kutta step, the solver may choose to update the step size by applying a
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
      make_adjoint_solver_fn: Callable that takes no arguments that constructs a
        `Solver` instance. The created solver is used in the adjoint senstivity
        analysis to compute gradients (if they are requested).
        Default value: A callable that returns this solver.
      validate_args: Whether to validate input with asserts. If `validate_args`
        is `False` and the inputs are invalid, correct behavior is not
        guaranteed.
        Default value: `False`.
      name: Python `str` name prefixed to Ops created by this function.
        Default value: `None` (i.e., 'dormand_prince').
    """
    super(DormandPrince, self).__init__(
        make_adjoint_solver_fn=make_adjoint_solver_fn,
        validate_args=validate_args,
        name=name,
    )
    # The default values of `rtol` and `atol` match `scipy.integrate.solve_ivp`.
    self._rtol = rtol
    self._atol = atol
    self._safety_factor = safety_factor
    # TODO(dkochkov) Add option to choose initial step size automatically.
    self._first_step_size = first_step_size
    self._min_step_size_factor = min_step_size_factor
    self._max_step_size_factor = max_step_size_factor
    self._max_num_steps = max_num_steps

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
    # Static assertions
    del jacobian_fn, jacobian_sparsity  # not used by DormandPrince
    if batch_ndims is not None and batch_ndims != 0:
      raise NotImplementedError('For homogeneous batching use `batch_ndims=0`.')
    solution_times_by_solver = isinstance(solution_times, base.ChosenBySolver)

    with tf.name_scope(self._name):
      # (2) Convert to tensors, determine dtypes.
      p = self._prepare_common_params(initial_state, initial_time)

      max_num_steps = self._max_num_steps
      max_ode_fn_evals = self._max_num_steps
      if max_num_steps is not None:
        max_num_steps = tf.convert_to_tensor(max_num_steps, dtype=tf.int32)
        max_ode_fn_evals = max_num_steps * self.ODE_FN_EVALS_PER_STEP
      step_size = tf.convert_to_tensor(
          self._first_step_size, dtype=p.real_dtype)
      rtol = tf.convert_to_tensor(tf.cast(self._rtol, p.real_dtype))
      atol = tf.convert_to_tensor(tf.cast(self._atol, p.real_dtype))
      safety = tf.convert_to_tensor(self._safety_factor, dtype=p.real_dtype)
      # Use i(d)factor notation for increasing and decreasing factors.
      ifactor, dfactor = self._max_step_size_factor, self._min_step_size_factor
      ifactor = tf.convert_to_tensor(ifactor, dtype=p.real_dtype)
      dfactor = tf.convert_to_tensor(dfactor, dtype=p.real_dtype)

      solver_internal_state = previous_solver_internal_state
      if solver_internal_state is None:
        solver_internal_state = self._initialize_solver_internal_state(
            ode_fn=ode_fn,
            initial_state=p.initial_state,
            initial_time=p.initial_time,
        )

      num_solution_times = 0
      if solution_times_by_solver:
        final_time = tf.cast(solution_times.final_time, p.real_dtype)
        times_array = tf.TensorArray(
            p.real_dtype,
            size=num_solution_times,
            dynamic_size=True,
            element_shape=tf.TensorShape([]))
      else:
        solution_times = tf.cast(solution_times, p.real_dtype)
        util.error_if_not_vector(solution_times, 'solution_times')
        num_solution_times = tf.size(solution_times)
        times_array = tf.TensorArray(
            p.real_dtype,
            size=num_solution_times,
            dynamic_size=False,
            element_shape=[]).unstack(solution_times)

      solutions_arrays = nest.map_structure_up_to(
          p.state_dtypes,
          lambda shape, dtype: tf.TensorArray(  # pylint: disable=g-long-lambda
              dtype=dtype,
              size=num_solution_times,
              element_shape=shape,
              dynamic_size=solution_times_by_solver),
          p.state_shapes,
          p.state_dtypes,
      )

      rk_step = functools.partial(
          self._step,
          max_ode_fn_evals=max_ode_fn_evals,
          ode_fn=ode_fn,
          atol=atol,
          rtol=rtol,
          safety=safety,
          ifactor=ifactor,
          dfactor=dfactor
      )
      advance_to_solution_time = functools.partial(
          _advance_to_solution_time,
          times_array=solution_times,
          step_fn=rk_step,
          validate_args=self._validate_args
      )

      assert_ops = self._assert_ops(
          ode_fn=ode_fn,
          initial_time=p.initial_time,
          initial_state=p.initial_state,
          solution_times=solution_times,
          previous_solver_state=previous_solver_internal_state,
          rtol=rtol,
          atol=atol,
          first_step_size=step_size,
          safety_factor=safety,
          min_step_size_factor=ifactor,
          max_step_size_factor=dfactor,
          max_num_steps=max_num_steps,
          solution_times_by_solver=solution_times_by_solver
      )
      with tf.control_dependencies(assert_ops):
        ode_evals_by_now = 1 if self._validate_args else 0
        ode_evals_by_now += 1 if solver_internal_state is None else 0
        diagnostics = _DopriDiagnostics(
            num_ode_fn_evaluations=ode_evals_by_now, num_jacobian_evaluations=0,
            num_matrix_factorizations=0, status=0)

        if solution_times_by_solver:
          r = _dense_solutions_to_final_time(
              final_time=final_time,
              solver_state=solver_internal_state,
              diagnostics=diagnostics,
              step_fn=rk_step,
              ode_fn=ode_fn,
              times_array=times_array,
              solutions_arrays=solutions_arrays,
              validate_args=self._validate_args
          )
          solver_internal_state, diagnostics, times_array, solutions_arrays = r
        else:
          def iterate_cond(time_id, *_):
            return time_id < num_solution_times

          [
              _, solver_internal_state, diagnostics, solutions_arrays
          ] = tf.while_loop(iterate_cond, advance_to_solution_time, [
              0, solver_internal_state, diagnostics, solutions_arrays
          ])

        times = times_array.stack()
        stack_components = lambda x: x.stack()
        states = tf.nest.map_structure(stack_components, solutions_arrays)
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
    p = self._prepare_common_params(initial_state, initial_time)

    initial_derivative = ode_fn(p.initial_time, p.initial_state)
    initial_derivative = tf.nest.map_structure(tf.convert_to_tensor,
                                               initial_derivative)
    step_size = tf.convert_to_tensor(self._first_step_size, dtype=p.real_dtype)

    return _RungeKuttaSolverInternalState(
        current_state=p.initial_state,
        current_derivative=initial_derivative,
        last_step_start=p.initial_time,
        current_time=p.initial_time,
        step_size=step_size,
        interpolating_coefficients=[p.initial_state] * self.ORDER,
    )

  def _prepare_common_params(self, initial_state, initial_time):
    error_if_wrong_dtype = functools.partial(
        util.error_if_not_real_or_complex, identifier='initial_state')

    initial_state = tf.nest.map_structure(tf.convert_to_tensor, initial_state)
    tf.nest.map_structure(error_if_wrong_dtype, initial_state)

    state_dtypes = tf.nest.map_structure(lambda x: x.dtype, initial_state)
    state_shapes = tf.nest.map_structure(lambda x: x.shape, initial_state)
    common_state_dtype = dtype_util.common_dtype(initial_state)
    real_dtype = dtype_util.real_dtype(common_state_dtype)

    initial_time = tf.cast(initial_time, real_dtype)

    return util.Bunch(
        initial_state=initial_state,
        state_dtypes=state_dtypes,
        state_shapes=state_shapes,
        real_dtype=real_dtype,
        initial_time=initial_time,
    )

  def _step(
      self,
      solver_state,
      diagnostics,
      max_ode_fn_evals,
      ode_fn,
      atol,
      rtol,
      safety,
      ifactor,
      dfactor
  ):
    """Take an adaptive Runge-Kutta step.

    Args:
      solver_state: `_DopriSolverInternalState` - solver internal state.
      diagnostics: `_DopriDiagnostics` - info on the current `_solve` call.
      max_ode_fn_evals: Integer `Tensor` specifying the maximum number of ode_fn
        evaluations.
      ode_fn: Callable(t, y) -> dy_dt.
      atol: Absolute tolerance allowed, see `_solve` method for details.
      rtol: Relative tolerance allowed, see `_solve` method for details.
      safety: Safety factor, see `_solve` method for details.
      ifactor: Maximum factor by which the step size can increase, see `_solve`
        method for details.
      dfactor: Minimum factor by which the step size can decrease, see `_solve`
        method for details.

    Returns:
      solver_state: `_RungeKuttaSolverInternalState` holding new solver state.
        Note that the step might not advance the time if the error tolerance
        criterias were not met. In this case step_size is decreased.
      diagnostics: `_DopriDiagnostics` holding diagnostic values after RK step.
    """
    y0, f0, _, t0, dt, interp_coeff = solver_state
    assertion_ops = []
    # TODO(dkochkov) Profile performance impact of `control_dependencies` here.
    with tf.name_scope('assertions'):
      if self._max_num_steps is not None:
        check_max_num_steps = tf.debugging.assert_less(
            diagnostics.num_ode_fn_evaluations, max_ode_fn_evals,
            'max_num_steps exceeded')
        assertion_ops.append(check_max_num_steps)

      if self._validate_args:
        check_underflow = tf.debugging.assert_greater(
            t0 + dt, t0, 'underflow in dt')
        assert_finite = functools.partial(
            tf.debugging.assert_all_finite,
            message='non-finite values in solution')
        check_numerics = tf.nest.map_structure(assert_finite, y0)
        assertion_ops.append(check_underflow)
        assertion_ops.append(check_numerics)

    with tf.control_dependencies(assertion_ops):
      y1, f1, y1_error, k = rk_util.runge_kutta_step(
          ode_fn, y0, f0, t0, dt, _TABLEAU)

    with tf.name_scope('error_ratio'):
      # We use the same criteria for accepting step as in scipy.
      abs_y0 = tf.nest.map_structure(tf.abs, y0)
      abs_y1 = tf.nest.map_structure(tf.abs, y1)
      max_y_vals = tf.nest.map_structure(tf.math.maximum, abs_y0, abs_y1)
      ones_nest = rk_util.nest_constant(abs_y0)
      error_tol = rk_util.weighted_sum([atol, rtol], [ones_nest, max_y_vals])
      def scale_errors(error_component, tol_scale):
        abs_square_error = rk_util.abs_square(error_component)
        abs_square_tol_scale = rk_util.abs_square(tol_scale)
        return tf.divide(abs_square_error, abs_square_tol_scale)

      scaled_errors = tf.nest.map_structure(scale_errors, y1_error, error_tol)
      error_ratio = rk_util.nest_rms_norm(scaled_errors)
      accept_step = error_ratio <= 1

    with tf.name_scope('update/state'):
      y_next = rk_util.nest_where(accept_step, y1, y0)
      f_next = rk_util.nest_where(accept_step, f1, f0)
      t_next = tf.where(accept_step, t0 + dt, t0)

      new_coefficients = rk_util.rk_fourth_order_interpolation_coefficients(
          y0, y1, k, dt, _TABLEAU)
      interp_coeff = rk_util.nest_where(
          accept_step, new_coefficients, interp_coeff)

      dt_next = util.next_step_size(
          dt, self.ORDER, error_ratio, safety, dfactor, ifactor)
      solver_state = _RungeKuttaSolverInternalState(
          y_next, f_next, t0, t_next, dt_next, interp_coeff)

    diagnostics = diagnostics._replace(
        num_ode_fn_evaluations=diagnostics.num_ode_fn_evaluations +
        self.ODE_FN_EVALS_PER_STEP)

    return solver_state, diagnostics

  def _assert_ops(
      self,
      ode_fn,
      initial_time,
      initial_state,
      solution_times,
      previous_solver_state,
      rtol,
      atol,
      first_step_size,
      safety_factor,
      min_step_size_factor,
      max_step_size_factor,
      max_num_steps,
      solution_times_by_solver
  ):
    """Constructs dynamic assertions that validate input values to `_solve`."""
    assert_ops = []
    if not self._validate_args:
      return assert_ops
    if solution_times_by_solver:
      final_time = solution_times.final_time
      assert_ops.append(
          util.assert_positive(final_time - initial_time,
                               'final_time - initial_time'))
    else:
      assert_ops += [
          util.assert_increasing(solution_times, 'solution_times'),
          util.assert_nonnegative(solution_times[0] - initial_time,
                                  'solution_times[0] - initial_time'),
      ]
    if previous_solver_state is not None:
      state_diff = initial_state - previous_solver_state.current_state
      assert_states_match = assert_util.assert_near(
          tf.norm(state_diff), 0., message='`previous_solver_state` does not '
          'match the `initial_state`.')
      assert_ops.append(assert_states_match)
    if self._max_num_steps is not None:
      assert_ops.append(util.assert_positive(max_num_steps, 'max_num_steps'))
    assert_ops += [
        util.assert_positive(rtol, 'rtol'),
        util.assert_positive(atol, 'atol'),
        util.assert_positive(first_step_size, 'first_step_size'),
        util.assert_positive(safety_factor, 'safety_factor'),
        util.assert_positive(
            min_step_size_factor, 'min_step_size_factor'),
        util.assert_positive(
            max_step_size_factor, 'max_step_size_factor'),
    ]
    derivative = ode_fn(initial_time, initial_state)
    tf.nest.assert_same_structure(initial_state, derivative)
    return assert_ops


class _DopriDiagnostics(
    collections.namedtuple('_DopriDiagnostics', [
        'num_ode_fn_evaluations',
        'num_jacobian_evaluations',
        'num_matrix_factorizations',
        'status',
    ]), base.Diagnostics):
  """See `tfp.math.ode.Diagnostics`."""
  __slots__ = ()


class _RungeKuttaSolverInternalState(
    collections.namedtuple('_RungeKuttaSolverInternalState', [
        'current_state',
        'current_derivative',
        'last_step_start',
        'current_time',
        'step_size',
        'interpolating_coefficients'
    ])):
  """Internal state of the Runge Kutta solver.

  Properties:
    current_state: Possibly nested structure of `Tensor`s representing the
      solution at `current_time`.
    current_derivative: Possibly nested structure of `Tensor`s representing
      time derivative of the solution at `current_time`. Must have same
      structure as `current_state`.
    last_step_start: Scalar `Tensor` representing start of the last time step.
    current_time: Scalar `Tensor` representing time of the `current_state`.
    step_size: Scalar `Tensor` representing the current time step size used by
      the solver.
    interpolating_coefficients: List of `Tensor`s giving coefficients for
      polynomial interpolation between `last_step_start` and `current_time`.
  """
  __slots__ = ()


def _dense_solutions_to_final_time(
    final_time,
    solver_state,
    diagnostics,
    step_fn,
    ode_fn,
    times_array,
    solutions_arrays,
    validate_args=False
):
  """Integrates `solver_state` to `final_time`.

  Performs integration of the `solver_state` to `final_time` while saving
  solutions at all intermediate time steps. This corresponds to the expected
  behavior of `ChosenBySolver` option. The solution at `final_time` is obtained
  by interpolation and is set as a final state of the solver.

  Args:
    final_time: Floating `Tensor` representing the final time of integration.
    solver_state: `_DopriSolverInternalState` - initial solver state.
    diagnostics: `_DopriDiagnostics` - info on the current `_solve` call.
    step_fn: Partial `Dopri._step` method that performs a single step updating
      the `solver_state`, `diagnostics` and `solver_state`.
    ode_fn: Callable(t, y) -> dy_dt.
    times_array: `TensorArray` where time values are recorded.
    solutions_arrays: `TensorArray`s where solutions are recorded.
    validate_args: Python `bool` indicating whether to validate inputs.
      Default value: False.

  Returns:
    solver_state: `_RungeKuttaSolverInternalState` holding final solver state.
    diagnostics: `_DopriDiagnostics` holding diagnostic values.
    times_array: `TensorArray` with recorded solution times.
    solutions_arrays: `TensorArray`s with solution values at time corresponding
      to times_array.
  """

  def step_and_record(solver_state, diagnostics, solutions_arrays, times_array):
    y = solver_state.current_state
    time_id = times_array.size()
    solutions_arrays = _write_solution_components(y, solutions_arrays, time_id)
    times_array = times_array.write(time_id, solver_state.current_time)
    solver_state, diagnostics = step_fn(solver_state, diagnostics)
    return (solver_state, diagnostics, solutions_arrays, times_array)

  def step_cond(solver_internal_state, *_):
    return solver_internal_state.current_time <= final_time

  [
      solver_state, diagnostics, solutions_arrays, times_array
  ] = tf.while_loop(step_cond, step_and_record, [
      solver_state, diagnostics, solutions_arrays, times_array
  ])
  # Interpolating the last time point, updating the state and write results.
  y, coefficients = _interpolate_solution_at(
      final_time, solver_state, validate_args)
  dy_dt = ode_fn(final_time, y)
  dy_dt = tf.nest.map_structure(tf.convert_to_tensor, dy_dt)

  time_id = times_array.size()
  times_array = times_array.write(time_id, final_time)
  solutions_arrays = _write_solution_components(y, solutions_arrays, time_id)
  solver_state = _RungeKuttaSolverInternalState(
      current_state=y,
      current_derivative=dy_dt,
      last_step_start=solver_state.last_step_start,
      current_time=final_time,
      step_size=solver_state.step_size,
      interpolating_coefficients=coefficients
  )
  return solver_state, diagnostics, times_array, solutions_arrays


def _advance_to_solution_time(
    time_id,
    solver_state,
    diagnostics,
    solutions_arrays,
    times_array,
    step_fn,
    validate_args=False
):
  """Advances solution to the next time point, integrating as necessary.

  Performs a series of Runge-Kutta steps updating `solver_state` until the
  target time `times_array[time_id]` is reached. The value of the solution at
  the target time is then obtained using 4th order interpolation.

  Args:
    time_id: Integer `Tensor` - index of target time in `times_array`.
    solver_state: `_DopriSolverInternalState` - solver state.
    diagnostics: `_DopriDiagnostics` - info on the current `_solve` call.
    solutions_arrays: `TensorArray`s for storing solutions at desired time steps
    times_array: `TensorArray` that specifies the list times where solutions
      should be obtained.
    step_fn: Partial `Dopri._step` method that performs a single step updating
      the `solver_state`, `diagnostics` and `solver_state`.
    validate_args: Python `bool` indicating whether to validate inputs.
      Default value: False.

  Returns:
    time_id: Integer index of next target time.
    solver_state: Updated solver state.
    diagnostics: Updated info on current `_solve` call.
    solutions_array: `TensorArray` storing solutions up to returned `time_id`
      target time.
  """
  with tf.name_scope('advance_to_solution_time'):
    # Perform integration steps until we are past the desired time step.
    def step_cond(solver_state, _):
      return times_array[time_id] >= solver_state.current_time

    solver_state, diagnostics = tf.while_loop(
        step_cond, step_fn, (solver_state, diagnostics))
    y, _ = _interpolate_solution_at(
        times_array[time_id], solver_state, validate_args)
    solutions_arrays = _write_solution_components(y, solutions_arrays, time_id)
    return time_id + 1, solver_state, diagnostics, solutions_arrays


def _interpolate_solution_at(target_time, solver_state, validate_args=False):
  """Computes the solution at `target_time` using 4th order interpolation.

  Args:
    target_time: Floating `Tensor` specifying the time at which to obtain the
      solution. Must be within the interval of the last time step of the
      `solver_state`: `solver_state.last_step_start` <= `target_time` <=
      `solver_state.current_time`.
    solver_state: `_DopriSolverInternalState` - solver state.
    validate_args: Python `bool` indicating whether to validate inputs.
      Default value: False.

  Returns:
    solution: Solution at `target_time` obtained by interpolation.
    coefficients: Interpolating coefficients used to construct the solution.
  """
  coefficients = solver_state.interpolating_coefficients
  t0 = solver_state.last_step_start
  t1 = solver_state.current_time
  solution = rk_util.evaluate_interpolation(
      coefficients, t0, t1, target_time, validate_args)
  return solution, coefficients


def _write_solution_components(solution, solutions_arrays, time_id):
  """Writes individual `Tensor` components of `solution` to `solutions_arrays`.

  Args:
    solution: Possibly nested structure of `Tensor`s representing current
      solution to be written to `solutions_arrays`.
    solutions_arrays: List of `TensorArray`s where `solution` components are
      written. Must have the same number of elements as components in
      `solution`.
    time_id: Integer `Tensor` representing the index at which `solution`
      components are written to `solutions_arrays`.

  Returns:
    updated_arrays: List of `TensorArray`s whose components at index `time_id`
      now contain components of the `solution`.
  """
  tf.nest.assert_same_structure(solution, solutions_arrays)
  write_solution = lambda array, tensor: array.write(time_id, tensor)
  updated_arrays = tf.nest.map_structure(
      write_solution, solutions_arrays, solution)
  return updated_arrays
