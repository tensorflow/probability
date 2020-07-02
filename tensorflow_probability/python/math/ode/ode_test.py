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
"""Tests for TensorFlow Probability ODE solvers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import numpy as np
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp

from tensorflow_probability.python.internal import test_util
from tensorflow.python import tf2  # pylint: disable=g-direct-tensorflow-import

_RTOL = 1e-8
_ATOL = 1e-12


class StepSizeHeuristicAdjointSolver(tfp.math.ode.Solver):
  """Adjoint solver which propagates step size between solves."""

  def __init__(self, make_solver_fn, first_step_size):
    self._first_step_size = first_step_size
    self._make_solver_fn = make_solver_fn
    solver = make_solver_fn(first_step_size)
    super(StepSizeHeuristicAdjointSolver, self).__init__(
        use_pfor_to_compute_jacobian=solver._use_pfor_to_compute_jacobian,
        validate_args=solver._validate_args,
        make_adjoint_solver_fn=None,
        name='Adjoint' + solver.name,
    )

  def _solve(self, **kwargs):
    step_size = kwargs.pop('previous_solver_internal_state')
    results = self._make_solver_fn(step_size).solve(**kwargs)
    return results._replace(
        solver_internal_state=results.solver_internal_state.step_size)

  def _initialize_solver_internal_state(self, **kwargs):
    del kwargs
    return self._first_step_size

  def _adjust_solver_internal_state_for_state_jump(self, **kwargs):
    return kwargs['previous_solver_internal_state']


@test_util.test_all_tf_execution_regimes
@parameterized.named_parameters([
    ('bdf', tfp.math.ode.BDF),
    ('dormand_prince', tfp.math.ode.DormandPrince)])
class NonStiffTest(test_util.TestCase):

  def test_zero_dims(self, solver):
    ode_fn = lambda time, state: -state
    initial_time = 0.
    initial_state = np.float64(1.)
    jacobian = np.float64([[-1.]])
    solver_instance = solver(rtol=_RTOL, atol=_ATOL)
    results = solver_instance.solve(
        ode_fn,
        initial_time,
        initial_state,
        solution_times=tfp.math.ode.ChosenBySolver(1.),
        jacobian_fn=jacobian)
    times, states = self.evaluate([results.times, results.states])
    states_exact = np.exp(-times) * initial_state
    self.assertAllClose(states, states_exact)

  def test_state_with_matrix_shape(self, solver):
    ode_fn = lambda time, state: -state
    initial_time = 0.
    initial_state = np.float64([[1., 2.], [3., 4.]])
    jacobian = np.reshape(
        np.diag(-np.ones([4])),
        np.concatenate([initial_state.shape, initial_state.shape]))
    solver_instance = solver(rtol=_RTOL, atol=_ATOL)
    results = solver_instance.solve(
        ode_fn,
        initial_time,
        initial_state,
        solution_times=tfp.math.ode.ChosenBySolver(1.),
        jacobian_fn=jacobian)
    times, states = self.evaluate([results.times, results.states])
    states_exact = np.exp(-times)[:, np.newaxis, np.newaxis] * initial_state
    self.assertAllClose(states, states_exact)

  def test_ode_fn_is_zero(self, solver):
    initial_time = 0.
    initial_state = np.float64([1., 2., 3.])
    ode_fn = lambda time, state: np.zeros_like(initial_state)
    jacobian = np.zeros((3, 3), dtype=np.float64)
    solver_instance = solver(rtol=_RTOL, atol=_ATOL)
    results = solver_instance.solve(
        ode_fn,
        initial_time,
        initial_state,
        solution_times=tfp.math.ode.ChosenBySolver(1.),
        jacobian_fn=jacobian)
    times, states = self.evaluate([results.times, results.states])
    states_exact = np.ones([times.size, initial_state.size]) * initial_state
    self.assertAllClose(states, states_exact)

  def test_linear_ode(self, solver):
    jacobian_diag_part = np.float64([-0.5, -1.])
    ode_fn = lambda time, state: jacobian_diag_part * state
    initial_time = 0.
    initial_state = np.float64([1., 2.])
    jacobian = np.diag(jacobian_diag_part)
    solver_instance = solver(rtol=_RTOL, atol=_ATOL)
    results = solver_instance.solve(
        ode_fn,
        initial_time,
        initial_state,
        solution_times=tfp.math.ode.ChosenBySolver(1.),
        jacobian_fn=jacobian)
    times, states = self.evaluate([results.times, results.states])
    states_exact = np.exp(jacobian_diag_part[np.newaxis, :] *
                          times[:, np.newaxis]) * initial_state
    self.assertAllClose(states, states_exact)

  def test_linear_ode_jacobian_fn_unspecified(self, solver):
    jacobian_diag_part = np.float64([-0.5, -1.])
    ode_fn = lambda time, state: jacobian_diag_part * state
    initial_time = 0.
    initial_state = np.float64([1., 2.])
    solver_instance = solver(rtol=_RTOL, atol=_ATOL)
    results = solver_instance.solve(
        ode_fn,
        initial_time,
        initial_state,
        solution_times=tfp.math.ode.ChosenBySolver(1.))
    times, states = self.evaluate([results.times, results.states])
    states_exact = np.exp(jacobian_diag_part[np.newaxis, :] *
                          times[:, np.newaxis]) * initial_state
    self.assertAllClose(states, states_exact)

  def test_linear_ode_complex(self, solver):
    jacobian_diag_part = np.complex128([1j - 0.1, 1j])
    ode_fn = lambda time, state: jacobian_diag_part * state
    initial_time = 0.
    initial_state = np.complex128([1., 2.])
    jacobian = np.diag(jacobian_diag_part)
    solver_instance = solver(rtol=_RTOL, atol=_ATOL)
    results = solver_instance.solve(
        ode_fn,
        initial_time,
        initial_state,
        solution_times=tfp.math.ode.ChosenBySolver(1.),
        jacobian_fn=jacobian)
    times, states = self.evaluate([results.times, results.states])
    states_exact = np.exp(jacobian_diag_part[np.newaxis, :] *
                          times[:, np.newaxis]) * initial_state
    self.assertAllClose(states, states_exact)

  def test_linear_ode_dense(self, solver):
    np.random.seed(0)
    initial_time = 0.
    num_odes = 20
    initial_state = np.float64([1.] * num_odes)
    jacobian = np.random.randn(num_odes, num_odes)

    def ode_fn(_, state):
      return tf.squeeze(tf.matmul(jacobian, state[:, tf.newaxis]))

    final_time = 1.
    solver_instance = solver(rtol=_RTOL, atol=_ATOL)
    results = solver_instance.solve(
        ode_fn,
        initial_time,
        initial_state,
        solution_times=[final_time],
        jacobian_fn=jacobian)
    final_state = self.evaluate(results.states[-1])
    # Exact solution is obtained by diagonalizing the Jacobian by
    # `jacobian = V diag(w) V^{-1}` and making the change of variables `Vz = y`.
    eigvals, eigvecs = np.linalg.eig(jacobian)
    initial_state_changed = np.matmul(np.linalg.inv(eigvecs), initial_state)
    final_state_changed_exact = np.exp(
        eigvals * final_time) * initial_state_changed
    final_state_exact = np.matmul(eigvecs, final_state_changed_exact)
    self.assertAllClose(final_state, final_state_exact)

  def test_riccati(self, solver):
    ode_fn = lambda time, state: (state - time)**2 + 1.
    initial_time = 0.
    initial_state = np.float64(0.5)
    jacobian_fn = lambda time, state: 2. * (state - time)
    solver_instance = solver(rtol=_RTOL, atol=_ATOL)
    results = solver_instance.solve(
        ode_fn,
        initial_time,
        initial_state,
        solution_times=tfp.math.ode.ChosenBySolver(1.),
        jacobian_fn=jacobian_fn)
    times, states = self.evaluate([results.times, results.states])
    states_exact = (1. / (1. / initial_state - times) + times)
    self.assertAllClose(states, states_exact)

  def test_forward_tuple(self, solver):
    jacobian_diag_a = np.float64([0.5, 1.])
    jacobian_diag_b = np.float64([1.5, 0.734])
    initial_time = 0.
    solution_times = [0.2]
    initial_state = (np.float64([1., 2.]), np.float64([5.23, 0.2354]))

    def ode_fn(time, state):
      del time
      state_a, state_b = state
      f_a = jacobian_diag_a * state_a
      f_b = jacobian_diag_b * state_b
      return f_a, f_b

    solver_instance = solver(rtol=_RTOL, atol=_ATOL)
    results = solver_instance.solve(
        ode_fn,
        initial_time,
        initial_state,
        solution_times=solution_times
    )
    times, states = self.evaluate([results.times, results.states])

    states_exact_a = initial_state[0] * np.exp(
        jacobian_diag_a[np.newaxis, :] * times[:, np.newaxis])

    states_exact_b = initial_state[1] * np.exp(
        jacobian_diag_b[np.newaxis, :] * times[:, np.newaxis])

    self.assertAllClose(states[0], states_exact_a)
    self.assertAllClose(states[1], states_exact_b)

  def test_forward_multilevel(self, solver):
    jacobian_diag_a = np.float64([0.5, 1.])
    jacobian_diag_b = np.float64([1.5, 0.734])
    jacobian_diag_c = np.float64([-2.5])
    initial_time = 0.
    solution_times = [0.2]
    initial_state = (
        np.float64([1., 2.]), (np.float64([5., 2.]), np.float64([9.])))

    def ode_fn(time, state):
      del time
      state_a, state_b_and_c = state
      state_b, state_c = state_b_and_c
      f_a = jacobian_diag_a * state_a
      f_b = jacobian_diag_b * state_b
      f_c = jacobian_diag_c * state_c
      return f_a, (f_b, f_c)

    solver_instance = solver(rtol=_RTOL, atol=_ATOL)
    results = solver_instance.solve(
        ode_fn,
        initial_time,
        initial_state,
        solution_times=solution_times
    )
    times, states = self.evaluate([results.times, results.states])

    states_exact_a = initial_state[0] * np.exp(
        jacobian_diag_a[np.newaxis, :] * times[:, np.newaxis])

    states_exact_b = initial_state[1][0] * np.exp(
        jacobian_diag_b[np.newaxis, :] * times[:, np.newaxis])

    states_exact_c = initial_state[1][1] * np.exp(
        jacobian_diag_c[np.newaxis, :] * times[:, np.newaxis])

    self.assertAllClose(states[0], states_exact_a)
    self.assertAllClose(states[1][0], states_exact_b)
    self.assertAllClose(states[1][1], states_exact_c)

  def test_linear_ode_dense_tuple(self, solver):
    np.random.seed(0)
    initial_time = 0.
    num_odes = 20
    initial_state = [np.float64(1.)] * num_odes
    jacobian = [
        [np.random.randn() for _ in range(num_odes)] for _ in range(num_odes)
    ]

    def ode_fn(_, state):
      state = tf.stack(state, axis=0)
      jacobian_tensor = tf.convert_to_tensor(jacobian, dtype=tf.float64)
      return tf.unstack(
          tf.squeeze(tf.matmul(jacobian_tensor, state[:, tf.newaxis])))

    final_time = 1.
    solver_instance = solver(rtol=_RTOL, atol=_ATOL)
    results = solver_instance.solve(
        ode_fn,
        initial_time,
        initial_state,
        solution_times=[final_time],
        jacobian_fn=jacobian)
    final_state = self.evaluate(
        tf.nest.map_structure(lambda s: s[-1], results.states))

    final_state = np.array(final_state)
    jacobian = np.array(jacobian)
    initial_state = np.array(initial_state)

    # Exact solution is obtained by diagonalizing the Jacobian by
    # `jacobian = V diag(w) V^{-1}` and making the change of variables `Vz = y`.
    eigvals, eigvecs = np.linalg.eig(jacobian)
    initial_state_changed = np.matmul(np.linalg.inv(eigvecs), initial_state)
    final_state_changed_exact = np.exp(
        eigvals * final_time) * initial_state_changed
    final_state_exact = np.matmul(eigvecs, final_state_changed_exact)
    self.assertAllClose(final_state, final_state_exact)


@test_util.test_all_tf_execution_regimes
@parameterized.named_parameters([('bdf', tfp.math.ode.BDF)])
class StiffTest(test_util.TestCase):

  def test_van_der_pol(self, solver):

    def ode_fn(_, state):
      return tf.stack([
          state[1],
          1000. * (1. - state[0]**2) * state[1] - state[0],
      ])

    def jacobian_fn(_, state):
      return tf.stack([
          [0., 1.],
          [-2000. * state[0] * state[1] - 1., 1000. * (1. - state[0]**2)],
      ])

    initial_time = 0.
    initial_state = np.float64([2., 0.])
    solver_instance = solver(rtol=1e-3, atol=1e-6)
    results = solver_instance.solve(
        ode_fn,
        initial_time,
        initial_state,
        solution_times=[3000.],
        jacobian_fn=jacobian_fn)
    self.assertAllClose(
        self.evaluate(results.states[-1, 0]), -1.5, rtol=0., atol=0.05)

  def test_van_der_pol_tuple(self, solver):

    def ode_fn(_, state):
      return (state[1], 1000. * (1. - state[0]**2) * state[1] - state[0])

    def jacobian_fn(_, state):
      return [
          [np.float64(0.), np.float64(1.)],
          [-2000. * state[0] * state[1] - 1., 1000. * (1. - state[0]**2)],
      ]

    initial_time = 0.
    initial_state = [np.float64(2.), np.float64(0.)]
    solver_instance = solver(rtol=1e-3, atol=1e-6)
    results = solver_instance.solve(
        ode_fn,
        initial_time,
        initial_state,
        solution_times=[3000.],
        jacobian_fn=jacobian_fn)
    self.assertAllClose(
        self.evaluate(results.states[0][-1]), -1.5, rtol=0., atol=0.05)


@test_util.test_all_tf_execution_regimes
@parameterized.named_parameters([
    ('bdf', tfp.math.ode.BDF),
    ('dormand_prince', tfp.math.ode.DormandPrince)])
class GradientTest(test_util.TestCase):

  def test_riccati(self, solver):
    ode_fn = lambda time, state: (state - time)**2 + 1.
    initial_time = 0.
    initial_state_value = 0.5
    initial_state = tf.constant(initial_state_value, dtype=tf.float64)
    final_time = 1.
    jacobian_fn = lambda time, state: 2. * (state - time)
    solver_instance = solver(rtol=_RTOL, atol=_ATOL)
    with tf.GradientTape() as tape:
      tape.watch(initial_state)
      results = solver_instance.solve(
          ode_fn,
          initial_time,
          initial_state,
          solution_times=[final_time],
          jacobian_fn=jacobian_fn)
      final_state = results.states[-1]
    grad = self.evaluate(tape.gradient(final_state, initial_state))
    grad_exact = 1. / (1. - initial_state_value * final_time)**2
    self.assertAllClose(grad, grad_exact, rtol=1e-3, atol=1e-3)

  def test_riccati_custom_adjoint_solver(self, solver):
    ode_fn = lambda time, state: (state - time)**2 + 1.
    initial_time = 0.
    initial_state_value = 0.5
    initial_state = tf.constant(initial_state_value, dtype=tf.float64)
    final_time = 1.
    solution_times = np.linspace(initial_time, final_time, 4)
    jacobian_fn = lambda time, state: 2. * (state - time)

    # Instrument the adjoint solver for testing.
    first_step_size = np.float64(1.)
    last_initial_step_size = tf.Variable(0., dtype=tf.float64)
    self.evaluate(last_initial_step_size.initializer)

    class _InstrumentedSolver(StepSizeHeuristicAdjointSolver):

      def solve(self, **kwargs):
        with tf.control_dependencies([
            last_initial_step_size.assign(
                kwargs['previous_solver_internal_state'])
        ]):
          return super(_InstrumentedSolver, self).solve(**kwargs)

    adjoint_solver = _InstrumentedSolver(
        make_solver_fn=lambda step_size: solver(  # pylint: disable=g-long-lambda
            rtol=_RTOL,
            atol=_ATOL,
            first_step_size=step_size),
        first_step_size=first_step_size)

    solver_instance = solver(
        rtol=_RTOL, atol=_ATOL, make_adjoint_solver_fn=lambda: adjoint_solver)
    with tf.GradientTape() as tape:
      tape.watch(initial_state)
      results = solver_instance.solve(
          ode_fn,
          initial_time,
          initial_state,
          solution_times=solution_times,
          jacobian_fn=jacobian_fn)
      final_state = results.states[-1]
    grad = self.evaluate(tape.gradient(final_state, initial_state))
    last_initial_step_size = self.evaluate(last_initial_step_size)
    grad_exact = 1. / (1. - initial_state_value * final_time)**2
    self.assertAllClose(grad, grad_exact, rtol=1e-3, atol=1e-3)
    # This indicates that the adaptation carried over to the final solve. We
    # expect the step size to decrease because we purposefully made the initial
    # step size way too large.
    self.assertLess(last_initial_step_size, first_step_size)

  def test_linear_ode(self, solver):
    if not tf2.enabled():
      self.skipTest('b/152464477')
    jacobian_diag_part = tf.constant([-0.5, -1.], dtype=tf.float64)
    ode_fn = lambda time, state, jacobian_diag_part: jacobian_diag_part * state
    initial_time = 0.
    initial_state = np.float64([1., 2.])
    solver_instance = solver(rtol=_RTOL, atol=_ATOL)
    with tf.GradientTape() as tape:
      tape.watch(jacobian_diag_part)
      results = solver_instance.solve(
          ode_fn,
          initial_time,
          initial_state,
          solution_times=tfp.math.ode.ChosenBySolver(1.),
          constants={'jacobian_diag_part': jacobian_diag_part})
    grad = tape.gradient(results.states, jacobian_diag_part)
    times, grad = self.evaluate([results.times, grad])

    with tf.GradientTape() as tape:
      tape.watch(jacobian_diag_part)
      states_exact = tf.exp(jacobian_diag_part[tf.newaxis, :] *
                            times[:, tf.newaxis]) * initial_state
    exact_grad = tape.gradient(states_exact, jacobian_diag_part)
    exact_grad = self.evaluate(exact_grad)
    self.assertAllClose(exact_grad, grad, atol=1e-5)


# Running pfor repeatedly to rebuild the Jacobian graph is too slow in Eager
# mode.
@test_util.test_graph_mode_only
@parameterized.named_parameters([
    ('bdf', tfp.math.ode.BDF),
    ('dormand_prince', tfp.math.ode.DormandPrince)])
class GradientTestPforJacobian(test_util.TestCase):

  def test_linear_ode_dense(self, solver):
    initial_time = 0.
    jacobian = -np.float64([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]])
    num_odes = jacobian.shape[0]
    initial_state_value = np.float64([1.] * num_odes)
    initial_state = tf.constant(initial_state_value, dtype=tf.float64)

    def ode_fn(_, state):
      return tf.squeeze(tf.matmul(jacobian, state[:, tf.newaxis]))

    intermediate_time = 1.
    final_time = 2.
    solver_instance = solver(rtol=_RTOL, atol=_ATOL)
    with tf.GradientTape() as tape:
      tape.watch(initial_state)
      results = solver_instance.solve(
          ode_fn,
          initial_time,
          initial_state,
          solution_times=[intermediate_time, final_time])
      intermediate_state = results.states[0]
    grad = self.evaluate(tape.gradient(intermediate_state, initial_state))
    matrix_exponential_of_jacobian = np.float64(
        [[+2.3703878775011322, +0.2645063729368097, -0.8413751316275110],
         [-0.0900545996427410, +0.7326649140674798, -0.4446155722222950],
         [-1.5504970767866180, -0.7991765448018465, +0.9521439871829201]])
    grad_exact = np.dot(np.ones([num_odes]), matrix_exponential_of_jacobian)
    self.assertAllClose(grad, grad_exact)

  def test_tuple(self, solver):
    alpha = np.float32(0.7)
    beta = np.float32(2.2)
    variable_1 = tf.Variable(alpha, name='alpha')
    variable_2 = tf.Variable(beta, name='beta')

    def ode_fn(time, state):
      del time
      x, y = state
      return variable_1 * x, (variable_1 + variable_2) * y

    initial_time = 0.
    initial_x = np.float32(3.2)
    initial_y = np.float32(-4.2)
    initial_state = (initial_x, initial_y)
    final_time = 0.5

    solver_instance = solver(rtol=_RTOL, atol=_ATOL)

    with tf.GradientTape(persistent=True) as tape:
      tape.watch([variable_1, variable_2])
      results = solver_instance.solve(
          ode_fn, initial_time, initial_state, solution_times=[final_time])
      final_state = results.states
    self.evaluate([variable_1.initializer, variable_2.initializer])
    actual_grad_1 = self.evaluate(tape.gradient(final_state[0], [variable_1]))
    actual_grad_2 = self.evaluate(tape.gradient(final_state[1], [variable_1]))
    actual_grad_3 = self.evaluate(tape.gradient(final_state[1], [variable_2]))

    expected_grad_1 = (
        initial_x * final_time * np.exp(alpha * final_time)[np.newaxis])
    expected_grad_2 = (
        initial_y * final_time *
        np.exp((alpha + beta) * final_time)[np.newaxis])
    expected_grad_3 = expected_grad_2
    self.assertAllClose(actual_grad_1, expected_grad_1, rtol=1e-4, atol=1e-4)
    self.assertAllClose(actual_grad_2, expected_grad_2, rtol=1e-4, atol=1e-4)
    self.assertAllClose(actual_grad_3, expected_grad_3, rtol=1e-4, atol=1e-4)


@test_util.test_all_tf_execution_regimes
@parameterized.named_parameters([
    ('bdf', tfp.math.ode.BDF),
    ('dormand_prince', tfp.math.ode.DormandPrince)])
class GeneralTest(test_util.TestCase):

  def test_bad_initial_state_dtype(self, solver):
    ode_fn = lambda time, state: -state
    initial_time = 0.
    initial_state = np.int32(1)
    with self.assertRaisesRegexp(
        TypeError, ('`initial_state` must have a floating point or complex '
                    'floating point dtype')):
      solver(validate_args=True).solve(
          ode_fn, initial_time, initial_state, solution_times=[1.])

  def test_diagnostics(self, solver):
    ode_fn = lambda time, state: -state
    initial_time = 0.
    initial_state = np.float64(1.)
    results = solver(validate_args=True).solve(
        ode_fn, initial_time, initial_state, solution_times=[1.])
    (
        num_ode_fn_evaluations,
        num_jacobian_evaluations,
        num_matrix_factorizations,
        status,
    ) = self.evaluate([
        results.diagnostics.num_ode_fn_evaluations,
        results.diagnostics.num_jacobian_evaluations,
        results.diagnostics.num_matrix_factorizations,
        results.diagnostics.status,
    ])
    self.assertEqual(status, 0)
    self.assertGreater(num_ode_fn_evaluations, 0)
    self.assertGreaterEqual(num_jacobian_evaluations, 0)
    self.assertGreaterEqual(num_matrix_factorizations, 0)

  def test_previous_solver_internal_state(self, solver):
    jacobian_diag_part = np.float64([-0.5, -1.])
    ode_fn = lambda time, state: jacobian_diag_part * state
    initial_time = 0.
    initial_state = np.float64([1., 2.])
    intermediate_time = 1.
    final_time = 2.
    solver_instance = solver(rtol=_RTOL, atol=_ATOL, validate_args=True)
    previous_results = solver_instance.solve(
        ode_fn,
        initial_time,
        initial_state,
        solution_times=tfp.math.ode.ChosenBySolver(intermediate_time))
    results = solver_instance.solve(
        ode_fn,
        intermediate_time,
        previous_results.states[-1],
        solution_times=tfp.math.ode.ChosenBySolver(final_time),
        previous_solver_internal_state=previous_results.solver_internal_state)
    times, states = self.evaluate([results.times, results.states])
    states_exact = np.exp(jacobian_diag_part[np.newaxis, :] *
                          times[:, np.newaxis]) * initial_state
    self.assertAllClose(states, states_exact)

  def test_initialize_solver_internal_state(self, solver):
    jacobian_diag_part = np.float64([-0.5, -1.])
    ode_fn = lambda time, state: jacobian_diag_part * state
    initial_time = 0.
    initial_state = np.float64([1., 2.])
    final_time = 2.
    solver_instance = solver(rtol=_RTOL, atol=_ATOL, validate_args=True)
    solver_internal_state = solver_instance._initialize_solver_internal_state(
        ode_fn, initial_time, initial_state)
    results = solver_instance.solve(
        ode_fn,
        initial_time,
        initial_state,
        solution_times=tfp.math.ode.ChosenBySolver(final_time),
        previous_solver_internal_state=solver_internal_state)
    times, states = self.evaluate([results.times, results.states])
    states_exact = np.exp(jacobian_diag_part[np.newaxis, :] *
                          times[:, np.newaxis]) * initial_state
    self.assertAllClose(states, states_exact)


if __name__ == '__main__':
  tf.test.main()
