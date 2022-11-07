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

from absl.testing import parameterized
import numpy as np
import tensorflow.compat.v1 as tf1
import tensorflow.compat.v2 as tf

from tensorflow_probability.python.internal import test_util
from tensorflow_probability.python.math import gradient as tfp_gradient
from tensorflow_probability.python.math.ode import base
from tensorflow_probability.python.math.ode import bdf
from tensorflow_probability.python.math.ode import dormand_prince

_RTOL = 1e-8
_ATOL = 1e-12

NUMPY_MODE = False
JAX_MODE = False


def _test_cases(bdf_only=False):
  # JAX cannot deal with ChosenBySolver as that requires dynamic TensorArrays.
  # NumPy cannot deal with BDF due to the Newton's method inside of it.

  def _fixed_fn(final_time):
    return [final_time]

  test_cases = []
  if not bdf_only:
    if not JAX_MODE:
      test_cases.append(
          ('dormand_prince', dormand_prince.DormandPrince, base.ChosenBySolver))
    test_cases.append(
        ('dormand_prince_fixed', dormand_prince.DormandPrince, _fixed_fn))
  if not NUMPY_MODE:
    if not JAX_MODE:
      test_cases.append(('bdf', bdf.BDF, base.ChosenBySolver))
    test_cases.append(('bdf_fixed', bdf.BDF, _fixed_fn))
  if not test_cases:
    # This is here just to appease parameterized.named_parameters, as it needs
    # at least 1 test case. We'll be disabling the entire test manually in this
    # case.
    test_cases.append(('disabled', None, None))

  return test_cases


class StepSizeHeuristicAdjointSolver(base.Solver):
  """Adjoint solver which propagates step size between solves."""

  def __init__(self, make_solver_fn, first_step_size):
    self._first_step_size = first_step_size
    self._make_solver_fn = make_solver_fn
    solver = make_solver_fn(first_step_size)
    super(StepSizeHeuristicAdjointSolver, self).__init__(
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
@parameterized.named_parameters(_test_cases())
class NonStiffTest(test_util.TestCase):

  def test_zero_dims(self, solver, solution_times_fn):
    ode_fn = lambda time, state: -state
    initial_time = 0.
    initial_state = np.float64(1.)
    jacobian = np.float64([[-1.]])
    solver_instance = solver(rtol=_RTOL, atol=_ATOL)
    results = solver_instance.solve(
        ode_fn,
        initial_time,
        initial_state,
        solution_times=solution_times_fn(1.),
        jacobian_fn=jacobian)
    times, states = self.evaluate([results.times, results.states])
    states_exact = np.exp(-times) * initial_state
    self.assertAllClose(states, states_exact)

  def test_state_with_matrix_shape(self, solver, solution_times_fn):
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
        solution_times=solution_times_fn(1.),
        jacobian_fn=jacobian)
    times, states = self.evaluate([results.times, results.states])
    states_exact = np.exp(-times)[:, np.newaxis, np.newaxis] * initial_state
    self.assertAllClose(states, states_exact)

  def test_ode_fn_is_zero(self, solver, solution_times_fn):
    initial_time = 0.
    initial_state = np.float64([1., 2., 3.])
    ode_fn = lambda time, state: np.zeros_like(initial_state)
    jacobian = np.zeros((3, 3), dtype=np.float64)
    solver_instance = solver(rtol=_RTOL, atol=_ATOL)
    results = solver_instance.solve(
        ode_fn,
        initial_time,
        initial_state,
        solution_times=solution_times_fn(1.),
        jacobian_fn=jacobian)
    times, states = self.evaluate([results.times, results.states])
    states_exact = np.ones([times.size, initial_state.size]) * initial_state
    self.assertAllClose(states, states_exact)

  def test_linear_ode(self, solver, solution_times_fn):
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
        solution_times=solution_times_fn(1.),
        jacobian_fn=jacobian)
    times, states = self.evaluate([results.times, results.states])
    states_exact = np.exp(jacobian_diag_part[np.newaxis, :] *
                          times[:, np.newaxis]) * initial_state
    self.assertAllClose(states, states_exact)

  def test_linear_ode_jacobian_fn_unspecified(self, solver, solution_times_fn):
    jacobian_diag_part = np.float64([-0.5, -1.])
    ode_fn = lambda time, state: jacobian_diag_part * state
    initial_time = 0.
    initial_state = np.float64([1., 2.])
    solver_instance = solver(rtol=_RTOL, atol=_ATOL)
    results = solver_instance.solve(
        ode_fn,
        initial_time,
        initial_state,
        solution_times=solution_times_fn(1.))
    times, states = self.evaluate([results.times, results.states])
    states_exact = np.exp(jacobian_diag_part[np.newaxis, :] *
                          times[:, np.newaxis]) * initial_state
    self.assertAllClose(states, states_exact)

  def test_linear_ode_complex(self, solver, solution_times_fn):
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
        solution_times=solution_times_fn(1.),
        jacobian_fn=jacobian)
    times, states = self.evaluate([results.times, results.states])
    states_exact = np.exp(jacobian_diag_part[np.newaxis, :] *
                          times[:, np.newaxis]) * initial_state
    self.assertAllClose(states, states_exact)

  def test_linear_ode_dense(self, solver, solution_times_fn):
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
        solution_times=solution_times_fn(final_time),
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

  def test_riccati(self, solver, solution_times_fn):
    ode_fn = lambda time, state: (state - time)**2 + 1.
    initial_time = 0.
    initial_state = np.float64(0.5)
    jacobian_fn = lambda time, state: 2. * (state - time)
    solver_instance = solver(rtol=_RTOL, atol=_ATOL)
    results = solver_instance.solve(
        ode_fn,
        initial_time,
        initial_state,
        solution_times=solution_times_fn(1.),
        jacobian_fn=jacobian_fn)
    times, states = self.evaluate([results.times, results.states])
    states_exact = (1. / (1. / initial_state - times) + times)
    self.assertAllClose(states, states_exact)

  def test_forward_tuple(self, solver, solution_times_fn):
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

  def test_forward_multilevel(self, solver, solution_times_fn):
    jacobian_diag_a = np.float64([0.5, 1.])
    jacobian_diag_b = np.float64([1.5, 0.734])
    jacobian_diag_c = np.float64([-2.5])
    initial_time = 0.
    solution_times = solution_times_fn(0.2)
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

  def test_linear_ode_dense_tuple(self, solver, solution_times_fn):
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
        solution_times=solution_times_fn(final_time),
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


@test_util.numpy_disable_gradient_test
@test_util.test_all_tf_execution_regimes
@parameterized.named_parameters(_test_cases(bdf_only=True))
class StiffTest(test_util.TestCase):

  def test_van_der_pol(self, solver, solution_times_fn):

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
        solution_times=solution_times_fn(3000.),
        jacobian_fn=jacobian_fn)
    self.assertAllClose(
        self.evaluate(results.states[-1, 0]), -1.5, rtol=0., atol=0.05)

  def test_van_der_pol_tuple(self, solver, solution_times_fn):

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
        solution_times=solution_times_fn(3000.),
        jacobian_fn=jacobian_fn)
    self.assertAllClose(
        self.evaluate(results.states[0][-1]), -1.5, rtol=0., atol=0.05)


@test_util.numpy_disable_gradient_test
@test_util.test_graph_mode_only
@parameterized.named_parameters(_test_cases())
class GradientTest(test_util.TestCase):

  def test_riccati(self, solver, solution_times_fn):
    ode_fn = lambda time, state: (state - time)**2 + 1.
    initial_time = 0.
    initial_state_value = 0.5
    initial_state = tf.constant(initial_state_value, dtype=tf.float64)
    final_time = 1.
    jacobian_fn = lambda time, state: 2. * (state - time)
    solver_instance = solver(rtol=_RTOL, atol=_ATOL)

    def grad_fn(initial_state):
      results = solver_instance.solve(
          ode_fn,
          initial_time,
          initial_state,
          solution_times=solution_times_fn(final_time),
          jacobian_fn=jacobian_fn)
      return results.states[-1]

    grad = self.evaluate(
        tfp_gradient.value_and_gradient(grad_fn, initial_state)[1])
    grad_exact = 1. / (1. - initial_state_value * final_time)**2
    self.assertAllClose(grad, grad_exact, rtol=1e-3, atol=1e-3)

  @test_util.jax_disable_variable_test
  def test_riccati_custom_adjoint_solver(self, solver, solution_times_fn):
    ode_fn = lambda time, state: (state - time)**2 + 1.
    initial_time = 0.
    initial_state_value = 0.5
    initial_state = tf.constant(initial_state_value, dtype=tf.float64)
    final_time = 1.
    solution_times = solution_times_fn(final_time)
    jacobian_fn = lambda time, state: 2. * (state - time)

    if not isinstance(solution_times, base.ChosenBySolver):
      self.skipTest('b/194468619')

    # Instrument the adjoint solver for testing. We have to do this because the
    # API doesn't provide access to the adjoint solver's diagnostics.
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

    def grad_fn(initial_state):
      results = solver_instance.solve(
          ode_fn,
          initial_time,
          initial_state,
          solution_times=solution_times,
          jacobian_fn=jacobian_fn)
      final_state = results.states[-1]
      return final_state
    _, grad = tfp_gradient.value_and_gradient(grad_fn, initial_state)
    grad, last_initial_step_size = self.evaluate((grad, last_initial_step_size))
    grad_exact = 1. / (1. - initial_state_value * final_time)**2
    self.assertAllClose(grad, grad_exact, rtol=1e-3, atol=1e-3)
    # This indicates that the adaptation carried over to the final solve. We
    # expect the step size to decrease because we purposefully made the initial
    # step size way too large.
    self.assertLess(last_initial_step_size, first_step_size)

  def test_linear_ode(self, solver, solution_times_fn):
    if not tf1.control_flow_v2_enabled():
      self.skipTest('b/152464477')
    jacobian_diag_part = tf.constant([-0.5, -1.], dtype=tf.float64)
    ode_fn = lambda time, state, jacobian_diag_part: jacobian_diag_part * state
    initial_time = 0.
    initial_state = np.float64([1., 2.])
    solver_instance = solver(rtol=_RTOL, atol=_ATOL)

    def grad_fn(jacobian_diag_part):
      results = solver_instance.solve(
          ode_fn,
          initial_time,
          initial_state,
          solution_times=solution_times_fn(1.),
          constants={'jacobian_diag_part': jacobian_diag_part})
      return results.states, results.times

    (_, times), grad = tfp_gradient.value_and_gradient(
        grad_fn, jacobian_diag_part, has_aux=True)
    times, grad = self.evaluate([times, grad])

    def exact_grad_fn(jacobian_diag_part):
      states_exact = tf.exp(jacobian_diag_part[tf.newaxis, :] *
                            times[:, tf.newaxis]) * initial_state
      return states_exact
    _, exact_grad = tfp_gradient.value_and_gradient(
        exact_grad_fn, jacobian_diag_part)
    exact_grad = self.evaluate(exact_grad)
    self.assertAllClose(exact_grad, grad, atol=1e-5)


# Running pfor repeatedly to rebuild the Jacobian graph is too slow in Eager
# mode.
@test_util.numpy_disable_gradient_test
@test_util.test_graph_mode_only
@parameterized.named_parameters([('bdf', bdf.BDF),
                                 ('dormand_prince',
                                  dormand_prince.DormandPrince)])
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

    def grad_fn(initial_state):
      results = solver_instance.solve(
          ode_fn,
          initial_time,
          initial_state,
          solution_times=[intermediate_time, final_time])
      intermediate_state = results.states[0]
      return intermediate_state

    grad = self.evaluate(
        tfp_gradient.value_and_gradient(grad_fn, initial_state)[1])
    matrix_exponential_of_jacobian = np.float64(
        [[+2.3703878775011322, +0.2645063729368097, -0.8413751316275110],
         [-0.0900545996427410, +0.7326649140674798, -0.4446155722222950],
         [-1.5504970767866180, -0.7991765448018465, +0.9521439871829201]])
    grad_exact = np.dot(np.ones([num_odes]), matrix_exponential_of_jacobian)
    self.assertAllClose(grad, grad_exact)

  def test_tuple(self, solver):
    if not tf1.control_flow_v2_enabled():
      self.skipTest('b/152464477')

    alpha = np.float32(0.7)
    beta = np.float32(2.2)

    def ode_fn(time, state, alpha, beta):
      del time
      x, y = state
      return alpha * x, (alpha + beta) * y

    initial_time = 0.
    initial_x = np.float32(3.2)
    initial_y = np.float32(-4.2)
    initial_state = (initial_x, initial_y)
    final_time = 0.5

    solver_instance = solver(rtol=_RTOL, atol=_ATOL)

    def grad_fn(alpha, beta):
      results = solver_instance.solve(
          ode_fn, initial_time, initial_state, solution_times=[final_time],
          constants={'alpha': alpha, 'beta': beta})
      final_state = results.states
      return final_state

    actual_grad_1 = self.evaluate(
        tfp_gradient.value_and_gradient(lambda alpha: grad_fn(alpha, beta)[0],
                                        alpha)[1])
    actual_grad_2 = self.evaluate(
        tfp_gradient.value_and_gradient(lambda alpha: grad_fn(alpha, beta)[1],
                                        alpha)[1])
    actual_grad_3 = self.evaluate(
        tfp_gradient.value_and_gradient(lambda beta: grad_fn(alpha, beta)[1],
                                        beta)[1])

    expected_grad_1 = (
        initial_x * final_time * np.exp(alpha * final_time))
    expected_grad_2 = (
        initial_y * final_time *
        np.exp((alpha + beta) * final_time))
    expected_grad_3 = expected_grad_2
    self.assertAllClose(actual_grad_1, expected_grad_1, rtol=1e-4, atol=1e-4)
    self.assertAllClose(actual_grad_2, expected_grad_2, rtol=1e-4, atol=1e-4)
    self.assertAllClose(actual_grad_3, expected_grad_3, rtol=1e-4, atol=1e-4)

  @test_util.jax_disable_variable_test
  def test_tuple_variables(self, solver):
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
@parameterized.named_parameters(
    [('dormand_prince', dormand_prince.DormandPrince)] +
    ([] if NUMPY_MODE else [('bdf', bdf.BDF)]))
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

  @test_util.jax_disable_test_missing_functionality('b/194468988')
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
        solution_times=base.ChosenBySolver(intermediate_time))
    results = solver_instance.solve(
        ode_fn,
        intermediate_time,
        previous_results.states[-1],
        solution_times=base.ChosenBySolver(final_time),
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
        solution_times=[final_time],
        previous_solver_internal_state=solver_internal_state)
    times, states = self.evaluate([results.times, results.states])
    states_exact = np.exp(jacobian_diag_part[np.newaxis, :] *
                          times[:, np.newaxis]) * initial_state
    self.assertAllClose(states, states_exact)


if __name__ == '__main__':
  test_util.main()
