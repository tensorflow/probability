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
"""Base classes for TensorFlow Probability ODE solvers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import collections
import six
import tensorflow.compat.v1 as tf1
import tensorflow.compat.v2 as tf

from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.math.ode import runge_kutta_util as rk_util
from tensorflow_probability.python.math.ode import util

# TODO(b/138303336): Support MATLAB-style events.
# TODO(b/138303018): Support nested structure state.

__all__ = [
    'ChosenBySolver',
    'Diagnostics',
    'Results',
    'Solver',
]


@six.add_metaclass(abc.ABCMeta)
class Solver(object):
  """Base class for an ODE solver."""

  def __init__(self, use_pfor_to_compute_jacobian, validate_args, name):
    self._use_pfor_to_compute_jacobian = use_pfor_to_compute_jacobian
    self._validate_args = validate_args
    self._name = name

  def solve(
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
    """Solves an initial value problem.

    An initial value problem consists of a system of ODEs and an initial
    condition:

    ```none
    dy/dt(t) = ode_fn(t, y(t))
    y(initial_time) = initial_state
    ```

    Here, `t` (also called time) is a scalar float `Tensor` and `y(t)` (also
    called the state at time `t`) is an N-D float or complex `Tensor`.

    ### Example

    The ODE `dy/dt(t) = dot(A, y(t))` is solved below.

    ```python
    t_init, t0, t1 = 0., 0.5, 1.
    y_init = tf.constant([1., 1.], dtype=tf.float64)
    A = tf.constant([[-1., -2.], [-3., -4.]], dtype=tf.float64)

    def ode_fn(t, y):
      return tf.linalg.matvec(A, y)

    results = tfp.math.ode.BDF().solve(ode_fn, t_init, y_init,
                                       solution_times=[t0, t1])
    y0 = results.states[0]  # == dot(matrix_exp(A * t0), y_init)
    y1 = results.states[1]  # == dot(matrix_exp(A * t1), y_init)
    ```

    Using instead `solution_times=tfp.math.ode.ChosenBySolver(final_time=1.)`
    yields the state at various times between `t_init` and `final_time` chosen
    automatically by the solver. In this case, `results.states[i]` is the state
    at time `results.times[i]`.

    #### Gradient

    The gradient of the result is computed using the adjoint sensitivity method
    described in [Chen et al. (2018)][1].

    ```python
    grad = tf.gradients(y1, y0) # == dot(e, J)
    # J is the Jacobian of y1 with respect to y0. In this case, J = exp(A * t1).
    # e = [1, ..., 1] is the row vector of ones.
    ```

    #### References

    [1]: Chen, Tian Qi, et al. "Neural ordinary differential equations."
         Advances in Neural Information Processing Systems. 2018.

    Args:
      ode_fn: Function of the form `ode_fn(t, y)`. The input `t` is a scalar
        float `Tensor`. The input `y` and output are both `Tensor`s with the
        same shape and `dtype` as `initial_state`.
      initial_time: Scalar float `Tensor` specifying the initial time.
      initial_state: N-D float or complex `Tensor` specifying the initial state.
        The `dtype` of `initial_state` must be complex for problems with
        complex-valued states (even if the initial state is real).
      solution_times: 1-D float `Tensor` specifying a list of times. The solver
        stores the computed state at each of these times in the returned
        `Results` object. Must satisfy `initial_time <= solution_times[0]` and
        `solution_times[i] < solution_times[i+1]`. Alternatively, the user can
        pass `tfp.math.ode.ChosenBySolver(final_time)` where `final_time` is a
        scalar float `Tensor` satisfying `initial_time < final_time`. Doing so
        requests that the solver automatically choose suitable times up to and
        including `final_time` at which to store the computed state.
      jacobian_fn: Optional function of the form `jacobian_fn(t, y)`. The input
        `t` is a scalar float `Tensor`. The input `y` has the same shape and
        `dtype` as `initial_state`. The output is a 2N-D `Tensor` whose shape is
        `initial_state.shape + initial_state.shape` and whose `dtype` is the
        same as `initial_state`. In particular, the `(i1, ..., iN, j1, ...,
        jN)`-th entry of `jacobian_fn(t, y)` is the derivative of the `(i1, ...,
        iN)`-th entry of `ode_fn(t, y)` with respect to the `(j1, ..., jN)`-th
        entry of `y`. If this argument is left unspecified, the solver
        automatically computes the Jacobian if and when it is needed.
        Default value: `None`.
      jacobian_sparsity: Optional 2N-D boolean `Tensor` whose shape is
        `initial_state.shape + initial_state.shape` specifying the sparsity
        pattern of the Jacobian. This argument is ignored if `jacobian_fn` is
        specified.
        Default value: `None`.
      batch_ndims: Optional nonnegative integer. When specified, the first
        `batch_ndims` dimensions of `initial_state` are batch dimensions.
        Default value: `None`.
      previous_solver_internal_state: Optional solver-specific argument used to
        warm-start this invocation of `solve`.
        Default value: `None`.

    Returns:
      Object of type `Results`.
    """
    input_state_structure = initial_state

    @tf.custom_gradient
    def gradient_helper(*flat_initial_state_components):
      """Inner method that restricts gradient to initial state components."""
      flat_initial_state = list(flat_initial_state_components)
      flat_initial_state = [tf.convert_to_tensor(c) for c in flat_initial_state]
      initial_state = tf.nest.pack_sequence_as(
          input_state_structure, flat_initial_state)

      results = self._solve(
          ode_fn,
          initial_time,
          initial_state,
          solution_times,
          jacobian_fn,
          jacobian_sparsity,
          batch_ndims,
          previous_solver_internal_state,
      )
      results = Results(
          times=tf.stop_gradient(results.times),
          states=results.states,
          diagnostics=util.stop_gradient_of_real_or_complex_entries(
              results.diagnostics),
          solver_internal_state=util.stop_gradient_of_real_or_complex_entries(
              results.solver_internal_state))

      def grad_fn(*dresults, **kwargs):
        """Adjoint sensitivity method to compute gradients."""
        dresults = tf.nest.pack_sequence_as(results, dresults)
        dstates = dresults.states
        # The signature grad_fn(*dresults, variables=None) is not valid Python 2
        # so use kwargs instead.
        variables = kwargs.pop('variables', [])
        assert not kwargs  # This assert should never fail.
        # TODO(b/138304303): Support complex types.
        with tf.name_scope('{}Gradients'.format(self._name)):
          get_dtype = lambda x: x.dtype
          def error_if_complex(dtype):
            if dtype.is_complex:
              raise NotImplementedError('The adjoint sensitivity method does '
                                        'not support complex dtypes.')

          state_dtypes = tf.nest.map_structure(get_dtype, initial_state)
          tf.nest.map_structure(error_if_complex, state_dtypes)
          common_state_dtype = dtype_util.common_dtype(initial_state)
          real_dtype = dtype_util.real_dtype(common_state_dtype)

          # We add initial_time to ensure that we know where to stop.
          result_times = tf.concat(
              [[tf.cast(initial_time, real_dtype)], results.times], 0)
          num_result_times = tf.size(result_times)

          # First two components correspond to reverse and adjoint states.
          # the last component is adjoint state for variables.
          terminal_augmented_state = tuple([
              rk_util.nest_constant(initial_state, 0.0),
              rk_util.nest_constant(initial_state, 0.0),
              tuple(
                  rk_util.nest_constant(variable, 0.0) for variable in variables
              )
          ])

          # The XLA compiler does not compile code which slices/indexes using
          # integer `Tensor`s. `TensorArray`s are used to get around this.
          result_time_array = tf.TensorArray(
              results.times.dtype,
              clear_after_read=False,
              size=num_result_times,
              element_shape=[]).unstack(result_times)

          # TensorArray shape should not include time dimension, hence shape[1:]
          result_state_arrays = [
              tf.TensorArray(  # pylint: disable=g-complex-comprehension
                  dtype=component.dtype, size=num_result_times - 1,
                  element_shape=component.shape[1:]).unstack(component)
              for component in tf.nest.flatten(results.states)
          ]
          result_state_arrays = tf.nest.pack_sequence_as(
              results.states, result_state_arrays)
          dresult_state_arrays = [
              tf.TensorArray(  # pylint: disable=g-complex-comprehension
                  dtype=component.dtype, size=num_result_times - 1,
                  element_shape=component.shape[1:]).unstack(component)
              for component in tf.nest.flatten(dstates)
          ]
          dresult_state_arrays = tf.nest.pack_sequence_as(
              results.states, dresult_state_arrays)

          def augmented_ode_fn(backward_time, augmented_state):
            """Dynamics function for the augmented system.

            Describes a differential equation that evolves the augmented state
            backwards in time to compute gradients using the adjoint method.
            Augmented state consists of 3 components `(state, adjoint_state,
            vars)` all evaluated at time `backward_time`:

            state: represents the solution of user provided `ode_fn`. The
              structure coincides with the `initial_state`.
            adjoint_state: represents the solution of adjoint sensitivity
              differential equation as discussed below. Has the same structure
              and shape as `state`.
            vars: represent the solution of the adjoint equation for variable
              gradients. Represented as a `Tuple(Tensor, ...)` with as many
              tensors as there are `variables`.

            Adjoint sensitivity equation describes the gradient of the solution
            with respect to the value of the solution at previous time t. Its
            dynamics are given by
            d/dt[adj(t)] = -1 * adj(t) @ jacobian(ode_fn(t, z), z)
            Which is computed as:
            d/dt[adj(t)]_i = -1 * sum_j(adj(t)_j * d/dz_i[ode_fn(t, z)_j)]
            d/dt[adj(t)]_i = -1 * d/dz_i[sum_j(no_grad_adj_j * ode_fn(t, z)_j)]
            where in the last line we moved adj(t)_j under derivative by
            removing gradient from it.

            Adjoint equation for the gradient with respect to every
            `tf.Variable` theta follows:
            d/dt[grad_theta(t)] = -1 * adj(t) @ jacobian(ode_fn(t, z), theta)
            = -1 * d/d theta_i[sum_j(no_grad_adj_j * ode_fn(t, z)_j)]

            Args:
              backward_time: Floating `Tensor` representing current time.
              augmented_state: `Tuple(state, adjoint_state, variable_grads)`

            Returns:
              negative_derivatives: Structure of `Tensor`s equal to backwards
                time derivative of the `state` componnent.
              adjoint_ode: Structure of `Tensor`s equal to backwards time
                derivative of the `adjoint_state` component.
              adjoint_variables_ode: Structure of `Tensor`s equal to backwards
                time derivative of the `vars` component.
            """
            # The negative signs disappears after the change of variables.
            # The ODE solver cannot handle the case initial_time > final_time
            # and hence a change of variables backward_time = -time is used.
            time = -backward_time
            state, adjoint_state, _ = augmented_state

            with tf.GradientTape() as tape:
              tape.watch(variables)
              tape.watch(state)
              derivatives = ode_fn(time, state)
              adjoint_no_grad = tf.nest.map_structure(
                  tf.stop_gradient, adjoint_state)
              negative_derivatives = rk_util.weighted_sum([-1.0], [derivatives])

              def dot_prod(tensor_a, tensor_b):
                return tf.reduce_sum(tensor_a * tensor_b)
              # See docstring for details.
              adjoint_dot_derivatives = tf.nest.map_structure(
                  dot_prod, adjoint_no_grad, derivatives)
              adjoint_dot_derivatives = tf.squeeze(
                  tf.add_n(tf.nest.flatten(adjoint_dot_derivatives)))

            adjoint_ode, adjoint_variables_ode = tape.gradient(
                adjoint_dot_derivatives, (state, tuple(variables)),
                unconnected_gradients=tf.UnconnectedGradients.ZERO)
            return negative_derivatives, adjoint_ode, adjoint_variables_ode

          def reverse_to_result_time(n, augmented_state, _):
            """Integrates the augmented system backwards in time."""
            lower_bound_of_integration = result_time_array.read(n)
            upper_bound_of_integration = result_time_array.read(n - 1)
            _, adjoint_state, adjoint_variable_state = augmented_state
            initial_state = _read_solution_components(
                result_state_arrays, input_state_structure, n - 1)
            initial_adjoint = _read_solution_components(
                dresult_state_arrays, input_state_structure, n - 1)
            initial_adjoint_state = rk_util.weighted_sum(
                [1.0, 1.0], [adjoint_state, initial_adjoint])
            initial_augmented_state = (
                initial_state, initial_adjoint_state, adjoint_variable_state)
            # TODO(b/138304303): Allow the user to specify the Hessian of
            # `ode_fn` so that we can get the Jacobian of the adjoint system.
            # TODO(b/143624114): Support higher order derivatives.
            augmented_results = self._solve(
                ode_fn=augmented_ode_fn,
                initial_time=-lower_bound_of_integration,
                initial_state=initial_augmented_state,
                solution_times=[-upper_bound_of_integration],
                batch_ndims=batch_ndims
            )
            # Results added an extra time dim of size 1, squeeze it.
            select_result = lambda x: tf.squeeze(x, [0])
            result_state = augmented_results.states
            result_state = tf.nest.map_structure(select_result, result_state)
            status = augmented_results.diagnostics.status
            return n - 1, result_state, status

          _, augmented_state, _ = tf.while_loop(
              lambda n, _, status: (n >= 1) & tf.equal(status, 0),
              reverse_to_result_time,
              (num_result_times - 1, terminal_augmented_state, 0),
              back_prop=False
          )
          _, adjoint_state, adjoint_variables = augmented_state
          return adjoint_state, list(adjoint_variables)

      return results, grad_fn

    # TODO(b/140760650): We must use a resource-using variable scope, otherwise
    # custom_gradient will complain even if there are no variables in `ode_fn`.
    flat_initial_state = tf.nest.flatten(initial_state)
    with tf1.variable_scope(tf1.get_variable_scope(), use_resource=True):
      return gradient_helper(*flat_initial_state)

  @abc.abstractmethod
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
    """Abstract method called by `solve`; to be implemented by child classes."""
    pass


class Results(
    collections.namedtuple(
        'Results',
        ['times', 'states', 'diagnostics', 'solver_internal_state'])):
  """Results returned by a Solver.

  Properties:
    times: A 1-D float `Tensor` satisfying `times[i] < times[i+1]`.
    states: A (1+N)-D `Tensor` containing the state at each time. In particular,
      `states[i]` is the state at time `times[i]`.
    diagnostics: Object of type `Diagnostics` containing performance
      information.
    solver_internal_state: Solver-specific object which can be used to
    warm-start the solver on a future invocation of `solve`.
  """
  __slots__ = ()


@six.add_metaclass(abc.ABCMeta)
class Diagnostics(object):
  """Diagnostics returned by a Solver."""

  @abc.abstractproperty
  def num_ode_fn_evaluations(self):
    """Number of function evaluations.

    Returns:
      num_ode_fn_evaluations: Scalar integer `Tensor` containing the number of
        function evaluations.
    """
    pass

  @abc.abstractproperty
  def num_jacobian_evaluations(self):
    """Number of Jacobian evaluations.

    Returns:
      num_jacobian_evaluations: Scalar integer `Tensor` containing number of
        Jacobian evaluations.
    """
    pass

  @abc.abstractproperty
  def num_matrix_factorizations(self):
    """Number of matrix factorizations.

    Returns:
      num_matrix_factorizations: Scalar integer `Tensor` containing the number
        of matrix factorizations.
    """
    pass

  @abc.abstractproperty
  def status(self):
    """Completion status.

    Returns:
      status: Scalar integer `Tensor` containing the reason for termination. -1
        on failure, 1 on termination by an event, and 0 otherwise.
    """
    pass

  @property
  def success(self):
    """Boolean indicating whether or not the method succeeded.

    Returns:
      success: Boolean `Tensor` equivalent to `status >= 0`.
    """
    return self.status >= 0


class ChosenBySolver(collections.namedtuple('ChosenBySolver', ['final_time'])):
  """Sentinel used to modify the behaviour of the `solve` method of a solver.

  Can be passed as the `solution_times` argument in the `solve` method of a
  solver. Doing so requests that the solver automatically choose suitable times
  at which to store the computed state (see `tfp.math.ode.Base.solve`).

  Properties:
    final_time: Scalar float `Tensor` specifying the largest time at which to
      store the computed state.
  """
  __slots__ = ()


def _read_solution_components(solutions_arrays, structure, time_id):
  """Composes `struct` from `time_id` slices of `solutions_arrays`.

  Args:
    solutions_arrays: List of `TensorArray`s holding components of solutions at
      different time steps.
    structure: Possibly nested structure of `Tensor`s representing solution
      state as defined in corresponding ODE.
    time_id: Scalar integer indicating which time steo to read.

  Returns:
    solution: Solution of the same structure as `structure` assembled from
      components in solutions array.
  """
  tf.nest.assert_same_structure(structure, solutions_arrays)
  read_solution = lambda array: array.read(time_id)
  solution = tf.nest.map_structure(read_solution, solutions_arrays)
  return solution
