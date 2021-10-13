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

import abc
import collections
import functools
import six
import tensorflow.compat.v1 as tf1
import tensorflow.compat.v2 as tf

from tensorflow_probability.python.internal import custom_gradient as tfp_custom_gradient
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.math import gradient as tfp_gradient
from tensorflow_probability.python.math.ode import runge_kutta_util as rk_util
from tensorflow_probability.python.math.ode import util
from tensorflow.python.util import deprecation  # pylint: disable=g-direct-tensorflow-import

# TODO(b/138303336): Support MATLAB-style events.

__all__ = [
    'ChosenBySolver',
    'Diagnostics',
    'Results',
    'Solver',
]


@six.add_metaclass(abc.ABCMeta)
class Solver(object):
  """Base class for an ODE solver."""

  @deprecation.deprecated_args(
      '2021-11-01',
      'use_pfor_to_compute_jacobian is deprecated, and does nothing.',
      'use_pfor_to_compute_jacobian')
  def __init__(self,
               make_adjoint_solver_fn,
               validate_args,
               name,
               use_pfor_to_compute_jacobian=True):
    del use_pfor_to_compute_jacobian
    self._validate_args = validate_args
    self._name = name
    if make_adjoint_solver_fn is None:
      make_adjoint_solver_fn = lambda: self
    self._make_adjoint_solver_fn = make_adjoint_solver_fn

  @property
  def name(self):
    return self._name

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
      constants=None,
  ):
    """Solves an initial value problem.

    An initial value problem consists of a system of ODEs and an initial
    condition:

    ```none
    dy/dt(t) = ode_fn(t, y(t), **constants)
    y(initial_time) = initial_state
    ```

    Here, `t` (also called time) is a scalar float `Tensor` and `y(t)` (also
    called the state at time `t`) is an N-D float or complex `Tensor`.
    `constants` is are values that are constant with respect to time. Passing
    the constants here rather than just closing over them in `ode_fn` is only
    necessary if you want gradients with respect to these values.

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

    If the exact solution times are not important, it can be much
    more efficient to let the solver choose them using
    `solution_times=tfp.math.ode.ChosenBySolver(final_time=1.)`.
    This yields the state at various times between `t_init` and `final_time`,
    in which case `results.states[i]` is the state at time `results.times[i]`.

    #### Gradients

    The gradients are computed using the adjoint sensitivity method described in
    [Chen et al. (2018)][1].

    ```python
    grad = tf.gradients(y1, y0) # == dot(e, J)
    # J is the Jacobian of y1 with respect to y0. In this case, J = exp(A * t1).
    # e = [1, ..., 1] is the row vector of ones.
    ```

    This is not capable of computing gradients with respect to values closed
    over by `ode_fn`, e.g., in the example above:

    ```python
    def ode_fn(t, y):
      return tf.linalg.matvec(A, y)

    with tf.GradientTape() as tape:
      tape.watch(A)
      results = tfp.math.ode.BDF().solve(ode_fn, t_init, y_init,
                                         solution_times=[t0, t1])
    tape.gradient(results.states, A)  # Undefined!
    ```

    There are two options to get the gradients flowing through these values:

    1. Use `tf.Variable` for these values.
    2. Pass the values in explicitly using the `constants` argument:

    ```python
    def ode_fn(t, y, A):
      return tf.linalg.matvec(A, y)

    with tf.GradientTape() as tape:
      tape.watch(A)
      results = tfp.math.ode.BDF().solve(ode_fn, t_init, y_init,
                                         solution_times=[t0, t1],
                                         constants={'A': A})
    tape.gradient(results.states, A)  # Fine.
    ```

    By default, this uses the same solver for the augmented ODE. This can be
    controlled via `make_adjoint_solver_fn`.

    #### References

    [1]: Chen, Tian Qi, et al. "Neural ordinary differential equations."
         Advances in Neural Information Processing Systems. 2018.

    Args:
      ode_fn: Function of the form `ode_fn(t, y, **constants)`. The input `t` is
        a scalar float `Tensor`. The input `y` and output are both `Tensor`s
        with the same shape and `dtype` as `initial_state`. `constants` is are
        values that are constant with respect to time. Passing the constants
        here rather than just closing over them in `ode_fn` is only necessary if
        you want gradients with respect to these values.
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
      constants: Optional dictionary with string keys and values being (possibly
        nested) float `Tensor`s. These represent values that are constant with
        respect to time. Specifying these here allows the adjoint sentitivity
        method to compute gradients of the results with respect to these values.

    Returns:
      Object of type `Results`.
    """
    if constants is None:
      constants = {}
    input_state_structure = initial_state
    initial_state, constants = tf.nest.map_structure(tf.convert_to_tensor,
                                                     (initial_state, constants))

    def vjp_fwd(initial_state, constants):
      results = self._solve(
          ode_fn=functools.partial(ode_fn, **constants),
          initial_time=initial_time,
          initial_state=initial_state,
          solution_times=solution_times,
          jacobian_fn=jacobian_fn,
          jacobian_sparsity=jacobian_sparsity,
          batch_ndims=batch_ndims,
          previous_solver_internal_state=previous_solver_internal_state,
      )
      results = Results(
          times=tf.stop_gradient(results.times),
          states=results.states,
          diagnostics=util.stop_gradient_of_real_or_complex_entries(
              results.diagnostics),
          solver_internal_state=util.stop_gradient_of_real_or_complex_entries(
              results.solver_internal_state))
      return results, (results, constants)

    def vjp_bwd(results_constants, dresults, variables=()):
      """Adjoint sensitivity method to compute gradients."""
      results, constants = results_constants
      adjoint_solver = self._make_adjoint_solver_fn()
      dstates = dresults.states
      # TODO(b/138304303): Support complex types.
      with tf.name_scope('{}Gradients'.format(self._name)):
        get_dtype = lambda x: x.dtype
        def error_if_complex(dtype):
          if dtype_util.is_complex(dtype):
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
        # the last two component is adjoint state for variables and constants.
        terminal_augmented_state = tuple([
            rk_util.nest_constant(initial_state, 0.0),
            rk_util.nest_constant(initial_state, 0.0),
            tuple(
                rk_util.nest_constant(variable, 0.0) for variable in variables
            ),
            rk_util.nest_constant(constants, 0.0),
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
                clear_after_read=False,
                element_shape=component.shape[1:]).unstack(component)
            for component in tf.nest.flatten(results.states)
        ]
        result_state_arrays = tf.nest.pack_sequence_as(
            results.states, result_state_arrays)
        dresult_state_arrays = [
            tf.TensorArray(  # pylint: disable=g-complex-comprehension
                dtype=component.dtype, size=num_result_times - 1,
                clear_after_read=False,
                element_shape=component.shape[1:]).unstack(component)
            for component in tf.nest.flatten(dstates)
        ]
        dresult_state_arrays = tf.nest.pack_sequence_as(
            results.states, dresult_state_arrays)

        def augmented_ode_fn(backward_time, augmented_state):
          """Dynamics function for the augmented system.

          Describes a differential equation that evolves the augmented state
          backwards in time to compute gradients using the adjoint method.
          Augmented state consists of 4 components `(state, adjoint_state,
          vars, constants)` all evaluated at time `backward_time`:

          state: represents the solution of user provided `ode_fn`. The
            structure coincides with the `initial_state`.
          adjoint_state: represents the solution of the adjoint sensitivity
            differential equation as discussed below. Has the same structure
            and shape as `state`.
          variables: represent the solution of the adjoint equation for
            variable gradients. Represented as a `Tuple(Tensor, ...)` with as
            many tensors as there are `variables` variable outside this
            function.
          constants: represent the solution of the adjoint equation for
            constant gradients. Has the same structure and shape as
            `constants` variable outside this function.

          The adjoint sensitivity equation describes the gradient of the
          solution with respect to the value of the solution at a previous
          time t. Its dynamics are given by
          d/dt[adj(t)] = -1 * adj(t) @ jacobian(ode_fn(t, z), z)
          Which is computed as:
          d/dt[adj(t)]_i = -1 * sum_j(adj(t)_j * d/dz_i[ode_fn(t, z)_j)]
          d/dt[adj(t)]_i = -1 * d/dz_i[sum_j(no_grad_adj_j * ode_fn(t, z)_j)]
          where in the last line we moved adj(t)_j under derivative by
          removing gradient from it.

          Adjoint equation for the gradient with respect to every
          `tf.Variable` and constant theta follows:
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
            adjoint_constants_ode: Structure of `Tensor`s equal to backwards
              time derivative of the `constants` component.
          """
          # The negative signs disappears after the change of variables.
          # The ODE solver cannot handle the case initial_time > final_time
          # and hence a change of variables backward_time = -time is used.
          time = -backward_time
          state, adjoint_state, _, _ = augmented_state

          # TODO(b/152464477): Doesn't work reliably in TF1.
          def grad_fn(state, variables, constants):
            del variables  # We compute these gradients via the GradientTape
            # capturing them.
            derivatives = ode_fn(time, state, **constants)
            adjoint_no_grad = tf.nest.map_structure(tf.stop_gradient,
                                                    adjoint_state)
            negative_derivatives = rk_util.weighted_sum([-1.0], [derivatives])

            def dot_prod(tensor_a, tensor_b):
              return tf.reduce_sum(tensor_a * tensor_b)

            # See docstring for details.
            adjoint_dot_derivatives = tf.nest.map_structure(
                dot_prod, adjoint_no_grad, derivatives)
            adjoint_dot_derivatives = tf.squeeze(
                tf.add_n(tf.nest.flatten(adjoint_dot_derivatives)))
            return adjoint_dot_derivatives, negative_derivatives

          values = (state, tuple(variables), constants)
          ((_, negative_derivatives),
           gradients) = tfp_gradient.value_and_gradient(
               grad_fn, values, has_aux=True, use_gradient_tape=True)

          (adjoint_ode, adjoint_variables_ode,
           adjoint_constants_ode) = tf.nest.map_structure(
               lambda v, g: tf.zeros_like(v) if g is None else g, values,
               tuple(gradients))
          return (negative_derivatives, adjoint_ode, adjoint_variables_ode,
                  adjoint_constants_ode)

        def make_augmented_state(n, prev_augmented_state):
          """Constructs the augmented state for step `n`."""
          (_, adjoint_state, adjoint_variable_state,
           adjoint_constant_state) = prev_augmented_state
          initial_state = _read_solution_components(
              result_state_arrays,
              input_state_structure,
              n - 1,
          )
          initial_adjoint = _read_solution_components(
              dresult_state_arrays,
              input_state_structure,
              n - 1,
          )
          initial_adjoint_state = rk_util.weighted_sum(
              [1.0, 1.0], [adjoint_state, initial_adjoint])
          augmented_state = (
              initial_state,
              initial_adjoint_state,
              adjoint_variable_state,
              adjoint_constant_state,
          )
          return augmented_state

        def reverse_to_result_time(n, augmented_state, solver_internal_state,
                                   _):
          """Integrates the augmented system backwards in time."""
          lower_bound_of_integration = result_time_array.read(n)
          upper_bound_of_integration = result_time_array.read(n - 1)
          initial_augmented_state = make_augmented_state(n, augmented_state)
          # TODO(b/138304303): Allow the user to specify the Hessian of
          # `ode_fn` so that we can get the Jacobian of the adjoint system.
          # TODO(b/143624114): Support higher order derivatives.
          solver_internal_state = (
              adjoint_solver._adjust_solver_internal_state_for_state_jump(  # pylint: disable=protected-access
                  ode_fn=augmented_ode_fn,
                  initial_time=-lower_bound_of_integration,
                  initial_state=initial_augmented_state,
                  previous_solver_internal_state=solver_internal_state,
                  previous_state=augmented_state,
              ))
          augmented_results = adjoint_solver.solve(
              ode_fn=augmented_ode_fn,
              initial_time=-lower_bound_of_integration,
              initial_state=initial_augmented_state,
              solution_times=[-upper_bound_of_integration],
              batch_ndims=batch_ndims,
              previous_solver_internal_state=solver_internal_state,
          )
          # Results added an extra time dim of size 1, squeeze it.
          select_result = lambda x: tf.squeeze(x, [0])
          result_state = augmented_results.states
          result_state = tf.nest.map_structure(select_result, result_state)
          status = augmented_results.diagnostics.status
          return (n - 1, result_state,
                  augmented_results.solver_internal_state, status)

        initial_n = num_result_times - 1
        solver_internal_state = adjoint_solver._initialize_solver_internal_state(  # pylint: disable=protected-access
            ode_fn=augmented_ode_fn,
            initial_time=result_time_array.read(initial_n),
            initial_state=make_augmented_state(initial_n,
                                               terminal_augmented_state),
        )

        _, augmented_state, _, _ = tf.while_loop(
            lambda n, _as, _sis, status: (n >= 1) & tf.equal(status, 0),
            reverse_to_result_time,
            (initial_n, terminal_augmented_state, solver_internal_state, 0),
            back_prop=False,
        )
        (_, adjoint_state, adjoint_variables,
         adjoint_constants) = augmented_state

        if variables:
          return (adjoint_state, adjoint_constants), list(adjoint_variables)
        else:
          return adjoint_state, adjoint_constants

    @tfp_custom_gradient.custom_gradient(
        vjp_fwd=vjp_fwd,
        vjp_bwd=vjp_bwd,
    )
    def gradient_helper(initial_state, constants):
      """Restricts gradient to initial state components and constants."""
      return vjp_fwd(initial_state, constants)[0]

    # TODO(b/140760650): We must use a resource-using variable scope, otherwise
    # custom_gradient will complain even if there are no variables in `ode_fn`.
    with tf1.variable_scope(tf1.get_variable_scope(), use_resource=True):
      return gradient_helper(initial_state, constants)

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

  @abc.abstractmethod
  def _initialize_solver_internal_state(
      self,
      ode_fn,
      initial_time,
      initial_state,
  ):
    """Initializes the solver internal state."""
    pass

  def _adjust_solver_internal_state_for_state_jump(
      self,
      ode_fn,
      initial_time,
      initial_state,
      previous_solver_internal_state,
      previous_state,
  ):
    """Adjust the previous internal state in response to a state jump."""
    del previous_solver_internal_state
    del previous_state
    return self._initialize_solver_internal_state(
        ode_fn=ode_fn,
        initial_time=initial_time,
        initial_state=initial_state,
    )


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
