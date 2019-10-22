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
import numpy as np
import six
import tensorflow.compat.v2 as tf

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

    @tf.custom_gradient
    def gradient_helper(initial_state):
      """Inner method used to restrict gradient op to `initial_state`."""
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
      # Call stop_gradient on members whose gradients we do not compute.
      results = Results(
          times=tf.stop_gradient(results.times),
          states=results.states,
          diagnostics=util.stop_gradient_of_real_or_complex_entries(
              results.diagnostics),
          solver_internal_state=util.stop_gradient_of_real_or_complex_entries(
              results.solver_internal_state))

      def grad_fn(*dresults):
        """Adjoint sensitivity method to compute gradients."""
        dresults = tf.nest.pack_sequence_as(results, dresults)
        dstates = dresults.states
        # TODO(b/138304303): Support complex types.
        state_dtype = initial_state.dtype
        if state_dtype.is_complex:
          raise NotImplementedError('The adjoint sensitivity method does not '
                                    'support complex dtypes.')
        with tf.name_scope('{}Gradients'.format(self._name)):
          state_shape = tf.shape(initial_state)
          state_vec_tensor_shape = tf.reshape(initial_state, [-1]).get_shape()
          num_odes = tf.size(initial_state)
          ode_fn_vec = util.get_ode_fn_vec(ode_fn, state_shape)
          real_dtype = tf.abs(initial_state).dtype
          result_times = tf.concat(
              [[tf.cast(initial_time, real_dtype)], results.times], 0)
          num_result_times = tf.size(result_times)
          # The XLA compiler does not compile code which slices/indexes using
          # integer `Tensor`s. `TensorArray`s are used to get around this.
          result_time_array = tf.TensorArray(
              results.times.dtype,
              clear_after_read=False,
              size=num_result_times,
              element_shape=[]).unstack(result_times)
          jacobian_fn_mat = util.get_jacobian_fn_mat(
              jacobian_fn,
              ode_fn_vec,
              state_shape,
              use_pfor=self._use_pfor_to_compute_jacobian)
          result_state_vec_array = tf.TensorArray(
              state_dtype,
              size=num_result_times,
              dynamic_size=False,
              element_shape=state_vec_tensor_shape).unstack(
                  tf.reshape(results.states, [num_result_times - 1, -1]))
          dstate_vec_array = tf.TensorArray(
              state_dtype,
              size=num_result_times - 1,
              dynamic_size=False,
              element_shape=state_vec_tensor_shape).unstack(
                  tf.reshape(dstates, [num_result_times - 1, -1]))
          terminal_augmented_state_vec = tf.zeros([num_odes * 2],
                                                  dtype=state_dtype)

          def augmented_ode_fn_vec(backward_time, augmented_state_vec):
            """Dynamics function for the augmented system."""
            # The ODE solver cannot handle the case initial_time > final_time
            # and hence a change of variables backward_time = -time is used.
            time = -backward_time
            state_vec, adjoint_state_vec = _decompose_augmented(
                augmented_state_vec)
            ode_vec = ode_fn_vec(time, state_vec)
            # The adjoint ODE is
            # adj'(t) = -dot(adj(t).transpose(), jacobian_fn(t, state(t)).
            # The negative sign disappears after the change of variables.
            adjoint_ode_vec = util.right_mult_by_jacobian_mat(
                jacobian_fn_mat, ode_fn_vec, time, state_vec, adjoint_state_vec)
            augmented_ode_vec = _compose_augmented(-ode_vec, adjoint_ode_vec)
            return augmented_ode_vec

          def reverse_to_result_time(n, augmented_state_vec, _):
            """Integrates the augmented system backwards in time."""
            lower_bound_of_integration = result_time_array.read(n)
            upper_bound_of_integration = result_time_array.read(n - 1)
            _, adjoint_state_vec = _decompose_augmented(augmented_state_vec)
            adjoint_state_vec.set_shape(state_vec_tensor_shape)
            augmented_state_vec = _compose_augmented(
                result_state_vec_array.read(n - 1),
                adjoint_state_vec + dstate_vec_array.read(n - 1))
            # TODO(b/138304303): Allow the user to specify the Hessian of
            # `ode_fn` so that we can get the Jacobian of the adjoint system.
            augmented_results = self._solve(
                augmented_ode_fn_vec,
                -lower_bound_of_integration,
                augmented_state_vec,
                [-upper_bound_of_integration],
                jacobian_fn=None,
                jacobian_sparsity=None,
                batch_ndims=batch_ndims,
                previous_solver_internal_state=None,
            )
            return (n - 1, augmented_results.states[0],
                    augmented_results.diagnostics.status)

          _, initial_augmented_state_vec, status = tf.while_loop(
              lambda n, _, status: (n >= 1) & tf.equal(status, 0),
              reverse_to_result_time,
              (num_result_times - 1, terminal_augmented_state_vec, 0),
          )
          _, initial_adjoint_state_vec = _decompose_augmented(
              initial_augmented_state_vec)
          on_success = tf.reshape(initial_adjoint_state_vec, state_shape)
          on_failure = np.nan * tf.ones(state_shape, dtype=state_dtype)
          return tf.where(tf.equal(status, 0), on_success, on_failure)

      return results, grad_fn

    return gradient_helper(initial_state)

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


def _compose_augmented(state_vec, adjoint_state_vec):
  """Forms the augmented state from individual components."""
  return tf.concat([state_vec, adjoint_state_vec], 0)


def _decompose_augmented(augmented_state_vec):
  """Splits up the augmented state into individual components."""
  return tf.split(augmented_state_vec, 2, 0)
