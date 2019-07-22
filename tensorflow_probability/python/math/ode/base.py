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

# TODO(parsiad): Support MATLAB-style events.
# TODO(parsiad): Support nested structures for initial_state.

__all__ = [
    'ChosenBySolver',
    'Diagnostics',
    'Results',
    'Solver',
]


@six.add_metaclass(abc.ABCMeta)
class Solver(object):
  """Base class for an ODE solver."""

  @abc.abstractmethod
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
