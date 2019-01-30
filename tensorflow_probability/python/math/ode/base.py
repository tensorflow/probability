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
    'Solver',
    'Solution',
    'Diagnostics',
]


@six.add_metaclass(abc.ABCMeta)
class Solver(object):
  """Base class for an ODE solver."""

  @abc.abstractmethod
  def solve(
      self,
      ode_fn,
      initial_state,
      initial_time=0.,
      final_time=None,
      solution_times=None,
      jacobian_fn=None,
      jacobian_sparsity=None,
      batch_ndims=None,
      previous_results=None,
  ):
    """Solves an initial value problem.

    An initial value problem consists of a system of ODEs and an initial
    condition:

    ```none
    dy/dt(t) = ode_fn(t, y(t))
    y(initial_time) = initial_state
    ```

    Here, `t` (also called time) is a scalar `Tensor` of real `dtype` and `y(t)`
    (also called the state at time `t`) is an N-D float or complex `Tensor`.

    Args:
      ode_fn: Function of the form `ode_fn(t, y)`. The input `t` is a scalar
        `Tensor` of real `dtype`. The input `y` and output are both `Tensor`s
        with the same shape and `dtype` as `initial_state`.
      initial_state: N-D float or complex `Tensor` specifying the initial state.
        The `dtype` of `initial_state` must be complex for problems with
        complex-valued states (even if the initial state is real).
      initial_time: Scalar `Tensor` of real `dtype` specifying the initial time.
        Default value: `0.`.
      final_time: Optional scalar `Tensor` of real `dtype` specifying the final
        time to integrate up to. Must satisfy `initial_time < final_time`. If
        unspecified, the solver will use `solution_times[-1]` (see below) as the
        final time. Therefore, at least one of `final_time` or `solution_times`
        must be specified.
        Default value: `None` (i.e., `solution_times[-1]` or error).
      solution_times: Optional 1-D float `Tensor` specifying a list of times. If
        specified, the solver stores the computed state at each of these times
        in the returned `Solution` object. If unspecified, the solver will
        choose this list of times automatically. Must satisfy `initial_time <=
        solution_times[0]`, `solution_times[i] < solution_times[i+1]`, and
        `solution_times[-1] <= final_time` if `final_time` is specified.
        Default value: `None`.
      jacobian_fn: Optional function of the form `jacobian_fn(t, y)`. The input
        `t` is a scalar `Tensor` of real `dtype`. The input `y` has the same
        shape and `dtype` as `initial_state`. The output is a 2N-D `Tensor`
        whose shape is `initial_state.shape + initial_state.shape` and whose
        `dtype` is the same as `initial_state`. In particular, the `(i1, ...,
        iN, j1, ..., jN)`-th entry of `jacobian_fn(t, y)` is the derivative of
        the `(i1, ..., iN)`-th entry of `ode_fn(t, y)` with respect to the `(j1,
        ..., jN)`-th entry of `y`. If this argument is left unspecified, the
        solver will automatically compute the Jacobian if and when it is needed.
        Default value: `None`.
      jacobian_sparsity: Optional 2N-D boolean `Tensor` whose shape is
        `initial_state.shape + initial_state.shape` specifying the sparsity
        pattern of the Jacobian. This argument is ignored if `jacobian_fn` is
        specified.
        Default value: `None`.
      batch_ndims: Optional nonnegative integer. When specified, the first
        `batch_ndims` dimensions of `initial_state` are batch dimensions.
        Default value: `None`.
      previous_results: Optional solver-specific argument used to warm-start
        this invocation of `solve`.
        Default value: `None`.

    Returns:
      solution: Object of type `Solution` containing the solution.
      diagnostics: Object of type `Diagnostics` containing performance
        information.
      results: Solver-specific object which can be used to warm-start the solver
        on a future invocation of `solve`.
    """


class Solution(collections.namedtuple('Solution', ['times', 'state'])):
  """Solution returned by a Solver.

  Properties:
    times: A 1-D float `Tensor` satisfying `times[i] < times[i+1]`.
    state: A (1+N)-D `Tensor` containing the state at each time. In particular,
      `state[i]` is the state at time `times[i]`.
  """
  pass


@six.add_metaclass(abc.ABCMeta)
class Diagnostics(object):
  """Diagnostics returned by a Solver."""

  @abc.abstractproperty
  def num_ode_fn_evaluations(self):
    """Number of function evaluations.

    Returns:
      num_ode_fn_evaluations: Scalar `Tensor` of integral `dtype` containing the
        number of function evaluations.
    """
    pass

  @abc.abstractproperty
  def num_jacobian_evaluations(self):
    """Number of Jacobian evaluations.

    Returns:
      num_jacobian_evaluations: Scalar `Tensor` of integral `dtype` containing
        number of Jacobian evaluations.
    """
    pass

  @abc.abstractproperty
  def num_matrix_factorizations(self):
    """Number of matrix factorizations.

    Returns:
      num_matrix_factorizations: Scalar `Tensor` of integral `dtype` containing
        the number of matrix factorizations.
    """
    pass

  @abc.abstractproperty
  def status(self):
    """Completion status.

    Returns:
      status: Scalar `Tensor` of integral `dtype` containing the reason for
        termination. -1 on failure, 1 on termination by an event, and 0
        otherwise.
    """
    pass

  @property
  def success(self):
    """Boolean indicating whether or not the method succeeded.

    Returns:
      success: Boolean `Tensor` equivalent to `status >= 0`.
    """
    return self.status >= 0
