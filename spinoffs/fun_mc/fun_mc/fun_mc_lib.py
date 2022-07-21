# Copyright 2021 The TensorFlow Probability Authors.
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
"""Functional MC: A functional API for creating new Markov Chains.

The core convention of this API is that transition operators have the following
form:

```
transition_operator(state...) -> (new_state..., extra_outputs)
```

Where 'x...', reresents one or more values. This operator can then be called
recursively as follows:

```
state = ...
while not_done:
  state, extra = transition_operator(*state)
```
"""

import collections
import functools
import typing
from typing import Any, Callable, List, Mapping, NamedTuple, Optional, Sequence, Text, Tuple, Union  # pylint: disable=unused-import

import numpy as np

from fun_mc import backend

tf = backend.tf
tfp = backend.tfp
util = backend.util
ps = backend.prefer_static
tfb = tfp.bijectors

__all__ = [
    'adam_init',
    'adam_step',
    'AdamExtra',
    'AdamState',
    'blanes_3_stage_step',
    'blanes_4_stage_step',
    'call_fn',
    'call_potential_fn',
    'call_potential_fn_with_grads',
    'call_transition_operator',
    'call_transport_map',
    'call_transport_map_with_ldj',
    'choose',
    'clip_grads',
    'gaussian_proposal',
    'gaussian_momentum_sample',
    'gradient_descent_init',
    'gradient_descent_step',
    'GradientDescentExtra',
    'GradientDescentState',
    'hamiltonian_integrator',
    'hamiltonian_monte_carlo_step',
    'hamiltonian_monte_carlo_init',
    'HamiltonianMonteCarloExtra',
    'HamiltonianMonteCarloState',
    'IntegratorExtras',
    'IntegratorState',
    'IntegratorStep',
    'IntegratorStepState',
    'leapfrog_step',
    'make_gaussian_kinetic_energy_fn',
    'make_surrogate_loss_fn',
    'maximal_reflection_coupling_proposal',
    'MaximalReflectiveCouplingProposalExtra',
    'maybe_broadcast_structure',
    'mclachlan_optimal_4th_order_step',
    'metropolis_hastings_step',
    'MetropolisHastingsExtra',
    'obabo_langevin_integrator',
    'persistent_metropolis_hastings_init',
    'persistent_metropolis_hastings_step',
    'PersistentMetropolistHastingsExtra',
    'PersistentMetropolistHastingsState',
    'potential_scale_reduction_extract',
    'potential_scale_reduction_init',
    'potential_scale_reduction_step',
    'PotentialFn',
    'PotentialScaleReductionState',
    'random_walk_metropolis_init',
    'random_walk_metropolis_step',
    'RandomWalkMetropolisExtra',
    'RandomWalkMetropolisState',
    'recover_state_from_args',
    'reparameterize_potential_fn',
    'running_approximate_auto_covariance_init',
    'running_approximate_auto_covariance_step',
    'running_covariance_init',
    'running_covariance_step',
    'running_mean_init',
    'running_mean_step',
    'running_variance_init',
    'running_variance_step',
    'RunningApproximateAutoCovarianceState',
    'RunningCovarianceState',
    'RunningMeanState',
    'RunningVarianceState',
    'ruth4_step',
    'sign_adaptation',
    'simple_dual_averages_init',
    'simple_dual_averages_step',
    'SimpleDualAveragesExtra',
    'SimpleDualAveragesState',
    'splitting_integrator_step',
    'State',
    'trace',
    'transform_log_prob_fn',
    'TransitionOperator',
    'TransportMap',
]

# We quote tf types to avoid unconditionally loading the TF backend.
AnyTensor = Union['tf.Tensor', np.ndarray, np.generic]
BooleanTensor = Union[bool, 'tf.Tensor', np.ndarray, np.bool_]
IntTensor = Union[int, 'tf.Tensor', np.ndarray, np.integer]
FloatTensor = Union[float, 'tf.Tensor', np.ndarray, np.floating]
# TODO(b/109648354): Correctly represent the recursive nature of this type.
TensorNest = Union[AnyTensor, Sequence[AnyTensor], Mapping[Any, AnyTensor]]
TensorSpecNest = Union['tf.TensorSpec', Sequence['tf.TensorSpec'],
                       Mapping[Any, 'tf.TensorSpec']]
BijectorNest = Union[tfb.Bijector, Sequence[tfb.Bijector],
                     Mapping[Any, tfb.Bijector]]
BooleanNest = Union[BooleanTensor, Sequence[BooleanTensor],
                    Mapping[Any, BooleanTensor]]
FloatNest = Union[FloatTensor, Sequence[FloatTensor], Mapping[Any, FloatTensor]]
IntNest = Union[IntTensor, Sequence[IntTensor], Mapping[Any, IntTensor]]
StringNest = Union[Text, Sequence[Text], Mapping[Any, Text]]
DTypeNest = Union['tf.DType', Sequence['tf.DType'], Mapping[Any, 'tf.DType']]
State = TensorNest  # pylint: disable=invalid-name
TransitionOperator = Callable[..., Tuple[State, TensorNest]]
TransportMap = Callable[..., Tuple[State, TensorNest]]
PotentialFn = Union[Callable[[TensorNest], Tuple['tf.Tensor', TensorNest]],
                    Callable[..., Tuple['tf.Tensor', TensorNest]]]
GradFn = Union[Callable[[TensorNest], Tuple[TensorNest, TensorNest]],
               Callable[..., Tuple[TensorNest, TensorNest]]]


def _trace_extra(state: 'State', extra: 'TensorNest') -> 'TensorNest':
  del state
  return extra


@util.named_call
def trace(
    state: 'State',
    fn: 'TransitionOperator',
    num_steps: 'IntTensor',
    trace_fn: 'Callable[[State, TensorNest], TensorNest]' = _trace_extra,
    trace_mask: 'BooleanNest' = True,
    unroll: 'bool' = False,
    parallel_iterations: 'int' = 10,
) -> 'Tuple[State, TensorNest]':
  """`TransitionOperator` that runs `fn` repeatedly and traces its outputs.

  Args:
    state: A nest of `Tensor`s or None.
    fn: A `TransitionOperator`.
    num_steps: Number of steps to run the function for. Must be greater than 1.
    trace_fn: Callable that the unpacked outputs of `fn` and returns a nest of
      `Tensor`s. These will potentially be stacked and returned as the second
      return value. By default, just the `extra` return from `fn` is returned.
    trace_mask: A potentially shallow nest with boolean leaves applied to the
      return value of `trace_fn`. This controls whether or not to actually trace
      the quantities returned from `trace_fn`. For subtrees of return value of
      `trace_fn` where the mask leaf is `True`, those subtrees are traced (i.e.
      the corresponding subtrees in `traces` will contain an extra leading
      dimension equalling `num_steps`). For subtrees of the return value of
      `trace_fn` where the mask leaf is `False`, those subtrees are merely
      propagated, and their corresponding subtrees in `traces` correspond to
      their final value.
    unroll: Whether to unroll the loop. This can occasionally lead to improved
      performance at the cost of increasing the XLA optimization time. Only
      works if `num_steps` is statically known.
    parallel_iterations: Number of iterations of the while loop to run in
      parallel.

  Returns:
    state: The final state returned by `fn`.
    traces: A nest with the same structure as that of `trace_fn` return value,
      but with leaves replaced with stacked and unstacked values according to
      the `trace_mask`.

  #### Example

  ```python
  def fn(x):
    return x + 1, (2 * x, 3 * x)

  # Full trace.
  state, traces = trace(0, fn, num_steps=3)
  assert(state == 3)
  assert(traces[0] == [0, 2, 4])
  assert(traces[1] == [0, 3, 6])

  # Partial trace.
  state, traces = trace(0, fn, num_steps=3, trace_mask=(True, False))
  assert(state == 3)
  assert(traces[0] == [0, 2, 4])
  assert(traces[1] == 6)

  # No trace.
  state, traces = trace(0, fn, num_steps=3, trace_mask=False)
  assert(state == 3)
  assert(traces[0] == 4)
  assert(traces[1] == 6)
  ```

  """

  def split_trace(trace_element):
    # The two return values of this function share the same shallow structure as
    # `trace_mask`, with the first return value being a version meant for
    # tracing, and the second return value meant for mere propagation.
    return (
        util.map_tree_up_to(
            trace_mask,
            lambda m, s: () if m else s,
            trace_mask,
            trace_element,
        ),
        util.map_tree_up_to(
            trace_mask,
            lambda m, s: s if m else (),
            trace_mask,
            trace_element,
        ),
    )

  def combine_trace(untraced, traced):
    # Reconstitute the structure returned by `trace_fn`, with leaves replaced
    # with traced and untraced elements according to the trace_mask. This is the
    # inverse operation of `split_trace`.
    def _select(trace_mask, traced, untraced):
      return traced if trace_mask else untraced

    return util.map_tree_up_to(trace_mask, _select, trace_mask, traced,
                               untraced)

  def wrapper(state):
    state, extra = util.map_tree(tf.convert_to_tensor,
                                 call_transition_operator(fn, state))
    trace_element = util.map_tree(tf.convert_to_tensor, trace_fn(state, extra))
    untraced, traced = split_trace(trace_element)
    return state, untraced, traced

  state = util.map_tree(lambda t: (t if t is None else tf.convert_to_tensor(t)),
                        state)

  state, untraced, traced = util.trace(
      state=state,
      fn=wrapper,
      num_steps=num_steps,
      unroll=unroll,
      parallel_iterations=parallel_iterations,
  )

  return state, combine_trace(untraced, traced)


def _tree_repr(tree: 'Any') -> 'Text':
  """Utility to get a string representation of the the structure of `tree`."""

  class _LeafSentinel(object):

    def __repr__(self):
      return '.'

  return str(util.map_tree(lambda _: _LeafSentinel(), tree))


def _is_namedtuple_like(x):
  """Helper which returns `True` if input is `collections.namedtuple`-like."""
  try:
    for fn in getattr(x, '_fields'):
      _ = getattr(x, fn)
    return True
  except AttributeError:
    return False


def call_fn(
    fn: 'TransitionOperator',
    args: 'Union[Tuple[Any], Mapping[Text, Any], Any]',
) -> 'Any':
  """Calls a function with `args`.

  If `args` is a sequence, `fn` is called like `fn(*args)`. If `args` is a
  mapping, `fn` is called like `fn(**args)`. Otherwise, it is called `fn(args)`.

  Args:
    fn: A `TransitionOperator`.
    args: Arguments to `fn`

  Returns:
    ret: Return value of `fn`.
  """
  if (isinstance(args, collections.abc.Sequence) and
      not _is_namedtuple_like(args)):
    return fn(*args)
  elif isinstance(args, collections.abc.Mapping):
    return fn(**args)
  else:
    return fn(args)


def recover_state_from_args(args: 'Sequence[Any]', kwargs: 'Mapping[Text, Any]',
                            state_structure: 'Any') -> 'Any':
  """Attempts to recover the state that was transmitted via *args, **kwargs."""
  orig_args = args
  if isinstance(state_structure, collections.abc.Mapping):
    state = type(state_structure)()
    # Mappings can be ordered and not ordered, and this information is lost when
    # passed via **kwargs. We iterate using the reference structure order so we
    # can reconstruct it. For unordered mappings the order doesn't matter. We
    # specifically don't use any sort of explicit sorting here, as that would
    # destroy the order of orderered mappings.
    for k in state_structure.keys():
      # This emulates the positional argument passing.
      if args:
        state[k] = args[0]
        args = args[1:]
      else:
        if k not in kwargs:
          raise ValueError(
              ('Missing \'{}\' from kwargs.\nargs=\n{}\nkwargs=\n{}\n'
               'state_structure=\n{}').format(k, orig_args, kwargs,
                                              _tree_repr(state_structure)))
        state[k] = kwargs[k]
    return state
  elif (isinstance(state_structure, collections.abc.Sequence) and
        not _is_namedtuple_like(state_structure)):
    # Sadly, we have no way of inferring the state index from kwargs, so we
    # disallow them.
    # TODO(siege): We could support length-1 sequences in principle.
    if kwargs:
      raise ValueError('This wrapper does not accept keyword arguments for a '
                       'sequence-like state structure=\n{}'.format(
                           _tree_repr(state_structure)))
    return type(state_structure)(args)
  elif args:
    return args[0]
  elif kwargs:
    return next(iter(kwargs.values()))
  else:
    # Must be the case that the state_structure is actually empty.
    assert not state_structure
    return state_structure


def call_potential_fn(
    fn: 'PotentialFn',
    args: 'Union[Tuple[Any], Mapping[Text, Any], Any]',
) -> 'Tuple[tf.Tensor, Any]':
  """Calls a transition operator with `args`.

  `fn` must fulfill the `PotentialFn` contract:

  ```python
  potential, extra = call_fn(fn, args)
  ```

  Args:
    fn: `PotentialFn`.
    args: Arguments to `fn`.

  Returns:
    ret: Return value of `fn`.

  Raises:
    TypeError: If `fn` doesn't fulfill the contract.
  """
  ret = call_fn(fn, args)
  error_template = ('`{fn:}` must have a signature '
                    '`fn(args) -> (tf.Tensor, extra)`'
                    ' but when called with `args=`\n{args:}\nreturned '
                    '`ret=`\n{ret:}\ninstead. The structure of '
                    '`args=`\n{args_s:}\nThe structure of `ret=`\n{ret_s:}\n'
                    'A common solution is to adjust the `return`s in `fn` to '
                    'be `return args, ()`.')

  if not isinstance(ret, collections.abc.Sequence) or len(ret) != 2:
    args_s = _tree_repr(args)
    ret_s = _tree_repr(ret)
    raise TypeError(
        error_template.format(
            fn=fn, args=args, ret=ret, args_s=args_s, ret_s=ret_s))
  return ret


def call_transition_operator(
    fn: 'TransitionOperator',
    args: 'State',
) -> 'Tuple[State, TensorNest]':
  """Calls a transition operator with `args`.

  `fn` must fulfill the `TransitionOperator` contract:

  ```python
  args_out, extra = call_fn(fn, args)
  assert_same_shallow_tree(args, args_out)
  ```

  Args:
    fn: `TransitionOperator`.
    args: Arguments to `fn`.

  Returns:
    ret: Return value of `fn`.

  Raises:
    TypeError: If `fn` doesn't fulfill the contract.
  """
  ret = call_fn(fn, args)
  error_template = ('`{fn:}` must have a signature '
                    '`fn(args) -> (new_args, extra)`'
                    ' but when called with `args=`\n{args:}\nreturned '
                    '`ret=`\n{ret:}\ninstead. The structure of '
                    '`args=`\n{args_s:}\nThe structure of `ret=`\n{ret_s:}\n'
                    'A common solution is to adjust the `return`s in `fn` to '
                    'be `return args, ()`.')

  if not isinstance(ret, collections.abc.Sequence) or len(ret) != 2:
    args_s = _tree_repr(args)
    ret_s = _tree_repr(ret)
    raise TypeError(
        error_template.format(
            fn=fn, args=args, ret=ret, args_s=args_s, ret_s=ret_s))

  error_template = (
      '`{fn:}` must have a signature '
      '`fn(args) -> (new_args, extra)`'
      ' but when called with `args=`\n{args:}\nreturned '
      '`new_args=`\n{new_args:}\ninstead. The structure of '
      '`args=`\n{args_s:}\nThe structure of `new_args=`\n{new_args_s:}\n')
  new_args, extra = ret
  try:
    util.assert_same_shallow_tree(args, new_args)
  except:
    args_s = _tree_repr(args)
    new_args_s = _tree_repr(new_args)
    raise TypeError(
        error_template.format(
            fn=fn,
            args=args,
            new_args=new_args,
            args_s=args_s,
            new_args_s=new_args_s))
  return new_args, extra


def call_transport_map(
    fn: 'TransportMap',
    args: 'State',
) -> 'Tuple[State, TensorNest]':
  """Calls a transport map with `args`.

  `fn` must fulfill the `TransportMap` contract:

  ```python
  out, extra = call_fn(fn, args)
  ```

  Args:
    fn: `TransitionOperator`.
    args: Arguments to `fn`.

  Returns:
    ret: Return value of `fn`.

  Raises:
    TypeError: If `fn` doesn't fulfill the contract.
  """

  ret = call_fn(fn, args)
  error_template = ('`{fn:}` must have a signature '
                    '`fn(args) -> (out, extra)`'
                    ' but when called with `args=`\n{args:}\nreturned '
                    '`ret=`\n{ret:}\ninstead. The structure of '
                    '`args=`\n{args_s:}\nThe structure of `ret=`\n{ret_s:}\n'
                    'A common solution is to adjust the `return`s in `fn` to '
                    'be `return args, ()`.')

  if not isinstance(ret, collections.abc.Sequence) or len(ret) != 2:
    args_s = _tree_repr(args)
    ret_s = _tree_repr(ret)
    raise TypeError(
        error_template.format(
            fn=fn, args=args, ret=ret, args_s=args_s, ret_s=ret_s))
  return ret


def call_transport_map_with_ldj(
    fn: 'TransitionOperator',
    args: 'State',
) -> 'Tuple[State, TensorNest, TensorNest]':
  """Calls `fn` and returns the log-det jacobian to `fn`'s first output.

  Args:
    fn: A `TransitionOperator`.
    args: Arguments to `fn`.

  Returns:
    ret: First output of `fn`.
    extra: Second output of `fn`.
    ldj: Log-det jacobian of `fn`.
  """

  def wrapper(args):
    return call_transport_map(fn, args)

  return util.value_and_ldj(wrapper, args)


def call_potential_fn_with_grads(
    fn: 'PotentialFn', args: 'Union[Tuple[Any], Mapping[Text, Any], Any]'
) -> 'Tuple[tf.Tensor, TensorNest, TensorNest]':
  """Calls `fn` and returns the gradients with respect to `fn`'s first output.

  Args:
    fn: A `PotentialFn`.
    args: Arguments to `fn`.

  Returns:
    ret: First output of `fn`.
    extra: Second output of `fn`.
    grads: Gradients of `ret` with respect to `args`.
  """

  def wrapper(args):
    return call_potential_fn(fn, args)

  return util.value_and_grad(wrapper, args)


def maybe_broadcast_structure(from_structure: 'Any',
                              to_structure: 'Any') -> 'Any':
  """Maybe broadcasts `from_structure` to `to_structure`.

  If `from_structure` is a singleton, it is tiled to match the structure of
  `to_structure`. Note that the elements in `from_structure` are not copied if
  this tiling occurs.

  Args:
    from_structure: A structure.
    to_structure: A structure.

  Returns:
    new_from_structure: Same structure as `to_structure`.
  """
  flat_from = util.flatten_tree(from_structure)
  flat_to = util.flatten_tree(to_structure)
  if len(flat_from) == 1:
    flat_from *= len(flat_to)
  return util.unflatten_tree(to_structure, flat_from)


def reparameterize_potential_fn(
    potential_fn: 'PotentialFn',
    transport_map_fn: 'TransportMap',
    init_state: 'Optional[State]' = None,
    state_structure: 'Optional[Any]' = None,
    track_volume: 'bool' = True,
) -> 'Tuple[PotentialFn, Optional[State]]':
  """Performs a change of variables of a potential function.

  This takes a potential function and creates a new potential function that now
  takes takes state in the domain of the `transport_map_fn`, transforms that
  state and calls the original potential function. If `track_volume` is True,
  then thay potential is treated as a log density of a volume element and is
  corrected with the log-det jacobian of `transport_map_fn`.

  This can be used to pre-condition and constrain probabilistic inference and
  optimization algorithms.

  The wrapped function has the following signature:
  ```none
    (*args, **kwargs) ->
      transformed_potential, [original_space_state, potential_extra,
                              transport_map_fn_extra]
  ```

  You can also pass `init_state` in the original space and this function will
  return the inverse transformed state as the 2nd return value. This requires
  that the `transport_map_fn` is invertible. If it is not, or the inverse is too
  expensive, you can skip passing `init_state` but then you need to pass
  `state_structure` so the code knows what state structure it should return (it
  is inferred from `init_state` otherwise).

  Args:
    potential_fn: A potential function.
    transport_map_fn: A `TransitionOperator` representing the transport map
      operating on the state. The action of this operator should take states
      from the transformed space to the original space.
    init_state: Initial state, in the original space.
    state_structure: Same structure as `init_state`. Mandatory if `init_state`
      is not provided.
    track_volume: Indicates that `potential_fn` represents a density and you
      want to transform the volume element. For example, this is True when
      `potential_fn` is used as `target_log_prob_fn` in probabilistic inference,
      and False when `potential_fn` is used in optimization.

  Returns:
    transformed_potential_fn: Transformed log prob fn.
    transformed_init_state: Initial state in the transformed space.
  """

  if state_structure is None and init_state is None:
    raise ValueError(
        'At least one of `state_structure` or `init_state` must be '
        'passed in.')

  def wrapper(*args, **kwargs):
    """Transformed wrapper."""
    real_state_structure = (
        state_structure if state_structure is not None else init_state)
    transformed_state = recover_state_from_args(args, kwargs,
                                                real_state_structure)

    if track_volume:
      state, map_extra, ldj = call_transport_map_with_ldj(
          transport_map_fn, transformed_state)
    else:
      state, map_extra = call_transport_map(transport_map_fn, transformed_state)

    potential, extra = call_potential_fn(potential_fn, state)

    if track_volume:
      potential += ldj

    return potential, [state, extra, map_extra]

  if init_state is not None:
    inverse_transform_map_fn = util.inverse_fn(transport_map_fn)
    transformed_state, _ = call_transport_map(inverse_transform_map_fn,
                                              init_state)
  else:
    transformed_state = None

  return wrapper, transformed_state


def transform_log_prob_fn(log_prob_fn: 'PotentialFn',
                          bijector: 'BijectorNest',
                          init_state: 'Optional[State]' = None) -> 'Any':
  """Transforms a log-prob function using a bijector.

  This takes a log-prob function and creates a new log-prob function that now
  takes takes state in the domain of the bijector, forward transforms that state
  and calls the original log-prob function. It then returns the log-probability
  that correctly accounts for this transformation.

  The wrapped function has the following signature:
  ```none
    (*args, **kwargs) ->
      transformed_space_state, [original_space_state, original_log_prob_extra]
  ```

  For convenience you can also pass the initial state (in the original space),
  and this function will return the inverse transformed state as the 2nd return
  value. You'd use this to initialize MCMC operators that operate in the
  transformed space.

  Args:
    log_prob_fn: Log prob fn.
    bijector: Bijector(s), must be of the same structure as the `log_prob_fn`
      inputs.
    init_state: Initial state, in the original space.

  Returns:
    transformed_log_prob_fn: Transformed log prob fn.
    transformed_init_state: If `init_state` is provided. Initial state in the
      transformed space.
  """

  bijector_structure = util.get_shallow_tree(
      lambda b: isinstance(b, tfb.Bijector), bijector)

  def wrapper(*args, **kwargs):
    """Transformed wrapper."""
    bijector_ = bijector

    args = recover_state_from_args(args, kwargs, bijector_)
    args = util.map_tree(lambda x: 0. + x, args)

    original_space_args = util.map_tree_up_to(bijector_structure,
                                              lambda b, x: b.forward(x),
                                              bijector_, args)
    original_space_log_prob, extra = call_potential_fn(log_prob_fn,
                                                       original_space_args)
    event_ndims = util.map_tree(
        lambda x: len(x.shape) - len(original_space_log_prob.shape), args)

    return original_space_log_prob + sum(
        util.flatten_tree(
            util.map_tree_up_to(
                bijector_structure,
                lambda b, x, e: b.forward_log_det_jacobian(x, event_ndims=e),
                bijector_, args, event_ndims))), [original_space_args, extra]

  if init_state is None:
    return wrapper
  else:
    return wrapper, util.map_tree_up_to(bijector_structure,
                                        lambda b, s: b.inverse(s), bijector,
                                        init_state)


class IntegratorStepState(NamedTuple):
  """Integrator step state."""
  state: 'State'
  state_grads: 'State'
  momentum: 'State'


class IntegratorStepExtras(NamedTuple):
  """Integrator step extras."""
  target_log_prob: 'FloatTensor'
  state_extra: 'Any'
  kinetic_energy: 'FloatTensor'
  kinetic_energy_extra: 'Any'
  momentum_grads: 'State'


IntegratorStep = Callable[[IntegratorStepState], Tuple[IntegratorStepState,
                                                       IntegratorStepExtras]]


@util.named_call
def splitting_integrator_step(
    integrator_step_state: 'IntegratorStepState',
    step_size: 'FloatTensor',
    target_log_prob_fn: 'PotentialFn',
    kinetic_energy_fn: 'PotentialFn',
    coefficients: 'Sequence[FloatTensor]',
    forward: 'bool' = True,
) -> 'Tuple[IntegratorStepState, IntegratorStepExtras]':
  """Symmetric symplectic integrator `TransitionOperator`.

  This implementation is based on Hamiltonian splitting, with the splits
  weighted by coefficients. We update the momentum first, if `forward` argument
  is `True`. See [1] for an overview of the method.

  Args:
    integrator_step_state: IntegratorStepState.
    step_size: Step size, structure broadcastable to the `target_log_prob_fn`
      state.
    target_log_prob_fn: Target log prob fn.
    kinetic_energy_fn: Kinetic energy fn.
    coefficients: Integrator coefficients.
    forward: Whether to run the integrator in the forward direction.

  Returns:
    integrator_step_state: IntegratorStepState.
    integrator_step_extras: IntegratorStepExtras.

  #### References:

  [1]: Sergio Blanes, Fernando Casas, J.M. Sanz-Serna. Numerical integrators for
       the Hybrid Monte Carlo method. SIAM J. Sci. Comput., 36(4), 2014.
       https://arxiv.org/pdf/1405.3153.pdf
  """
  if len(coefficients) < 2:
    raise ValueError('Too few coefficients. Need at least 2.')
  state = integrator_step_state.state
  state_grads = integrator_step_state.state_grads
  momentum = integrator_step_state.momentum
  # TODO(siege): Consider amortizing this across steps. The tricky bit here
  # is that only a few integrators take these grads.
  momentum_grads = None
  step_size = maybe_broadcast_structure(step_size, state)

  state = util.map_tree(tf.convert_to_tensor, state)
  momentum = util.map_tree(tf.convert_to_tensor, momentum)
  state = util.map_tree(tf.convert_to_tensor, state)

  idx_and_coefficients = enumerate(coefficients)
  if not forward:
    idx_and_coefficients = reversed(list(idx_and_coefficients))

  for i, c in idx_and_coefficients:
    # pylint: disable=cell-var-from-loop
    if i % 2 == 0:
      # Update momentum.
      state_grads = util.map_tree(tf.convert_to_tensor, state_grads)

      momentum = util.map_tree(lambda m, sg, s: m + c * sg * s, momentum,
                               state_grads, step_size)

      kinetic_energy, kinetic_energy_extra, momentum_grads = call_potential_fn_with_grads(
          kinetic_energy_fn, momentum)
    else:
      # Update position.
      if momentum_grads is None:
        _, _, momentum_grads = call_potential_fn_with_grads(
            kinetic_energy_fn, momentum)

      state = util.map_tree(lambda x, mg, s: x + c * mg * s, state,
                            momentum_grads, step_size)

      target_log_prob, state_extra, state_grads = call_potential_fn_with_grads(
          target_log_prob_fn, state)

  return (IntegratorStepState(state, state_grads, momentum),
          IntegratorStepExtras(target_log_prob, state_extra, kinetic_energy,
                               kinetic_energy_extra, momentum_grads))


@util.named_call
def leapfrog_step(
    integrator_step_state: 'IntegratorStepState',
    step_size: 'FloatTensor',
    target_log_prob_fn: 'PotentialFn',
    kinetic_energy_fn: 'PotentialFn',
) -> 'Tuple[IntegratorStepState, IntegratorStepExtras]':
  """Leapfrog integrator `TransitionOperator`.

  Args:
    integrator_step_state: IntegratorStepState.
    step_size: Step size, structure broadcastable to the `target_log_prob_fn`
      state.
    target_log_prob_fn: Target log prob fn.
    kinetic_energy_fn: Kinetic energy fn.

  Returns:
    integrator_step_state: IntegratorStepState.
    integrator_step_extras: IntegratorStepExtras.
  """
  coefficients = [0.5, 1., 0.5]
  return splitting_integrator_step(
      integrator_step_state,
      step_size,
      target_log_prob_fn,
      kinetic_energy_fn,
      coefficients=coefficients)


@util.named_call
def ruth4_step(
    integrator_step_state: 'IntegratorStepState',
    step_size: 'FloatTensor',
    target_log_prob_fn: 'PotentialFn',
    kinetic_energy_fn: 'PotentialFn',
) -> 'Tuple[IntegratorStepState, IntegratorStepExtras]':
  """Ruth 4th order integrator `TransitionOperator`.

  See [1] for details.

  Args:
    integrator_step_state: IntegratorStepState.
    step_size: Step size, structure broadcastable to the `target_log_prob_fn`
      state.
    target_log_prob_fn: Target log prob fn.
    kinetic_energy_fn: Kinetic energy fn.

  Returns:
    integrator_step_state: IntegratorStepState.
    integrator_step_extras: IntegratorStepExtras.

  #### References:

  [1]: Ruth, Ronald D. (August 1983). "A Canonical Integration Technique".
       Nuclear Science, IEEE Trans. on. NS-30 (4): 2669-2671
  """
  c = 2**(1. / 3)
  coefficients = (1. / (2 - c)) * np.array([0.5, 1., 0.5 - 0.5 * c, -c])
  coefficients = list(coefficients) + list(reversed(coefficients))[1:]
  return splitting_integrator_step(
      integrator_step_state,
      step_size,
      target_log_prob_fn,
      kinetic_energy_fn,
      coefficients=coefficients)


@util.named_call
def blanes_3_stage_step(
    integrator_step_state: 'IntegratorStepState',
    step_size: 'FloatTensor',
    target_log_prob_fn: 'PotentialFn',
    kinetic_energy_fn: 'PotentialFn',
) -> 'Tuple[IntegratorStepState, IntegratorStepExtras]':
  """Blanes 3 stage integrator `TransitionOperator`.

  This integrator has second order. See [1] for details and [2] for further
  analysis.

  Args:
    integrator_step_state: IntegratorStepState.
    step_size: Step size, structure broadcastable to the `target_log_prob_fn`
      state.
    target_log_prob_fn: Target log prob fn.
    kinetic_energy_fn: Kinetic energy fn.

  Returns:
    integrator_step_state: IntegratorStepState.
    integrator_step_extras: IntegratorStepExtras.

  #### References:

  [1]: Sergio Blanes, Fernando Casas, J.M. Sanz-Serna. Numerical integrators for
       the Hybrid Monte Carlo method. SIAM J. Sci. Comput., 36(4), 2014.
       https://arxiv.org/pdf/1405.3153.pdf

  [2]: Campos, C. M., & Sanz-Serna, J. M. Palindromic 3-stage splitting
       integrators, a roadmap, (8), 2017. https://arxiv.org/pdf/1703.09958.pdf
  """
  a1 = 0.11888010966
  b1 = 0.29619504261
  coefficients = [a1, b1, 0.5 - a1, 1. - 2. * b1]
  coefficients = coefficients + list(reversed(coefficients))[1:]
  return splitting_integrator_step(
      integrator_step_state,
      step_size,
      target_log_prob_fn,
      kinetic_energy_fn,
      coefficients=coefficients)


@util.named_call
def blanes_4_stage_step(
    integrator_step_state: 'IntegratorStepState',
    step_size: 'FloatTensor',
    target_log_prob_fn: 'PotentialFn',
    kinetic_energy_fn: 'PotentialFn',
) -> 'Tuple[IntegratorStepState, IntegratorStepExtras]':
  """Blanes 4 stage integrator `TransitionOperator`.

  See [1] for details. This integrator likely has second order.

  Args:
    integrator_step_state: IntegratorStepState.
    step_size: Step size, structure broadcastable to the `target_log_prob_fn`
      state.
    target_log_prob_fn: Target log prob fn.
    kinetic_energy_fn: Kinetic energy fn.

  Returns:
    integrator_step_state: IntegratorStepState.
    integrator_step_extras: IntegratorStepExtras.

  #### References:

  [1]: Sergio Blanes, Fernando Casas, J.M. Sanz-Serna. Numerical integrators for
       the Hybrid Monte Carlo method. SIAM J. Sci. Comput., 36(4), 2014.
       https://arxiv.org/pdf/1405.3153.pdf
  """
  a1 = 0.071353913
  a2 = 0.268548791
  b1 = 0.191667800
  coefficients = [a1, b1, a2, 0.5 - b1, 1. - 2. * (a1 + a2)]
  coefficients = coefficients + list(reversed(coefficients))[1:]
  return splitting_integrator_step(
      integrator_step_state,
      step_size,
      target_log_prob_fn,
      kinetic_energy_fn,
      coefficients=coefficients)


@util.named_call
def mclachlan_optimal_4th_order_step(
    integrator_step_state: 'IntegratorStepState',
    step_size: 'FloatTensor',
    target_log_prob_fn: 'PotentialFn',
    kinetic_energy_fn: 'PotentialFn',
    forward: 'BooleanTensor',
) -> 'Tuple[IntegratorStepState, IntegratorStepExtras]':
  """4th order integrator for Hamiltonians with a quadratic kinetic energy.

  See [1] for details. Note that this integrator step is not reversible, so for
  use in HMC you should randomly reverse the integration direction to preserve
  detailed balance.

  Args:
    integrator_step_state: IntegratorStepState.
    step_size: Step size, structure broadcastable to the `target_log_prob_fn`
      state.
    target_log_prob_fn: Target log prob fn.
    kinetic_energy_fn: Kinetic energy fn.
    forward: A scalar `bool` Tensor. Whether to run this integrator in the
      forward direction. Note that this is done for the entire state, not
      per-batch.

  Returns:
    integrator_step_state: IntegratorStepState.
    integrator_step_extras: IntegratorStepExtras.

  #### References:

  [1]: McLachlan R. I., & Atela P. (1992). The accuracy of symplectic
       integrators. Nonlinearity, 5, 541-562.
  """
  # N.B. a's and b's are used in the opposite sense than the Blanes integrators
  # above.
  a1 = 0.5153528374311229364
  a2 = -0.085782019412973646
  a3 = 0.4415830236164665242
  a4 = 0.1288461583653841854

  b1 = 0.1344961992774310892
  b2 = -0.2248198030794208058
  b3 = 0.7563200005156682911
  b4 = 0.3340036032863214255
  coefficients = [b1, a1, b2, a2, b3, a3, b4, a4]

  def _step(direction):
    return splitting_integrator_step(
        integrator_step_state,
        step_size,
        target_log_prob_fn,
        kinetic_energy_fn,
        coefficients=coefficients,
        forward=direction)

  # In principle we can avoid the cond, and use `tf.where` to select between the
  # coefficients. This would require a superfluous momentum update, but in
  # principle is feasible. We're not doing it because it would complicate the
  # code slightly, and there is limited motivation to do it since reversing the
  # directions for all the chains at once is typically valid as well.
  return tf.cond(forward, lambda: _step(True), lambda: _step(False))


class MetropolisHastingsExtra(NamedTuple):
  """Metropolis-hastings extra outputs."""
  is_accepted: 'FloatTensor'
  log_uniform: 'FloatTensor'


@util.named_call
def metropolis_hastings_step(
    current_state: 'State',
    proposed_state: 'State',
    energy_change: 'FloatTensor',
    log_uniform: 'Optional[FloatTensor]' = None,
    seed=None) -> 'Tuple[State, MetropolisHastingsExtra]':
  """Metropolis-Hastings step.

  This probabilistically chooses between `current_state` and `proposed_state`
  based on the `energy_change` so as to preserve detailed balance.

  Energy change is the negative of `log_accept_ratio`.

  Args:
    current_state: Current state.
    proposed_state: Proposed state.
    energy_change: E(proposed_state) - E(previous_state).
    log_uniform: Optional logarithm of a uniformly distributed random sample in
      [0, 1]. It is used to accept/reject the current and proposed state.
    seed: For reproducibility.

  Returns:
    new_state: The chosen state.
    mh_extra: MetropolisHastingsExtra.
  """
  current_state = util.map_tree(tf.convert_to_tensor, current_state)
  proposed_state = util.map_tree(tf.convert_to_tensor, proposed_state)
  energy_change = tf.convert_to_tensor(energy_change)

  log_accept_ratio = -energy_change

  if log_uniform is None:
    log_uniform = tf.math.log(
        util.random_uniform(
            shape=log_accept_ratio.shape,
            dtype=log_accept_ratio.dtype,
            seed=seed))
  is_accepted = log_uniform < log_accept_ratio

  next_state = choose(is_accepted, proposed_state, current_state)
  return next_state, MetropolisHastingsExtra(
      is_accepted=is_accepted, log_uniform=log_uniform)


class PersistentMetropolistHastingsState(NamedTuple):
  """Persistent Metropolis Hastings state.

  Attributes:
    level: Value uniformly distributed on [-1, 1], absolute value of which is
      used as the slice variable for the acceptance test.
  """
  # We borrow the [-1, 1] encoding from the original paper; it has the effect of
  # flipping the drift direction automatically, which has the effect of
  # prolonging the persistent bouts of acceptance.
  level: 'FloatTensor'


class PersistentMetropolistHastingsExtra(NamedTuple):
  """Persistent Metropolis Hastings extra outputs.

  Attributes:
    is_accepted: Whether the proposed state was accepted.
    accepted_state: The accepted state.
  """
  is_accepted: 'BooleanTensor'
  accepted_state: 'State'


@util.named_call
def persistent_metropolis_hastings_init(
    shape: 'IntTensor',
    dtype: 'tf.DType' = tf.float32,
    init_level: 'FloatTensor' = 0.,
) -> 'PersistentMetropolistHastingsState':
  """Initializes `PersistentMetropolistHastingsState`.

  Args:
    shape: Shape of the independent levels.
    dtype: Dtype for the levels.
    init_level: Initial value for the level. Broadcastable to `shape`.

  Returns:
    pmh_state: `PersistentMetropolistHastingsState`
  """
  return PersistentMetropolistHastingsState(level=init_level +
                                            tf.zeros(shape, dtype))


@util.named_call
def persistent_metropolis_hastings_step(
    pmh_state: 'PersistentMetropolistHastingsState',
    current_state: 'State',
    proposed_state: 'State',
    energy_change: 'FloatTensor',
    drift: 'FloatTensor',
) -> ('Tuple[PersistentMetropolistHastingsState, '
      'PersistentMetropolistHastingsExtra]'):
  """Persistent metropolis hastings step.

  This implements the algorithm from [1]. The net effect of this algorithm is
  that accepts/rejects are clustered in time, which helps algorithms that rely
  on persistent momenta. The overall acceptance rate is unaffected. This
  algorithm assumes that the `energy_change` has a continuous distribution
  symmetric about 0 to maintain ergodicity.

  Args:
    pmh_state: `PersistentMetropolistHastingsState`
    current_state: Current state.
    proposed_state: Proposed state.
    energy_change: E(proposed_state) - E(previous_state).
    drift: How much to shift the level variable at each step.

  Returns:
    pmh_state: New `PersistentMetropolistHastingsState`.
    pmh_extra: `PersistentMetropolistHastingsExtra`.

  #### References

  [1]: Neal, R. M. (2020). Non-reversibly updating a uniform [0,1] value for
       Metropolis accept/reject decisions.
  """
  log_accept_ratio = -energy_change
  is_accepted = tf.math.log(tf.abs(pmh_state.level)) < log_accept_ratio
  # N.B. we'll never accept when energy_change is NaN, so `level` should remain
  # non-NaN at all times.
  level = pmh_state.level
  level = tf.where(is_accepted, level * tf.exp(energy_change), level)
  level += drift
  level = (1 + level) % 2 - 1
  return pmh_state._replace(level=level), PersistentMetropolistHastingsExtra(
      is_accepted=is_accepted,
      accepted_state=choose(is_accepted, proposed_state, current_state),
  )


MomentumSampleFn = Callable[[Any], State]


@util.named_call
def gaussian_momentum_sample(state: 'Optional[State]' = None,
                             shape: 'Optional[IntTensor]' = None,
                             dtype: 'Optional[DTypeNest]' = None,
                             named_axis: 'Optional[StringNest]' = None,
                             seed=None) -> 'State':
  """Generates a sample from a Gaussian (Normal) momentum distribution.

  One of `state` or the pair of `shape`/`dtype` need to be specified to obtain
  the correct
  structure.

  Args:
    state: A nest of `Tensor`s with the shape and dtype being the same as the
      output.
    shape: A nest of shapes, which matches the output shapes.
    dtype: A nest of dtypes, which matches the output dtypes.
    named_axis: Named axes of the state, same structure as `state`.
    seed: For reproducibility.

  Returns:
    sample: A nest of `Tensor`s with the same structure, shape and dtypes as one
      of the two sets inputs, distributed with Normal distribution.
  """
  if dtype is None or shape is None:
    shape = util.map_tree(lambda t: t.shape, state)
    dtype = util.map_tree(lambda t: t.dtype, state)
  if named_axis is None:
    named_axis = util.map_tree(lambda _: [], dtype)

  num_seeds_needed = len(util.flatten_tree(dtype))
  seeds = list(util.split_seed(seed, num_seeds_needed))
  seeds = util.unflatten_tree(dtype, seeds)

  def _one_part(dtype, shape, seed, named_axis):
    seed = backend.distribute_lib.fold_in_axis_index(seed, named_axis)
    return util.random_normal(shape=shape, dtype=dtype, seed=seed)

  return util.map_tree_up_to(dtype, _one_part, dtype, shape, seeds, named_axis)


def make_gaussian_kinetic_energy_fn(
    chain_ndims: 'IntTensor',
    named_axis: 'Optional[StringNest]' = None,
) -> 'Callable[..., Tuple[tf.Tensor, TensorNest]]':
  """Returns a function that computes the kinetic energy of a state.

  Args:
    chain_ndims: How many leading dimensions correspond to independent
      particles.
    named_axis: Named axes of the state, same structure as `state`.

  Returns:
    kinetic_energy_fn: A callable that takes in the expanded state (see
      `call_potential_fn`) and returns the kinetic energy + dummy auxiliary
      output.
  """

  @util.named_call
  def kinetic_energy_fn(*args, **kwargs):
    state_args = (args, kwargs)

    if named_axis is None:
      named_axis_args = util.map_tree(lambda _: [], state_args)
    else:
      # We need named_axis to line up with state, but state has been decomposed
      # into args, kwargs via call_fn which called this function. Normally, we'd
      # reconstruct the state via recover_state_from_args, but we don't have a
      # good reference structure (named_axis is no good as it can have tuples as
      # leaves). Instead, we go the other way, and decompose named_axis into
      # args, kwargs. These new objects are guaranteed to line up with the
      # decomposed state.
      named_axis_args = call_fn(lambda *args, **kwargs: (args, kwargs),
                                named_axis)

    def _one_part(x, named_axis):
      return backend.distribute_lib.reduce_sum(
          tf.square(x), tuple(range(chain_ndims, len(x.shape))), named_axis)

    return 0.5 * sum(
        util.flatten_tree(
            util.map_tree_up_to(state_args, _one_part, state_args,
                                named_axis_args))), ()

  return kinetic_energy_fn


class HamiltonianMonteCarloState(NamedTuple):
  """Hamiltonian Monte Carlo state."""
  state: 'State'
  state_grads: 'State'
  target_log_prob: 'FloatTensor'
  state_extra: 'Any'


class HamiltonianMonteCarloExtra(NamedTuple):
  """Hamiltonian Monte Carlo extra outputs."""
  is_accepted: 'BooleanTensor'
  log_accept_ratio: 'FloatTensor'
  proposed_hmc_state: 'State'
  integrator_state: 'IntegratorState'
  integrator_extra: 'IntegratorExtras'
  initial_momentum: 'State'
  energy_change_extra: 'Any'


@util.named_call
def hamiltonian_monte_carlo_init(
    state: 'TensorNest',
    target_log_prob_fn: 'PotentialFn') -> 'HamiltonianMonteCarloState':
  """Initializes the `HamiltonianMonteCarloState`.

  Args:
    state: State of the chain.
    target_log_prob_fn: Target log prob fn.

  Returns:
    hmc_state: State of the `hamiltonian_monte_carlo_step` `TransitionOperator`.
  """
  state = util.map_tree(tf.convert_to_tensor, state)
  target_log_prob, state_extra, state_grads = util.map_tree(
      tf.convert_to_tensor,
      call_potential_fn_with_grads(target_log_prob_fn, state),
  )
  return HamiltonianMonteCarloState(state, state_grads, target_log_prob,
                                    state_extra)


@util.named_call
def _default_hamiltonian_monte_carlo_energy_change_fn(
    current_integrator_state: 'IntegratorState',
    proposed_integrator_state: 'IntegratorState',
    integrator_extra: 'IntegratorExtras',
) -> 'Tuple[FloatTensor, Any]':
  """Default HMC energy change function."""
  del current_integrator_state
  del proposed_integrator_state
  return integrator_extra.energy_change, ()


@util.named_call
def hamiltonian_monte_carlo_step(
    hmc_state: 'HamiltonianMonteCarloState',
    target_log_prob_fn: 'PotentialFn',
    step_size: 'Optional[Any]' = None,
    num_integrator_steps: 'Optional[IntTensor]' = None,
    momentum: 'Optional[State]' = None,
    kinetic_energy_fn: 'Optional[PotentialFn]' = None,
    momentum_sample_fn: 'Optional[MomentumSampleFn]' = None,
    integrator_trace_fn: 'Callable[[IntegratorStepState, IntegratorStepExtras],'
    'TensorNest]' = lambda *args: (),
    log_uniform: 'Optional[FloatTensor]' = None,
    integrator_fn=None,
    unroll_integrator: 'bool' = False,
    max_num_integrator_steps: 'Optional[IntTensor]' = None,
    energy_change_fn:
    'Callable[[IntegratorState, IntegratorState, IntegratorExtras], '
    'Tuple[FloatTensor, Any]]' = _default_hamiltonian_monte_carlo_energy_change_fn,
    named_axis: 'Optional[StringNest]' = None,
    seed=None,
) -> 'Tuple[HamiltonianMonteCarloState, HamiltonianMonteCarloExtra]':
  """Hamiltonian Monte Carlo `TransitionOperator`.

  #### Example

  ```python
  step_size = 0.2
  num_steps = 2000
  num_integrator_steps = 10
  state = tf.ones([16, 2])

  base_mean = [1., 0]
  base_cov = [[1, 0.5], [0.5, 1]]

  bijector = tfb.Softplus()
  base_dist = tfd.MultivariateNormalFullCovariance(
      loc=base_mean, covariance_matrix=base_cov)
  target_dist = bijector(base_dist)

  def orig_target_log_prob_fn(x):
    return target_dist.log_prob(x), ()

  target_log_prob_fn, state = fun_mc.transform_log_prob_fn(
      orig_target_log_prob_fn, bijector, state)

  kernel = tf.function(lambda state: fun_mc.hamiltonian_monte_carlo_step(
      state,
      step_size=step_size,
      num_integrator_steps=num_integrator_steps,
      target_log_prob_fn=target_log_prob_fn,
      seed=tfp_test_util.test_seed()))

  _, chain = fun_mc.trace(
      state=fun_mc.hamiltonian_monte_carlo_init(state, target_log_prob_fn),
      fn=kernel,
      num_steps=num_steps,
      trace_fn=lambda state, extra: state.state_extra[0])
  ```

  Args:
    hmc_state: HamiltonianMonteCarloState.
    target_log_prob_fn: Target log prob fn.
    step_size: Step size, structure broadcastable to the `target_log_prob_fn`
      state. Optional if `integrator_fn` is specified.
    num_integrator_steps: Number of integrator steps to take. Optional if
      `integrator_fn` is specified.
    momentum: Initial momentum, by default sampled from a standard gaussian.
    kinetic_energy_fn: Kinetic energy function.
    momentum_sample_fn: Sampler for the momentum.
    integrator_trace_fn: Trace function for the integrator.
    log_uniform: Optional logarithm of a uniformly distributed random sample in
      [0, 1], used for the MH accept/reject step.
    integrator_fn: Integrator to use for the HMC dynamics. Uses a
      `hamiltonian_integrator` with `leapfrog_step` by default.
    unroll_integrator: Whether to unroll the loop in the integrator. Only works
      if `num_integrator_steps`/`max_num_integrator_steps' is statically known.
      Ignored if `integrator_fn` is specified.
    max_num_integrator_steps: Maximum number of integrator steps to take. Useful
      when `num_integrator_steps` is dynamic, and yet you still want
      gradients/tracing to work. Ignored if `integrator_fn` is specified.
    energy_change_fn: Callable with signature: `(current_integrator_state,
      proposed_integrator_state,) -> (energy_change, energy_change_extra)`.
      Computes the change in energy between current and proposed states. By
      default, it just substracts the current and proposed energies. A typical
      reason to override this is to improve numerical stability.
    named_axis: Named axes of the state, same structure as `hmc_state.state`.
    seed: For reproducibility.

  Returns:
    hmc_state: HamiltonianMonteCarloState
    hmc_extra: HamiltonianMonteCarloExtra
  """
  state = hmc_state.state
  state_grads = hmc_state.state_grads
  target_log_prob = hmc_state.target_log_prob
  state_extra = hmc_state.state_extra

  if kinetic_energy_fn is None:
    kinetic_energy_fn = make_gaussian_kinetic_energy_fn(
        len(target_log_prob.shape) if target_log_prob.shape is not None else tf  # pytype: disable=attribute-error
        .rank(target_log_prob),
        named_axis=named_axis)

  if momentum_sample_fn is None:
    momentum_sample_fn = lambda seed: gaussian_momentum_sample(  # pylint: disable=g-long-lambda
        state=state,
        seed=seed,
        named_axis=named_axis)

  if integrator_fn is None:
    integrator_fn = lambda state: hamiltonian_integrator(  # pylint: disable=g-long-lambda
        state,
        num_steps=num_integrator_steps,
        integrator_step_fn=lambda state: leapfrog_step(  # pylint: disable=g-long-lambda
            state,
            step_size=step_size,
            target_log_prob_fn=target_log_prob_fn,
            kinetic_energy_fn=kinetic_energy_fn),
        kinetic_energy_fn=kinetic_energy_fn,
        unroll=unroll_integrator,
        max_num_steps=max_num_integrator_steps,
        integrator_trace_fn=integrator_trace_fn)

  if momentum is None:
    seed, sample_seed = util.split_seed(seed, 2)
    momentum = momentum_sample_fn(sample_seed)

  initial_integrator_state = IntegratorState(
      target_log_prob=target_log_prob,
      momentum=momentum,
      state=state,
      state_grads=state_grads,
      state_extra=state_extra,
  )

  integrator_state, integrator_extra = integrator_fn(initial_integrator_state)

  proposed_state = HamiltonianMonteCarloState(
      state=integrator_state.state,
      state_grads=integrator_state.state_grads,
      target_log_prob=integrator_state.target_log_prob,
      state_extra=integrator_state.state_extra)

  energy_change, energy_change_extra = energy_change_fn(
      initial_integrator_state,
      integrator_state,
      integrator_extra,
  )

  hmc_state, mh_extra = metropolis_hastings_step(
      hmc_state,
      proposed_state,
      energy_change,
      log_uniform=log_uniform,
      seed=seed)

  hmc_state = typing.cast(HamiltonianMonteCarloState, hmc_state)
  return hmc_state, HamiltonianMonteCarloExtra(
      is_accepted=mh_extra.is_accepted,
      proposed_hmc_state=proposed_state,
      log_accept_ratio=-energy_change,
      integrator_state=integrator_state,
      integrator_extra=integrator_extra,
      energy_change_extra=energy_change_extra,
      initial_momentum=momentum)


class IntegratorState(NamedTuple):
  """Integrator state."""
  state: 'State'
  state_extra: Any
  state_grads: 'State'
  target_log_prob: 'FloatTensor'
  momentum: 'State'


class IntegratorExtras(NamedTuple):
  """Integrator extra outputs."""
  initial_energy: 'FloatTensor'
  initial_kinetic_energy: 'FloatTensor'
  initial_kinetic_energy_extra: Any
  final_energy: 'FloatTensor'
  final_kinetic_energy: 'FloatTensor'
  final_kinetic_energy_extra: Any
  energy_change: 'FloatTensor'
  integrator_trace: Any
  momentum_grads: 'State'


@util.named_call
def hamiltonian_integrator(
    int_state: 'IntegratorState',
    num_steps: 'IntTensor',
    integrator_step_fn: 'IntegratorStep',
    kinetic_energy_fn: 'PotentialFn',
    integrator_trace_fn: 'Callable[[IntegratorStepState, IntegratorStepExtras],'
    'TensorNest]' = lambda *args: (),
    unroll: 'bool' = False,
    max_num_steps: 'Optional[IntTensor]' = None,
) -> 'Tuple[IntegratorState, IntegratorExtras]':
  """Intergrates a discretized set of Hamiltonian equations.

  This function will use the passed `integrator_step_fn` to evolve the system
  for `num_steps`. The `integrator_step_fn` is assumed to be reversible.

  Args:
    int_state: Current `IntegratorState`.
    num_steps: Integer scalar or N-D `Tensor`. Number of steps to take. If this
      is not a scalar, then each corresponding independent system will be
      evaluated for that number of steps, followed by copying the final state to
      avoid creating a ragged Tensor. Keep this in mind when interpreting the
      `integrator_trace` in the auxiliary output.
    integrator_step_fn: Instance of `IntegratorStep`.
    kinetic_energy_fn: Function to compute the kinetic energy from momentums.
    integrator_trace_fn: Trace function for the integrator.
    unroll: Whether to unroll the loop in the integrator. Only works if
      `num_steps`/`max_num_steps' is statically known.
    max_num_steps: Maximum number of integrator steps to take. Useful when
      `num_steps` is dynamic, and yet you still want gradients/tracing to work.

  Returns:
    integrator_state: `IntegratorState`
    integrator_exras: `IntegratorExtras`
  """
  target_log_prob = int_state.target_log_prob
  momentum = int_state.momentum
  state = int_state.state
  state_grads = int_state.state_grads
  state_extra = int_state.state_extra

  num_steps = ps.convert_to_shape_tensor(num_steps)
  is_ragged = len(num_steps.shape) > 0 or max_num_steps is not None  # pylint: disable=g-explicit-length-test

  initial_kinetic_energy, initial_kinetic_energy_extra = call_potential_fn(
      kinetic_energy_fn, momentum)
  initial_energy = -target_log_prob + initial_kinetic_energy

  if is_ragged:
    step = 0
    # TODO(siege): Use cond for the common padding, ala the old RNN code.
    if max_num_steps is None:
      max_num_steps = ps.reduce_max(num_steps)
  else:
    step = []
    max_num_steps = num_steps

  # We need to carry around the integrator state extras so we can properly do
  # the ragged computation.
  # TODO(siege): In principle we can condition this on `is_ragged`, but doesn't
  # seem worthwhile at the time.
  integrator_wrapper_state = (
      step,
      IntegratorStepState(state, state_grads, momentum),
      IntegratorStepExtras(target_log_prob, state_extra, initial_kinetic_energy,
                           initial_kinetic_energy_extra,
                           util.map_tree(tf.zeros_like, momentum)),
  )

  def integrator_wrapper(step, integrator_step_state, integrator_step_extra):
    """Integrator wrapper that tracks extra state."""
    old_integrator_step_state = integrator_step_state
    old_integrator_step_extra = integrator_step_extra
    integrator_step_state, integrator_step_extra = integrator_step_fn(
        integrator_step_state)

    if is_ragged:
      integrator_step_state = choose(step < num_steps, integrator_step_state,
                                     old_integrator_step_state)
      integrator_step_extra = choose(step < num_steps, integrator_step_extra,
                                     old_integrator_step_extra)
      step = step + 1

    return (step, integrator_step_state, integrator_step_extra), []

  def integrator_trace_wrapper_fn(args, _):
    return integrator_trace_fn(args[1], args[2])

  [_, integrator_step_state, integrator_step_extra], integrator_trace = trace(
      integrator_wrapper_state,
      integrator_wrapper,
      max_num_steps,
      trace_fn=integrator_trace_wrapper_fn,
      unroll=unroll,
  )

  final_energy = (-integrator_step_extra.target_log_prob +
                  integrator_step_extra.kinetic_energy)

  state = IntegratorState(
      state=integrator_step_state.state,
      state_extra=integrator_step_extra.state_extra,
      state_grads=integrator_step_state.state_grads,
      target_log_prob=integrator_step_extra.target_log_prob,
      momentum=integrator_step_state.momentum)

  extra = IntegratorExtras(
      initial_energy=initial_energy,
      initial_kinetic_energy=integrator_step_extra.kinetic_energy,
      initial_kinetic_energy_extra=integrator_step_extra.kinetic_energy_extra,
      final_energy=final_energy,
      final_kinetic_energy=integrator_step_extra.kinetic_energy,
      final_kinetic_energy_extra=integrator_step_extra.kinetic_energy_extra,
      energy_change=final_energy - initial_energy,
      integrator_trace=integrator_trace,
      momentum_grads=integrator_step_extra.momentum_grads)

  return state, extra


@util.named_call
def obabo_langevin_integrator(
    int_state: 'IntegratorState',
    num_steps: 'IntTensor',
    integrator_step_fn: 'IntegratorStep',
    momentum_refresh_fn: 'Callable[[State, Any], State]',
    energy_change_fn: 'Callable[[IntegratorState, IntegratorState], '
                      'Tuple[FloatTensor, Any]]',
    integrator_trace_fn: 'Callable[[IntegratorState, IntegratorStepState, '
                         'IntegratorStepExtras], '
                         'TensorNest]' = lambda *args: (),
    unroll: 'bool' = False,
    seed: Any = None,
) -> 'Tuple[IntegratorState, IntegratorExtras]':
  """Integrates discretized Langevin dynamics.

  This implements a generalized version of the OBABO integrator from [1]. In
  this integrator, the Langevin dynamics are split into Hamiltonian dynamics and
  an thermostat moves. The former is solved typically solved using the leapfrog
  integrator (generalized here to the `integrator_step_fn`), corresponding to
  the BAB steps. The latter is exactly solved by the `momentum_refresh_fn`, and
  corresponds to the two O steps.

  In the typical configuration where a leapfrog integrator is used for the
  Hamiltonian part, the momentum refreshment steps should implement half-steps.
  The integration error is assumed to only arise from `integrator_step_fn`.

  Args:
    int_state: Current `IntegratorState`.
    num_steps: Integer scalar. Number of steps to take.
    integrator_step_fn: Instance of `IntegratorStep`.
    momentum_refresh_fn: Momentum refreshment step. This must preserve the
      momentum density implied by the `integrator_step_fn`.
    energy_change_fn: Computes the change in energy between two integrator
      states.
    integrator_trace_fn: Trace function for the integrator.
    unroll: Whether to unroll the loop in the integrator. Only works if
      `num_steps` is statically known.
    seed: For reproducibility.

  Returns:
    integrator_state: `IntegratorState`
    integrator_exras: `IntegratorExtras`. Due to the structure of this
      integrator, the initial/final energy terms in `IntegratorExtras` return
      value are set to empty tuples.

  #### References

  [1]: Bussi, G., & Parrinello, M. (2008). Accurate sampling using Langevin
       dynamics. http://arxiv.org/abs/0803.4083
  """

  def step_fn(int_state, energy_change, seed):
    seed, refresh_seed_1, refresh_seed_2 = util.split_seed(seed, 3)

    # Refresh.
    momentum = momentum_refresh_fn(int_state.momentum, refresh_seed_1)
    int_state = int_state._replace(momentum=momentum)

    integrator_step_state = IntegratorStepState(
        state=int_state.state,
        state_grads=int_state.state_grads,
        momentum=int_state.momentum)

    # Integrate.
    integrator_step_state, integrator_step_extra = integrator_step_fn(
        integrator_step_state)

    new_int_state = int_state._replace(
        state=integrator_step_state.state,
        state_extra=integrator_step_extra.state_extra,
        state_grads=integrator_step_state.state_grads,
        momentum=integrator_step_state.momentum,
        target_log_prob=integrator_step_extra.target_log_prob,
    )

    # TODO(siege): Somehow return the energy change extra.
    new_energy_change, _ = energy_change_fn(int_state, new_int_state)
    energy_change += new_energy_change

    # Refresh.
    momentum = momentum_refresh_fn(new_int_state.momentum, refresh_seed_2)
    new_int_state = new_int_state._replace(momentum=momentum)

    if integrator_trace_fn is None:
      integrator_trace = ()
    else:
      integrator_trace = integrator_trace_fn(new_int_state,
                                             integrator_step_state,
                                             integrator_step_extra)
    return (new_int_state, energy_change, seed), (integrator_trace,
                                                  integrator_step_extra)

  (int_state, energy_change,
   _), (integrator_trace, integrator_step_extra) = trace(
       (int_state, tf.zeros_like(int_state.target_log_prob), seed),
       step_fn,
       num_steps,
       unroll=unroll,
       trace_mask=(True, False),
   )

  extra = IntegratorExtras(
      initial_energy=(),
      initial_kinetic_energy=(),
      initial_kinetic_energy_extra=(),
      final_energy=(),
      final_kinetic_energy=(),
      final_kinetic_energy_extra=(),
      energy_change=energy_change,
      integrator_trace=integrator_trace,
      momentum_grads=integrator_step_extra.momentum_grads)

  return int_state, extra


@util.named_call
def sign_adaptation(control: 'FloatNest',
                    output: 'FloatTensor',
                    set_point: 'FloatTensor',
                    adaptation_rate: 'FloatTensor' = 0.01) -> 'FloatNest':
  """A function to do simple sign-based control of a variable.

  ```
  control = control * (1. + adaptation_rate) ** sign(output - set_point)
  ```

  Args:
    control: The control variable.
    output: The output variable.
    set_point: The set point for `output`. This function will adjust `control`
      so that `output` matches `set_point`.
    adaptation_rate: Adaptation rate.

  Returns:
    control: New control.
  """

  def _get_new_control(control, output, set_point):
    new_control = choose(output > set_point, control * (1. + adaptation_rate),
                         control / (1. + adaptation_rate))
    return new_control

  output = maybe_broadcast_structure(output, control)
  set_point = maybe_broadcast_structure(set_point, control)

  return util.map_tree(_get_new_control, control, output, set_point)


@util.named_call
def choose(condition, x, y):
  """A nest-aware, left-broadcasting `tf.where`.

  Args:
    condition: Boolean nest. Must left-broadcast with `x` and `y`.
    x: Input one.
    y: Input two.

  Returns:
    z: Elements of `x` where 'condition' is `True`, and `y` otherwise.
  """

  def _choose_base_case(condition, x, y):
    """Choose base case for one tensor."""

    def _expand_condition_like(x):
      """Helper to expand `condition` like the shape of some input arg."""
      expand_shape = list(condition.shape) + [1] * (
          len(x.shape) - len(condition.shape))
      return tf.reshape(condition, expand_shape)

    if x is y:
      return x
    x = tf.convert_to_tensor(x)
    y = tf.convert_to_tensor(y)
    return tf.where(_expand_condition_like(x), x, y)

  condition = tf.convert_to_tensor(condition)
  return util.map_tree(lambda a, r: _choose_base_case(condition, a, r), x, y)


class AdamState(NamedTuple):
  """Adam state."""
  state: 'State'
  m: 'State'
  v: 'State'
  t: 'IntTensor'


class AdamExtra(NamedTuple):
  """Adam extra outputs."""
  loss: 'FloatTensor'
  loss_extra: Any
  grads: 'State'


@util.named_call
def adam_init(state: 'FloatNest') -> 'AdamState':
  """Initializes `AdamState`."""
  state = util.map_tree(tf.convert_to_tensor, state)
  return AdamState(
      state=state,
      m=util.map_tree(tf.zeros_like, state),
      v=util.map_tree(tf.zeros_like, state),
      t=tf.constant(0, dtype=tf.int32))


@util.named_call
def adam_step(adam_state: 'AdamState',
              loss_fn: 'PotentialFn',
              learning_rate: 'FloatNest',
              beta_1: 'FloatNest' = 0.9,
              beta_2: 'FloatNest' = 0.999,
              epsilon: 'FloatNest' = 1e-8) -> 'Tuple[AdamState, AdamExtra]':
  """Performs one step of the Adam optimization method.

  Args:
    adam_state: Current `AdamState`.
    loss_fn: A function whose output will be minimized.
    learning_rate: Learning rate, broadcastable with the state.
    beta_1: Adaptation rate for the first order gradient statistics,
      broadcastable with the state.
    beta_2: Adaptation rate for the second order gradient statistics,
      broadcastable with the state.
    epsilon: Epsilon to stabilize the algorithm, broadcastable with the state.
      Note that the `epsilon` is actually the `epsilon_hat` from introduction to
      Section 2 in [1].

  Returns:
    adam_state: `AdamState`
    adam_extra: `AdamExtra`


  #### References:

  [1]: Kingma, D. P., & Ba, J. L. (2015). Adam: a Method for Stochastic
       Optimization. International Conference on Learning Representations
       2015, 1-15.
  """
  state = adam_state.state
  m = adam_state.m
  v = adam_state.v
  learning_rate = maybe_broadcast_structure(learning_rate, state)
  beta_1 = maybe_broadcast_structure(beta_1, state)
  beta_2 = maybe_broadcast_structure(beta_2, state)
  epsilon = maybe_broadcast_structure(epsilon, state)
  t = adam_state.t + 1

  def _one_part(state, g, m, v, learning_rate, beta_1, beta_2, epsilon):
    """Updates one part of the state."""
    t_f = tf.cast(t, state.dtype)
    beta_1 = tf.convert_to_tensor(beta_1, state.dtype)
    beta_2 = tf.convert_to_tensor(beta_2, state.dtype)
    learning_rate = learning_rate * (
        tf.math.sqrt(1. - tf.math.pow(beta_2, t_f)) /
        (1. - tf.math.pow(beta_1, t_f)))

    m_t = beta_1 * m + (1. - beta_1) * g
    v_t = beta_2 * v + (1. - beta_2) * tf.square(g)
    state = state - learning_rate * m_t / (tf.math.sqrt(v_t) + epsilon)
    return state, m_t, v_t

  loss, loss_extra, grads = call_potential_fn_with_grads(loss_fn, state)

  state_m_v = util.map_tree(_one_part, state, grads, m, v, learning_rate,
                            beta_1, beta_2, epsilon)

  adam_state = AdamState(
      state=util.map_tree_up_to(state, lambda x: x[0], state_m_v),
      m=util.map_tree_up_to(state, lambda x: x[1], state_m_v),
      v=util.map_tree_up_to(state, lambda x: x[2], state_m_v),
      t=adam_state.t + 1)

  return adam_state, AdamExtra(loss_extra=loss_extra, loss=loss, grads=grads)


class GradientDescentState(NamedTuple):
  """Gradient Descent state."""
  state: 'State'


class GradientDescentExtra(NamedTuple):
  """Gradient Descent extra outputs."""
  loss: 'FloatTensor'
  loss_extra: Any
  grads: 'State'


@util.named_call
def gradient_descent_init(state: 'FloatNest') -> 'GradientDescentState':
  """Initializes `GradientDescentState`."""
  state = util.map_tree(tf.convert_to_tensor, state)
  return GradientDescentState(state=state)


@util.named_call
def gradient_descent_step(
    gd_state: 'GradientDescentState', loss_fn: 'PotentialFn',
    learning_rate: 'FloatNest'
) -> 'Tuple[GradientDescentState, GradientDescentExtra]':
  """Performs a step of regular gradient descent.

  Args:
    gd_state: Current `GradientDescentState`.
    loss_fn: A function whose output will be minimized.
    learning_rate: Learning rate, broadcastable with the state.

  Returns:
    gd_state: `GradientDescentState`
    gd_extra: `GradientDescentExtra`
  """

  state = gd_state.state
  learning_rate = maybe_broadcast_structure(learning_rate, state)

  def _one_part(state, g, learning_rate):
    return state - learning_rate * g

  loss, loss_extra, grads = call_potential_fn_with_grads(loss_fn, state)

  state = util.map_tree(_one_part, state, grads, learning_rate)

  gd_state = GradientDescentState(state=state)

  return gd_state, GradientDescentExtra(
      loss_extra=loss_extra, loss=loss, grads=grads)


@util.named_call
def gaussian_proposal(
    state: 'State',
    scale: 'FloatNest' = 1.,
    named_axis: 'Optional[StringNest]' = None,
    seed: 'Optional[Any]' = None) -> 'Tuple[State, Tuple[Tuple[()], float]]':
  """Axis-aligned gaussian random-walk proposal.

  Args:
    state: Current state.
    scale: Scale of the proposal.
    named_axis: Named axes of the state, same structure as `state`.
    seed: Random seed.

  Returns:
    new_state: Proposed state.
    extra: A 2-tuple of an empty tuple and the scalar 0.
  """
  scale = maybe_broadcast_structure(scale, state)
  num_parts = len(util.flatten_tree(state))
  seeds = util.unflatten_tree(state, util.split_seed(seed, num_parts))
  if named_axis is None:
    named_axis = util.map_tree(lambda _: [], state)

  def _sample_part(x, scale, seed, named_axis):
    seed = backend.distribute_lib.fold_in_axis_index(seed, named_axis)
    return x + scale * util.random_normal(  # pylint: disable=g-long-lambda
        x.shape, x.dtype, seed)

  new_state = util.map_tree_up_to(state, _sample_part, state, scale, seeds,
                                  named_axis)

  return new_state, ((), 0.)


class MaximalReflectiveCouplingProposalExtra(NamedTuple):
  """Extra results from the `maximal_reflection_coupling_proposal`."""
  log_couple_ratio: 'FloatTensor'
  coupling_proposed: 'BooleanTensor'


@util.named_call
def maximal_reflection_coupling_proposal(
    state: 'State',
    chain_ndims: 'int' = 0,
    scale: 'FloatNest' = 1,
    named_axis: 'Optional[StringNest]' = None,
    epsilon: 'FloatTensor' = 1e-20,
    seed: 'Optional[Any]' = None
) -> 'Tuple[State, Tuple[MaximalReflectiveCouplingProposalExtra, float]]':
  """Maximal reflection coupling proposal.

  Given two MCMC chains with the same, axis-aligned symmetrical Gaussian
  proposal, this function will produce a joint proposal with the maximum
  probability of the proposed state being the same for both chains. When that
  proposal is not generated, the two states are reflectionary symmetric across
  the hyperplane separating the two chain states.

  The chains are encoded into the `state` arg by concatenating along the first
  dimension such that `chain_i` is coupled with `chain_i + num_chains`, where
  `num_chains = state.shape[0] // 2`

  This function supports SPMD via sharded states in the same sense as TensorFlow
  Probability's `tfp.experimental.distribute.Sharded`.

  Args:
    state: Current state of the two sets of chains.
    chain_ndims: How many leading dimensions correspond to independent chains
      (not counting the first one).
    scale: Scale of the proposal.
    named_axis: Shard axes names, used for SPMD.
    epsilon: Small offset for numerical stability.
    seed: Random seed.

  Returns:
    new_state: Proposed state.
    extra: A 2-tuple of an empty tuple and the scalar 0.

  #### References:

  O'Leary, J. (2021). Couplings of the Random-Walk Metropolis algorithm. 1–28.
    Retrieved from http://arxiv.org/abs/2102.01790
  """

  _sum = backend.distribute_lib.reduce_sum  # pylint: disable=invalid-name

  def _struct_sum(s):
    return sum(util.flatten_tree(s))

  if named_axis is None:
    named_axis = util.map_tree(lambda _: [], state)
  scale = maybe_broadcast_structure(scale, state)
  num_chains = util.flatten_tree(state)[0].shape[0] // 2
  mu1 = util.map_tree(lambda x: x[:num_chains], state)
  mu2 = util.map_tree(lambda x: x[num_chains:], state)
  event_dims = util.map_tree(
      lambda x: tuple(range(1 + chain_ndims, len(x.shape))), mu1)
  z = util.map_tree(lambda s, x1, x2: (x1 - x2) / s, scale, mu1, mu2)
  z_norm = tf.sqrt(
      _struct_sum(
          util.map_tree_up_to(z, lambda z, ed, na: _sum(tf.square(z), ed, na),
                              z, event_dims, named_axis)))
  e = util.map_tree(
      lambda z: z /  # pylint: disable=g-long-lambda
      (tf.reshape(z_norm, z_norm.shape + (1,) *
                  (len(z.shape) - len(z_norm.shape))) + epsilon),
      z)
  batch_shape = util.flatten_tree(mu1)[0].shape[1:1 + chain_ndims]

  num_parts = len(util.flatten_tree(state))
  all_seeds = util.split_seed(seed, num_parts + 1)
  x_seeds = util.unflatten_tree(state, all_seeds[:num_parts])
  couple_seed = all_seeds[-1]

  def _sample_part(x, seed, named_axis):
    seed = backend.distribute_lib.fold_in_axis_index(seed, named_axis)
    return util.random_normal(x.shape, x.dtype, seed)

  x = util.map_tree_up_to(mu1, _sample_part, mu1, x_seeds, named_axis)

  e_dot_x = _struct_sum(
      util.map_tree_up_to(x, lambda x, e, ed, na: _sum(x * e, ed, na), x, e,
                          event_dims, named_axis))

  log_couple_ratio = _struct_sum(
      util.map_tree_up_to(
          x, lambda x, z, ed, na: -_sum(x * z + tf.square(z) / 2, ed, na), x, z,
          event_dims, named_axis))

  p_couple = tf.exp(tf.minimum(0., log_couple_ratio))
  coupling_proposed = util.random_uniform(
      batch_shape, dtype=p_couple.dtype, seed=couple_seed) < p_couple

  y_reflected = util.map_tree(
      lambda x, e: x - 2 * tf.reshape(  # pylint: disable=g-long-lambda
          e_dot_x, e_dot_x.shape + (1,) *
          (len(e.shape) - len(e_dot_x.shape))) * e,
      x,
      e)

  x2 = util.map_tree(lambda x, mu1, s: mu1 + s * x, x, mu1, scale)
  y2 = util.map_tree(lambda y, mu2, s: mu2 + s * y, y_reflected, mu2, scale)
  y2 = choose(coupling_proposed, x2, y2)

  new_state = util.map_tree(lambda x, y: tf.concat([x, y], axis=0), x2, y2)

  extra = MaximalReflectiveCouplingProposalExtra(
      log_couple_ratio=log_couple_ratio,
      coupling_proposed=coupling_proposed,
  )
  return new_state, (extra, 0.)


class RandomWalkMetropolisState(NamedTuple):
  """Random Walk Metropolis state."""
  state: 'State'
  target_log_prob: 'FloatTensor'
  state_extra: Any


class RandomWalkMetropolisExtra(NamedTuple):
  """Random Walk Metropolis extra outputs."""
  is_accepted: 'BooleanTensor'
  log_accept_ratio: 'FloatTensor'
  proposal_extra: Any
  proposed_rwm_state: 'RandomWalkMetropolisState'


@util.named_call
def random_walk_metropolis_init(
    state: 'State',
    target_log_prob_fn: 'PotentialFn') -> 'RandomWalkMetropolisState':
  """Initializes the `RandomWalkMetropolisState`.

  Args:
    state: State of the chain.
    target_log_prob_fn: Target log prob fn.

  Returns:
    hmc_state: State of the `random_walk_metropolis_init` `TransitionOperator`.
  """
  target_log_prob, state_extra = call_potential_fn(target_log_prob_fn, state)
  return RandomWalkMetropolisState(
      state=state,
      target_log_prob=target_log_prob,
      state_extra=state_extra,
  )


@util.named_call
def random_walk_metropolis_step(
    rwm_state: 'RandomWalkMetropolisState',
    target_log_prob_fn: 'PotentialFn',
    proposal_fn: 'TransitionOperator',
    log_uniform: 'Optional[FloatTensor]' = None,
    seed=None) -> 'Tuple[RandomWalkMetropolisState, RandomWalkMetropolisExtra]':
  """Random Walk Metropolis Hastings `TransitionOperator`.

  The `proposal_fn` takes in the current state, and must return a proposed
  state. It also must return a 2-tuple as its `extra` output, with the first
  element being arbitrary (returned in `rwm_extra`), and the second element
  being the log odds of going from the current state to the proposed state
  instead of reverse. If the proposal is symmetric about the current state, you
  can return `0.`.

  Args:
    rwm_state: RandomWalkMetropolisState.
    target_log_prob_fn: Target log prob fn.
    proposal_fn: Proposal fn.
    log_uniform: Optional logarithm of a uniformly distributed random sample in
      [0, 1], used for the MH accept/reject step.
    seed: For reproducibility.

  Returns:
    rwm_state: RandomWalkMetropolisState
    rwm_extra: RandomWalkMetropolisExtra
  """
  seed, sample_seed = util.split_seed(seed, 2)
  proposed_state, (proposal_extra,
                   log_proposed_bias) = proposal_fn(rwm_state.state,
                                                    sample_seed)

  proposed_target_log_prob, proposed_state_extra = call_potential_fn(
      target_log_prob_fn, proposed_state)

  # TODO(siege): Is it really a "log accept ratio" if we need to clamp it to 0?
  log_accept_ratio = (
      proposed_target_log_prob - rwm_state.target_log_prob - log_proposed_bias)

  proposed_rwm_state = RandomWalkMetropolisState(
      state=proposed_state,
      target_log_prob=proposed_target_log_prob,
      state_extra=proposed_state_extra,
  )

  rwm_state, mh_extra = metropolis_hastings_step(
      rwm_state,
      proposed_rwm_state,
      -log_accept_ratio,
      log_uniform=log_uniform,
      seed=seed,
  )

  rwm_extra = RandomWalkMetropolisExtra(
      proposal_extra=proposal_extra,
      proposed_rwm_state=proposed_rwm_state,
      log_accept_ratio=log_accept_ratio,
      is_accepted=mh_extra.is_accepted,
  )

  rwm_state = typing.cast(RandomWalkMetropolisState, rwm_state)
  return rwm_state, rwm_extra


class RunningVarianceState(NamedTuple):
  num_points: 'IntTensor'
  mean: 'FloatNest'
  variance: 'FloatNest'


@util.named_call
def running_variance_init(shape: 'IntTensor',
                          dtype: 'DTypeNest') -> 'RunningVarianceState':
  """Initializes the `RunningVarianceState`.

  Args:
    shape: Shape of the computed statistics.
    dtype: DType of the computed statistics.

  Returns:
    state: `RunningVarianceState`.
  """
  return RunningVarianceState(
      num_points=util.map_tree(lambda _: tf.zeros([], tf.int32), dtype),
      mean=util.map_tree_up_to(dtype, tf.zeros, shape, dtype),
      # The initial value of variance is discarded upon the first update, but
      # setting it to something reasonable (ones) is convenient in case the
      # state is read before an update.
      variance=util.map_tree_up_to(dtype, tf.ones, shape, dtype),
  )


@util.named_call
def running_variance_step(
    state: 'RunningVarianceState',
    vec: 'FloatNest',
    axis: 'Optional[Union[int, List[int], Tuple[int]]]' = None,
    window_size: 'Optional[IntNest]' = None,
) -> 'Tuple[RunningVarianceState, Tuple[()]]':
  """Updates the `RunningVarianceState`.

  As a computational convenience, this allows computing both independent
  variance estimates, as well as aggregating across an axis of `vec`. For
  example:

  - vec shape: [3, 4], axis=None -> mean/var shape: [3, 4]
  - vec shape: [3, 4], axis=0 -> mean/var shape: [4]
  - vec shape: [3, 4], axis=1 -> mean/var shape: [3]
  - vec shape: [3, 4], axis=[0, 1] -> mean/var shape: []

  Note that this produces a biased estimate of variance, for simplicity. If the
  unbiased estimate is required, compute it as follows: `state.variance *
  state.num_points / (state.num_points - 1)`.

  The algorithm is adapted from [1].

  Args:
    state: `RunningVarianceState`.
    vec: A Tensor to incorporate into the variance estimate.
    axis: If not `None`, treat these axes as being additional axes to aggregate
      over.
    window_size: A nest of ints, broadcastable with the structure of `vec`. If
      set, this will aggregate up to this many points. After the number of
      points is exceeded, old points are discarded as if doing exponential
      moving average with `alpha = window_size / (window_size + 1)` (this is
      only exactly true if `axis` is `None` however). The estimator retains the
      same bias guarantees if the distribution over the sequence of `vec` is
      stationary.

  Returns:
    state: `RunningVarianceState`.
    extra: Empty tuple.

  #### References

  [1]: <https://notmatthancock.github.io/2017/03/23/
       simple-batch-stat-updates.html>
  """

  def _one_part(vec, mean, variance, num_points):
    """Updates a single part."""
    vec = tf.convert_to_tensor(vec, mean.dtype)

    if axis is None:
      vec_mean = vec
      vec_variance = tf.zeros_like(variance)
    else:
      vec_mean = tf.reduce_mean(vec, axis)
      vec_variance = tf.math.reduce_variance(vec, axis)
    mean_diff = vec_mean - mean
    mean_diff_sq = tf.square(mean_diff)
    variance_diff = vec_variance - variance

    additional_points = tf.size(vec) // tf.size(mean)
    additional_points_f = tf.cast(additional_points, vec.dtype)
    num_points_f = tf.cast(num_points, vec.dtype)
    weight = additional_points_f / (num_points_f + additional_points_f)

    new_mean = mean + mean_diff * weight
    new_variance = (
        variance + variance_diff * weight + weight *
        (1. - weight) * mean_diff_sq)
    return new_mean, new_variance, num_points + additional_points

  new_mean_variance_num_points = util.map_tree(_one_part, vec, state.mean,
                                               state.variance, state.num_points)

  new_mean = util.map_tree_up_to(state.mean, lambda x: x[0],
                                 new_mean_variance_num_points)
  new_variance = util.map_tree_up_to(state.mean, lambda x: x[1],
                                     new_mean_variance_num_points)
  new_num_points = util.map_tree_up_to(state.mean, lambda x: x[2],
                                       new_mean_variance_num_points)
  if window_size is not None:
    window_size = maybe_broadcast_structure(window_size, new_num_points)
    new_num_points = util.map_tree(tf.minimum, new_num_points, window_size)
  return RunningVarianceState(
      num_points=new_num_points, mean=new_mean, variance=new_variance), ()


class RunningCovarianceState(NamedTuple):
  """Running Covariance state."""
  num_points: 'IntNest'
  mean: 'FloatNest'
  covariance: 'FloatNest'


@util.named_call
def running_covariance_init(shape: 'IntTensor',
                            dtype: 'DTypeNest') -> 'RunningCovarianceState':
  """Initializes the `RunningCovarianceState`.

  Args:
    shape: Shape of the computed mean.
    dtype: DType of the computed statistics.

  Returns:
    state: `RunningCovarianceState`.
  """
  return RunningCovarianceState(
      num_points=util.map_tree(lambda _: tf.zeros([], tf.int32), dtype),
      mean=util.map_tree_up_to(dtype, tf.zeros, shape, dtype),
      covariance=util.map_tree_up_to(
          # The initial value of covariance is discarded upon the first update,
          # but setting it to something reasonable (the identity matrix) is
          # convenient in case the state is read before an update.
          dtype,
          lambda shape, dtype: tf.eye(  # pylint: disable=g-long-lambda
              shape[-1],
              batch_shape=shape[:-1],
              dtype=dtype,
          ),
          shape,
          dtype),
  )


@util.named_call
def running_covariance_step(
    state: 'RunningCovarianceState',
    vec: 'FloatNest',
    axis: 'Optional[Union[int, List[int], Tuple[int]]]' = None,
    window_size: 'Optional[IntNest]' = None,
) -> 'Tuple[RunningCovarianceState, Tuple[()]]':
  """Updates the `RunningCovarianceState`.

  As a computational convenience, this allows computing both independent
  covariance estimates, as well as aggregating across an axis of `vec`. For
  example:

  - vec shape: [3, 4], axis=None -> mean shape: [3, 4], cov shape [3, 4, 4]
  - vec shape: [3, 4], axis=0 -> mean shape: [4], cov shape [4, 4]

  Note that the final unreduced dimension must be the last one (and there must
  be at least one unreduced dimension); thus, the following are illegal:

  - vec shape: [3, 4], axis=1 -> Illegal, unreduced dimension is not last.
  - vec shape: [3, 4], axis=[0, 1] -> Illegal, no unreduced dimensions.

  Note that this produces a biased estimate of covariance, for simplicity. If
  the unbiased estimate is required, compute it as follows: `state.covariance *
  state.num_points / (state.num_points - 1)`.

  The algorithm is adapted from [1].

  Args:
    state: `RunningCovarianceState`.
    vec: A Tensor to incorporate into the variance estimate.
    axis: If not `None`, treat these axes as being additional axes to aggregate
      over.
    window_size: A nest of ints, broadcastable with the structure of `vec`. If
      set, this will aggregate up to this many points. After the number of
      points is exceeded, old points are discarded as if doing exponential
      moving average with `alpha = window_size / (window_size + 1)` (this is
      only exactly true if `axis` is `None` however). The estimator retains the
      same bias guarantees if the distribution over the sequence of `vec` is
      stationary.

  Returns:
    state: `RunningCovarianceState`.
    extra: Empty tuple.

  #### References

  [1]: <https://notmatthancock.github.io/2017/03/23/
       simple-batch-stat-updates.html>
  """

  def _one_part(vec, mean, covariance, num_points):
    """Updates a single part."""
    vec = tf.convert_to_tensor(vec, mean.dtype)
    if axis is None:
      vec_mean = vec
      vec_covariance = tf.zeros_like(covariance)
    else:
      vec_mean = tf.reduce_mean(vec, axis)
      vec_covariance = tfp.stats.covariance(vec, sample_axis=axis)
    mean_diff = vec_mean - mean
    mean_diff_sq = (
        mean_diff[..., :, tf.newaxis] * mean_diff[..., tf.newaxis, :])
    covariance_diff = vec_covariance - covariance

    additional_points = tf.size(vec) // tf.size(mean)
    additional_points_f = tf.cast(additional_points, vec.dtype)
    num_points_f = tf.cast(num_points, vec.dtype)
    weight = additional_points_f / (num_points_f + additional_points_f)

    new_mean = mean + mean_diff * weight
    new_covariance = (
        covariance + covariance_diff * weight + weight *
        (1. - weight) * mean_diff_sq)
    return new_mean, new_covariance, num_points + additional_points

  new_mean_covariance_num_points = util.map_tree(_one_part, vec, state.mean,
                                                 state.covariance,
                                                 state.num_points)

  new_mean = util.map_tree_up_to(state.mean, lambda x: x[0],
                                 new_mean_covariance_num_points)
  new_covariance = util.map_tree_up_to(state.mean, lambda x: x[1],
                                       new_mean_covariance_num_points)
  new_num_points = util.map_tree_up_to(state.mean, lambda x: x[2],
                                       new_mean_covariance_num_points)
  if window_size is not None:
    window_size = maybe_broadcast_structure(window_size, new_num_points)
    new_num_points = util.map_tree(tf.minimum, new_num_points, window_size)
  return RunningCovarianceState(
      num_points=new_num_points, mean=new_mean, covariance=new_covariance), ()


class RunningMeanState(NamedTuple):
  """Running Mean state."""
  num_points: 'IntNest'
  mean: 'FloatTensor'


@util.named_call
def running_mean_init(shape: 'IntTensor',
                      dtype: 'DTypeNest') -> 'RunningMeanState':
  """Initializes the `RunningMeanState`.

  Args:
    shape: Shape of the computed statistics.
    dtype: DType of the computed statistics.

  Returns:
    state: `RunningMeanState`.
  """
  return RunningMeanState(
      num_points=util.map_tree(lambda _: tf.zeros([], tf.int32), dtype),
      mean=util.map_tree_up_to(dtype, tf.zeros, shape, dtype),
  )


@util.named_call
def running_mean_step(
    state: 'RunningMeanState',
    vec: 'FloatNest',
    axis: 'Optional[Union[int, List[int], Tuple[int]]]' = None,
    window_size: 'Optional[IntNest]' = None,
) -> 'Tuple[RunningMeanState, Tuple[()]]':
  """Updates the `RunningMeanState`.

  As a computational convenience, this allows computing both independent
  mean estimates, as well as aggregating across an axis of `vec`. For example:

  - vec shape: [3, 4], axis=None -> mean shape: [3, 4]
  - vec shape: [3, 4], axis=0 -> mean shape: [4]
  - vec shape: [3, 4], axis=1 -> mean shape: [3]
  - vec shape: [3, 4], axis=[0, 1] -> mean shape: []

  The algorithm is adapted from [1].

  Args:
    state: `RunningMeanState`.
    vec: A Tensor to incorporate into the mean.
    axis: If not `None`, treat these axes as being additional axes to aggregate
      over.
    window_size: A nest of ints, broadcastable with the structure of `vec`. If
      set, this will aggregate up to this many points. After the number of
      points is exceeded, old points are discarded as if doing exponential
      moving average with `alpha = window_size / (window_size + 1)` (this is
      only exactly true if `axis` is `None` however). The estimator retains the
      same bias guarantees if the distribution over the sequence of `vec` is
      stationary.

  Returns:
    state: `RunningMeanState`.
    extra: Empty tuple.

  #### References

  [1]: <https://notmatthancock.github.io/2017/03/23/
       simple-batch-stat-updates.html>
  """

  def _one_part(vec, mean, num_points):
    """Updates a single part."""
    vec = tf.convert_to_tensor(vec, mean.dtype)
    if axis is None:
      vec_mean = vec
    else:
      vec_mean = tf.reduce_mean(vec, axis)
    mean_diff = vec_mean - mean

    additional_points = tf.size(vec) // tf.size(mean)
    additional_points_f = tf.cast(additional_points, vec.dtype)
    num_points_f = tf.cast(num_points, vec.dtype)
    weight = additional_points_f / (num_points_f + additional_points_f)

    new_mean = mean + mean_diff * weight
    return new_mean, num_points + additional_points

  new_mean_num_points = util.map_tree(_one_part, vec, state.mean,
                                      state.num_points)

  new_mean = util.map_tree_up_to(state.mean, lambda x: x[0],
                                 new_mean_num_points)
  new_num_points = util.map_tree_up_to(state.mean, lambda x: x[1],
                                       new_mean_num_points)
  if window_size is not None:
    window_size = maybe_broadcast_structure(window_size, new_num_points)
    new_num_points = util.map_tree(tf.minimum, new_num_points, window_size)
  return RunningMeanState(num_points=new_num_points, mean=new_mean), ()


class PotentialScaleReductionState(RunningVarianceState):
  """Potential Scale Reduction state."""
  pass


@util.named_call
def potential_scale_reduction_init(shape,
                                   dtype) -> 'PotentialScaleReductionState':
  """Initializes `PotentialScaleReductionState`.

  Args:
    shape: Shape of the MCMC state.
    dtype: DType of the MCMC state.

  Returns:
    state: `PotentialScaleReductionState`.
  """
  # We are wrapping running variance so that the user doesn't get the chance to
  # set the reduction axis, which would break the assumptions of
  # `potential_scale_reduction_extract`.
  return PotentialScaleReductionState(*running_variance_init(shape, dtype))


@util.named_call
def potential_scale_reduction_step(
    state: 'PotentialScaleReductionState',
    sample) -> 'Tuple[PotentialScaleReductionState, Tuple[()]]':
  """Updates `PotentialScaleReductionState`.

  This computes the 'potential scale reduction' statistic from [1]. Note that
  this actually refers to the potential *variance* reduction, but the scale
  terminology has stuck. When this is close to 1, the chains are often
  considered converged.

  To extract the actual value of the statistic, use
  `potential_scale_reduction_extract`.

  Args:
    state: `PotentialScaleReductionState`
    sample: A sample from an MCMC chain. The leading dimension must have shape
      of at least 1.

  Returns:
    state: `PotentialScaleReductionState`.
    extra: Empty tuple.

  #### References

  [1]: Rooks, S. P. B., & Elman, A. G. (1998). General Methods for Monitoring
       Convergence of Iterative Simulations, 7(4), 434-455.
  """
  # We are wrapping running variance so that the user doesn't get the chance to
  # set the reduction axis, which would break the assumptions of
  # `potential_scale_reduction_extract`.
  return PotentialScaleReductionState(
      *running_variance_step(state, sample)[0]), ()


@util.named_call
def potential_scale_reduction_extract(
    state: 'PotentialScaleReductionState',
    independent_chain_ndims: 'IntNest' = 1) -> 'FloatNest':
  """Extracts the 'potential scale reduction' statistic.

  Args:
    state: `PotentialScaleReductionState`.
    independent_chain_ndims: Number of initial dimensions that are treated as
      indexing independent chains. Must be at least 1.

  Returns:
    rhat: Potential scale reduction.
  """
  independent_chain_ndims = maybe_broadcast_structure(independent_chain_ndims,
                                                      state.mean)

  def _psr_part(num_points, mean, variance, independent_chain_ndims):
    """Compute PSR for a single part."""
    # TODO(siege): Keeping these per-component points is mildly wasteful because
    # unlike general running variance estimation, these are always the same
    # across parts.
    num_points = tf.cast(num_points, mean.dtype)
    num_chains = tf.cast(
        np.prod(mean.shape[:independent_chain_ndims]), mean.dtype)

    independent_dims = list(range(independent_chain_ndims))
    # Within chain variance.
    var_w = num_points / (num_points - 1) * tf.reduce_mean(
        variance, independent_dims)
    # Between chain variance.
    var_b = num_chains / (num_chains - 1) * tf.math.reduce_variance(
        mean, independent_dims)
    # Estimate of the true variance of the target distribution.
    sigma2p = (num_points - 1) / num_points * var_w + var_b
    return ((num_chains + 1) / num_chains * sigma2p / var_w - (num_points - 1) /
            (num_chains * num_points))

  return util.map_tree(_psr_part, state.num_points, state.mean, state.variance,
                       independent_chain_ndims)


class RunningApproximateAutoCovarianceState(NamedTuple):
  """Running Approximate Auto-Covariance state."""
  buffer: 'FloatNest'
  num_steps: 'IntTensor'
  mean: 'FloatNest'
  auto_covariance: 'FloatNest'


@util.named_call
def running_approximate_auto_covariance_init(
    max_lags: 'int',
    state_shape: 'IntTensor',
    dtype: 'DTypeNest',
    axis: 'Optional[Union[int, List[int], Tuple[int]]]' = None,
) -> 'RunningApproximateAutoCovarianceState':
  """Initializes `RunningApproximateAutoCovarianceState`.

  Args:
    max_lags: Maximum lag for the computed auto-covariance.
    state_shape: Shape of the sequence elements that the auto-covariance is
      computed over. Note that this is before the averaging by the `axis`
      argument.
    dtype: DType of the state.
    axis: Axes to average over. See `running_approximate_auto_covariance_step`
      for details.

  Returns:
    state: `RunningApproximateAutoCovarianceState`.
  """
  if axis is None:
    mean_shape = state_shape
  else:
    # TODO(siege): Can this be done without doing the surrogate computation?
    mean_shape = util.map_tree_up_to(
        dtype, lambda s: tf.reduce_sum(tf.zeros(s), axis).shape, state_shape)

  def _shape_with_lags(shape):
    if isinstance(shape, (tuple, list)):
      return [max_lags + 1] + list(shape)
    else:
      return tf.concat([[max_lags + 1],
                        tf.convert_to_tensor(shape, tf.int32)],
                       axis=0)

  return RunningApproximateAutoCovarianceState(
      buffer=util.map_tree_up_to(
          dtype, lambda d, s: tf.zeros(_shape_with_lags(s), dtype=d), dtype,
          state_shape),
      num_steps=tf.zeros([], dtype=tf.int32),
      mean=util.map_tree_up_to(dtype, lambda d, s: tf.zeros(s, dtype=d), dtype,
                               mean_shape),
      auto_covariance=util.map_tree_up_to(
          dtype, lambda d, s: tf.zeros(_shape_with_lags(s), dtype=d), dtype,
          mean_shape),
  )


@util.named_call
def running_approximate_auto_covariance_step(
    state: 'RunningApproximateAutoCovarianceState',
    vec: 'TensorNest',
    axis: 'Optional[Union[int, List[int], Tuple[int]]]' = None,
) -> 'Tuple[RunningApproximateAutoCovarianceState, Tuple[()]]':
  """Updates `RunningApproximateAutoCovarianceState`.

  This computes a running auto-covariance of a sequence using a biased
  approximation. The algorithm effectively performs `max_lags + 1` separate
  covariance estimates, except with the running mean terms replaced by a shared
  mean computed at lag 0. This is not mathematically correct for lag > 0, but
  empirically the bias is manageable. The bias is large when the `max_lags` is
  large compared to the sequence length: a factor of about 3x is often adequate.

  This used a very naive algorithm based on keeping the last `max_lags + 1`
  elements of the sequence as part of the state. The time complexity is
  `O(max_lags * sequence_length)`, so this should only be used instead of the
  versions based on FFT when the memory requrements for materializing the whole
  sequence are excessive.

  For convenience, this function supports computing the average auto-correlation
  across dimensions of the elements by specifying the `axis` argument. This must
  either be `None` or refer to the leading dimensions of `vec`. For example:

  - vec shape: [3, 4], axis=None -> auto_covariance shape: [max_lags + 1, 3, 4]
  - vec shape: [3, 4], axis=0 -> auto_covariance shape: [max_lags + 1, 4]
  - vec shape: [3, 4], axis=[0, 1] -> auto_covariance shape: [max_lags + 1]

  Args:
    state: `RunningApproximateAutoCovarianceState`
    vec: An element of a sequence. This must have the same shape as was passed
      to `running_approximate_auto_covariance_init`.
    axis: If not `None`, treat these axes as being axes to average over.

  Returns:
    state: `RunningApproximateAutoCovarianceState`.
    extra: Empty tuple.
  """

  def _one_part(vec, buf, mean, auto_cov):
    """Compute the auto-covariance for one part."""
    buf_size = buf.shape[0]
    tail_idx = tf.range(0, buf_size - 1)
    num_steps = state.num_steps - tf.range(buf_size)
    num_steps = tf.maximum(0, num_steps)

    buf = tf.gather(buf, tail_idx)
    buf = tf.concat([vec[tf.newaxis], buf], 0)
    centered_buf = buf - mean
    centered_vec = vec - mean

    num_steps_0 = num_steps[0]
    # Need to broadcast on the right with autocov.
    steps_shape = ([-1] + [1] * (len(auto_cov.shape) - len(num_steps.shape)))
    num_steps = tf.reshape(num_steps, steps_shape)

    # TODO(siege): Simplify this to look like running_variance_step.
    # pyformat: disable
    if axis is None:
      additional_points = 1
      additional_points_f = 1
      # This assumes `additional_points` is the same for every step,
      # verified by the buf update logic above.
      num_points_f = additional_points_f * tf.cast(num_steps, mean.dtype)

      auto_cov = ((
          num_points_f * (num_points_f + additional_points_f) * auto_cov +
          num_points_f * centered_vec * centered_buf) /
                  tf.square(num_points_f + additional_points_f))
    else:
      vec_shape = tf.convert_to_tensor(vec.shape)
      additional_points = tf.math.reduce_prod(tf.gather(vec_shape, axis))
      additional_points_f = tf.cast(additional_points, vec.dtype)
      num_points_f = additional_points_f * tf.cast(num_steps, mean.dtype)
      buf_axis = util.map_tree(lambda a: a + 1, axis)

      auto_cov = (
          num_points_f * (num_points_f + additional_points_f) * auto_cov +
          num_points_f * tf.reduce_sum(centered_vec * centered_buf, buf_axis) -
          tf.reduce_sum(vec, axis) * tf.reduce_sum(buf, buf_axis) +
          additional_points_f * tf.reduce_sum(vec * buf, buf_axis)) / (
              tf.square(num_points_f + additional_points_f))
      centered_vec = tf.reduce_sum(centered_vec, axis)
    # pyformat: enable
    num_points_0_f = additional_points_f * tf.cast(num_steps_0, mean.dtype)
    mean = mean + centered_vec / (num_points_0_f + additional_points_f)
    return buf, auto_cov, mean

  new_buffer_auto_cov_mean = util.map_tree(_one_part, vec, state.buffer,
                                           state.mean, state.auto_covariance)

  new_buffer = util.map_tree_up_to(state.buffer, lambda x: x[0],
                                   new_buffer_auto_cov_mean)
  new_auto_cov = util.map_tree_up_to(state.buffer, lambda x: x[1],
                                     new_buffer_auto_cov_mean)
  new_mean = util.map_tree_up_to(state.buffer, lambda x: x[2],
                                 new_buffer_auto_cov_mean)

  state = RunningApproximateAutoCovarianceState(
      num_steps=state.num_steps + 1,
      buffer=new_buffer,
      auto_covariance=new_auto_cov,
      mean=new_mean,
  )
  return state, ()


def make_surrogate_loss_fn(
    grad_fn: 'Optional[GradFn]' = None,
    loss_value: 'tf.Tensor' = 0.,
) -> 'Any':
  """Creates a surrogate loss function with specified gradients.

  This wrapper converts `grad_fn` with signature `state -> grad, extra` which
  computes the gradients of its arguments with respect to a surrogate loss given
  the values of its arguments, namely:

  ```python
  loss_fn = make_surrogate_loss_fn(grad_fn, loss_value=loss_value)
  loss, extra, grad = call_potential_fn_with_grads(loss_fn, state)

  grad2, extra2 = grad(state)
  assert loss == loss_value
  assert extra == extra2
  assert grad == grad2
  ```

  Args:
    grad_fn: Wrapped gradient function. If `None`, then this function returns a
      decorator with the signature of `GradFn -> PotentialFn`.
    loss_value: A tensor that will be returned from the surrogate loss function.

  Returns:
    loss_fn: If grad_fn is not None. A `PotentialFn`, the surrogate loss
      function.
    make_surrogate_loss_fn: If grad_fn is None, this returns itself with
      `loss_value` bound to the value passed to this function.
  """
  if grad_fn is None:
    return functools.partial(make_surrogate_loss_fn, loss_value=loss_value)

  def loss_fn(*args, **kwargs):
    """The surrogate loss function."""

    @tf.custom_gradient
    def grad_wrapper(*flat_args_kwargs):
      new_args, new_kwargs = util.unflatten_tree((args, kwargs),
                                                 flat_args_kwargs)
      g, e = grad_fn(*new_args, **new_kwargs)  # pytype: disable=wrong-arg-count

      def inner_grad_fn(*_):
        return tuple(util.flatten_tree(g))

      return (loss_value, e), inner_grad_fn

    return grad_wrapper(*util.flatten_tree((args, kwargs)))

  return loss_fn


class SimpleDualAveragesState(NamedTuple):
  """Simple Dual Averages state."""
  state: 'State'
  step: 'IntTensor'
  grad_running_mean_state: 'RunningMeanState'


class SimpleDualAveragesExtra(NamedTuple):
  loss: 'FloatTensor'
  loss_extra: Any
  grads: 'State'


@util.named_call
def simple_dual_averages_init(
    state: 'FloatNest',
    grad_mean_smoothing_steps: 'IntNest' = 0,
) -> 'SimpleDualAveragesState':
  """Initializes Simple Dual Averages state.

  Note that the `state` argument only affects the initial value read from the
  state, it has no effect on any other step of the algorithm. Typically, you'd
  set this to the same value as `shrink_point`.

  Args:
    state: The state of the problem.
    grad_mean_smoothing_steps: Smoothes out the initial gradient running mean.
      For some algorithms it improves stability to make this non-zero.

  Returns:
    sda_state: `SimpleDualAveragesState`.
  """
  grad_rms = running_mean_init(
      util.map_tree(lambda s: s.shape, state),
      util.map_tree(lambda s: s.dtype, state))
  grad_rms = grad_rms._replace(
      num_points=util.map_tree(lambda _: grad_mean_smoothing_steps,
                               grad_rms.num_points))

  return SimpleDualAveragesState(
      state=state,
      # The algorithm assumes this starts at 1.
      step=1,
      grad_running_mean_state=grad_rms,
  )


@util.named_call
def simple_dual_averages_step(
    sda_state: 'SimpleDualAveragesState',
    loss_fn: 'PotentialFn',
    shrink_weight: 'FloatNest',
    shrink_point: 'State' = 0.,
) -> 'Tuple[SimpleDualAveragesState, SimpleDualAveragesExtra]':
  """Performs one step of the Simple Dual Averages algorithm [1].

  This function implements equation 3.4 from [1], with the following choices:

  ```none
  d(x) = 0.5 * (x - shrink_point)**2
  mu_k = shrink_weight / step**0.5
  ```

  Strictly speaking, this algorithm only applies to convex problems. The
  `loss_fn` need not have true gradients: sub-gradients are sufficient. The
  sequence of `state` is not actually convergent. To get a convergent sequence,
  you can compute a running mean of `state` (e.g. using `running_mean_step`),
  although that is not the sole choice.

  Args:
    sda_state: `SimpleDualAveragesState`.
    loss_fn: A function whose output will be minimized.
    shrink_weight: Weight of the shrinkage term. Must broadcast with `state`.
    shrink_point: Where the algorithm initially shrinks `state` to. Must
      broadcast with `state`.

  Returns:
    sda_state: `SimpleDualAveragesState`.
    sda_extra: `SimpleDualAveragesExtra`.

  #### References

  [1]: Nesterov, Y. (2009). Primal-dual subgradient methods for convex problems.
       Mathematical Programming, 120(1), 221-259.
  """
  state = sda_state.state
  step = sda_state.step
  shrink_point = maybe_broadcast_structure(shrink_point, state)
  shrink_weight = maybe_broadcast_structure(shrink_weight, state)

  loss, loss_extra, grads = call_potential_fn_with_grads(loss_fn, state)

  grad_rms, _ = running_mean_step(sda_state.grad_running_mean_state, grads)

  def _one_part(shrink_point, shrink_weight, grad_running_mean):
    shrink_point = tf.convert_to_tensor(shrink_point, grad_running_mean.dtype)
    step_f = tf.cast(step, grad_running_mean.dtype)
    return shrink_point - tf.sqrt(step_f) / shrink_weight * grad_running_mean

  state = util.map_tree(_one_part, shrink_point, shrink_weight, grad_rms.mean)

  sda_state = SimpleDualAveragesState(
      state=state,
      step=step + 1,
      grad_running_mean_state=grad_rms,
  )
  sda_extra = SimpleDualAveragesExtra(
      loss=loss,
      loss_extra=loss_extra,
      grads=grads,
  )

  return sda_state, sda_extra


def _global_norm(x: 'FloatNest') -> 'FloatTensor':
  return tf.sqrt(sum(tf.reduce_sum(tf.square(v)) for v in util.flatten_tree(x)))


def clip_grads(x: 'FloatNest',
               max_global_norm: 'FloatTensor',
               eps: 'FloatTensor' = 1e-9,
               zero_out_nan: 'bool' = True) -> 'FloatNest':
  """Clip gradients flowing through x.

  By default, non-finite gradients are zeroed out.

  Args:
    x: (Possibly nested) floating point `Tensor`.
    max_global_norm: Floating point `Tensor`. Maximum global norm of gradients
      flowing through `x`.
    eps: Floating point `Tensor`. Epsilon used when normalizing the gradient
      norm.
    zero_out_nan: Boolean. If `True` non-finite gradients are zeroed out.

  Returns:
    x: Same value as the input.
  """

  @tf.custom_gradient
  def grad_wrapper(*x):

    def grad_fn(*g):
      g = util.flatten_tree(g)
      if zero_out_nan:
        g = [tf.where(tf.math.is_finite(v), v, tf.zeros_like(v)) for v in g]
      norm = _global_norm(g) + eps

      def clip_part(v):
        return tf.where(norm < max_global_norm, v, v * max_global_norm / norm)

      res = tuple(clip_part(v) for v in g)
      if len(res) == 1:
        res = res[0]
      return res

    if len(x) == 1:
      x = x[0]
    return x, grad_fn

  return util.unflatten_tree(
      x, util.flatten_tree(grad_wrapper(*util.flatten_tree(x))))
