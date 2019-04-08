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
"""Internal utility functions for implementing TransitionKernels."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import warnings

# Dependency imports
import numpy as np
import tensorflow as tf

from tensorflow_probability.python.math.gradient import value_and_gradient as tfp_math_value_and_gradients

from tensorflow.python.ops import control_flow_util  # pylint: disable=g-direct-tensorflow-import
from tensorflow.python.util import deprecation  # pylint: disable=g-direct-tensorflow-import


__all__ = [
    'choose',
    'is_list_like',
    'is_namedtuple_like',
    'make_innermost_setter',
    'make_innermost_getter',
    'make_name',
    'maybe_call_fn_and_grads',
    'safe_sum',
    'set_doc',
    'smart_for_loop',
    'trace_scan',
    'enable_store_parameters_in_results',
    'warn_if_parameters_are_not_simple_tensors',
]


def is_list_like(x):
  """Helper which returns `True` if input is `list`-like."""
  return isinstance(x, (tuple, list))


def is_namedtuple_like(x):
  """Helper which returns `True` if input is `collections.namedtuple`-like."""
  try:
    for fn in x._fields:
      _ = getattr(x, fn)
    return True
  except AttributeError:
    return False


def make_name(super_name, default_super_name, sub_name):
  """Helper which makes a `str` name; useful for tf.compat.v1.name_scope."""
  name = super_name if super_name is not None else default_super_name
  if sub_name is not None:
    name += '_' + sub_name
  return name


def _choose_base_case(is_accepted,
                      accepted,
                      rejected,
                      name=None):
  """Helper to `choose` which expand_dims `is_accepted` and applies tf.where."""
  def _expand_is_accepted_like(x):
    """Helper to expand `is_accepted` like the shape of some input arg."""
    with tf.compat.v1.name_scope('expand_is_accepted_like'):
      expand_shape = tf.concat([
          tf.shape(input=is_accepted),
          tf.ones([tf.rank(x) - tf.rank(is_accepted)], dtype=tf.int32),
      ],
                               axis=0)
      multiples = tf.concat([
          tf.ones([tf.rank(is_accepted)], dtype=tf.int32),
          tf.shape(input=x)[tf.rank(is_accepted):],
      ],
                            axis=0)
      m = tf.tile(tf.reshape(is_accepted, expand_shape),
                  multiples)
      m.set_shape(m.shape.merge_with(x.shape))
      return m
  def _where(accepted, rejected):
    if accepted is rejected:
      return accepted
    accepted = tf.convert_to_tensor(value=accepted, name='accepted')
    rejected = tf.convert_to_tensor(value=rejected, name='rejected')
    r = tf.where(_expand_is_accepted_like(accepted), accepted, rejected)
    r.set_shape(r.shape.merge_with(accepted.shape.merge_with(rejected.shape)))
    return r

  with tf.compat.v1.name_scope(
      name, 'choose', values=[is_accepted, accepted, rejected]):
    if not is_list_like(accepted):
      return _where(accepted, rejected)
    return [(choose(is_accepted, a, r, name=name) if is_namedtuple_like(a)
             else _where(a, r))
            for a, r in zip(accepted, rejected)]


def choose(is_accepted, accepted, rejected, name=None):
  """Helper which expand_dims `is_accepted` then applies tf.where."""
  if not is_namedtuple_like(accepted):
    return _choose_base_case(is_accepted, accepted, rejected, name=name)
  if not isinstance(accepted, type(rejected)):
    raise TypeError('Type of `accepted` ({}) must be identical to '
                    'type of `rejected` ({})'.format(
                        type(accepted).__name__,
                        type(rejected).__name__))
  return type(accepted)(**dict(
      [(fn,
        choose(is_accepted,
               getattr(accepted, fn),
               getattr(rejected, fn),
               name=name))
       for fn in accepted._fields]))


def safe_sum(x, alt_value=-np.inf, name=None):
  """Elementwise adds list members, replacing non-finite results with alt_value.

  Typically the `alt_value` is chosen so the `MetropolisHastings`
  `TransitionKernel` always rejects the proposal.

  Args:
    x: Python `list` of `Tensors` to elementwise add.
    alt_value: Python scalar used to replace any elementwise sums which would
      otherwise be non-finite.
    name: Python `str` name prefixed to Ops created by this function.
      Default value: `None` (i.e., "safe_sum").

  Returns:
    safe_sum: `Tensor` representing the elementwise sum of list of `Tensor`s
      `x` or `alt_value` where sums are non-finite.

  Raises:
    TypeError: if `x` is not list-like.
    ValueError: if `x` is empty.
  """
  with tf.compat.v1.name_scope(name, 'safe_sum', [x, alt_value]):
    if not is_list_like(x):
      raise TypeError('Expected list input.')
    if not x:
      raise ValueError('Input should not be empty.')
    in_shape = x[0].shape
    x = tf.stack(x, axis=-1)
    x = tf.reduce_sum(input_tensor=x, axis=-1)
    alt_value = np.array(alt_value, x.dtype.as_numpy_dtype)
    alt_fill = tf.fill(tf.shape(input=x), value=alt_value)
    x = tf.where(tf.math.is_finite(x), x, alt_fill)
    x.set_shape(x.shape.merge_with(in_shape))
    return x


def set_doc(value):
  """Decorator to programmatically set a function docstring."""
  def _doc(func):
    func.__doc__ = value
    return func
  return _doc


def _value_and_gradients(fn, fn_arg_list, result=None, grads=None, name=None):
  """Helper to `maybe_call_fn_and_grads`."""
  with tf.compat.v1.name_scope(name, 'value_and_gradients',
                               [fn_arg_list, result, grads]):

    def _convert_to_tensor(x, name):
      ctt = lambda x_: x_ if x_ is None else tf.convert_to_tensor(
          value=x_, name=name)
      return [ctt(x_) for x_ in x] if is_list_like(x) else ctt(x)

    fn_arg_list = (list(fn_arg_list) if is_list_like(fn_arg_list)
                   else [fn_arg_list])
    fn_arg_list = _convert_to_tensor(fn_arg_list, 'fn_arg')

    if result is None:
      result = fn(*fn_arg_list)
      if grads is None and tf.executing_eagerly():
        # Ensure we disable bijector cacheing in eager mode.
        # TODO(b/72831017): Remove this once bijector cacheing is fixed for
        # eager mode.
        fn_arg_list = [0 + x for x in fn_arg_list]

    result = _convert_to_tensor(result, 'fn_result')

    if grads is not None:
      grads = _convert_to_tensor(grads, 'fn_grad')
      return result, grads

    if is_list_like(result) and len(result) == len(fn_arg_list):
      # Compute the block diagonal of Jacobian.
      # TODO(b/79158574): Guard this calculation by an arg which explicitly
      # requests block diagonal Jacobian calculation.
      def fn_slice(i):
        """Needed to prevent `cell-var-from-loop` pylint warning."""
        return lambda x: fn(*(fn_arg_list[:i] + [x] + fn_arg_list[i+1:]))
      grads = [
          tfp_math_value_and_gradients(fn_slice(i), fn_arg_list[i])[1]
          for i in range(len(result))
      ]
    else:
      _, grads = tfp_math_value_and_gradients(fn, fn_arg_list)

    return result, grads


def maybe_call_fn_and_grads(fn,
                            fn_arg_list,
                            result=None,
                            grads=None,
                            check_non_none_grads=True,
                            name=None):
  """Calls `fn` and computes the gradient of the result wrt `args_list`."""
  with tf.compat.v1.name_scope(name, 'maybe_call_fn_and_grads',
                               [fn_arg_list, result, grads]):
    fn_arg_list = (list(fn_arg_list) if is_list_like(fn_arg_list)
                   else [fn_arg_list])
    result, grads = _value_and_gradients(fn, fn_arg_list, result, grads)
    if not all(r.dtype.is_floating
               for r in (result if is_list_like(result) else [result])):  # pylint: disable=superfluous-parens
      raise TypeError('Function result must be a `Tensor` with `float` '
                      '`dtype`.')
    if len(fn_arg_list) != len(grads):
      raise ValueError('Function args must be in one-to-one correspondence '
                       'with grads.')
    if check_non_none_grads and any(g is None for g in grads):
      raise ValueError('Encountered `None` gradient.\n'
                       '  fn_arg_list: {}\n'
                       '  grads: {}'.format(fn_arg_list, grads))
    return result, grads


def smart_for_loop(loop_num_iter, body_fn, initial_loop_vars,
                   parallel_iterations=10, name=None):
  """Construct a for loop, preferring a python loop if `n` is staticaly known.

  Given `loop_num_iter` and `body_fn`, return an op corresponding to executing
  `body_fn` `loop_num_iter` times, feeding previous outputs of `body_fn` into
  the next iteration.

  If `loop_num_iter` is statically known, the op is constructed via python for
  loop, and otherwise a `tf.while_loop` is used.

  Args:
    loop_num_iter: `Integer` `Tensor` representing the number of loop
      iterations.
    body_fn: Callable to be executed `loop_num_iter` times.
    initial_loop_vars: Listlike object of `Tensors` to be passed in to
      `body_fn`'s first execution.
    parallel_iterations: The number of iterations allowed to run in parallel.
      It must be a positive integer. See `tf.while_loop` for more details.
      Default value: `10`.
    name: Python `str` name prefixed to Ops created by this function.
      Default value: `None` (i.e., "smart_for_loop").
  Returns:
    result: `Tensor` representing applying `body_fn` iteratively `n` times.
  """
  with tf.compat.v1.name_scope(name, 'smart_for_loop',
                               [loop_num_iter, initial_loop_vars]):
    loop_num_iter_ = tf.get_static_value(loop_num_iter)
    if (loop_num_iter_ is None or tf.executing_eagerly() or
        control_flow_util.GraphOrParentsInXlaContext(
            tf.compat.v1.get_default_graph())):
      # Cast to int32 to run the comparison against i in host memory,
      # where while/LoopCond needs it.
      loop_num_iter = tf.cast(loop_num_iter, dtype=tf.int32)
      return tf.while_loop(
          cond=lambda i, *args: i < loop_num_iter,
          body=lambda i, *args: [i + 1] + list(body_fn(*args)),
          loop_vars=[np.int32(0)] + initial_loop_vars,
          parallel_iterations=parallel_iterations
      )[1:]
    result = initial_loop_vars
    for _ in range(loop_num_iter_):
      result = body_fn(*result)
    return result


def trace_scan(loop_fn,
               initial_state,
               elems,
               trace_fn,
               parallel_iterations=10,
               name=None):
  """A simplified version of `tf.scan` that has configurable tracing.

  This function repeatedly calls `loop_fn(state, elem)`, where `state` is the
  `initial_state` during the first iteration, and the return value of `loop_fn`
  for every iteration thereafter. `elem` is a slice of `elements` along the
  first dimension, accessed in order. Additionally, it calls `trace_fn` on the
  return value of `loop_fn`. The `Tensor`s in return values of `trace_fn` are
  stacked and returned from this function, such that the first dimension of
  those `Tensor`s matches the size of `elems`.

  Args:
    loop_fn: A callable that takes in a `Tensor` or a nested collection of
      `Tensor`s with the same structure as `initial_state`, a slice of `elems`
      and returns the same structure as `initial_state`.
    initial_state: A `Tensor` or a nested collection of `Tensor`s passed to
      `loop_fn` in the first iteration.
    elems: A `Tensor` that is split along the first dimension and each element
      of which is passed to `loop_fn`.
    trace_fn: A callable that takes in the return value of `loop_fn` and returns
      a `Tensor` or a nested collection of `Tensor`s.
    parallel_iterations: Passed to the internal `tf.while_loop`.
    name: Name scope used in this function. Default: 'trace_scan'.

  Returns:
    final_state: The final return value of `loop_fn`.
    trace: The same structure as the return value of `trace_fn`, but with each
      `Tensor` being a stack of the corresponding `Tensors` in the return value
      of `trace_fn` for each slice of `elems`.
  """
  with tf.compat.v1.name_scope(
      name, 'trace_scan', [initial_state, elems]), tf.compat.v1.variable_scope(
          tf.compat.v1.get_variable_scope()) as vs:
    if vs.caching_device is None and not tf.executing_eagerly():
      vs.set_caching_device(lambda op: op.device)

    initial_state = tf.nest.map_structure(
        lambda x: tf.convert_to_tensor(value=x, name='initial_state'),
        initial_state)
    elems = tf.convert_to_tensor(value=elems, name='elems')

    static_length = elems.shape[0]
    if tf.compat.dimension_value(static_length) is None:
      length = tf.shape(input=elems)[0]
    else:
      length = tf.convert_to_tensor(
          value=static_length, dtype=tf.int32, name='length')

    # This is an TensorArray in part because of XLA, which had trouble with
    # non-statically known indices. I.e. elems[i] errored, but
    # elems_array.read(i) worked.
    elems_array = tf.TensorArray(
        elems.dtype, size=length, element_shape=elems.shape[1:])
    elems_array = elems_array.unstack(elems)

    trace_arrays = tf.nest.map_structure(
        lambda x: tf.TensorArray(x.dtype, size=length, element_shape=x.shape),
        trace_fn(initial_state))

    def _body(i, state, trace_arrays):
      state = loop_fn(state, elems_array.read(i))
      trace_arrays = tf.nest.pack_sequence_as(trace_arrays, [
          a.write(i, v) for a, v in zip(
              tf.nest.flatten(trace_arrays), tf.nest.flatten(trace_fn(state)))
      ])
      return i + 1, state, trace_arrays

    _, final_state, trace_arrays = tf.while_loop(
        cond=lambda i, *args: i < length,
        body=_body,
        loop_vars=(0, initial_state, trace_arrays),
        parallel_iterations=parallel_iterations)

    stacked_trace = tf.nest.map_structure(lambda x: x.stack(), trace_arrays)

    # Restore the static length if we know it.
    def _merge_static_length(x):
      x.set_shape(tf.TensorShape(static_length).concatenate(x.shape[1:]))
      return x

    stacked_trace = tf.nest.map_structure(_merge_static_length, stacked_trace)
    return final_state, stacked_trace


def make_innermost_setter(setter):
  """Wraps a setter so it applies to the inner-most results in `kernel_results`.

  The wrapped setter unwraps `kernel_results` and applies `setter` to the first
  results without an `inner_results` attribute.

  Args:
    setter: A callable that takes the kernel results as well as some `*args` and
      `**kwargs` and returns a modified copy of those kernel results.

  Returns:
    new_setter: A wrapped `setter`.
  """

  @functools.wraps(setter)
  def _new_setter(kernel_results, *args, **kwargs):
    """Wrapped setter."""
    results_stack = []
    while hasattr(kernel_results, 'inner_results'):
      results_stack.append(kernel_results)
      kernel_results = kernel_results.inner_results

    new_kernel_results = setter(kernel_results, *args, **kwargs)
    for outer_results in reversed(results_stack):
      new_kernel_results = outer_results._replace(
          inner_results=new_kernel_results)

    return new_kernel_results

  return _new_setter


def make_innermost_getter(getter):
  """Wraps a getter so it applies to the inner-most results in `kernel_results`.

  The wrapped getter unwraps `kernel_results` and returns the return value of
  `getter` called with the first results without an `inner_results` attribute.

  Args:
    getter: A callable that takes Kernel results and returns some value.

  Returns:
    new_getter: A wrapped `getter`.
  """

  @functools.wraps(getter)
  def _new_getter(kernel_results, *args, **kwargs):
    """Wrapped getter."""
    results_stack = []
    while hasattr(kernel_results, 'inner_results'):
      results_stack.append(kernel_results)
      kernel_results = kernel_results.inner_results

    return getter(kernel_results, *args, **kwargs)

  return _new_getter


def enable_store_parameters_in_results(kernel):
  """Enables the `store_parameters_in_results` parameter in a chain of kernels.

  This is a temporary utility for use during the transition period of the
  parameter storage methods.

  Args:
    kernel: A TransitionKernel.

  Returns:
    kernel: The same kernel, but recreated with `store_parameters_in_results`
        recursively set to `True` in its parameters and its inner kernels (as
        appropriate).
  """
  kernel_stack = []
  while hasattr(kernel, 'parameters') and 'inner_kernel' in kernel.parameters:
    kernel_stack.append(kernel)
    kernel = kernel.parameters['inner_kernel']

  def _recreate_kernel(kernel, parameters):
    new_parameters = kernel.parameters.copy()
    new_parameters.update(parameters)
    if 'store_parameters_in_results' in new_parameters:
      new_parameters['store_parameters_in_results'] = True
    with deprecation.silence():
      return type(kernel)(**new_parameters)

  if hasattr(kernel, 'parameters'):
    kernel = _recreate_kernel(kernel, {})

  for outer_kernel in reversed(kernel_stack):
    outer_kernel = _recreate_kernel(outer_kernel, {'inner_kernel': kernel})
    kernel = outer_kernel

  return kernel


def warn_if_parameters_are_not_simple_tensors(params_dict):
  for param_name, param in params_dict.items():
    if not isinstance(param, tf.Tensor) and np.array(param).dtype == np.object:
      warnings.warn(
          '`{}` is not a `tf.Tensor`, Python number, or Numpy array. If this '
          'parameter is mutable (e.g., a `tf.Variable`), then the '
          'behavior implied by `store_parameters_in_results` will silently '
          'change on 2019-08-01. Please consult the docstring for '
          '`store_parameters_in_results` details and use '
          '`store_parameters_in_results=True` to silence this warning.'.format(
              param_name))
