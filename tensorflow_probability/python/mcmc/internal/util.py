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
import numpy as onp  # JAX rewrites numpy import  # pylint: disable=reimported
import tensorflow.compat.v1 as tf1
import tensorflow.compat.v2 as tf

from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import prefer_static
from tensorflow_probability.python.internal import tensorshape_util
from tensorflow_probability.python.math.gradient import value_and_gradient as tfp_math_value_and_gradients
from tensorflow.python.ops import control_flow_util  # pylint: disable=g-direct-tensorflow-import
from tensorflow.python.util import deprecation  # pylint: disable=g-direct-tensorflow-import


__all__ = [
    'choose',
    'enable_store_parameters_in_results',
    'left_justified_expand_dims_like',
    'left_justified_expand_dims_to',
    'left_justified_broadcast_like',
    'left_justified_broadcast_to',
    'index_remapping_gather',
    'is_list_like',
    'is_namedtuple_like',
    'make_innermost_getter',
    'make_innermost_setter',
    'make_name',
    'maybe_call_fn_and_grads',
    'prepare_state_parts',
    'safe_sum',
    'set_doc',
    'smart_for_loop',
    'trace_scan',
    'warn_if_parameters_are_not_simple_tensors',
]


def left_justified_expand_dims_like(x, reference, name=None):
  """Right pads `x` with `rank(reference) - rank(x)` ones."""
  with tf.name_scope(name or 'left_justified_expand_dims_like'):
    return left_justified_expand_dims_to(x, prefer_static.rank(reference))


def left_justified_expand_dims_to(x, rank, name=None):
  """Right pads `x` with `rank - rank(x)` ones."""
  with tf.name_scope(name or 'left_justified_expand_dims_to'):
    rank = tf.convert_to_tensor(rank, dtype=tf.int32)
    expand_ndims = prefer_static.maximum(rank - prefer_static.rank(x), 0)
    expand_shape = prefer_static.pad(
        prefer_static.shape(x),
        paddings=[[0, expand_ndims]],
        constant_values=1)
    return prefer_static.reshape(x, expand_shape)


def left_justified_broadcast_like(x, reference, name=None):
  """Broadcasts `x` to shape of reference, in a left-justified manner."""
  with tf.name_scope(name or 'left_justified_broadcast_like'):
    return left_justified_broadcast_to(x, prefer_static.shape(reference))


def left_justified_broadcast_to(x, shape, name=None):
  """Broadcasts `x` to shape, in a left-justified manner."""
  with tf.name_scope(name or 'left_justified_broadcast_to'):
    return tf.broadcast_to(
        left_justified_expand_dims_to(x, prefer_static.size(shape)), shape)


def prepare_state_parts(state_or_state_part, dtype=None, name=None):
  """Calls c2t on each element or the entirety if not iterable; returns list."""
  # Don't use tf.name_scope since this function has ct2-like semantics.
  is_multipart = is_list_like(state_or_state_part)
  state_parts = state_or_state_part if is_multipart else [state_or_state_part]
  state_parts = [tf.convert_to_tensor(x, dtype=dtype, name=name)
                 for x in state_parts]
  return state_parts, is_multipart


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
  """Helper which makes a `str` name; useful for tf.name_scope."""
  name = super_name if super_name is not None else default_super_name
  if sub_name is not None:
    name += '_' + sub_name
  return name


def _choose_base_case(is_accepted,
                      accepted,
                      rejected,
                      name=None):
  """Helper to `choose` which expand_dims `is_accepted` and applies tf.where."""
  def _where(accepted, rejected):
    """Wraps `tf.where`."""
    if accepted is rejected:
      return accepted
    # Preserve the name from `rejected` so names can propagate from
    # `bootstrap_results`.
    name = getattr(rejected, 'name', None)
    if name is not None:
      name = name.rpartition('/')[2].rsplit(':', 1)[0]
    # Since this is an internal utility it is ok to assume
    # tf.shape(accepted) == tf.shape(rejected).
    return tf.where(left_justified_expand_dims_like(is_accepted, accepted),
                    accepted, rejected, name=name)
  with tf.name_scope(name or 'choose'):
    if not is_list_like(accepted):
      return _where(accepted, rejected)
    return [(choose(is_accepted, a, r, name=name) if is_namedtuple_like(a)
             else _where(a, r))
            for a, r in zip(accepted, rejected)]


def choose(is_accepted, accepted, rejected, name=None):
  """Helper which expand_dims `is_accepted` then applies tf.where."""
  with tf.name_scope(name or 'choose'):
    if not is_namedtuple_like(accepted):
      return _choose_base_case(is_accepted, accepted, rejected, name=name)
    if not isinstance(accepted, type(rejected)):
      raise TypeError('Type of `accepted` ({}) must be identical to '
                      'type of `rejected` ({})'.format(
                          type(accepted).__name__,
                          type(rejected).__name__))
    return type(accepted)(**dict(
        [(fn,  # pylint: disable=g-complex-comprehension
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
  with tf.name_scope(name or 'safe_sum'):
    if not is_list_like(x):
      raise TypeError('Expected list input.')
    if not x:
      raise ValueError('Input should not be empty.')
    in_shape = x[0].shape
    x = tf.add_n(x)
    x = tf.where(tf.math.is_finite(x), x, tf.constant(alt_value, dtype=x.dtype))
    tensorshape_util.set_shape(x, in_shape)
    return x


def set_doc(value):
  """Decorator to programmatically set a function docstring."""
  def _doc(func):
    func.__doc__ = value
    return func
  return _doc


def _value_and_gradients(fn, fn_arg_list, result=None, grads=None, name=None):
  """Helper to `maybe_call_fn_and_grads`."""
  with tf.name_scope(name or 'value_and_gradients'):

    def _convert_to_tensor(x, name):
      ctt = lambda x_: None if x_ is None else tf.convert_to_tensor(  # pylint: disable=g-long-lambda
          x_, name=name)
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
  with tf.name_scope(name or 'maybe_call_fn_and_grads'):
    fn_arg_list = (list(fn_arg_list) if is_list_like(fn_arg_list)
                   else [fn_arg_list])
    result, grads = _value_and_gradients(fn, fn_arg_list, result, grads)
    if not all(dtype_util.is_floating(r.dtype)
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
  with tf.name_scope(name or 'smart_for_loop'):
    loop_num_iter_ = tf.get_static_value(loop_num_iter)
    if (loop_num_iter_ is None or tf.executing_eagerly() or
        control_flow_util.GraphOrParentsInXlaContext(
            tf1.get_default_graph())):
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
  with tf.name_scope(name or 'trace_scan'), tf1.variable_scope(
      tf1.get_variable_scope()) as vs:
    if vs.caching_device is None and not tf.executing_eagerly():
      vs.set_caching_device(lambda op: op.device)

    initial_state = tf.nest.map_structure(
        lambda x: tf.convert_to_tensor(x, name='initial_state'),
        initial_state)
    elems = tf.convert_to_tensor(elems, name='elems')

    length = prefer_static.size0(elems)

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
    static_length = tf.TensorShape(
        length if prefer_static.is_numpy(length) else None)
    def _merge_static_length(x):
      tensorshape_util.set_shape(x, static_length.concatenate(x.shape[1:]))
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


def _is_tensor_like(param):
  if is_list_like(param):
    return all([_is_tensor_like(p) for p in param])
  return isinstance(param, tf.Tensor) or (np.array(param).dtype != onp.object)


def warn_if_parameters_are_not_simple_tensors(params_dict):
  for param_name, param in params_dict.items():
    if not _is_tensor_like(param):
      warnings.warn(
          '`{}` is not a `tf.Tensor`, Python number, or Numpy array. If this '
          'parameter is mutable (e.g., a `tf.Variable`), then the '
          'behavior implied by `store_parameters_in_results` will silently '
          'change on 2019-08-01. Please consult the docstring for '
          '`store_parameters_in_results` details and use '
          '`store_parameters_in_results=True` to silence this warning.'.format(
              param_name))


def index_remapping_gather(params, indices, name='index_remapping_gather'):
  """Uses `indices` to remap values from `axis` of `params`.

  If `rank(params) = rank(indices) = 3`, this returns `remapped`:
  `remapped[i, j, k] = params[indices[i, j, k], j, k]`.

  In general, with `rank(indices) = K <= N = rank(params)`,

  ```remapped[i, ..., N] = params[indices[i,...,K], 1,..., N].```

  Args:
    params:  `N-D` `Tensor` (`N > 0`) from which to gather values.
      Number of dimensions must be known statically.
    indices: `Tensor` with values in `{0, ..., params.shape[0]-1}`, and
      `indices.shape[1:]` able to do a left-justified broadcast with
      `params.shape[1:]`.
    name: String name for scoping created ops.

  Returns:
    `Tensor` composed of elements of `params`.

  Raises:
    ValueError: If shape/rank requirements are not met.
  """
  with tf.name_scope(name):
    params = tf.convert_to_tensor(params, name='params')
    indices = tf.convert_to_tensor(indices, name='indices')

    params_ndims = params.shape.ndims
    indices_ndims = indices.shape.ndims

    if params_ndims is None:
      raise ValueError(
          'Rank of `params`, must be known statically. This is due to '
          'tf.gather not accepting a `Tensor` for `batch_dims`.')

    if params_ndims < 1:
      raise ValueError(
          'Rank of params should be `> 0`, but was {}'.format(params_ndims))

    if indices_ndims is not None and indices_ndims < 1:
      raise ValueError(
          'Rank of indices should be `> 0`, but was {}'.format(indices_ndims))

    if indices_ndims is not None and indices_ndims > params_ndims:
      raise ValueError(
          'Rank of `params` ({}) must be >= rank of `indices` ({}), but was '
          'not'.format(params_ndims, indices_ndims))

    # tf.gather requires batch dims to have identical shape.
    bcast_shape = prefer_static.pad(
        prefer_static.shape(params)[1:],
        paddings=[[1, 0]],
        constant_values=prefer_static.size0(indices))
    indices = left_justified_broadcast_to(indices, bcast_shape)

    # perm_fwd rotates dimensions left, perm_rev rotates right.
    perm_fwd = prefer_static.pad(prefer_static.range(1, params_ndims),
                                 paddings=[[0, 1]],
                                 constant_values=0)
    perm_rev = prefer_static.pad(prefer_static.range(params_ndims - 1),
                                 paddings=[[1, 0]],
                                 constant_values=params_ndims - 1)

    # result_t[i, ..., N] = params_t[i, ..., N-1, indices_t[i, ..., N]].
    # I.e., we're gathering on axis=-1, with all but the last dim a batch dim.
    result_t = tf.gather(
        # Transpose params/indices so that the `axis` dimension is rightmost.
        tf.transpose(params, perm_fwd),
        tf.transpose(indices, perm_fwd),
        batch_dims=params_ndims - 1, axis=-1)

    return tf.transpose(result_t, perm_rev)
