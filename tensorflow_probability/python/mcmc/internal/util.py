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

import warnings

# Dependency imports
import numpy as np
import tensorflow.compat.v1 as tf1
import tensorflow.compat.v2 as tf

from tensorflow_probability.python.internal import broadcast_util as bu
from tensorflow_probability.python.internal import distribution_util as dist_util
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import prefer_static as ps
from tensorflow_probability.python.internal import tensorshape_util
from tensorflow_probability.python.math.gradient import value_and_gradient as tfp_math_value_and_gradients
from tensorflow.python.ops import control_flow_util  # pylint: disable=g-direct-tensorflow-import
from tensorflow.python.util import deprecation  # pylint: disable=g-direct-tensorflow-import


__all__ = [
    'choose',
    'enable_store_parameters_in_results',
    'index_remapping_gather',
    'is_list_like',
    'is_namedtuple_like',
    'make_name',
    'maybe_call_fn_and_grads',
    'prepare_state_parts',
    'PrettyNamedTupleMixin',
    'safe_sum',
    'SEED_CTOR_ARG_DEPRECATION_MSG',
    'set_doc',
    'smart_for_loop',
    'strip_seeds',
    'trace_scan',
    'warn_if_parameters_are_not_simple_tensors',
]


JAX_MODE = False

SEED_CTOR_ARG_DEPRECATION_MSG = (
    'Seeding `tfp.mcmc.TransitionKernel` instances by constructor argument is '
    'deprecated. Use the `seed` argument to `tfp.mcmc.sample_chain` or '
    'directly on `one_step`. The legacy behavior is still supported and should '
    'be through 2020-09-20.')


class PrettyNamedTupleMixin(object):
  """Mixin adding a nicer `__repr__` for `namedtuple`s."""
  __slots__ = ()

  def __repr__(self):
    return '{}(\n{}\n)'.format(
        type(self).__name__,
        ',\n'.join('  {}={}'.format(k, repr(v).replace('\n', '\n    '))
                   for (k, v) in self._asdict().items()))


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
                      proposed,
                      current,
                      name=None,
                      addr=None,):
  """Helper to `choose` which expand_dims `is_accepted` and applies tf.where."""
  def _where(proposed, current):
    """Wraps `tf.where`."""
    if proposed is current:
      return proposed
    # Preserve the name from `current` so names can propagate from
    # `bootstrap_results`.
    name = getattr(current, 'name', None)
    if name is not None:
      name = name.rpartition('/')[2].rsplit(':', 1)[0]
    # Since this is an internal utility it is ok to assume
    # tf.shape(proposed) == tf.shape(current).
    return tf.where(bu.left_justified_expand_dims_like(is_accepted, proposed),
                    proposed, current, name=name)
  with tf.name_scope(name or 'choose'):
    if not is_list_like(proposed):
      return _where(proposed, current)
    return tf.nest.pack_sequence_as(
        current,
        [(_choose_recursive(is_accepted, p, c, name=name, addr=f'{addr}[i]')
          if is_namedtuple_like(p) else
          _where(p, c)) for i, (p, c) in enumerate(zip(proposed, current))])


def _choose_recursive(is_accepted, proposed, current, name=None, addr='<root>'):
  """Recursion helper which also reports the address of any failures."""
  with tf.name_scope(name or 'choose'):
    if not is_namedtuple_like(proposed):
      return _choose_base_case(is_accepted, proposed, current, name=name,
                               addr=addr)
    if not isinstance(proposed, type(current)):
      raise TypeError(
          f'Type of `proposed` ({type(proposed).__name__}) must be identical '
          f'to type of `current` ({type(current).__name__}). (At "{addr}".)')
    items = {}
    for fn in proposed._fields:
      items[fn] = _choose_recursive(is_accepted,
                                    getattr(proposed, fn),
                                    getattr(current, fn),
                                    name=name,
                                    addr=f'{addr}/{fn}')
    return type(proposed)(**items)


def choose(is_accepted, proposed, current, name=None):
  """Helper which expand_dims `is_accepted` then applies tf.where."""
  return _choose_recursive(is_accepted, proposed, current, name=name)


def strip_seeds(obj):
  if not is_namedtuple_like(obj):
    return obj
  return type(obj)(**{fn: strip_seeds(fv) if fn != 'seed' else []
                      for fn, fv in obj._asdict().items()})


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

    if result is None and grads is None and (JAX_MODE or
                                             not tf.executing_eagerly()):
      # Currently, computing gradient is not working well with caching in
      # tensorflow eager mode (see below), so we will handle that case
      # separately.
      return tfp_math_value_and_gradients(fn, fn_arg_list)

    if result is None:
      result = fn(*fn_arg_list)
      if grads is None:
        assert tf.executing_eagerly()
        # Ensure we disable bijector cacheing in eager mode.
        # TODO(b/72831017): Remove this once bijector cacheing is fixed for
        # eager mode.
        fn_arg_list = [0 + x for x in fn_arg_list]

    result = _convert_to_tensor(result, 'fn_result')

    if grads is not None:
      grads = _convert_to_tensor(grads, 'fn_grad')
      return result, grads

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
                   parallel_iterations=10, unroll_threshold=1, name=None):
  """Construct a for loop, preferring a python loop if `n` is statically known.

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
    unroll_threshold: Integer denoting the maximum number of iterations to
      unroll, if possible. If `loop_num_iter > unroll_threshold` a
      `tf.while_loop` will always be used, even if `loop_num_iter` is
      statically known.
      Default value: `1`.
    name: Python `str` name prefixed to Ops created by this function.
      Default value: `None` (i.e., "smart_for_loop").
  Returns:
    result: `Tensor` representing applying `body_fn` iteratively `n` times.
  """
  with tf.name_scope(name or 'smart_for_loop'):
    loop_num_iter_ = tf.get_static_value(loop_num_iter)
    if (loop_num_iter_ is None
        or tf.executing_eagerly()
        # large values for loop_num_iter_ will cause ridiculously slow
        # graph compilation time (GitHub issue #1033)
        or loop_num_iter_ > unroll_threshold
        or control_flow_util.GraphOrParentsInXlaContext(
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
               trace_criterion_fn=None,
               static_trace_allocation_size=None,
               condition_fn=None,
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
    trace_criterion_fn: Optional callable that takes in the return value of
      `loop_fn` and returns a boolean `Tensor` indicating whether to trace it.
      If `None`, all steps are traced.
      Default value: `None`.
    static_trace_allocation_size: Optional Python `int` size of trace to
      allocate statically. This should be an upper bound on the number of steps
      traced and is used only when the length cannot be
      statically inferred (for example, if a `trace_criterion_fn` is specified).
      It is primarily intended for contexts where static shapes are required,
      such as in XLA-compiled code.
      Default value: `None`.
    condition_fn: Python `callable` additional loop termination condition, with
     signature `should_continue = condition_fn(step, state, num_traced, trace)`;
     returning `False` will terminate early and not scan over all of `elems`.
     Default value: `None`, which means no additional termination condition.
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
        initial_state, expand_composites=True)
    elems = tf.convert_to_tensor(elems, name='elems')

    length = ps.size0(elems)

    # This is an TensorArray in part because of XLA, which had trouble with
    # non-statically known indices. I.e. elems[i] errored, but
    # elems_array.read(i) worked.
    elems_array = tf.TensorArray(
        elems.dtype, size=length, element_shape=elems.shape[1:])
    elems_array = elems_array.unstack(elems)

    # Initialize trace arrays.
    if trace_criterion_fn is None and condition_fn is None:
      dynamic_size, initial_size = tf.is_tensor(length), length
    elif static_trace_allocation_size is not None:
      dynamic_size, initial_size = False, static_trace_allocation_size
    elif JAX_MODE or (not tf.executing_eagerly() and
                      control_flow_util.GraphOrParentsInXlaContext(
                          tf1.get_default_graph())):
      dynamic_size, initial_size = False, length
    else:
      dynamic_size, initial_size = True, 0
    initial_trace = trace_fn(initial_state)
    flat_initial_trace = tf.nest.flatten(initial_trace, expand_composites=True)
    trace_arrays = []
    for trace_elt in flat_initial_trace:
      trace_arrays.append(
          tf.TensorArray(
              trace_elt.dtype,
              size=initial_size,
              dynamic_size=dynamic_size,
              element_shape=trace_elt.shape))

    # Helper for writing a (structured) state to (structured) arrays.
    def trace_one_step(num_steps_traced, trace_arrays, state):
      return [ta.write(num_steps_traced, x) for ta, x in
              zip(trace_arrays,
                  tf.nest.flatten(trace_fn(state), expand_composites=True))]

    def _body(i, state, num_steps_traced, trace_arrays):
      elem = elems_array.read(i)
      state = loop_fn(state, elem)

      trace_arrays, num_steps_traced = ps.cond(
          trace_criterion_fn(state) if trace_criterion_fn else True,
          lambda: (trace_one_step(num_steps_traced, trace_arrays, state),  # pylint: disable=g-long-lambda
                   num_steps_traced + 1),
          lambda: (trace_arrays, num_steps_traced))

      return i + 1, state, num_steps_traced, trace_arrays

    if condition_fn is None:
      cond = lambda i, *_: i < length
    else:
      cond = lambda i, *rest: (i < length) & condition_fn(i, *rest)

    _, final_state, _, trace_arrays = tf.while_loop(
        cond=cond,
        body=_body,
        loop_vars=(0, initial_state, 0, trace_arrays),
        parallel_iterations=parallel_iterations)

    # unflatten
    stacked_trace = tf.nest.pack_sequence_as(
        initial_trace, [ta.stack() for ta in trace_arrays],
        expand_composites=True)

    # Restore the static length if we know it.
    static_length = tf.TensorShape(None if dynamic_size else initial_size)

    def _merge_static_length(x):
      tensorshape_util.set_shape(x, static_length.concatenate(x.shape[1:]))
      return x

    stacked_trace = tf.nest.map_structure(
        _merge_static_length, stacked_trace, expand_composites=True)
    return final_state, stacked_trace


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
  if isinstance(param, tf.Tensor):
    return True
  elif isinstance(param, tf.Variable):
    return False
  else:
    return np.array(param).dtype != np.object_


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


def index_remapping_gather(params,
                           indices,
                           axis=0,
                           indices_axis=0,
                           name='index_remapping_gather'):
  """Gather values from `axis` of `params` using `indices_axis` of `indices`.

  The shape of `indices` must broadcast to that of `params` when
  their `indices_axis` and `axis` (respectively) are aligned:

  ```python
  # params.shape:
  [p[0],  ..., ...,         p[axis], ..., ..., p[rank(params)] - 1])
  # indices.shape:
        [i[0], ..., i[indices_axis], ..., i[rank(indices)] - 1])
  ```

  In particular, `params` must have at least as many
  leading dimensions as `indices` (`axis >= indices_axis`), and at least as many
  trailing dimensions (`rank(params) - axis >= rank(indices) - indices_axis`).

  The `result` has the same shape as `params`, except that the dimension
  of size `p[axis]` is replaced by one of size `i[indices_axis]`:

  ```python
  # result.shape:
  [p[0],  ..., ..., i[indices_axis], ..., ..., p[rank(params) - 1]]
  ```

  In the case where `rank(params) == 5`, `rank(indices) == 3`, `axis = 2`, and
  `indices_axis = 1`, the result is given by

   ```python
   # alignment is:                       v axis
   # params.shape    ==   [p[0], p[1], p[2], p[3], p[4]]
   # indices.shape   ==         [i[0], i[1], i[2]]
   #                                     ^ indices_axis
   result[i, j, k, l, m] = params[i, j, indices[j, k, l], l, m]
  ```

  Args:
    params:  `N-D` `Tensor` (`N > 0`) from which to gather values.
      Number of dimensions must be known statically.
    indices: `Tensor` with values in `{0, ..., params.shape[axis] - 1}`, whose
      shape broadcasts to that of `params` as described above.
    axis: Python `int` axis of `params` from which to gather.
    indices_axis: Python `int` axis of `indices` to align with the `axis`
      over which `params` is gathered.
    name: String name for scoping created ops.

  Returns:
    `Tensor` composed of elements of `params`.

  Raises:
    ValueError: If shape/rank requirements are not met.
  """
  with tf.name_scope(name):
    params = tf.convert_to_tensor(params, name='params')
    indices = tf.convert_to_tensor(indices, name='indices')

    params_ndims = tensorshape_util.rank(params.shape)
    indices_ndims = tensorshape_util.rank(indices.shape)
    # `axis` dtype must match ndims, which are 64-bit Python ints.
    axis = tf.get_static_value(ps.convert_to_shape_tensor(axis, dtype=tf.int64))
    indices_axis = tf.get_static_value(
        ps.convert_to_shape_tensor(indices_axis, dtype=tf.int64))

    if params_ndims is None:
      raise ValueError(
          'Rank of `params`, must be known statically. This is due to '
          'tf.gather not accepting a `Tensor` for `batch_dims`.')

    if axis is None:
      raise ValueError(
          '`axis` must be known statically. This is due to '
          'tf.gather not accepting a `Tensor` for `batch_dims`.')

    if indices_axis is None:
      raise ValueError(
          '`indices_axis` must be known statically. This is due to '
          'tf.gather not accepting a `Tensor` for `batch_dims`.')

    if indices_axis > axis:
      raise ValueError(
          '`indices_axis` should be <= `axis`, but was {} > {}'.format(
              indices_axis, axis))

    if params_ndims < 1:
      raise ValueError(
          'Rank of params should be `> 0`, but was {}'.format(params_ndims))

    if indices_ndims is not None and indices_ndims < 1:
      raise ValueError(
          'Rank of indices should be `> 0`, but was {}'.format(indices_ndims))

    if (indices_ndims is not None and
        (indices_ndims - indices_axis > params_ndims - axis)):
      raise ValueError(
          '`rank(params) - axis` ({} - {}) must be >= `rank(indices) - '
          'indices_axis` ({} - {}), but was not.'.format(
              params_ndims, axis, indices_ndims, indices_axis))

    # `tf.gather` requires the axis to be the rightmost batch ndim. So, we
    # transpose `indices_axis` to be the rightmost dimension of `indices`...
    transposed_indices = dist_util.move_dimension(indices,
                                                  source_idx=indices_axis,
                                                  dest_idx=-1)

    # ... and `axis` to be the corresponding (aligned as in the docstring)
    # dimension of `params`.
    broadcast_indices_ndims = indices_ndims + (axis - indices_axis)
    transposed_params = dist_util.move_dimension(
        params,
        source_idx=axis,
        dest_idx=broadcast_indices_ndims - 1)

    # Next we broadcast `indices` so that its shape has the same prefix as
    # `params.shape`.
    transposed_params_shape = ps.shape(transposed_params)
    result_shape = ps.concat([
        transposed_params_shape[:broadcast_indices_ndims - 1],
        ps.shape(indices)[indices_axis:indices_axis + 1],
        transposed_params_shape[broadcast_indices_ndims:]], axis=0)
    broadcast_indices = ps.broadcast_to(
        transposed_indices,
        result_shape[:broadcast_indices_ndims])

    result_t = tf.gather(transposed_params,
                         broadcast_indices,
                         batch_dims=broadcast_indices_ndims - 1,
                         axis=broadcast_indices_ndims - 1)
    return dist_util.move_dimension(result_t,
                                    source_idx=broadcast_indices_ndims - 1,
                                    dest_idx=axis)
