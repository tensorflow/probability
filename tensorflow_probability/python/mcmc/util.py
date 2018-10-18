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
"""Internal utiltiy functions for implementing TransitionKernels."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
import numpy as np
import tensorflow as tf

from tensorflow.contrib import eager as tfe
from tensorflow.python.framework import tensor_util


__all__ = [
    'choose',
    'is_list_like',
    'is_namedtuple_like',
    'make_name',
    'maybe_call_fn_and_grads',
    'safe_sum',
    'set_doc',
    'smart_for_loop',
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
  def _expand_is_accepted_like(x):
    """Helper to expand `is_accepted` like the shape of some input arg."""
    with tf.name_scope('expand_is_accepted_like'):
      expand_shape = tf.concat([
          tf.shape(is_accepted),
          tf.ones([tf.rank(x) - tf.rank(is_accepted)],
                  dtype=tf.int32),
      ], axis=0)
      multiples = tf.concat([
          tf.ones([tf.rank(is_accepted)], dtype=tf.int32),
          tf.shape(x)[tf.rank(is_accepted):],
      ], axis=0)
      m = tf.tile(tf.reshape(is_accepted, expand_shape),
                  multiples)
      m.set_shape(m.shape.merge_with(x.shape))
      return m
  def _where(accepted, rejected):
    accepted = tf.convert_to_tensor(accepted, name='accepted')
    rejected = tf.convert_to_tensor(rejected, name='rejected')
    r = tf.where(_expand_is_accepted_like(accepted), accepted, rejected)
    r.set_shape(r.shape.merge_with(accepted.shape.merge_with(rejected.shape)))
    return r
  with tf.name_scope(name, 'choose', values=[
      is_accepted, accepted, rejected]):
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
  with tf.name_scope(name, 'safe_sum', [x, alt_value]):
    if not is_list_like(x):
      raise TypeError('Expected list input.')
    if not x:
      raise ValueError('Input should not be empty.')
    n = np.int64(len(x))
    in_shape = x[0].shape
    x = tf.stack(x, axis=-1)
    # The sum is NaN if any element is NaN or we see both +Inf and -Inf.  Thus
    # we will replace such rows with the `alt_value`. Typically the `alt_value`
    # is chosen so the `MetropolisHastings` `TransitionKernel` always rejects
    # the proposal.  rejection.
    # Regarding the following float-comparisons, recall comparing with NaN is
    # always False, i.e., we're implicitly capturing NaN and explicitly
    # capturing +/- Inf.
    is_sum_determinate = (
        tf.reduce_all(tf.is_finite(x) | (x >= 0.), axis=-1) &
        tf.reduce_all(tf.is_finite(x) | (x <= 0.), axis=-1))
    is_sum_determinate = tf.tile(
        is_sum_determinate[..., tf.newaxis],
        multiples=tf.concat([tf.ones(tf.rank(x) - 1, dtype=tf.int64), [n]],
                            axis=0))
    alt_value = np.array(alt_value, x.dtype.as_numpy_dtype)
    x = tf.where(is_sum_determinate, x, tf.fill(tf.shape(x), value=alt_value))
    x = tf.reduce_sum(x, axis=-1)
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
  with tf.name_scope(name, 'value_and_gradients', [fn_arg_list, result, grads]):
    def _convert_to_tensor(x, name):
      ctt = lambda x_: x_ if x_ is None else tf.convert_to_tensor(x_, name=name)
      return [ctt(x_) for x_ in x] if is_list_like(x) else ctt(x)

    fn_arg_list = (list(fn_arg_list) if is_list_like(fn_arg_list)
                   else [fn_arg_list])
    fn_arg_list = _convert_to_tensor(fn_arg_list, 'fn_arg')

    if result is None:
      result = fn(*fn_arg_list)
      if grads is None and tfe.executing_eagerly():
        # Ensure we disable bijector cacheing in eager mode.
        # TODO(b/72831017): Remove this once bijector cacheing is fixed for
        # eager mode.
        fn_arg_list = [0 + x for x in fn_arg_list]

    result = _convert_to_tensor(result, 'fn_result')

    if grads is not None:
      grads = _convert_to_tensor(grads, 'fn_grad')
      return result, grads

    if tfe.executing_eagerly():
      if is_list_like(result) and len(result) == len(fn_arg_list):
        # Compute the block diagonal of Jacobian.
        # TODO(b/79158574): Guard this calculation by an arg which explicitly
        # requests block diagonal Jacobian calculation.
        def make_fn_slice(i):
          """Needed to prevent `cell-var-from-loop` pylint warning."""
          return lambda *args: fn(*args)[i]
        grads = [
            tfe.gradients_function(make_fn_slice(i))(*fn_arg_list)[i]
            for i in range(len(result))
        ]
      else:
        grads = tfe.gradients_function(fn)(*fn_arg_list)
    else:
      if is_list_like(result) and len(result) == len(fn_arg_list):
        # Compute the block diagonal of Jacobian.
        # TODO(b/79158574): Guard this calculation by an arg which explicitly
        # requests block diagonal Jacobian calculation.
        grads = [tf.gradients(result[i], fn_arg_list[i])[0]
                 for i in range(len(result))]
      else:
        grads = tf.gradients(result, fn_arg_list)

    return result, grads


def maybe_call_fn_and_grads(fn,
                            fn_arg_list,
                            result=None,
                            grads=None,
                            check_non_none_grads=True,
                            name=None):
  """Calls `fn` and computes the gradient of the result wrt `args_list`."""
  with tf.name_scope(name, 'maybe_call_fn_and_grads',
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
  with tf.name_scope(
      name, 'smart_for_loop', [loop_num_iter, initial_loop_vars]):
    loop_num_iter_ = tensor_util.constant_value(tf.convert_to_tensor(
        loop_num_iter, dtype=tf.int64, name='loop_num_iter'))
    if loop_num_iter_ is None or tf.contrib.eager.executing_eagerly():
      return tf.while_loop(
          cond=lambda i, *args: i < loop_num_iter,
          body=lambda i, *args: [i + 1] + list(body_fn(*args)),
          loop_vars=[np.int64(0)] + initial_loop_vars,
          parallel_iterations=parallel_iterations
      )[1:]
    result = initial_loop_vars
    for _ in range(loop_num_iter_):
      result = body_fn(*result)
    return result
