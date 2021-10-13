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
"""Numpy implementations of TensorFlow top-level control flow functions."""

import collections
import functools
import numpy as np

from tensorflow_probability.python.internal.backend.numpy import _utils as utils
from tensorflow_probability.python.internal.backend.numpy import dtype
from tensorflow_probability.python.internal.backend.numpy import nest
from tensorflow_probability.python.internal.backend.numpy import ops


__all__ = [
    'cond',
    'group',
    'no_op',
    'while_loop',
    # 'case',
    # 'dynamic_partition',
    # 'dynamic_stitch',
]


JAX_MODE = False


def _cond_jax(pred, true_fn=None, false_fn=None, name=None):  # pylint: disable=missing-docstring
  from jax import lax  # pylint: disable=g-import-not-at-top

  del name
  def overridden_true_fn(x):
    del x
    return true_fn()

  def overridden_false_fn(x):
    del x
    return false_fn()
  return lax.cond(pred, None, overridden_true_fn, None, overridden_false_fn)


def _cond(pred, true_fn=None, false_fn=None, name=None):  # pylint: disable=unused-argument
  return true_fn() if pred else false_fn()


def _no_op(name=None):  # pylint: disable=unused-argument
  pass


def _while_loop(cond, body, loop_vars,  # pylint: disable=redefined-outer-name
                shape_invariants=None, parallel_iterations=10,  # pylint: disable=unused-argument
                back_prop=True, swap_memory=False,  # pylint: disable=unused-argument
                maximum_iterations=None, name=None):  # pylint: disable=unused-argument
  """Numpy implementation of `tf.while_loop`."""
  i = 0
  while (cond(*loop_vars) and
         (maximum_iterations is None or i < maximum_iterations)):
    loop_vars = body(*loop_vars)
    i += 1
  return loop_vars


def _while_loop_jax(cond, body, loop_vars,  # pylint: disable=redefined-outer-name
                    shape_invariants=None, parallel_iterations=10,  # pylint: disable=unused-argument
                    back_prop=True, swap_memory=False,  # pylint: disable=unused-argument
                    maximum_iterations=None, name=None):  # pylint: disable=unused-argument
  """Jax implementation of `tf.while_loop`."""
  from jax import lax  # pylint: disable=g-import-not-at-top

  pack_body = lambda x: nest.pack_sequence_as(loop_vars, nest.flatten(x))

  if maximum_iterations is None:
    def override_body_fn(args):
      return pack_body(body(*args))
    def override_cond_fn(args):
      return cond(*args)
    return lax.while_loop(override_cond_fn, override_body_fn, loop_vars)
  elif back_prop:
    def override_body_fn(args, _):
      c = cond(*args)
      sc = ops.get_static_value(c)
      if sc is None:
        args = lax.cond(c, args, lambda args: pack_body(body(*args)), args,
                        lambda args: args)
      elif sc:
        args = pack_body(body(*args))
      return args, ()

    loop_vars, _ = lax.scan(
        override_body_fn, loop_vars, xs=None, length=maximum_iterations)
    return loop_vars
  else:
    def override_body_fn(args):
      i, args = args
      return i + 1, pack_body(body(*args))
    def override_cond_fn(args):
      i, args = args
      return cond(*args) & (i < maximum_iterations)
    return lax.while_loop(
        override_cond_fn, override_body_fn, (np.array(0), loop_vars))[1]


def _case_create_default_action(predicates, actions):
  """Creates default action for a list of actions and their predicates.

  It uses the input actions to select an arbitrary as default and makes sure
  that corresponding predicates have valid values.
  Args:
    predicates: a list of bool scalar tensors
    actions: a list of callable objects which return tensors.
  Returns:
    a callable
  """
  k = len(predicates) - 1  # could pick any
  action = actions[k]
  other_predicates, other_actions = predicates[:k], actions[:k]

  def default_action():
    return action()

  return default_action, other_predicates, other_actions


def _case_verify_and_canonicalize_args(pred_fn_pairs, exclusive, name,
                                       allow_python_preds):
  """Verifies input arguments for the case function.

  Args:
    pred_fn_pairs: Dict or list of pairs of a boolean scalar tensor, and a
      callable which returns a list of tensors.
    exclusive: True iff at most one predicate is allowed to evaluate to `True`.
    name: A name for the case operation.
    allow_python_preds: if true, pred_fn_pairs may contain Python bools in
      addition to boolean Tensors
  Raises:
    TypeError: If `pred_fn_pairs` is not a list/dictionary.
    TypeError: If `pred_fn_pairs` is a list but does not contain 2-tuples.
    TypeError: If `fns[i]` is not callable for any i, or `default` is not
               callable.
  Returns:
    a tuple <list of scalar bool tensors, list of callables>.
  """
  del name
  if not isinstance(pred_fn_pairs, (list, tuple, dict)):
    raise TypeError('fns must be a list, tuple, or dict')

  if isinstance(pred_fn_pairs, collections.OrderedDict):
    pred_fn_pairs = pred_fn_pairs.items()
  elif isinstance(pred_fn_pairs, dict):
    # No name to sort on in eager mode. Use dictionary traversal order,
    # which is nondeterministic in versions of Python < 3.6
    if not exclusive:
      raise ValueError('Unordered dictionaries are not supported for the '
                       '`pred_fn_pairs` argument when `exclusive=False` and '
                       'eager mode is enabled.')
    pred_fn_pairs = list(pred_fn_pairs.items())
  for pred_fn_pair in pred_fn_pairs:
    if not isinstance(pred_fn_pair, tuple) or len(pred_fn_pair) != 2:
      raise TypeError('Each entry in pred_fn_pairs must be a 2-tuple')
    pred, fn = pred_fn_pair

    if ops.is_tensor(pred):
      if pred.dtype != dtype.bool:
        raise TypeError('pred must be Tensor of type bool: %s' % pred.name)
    elif not allow_python_preds:
      raise TypeError('pred must be a Tensor, got: %s' % pred)
    elif not isinstance(pred, bool):
      raise TypeError('pred must be a Tensor or bool, got: %s' % pred)

    if not callable(fn):
      raise TypeError('fn for pred %s must be callable.' % pred.name)

  predicates, actions = zip(*pred_fn_pairs)
  return predicates, actions


def _case_helper(cond_fn,
                 pred_fn_pairs,
                 default,
                 exclusive,
                 name,
                 allow_python_preds=False,
                 **cond_kwargs):
  """Implementation of case that allows for different cond functions.

  Args:
    cond_fn: method that has signature and semantics of `cond` above.
    pred_fn_pairs: Dict or list of pairs of a boolean scalar tensor, and a
      callable which returns a list of tensors.
    default: Optional callable that returns a list of tensors.
    exclusive: True iff at most one predicate is allowed to evaluate to `True`.
    name: A name for this operation (optional).
    allow_python_preds: if true, pred_fn_pairs may contain Python bools in
      addition to boolean Tensors
    **cond_kwargs: keyword arguments that will be passed to `cond_fn`.
  Returns:
    The tensors returned by the first pair whose predicate evaluated to True, or
    those returned by `default` if none does.
  Raises:
    TypeError: If `pred_fn_pairs` is not a list/dictionary.
    TypeError: If `pred_fn_pairs` is a list but does not contain 2-tuples.
    TypeError: If `fns[i]` is not callable for any i, or `default` is not
               callable.
  """
  predicates, actions = _case_verify_and_canonicalize_args(
      pred_fn_pairs, exclusive, name, allow_python_preds)
  if default is None:
    default, predicates, actions = _case_create_default_action(
        predicates, actions)
  fn = default
  # To eval conditions in direct order we create nested conditions in reverse:
  #   cond_fn(c[0], true_fn=.., false_fn=cond_fn(c[1], ...))
  for predicate, action in reversed(list(zip(predicates, actions))):
    fn = functools.partial(
        cond_fn, predicate, true_fn=action, false_fn=fn, **cond_kwargs)
  return fn()


def with_dependencies(deps, value):
  del deps
  return value


# --- Begin Public Functions --------------------------------------------------

cond = utils.copy_docstring(
    'tf.cond',
    _cond_jax if JAX_MODE else _cond)

group = utils.copy_docstring(
    'tf.group',
    lambda *inputs, **kwargs: None)

no_op = utils.copy_docstring(
    'tf.no_op',
    _no_op)

while_loop = utils.copy_docstring(
    'tf.while_loop',
    _while_loop_jax if JAX_MODE else _while_loop)
