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
"""Operations that use static values when available."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

import numpy as np
import tensorflow as tf

from tensorflow.python import pywrap_tensorflow as c_api  # pylint: disable=g-direct-tensorflow-import
from tensorflow.python.ops import control_flow_ops  # pylint: disable=g-direct-tensorflow-import


def _maybe_get_static_args(args):
  maybe_static_args = tf.compat.v2.nest.map_structure(tf.get_static_value, args)
  all_static = all([arg is not None
                    for arg in tf.compat.v2.nest.flatten(maybe_static_args)])
  return maybe_static_args, all_static


def _prefer_static(static_fn):
  def decorator(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
      [args_, kwargs_], all_static = _maybe_get_static_args([args, kwargs])
      if all_static:
        return tf.constant(static_fn(*args_, **kwargs_))
      return fn(*args, **kwargs)
    return wrapper
  return decorator


# The operations below function as drop-in replacements for their standard
# TensorFlow counterparts. However, they will attempt to statically evaluate
# arguments and apply the given `static_fn` instead of the TensorFlow operation.

greater = _prefer_static(lambda x, y, *_, **__: (x > y))(tf.greater)
less = _prefer_static(lambda x, y, *_, **__: (x < y))(tf.less)
equal = _prefer_static(lambda x, y, *_, **__: (x == y))(tf.equal)
logical_and = _prefer_static(lambda x, y, *_, **__: (x and y))(tf.logical_and)
logical_or = _prefer_static(lambda x, y, *_, **__: (x or y))(tf.logical_or)


def _static_all(input_tensor, axis=None, keepdims=False, name=None):
  del name  # unused
  return np.all(a=input_tensor, axis=axis, keepdims=keepdims)
reduce_all = _prefer_static(_static_all)(tf.reduce_all)


def _static_any(input_tensor, axis=None, keepdims=False, name=None):
  del name  # unused
  return np.any(a=input_tensor, axis=axis, keepdims=keepdims)
reduce_any = _prefer_static(_static_any)(tf.reduce_any)


def _get_static_predicate(pred):
  """Helper function for statically evaluating predicates in `cond`."""
  if pred in {0, 1}:  # Accept 1/0 as valid boolean values
    pred_value = bool(pred)
  elif isinstance(pred, bool):
    pred_value = pred
  elif isinstance(pred, tf.Tensor):
    pred_value = tf.get_static_value(pred)

    # TODO(jamieas): remove the dependency on `pywrap_tensorflow`.
    # pylint: disable=protected-access
    if pred_value is None:
      pred_value = c_api.TF_TryEvaluateConstant_wrapper(pred.graph._c_graph,
                                                        pred._as_tf_output())
    # pylint: enable=protected-access

  else:
    raise TypeError("`pred` must be a Tensor, or a Python bool, or 1 or 0. "
                    "Found instead: %s" % pred)
  return pred_value


def cond(pred, true_fn=None, false_fn=None, name=None):
  """Return either `true_fn()` if predicate `pred` is true else `false_fn()`.

  If `pred` is a bool or has a constant value, we return either `true_fn()`
  or `false_fn()`, otherwise we use `tf.cond` to dynamically route to both.

  Arguments:
    pred: A scalar determining whether to return the result of `true_fn` or
      `false_fn`.
    true_fn: The callable to be performed if pred is true.
    false_fn: The callable to be performed if pred is false.
    name: Optional name prefix when using `tf.cond`.

  Returns:
    Tensors returned by the call to either `true_fn` or `false_fn`.

  Raises:
    TypeError: If `true_fn` or `false_fn` is not callable.
  """
  if not callable(true_fn):
    raise TypeError("`true_fn` must be callable.")
  if not callable(false_fn):
    raise TypeError("`false_fn` must be callable.")

  pred_value = _get_static_predicate(pred)
  if pred_value is not None:
    if pred_value:
      return true_fn()
    else:
      return false_fn()
  else:
    return tf.cond(pred=pred, true_fn=true_fn, false_fn=false_fn, name=name)


def case(pred_fn_pairs, default=None, exclusive=False, name="smart_case"):
  """Like tf.case, except attempts to statically evaluate predicates.

  If any predicate in `pred_fn_pairs` is a bool or has a constant value, the
  associated callable will be called or omitted depending on its value.
  Otherwise this functions like tf.case.

  Args:
    pred_fn_pairs: Dict or list of pairs of a boolean scalar tensor and a
                   callable which returns a list of tensors.
    default: Optional callable that returns a list of tensors.
    exclusive: True iff at most one predicate is allowed to evaluate to `True`.
    name: A name for this operation (optional).

  Returns:
    The tensors returned by the first pair whose predicate evaluated to True, or
    those returned by `default` if none does.

  Raises:
    TypeError: If `pred_fn_pairs` is not a list/dictionary.
    TypeError: If `pred_fn_pairs` is a list but does not contain 2-tuples.
    TypeError: If `fns[i]` is not callable for any i, or `default` is not
               callable.
  """
  return control_flow_ops._case_helper(  # pylint: disable=protected-access
      cond, pred_fn_pairs, default, exclusive, name, allow_python_preds=True)
