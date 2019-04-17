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
"""Operations that use static values when possible."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
import decorator
import numpy as np

import tensorflow.compat.v2 as tf

from tensorflow_probability.python.internal import tensorshape_util
from tensorflow.python import pywrap_tensorflow as c_api  # pylint: disable=g-direct-tensorflow-import
from tensorflow.python.ops import control_flow_ops  # pylint: disable=g-direct-tensorflow-import
from tensorflow.python.util import tf_inspect  # pylint: disable=g-direct-tensorflow-import


def _maybe_get_static_args(args):
  flat_args = tf.nest.flatten(args)
  flat_args_ = [tf.get_static_value(a) for a in flat_args]
  all_static = all(arg is None or arg_ is not None
                   for arg, arg_ in zip(flat_args, flat_args_))
  return tf.nest.pack_sequence_as(args, flat_args_), all_static


def _prefer_static(original_fn, static_fn):
  """Wraps original_fn, preferring to call static_fn when inputs are static."""
  original_spec = tf_inspect.getfullargspec(original_fn)
  static_spec = tf_inspect.getfullargspec(static_fn)
  if original_spec != static_spec:
    raise ValueError(
        'Arg specs do not match: original={}, static={}, fn={}'.format(
            original_spec, static_spec, original_fn))
  @decorator.decorator
  def wrap(wrapped_fn, *args, **kwargs):
    del wrapped_fn
    [args_, kwargs_], all_static = _maybe_get_static_args([args, kwargs])
    if all_static:
      return static_fn(*args_, **kwargs_)
    return original_fn(*args, **kwargs)
  return wrap(original_fn)


def _copy_docstring(original_fn, new_fn):
  """Wraps new_fn with the doc of original_fn."""
  original_spec = tf_inspect.getfullargspec(original_fn)
  new_spec = tf_inspect.getfullargspec(new_fn)
  if original_spec != new_spec:
    raise ValueError(
        'Arg specs do not match: original={}, new={}, fn={}'.format(
            original_spec, new_spec, original_fn))
  @decorator.decorator
  def wrap(wrapped_fn, *args, **kwargs):
    del wrapped_fn
    return new_fn(*args, **kwargs)
  return wrap(original_fn)


def _numpy_dtype(dtype):
  return dtype.as_numpy_dtype if hasattr(dtype, 'as_numpy_dtype') else dtype


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
    raise TypeError('`pred` must be a Tensor, or a Python bool, or 1 or 0. '
                    'Found instead: {}'.format(pred))
  return pred_value


def rank_from_shape(shape_tensor_fn, tensorshape=None):
  """Computes `rank` given a `Tensor`'s `shape`."""

  if tensorshape is None:
    shape_tensor = (shape_tensor_fn() if callable(shape_tensor_fn)
                    else shape_tensor_fn)
    if (hasattr(shape_tensor, 'shape') and
        hasattr(shape_tensor.shape, 'num_elements')):
      ndims_ = tensorshape_util.num_elements(shape_tensor.shape)
    else:
      ndims_ = len(shape_tensor)
    ndims_fn = lambda: tf.size(input=shape_tensor)
  else:
    ndims_ = tensorshape_util.rank(tensorshape)
    ndims_fn = lambda: tf.size(input=shape_tensor_fn()  # pylint: disable=g-long-lambda
                               if callable(shape_tensor_fn)
                               else shape_tensor_fn)
  return ndims_fn() if ndims_ is None else ndims_


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
    raise TypeError('`true_fn` must be callable.')
  if not callable(false_fn):
    raise TypeError('`false_fn` must be callable.')

  pred_value = _get_static_predicate(pred)
  if pred_value is not None:
    if pred_value:
      return true_fn()
    else:
      return false_fn()
  else:
    return tf.cond(pred=pred, true_fn=true_fn, false_fn=false_fn, name=name)


def case(pred_fn_pairs, default=None, exclusive=False, name='smart_case'):
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

# The following functions are intended as drop-in replacements for their
# TensorFlow counterparts.

concat = _prefer_static(
    tf.concat,
    lambda values, axis, name='concat': np.concatenate(values, axis))

equal = _prefer_static(
    tf.equal,
    lambda x, y, name=None: np.equal(x, y))

greater = _prefer_static(
    tf.greater,
    lambda x, y, name=None: np.greater(x, y))

less = _prefer_static(
    tf.less,
    lambda x, y, name=None: np.less(x, y))

logical_and = _prefer_static(
    tf.logical_and,
    lambda x, y, name=None: np.logical_and(x, y))

logical_not = _prefer_static(
    tf.logical_not,
    lambda x, name=None: np.logical_not(x))

logical_or = _prefer_static(
    tf.logical_or,
    lambda x, y, name=None: np.logical_or(x, y))

maximum = _prefer_static(
    tf.maximum,
    lambda x, y, name=None: np.maximum(x, y))

minimum = _prefer_static(
    tf.minimum,
    lambda x, y, name=None: np.minimum(x, y))

ones = _prefer_static(
    tf.ones,
    lambda shape, dtype=tf.float32, name=None: np.ones(  # pylint: disable=g-long-lambda
        shape, _numpy_dtype(dtype)))


def _ones_like(input, dtype=None, name=None):  # pylint: disable=redefined-builtin
  s = _shape(input)
  if isinstance(s, (np.ndarray, np.generic)):
    return np.ones(s, _numpy_dtype(dtype or input.dtype))
  return tf.ones(s, dtype or s.dtype, name)
ones_like = _copy_docstring(tf.ones_like, _ones_like)

range = _prefer_static(  # pylint: disable=redefined-builtin
    tf.range,
    lambda start, limit=None, delta=1, dtype=None, name='range': np.arange(  # pylint: disable=g-long-lambda
        start, limit, delta, _numpy_dtype(dtype)))

rank = _copy_docstring(
    tf.rank,
    lambda input, name=None: (  # pylint: disable=redefined-builtin,g-long-lambda
        tf.rank(input=input)
        if tensorshape_util.rank(input.shape) is None
        else tensorshape_util.rank(input.shape)))

reduce_all = _prefer_static(
    tf.reduce_all,
    lambda input_tensor, axis=None, keepdims=False, name=None: np.all(  # pylint: disable=g-long-lambda
        input_tensor, axis, keepdims=keepdims))

reduce_any = _prefer_static(
    tf.reduce_any,
    lambda input_tensor, axis=None, keepdims=False, name=None: np.any(  # pylint: disable=g-long-lambda
        input_tensor, axis, keepdims=keepdims))

reduce_prod = _prefer_static(
    tf.reduce_prod,
    lambda input_tensor, axis=None, keepdims=False, name=None: np.prod(  # pylint: disable=g-long-lambda
        input_tensor, axis, keepdims=keepdims))

reduce_sum = _prefer_static(
    tf.reduce_sum,
    lambda input_tensor, axis=None, keepdims=False, name=None: np.sum(  # pylint: disable=g-long-lambda
        input_tensor, axis, keepdims=keepdims))


def _shape(input, out_type=tf.int32, name=None):  # pylint: disable=redefined-builtin
  if not hasattr(input, 'shape'):
    x = np.array(input)
    input = tf.convert_to_tensor(value=input) if x.dtype is np.object else x
  input_shape = tf.TensorShape(input.shape)
  if tensorshape_util.is_fully_defined(input.shape):
    return np.array(tensorshape_util.as_list(input_shape)).astype(
        _numpy_dtype(out_type))
  return tf.shape(input=input, out_type=out_type, name=name)
shape = _copy_docstring(tf.shape, _shape)

where = _prefer_static(
    tf.where,
    lambda condition, x=None, y=None, name=None: np.where(condition, x, y))

zeros = _prefer_static(
    tf.zeros,
    lambda shape, dtype=tf.float32, name=None: np.zeros(  # pylint: disable=g-long-lambda
        shape, _numpy_dtype(dtype)))


def _zeros_like(input, dtype=None, name=None):  # pylint: disable=redefined-builtin
  s = _shape(input)
  if isinstance(s, (np.ndarray, np.generic)):
    return np.zeros(s, _numpy_dtype(dtype or input.dtype))
  return tf.zeros(s, dtype or s.dtype, name)
zeros_like = _copy_docstring(tf.zeros_like, _zeros_like)
