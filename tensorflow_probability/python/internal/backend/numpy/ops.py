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
"""Numpy implementations of TensorFlow functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

# Dependency imports
import numpy as np
import numpy as onp  # Avoid JAX rewrite.  # pylint: disable=reimported
import six

# TODO(b/151669121): Remove remaining TF imports
import tensorflow.compat.v1 as tf1
import tensorflow.compat.v2 as tf

from tensorflow_probability.python.internal.backend.numpy import _utils as utils
import wrapt
from tensorflow.python.ops.unconnected_gradients import UnconnectedGradients  # pylint: disable=g-direct-tensorflow-import

__all__ = [
    'bitcast',
    'broadcast_dynamic_shape',
    'broadcast_static_shape',
    'broadcast_to',
    'cast',
    'clip_by_value',
    'constant',
    'control_dependencies',
    'convert_to_tensor',
    'custom_gradient',
    'device',
    'dimension_value',
    'enable_v2_behavior',
    'executing_eagerly',
    'get_static_value',
    'group',
    'identity',
    'is_tensor',
    'name_scope',
    'newaxis',
    'stop_gradient',
    'GradientTape',
    'Module',
    'Tensor',
    'TensorShape',
    'Variable',
    # 'gradients',
]


JAX_MODE = False


class _NullContext(object):

  def __init__(self, *args, **kwargs):
    pass

  def __enter__(self):
    pass

  def __exit__(self, type_arg, value_arg, traceback_arg):
    return False  # False values do not suppress exceptions.


def _base_broadcast_static_shape(
    shape_x, shape_y, as_tensorshape=False, static_shape=False):
  shape_x = TensorShape(shape_x)
  shape_y = TensorShape(shape_y)
  shape_xy = tf.broadcast_static_shape(shape_x, shape_y)
  if as_tensorshape:
    return shape_xy
  if static_shape:
    return onp.array(shape_xy.as_list(), dtype=onp.int32)
  return np.array(shape_xy.as_list(), dtype=np.int32)


def _broadcast_static_shape(shape_x, shape_y):
  return _base_broadcast_static_shape(shape_x, shape_y, static_shape=True)


def _broadcast_dynamic_shape(shape_x, shape_y):
  return _base_broadcast_static_shape(shape_x, shape_y, static_shape=False)


def _constant(value, dtype=None, shape=None, name='Const'):  # pylint: disable=unused-argument
  x = convert_to_tensor(value, dtype=dtype)
  if shape is None:
    return x
  if not x.shape:
    return np.full(shape, x)
  return np.reshape(x, shape)


def _control_dependencies(control_inputs):
  if control_inputs:
    for control in control_inputs:
      if callable(control):
        control()
  return _NullContext()


def _convert_to_tensor(value, dtype=None, dtype_hint=None, name=None):  # pylint: disable=unused-argument
  """Emulates tf.convert_to_tensor."""
  assert not tf.is_tensor(value), value
  if is_tensor(value):
    if dtype is not None:
      dtype = utils.numpy_dtype(dtype)
      # if np.result_type(value, dtype) != dtype:
      #   raise ValueError('Expected dtype {} but got {} with dtype {}.'.format(
      #       dtype, value, value.dtype))
      return value.astype(dtype)
    return value
  if isinstance(value, Dimension):
    value = _dimension_value(value)
  elif isinstance(value, TensorShape):
    value = value.as_list()
  # In JAX mode, onp.ndarray/onp.generic are not identified as Tensor's.
  # By default, use the dtype of the values passed in.
  elif hasattr(value, 'dtype'):
    if dtype is not None:
      dtype = utils.numpy_dtype(dtype)
      return np.array(value).astype(dtype)
    return np.array(value)
  if dtype is None and dtype_hint is not None:
    dtype_hint = utils.numpy_dtype(dtype_hint)
    value = np.array(value)
    if np.size(value):
      # Match TF behavior, which won't downcast e.g. float to int.
      if np.issubdtype(value.dtype, np.complexfloating):
        if not np.issubdtype(dtype_hint, np.complexfloating):
          return value
      if np.issubdtype(value.dtype, np.floating):
        if not (np.issubdtype(dtype_hint, np.floating)
                or np.issubdtype(dtype_hint, np.complexfloating)):
          return value
      if np.issubdtype(value.dtype, np.integer):
        if not (np.issubdtype(dtype_hint, np.integer)
                or np.issubdtype(dtype_hint, np.floating)
                or np.issubdtype(dtype_hint, np.complexfloating)):
          return value
    return value.astype(dtype_hint)

  np_value = np.array(value, dtype=utils.numpy_dtype(dtype or dtype_hint))
  if np.issubdtype(np_value.dtype, np.object_):
    raise ValueError('Numpy `object`s cannot be converted to `Tensor`s.')
  # We have no hints. By default JAX (in x64 mode) and Numpy default to
  # {int64,float64} which does not match with TF's default.
  if dtype is None and dtype_hint is None:
    # If the integer doesn't fit in int32, return an int64. This matches TF.
    if isinstance(value, int):
      if value > onp.iinfo(onp.int32).max or value < onp.iinfo(onp.int32).min:
        return np.array(value, dtype=np.int64)
    if np.issubdtype(np_value.dtype, np.floating):
      return np_value.astype(np.float32)
    if np.issubdtype(np_value.dtype, np.integer):
      return np_value.astype(np.int32)
  return np_value


def _dimension_value(dimension):
  if dimension is None:
    return None
  return int(dimension)


# --- Begin Public Functions --------------------------------------------------

dimension_value = utils.copy_docstring(
    'tf.compat.dimension_value',
    _dimension_value)


class GradientTape(object):
  """tf.GradientTape stub."""

  def __init__(self, persistent=False, watch_accessed_variables=True):  # pylint: disable=unused-argument
    pass

  def __enter__(self):
    return self

  def __exit__(self, typ, value, traceback):  # pylint: disable=unused-argument
    pass

  def watch(self, tensor):  # pylint: disable=unused-argument
    pass

  def gradient(self, target, sources, output_gradients=None,  # pylint: disable=unused-argument
               unconnected_gradients=UnconnectedGradients.NONE):  # pylint: disable=unused-argument
    return sources

  def batch_jacobian(self, target, source,  # pylint: disable=unused-argument
                     unconnected_gradients=UnconnectedGradients.NONE,  # pylint: disable=unused-argument
                     parallel_iterations=None, experimental_use_pfor=True):  # pylint: disable=unused-argument
    return source

bitcast = utils.copy_docstring(
    'tf.bitcast',
    lambda input, type, name=None: convert_to_tensor(  # pylint: disable=g-long-lambda
        input, dtype_hint=type).view(type))

broadcast_dynamic_shape = utils.copy_docstring(
    'tf.broadcast_static_shape', _broadcast_dynamic_shape)

broadcast_static_shape = utils.copy_docstring(
    'tf.broadcast_static_shape', _broadcast_static_shape)

broadcast_static_shape_as_tensorshape = utils.copy_docstring(
    'tf.broadcast_static_shape',
    functools.partial(_base_broadcast_static_shape, as_tensorshape=True))

broadcast_to = utils.copy_docstring(
    'tf.broadcast_to',
    lambda input, shape, name=None: np.broadcast_to(input, shape))

cast = utils.copy_docstring(
    'tf.cast',
    lambda x, dtype, name=None: np.array(x, dtype=utils.numpy_dtype(dtype)))

clip_by_value = utils.copy_docstring(
    'tf.clip_by_value',
    lambda t, clip_value_min, clip_value_max, name=None:  # pylint: disable=g-long-lambda
    np.clip(t, clip_value_min, clip_value_max))

constant = utils.copy_docstring(
    'tf.constant',
    _constant)

control_dependencies = utils.copy_docstring(
    'tf.control_dependencies',
    _control_dependencies)

convert_to_tensor = utils.copy_docstring(
    'tf.convert_to_tensor',
    _convert_to_tensor)


def _custom_gradient(f):
  """Jax implementation of tf.custom_gradient."""
  if not JAX_MODE:
    # Numpy backend ignores custom gradients, so we do too.
    return lambda *args, **kwargs: f(*args, **kwargs)[0]
  def f_(*args, **kwargs):
    value, vjp = f(*args, **kwargs)
    def vjp_(cts_out):
      cts_in = vjp(cts_out)
      if isinstance(cts_in, list):
        cts_in = tuple(cts_in)
      elif not isinstance(cts_in, tuple):
        cts_in = (cts_in,)
      return cts_in
    return value, vjp_
  @jax.custom_transforms
  def wrapped(*args, **kwargs):
    value, _ = f(*args, **kwargs)
    return value
  jax.defvjp_all(wrapped, f_)
  return wrapped

custom_gradient = utils.copy_docstring(
    'tf.custom_gradient', _custom_gradient)

device = lambda _: _NullContext()

executing_eagerly = utils.copy_docstring(
    'tf.executing_eagerly',
    lambda: True)


def _get_static_value_jax(tensor, partial=False):
  del partial
  if isinstance(tensor, jax.core.Tracer):
    return None
  if isinstance(tensor, np.ndarray):
    return onp.array(tensor)
  return tensor

get_static_value = utils.copy_docstring(
    'tf.get_static_value',
    _get_static_value_jax if JAX_MODE else
    lambda tensor, partial=False: tensor)

group = utils.copy_docstring(
    'tf.group',
    lambda *inputs, **kwargs: None)

identity = utils.copy_docstring(
    'tf.identity',
    lambda input, name=None: np.array(input))

is_tensor = utils.copy_docstring(
    'tf.is_tensor',
    lambda x: isinstance(x, Tensor))


class name_scope(object):  # pylint: disable=invalid-name
  """A context manager for use when defining a Python op.

  This context manager pushes a name scope, which will make the name of all
  operations added within it have a prefix.

  For example, to define a new Python op called `my_op`:

  ```python
  def my_op(a, b, c, name=None):
    with tf.name_scope("MyOp") as scope:
      a = tf.convert_to_tensor(a, name="a")
      b = tf.convert_to_tensor(b, name="b")
      c = tf.convert_to_tensor(c, name="c")
      # Define some computation that uses `a`, `b`, and `c`.
      return foo_op(..., name=scope)
  ```

  When executed, the Tensors `a`, `b`, `c`, will have names `MyOp/a`, `MyOp/b`,
  and `MyOp/c`.

  If the scope name already exists, the name will be made unique by appending
  `_n`. For example, calling `my_op` the second time will generate `MyOp_1/a`,
  etc.
  """

  @property
  def name(self):
    return self._name

  def __init__(self, name, *args, **kwargs):
    del args, kwargs
    self._name = name

  def __enter__(self):
    return self._name

  def __exit__(self, type_arg, value_arg, traceback_arg):
    return False  # False values do not suppress exceptions.


newaxis = np.newaxis

if JAX_MODE:
  from jax import lax  # pylint: disable=g-import-not-at-top
  stop_gradient = utils.copy_docstring(
      'tf.stop_gradient',
      lambda input, name=None: lax.stop_gradient(input))
else:
  stop_gradient = utils.copy_docstring(
      'tf.stop_gradient',
      lambda input, name=None: np.array(input))

TensorShape = tf.TensorShape
Dimension = tf1.Dimension


def dimension_at_index(shape, index):
  if isinstance(shape, TensorShape):
    return shape.dims[index]
  return Dimension(int(shape[index]))


class NumpyVariable(wrapt.ObjectProxy):
  """Stand-in for tf.Variable."""

  __slots__ = ('initializer',)

  # pylint: disable=unused-argument
  def __init__(
      self,
      initial_value=None,
      trainable=True,
      validate_shape=True,
      caching_device=None,
      name=None,
      variable_def=None,
      dtype=None,
      import_scope=None,
      constraint=None,
      shape=None):
    assert constraint is None
    v = convert_to_tensor(initial_value)
    if dtype is not None:
      v = v.astype(utils.numpy_dtype(dtype))
    super(NumpyVariable, self).__init__(v)
    self._self_name = name
    self.initializer = None
  # pylint: enable=unused-argument

  @property
  def name(self):
    return self._self_name if self._self_name is not None else str(id(self))

  def __array__(self, dtype=None):
    if dtype is not None:
      dtype = utils.numpy_dtype(dtype)
      return self.__wrapped__.__array__(dtype)
    # Passing in dtype=None to __array__ has differing behavior in numpy.
    # When an `np.ndarray` has `.__array__(None)` invoked, the array is casted
    # to `float64`. Thus we handle this case separately.
    return self.__wrapped__.__array__()

  def assign(self, value):
    super(NumpyVariable, self).__init__(onp.array(value, dtype=self.dtype))
    return self

  def assign_add(self, value):
    super(NumpyVariable, self).__init__(
        onp.array(self, dtype=self.dtype) + onp.array(value, dtype=self.dtype))
    return self

  def assign_sub(self, value):
    super(NumpyVariable, self).__init__(
        onp.array(self, dtype=self.dtype) - onp.array(value, dtype=self.dtype))
    return self


if JAX_MODE:
  import jax  # pylint: disable=g-import-not-at-top
  jax.interpreters.xla.canonicalize_dtype_handlers[NumpyVariable] = (
      jax.interpreters.xla.canonicalize_dtype_handlers[onp.ndarray])
  jax.interpreters.xla.pytype_aval_mappings[NumpyVariable] = (
      jax.interpreters.xla.pytype_aval_mappings[onp.ndarray])
  jax.core.pytype_aval_mappings[NumpyVariable] = (
      jax.core.pytype_aval_mappings[onp.ndarray])


Variable = NumpyVariable


class _TensorMeta(type(np.ndarray)):

  @classmethod
  def __instancecheck__(cls, instance):
    if JAX_MODE:
      return isinstance(instance, (jax.xla.DeviceArray,
                                   jax.abstract_arrays.UnshapedArray,
                                   jax.core.Tracer))
    return isinstance(instance, np.ndarray)


class Tensor(six.with_metaclass(_TensorMeta)):
  OVERLOADABLE_OPERATORS = ()


class Module(object):
  """tf.Module."""

  _TF_MODULE_IGNORED_PROPERTIES = frozenset()

  def __init__(self, name):
    self._name = name

  def _no_dependency(self, x):
    return x

  @property
  def trainable_variables(self):
    return []

  @property
  def variables(self):
    return []


enable_v2_behavior = lambda: None
