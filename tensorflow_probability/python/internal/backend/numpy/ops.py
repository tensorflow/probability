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

import contextlib
import functools

# Dependency imports
import numpy as np
import numpy as onp  # Avoid JAX rewrite.  # pylint: disable=reimported
import six

from tensorflow_probability.python.internal.backend.numpy import _utils as utils
from tensorflow_probability.python.internal.backend.numpy import nest
from tensorflow_probability.python.internal.backend.numpy.gen import tensor_shape
try:  # May not be available, not a core dep for TFP.
  import wrapt  # pylint: disable=g-import-not-at-top
except ImportError:
  wrapt = None

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
    'enable_v2_behavior',
    'ensure_shape',
    'executing_eagerly',
    'get_static_value',
    'identity',
    'init_scope',
    'is_tensor',
    'name_scope',
    'newaxis',
    'recompute_grad',
    'register_tensor_conversion_function',
    'stop_gradient',
    'GradientTape',
    'Module',
    'Tensor',
    'Variable',
    # 'gradients',
]


JAX_MODE = False


if JAX_MODE:
  import jax  # pylint: disable=g-import-not-at-top


class _NullContext(object):

  def __init__(self, *args, **kwargs):
    pass

  def __enter__(self):
    pass

  def __exit__(self, type_arg, value_arg, traceback_arg):
    return False  # False values do not suppress exceptions.


def _broadcast_static_shape(shape_x, shape_y):
  """Reimplements `tf.broadcast_static_shape` in JAX/NumPy."""
  if (tensor_shape.TensorShape(shape_x).ndims is None or
      tensor_shape.TensorShape(shape_y).ndims is None):
    return tensor_shape.TensorShape(None)
  shape_x = tuple(tensor_shape.TensorShape(shape_x).as_list())
  shape_y = tuple(tensor_shape.TensorShape(shape_y).as_list())
  try:
    if JAX_MODE:
      error_message = 'Incompatible shapes for broadcasting'
      return tensor_shape.TensorShape(lax.broadcast_shapes(shape_x, shape_y))
    error_message = ('shape mismatch: objects cannot be broadcast to'
                     ' a single shape')
    return tensor_shape.TensorShape(
        np.broadcast(np.zeros(shape_x), np.zeros(shape_y)).shape)
  except ValueError as e:
    # Match TF error message
    if error_message in str(e):
      raise ValueError(
          'Incompatible shapes for broadcasting: {} and {}'.format(
              shape_x, shape_y))
    raise


def _broadcast_dynamic_shape(shape_x, shape_y):
  """Reimplements `tf.broadcast_dynamic_shape` in JAX/NumPy."""
  return convert_to_tensor(_broadcast_static_shape(shape_x, shape_y))

broadcast_shape = _broadcast_static_shape


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


tensor_conversion_registry = {}


def register_tensor_conversion_function(base_type, conversion_func):
  # No priority system like TensorFlow yet
  tensor_conversion_registry[base_type] = conversion_func


def _convert_to_tensor(value, dtype=None, dtype_hint=None, name=None):  # pylint: disable=unused-argument
  """Emulates tf.convert_to_tensor."""
  dtype = utils.numpy_dtype(dtype)
  dtype_hint = utils.numpy_dtype(dtype_hint)
  if is_tensor(value) and not isinstance(value, Variable):
    # In NumPy mode, we are lenient on the dtype compatibility check because
    # some codepaths rely on flexible conversion from int/float64 to 32.
    if dtype is not None and value.dtype != dtype:
      if JAX_MODE:
        raise TypeError(('Tensor conversion requested dtype {} for array with '
                         'dtype {}: {}').format(dtype, value.dtype, value))
      return value.astype(dtype)
    return value

  conversion_func = tensor_conversion_registry.get(type(value),
                                                   _default_convert_to_tensor)
  ret = None
  if dtype is None and dtype_hint is not None:
    try:
      ret = conversion_func(value, dtype=dtype_hint)
    except (TypeError, ValueError):
      pass

  if ret is None:
    ret = conversion_func(value, dtype=dtype)
  return ret


def _infer_dtype(value, default_dtype):
  """Guesses an object's dtype."""
  # Need to check for onp type first because onp types are subclasses of Python
  # types.
  if hasattr(value, 'dtype'):
    # Duck-typing onp types
    return value.dtype
  elif isinstance(value, bool):
    return np.bool_
  elif isinstance(value, six.integer_types):
    return np.int32
  elif isinstance(value, float):
    return np.float32
  elif isinstance(value, complex):
    return np.complex128
  elif isinstance(value, (tuple, list)):
    # Try inferring the type from items in the object if possible.
    for v in nest.flatten(value):
      if hasattr(v, 'dtype'):
        return v.dtype
    try:  # Finally fall back to raw types (int, bool).
      return _infer_dtype(value[0], default_dtype)
    except (IndexError, TypeError):
      return default_dtype
  raise ValueError(('Attempt to convert a value ({})'
                    ' with an unsupported type ({}) to a Tensor.').format(
                        value, type(value)))


class _Int64ToInt32Error(TypeError):
  """Error thrown when trying to convert an int64 to int32."""

  def __init__(self, int_value):
    self.int_value = int_value
    super(_Int64ToInt32Error, self).__init__('Overflow when casting an int64 to'
                                             ' an int32.')


class _FloatToIntError(TypeError):
  """Error thrown when trying to convert a float to an int."""


def _is_int64(value):
  return value > onp.iinfo(onp.int32).max or value < onp.iinfo(onp.int32).min


def _default_convert_to_tensor(value, dtype=None):
  """Default tensor conversion function for array, bool, int, float, and complex."""
  if JAX_MODE:
    # TODO(b/223267515): We shouldn't need to specialize here.
    if 'PRNGKeyArray' in str(type(value)):
      return value
    if isinstance(value, (list, tuple)) and value:
      if 'PRNGKeyArray' in str(type(value[0])):
        return np.stack(value, axis=0)

  inferred_dtype = _infer_dtype(value, np.float32)
  # When a dtype is provided, we can go ahead and try converting to the dtype
  # and force overflow/underflow if an int64 is converted to an int32.
  if dtype is not None:
    try:
      return _default_convert_to_tensor_with_dtype(value, dtype)
    except _Int64ToInt32Error as e:
      # Force conversion to int32 if requested
      return e.int_value
  # If no dtype is provided, we try the inferred dtype and fallback to int64 or
  # float32 depending on the type of conversion error we see.
  try:
    return _default_convert_to_tensor_with_dtype(value, inferred_dtype)
  except _Int64ToInt32Error as e:
    return np.array(value, dtype=np.int64)
  except _FloatToIntError as e:
    return np.array(value, dtype=np.float32)


class TypeConversionError(TypeError):

  def __init__(self, value, dtype):
    super(TypeConversionError, self).__init__(
        'Cannot convert {} to array of dtype {}'.format(value, dtype))


class MixedTypesError(ValueError):

  def __init__(self):
    super(MixedTypesError, self).__init__('Can\'t convert Python sequence with'
                                          ' mixed types to Tensor.')


def _default_convert_to_tensor_with_dtype(value, dtype,
                                          error_if_mismatch=False):
  """Converts a value to a tensor with a given dtype.

  Args:
    value: An object to be converted to tensor.
    dtype: A NPTF dtype.
    error_if_mismatch: Enables a stricter check for use when converting an
                       iterable from a tensor.
  Returns:
    A tensor.

  Raises:
    TypeConversionError: If type conversion fails.
    MixedTypesError: If types are mismatched in an iterable context.
    ValueError: If object isn't convertible to tensor.
    _Int64ToInt32Error: If trying to convert an int64 to an int32.
    _FloatToIntError: If trying to convert a float to an int.
  """
  is_arraylike = hasattr(value, 'dtype')
  if is_arraylike:
    # Duck-typed for `onp.array`/`onp.generic`
    arr = np.array(value)
    if dtype is not None:
      # arr.astype(None) forces conversion to float64
      return arr.astype(dtype)
    return arr
  elif isinstance(value, complex):
    dtype_compatible = np.issubdtype(dtype, np.complexfloating)
    if not dtype_compatible:
      if error_if_mismatch:
        raise MixedTypesError()
      raise TypeConversionError(value, dtype)
  elif isinstance(value, bool):
    # Bool check needs to happen before int check because bools are instances of
    # int.
    dtype_compatible = (dtype == np.bool_ or np.issubdtype(dtype, np.integer)
                        or np.issubdtype(dtype, np.floating))
    if not dtype_compatible:
      if error_if_mismatch:
        raise MixedTypesError()
      raise TypeError(value, dtype)
  elif isinstance(value, six.integer_types):
    if error_if_mismatch and not (np.issubdtype(dtype, np.integer)
                                  or np.issubdtype(dtype, np.floating)):
      raise MixedTypesError()
    if dtype == np.int32 and _is_int64(value):
      raise _Int64ToInt32Error(np.array(value, dtype=dtype))
    if dtype == np.bool_:
      # Can't downcast an int to a bool
      raise TypeConversionError(value, dtype)
  elif isinstance(value, float):
    if error_if_mismatch and not (np.issubdtype(dtype, np.integer)
                                  or np.issubdtype(dtype, np.floating)):
      raise MixedTypesError()
    if np.issubdtype(dtype, np.integer):
      raise _FloatToIntError(
          'Cannot convert {} to array of dtype {}'.format(value, dtype))
    if not (np.issubdtype(dtype, np.floating)
            or np.issubdtype(dtype, np.complexfloating)):
      raise TypeConversionError(value, dtype)
  else:
    # Try to iterate through object and throw ValueError if we can't.
    if hasattr(value, '__getitem__'):
      ret = []
      error_in_list = False
      for v in value:
        ret.append(_default_convert_to_tensor_with_dtype(
            v, dtype, error_if_mismatch=error_in_list))
        error_in_list = True
      value = ret
    else:
      raise ValueError(
          ('Attempting to convert a value {} with an'
           ' unsupported type {} to a Tensor.').format(value, type(value)))
  return np.array(value, dtype=dtype)


@contextlib.contextmanager
def _init_scope():
  yield


# --- Begin Public Functions --------------------------------------------------


class GradientTape(object):
  """tf.GradientTape stub."""

  def __init__(self, persistent=False, watch_accessed_variables=True):  # pylint: disable=unused-argument
    raise NotImplementedError('GradientTape not currently supported in JAX and '
                              'NumPy backends.')

  def __enter__(self):
    return self

  def __exit__(self, typ, value, traceback):  # pylint: disable=unused-argument
    pass

  def watch(self, tensor):  # pylint: disable=unused-argument
    pass

  def gradient(self, target, sources, output_gradients=None,  # pylint: disable=unused-argument
               unconnected_gradients=None):  # pylint: disable=unused-argument
    raise NotImplementedError

  def batch_jacobian(self, target, source,  # pylint: disable=unused-argument
                     unconnected_gradients=None,  # pylint: disable=unused-argument
                     parallel_iterations=None, experimental_use_pfor=True):  # pylint: disable=unused-argument
    raise NotImplementedError

bitcast = utils.copy_docstring(
    'tf.bitcast',
    lambda input, type, name=None: convert_to_tensor(  # pylint: disable=g-long-lambda
        input, dtype_hint=type).view(type))

broadcast_dynamic_shape = utils.copy_docstring(
    'tf.broadcast_dynamic_shape', _broadcast_dynamic_shape)

broadcast_static_shape = utils.copy_docstring(
    'tf.broadcast_static_shape', _broadcast_static_shape)

broadcast_to = utils.copy_docstring(
    'tf.broadcast_to',
    lambda input, shape, name=None: np.broadcast_to(input, shape))


def _cast(x, dtype):
  x = convert_to_tensor(x)
  if (np.issubdtype(x.dtype, np.complexfloating) and
      not np.issubdtype(dtype, np.complexfloating)):
    x = np.real(x)
  return x.astype(dtype)


cast = utils.copy_docstring(
    'tf.cast',
    lambda x, dtype, name=None: _cast(x, utils.numpy_dtype(dtype)))

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
  """JAX implementation of tf.custom_gradient."""
  if not JAX_MODE:
    # Numpy backend ignores custom gradients, so we do too.
    return lambda *args, **kwargs: f(*args, **kwargs)[0]

  @jax.custom_gradient
  @functools.wraps(f)
  def wrapped(*args, **kwargs):
    value, vjp = f(*args, **kwargs)
    def vjp_(cts_out):
      cts_in = vjp(cts_out)
      if isinstance(cts_in, list):
        cts_in = tuple(cts_in)
      return cts_in
    return value, vjp_
  return wrapped

custom_gradient = utils.copy_docstring(
    'tf.custom_gradient', _custom_gradient)

device = lambda _: _NullContext()


def _ensure_shape(x, shape, name=None):  # pylint: disable=unused-argument
  x_shape = tensor_shape.TensorShape(x.shape)
  shape = tensor_shape.TensorShape(shape)
  if not shape.is_compatible_with(x_shape):
    msg = 'Shape of tensor x {} is not compatible with expected shape {}'
    raise ValueError(msg.format(x_shape, shape))
  return x


ensure_shape = utils.copy_docstring(
    'tf.ensure_shape', _ensure_shape)

executing_eagerly = utils.copy_docstring(
    'tf.executing_eagerly',
    lambda: True)


def _get_static_value_jax(tensor, partial=False):
  """JAX implementation of tf.get_static_value."""
  del partial
  if isinstance(tensor, jax.core.Tracer):
    return None
  if isinstance(tensor, NumpyVariable):
    return None
  if isinstance(tensor, Module):
    return None
  if isinstance(tensor, (onp.ndarray, np.ndarray)):
    return onp.array(tensor)
  return tensor


def _get_static_value_numpy(tensor, partial=False):
  """NumPy implementation of tf.get_static_value."""
  del partial
  if isinstance(tensor, NumpyVariable):
    return None
  if isinstance(tensor, Module):
    return None
  return tensor

get_static_value = utils.copy_docstring(
    'tf.get_static_value',
    _get_static_value_jax if JAX_MODE else _get_static_value_numpy)

identity = utils.copy_docstring(
    'tf.identity',
    lambda input, name=None: np.array(input))

is_tensor = utils.copy_docstring(
    'tf.is_tensor',
    lambda x: isinstance(x, Tensor))

init_scope = utils.copy_docstring('tf.init_scope', _init_scope)


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
    if JAX_MODE and name is not None:
      jax_named_scope = jax.named_scope(name)
    else:
      jax_named_scope = None
    self._jax_named_scope = jax_named_scope

  def __enter__(self):
    if self._jax_named_scope is not None:
      self._jax_named_scope.__enter__()
    return self._name

  def __exit__(self, type_arg, value_arg, traceback_arg):
    if self._jax_named_scope is not None:
      self._jax_named_scope.__exit__(type_arg, value_arg, traceback_arg)
    return False  # False values do not suppress exceptions.


newaxis = np.newaxis

if JAX_MODE:
  from jax import remat  # pylint: disable=g-import-not-at-top
  recompute_grad = utils.copy_docstring(
      'tf.recompute_grad',
      remat)
else:
  recompute_grad = lambda x: x

if JAX_MODE:
  from jax import lax  # pylint: disable=g-import-not-at-top
  stop_gradient = utils.copy_docstring(
      'tf.stop_gradient',
      lambda input, name=None: lax.stop_gradient(input))
else:
  stop_gradient = utils.copy_docstring(
      'tf.stop_gradient',
      lambda input, name=None: np.array(input))


def _convert_tensorshape_to_tensor(value, dtype=None):
  """Copied from TF's TensorShape conversion."""
  if not value.is_fully_defined():
    raise ValueError(
        'Cannot convert a partially known TensorShape to a Tensor: {}'.format(
            value))
  value_list = value.as_list()
  int64_value = 0
  for dim in value_list:
    if dim >= 2**31:
      int64_value = dim
      break
  if dtype is not None:
    if dtype not in (np.int32, np.int64):
      raise TypeConversionError(value, dtype)
    if dtype == np.int32 and int64_value:
      raise ValueError('Cannot convert a TensorShape to dtype int32; '
                       'a dimension is too large ({})'.format(int64_value))
  else:
    dtype = np.int64 if int64_value else np.int32
  return convert_to_tensor(value_list, dtype=dtype)
register_tensor_conversion_function(tensor_shape.TensorShape,
                                    _convert_tensorshape_to_tensor)


def _convert_dimension_to_tensor(value, dtype=None):
  dtype = dtype or np.int32
  if dtype not in (np.int32, np.int64):
    raise TypeConversionError(value, dtype)
  return convert_to_tensor(tensor_shape.dimension_value(value), dtype=dtype)
register_tensor_conversion_function(tensor_shape.Dimension,
                                    _convert_dimension_to_tensor)


class NumpyVariable(getattr(wrapt, 'ObjectProxy', object)):
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

  def assign(self, value, **_):
    super(NumpyVariable, self).__init__(onp.array(value, dtype=self.dtype))
    return self

  def assign_add(self, value, **_):
    super(NumpyVariable, self).__init__(
        onp.array(self, dtype=self.dtype) + onp.array(value, dtype=self.dtype))
    return self

  def assign_sub(self, value, **_):
    super(NumpyVariable, self).__init__(
        onp.array(self, dtype=self.dtype) - onp.array(value, dtype=self.dtype))
    return self

if JAX_MODE:
  jax.interpreters.xla.canonicalize_dtype_handlers[NumpyVariable] = (
      jax.interpreters.xla.canonicalize_dtype_handlers[onp.ndarray])
  jax.interpreters.xla.pytype_aval_mappings[NumpyVariable] = (
      jax.interpreters.xla.pytype_aval_mappings[onp.ndarray])
  jax.core.pytype_aval_mappings[NumpyVariable] = (
      jax.core.pytype_aval_mappings[onp.ndarray])


def _convert_variable_to_tensor(value, dtype=None):
  return convert_to_tensor(value.__wrapped__, dtype=dtype)
register_tensor_conversion_function(NumpyVariable, _convert_variable_to_tensor)

Variable = NumpyVariable


class _TensorMeta(type(np.ndarray)):

  @classmethod
  def __instancecheck__(cls, instance):
    if JAX_MODE:
      return isinstance(instance, (jax.xla.DeviceArray,
                                   jax.core.Tracer))
    return isinstance(instance, np.ndarray)


class Tensor(six.with_metaclass(_TensorMeta)):
  OVERLOADABLE_OPERATORS = frozenset((
      # Binary.
      '__add__',
      '__radd__',
      '__sub__',
      '__rsub__',
      '__mul__',
      '__rmul__',
      '__truediv__',
      '__rtruediv__',
      '__floordiv__',
      '__rfloordiv__',
      '__mod__',
      '__rmod__',
      '__lt__',
      '__le__',
      '__gt__',
      '__ge__',
      '__ne__',
      '__eq__',
      '__and__',
      '__rand__',
      '__or__',
      '__ror__',
      '__xor__',
      '__rxor__',
      '__getitem__',
      '__pow__',
      '__rpow__',
      # Unary.
      '__invert__',
      '__neg__',
      '__abs__',
      '__matmul__',
      '__rmatmul__'
  ))


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
