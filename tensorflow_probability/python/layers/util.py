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
"""Utilities for probabilistic layers."""

import binascii
import codecs
import marshal
import os
import types
# Dependency imports
import numpy as np
import tensorflow.compat.v2 as tf

from tensorflow_probability.python import util as tfp_util
from tensorflow_probability.python.distributions import deterministic as deterministic_lib
from tensorflow_probability.python.distributions import independent as independent_lib
from tensorflow_probability.python.distributions import normal as normal_lib

from tensorflow_probability.python.internal import tf_keras


__all__ = [
    'default_loc_scale_fn',
    'default_mean_field_normal_fn',
    'default_multivariate_normal_fn',
    'deserialize_function',
    'serialize_function',
]


def default_loc_scale_fn(
    is_singular=False,
    loc_initializer=tf_keras.initializers.RandomNormal(stddev=0.1),
    untransformed_scale_initializer=tf_keras.initializers.RandomNormal(
        mean=-3., stddev=0.1),
    loc_regularizer=None,
    untransformed_scale_regularizer=None,
    loc_constraint=None,
    untransformed_scale_constraint=None):
  """Makes closure which creates `loc`, `scale` params from `tf.get_variable`.

  This function produces a closure which produces `loc`, `scale` using
  `tf.get_variable`. The closure accepts the following arguments:

    dtype: Type of parameter's event.
    shape: Python `list`-like representing the parameter's event shape.
    name: Python `str` name prepended to any created (or existing)
      `tf.Variable`s.
    trainable: Python `bool` indicating all created `tf.Variable`s should be
      added to the graph collection `GraphKeys.TRAINABLE_VARIABLES`.
    add_variable_fn: `tf.get_variable`-like `callable` used to create (or
      access existing) `tf.Variable`s.

  Args:
    is_singular: Python `bool` indicating if `scale is None`. Default: `False`.
    loc_initializer: Initializer function for the `loc` parameters.
      The default is `tf.random_normal_initializer(mean=0., stddev=0.1)`.
    untransformed_scale_initializer: Initializer function for the `scale`
      parameters. Default value: `tf.random_normal_initializer(mean=-3.,
      stddev=0.1)`. This implies the softplus transformed result is initialized
      near `0`. It allows a `Normal` distribution with `scale` parameter set to
      this value to approximately act like a point mass.
    loc_regularizer: Regularizer function for the `loc` parameters.
      The default (`None`) is to use the `tf.get_variable` default.
    untransformed_scale_regularizer: Regularizer function for the `scale`
      parameters. The default (`None`) is to use the `tf.get_variable` default.
    loc_constraint: An optional projection function to be applied to the
      loc after being updated by an `Optimizer`. The function must take as input
      the unprojected variable and must return the projected variable (which
      must have the same shape). Constraints are not safe to use when doing
      asynchronous distributed training.
      The default (`None`) is to use the `tf.get_variable` default.
    untransformed_scale_constraint: An optional projection function to be
      applied to the `scale` parameters after being updated by an `Optimizer`
      (e.g. used to implement norm constraints or value constraints). The
      function must take as input the unprojected variable and must return the
      projected variable (which must have the same shape). Constraints are not
      safe to use when doing asynchronous distributed training. The default
      (`None`) is to use the `tf.get_variable` default.

  Returns:
    default_loc_scale_fn: Python `callable` which instantiates `loc`, `scale`
    parameters from args: `dtype, shape, name, trainable, add_variable_fn`.
  """
  def _fn(dtype, shape, name, trainable, add_variable_fn):
    """Creates `loc`, `scale` parameters."""
    loc = add_variable_fn(
        name=name + '_loc',
        shape=shape,
        initializer=loc_initializer,
        regularizer=loc_regularizer,
        constraint=loc_constraint,
        dtype=dtype,
        trainable=trainable)
    if is_singular:
      return loc, None
    untransformed_scale = add_variable_fn(
        name=name + '_untransformed_scale',
        shape=shape,
        initializer=untransformed_scale_initializer,
        regularizer=untransformed_scale_regularizer,
        constraint=untransformed_scale_constraint,
        dtype=dtype,
        trainable=trainable)
    scale = tfp_util.DeferredTensor(
        untransformed_scale,
        lambda x: (np.finfo(dtype.as_numpy_dtype).eps + tf.nn.softplus(x)))
    return loc, scale
  return _fn


def default_mean_field_normal_fn(
    is_singular=False,
    loc_initializer=tf_keras.initializers.RandomNormal(stddev=0.1),
    untransformed_scale_initializer=tf_keras.initializers.RandomNormal(
        mean=-3., stddev=0.1),
    loc_regularizer=None,
    untransformed_scale_regularizer=None,
    loc_constraint=None,
    untransformed_scale_constraint=None):
  """Creates a function to build Normal distributions with trainable params.

  This function produces a closure which produces `tfd.Normal`
  parameterized by a `loc` and `scale` each created using `tf.get_variable`.

  Args:
    is_singular: Python `bool` if `True`, forces the special case limit of
      `scale->0`, i.e., a `Deterministic` distribution.
    loc_initializer: Initializer function for the `loc` parameters.
      The default is `tf.random_normal_initializer(mean=0., stddev=0.1)`.
    untransformed_scale_initializer: Initializer function for the `scale`
      parameters. Default value: `tf.random_normal_initializer(mean=-3.,
      stddev=0.1)`. This implies the softplus transformed result is initialized
      near `0`. It allows a `Normal` distribution with `scale` parameter set to
      this value to approximately act like a point mass.
    loc_regularizer: Regularizer function for the `loc` parameters.
    untransformed_scale_regularizer: Regularizer function for the `scale`
      parameters.
    loc_constraint: An optional projection function to be applied to the
      loc after being updated by an `Optimizer`. The function must take as input
      the unprojected variable and must return the projected variable (which
      must have the same shape). Constraints are not safe to use when doing
      asynchronous distributed training.
    untransformed_scale_constraint: An optional projection function to be
      applied to the `scale` parameters after being updated by an `Optimizer`
      (e.g. used to implement norm constraints or value constraints). The
      function must take as input the unprojected variable and must return the
      projected variable (which must have the same shape). Constraints are not
      safe to use when doing asynchronous distributed training.

  Returns:
    make_normal_fn: Python `callable` which creates a `tfd.Normal`
      using from args: `dtype, shape, name, trainable, add_variable_fn`.
  """
  loc_scale_fn = default_loc_scale_fn(
      is_singular=is_singular,
      loc_initializer=loc_initializer,
      untransformed_scale_initializer=untransformed_scale_initializer,
      loc_regularizer=loc_regularizer,
      untransformed_scale_regularizer=untransformed_scale_regularizer,
      loc_constraint=loc_constraint,
      untransformed_scale_constraint=untransformed_scale_constraint)
  def _fn(dtype, shape, name, trainable, add_variable_fn):
    """Creates multivariate `Deterministic` or `Normal` distribution.

    Args:
      dtype: Type of parameter's event.
      shape: Python `list`-like representing the parameter's event shape.
      name: Python `str` name prepended to any created (or existing)
        `tf.Variable`s.
      trainable: Python `bool` indicating all created `tf.Variable`s should be
        added to the graph collection `GraphKeys.TRAINABLE_VARIABLES`.
      add_variable_fn: `tf.get_variable`-like `callable` used to create (or
        access existing) `tf.Variable`s.

    Returns:
      Multivariate `Deterministic` or `Normal` distribution.
    """
    loc, scale = loc_scale_fn(dtype, shape, name, trainable, add_variable_fn)
    if scale is None:
      dist = deterministic_lib.Deterministic(loc=loc)
    else:
      dist = normal_lib.Normal(loc=loc, scale=scale)
    batch_ndims = tf.size(dist.batch_shape_tensor())
    return independent_lib.Independent(
        dist, reinterpreted_batch_ndims=batch_ndims)
  return _fn


def default_multivariate_normal_fn(dtype, shape, name, trainable,
                                   add_variable_fn):
  """Creates multivariate standard `Normal` distribution.

  Args:
    dtype: Type of parameter's event.
    shape: Python `list`-like representing the parameter's event shape.
    name: Python `str` name prepended to any created (or existing)
      `tf.Variable`s.
    trainable: Python `bool` indicating all created `tf.Variable`s should be
      added to the graph collection `GraphKeys.TRAINABLE_VARIABLES`.
    add_variable_fn: `tf.get_variable`-like `callable` used to create (or
      access existing) `tf.Variable`s.

  Returns:
    Multivariate standard `Normal` distribution.
  """
  del name, trainable, add_variable_fn   # unused
  dist = normal_lib.Normal(
      loc=tf.zeros(shape, dtype), scale=dtype.as_numpy_dtype(1))
  batch_ndims = tf.size(dist.batch_shape_tensor())
  return independent_lib.Independent(
      dist, reinterpreted_batch_ndims=batch_ndims)


def deserialize_function(serial, function_type):
  """Deserializes the Keras-serialized function.

  (De)serializing Python functions from/to bytecode is unsafe. Therefore we
  also use the function's type as an anonymous function ('lambda') or named
  function in the Python environment ('function'). In the latter case, this lets
  us use the Python scope to obtain the function rather than reload it from
  bytecode. (Note that both cases are brittle!)

  Keras-deserialized functions do not perform lexical scoping. Any modules that
  the function requires must be imported within the function itself.

  This serialization mimicks the implementation in `tf_keras.layers.Lambda`.

  Args:
    serial: Serialized Keras object: typically a dict, string, or bytecode.
    function_type: Python string denoting 'function' or 'lambda'.

  Returns:
    function: Function the serialized Keras object represents.

  #### Examples

  ```python
  serial, function_type = serialize_function(lambda x: x)
  function = deserialize_function(serial, function_type)
  assert function(2.3) == 2.3  # function is identity
  ```

  """
  if function_type == 'function':
    # Simple lookup in custom objects
    function = tf_keras.utils.legacy.deserialize_keras_object(serial)
  elif function_type == 'lambda':
    # Unsafe deserialization from bytecode
    function = _func_load(serial)
  else:
    raise TypeError('Unknown function type:', function_type)
  return function


def serialize_function(func):
  """Serializes function for Keras.

  (De)serializing Python functions from/to bytecode is unsafe. Therefore we
  return the function's type as an anonymous function ('lambda') or named
  function in the Python environment ('function'). In the latter case, this lets
  us use the Python scope to obtain the function rather than reload it from
  bytecode. (Note that both cases are brittle!)

  This serialization mimicks the implementation in `tf_keras.layers.Lambda`.

  Args:
    func: Python function to serialize.

  Returns:
    (serial, function_type): Serialized object, which is a tuple of its
    bytecode (if function is anonymous) or name (if function is named), and its
    function type.
  """
  if isinstance(func, types.LambdaType):
    return _func_dump(func), 'lambda'
  return func.__name__, 'function'


def _func_dump(func):
  """Serializes a user defined function.

  Args:
    func: the function to serialize.

  Returns:
    A tuple `(code, defaults, closure)`.
  """
  if os.name == 'nt':
    raw_code = marshal.dumps(func.__code__).replace(b'\\', b'/')
    code = codecs.encode(raw_code, 'base64').decode('ascii')
  else:
    raw_code = marshal.dumps(func.__code__)
    code = codecs.encode(raw_code, 'base64').decode('ascii')
  defaults = func.__defaults__
  if func.__closure__:
    closure = tuple(c.cell_contents for c in func.__closure__)
  else:
    closure = None
  return code, defaults, closure


def _func_load(code, defaults=None, closure=None, globs=None):
  """Deserializes a user defined function.

  Args:
    code: bytecode of the function.
    defaults: defaults of the function.
    closure: closure of the function.
    globs: dictionary of global objects.

  Returns:
    A function object.
  """
  if isinstance(code, (tuple, list)):  # unpack previous dump
    code, defaults, closure = code
    if isinstance(defaults, list):
      defaults = tuple(defaults)

  def ensure_value_to_cell(value):
    """Ensures that a value is converted to a python cell object.

    Args:
      value: Any value that needs to be casted to the cell type

    Returns:
      A value wrapped as a cell object (see function `_func_load`)
    """

    def dummy_fn():
      value  # just access it so it gets captured in .__closure__  # pylint:disable=pointless-statement

    cell_value = dummy_fn.__closure__[0]
    if not isinstance(value, type(cell_value)):
      return cell_value
    return value

  if closure is not None:
    closure = tuple(ensure_value_to_cell(_) for _ in closure)
  try:
    raw_code = codecs.decode(code.encode('ascii'), 'base64')
  except (UnicodeEncodeError, binascii.Error):
    raw_code = code.encode('raw_unicode_escape')
  code = marshal.loads(raw_code)
  if globs is None:
    globs = globals()
  return types.FunctionType(
      code, globs, name=code.co_name, argdefs=defaults, closure=closure)
