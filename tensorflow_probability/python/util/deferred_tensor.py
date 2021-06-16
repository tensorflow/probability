# Copyright 2019 The TensorFlow Probability Authors.
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
"""Class definitions for declaritive (vs imperative) `Tensors` & `Variables`."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import functools
import numpy as np
import six

import tensorflow.compat.v2 as tf
from tensorflow_probability.python.internal import auto_composite_tensor
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import name_util
from tensorflow_probability.python.internal import tensor_util
from tensorflow_probability.python.internal import tensorshape_util

from tensorflow.python.framework import type_spec  # pylint: disable=g-direct-tensorflow-import
from tensorflow.python.ops import resource_variable_ops  # pylint: disable=g-direct-tensorflow-import


__all__ = [
    'DeferredTensor',
    'TransformedVariable',
]


JAX_MODE = False
NUMPY_MODE = False


_identity = lambda x: x


def _numpy_text(tensor):
  """Human readable representation of a tensor's numpy value."""
  if dtype_util.is_numpy_compatible(tensor.dtype):
    value = np.array(tensor)
    if value.shape:
      text = repr(value)
    else:
      text = str(value)
  else:
    text = '<unprintable>'
  if '\n' in text:
    text = '\n' + text
  return text


def _wrap_method(attr):
  """Wraps a method to operate on the concretized value.

  Args:
    attr: Python `str` representing the `attr` to inject a new notion of `self`.

  Returns:
    dependency_injected_function: Python `callable`
      corresponding to `type(self).attr` but implemented by `new_fn`.
  """
  def new_fn_like_old_fn(self, *args, **kwargs):
    value = self._value()  # pylint: disable=protected-access
    old_fn = getattr(type(value), attr)
    return old_fn(value, *args, **kwargs)
  return new_fn_like_old_fn


def _tensorize(d, dtype=None, name=None, as_ref=False):
  """Tensor conversion function presuming `hasattr(d, '_value')`."""
  return d._value(dtype, name, as_ref)  # pylint: disable=protected-access


class TensorMetaClass(abc.ABCMeta):
  """A type of class which will make objects which act like Tensors."""

  def __new__(mcs, name, bases, attrs):  # pylint: disable=bad-mcs-classmethod-argument
    operators = set(tf.Tensor.OVERLOADABLE_OPERATORS)
    operators.difference_update({'__eq__', '__ne__'})
    operators.update({'__iter__'})
    attrs.update((attr, _wrap_method(attr)) for attr in operators)

    # Support methods for __iter__ and __bool__
    private_methods = {
        name for name in dir(tf.Tensor) if name.startswith('_disallow')
    }
    attrs.update(
        (attr, _wrap_method(attr))
        for attr in private_methods)

    if JAX_MODE or NUMPY_MODE:
      other_attrs = {'__array_priority__'}
      if six.PY2:
        other_attrs.add('__nonzero__')
      else:
        other_attrs.add('__bool__')
      attrs.update((attr, getattr(np.ndarray, attr)) for attr in other_attrs)
    else:
      attrs.update(
          (attr, getattr(tf.Tensor, attr))
          for attr in {'__bool__', '__array_priority__', '__nonzero__'})
    cls = super(TensorMetaClass, mcs).__new__(mcs, name, bases, attrs)  # pylint: disable=too-many-function-args
    tf.register_tensor_conversion_function(cls, conversion_func=_tensorize)
    return cls


NONE_SPECIFIED = 'None'


class DeferredTensor(six.with_metaclass(
    TensorMetaClass, tf.Module, tf.__internal__.CompositeTensor)):
  """Variable tracking object which applies function upon `convert_to_tensor`.

  #### Example

  ```python
  import tensorflow.compat.v2 as tf
  import tensorflow_probability as tfp
  tfb = tfp.bijectors
  tfd = tfp.distributions

  # Note: it'd be better to use `tfp.util.TransformedVariable`;
  #       this example is for illustration only.
  trainable_normal = tfd.Normal(
      loc=tf.Variable(0.),
      scale=tfp.util.DeferredTensor(tf.Variable(0.), tf.math.exp))

  trainable_normal.loc
  # ==> <tf.Variable 'Variable:0' shape=() dtype=float32, numpy=0.0>

  trainable_normal.scale
  # ==> <DeferredTensor: dtype=float32, shape=[], fn=exp>

  # Operators work with `DeferredTensor`.
  trainable_normal.scale + 1.
  # ==> 2.

  with tf.GradientTape() as tape:
    negloglik = -trainable_normal.log_prob(0.5)
  g = tape.gradient(negloglik, trainable_normal.trainable_variables)
  # ==> (-0.5, 0.75)
  ```

  Which we could then fit as:

  ```python
  opt = tf.optimizers.Adam(learning_rate=0.05)
  loss = tf.function(lambda: -trainable_normal.log_prob(0.5), autograph=True)
  for _ in range(int(1e3)):
    opt.minimize(loss, trainable_normal.trainable_variables)
  trainable_normal.mean()
  # ==> 0.5
  trainable_normal.stddev()
  # ==> (approximately) 0.0075
  ```

  It is also possible to parameterize a `DeferredTensor` with a bijector, e.g.:

  ```python
  # Note: it'd be better to use `tfp.util.TransformedVariable`;
  #       this example is for illustration only.
  d = tfd.Normal(loc=0.,
                 scale=tfp.util.DeferredTensor(tf.Variable([0.54, 1.85]),
                                               tfb.Softplus()))
  d.stddev()
  # ==> [1., 2.]
  tf.convert_to_tensor(d.scale)
  # ==> [1., 2.]
  ```

  """

  def __init__(self, pretransformed_input, transform_fn, dtype=None,
               shape=NONE_SPECIFIED, also_track=None, name=None):
    """Creates the `DeferredTensor` object.

    Args:
      pretransformed_input: object with `shape`, `dtype` properties (typically a
        `tf.Variable`) passed into `transform_fn` when this object is acted upon
        in a `Tensor` context, eg, `tf.convert_to_tensor`, `+`, `tf.math.exp`,
        etc.
      transform_fn: Python `callable` or `tfp.bijectors.Bijector`-like instance.
        When `callable`, should take `pretransformed_input` and
        return a `Tensor` (representing by this object).
      dtype: Equivalent to what would otherwise be
      `transform_fn(pretransformed_input).dtype`.
         Default value: `None` (i.e.,
         `getattr(transform_fn, 'dtype', None) or pretransformed_input.dtype`).
      shape: Equivalent to what would otherwise be
        `transform_fn(pretransformed_input).shape`.
         Default value: `'None'` (i.e.,
         `getattr(transform_fn, 'forward_event_shape', lambda x: x)(
              pretransformed_input.shape)`).
      also_track: Optional instance or structure of instances of `tf.Variable`
        and/or `tf.Module`, containing any additional trainable variables that
        the `transform_fn` may access beyond the given
        `pretransformed_input`. This ensures that such variables
        will be correctly tracked in `self.trainable_variables`.
        Default value: `None`.
      name: Python `str` representing this object's `name`; used only in graph
        mode.
        Default value: `None` (i.e.,
        `(getattr(transform_fn, 'name', None) or
          transform_fn.__name__ + '_' + pretransformed_input.name)`).

    Raises:
      TypeError: if `transform_fn` is not `callable`.
      TypeError: if `pretransformed_input` lacks `dtype` and/or `shape`
        properties (and `dtype` and/or `shape` arguments are unspecified).
    """
    pretransformed_input = tensor_util.convert_nonref_to_tensor(
        pretransformed_input,
        name='pretransformed_input')

    if dtype is None:
      dtype = (getattr(transform_fn, 'dtype', None) or
               dtype_util.base_dtype(pretransformed_input.dtype))
    try:
      dtype = None if dtype is None else tf.as_dtype(dtype)
    except TypeError:
      raise TypeError('Argument `dtype` must be convertible to a '
                      '`tf.dtypes.DType`; saw "{}" of type "{}".'.format(
                          repr(dtype), type(dtype)))

    if shape == NONE_SPECIFIED:
      shape = getattr(transform_fn, 'forward_event_shape', _identity)
      shape = shape(pretransformed_input.shape)
    try:
      shape = tf.TensorShape(shape)
    except TypeError:
      raise TypeError('Argument `shape` must be convertible to a '
                      '`tf.TensorShape`; saw "{}".'.format(shape))

    name = name or getattr(transform_fn, 'name', None)
    if not name:
      name = '_'.join([
          transform_fn.__name__,
          getattr(pretransformed_input, 'name', '')])
      name = name_util.strip_invalid_chars(name)
      name = name_util.camel_to_lower_snake(name)
    name = name_util.get_name_scope_name(name)
    name = name_util.strip_invalid_chars(name)

    if hasattr(transform_fn, 'forward'):
      fwd_name = '"{}"'.format(transform_fn.name)
    else:
      fwd_name = transform_fn.__name__
      if not callable(transform_fn):
        raise TypeError('Argument `transform_fn` must be `callable`.')

    super(DeferredTensor, self).__init__(name=name)
    self._pretransformed_input = pretransformed_input
    self._transform_fn = transform_fn
    self._dtype = dtype
    self._shape = shape
    self._also_track = also_track
    self._name = name
    self._fwd_name = fwd_name

    # Secret handshake with tf.is_tensor to return True for DT.
    #
    # Works around an exception in LinearOperator (which in 2.0.0 checks only
    # `tf.is_tensor`, not also `linear_operator_util.is_ref`:
    # ValueError: Graph parent item 0 is not a Tensor;
    #   <DeferredTensor: dtype=float32, shape=[2], fn=exp>.
    # TODO(b/140157055): Remove this shim after LinOp is patched in 2.0.
    self.is_tensor_like = True

  @property
  def transform_fn(self):
    """Function which characterizes the `Tensor`ization of this object."""
    if hasattr(self._transform_fn, 'forward'):
      return self._transform_fn.forward
    return self._transform_fn

  @property
  def pretransformed_input(self):
    """Input to `transform_fn`."""
    return self._pretransformed_input

  @property
  def dtype(self):
    """Represents the type of the elements in a `Tensor`."""
    return self._dtype

  @property
  def shape(self):
    """Represents the shape of a `Tensor`."""
    return self._shape

  # TODO(b/140157055): Remove this shim.
  def get_shape(self):
    """Legacy means of getting Tensor shape, for compat with 2.0.0 LinOp."""
    return self._shape

  @property
  def also_track(self):
    """Additional variables tracked by tf.Module in self.trainable_variables."""
    return self._also_track

  @property
  def name(self):
    """The string name of this object."""
    return self._name

  def numpy(self):
    """Returns (copy of) deferred values as a NumPy array or scalar."""
    value = self._value()
    if not tf.executing_eagerly():
      raise NotImplementedError(
          'DeferredTensor.numpy() is only supported in eager execution mode.')
    return np.array(value)

  def set_shape(self, shape):
    """Updates the shape of this pretransformed_input.

    This method can be called multiple times, and will merge the given `shape`
    with the current shape of this object. It can be used to provide additional
    information about the shape of this object that cannot be inferred from the
    graph alone.

    Args:
      shape: A `TensorShape` representing the shape of this
        `pretransformed_input`, a `TensorShapeProto`, a list, a tuple, or None.

    Raises:
      ValueError: If `shape` is not compatible with the current shape of this
        `pretransformed_input`.
    """
    self._shape = self._shape.merge_with(shape)

  def __repr__(self):
    if tf.executing_eagerly():
      try:
        value = self._value()
      except Exception as e:  # pylint: disable=broad-except
        value = e
      value_str = ', numpy={}'.format(value if isinstance(value, Exception)
                                      else _numpy_text(value))
    else:
      value_str = ''
    return '<{}: dtype={}, shape={}, fn={}{}>'.format(
        type(self).__name__,
        dtype_util.name(self.dtype) if self.dtype else '?',
        str(
            tensorshape_util.as_list(self.shape)
            if tensorshape_util.rank(self.shape) is not None else '?').replace(
                'None', '?'), self._fwd_name, value_str)

  def __getitem__(self, i):
    return self._value()[i]

  def _value(self, dtype=None, name=None, as_ref=False):
    y = self.transform_fn(self.pretransformed_input)  # pylint: disable=not-callable
    if dtype_util.base_dtype(y.dtype) != self.dtype:
      raise TypeError(
          'Actual dtype ({}) does not match deferred dtype ({}).'.format(
              dtype_util.name(dtype_util.base_dtype(y.dtype)),
              dtype_util.name(self.dtype)))
    if not tensorshape_util.is_compatible_with(y.shape, self.shape):
      raise TypeError(
          'Actual shape ({}) is incompatible with deferred shape ({}).'.format(
              y.shape, self.shape))
    return tf.convert_to_tensor(y, dtype=dtype, name=name)

  def __array__(self, dtype=None):
    if not tf.executing_eagerly():
      raise NotImplementedError(
          'Cannot convert a symbolic (graph mode) `DeferredTensor` to a '
          'numpy array.')
    return np.array(self._value(dtype=dtype))

  def _get_input_spec(self):
    if isinstance(self.pretransformed_input, tf.__internal__.CompositeTensor):
      return self.pretransformed_input._type_spec  # pylint: disable=protected-access
    if isinstance(self.pretransformed_input, tf.Variable):
      return resource_variable_ops.VariableSpec(
          self.pretransformed_input.shape,
          dtype=self.pretransformed_input.dtype,
          trainable=self.pretransformed_input.trainable)
    return tf.TensorSpec.from_tensor(self.pretransformed_input)

  @property
  def _type_spec(self):
    input_spec = self._get_input_spec()
    transform_or_spec = getattr(self._transform_fn, '_type_spec',
                                self._transform_fn)

    # Extract Variables from also_track.
    if self.also_track is None:
      also_track_spec = None
    else:
      also_track_vars = tf.nest.flatten(
          tf.nest.map_structure(
              lambda x: x.variables if isinstance(x, tf.Module) else x,
              self.also_track))
      also_track_spec = tf.nest.map_structure(
          lambda x: resource_variable_ops.VariableSpec(  # pylint: disable=g-long-lambda
              x.shape, x.dtype, trainable=x.trainable),
          also_track_vars)

    return _DeferredTensorSpec(
        input_spec, transform_or_spec, dtype=self.dtype, shape=self.shape,
        name=self.name, also_track_spec=also_track_spec)


class TransformedVariable(DeferredTensor):
  """Variable tracking object which applies a bijector upon `convert_to_tensor`.

  #### Example

  ```python
  import tensorflow.compat.v2 as tf
  import tensorflow_probability as tfp
  tfb = tfp.bijectors

  positive_variable = tfp.util.TransformedVariable(1., bijector=tfb.Exp())

  positive_variable
  # ==> <TransformedVariable: dtype=float32, shape=[], fn=exp>

  # Note that the initial value corresponds to the transformed output.
  tf.convert_to_tensor(positive_variable)
  # ==> 1.

  positive_variable.pretransformed_input
  # ==> <tf.Variable 'Variable:0' shape=() dtype=float32, numpy=0.0>

  # Operators work with `TransformedVariable`.
  positive_variable + 1.
  # ==> 2.

  # It is also possible to assign values to a TransformedVariable
  with tf.control_dependencies([positive_variable.assign_add(2.)]):
    positive_variable
  # ==> 3.

  A common use case for the `TransformedVariable` is to fit constrained
  parameters. E.g.:

  ```python
  import tensorflow.compat.v2 as tf
  import tensorflow_probability as tfp
  tfb = tfp.bijectors
  tfd = tfp.distributions

  trainable_normal = tfd.Normal(
      loc=tf.Variable(0.),
      scale=tfp.util.TransformedVariable(1., bijector=tfb.Exp()))

  trainable_normal.loc
  # ==> <tf.Variable 'Variable:0' shape=() dtype=float32, numpy=0.0>

  trainable_normal.scale
  # ==> <TransformedVariable: dtype=float32, shape=[], fn=exp>

  with tf.GradientTape() as tape:
    negloglik = -trainable_normal.log_prob(0.5)
  g = tape.gradient(negloglik, trainable_normal.trainable_variables)
  # ==> (-0.5, 0.75)

  opt = tf.optimizers.Adam(learning_rate=0.05)
  loss = tf.function(lambda: -trainable_normal.log_prob(0.5))
  for _ in range(int(1e3)):
    opt.minimize(loss, trainable_normal.trainable_variables)
  trainable_normal.mean()
  # ==> 0.5
  trainable_normal.stddev()
  # ==> (approximately) 0.0075
  ```

  """

  def __init__(self, initial_value, bijector, dtype=None, name=None, **kwargs):
    """Creates the `TransformedVariable` object.

    Args:
      initial_value: A `Tensor`, or Python object convertible to a `Tensor`,
        which is the initial value for the `TransformedVariable`. The underlying
        untransformed `tf.Variable` will be initialized with
        `bijector.inverse(initial_value)`. Can also be a callable with no
        argument that returns the initial value when called.
      bijector: A `Bijector`-like instance which defines the transformations
        applied to the underlying `tf.Variable`.
      dtype: `tf.dtype.DType` instance or otherwise valid `dtype` value to
        `tf.convert_to_tensor(..., dtype)`.
         Default value: `None` (i.e., `bijector.dtype`).
      name: Python `str` representing the underlying `tf.Variable`'s name.
         Default value: `None`.
      **kwargs: Keyword arguments forward to `tf.Variable`.
    """
    # Check if `bijector` is "`Bijector`-like".
    for attr in {'forward', 'forward_event_shape',
                 'inverse', 'inverse_event_shape',
                 'name', 'dtype'}:
      if not hasattr(bijector, attr):
        raise TypeError('Argument `bijector` missing required `Bijector` '
                        'attribute "{}".'.format(attr))

    if callable(initial_value):
      initial_value = initial_value()

    # Extra kwarg that TypeSpec._from_components uses to re-build the object
    # without re-initializing the variable.
    pretransformed_input = kwargs.pop('pretransformed_input', None)
    if pretransformed_input is None:
      initial_value = tf.convert_to_tensor(
          initial_value, dtype_hint=bijector.dtype, dtype=dtype)
      pretransformed_input = tf.Variable(
          initial_value=bijector.inverse(initial_value),
          name=name,
          dtype=dtype,
          **kwargs)
      shape = initial_value.shape
    else:
      shape = bijector.forward_event_shape(pretransformed_input.shape)
    super(TransformedVariable, self).__init__(
        pretransformed_input=pretransformed_input,
        transform_fn=bijector,
        shape=shape,
        name=bijector.name)
    self._bijector = bijector

  @property
  def bijector(self):
    return self._bijector

  @property
  def initializer(self):
    """The initializer operation for the underlying variable."""
    return self.pretransformed_input.initializer

  @functools.wraps(tf.Variable.assign)
  def assign(self, value, use_locking=False, name=None, read_value=True):
    return self.pretransformed_input.assign(
        self.bijector.inverse(value),
        use_locking=use_locking,
        name=name,
        read_value=read_value)

  @functools.wraps(tf.Variable.assign_add)
  def assign_add(self, value, use_locking=False, name=None, read_value=True):
    value = tf.convert_to_tensor(value, self.dtype)
    new_value = self.transform_fn(self.pretransformed_input) + value  # pylint: disable=not-callable
    return self.pretransformed_input.assign(
        self.bijector.inverse(new_value),
        use_locking=use_locking,
        name=name,
        read_value=read_value)

  @functools.wraps(tf.Variable.assign_sub)
  def assign_sub(self, value, use_locking=False, name=None, read_value=True):
    value = tf.convert_to_tensor(value, self.dtype)
    new_value = self.transform_fn(self.pretransformed_input) - value  # pylint: disable=not-callable
    return self.pretransformed_input.assign(
        self.bijector.inverse(new_value),
        use_locking=use_locking,
        name=name,
        read_value=read_value)

  @property
  def _type_spec(self):
    input_spec = self._get_input_spec()
    transform_or_spec = getattr(self.bijector, '_type_spec', self.bijector)
    return _TransformedVariableSpec(
        input_spec, transform_or_spec, self.dtype, self.name)


class _DeferredTensorSpecBase(object):
  """Common methods for '_DeferredTensorSpec' and '_TransformedVariableSpec."""

  @property
  def name(self):
    return self._name

  @property
  def dtype(self):
    return self._dtype

  @property
  def transform_or_spec(self):
    return self._transform_or_spec

  def most_specific_compatible_type(self, other):
    """Returns the most specific TypeSpec compatible with `self` and `other`.

    Args:
      other: A `TypeSpec`.

    Returns:
      compatible_spec: The `TypeSpec` most compatible with `self` and `other`.

    Raises:
      ValueError: If there is no TypeSpec that is compatible with both `self`
        and `other`.
      ValueError: If `self._transform_fn` is not a `CompositeTensor` and not
        equal to `other._transform_fn`.
    """
    if type(self) is not type(other):
      raise ValueError(
          f'No TypeSpec is compatible with both {self} and {other}.')
    specs, params = self._TypeSpec__most_specific_compatible_type_serialization(
        (self._specs, self._unique_id_params),
        (other._specs, other._unique_id_params))  # pylint: disable=protected-access
    kwargs = dict(specs, **params)
    if not self._transform_is_composite:
      if self.transform_or_spec != other.transform_or_spec:
        raise ValueError(
            f'{self.transform_or_spec} and {other.transform_or_spec} must be '
            f'identical.')
      kwargs['transform_or_spec'] = self.transform_or_spec
    return type(self)(**kwargs, name=None)

  def is_compatible_with(self, spec_or_value):
    """Returns True if `spec_or_value` is compatible with this TypeSpec."""
    if not isinstance(spec_or_value, tf.TypeSpec):
      spec_or_value = type_spec.type_spec_from_value(spec_or_value)
    if type(self) is not type(spec_or_value):
      return False
    if not self._transform_is_composite:
      if self.transform_or_spec != spec_or_value.transform_or_spec:
        return False
    return self._TypeSpec__is_compatible(
        (self._specs, self._unique_id_params),
        (spec_or_value._specs, spec_or_value._unique_id_params))  # pylint: disable=protected-access

  def _with_tensor_ranks_only(self):
    """Returns a TypeSpec compatible with `self`, with Tensor shapes relaxed.

    Returns:
      A `TypeSpec` that is compatible with `self`, where any `TensorShape`
      information has been relaxed to include only Tensor rank (and not
      the dimension sizes for individual axes).
    """
    def relax(value):
      if isinstance(value, tf.TypeSpec):
        return value._with_tensor_ranks_only()  # pylint: disable=protected-access
      elif (isinstance(value, tf.TensorShape) and
            value.rank is not None):
        return tf.TensorShape([None] * value.rank)
      else:
        return value

    specs = self._specs.copy()
    transform_or_spec = specs.pop(
        'transform_or_spec', self.transform_or_spec)
    return type(self)(
        **tf.nest.map_structure(
            relax,
            dict(specs,
                 transform_or_spec=transform_or_spec,
                 **self._unique_id_params,
                 name=self.name)))

  def _get_batched_input_spec(self, batch_size):
    """Returns the batched `input_spec` for the given `batch_size`."""
    if isinstance(self._input_spec, type_spec.BatchableTypeSpec):
      return self._input_spec._batch(batch_size)  # pylint: disable=protected-access
    if isinstance(self._input_spec, resource_variable_ops.VariableSpec):
      return resource_variable_ops.VariableSpec(
          shape=tf.TensorShape([batch_size]).concatenate(
              self._input_spec.shape),
          dtype=self._input_spec.dtype,
          trainable=self._input_spec.trainable)
    raise NotImplementedError(
        f'`{self.value_type.__name__}`s `TypeSpec` is not supported for '
        f'inputs of type {type(self._input_spec)}.')

  def _get_unbatched_input_spec(self):
    """Returns the `input_spec` with leading batch dimension removed."""
    if isinstance(self._input_spec, type_spec.BatchableTypeSpec):
      return self._input_spec._unbatch()  # pylint: disable=protected-access
    if isinstance(self._input_spec, resource_variable_ops.VariableSpec):
      return resource_variable_ops.VariableSpec(
          shape=(None if self._input_spec.shape is None
                 else self._input_spec.shape[1:]),
          dtype=self._input_spec.dtype,
          trainable=self._input_spec.trainable)
    else:
      raise NotImplementedError(
          f'`{self.value_type.__name__}`s `TypeSpec` is not supported for '
          f'inputs of type {type(self._input_spec)}.')

  def _serialize(self):
    if not self._transform_is_composite:
      raise ValueError(
          f'Cannot serialize non-`CompositeTensor: {self.transform_or_spec}.')
    return tuple(
        dict(self._specs, **self._unique_id_params, name=self.name).items())

  @classmethod
  def _deserialize(cls, serialization):
    return cls(**dict(serialization))

  def __get_cmp_key(self):
    fn_key = (None if self._transform_is_composite
              else id(self.transform_or_spec))
    return (type(self), self._TypeSpec__make_cmp_key(
        (self._specs, self._unique_id_params, fn_key)))

  def __repr__(self):
    kwargs = dict(self._specs, **self._unique_id_params, name=self.name)
    if not self._transform_is_composite:
      kwargs['transform_or_spec'] = self.transform_or_spec
    kwargs_str = ', '.join(f'{k}={v}' for k, v in kwargs.items())
    return f'{type(self).__name__}({kwargs_str})'

  def __reduce__(self):
    if not self._transform_is_composite:
      raise ValueError(
          f'Cannot serialize object with callable parameters that are not '
          f'`CompositeTensor`s: {self.transform_or_spec}.')
    super().__reduce__()

  def __eq__(self, other):
    return (type(other) is type(self) and
            self.__get_cmp_key() == other.__get_cmp_key())  # pylint: disable=protected-access

  def __ne__(self, other):
    return not self == other

  def __hash__(self):
    return hash(self.__get_cmp_key())


@auto_composite_tensor.type_spec_register('tfp.util.DeferredTensorSpec')
class _DeferredTensorSpec(_DeferredTensorSpecBase, type_spec.BatchableTypeSpec):
  """`tf.TypeSpec` for `tfp.util.DeferredTensor`."""

  __slots__ = ('_input_spec', '_transform_or_spec', '_also_track_spec',
               '_dtype', '_shape', '_name', '_specs', '_unique_id_params',
               '_transform_is_composite')

  def __init__(self, input_spec, transform_or_spec, dtype, shape, name,
               also_track_spec=None):
    """Initializes a new `_DeferredTensorSpec`.

    Args:
      input_spec: `tf.TypeSpec` instance describing the `DeferredTensor`s
        `pretransformed_input` attribute.
      transform_or_spec: The `transform_fn` passed to the `DeferredTensor`'s
        constructor, or `transform_fn._type_spec` if `transform_fn` is a
        `CompositeTensor`.
      dtype: `tf.DType`, `dtype` property of the `DeferredTensor`.
      shape: `tf.TensorShape`, `shape` property of the `DeferredTensor`.
      name: `str`, name of the `DeferredTensor`.
      also_track_spec: Python `list` of `VariableSpec` instances describing the
        additional variables tracked by the `DeferredTensor`.
    """
    self._input_spec = input_spec
    self._transform_or_spec = transform_or_spec
    self._also_track_spec = also_track_spec
    self._dtype = dtype
    self._shape = shape
    self._name = name

    self._transform_is_composite = isinstance(transform_or_spec, tf.TypeSpec)
    self._unique_id_params = {'dtype': dtype, 'shape': shape}

    self._specs = {'input_spec': input_spec}
    if self._transform_is_composite:
      self._specs['transform_or_spec'] = transform_or_spec
    if also_track_spec is not None:
      self._specs['also_track_spec'] = also_track_spec

  @property
  def value_type(self):
    return DeferredTensor

  @property
  def shape(self):
    return self._shape

  def _to_components(self, value):
    """Encodes `value` as a nested structure of Tensor/CompositeTensor."""
    components = dict(pretransformed_input=value.pretransformed_input)
    # pylint: disable=protected-access
    if isinstance(value._transform_fn, tf.__internal__.CompositeTensor):
      components['transform_fn'] = value._transform_fn
    if value.also_track is not None:
      components['also_track'] = tf.nest.flatten(
          tf.nest.map_structure(
              lambda x: x.variables if isinstance(x, tf.Module) else x,
              value.also_track))
    return components

  def _from_components(self, components):
    """Reconstructs a value from a structure of Tensor/CompositeTensor."""
    transform_fn = components.pop('transform_fn', self.transform_or_spec)
    return DeferredTensor(**components, transform_fn=transform_fn,
                          dtype=self.dtype, shape=self.shape, name=self.name)

  @property
  def _component_specs(self):
    """A nested structure of TypeSpecs for the DeferredTensor's components."""
    specs = dict(pretransformed_input=self._input_spec)
    if self._transform_is_composite:
      specs['transform_fn'] = self.transform_or_spec
    if self._also_track_spec is not None:
      specs['also_track'] = self._also_track_spec
    return specs

  def _batch(self, batch_size):
    """Returns a TypeSpec representing a batch of DeferredTensors."""
    transform_or_spec = self._specs.get(
        'transform_or_spec', self.transform_or_spec)
    return _DeferredTensorSpec(
        self._get_batched_input_spec(batch_size),
        transform_or_spec=transform_or_spec,
        dtype=self.dtype,
        shape=(None if self.shape is None
               else tf.TensorShape([batch_size]).concatenate(self.shape)),
        name=self.name,
        also_track_spec=self._also_track_spec)

  def _unbatch(self):
    """Returns a TypeSpec representing a single DeferredTensor."""
    transform_or_spec = self._specs.get(
        'transform_or_spec', self.transform_or_spec)
    return _DeferredTensorSpec(
        self._get_unbatched_input_spec(),
        transform_or_spec=transform_or_spec,
        dtype=self.dtype,
        shape=(None if self.shape is None else self.shape[1:]),
        name=self.name,
        also_track_spec=self._also_track_spec)


@auto_composite_tensor.type_spec_register('tfp.util.TransformedVariableSpec')
class _TransformedVariableSpec(
    _DeferredTensorSpecBase, type_spec.BatchableTypeSpec):
  """`tf.TypeSpec` for `tfp.util.TransformedVariable`."""

  __slots__ = ('_input_spec', '_transform_or_spec', '_dtype', '_name', '_specs',
               '_unique_id_params', '_transform_is_composite')

  def __init__(self, input_spec, transform_or_spec, dtype, name):
    """Initializes a new `_TransformedVariableSpec`.

    Args:
      input_spec: `tf.TypeSpec` instance describing the `TransformedVariable`s
        `pretransformed_input` attribute.
      transform_or_spec: The `bijector` passed to the `TransformedVariable`'s
        constructor, or `bijector._type_spec` if `bijector` is a
        `CompositeTensor`.
      dtype: `tf.DType`, `dtype` property of the `TransformedVariable`.
      name: `str`, name of the `TransformedVariable`.
    """
    self._input_spec = input_spec
    self._transform_or_spec = transform_or_spec
    self._dtype = dtype
    self._name = name

    self._unique_id_params = {'dtype': dtype}
    self._transform_is_composite = isinstance(transform_or_spec, tf.TypeSpec)

    self._specs = {'input_spec': input_spec}
    if self._transform_is_composite:
      self._specs['transform_or_spec'] = transform_or_spec

  @property
  def value_type(self):
    return TransformedVariable

  def _to_components(self, value):
    """Encodes `value` as a nested structure of Tensor/CompositeTensor."""
    components = dict(pretransformed_input=value.pretransformed_input)
    if isinstance(value.bijector, tf.__internal__.CompositeTensor):
      components['bijector'] = value.bijector
    return components

  def _from_components(self, components):
    """Reconstructs a value from a structure of Tensor/CompositeTensor."""
    bijector = components.pop('bijector', self.transform_or_spec)
    return TransformedVariable(
        **components, initial_value=None, bijector=bijector,
        dtype=self.dtype, name=self.name)

  @property
  def _component_specs(self):
    """A structure of TypeSpecs for the TransformedVariable's components."""
    specs = dict(pretransformed_input=self._input_spec)
    if self._transform_is_composite:
      specs['bijector'] = self.transform_or_spec
    return specs

  def _batch(self, batch_size):
    """Returns a TypeSpec representing a batch of TransformedVariable."""
    transform_or_spec = self._specs.get(
        'transform_or_spec', self.transform_or_spec)
    return _TransformedVariableSpec(
        self._get_batched_input_spec(batch_size),
        transform_or_spec=transform_or_spec,
        dtype=self.dtype,
        name=self.name)

  def _unbatch(self):
    """Returns a TypeSpec representing a single TransformedVariable."""
    transform_or_spec = self._specs.get(
        'transform_or_spec', self.transform_or_spec)
    return _TransformedVariableSpec(
        self._get_unbatched_input_spec(),
        transform_or_spec=transform_or_spec,
        dtype=self.dtype,
        name=self.name)
