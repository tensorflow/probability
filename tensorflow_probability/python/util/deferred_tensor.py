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

import functools
import numpy as np
import six

import tensorflow.compat.v2 as tf
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import name_util
from tensorflow_probability.python.internal import tensor_util
from tensorflow_probability.python.internal import tensorshape_util


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


class TensorMetaClass(type):
  """A type of class which will make objects which act like Tensors."""

  def __new__(mcs, name, bases, attrs):
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
    cls = super(TensorMetaClass, mcs).__new__(mcs, name, bases, attrs)
    tf.register_tensor_conversion_function(cls, conversion_func=_tensorize)
    return cls


NONE_SPECIFIED = 'None'


class DeferredTensor(six.with_metaclass(TensorMetaClass, tf.Module)):
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
               shape=NONE_SPECIFIED, name=None):
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


class TransformedVariable(DeferredTensor):
  """Variable tracking object which applies a bijector upon `convert_to_tensor`.

  #### Example

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

  tf.convert_to_tensor(trainable_normal.scale)
  # ==> 1.

  # Operators work with `TransformedVariable`.
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
  loss = tf.function(lambda: -trainable_normal.log_prob(0.5))
  for _ in range(int(1e3)):
    opt.minimize(loss, trainable_normal.trainable_variables)
  trainable_normal.mean()
  # ==> 0.5
  trainable_normal.stddev()
  # ==> (approximately) 0.0075
  ```

  It is also possible to assign values to a TransformedVariable, e.g.,

  ```python
  d = tfd.Normal(
      loc=tf.Variable(0.),
      scale=tfp.util.TransformedVariable([1., 2.], bijector=tfb.Softplus()))
  d.stddev()
  # ==> [1., 2.]
  with tf.control_dependencies([x.scale.assign_add([0.5, 1.])]):
    d.stddev()
    # ==> [1.5, 3.]
  ```

  """

  def __init__(self, initial_value, bijector, dtype=None, name=None, **kwargs):
    """Creates the `TransformedVariable` object.

    Args:
      initial_value: A `Tensor`, or Python object convertible to a `Tensor`,
        which is the initial value for the Variable. Can also be a callable with
        no argument that returns the initial value when called. Note: if
        `initial_value` is a `TransformedVariable` then the instantiated object
        does not create a new `tf.Variable`, but rather points to the underlying
        `Variable` and chains the `bijector` arg with the underlying bijector as
        `tfb.Chain([bijector, initial_value.bijector])`.
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
    initial_value = tf.convert_to_tensor(
        initial_value, dtype_hint=bijector.dtype, dtype=dtype)
    super(TransformedVariable, self).__init__(
        pretransformed_input=tf.Variable(
            initial_value=bijector.inverse(initial_value),
            name=name,
            dtype=dtype,
            **kwargs),
        transform_fn=bijector,
        shape=initial_value.shape,
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
