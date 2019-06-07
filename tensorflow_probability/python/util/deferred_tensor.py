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
"""The `DeferredTensor` class."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import six

import tensorflow.compat.v2 as tf
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import name_util
from tensorflow_probability.python.internal import tensorshape_util


__all__ = [
    'DeferredTensor',
]


def _wrap_method(cls, attr):
  """Replaces member function's first arg, `self`, to `self._value()`.

  This function is used by `_get_tensor_like_attributes` to take existing
  `Tensor` member functions and make them operate on `self._value()`, i.e., the
  concretization of a `Distribution`.

  Args:
    cls: The `class` from which we will look up the `attr`.
    attr: Python `str` representing the `attr` to inject a new notion of `self`.

  Returns:
    dependency_injected_function: Python `callable` (or `property`)
      corresponding to `cls.attr` with `self` replaced as `self._value()`.
  """
  fn = getattr(cls, attr)
  is_property = isinstance(fn, property)
  if is_property:
    fn = fn.fget
  @functools.wraps(fn)
  def wrapped(self, *args, **kwargs):
    return fn(self._value(), *args, **kwargs)  # pylint: disable=protected-access
  return property(wrapped) if is_property else wrapped


def _tensorize(d, dtype=None, name=None, as_ref=False):
  """Tensor conversion function presuming `hasattr(d, '_value')`."""
  return d._value(dtype, name, as_ref)  # pylint: disable=protected-access


class TensorMetaClass(type):
  """A type of class which will make objects which act like Tensors."""

  def __new__(mcs, name, bases, attrs):
    attrs.update(
        (attr, _wrap_method(tf.Tensor, attr))
        for attr in tf.Tensor.OVERLOADABLE_OPERATORS.union({'__iter__'}))
    attrs.update(
        (attr, getattr(tf.Tensor, attr))
        for attr in {'__nonzero__', '__bool__', '__array_priority__'})
    cls = super(TensorMetaClass, mcs).__new__(mcs, name, bases, attrs)
    tf.register_tensor_conversion_function(cls, conversion_func=_tensorize)
    return cls


NONE_SPECIFIED = 'None'


@six.add_metaclass(TensorMetaClass)
class DeferredTensor(tf.Module):
  """Variable tracking object which applies function upon `convert_to_tensor`.

  #### Example

  ```python
  import tensorflow.compat.v2 as tf

  trainable_normal = tfd.Normal(
      loc=tf.Variable(0.),
      scale=tfp.util.DeferredTensor(tf.math.exp, tf.Variable(0.)))

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

  """

  def __init__(self, transform_fn, pretransformed_input, dtype=None,
               shape=NONE_SPECIFIED, name=None):
    """Creates the `DeferredTensor` object.

    Args:
      transform_fn: Python `callable` taking `pretransformed_input` and
        returning a `Tensor` (representing by this object).
      pretransformed_input: object with `shape`, `dtype` properties (typically a
        `tf.Variable`) passed into `transform_fn` when this object is acted upon
        in a `Tensor` context, eg, `tf.convert_to_tensor`, `+`, `tf.math.exp`,
        etc.
      dtype: Equivalent to what would otherwise be
        `transform_fn(pretransformed_input).dtype`.
         Default value: `None` (i.e., `pretransformed_input.dtype`).
      shape: Equivalent to what would otherwise be
        `transform_fn(pretransformed_input).shape`.
         Default value: `'None'` (i.e., `pretransformed_input.shape`).
      name: Python `str` representing this object's `name`; used only in graph
        mode.
        Default value: `None` (i.e.,
        `transform_fn.__name__ + '_' + pretransformed_input.name`).

    Raises:
      TypeError: if `transform_fn` is not `callable`.
      TypeError: if `pretransformed_input` lacks `dtype` and/or `shape`
        properties (and `dtype` and/or `shape` arguments are unspecified).
    """
    if not callable(transform_fn):
      raise TypeError('Argument `transform_fn` must be a Python `callable`.')
    if ((dtype is None and not hasattr(pretransformed_input, 'dtype')) or
        (shape is None and not hasattr(pretransformed_input, 'shape'))):
      raise TypeError('Argument `pretransformed_input` must have `dtype` and '
                      '`shape` properties (unless `dtype`, `shape` arguments '
                      'are explicitly provided.')
    has_name = bool(name)
    if not has_name:
      name = '_'.join([
          transform_fn.__name__,
          getattr(pretransformed_input, 'name', '')])
      name = name_util.strip_invalid_chars(name)
      name = name_util.camel_to_lower_snake(name)
    name = name_util.get_name_scope_name(name)
    name = name_util.strip_invalid_chars(name)
    super(DeferredTensor, self).__init__(name=name)
    self._name = name

    self._transform_fn = transform_fn
    self._pretransformed_input = pretransformed_input
    self._dtype = dtype_util.base_dtype(dtype or pretransformed_input.dtype)
    self._shape = tf.TensorShape(
        pretransformed_input.shape if shape == 'None' else shape)

  @property
  def transform_fn(self):
    """Function which characterizes the `Tensor`ization of this object."""
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

  @property
  def name(self):
    """The string name of this object."""
    return self._name

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
    return '<DeferredTensor: dtype={}, shape={}, fn={}>'.format(
        self.dtype.name if self.dtype else '?',
        str(self.shape.as_list()
            if self.shape.ndims is not None else '?').replace('None', '?'),
        self.transform_fn.__name__)

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
    self._dtype = y.dtype
    self.set_shape(y.shape)
    return tf.convert_to_tensor(y, dtype=dtype, name=name)
