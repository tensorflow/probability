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
"""Class which enables `tfd.Distribution` to `tf.Tensor` coercion."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import functools
import six

import tensorflow as tf
from tensorflow_probability.python.distributions import distribution as tfd
from tensorflow.python.framework import composite_tensor  # pylint: disable=g-direct-tensorflow-import


__all__ = []  # We intend nothing public.


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


def _get_tensor_like_attributes():
  """Returns `Tensor` attributes related to shape and Python builtins."""
  # Enable "Tensor semantics" for distributions.
  # See tensorflow/python/framework/ops.py `class Tensor` for details.
  attrs = dict()
  # Setup overloadable operators and white-listed members / properties.
  attrs.update((attr, _wrap_method(tf.Tensor, attr))
               for attr in tf.Tensor.OVERLOADABLE_OPERATORS.union({'__iter__'}))
  # Copy some members straight-through.
  attrs.update((attr, getattr(tf.Tensor, attr))
               for attr in {'__nonzero__', '__bool__', '__array_priority__'})
  return attrs


def _value(self, dtype=None, name=None, as_ref=False):  # pylint: disable=g-doc-args
  """Get the value returned by `tf.convert_to_tensor(distribution)`.

  Note: this function may mutate the distribution instance state by caching
  the concretized `Tensor` value.

  Args:
    dtype: Must return a `Tensor` with the given `dtype` if specified.
    name: If the conversion function creates a new `Tensor`, it should use the
      given `name` if specified.
    as_ref: `as_ref` is true, the function must return a `Tensor` reference,
      such as a `Variable`.
  Returns:
    concretized_distribution_value: `Tensor` identical to
    `tf.convert_to_tensor(distribution)`.

  #### Examples

  ```python
  tfd = tfp.distributions
  x = tfd.Normal(0.5, 1).set_tensor_conversion(tfd.Distribution.mean)

  x._value()
  # ==> tf.convert_to_tensor(x) ==> 0.5

  x._value() + 2
  # ==> tf.convert_to_tensor(x) + 2. ==> 2.5

  x + 2
  # ==> tf.convert_to_tensor(x) + 2. ==> 2.5
  ```

  """
  # pylint: disable=protected-access
  if as_ref:
    raise NotImplementedError(
        'Cannot convert a `Distribution` to a reference '
        '(e.g., `tf.Variable`).')
  if self._concrete_value is None:
    if self._convert_to_tensor_fn is None:
      raise NotImplementedError(
          'Failed to convert object of type {} to Tensor. Contents: {}. '
          'Call `distribution.set_tensor_conversion(lambda self: ...)` to '
          'enable `tf.convert_to_tensor` capability. For example: '
          '`x = tfd.Normal(0,1).set_tensor_conversion(tfd.Distribution.mean)`'
          ' results in `tf.convert_to_tensor(x)` being identical to '
          '`x.mean()`.'.format(type(self), self))
    with self._name_scope('value'):
      self._concrete_value = (self._convert_to_tensor_fn(self)
                              if callable(self._convert_to_tensor_fn)
                              else self._convert_to_tensor_fn)
      if (not tf.is_tensor(self._concrete_value) and
          not isinstance(self._concrete_value,
                         composite_tensor.CompositeTensor)):
        self._concrete_value = tfd._convert_to_tensor(
            value=self._concrete_value,
            name=name or 'concrete_value',
            dtype=dtype,
            dtype_hint=self.dtype)
  return self._concrete_value
  # pylint: enable=protected-access


class _TensorCoercibleMeta(type):
  """A factory for classes which will act like Tensors."""

  def __new__(mcs, name, bases, attrs):
    # Embed Tensor-like attributes into the mcs class.
    attrs.update(_get_tensor_like_attributes())
    attrs['_value'] = _value
    cls = super(_TensorCoercibleMeta, mcs).__new__(mcs, name, bases, attrs)
    def _tensorize(d, dtype=None, name=None, as_ref=False):
      return d._value(dtype, name, as_ref)  # pylint: disable=protected-access
    tf.register_tensor_conversion_function(cls, conversion_func=_tensorize)
    return cls


# Define mixin type because Distribution already has its own metaclass.
class _DistributionAndTensorCoercibleMeta(
    type(tfd.Distribution), _TensorCoercibleMeta):
  pass


@six.add_metaclass(_DistributionAndTensorCoercibleMeta)
class _TensorCoercible(tfd.Distribution):
  """Docstring."""

  def __new__(cls, distribution, convert_to_tensor_fn=tfd.Distribution.sample):
    if isinstance(distribution, cls):
      return distribution
    if not isinstance(distribution, tfd.Distribution):
      raise TypeError('`distribution` argument must be a '
                      '`tfd.Distribution` instance; '
                      'saw "{}" of type "{}".'.format(
                          distribution, type(distribution)))
    self = copy.copy(distribution)
    distcls = distribution.__class__
    self.__class__ = type(distcls.__name__, (cls, distcls), {})
    self._concrete_value = None  # pylint: disable=protected-access
    self._convert_to_tensor_fn = convert_to_tensor_fn  # pylint: disable=protected-access
    return self

  def __init__(self,
               distribution,
               convert_to_tensor_fn=tfd.Distribution.sample):
    pass
