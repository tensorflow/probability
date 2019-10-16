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
import six

import tensorflow.compat.v2 as tf

from tensorflow_probability.python.distributions import distribution as tfd
from tensorflow_probability.python.util.deferred_tensor import TensorMetaClass
from tensorflow.python.framework import composite_tensor  # pylint: disable=g-direct-tensorflow-import


__all__ = []  # We intend nothing public.


# Define mixin type because Distribution already has its own metaclass.
class _DistributionAndTensorCoercibleMeta(type(tfd.Distribution),
                                          TensorMetaClass):
  pass


@six.add_metaclass(_DistributionAndTensorCoercibleMeta)
class _TensorCoercible(tfd.Distribution):
  """Docstring."""

  registered_class_list = {}

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
    self_class = _TensorCoercible.registered_class_list.get(distcls)
    if not self_class:
      self_class = type(distcls.__name__, (cls, distcls), {})
      _TensorCoercible.registered_class_list[distcls] = self_class
    self.__class__ = self_class
    return self

  def __init__(self,
               distribution,
               convert_to_tensor_fn=tfd.Distribution.sample):
    self._concrete_value = None  # pylint: disable=protected-access
    self._convert_to_tensor_fn = convert_to_tensor_fn  # pylint: disable=protected-access

  @property
  def shape(self):
    return (tf.TensorShape(None) if self._concrete_value is None
            else self._concrete_value.shape)

  def _value(self, dtype=None, name=None, as_ref=False):
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
      with self._name_and_control_scope('value'):
        self._concrete_value = (self._convert_to_tensor_fn(self)
                                if callable(self._convert_to_tensor_fn)
                                else self._convert_to_tensor_fn)
        if (not tf.is_tensor(self._concrete_value) and
            not isinstance(self._concrete_value,
                           composite_tensor.CompositeTensor)):
          self._concrete_value = tfd._convert_to_tensor(  # pylint: disable=protected-access
              self._concrete_value,
              name=name or 'concrete_value',
              dtype=dtype,
              dtype_hint=self.dtype)
    return self._concrete_value
