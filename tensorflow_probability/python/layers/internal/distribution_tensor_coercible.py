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

import six

import tensorflow.compat.v2 as tf

from tensorflow_probability.python.distributions import distribution as tfd
from tensorflow_probability.python.distributions import kullback_leibler
from tensorflow_probability.python.internal import nest_util
from tensorflow_probability.python.internal import parameter_properties
from tensorflow_probability.python.util.deferred_tensor import TensorMetaClass
from tensorflow.python.framework import composite_tensor  # pylint: disable=g-direct-tensorflow-import
from tensorflow.python.trackable import data_structures  # pylint: disable=g-direct-tensorflow-import


__all__ = []  # We intend nothing public.

_NOT_FOUND = object()


# Define mixin type because Distribution already has its own metaclass.
class _DistributionAndTensorCoercibleMeta(type(tfd.Distribution),
                                          TensorMetaClass):
  pass


# TODO(b/182603117): Convert _TensorCoercible to AutoCompositeTensor, or decide
# not to.
@six.add_metaclass(_DistributionAndTensorCoercibleMeta)
class _TensorCoercible(tfd.Distribution):
  """Docstring."""

  def __init__(self,
               distribution,
               convert_to_tensor_fn=tfd.Distribution.sample):
    self._concrete_value = None  # pylint: disable=protected-access
    self._convert_to_tensor_fn = convert_to_tensor_fn  # pylint: disable=protected-access
    self.tensor_distribution = distribution
    super(_TensorCoercible, self).__init__(
        dtype=distribution.dtype,
        reparameterization_type=distribution.reparameterization_type,
        validate_args=distribution.validate_args,
        allow_nan_stats=distribution.allow_nan_stats,
        parameters=distribution.parameters)

  def __setattr__(self, name, value):
    """Support self.foo = trackable syntax.

    Redefined from `tensorflow/python/trackable/tracking.py` to avoid
    calling `getattr`, which causes an infinite loop.

    Args:
      name: str, name of the attribute to be set.
      value: value to be set.
    """
    if vars(self).get(name, _NOT_FOUND) is value:
      return

    if vars(self).get('_self_setattr_tracking', True):
      value = data_structures.sticky_attribute_assignment(
          trackable=self, value=value, name=name)
    object.__setattr__(self, name, value)

  def __getattr__(self, name):
    # If the attribute is set in the _TensorCoercible object, return it. This
    # ensures that direct calls to `getattr` behave as expected.
    if name in vars(self):
      return vars(self)[name]
    # Look for the attribute in `tensor_distribution`, unless it's a `_tracking`
    # attribute accessed directly by `getattr` in the `Trackable` base class, in
    # which case the default passed to `getattr` should be returned.
    if 'tensor_distribution' in vars(self) and '_tracking' not in name:
      return getattr(vars(self)['tensor_distribution'], name)
    # Otherwise invoke `__getattribute__`, which will return the default passed
    # to `getattr` if the attribute was not found.
    return self.__getattribute__(name)

  @classmethod
  def _parameter_properties(cls, dtype, num_classes=None):
    return dict(distribution=parameter_properties.BatchedComponentProperties())

  # pylint: disable=protected-access
  def _batch_shape_tensor(self, **parameter_kwargs):
    return self.tensor_distribution._batch_shape_tensor(**parameter_kwargs)

  def _batch_shape(self):
    return self.tensor_distribution._batch_shape()

  def _event_shape_tensor(self):
    return self.tensor_distribution._event_shape_tensor()

  def _event_shape(self):
    return self.tensor_distribution._event_shape()

  def sample(self, sample_shape=(), seed=None, name='sample', **kwargs):
    return self.tensor_distribution.sample(
        sample_shape=sample_shape, seed=seed, name=name, **kwargs)

  def _log_prob(self, value, **kwargs):
    return self.tensor_distribution._log_prob(value, **kwargs)

  def _prob(self, value, **kwargs):
    return self.tensor_distribution._prob(value, **kwargs)

  def _log_cdf(self, value, **kwargs):
    return self.tensor_distribution._log_cdf(value, **kwargs)

  def _cdf(self, value, **kwargs):
    return self.tensor_distribution._cdf(value, **kwargs)

  def _log_survival_function(self, value, **kwargs):
    return self.tensor_distribution._log_survival_function(value, **kwargs)

  def _survival_function(self, value, **kwargs):
    return self.tensor_distribution._survival_function(value, **kwargs)

  def _entropy(self, **kwargs):
    return self.tensor_distribution._entropy(**kwargs)

  def _mean(self, **kwargs):
    return self.tensor_distribution._mean(**kwargs)

  def _quantile(self, value, **kwargs):
    return self.tensor_distribution._quantile(value, **kwargs)

  def _variance(self, **kwargs):
    return self.tensor_distribution._variance(**kwargs)

  def _stddev(self, **kwargs):
    return self.tensor_distribution._stddev(**kwargs)

  def _covariance(self, **kwargs):
    return self.tensor_distribution._covariance(**kwargs)

  def _mode(self, **kwargs):
    return self.tensor_distribution._mode(**kwargs)

  def _default_event_space_bijector(self, *args, **kwargs):
    return self.tensor_distribution._default_event_space_bijector(
        *args, **kwargs)

  def _parameter_control_dependencies(self, is_init):
    return self.tensor_distribution._parameter_control_dependencies(is_init)

  @property
  def shape(self):
    return self._shape

  def _shape(self):
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
        self._concrete_value = (
            self._convert_to_tensor_fn(self.tensor_distribution)
            if callable(self._convert_to_tensor_fn)
            else self._convert_to_tensor_fn)
        if (not tf.is_tensor(self._concrete_value) and
            not isinstance(self._concrete_value,
                           composite_tensor.CompositeTensor)):
          self._concrete_value = nest_util.convert_to_nested_tensor(  # pylint: disable=protected-access
              self._concrete_value,
              name=name or 'concrete_value',
              dtype=dtype,
              dtype_hint=self.tensor_distribution.dtype)
    return self._concrete_value


@kullback_leibler.RegisterKL(_TensorCoercible, tfd.Distribution)
def _kl_tensor_coercible_distribution(a, b, name=None):
  return kullback_leibler.kl_divergence(a.tensor_distribution, b, name=name)


@kullback_leibler.RegisterKL(tfd.Distribution, _TensorCoercible)
def _kl_distribution_tensor_coercible(a, b, name=None):
  return kullback_leibler.kl_divergence(a, b.tensor_distribution, name=name)
