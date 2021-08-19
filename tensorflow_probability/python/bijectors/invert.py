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
"""Invert bijector."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v2 as tf

from tensorflow_probability.python.bijectors import bijector as bijector_lib
from tensorflow_probability.python.internal import parameter_properties

__all__ = [
    'Invert',
]


class _Invert(bijector_lib.Bijector):
  """Bijector which inverts another Bijector.

  Example Use: [ExpGammaDistribution (see Background & Context)](
  https://reference.wolfram.com/language/ref/ExpGammaDistribution.html)
  models `Y=log(X)` where `X ~ Gamma`.

  ```python
  exp_gamma_distribution = TransformedDistribution(
    distribution=Gamma(concentration=1., rate=2.),
    bijector=bijector.Invert(bijector.Exp())
  ```

  """

  def __init__(self, bijector, validate_args=False, parameters=None, name=None):
    """Creates a `Bijector` which swaps the meaning of `inverse` and `forward`.

    Note: An inverted bijector's `inverse_log_det_jacobian` is often more
    efficient if the base bijector implements `_forward_log_det_jacobian`. If
    `_forward_log_det_jacobian` is not implemented then the following code is
    used:

    ```python
    y = self.inverse(x, **kwargs)
    return -self.inverse_log_det_jacobian(y, **kwargs)
    ```

    Args:
      bijector: Bijector instance.
      validate_args: Python `bool` indicating whether arguments should be
        checked for correctness.
      parameters: Locals dict captured by subclass constructor, to be used for
        copy/slice re-instantiation operators.
      name: Python `str`, name given to ops managed by this object.
    """

    parameters = dict(locals()) if parameters is None else parameters
    if not bijector._is_injective:  # pylint: disable=protected-access
      raise NotImplementedError(
          'Invert is not implemented for non-injective bijectors.')

    name = name or '_'.join(['invert', bijector.name])
    with tf.name_scope(name) as name:
      self._bijector = bijector
      super(_Invert, self).__init__(
          forward_min_event_ndims=bijector.inverse_min_event_ndims,
          inverse_min_event_ndims=bijector.forward_min_event_ndims,
          dtype=bijector.dtype,
          is_constant_jacobian=bijector.is_constant_jacobian,
          validate_args=validate_args,
          parameters=parameters,
          name=name)

  @classmethod
  def _parameter_properties(cls, dtype):
    return dict(
        bijector=parameter_properties.BatchedComponentProperties(
            event_ndims=(
                lambda self, x_event_ndims: self.bijector.inverse_event_ndims(  # pylint: disable=g-long-lambda
                    x_event_ndims))))

  def forward_event_shape(self, input_shape):
    return self.bijector.inverse_event_shape(input_shape)

  def forward_event_shape_tensor(self, input_shape):
    return self.bijector.inverse_event_shape_tensor(input_shape)

  def inverse_event_shape(self, output_shape):
    return self.bijector.forward_event_shape(output_shape)

  def inverse_event_shape_tensor(self, output_shape):
    return self.bijector.forward_event_shape_tensor(output_shape)

  @property
  def bijector(self):
    return self._bijector

  @property
  def _is_permutation(self):
    return self.bijector._is_permutation  # pylint: disable=protected-access

  @property
  def _parts_interact(self):
    return self.bijector._parts_interact  # pylint: disable=protected-access

  def _internal_is_increasing(self, **kwargs):
    return self.bijector._internal_is_increasing(**kwargs)  # pylint: disable=protected-access

  def forward(self, x, **kwargs):
    return self.bijector.inverse(x, **kwargs)

  def inverse(self, y, **kwargs):
    return self.bijector.forward(y, **kwargs)

  def inverse_log_det_jacobian(self, y, event_ndims=None, **kwargs):
    return self.bijector.forward_log_det_jacobian(y, event_ndims, **kwargs)

  def forward_log_det_jacobian(self, x, event_ndims=None, **kwargs):
    return self.bijector.inverse_log_det_jacobian(x, event_ndims, **kwargs)

  def forward_dtype(self, dtype=bijector_lib.UNSPECIFIED, **kwargs):
    return self.bijector.inverse_dtype(dtype, **kwargs)

  def inverse_dtype(self, dtype=bijector_lib.UNSPECIFIED, **kwargs):
    return self.bijector.forward_dtype(dtype, **kwargs)

  def inverse_event_ndims(self, event_ndims, **kwargs):
    return self.bijector.forward_event_ndims(event_ndims, **kwargs)

  def forward_event_ndims(self, event_ndims, **kwargs):
    return self.bijector.inverse_event_ndims(event_ndims, **kwargs)

  def __str__(self):
    return '{}, bijector={})'.format(
        super().__str__()[:-1],  # Strip final `)`.
        type(self.bijector).__name__)

  def __repr__(self):
    return '{} bijector={}>'.format(
        super().__repr__()[:-1],  # Strip final `>`.
        repr(self.bijector))


class Invert(_Invert, bijector_lib.AutoCompositeTensorBijector):

  def __new__(cls, *args, **kwargs):
    """Returns an `_Invert` instance if `bijector` is not a `CompositeTensor."""
    if cls is Invert:
      if args:
        bijector = args[0]
      elif 'bijector' in kwargs:
        bijector = kwargs['bijector']
      else:
        raise TypeError('`Invert.__new__()` is missing argument `bijector`.')

      if not isinstance(bijector, tf.__internal__.CompositeTensor):
        return _Invert(*args, **kwargs)
    return super(Invert, cls).__new__(cls)


Invert.__doc__ = _Invert.__doc__ + '/n' + (
    'When an `Invert` bijector is constructed, if its `bijector` arg is not a '
    '`CompositeTensor` instance, an `_Invert` instance is returned instead. '
    'Bijectors subclasses that inherit from `Invert` will also inherit from '
    ' `CompositeTensor`.')
