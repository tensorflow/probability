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
"""Base classes for building neural networks."""
import sys

import tensorflow.compat.v2 as tf

from tensorflow_probability.python.experimental.nn.util import utils as nn_util_lib
from tensorflow_probability.python.internal import name_util


__all__ = [
    'Layer',
    'Sequential',
]


class Layer(tf.Module):
  """A `callable` `tf.Module`."""

  def __init__(self, also_track=None, validate_args=False, name=None):
    name = name_util.strip_invalid_chars(name or type(self).__name__)
    self._also_track = [] if also_track is None else [also_track]
    super(Layer, self).__init__(name=name)
    self._trace = False
    self._validate_args = validate_args

  @property
  def also_track(self):
    return list(self._also_track)

  @property
  def validate_args(self):
    """Python `bool` indicating possibly expensive checks are enabled."""
    return self._validate_args

  def summary(self):
    return nn_util_lib.variables_summary(self.variables, self.name)

  def save(self, filename):
    return nn_util_lib.variables_save(filename, self.variables)

  def load(self, filename):
    return nn_util_lib.variables_load(filename, self.variables)

  def __repr__(self):
    return '<{}: name={}>'.format(type(self).__name__, self.name)


class Sequential(Layer):
  """A `Layer` characterized by iteratively given functions."""

  def __init__(self, layers, also_track=None, validate_args=False, name=None):
    layers = tuple(layers)
    if not layers:
      raise tf.errors.InvalidArgumentError(
          'Argument `layers` must contain at least one element.')
    name = name or '_'.join([_try_get_name(x) for x in layers])
    self._layers = tuple(layers)
    super(Sequential, self).__init__(
        also_track=also_track, validate_args=validate_args, name=name)

  def set_trace(self, trace):
    self._trace = bool(trace)
    return self

  @property
  def layers(self):
    return self._layers

  def __call__(self, inputs, **kwargs):
    if callable(inputs):
      return Sequential([inputs, self], **kwargs)
    x = inputs
    if self._trace:
      _trace(self, x, -1)
    for i, layer in enumerate(self.layers):
      x = _try_call(layer, [x], kwargs)
      if self._trace:
        _trace(layer, x, i)
    return x

  def __getitem__(self, i):
    r = Sequential(self.layers[i], name=self.name)
    r._also_track = self._also_track  # pylint: disable=protected-access
    return r


class KernelBiasLayer(Layer):
  """Linear layer."""

  def __init__(self,
               kernel,
               bias,
               apply_kernel_fn,
               activation_fn=None,
               dtype=tf.float32,
               validate_args=False,
               name=None):
    self._kernel = kernel
    self._bias = bias
    self._activation_fn = activation_fn
    self._apply_kernel_fn = apply_kernel_fn
    self._dtype = dtype
    super(KernelBiasLayer, self).__init__(
        validate_args=validate_args, name=name)

  @property
  def dtype(self):
    return self._dtype

  @property
  def kernel(self):
    return self._kernel

  @property
  def bias(self):
    return self._bias

  @property
  def activation_fn(self):
    return self._activation_fn

  def __call__(self, x):
    x = tf.convert_to_tensor(x, dtype_hint=self.dtype, name='x')
    y = x
    if self.kernel is not None:
      y = self._apply_kernel_fn(y, self.kernel)
    if self.bias is not None:
      y = y + self.bias
    if self.activation_fn is not None:
      y = self.activation_fn(y)  # pylint: disable=not-callable
    return y


def _try_call(fn, args, kwargs):
  """Convenience function for evaluating argument `fn`."""
  try:
    if fn is None:
      return args[0]
    try:
      return fn(*args, **kwargs)
    except TypeError:
      # Don't return from here or else we'll pick up a nested exception.
      # Seeing TypeError here isn't really an exception since it only means we
      # need to call `fn` differently).
      pass
    return fn(*args)
  except:
    print('------ EXCEPTION in {} ------'.format(_try_get_name(fn)))
    raise


def _try_get_name(fn, name_fallback='unknown'):
  return str(getattr(fn, '__name__', None) or
             getattr(fn, 'name', None) or
             getattr(type(fn), '__name__', name_fallback))


def _trace(layer, x, i):
  name = _try_get_name(layer)
  z = tf.nest.map_structure(lambda x_: '{:14} {:<24} {:>10}'.format(  # pylint: disable=g-long-lambda
      _try_get_name(x_),
      str(list(getattr(x_, 'shape', '?'))),
      _try_get_name(getattr(x_, 'dtype', x_), '?')), x)
  print('--- TRACE{:02}:  {:<24} {}'.format(i, name, z))
  sys.stdout.flush()
