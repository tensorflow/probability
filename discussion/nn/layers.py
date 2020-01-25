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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp

from discussion.nn import util as nn_util_lib
from tensorflow_probability.python.internal import name_util


__all__ = [
    'Lambda',
    'Layer',
    'Sequential',
]

tfd = tfp.distributions


class Layer(tf.Module):
  """A `callable` `tf.Module` characterized by `eval(input)`."""

  def __init__(self, name=None):
    name = name_util.strip_invalid_chars(name or type(self).__name__)
    super(Layer, self).__init__(name=name)
    self._extra_loss = None
    self._extra_result = None

  @property
  def extra_loss(self):
    return self._extra_loss

  @property
  def extra_result(self):
    return self._extra_result

  def eval(self, inputs, is_training=True, **kwargs):
    self._set_extra_loss(None)
    self._set_extra_result(None)
    return inputs

  def summary(self):
    return nn_util_lib.variables_summary(self.variables, self.name)

  def save(self, filename):
    return nn_util_lib.variables_save(filename, self.variables)

  def load(self, filename):
    return nn_util_lib.variables_load(filename, self.variables)

  def __call__(self, inputs, **kwargs):
    if callable(inputs):
      return Sequential([inputs, self], **kwargs)
    self._extra_loss = self._extra_result = None
    y = self.eval(inputs, **kwargs)
    # TODO(jvdillon): Consider adding provenance.
    # y.__tfp_nn_provenance = self
    return y

  def __repr__(self):
    return '<{}: name={}>'.format(type(self).__name__, self.name)

  def _set_extra_result(self, value):
    self._extra_result = value

  def _set_extra_loss(self, value):
    self._extra_loss = value


class Sequential(Layer):
  """A `Layer` characterized by iteratively given functions."""

  def __init__(self, layers, also_track=None, name=None):
    layers = tuple(layers)
    if not layers:
      raise tf.errors.InvalidArgumentError(
          'Argument `layers` must contain at least one element.')
    name = name or '_'.join([_try_get_name(x) for x in layers])
    self._layers = layers
    self._also_track = also_track
    super(Sequential, self).__init__(name=name)

  @property
  def layers(self):
    return self._layers

  def eval(self, inputs, is_training=True, **kwargs):
    kwargs.update({'is_training': is_training})
    all_extras = []
    x = inputs
    for layer in self.layers:
      _try_set_extra_results(layer, loss=None, result=None)
      x = _try_call(layer, [x], kwargs)
      extra_loss, extra_result = _try_get_extra_results(layer)
      all_extras.append((extra_loss, extra_result))
      _try_set_extra_results(layer, loss=extra_loss, result=extra_result)
    non_none_extra_losses = [extra_loss for (extra_loss, _) in all_extras
                             if extra_loss is not None]
    sum_extra_losses = (sum(non_none_extra_losses)
                        if non_none_extra_losses else None)
    self._set_extra_loss(sum_extra_losses)
    self._set_extra_result(None)
    return x

  def __getitem__(self, i):
    return Sequential(self.layers[i])


class Lambda(Layer):
  """A `Layer` which can be defined inline."""

  def __init__(self,
               eval_fn=None,
               extra_loss_fn=None,
               also_track=None,
               name=None):
    if not callable(eval_fn):
      raise tf.errors.InvalidArgumentError(
          'Argument `eval_fn` must be `callable`.')
    name = name or _try_get_name(eval_fn)
    self._eval_fn = eval_fn
    self._extra_loss_fn = extra_loss_fn
    self._also_track = also_track
    super(Lambda, self).__init__(name=name)

  def eval(self, inputs, is_training=True, **kwargs):
    kwargs.update({'is_training': is_training})
    if self._eval_fn is not None:
      r = _try_call(self._eval_fn, [inputs], kwargs)
    else:
      r = inputs
    self._last_call = r  # For variable tracking purposes.
    self._set_extra_loss(None if self._extra_loss_fn is None else
                         _try_call(self._extra_loss_fn, [r], kwargs))
    self._set_extra_result(None)
    return r


class KernelBiasLayer(Layer):
  """Linear layer."""

  def __init__(self,
               kernel,
               bias,
               apply_kernel_fn,
               dtype=tf.float32,
               name=None):
    self._kernel = kernel
    self._bias = bias
    self._apply_kernel_fn = apply_kernel_fn
    self._dtype = dtype
    super(KernelBiasLayer, self).__init__(name=name)

  @property
  def dtype(self):
    return self._dtype

  @property
  def kernel(self):
    return self._kernel

  @property
  def bias(self):
    return self._bias

  def eval(self, x, is_training=True):
    x = tf.convert_to_tensor(x, dtype_hint=self.dtype, name='x')
    y = x
    if self.kernel is not None:
      y = self._apply_kernel_fn(y, self.kernel)
    if self.bias is not None:
      y = y + self.bias
    return y


def _try_set_extra_results(layer, loss, result):
  """Convenience function for maybe calling `_set_extra_result`."""
  set_fn = getattr(layer, '_set_extra_loss', None)
  if callable(set_fn):
    set_fn(loss)
  set_fn = getattr(layer, '_set_extra_result', None)
  if callable(set_fn):
    set_fn(result)


def _try_get_extra_results(layer):
  """Convenience function for getting side data."""
  return (
      getattr(layer, 'extra_loss', None),
      getattr(layer, 'extra_result', None),
  )


def _try_call(fn, args, kwargs):
  """Convenience function for evaluating argument `fn`."""
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


def _try_get_name(fn, name_fallback='unknown'):
  return str(getattr(fn, 'name', None) or
             getattr(fn, '__name__', name_fallback))
