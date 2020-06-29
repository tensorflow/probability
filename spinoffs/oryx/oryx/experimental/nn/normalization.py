# Copyright 2020 The TensorFlow Probability Authors.
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
"""Contains building blocks for normalization layers."""

import collections
from jax import lax
from jax import random
from jax.experimental import stax
import jax.numpy as np

from oryx.core import kwargs_util
from oryx.experimental.nn import base

__all__ = [
    'BatchNorm',
]


class BatchNormParams(collections.namedtuple(
    'BatchNormParams', ('beta', 'gamma'))):
  pass


class BatchNormState(collections.namedtuple(
    'BatchNormState',
    ('moving_mean', 'moving_var'))):
  pass


class BatchNormInfo(collections.namedtuple(
    'BatchNormInfo',
    ('axis', 'epsilon', 'center', 'scale', 'decay', 'shape'))):
  pass


class BatchNorm(base.Layer):
  """Layer for Batch Normalization."""

  @classmethod
  def initialize(cls, key, in_spec, axis=(0, 1), momentum=0.99,
                 epsilon=1e-5, center=True, scale=True,
                 beta_init=stax.zeros, gamma_init=stax.ones):
    in_shape = in_spec.shape
    axis = (axis,) if np.isscalar(axis) else axis
    decay = 1.0 - momentum
    shape = tuple(d for i, d in enumerate(in_shape) if i not in axis)
    moving_shape = tuple(1 if i in axis else d for i, d in enumerate(in_shape))
    k1, k2, k3, k4 = random.split(key, 4)
    beta = base.create_parameter(k1, shape, init=beta_init) if center else ()
    gamma = base.create_parameter(k2, shape, init=gamma_init) if scale else ()
    moving_mean = base.create_parameter(k3, moving_shape, init=stax.zeros)
    moving_var = base.create_parameter(k4, moving_shape, init=stax.ones)
    params = BatchNormParams(beta, gamma)
    info = BatchNormInfo(axis, epsilon, center, scale, decay, in_shape)
    state = BatchNormState(moving_mean, moving_var)
    return base.LayerParams(params, info, state)

  @classmethod
  def spec(cls, in_spec, axis=(0, 1), momentum=0.99,
           epsilon=1e-5, center=True, scale=True,
           beta_init=stax.zeros, gamma_init=stax.ones):
    return in_spec

  def _call_and_update_batched(self, *args, has_rng=False, **kwargs):
    if has_rng:
      rng, args = args[0], args[1:]
      kwargs = dict(kwargs, rng=rng)
    call_kwargs = kwargs_util.filter_kwargs(self._call_batched, kwargs)
    update_kwargs = kwargs_util.filter_kwargs(self._update_batched, kwargs)
    layer = self.replace(state=lax.stop_gradient(self.state))
    return (layer._call_batched(*args, **call_kwargs),  # pylint: disable=protected-access
            layer._update_batched(*args, **update_kwargs))  # pylint: disable=protected-access

  def _call(self, x, training=True):
    if len(x.shape) != len(self.info.shape):
      raise ValueError('Need to `jax.vmap` in order to batch')
    if training:
      # BatchNorm on a single example while training=True is a no-op
      # The tracer will pass through this and hand off to _call_batched
      return x
    return self._call_batched(x[np.newaxis], training=False)[0]

  def _call_batched(self, x, training=True):
    params, info, state = self.params, self.info, self.state
    beta, gamma = params.beta, params.gamma
    axis = (0,) + tuple(a + 1 for a in info.axis)
    epsilon, center, scale = info.epsilon, info.center, info.scale
    ed = tuple(None if i in axis else slice(None) for i in range(np.ndim(x)))
    if center:
      beta = beta[ed]
    if scale:
      gamma = gamma[ed]
    if training:
      mean = np.mean(x, axis, keepdims=True)
      var = np.mean(x**2, axis, keepdims=True) - mean**2
    else:
      mean, var = state.moving_mean, state.moving_var
    z = (x - mean) / np.sqrt(var + epsilon)
    if center and scale:
      output = gamma * z + beta
    elif center:
      output = z + beta
    elif scale:
      output = gamma * z
    else:
      output = z
    return output

  def _update(self, x):
    return self._update_axis(x, self.info.axis)

  def _update_batched(self, x):
    axis = self.info.axis
    axis_diff = np.ndim(x) - len(self.info.shape)
    axis = tuple(range(axis_diff)) + tuple(a + axis_diff for a in axis)
    return self._update_axis(x, axis)

  def _update_axis(self, x, axis):
    info, state = self.info, self.state
    decay = info.decay

    mean = np.mean(x, axis, keepdims=True)
    var = np.mean(x**2, axis, keepdims=True) - mean**2
    mean, var = mean[0], var[0]

    moving_mean, moving_var = state.moving_mean, state.moving_var
    moving_mean -= (moving_mean - mean) * decay
    moving_var -= (moving_var - var) * decay

    new_state = BatchNormState(moving_mean, moving_var)
    return self.replace(state=new_state)
