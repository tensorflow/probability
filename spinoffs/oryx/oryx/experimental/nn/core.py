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
"""Contains important layers for neural network construction."""

import abc
import collections
import jax
from jax import lax
from jax import random
from jax.experimental import stax
import jax.numpy as np

from oryx.core import state
from oryx.experimental.nn import base

__all__ = [
    'Dense',
    'Dropout',
    'Activation',
    'Relu',
    'Tanh',
    'Softplus',
    'LogSoftmax',
    'Softmax',
]


DenseParams = collections.namedtuple('DenseParams', ['kernel', 'bias'])


class Dense(base.Layer):
  """Dense layer used for building neural networks."""

  @classmethod
  def initialize(cls, rng, in_spec, dim_out,
                 kernel_init=stax.glorot(),
                 bias_init=stax.zeros):
    """Initializes Dense Layer.

    Args:
      rng: Random key.
      in_spec: Input Spec.
      dim_out: Output dimensions.
      kernel_init: Kernel initialization function.
      bias_init: Bias initialization function.

    Returns:
      Tuple with the output shape and the LayerParams.
    """
    if rng is None:
      raise ValueError('Need valid RNG to instantiate Dense layer.')
    dim_in = in_spec.shape[-1]
    k1, k2 = random.split(rng)
    params = DenseParams(
        base.create_parameter(k1, (dim_in, dim_out), init=kernel_init),
        base.create_parameter(k2, (dim_out,), init=bias_init)
    )
    return base.LayerParams(params)

  @classmethod
  def spec(cls, in_spec, dim_out, **kwargs):
    in_shape = in_spec.shape
    out_shape = in_shape[:-1] + (dim_out,)
    return state.Shape(out_shape, dtype=in_spec.dtype)

  def _call(self, x):
    """Applies Dense multiplication of the params with the input x."""
    params = self.params
    kernel, bias = params.kernel, params.bias
    return np.dot(x, kernel) + bias

  @property
  def dim_in(self):
    """Input dimensions."""
    return self.params.kernel.shape[0]

  @property
  def dim_out(self):
    """Output dimensions."""
    return self.params.kernel.shape[1]

  def __str__(self):
    """String representation of the Layer."""
    return '{}({})'.format(self.__class__.__name__,
                           self.dim_out)


class DropoutInfo(collections.namedtuple(
    'DropoutInfo', ('rate'))):
  pass


class Dropout(base.Layer):
  """Dropout layer used for building neural networks."""

  @classmethod
  def initialize(cls, rng, in_shape, rate):
    del in_shape
    layer_params = base.LayerParams(info=DropoutInfo(rate))
    return layer_params

  @classmethod
  def spec(cls, in_spec, _):
    return in_spec

  def _call(self, x, training=True, rng=None):
    info = self.info
    if training:
      if rng is None:
        raise ValueError('rng is required when training is True')
      # Using tie_in to avoid materializing constants
      keep = lax.tie_in(x, random.bernoulli(rng, info.rate, x.shape))
      return np.where(keep, x / info.rate, 0)
    else:
      return x


class Activation(base.Layer, metaclass=abc.ABCMeta):
  """Parent abstract class for activation functions."""

  @abc.abstractmethod
  def _activate(self, x):
    raise NotImplementedError

  @classmethod
  def initialize(cls, rng, in_shape):
    """Initializes Activation Layer."""
    del in_shape, rng
    return base.LayerParams()

  @classmethod
  def spec(cls, in_spec):
    return in_spec

  def _call(self, x):
    """Applies Activation on the input x."""
    return self._activate(x)


class Relu(Activation):

  def _activate(self, x):
    return jax.nn.relu(x)


class Tanh(Activation):

  def _activate(self, x):
    return np.tanh(x)


class Softplus(Activation):

  def _activate(self, x):
    return jax.nn.softplus(x)


class LogSoftmax(Activation):

  def _activate(self, x):
    return jax.nn.log_softmax(x, axis=-1)


class Softmax(Activation):

  def _activate(self, x):
    return jax.nn.softmax(x, axis=-1)
