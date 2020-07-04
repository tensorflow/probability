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
"""Contains building blocks for convolutional neural networks."""

import collections
import itertools
from jax import lax
from jax import random
from jax.experimental import stax
import jax.numpy as np

from oryx.core import state
from oryx.experimental.nn import base

__all__ = [
    'conv_info',
    'Conv',
    'Deconv',
]


DIMENSION_NUMBERS = ('NHWC', 'HWIO', 'NHWC')
ConvParams = collections.namedtuple('ConvParams', ['kernel', 'bias'])
ConvInfo = collections.namedtuple('ConvInfo',
                                  ['strides', 'padding', 'one', 'use_bias'])


def conv_info(in_shape, out_chan, filter_shape,
              strides=None, padding='VALID',
              kernel_init=None, bias_init=stax.randn(1e-6),
              transpose=False):
  """Returns parameters and output shape information given input shapes."""
  # Essentially the `stax` implementation
  if len(in_shape) != 3:
    raise ValueError('Need to `jax.vmap` in order to batch')
  in_shape = (1,) + in_shape
  lhs_spec, rhs_spec, out_spec = DIMENSION_NUMBERS
  one = (1,) * len(filter_shape)
  strides = strides or one
  kernel_init = kernel_init or stax.glorot(
      rhs_spec.index('O'), rhs_spec.index('I'))
  filter_shape_iter = iter(filter_shape)
  kernel_shape = tuple([out_chan if c == 'O' else
                        in_shape[lhs_spec.index('C')] if c == 'I' else
                        next(filter_shape_iter) for c in rhs_spec])
  if transpose:
    out_shape = lax.conv_transpose_shape_tuple(
        in_shape, kernel_shape, strides, padding, DIMENSION_NUMBERS)
  else:
    out_shape = lax.conv_general_shape_tuple(
        in_shape, kernel_shape, strides, padding, DIMENSION_NUMBERS)
  bias_shape = [out_chan if c == 'C' else 1 for c in out_spec]
  bias_shape = tuple(itertools.dropwhile(lambda x: x == 1, bias_shape))
  out_shape = out_shape[1:]
  shapes = (out_shape, kernel_shape, bias_shape)
  inits = (kernel_init, bias_init)
  return shapes, inits, (strides, padding, one)


class Conv(base.Layer):
  """Neural network layer for 2D convolution."""

  @classmethod
  def initialize(cls, key, in_spec, out_chan, filter_shape,
                 strides=None, padding='VALID',
                 kernel_init=None, bias_init=stax.randn(1e-6),
                 use_bias=True):
    in_shape = in_spec.shape
    shapes, inits, (strides, padding, one) = conv_info(
        in_shape, out_chan, filter_shape,
        strides=strides, padding=padding,
        kernel_init=kernel_init, bias_init=bias_init
    )
    info = ConvInfo(strides, padding, one, use_bias)
    _, kernel_shape, bias_shape = shapes
    kernel_init, bias_init = inits
    k1, k2 = random.split(key)
    if use_bias:
      params = ConvParams(
          base.create_parameter(k1, kernel_shape, init=kernel_init),
          base.create_parameter(k2, bias_shape, init=bias_init),
      )
    else:
      params = ConvParams(
          base.create_parameter(k1, kernel_shape, init=kernel_init),
          None
      )
    return base.LayerParams(params, info=info)

  @classmethod
  def spec(cls, in_spec, out_chan, filter_shape,
           strides=None, padding='VALID',
           kernel_init=None, bias_init=stax.randn(1e-6),
           use_bias=True):
    del use_bias
    in_shape = in_spec.shape
    shapes, _, _ = conv_info(
        in_shape, out_chan, filter_shape,
        strides=strides, padding=padding,
        kernel_init=kernel_init, bias_init=bias_init
    )
    return state.Shape(shapes[0], dtype=in_spec.dtype)

  def _call_batched(self, x):
    params, info = self.params, self.info
    result = lax.conv_general_dilated(x, params.kernel,
                                      info.strides, info.padding,
                                      info.one, info.one,
                                      DIMENSION_NUMBERS)
    if info.use_bias:
      result += params.bias
    return result

  def _call(self, x):
    """Applies 2D convolution of the params with the input x."""
    if len(x.shape) != 3:
      raise ValueError('Need to `jax.vmap` in order to batch: {}'.format(
          x.shape))
    result = self._call_batched(x[np.newaxis])
    return result[0]


class Deconv(base.Layer):
  """Neural network layer for 2D transposed convolution."""

  @classmethod
  def initialize(cls, key, in_spec, out_chan, filter_shape,
                 strides=None, padding='VALID',
                 kernel_init=None, bias_init=stax.randn(1e-6),
                 use_bias=True):
    in_shape = in_spec.shape
    shapes, inits, (strides, padding, one) = conv_info(
        in_shape, out_chan, filter_shape,
        strides=strides, padding=padding,
        kernel_init=kernel_init, bias_init=bias_init,
        transpose=True
    )
    info = ConvInfo(strides, padding, one, use_bias)
    _, kernel_shape, bias_shape = shapes
    kernel_init, bias_init = inits
    k1, k2 = random.split(key)
    if use_bias:
      params = ConvParams(
          base.create_parameter(k1, kernel_shape, init=kernel_init),
          base.create_parameter(k2, bias_shape, init=bias_init),
      )
    else:
      params = ConvParams(
          base.create_parameter(k1, kernel_shape, init=kernel_init),
          None
      )
    return base.LayerParams(params, info=info)

  @classmethod
  def spec(cls, in_spec, out_chan, filter_shape,
           strides=None, padding='VALID',
           kernel_init=None, bias_init=stax.randn(1e-6),
           use_bias=True):
    del use_bias
    in_shape = in_spec.shape
    shapes, _, _ = conv_info(
        in_shape, out_chan, filter_shape,
        strides=strides, padding=padding,
        kernel_init=kernel_init, bias_init=bias_init,
        transpose=True
    )
    return state.Shape(shapes[0], dtype=in_spec.dtype)

  def _call_batched(self, x):
    params, info = self.params, self.info
    result = lax.conv_transpose(x, params.kernel,
                                info.strides, info.padding,
                                dimension_numbers=DIMENSION_NUMBERS)
    if info.use_bias:
      result += params.bias
    return result

  def _call(self, x):
    """Applies 2D transposed convolution of the params with the input x."""
    if len(x.shape) != 3:
      raise ValueError('Need to `jax.vmap` in order to batch')
    result = self._call_batched(x[np.newaxis])
    return result[0]
