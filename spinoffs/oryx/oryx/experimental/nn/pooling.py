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
"""Contains building blocks for pooling layers used for neural networks."""

import abc
import collections
from typing import Any

from jax import lax
from jax import numpy as np

from oryx.core import state
from oryx.experimental.nn import base

__all__ = [
    'Pooling',
    'MaxPooling',
    'SumPooling',
    'AvgPooling'
]


PoolingInfo = collections.namedtuple(
    'PoolingInfo', ['window_shape', 'dims', 'strides', 'padding'])


class Pooling(base.Layer, metaclass=abc.ABCMeta):
  """Base class for Pooling layers."""

  @classmethod
  def initialize(cls, rng, in_spec, window_shape,
                 strides=None, padding='VALID'):
    """Initializes Pooling layers.

    Args:
      rng: Random key.
      in_spec: Spec, specifying the input shape and dtype.
      window_shape: Int Tuple, specifying the Pooling window shape.
      strides: Optional tuple with pooling strides. If None, it will use
        stride 1 for each dimension in window_shape.
      padding: Either the string "SAME" or "VALID" indicating the type of
        padding algorithm to use. "SAME" would preserve the same input size,
        while "VALID" would reduce the input size.

    Returns:
      Tuple with the output shape and the LayerParams.
    """
    del in_spec
    strides = strides or (1,) * len(window_shape)
    dims = (1,) + window_shape + (1,)  # NHWC or NHC
    strides = (1,) + strides + (1,)
    info = PoolingInfo(window_shape, dims, strides, padding)
    return base.LayerParams(info=info)

  @classmethod
  def spec(cls, in_spec, window_shape,
           strides=None, padding='VALID'):
    in_shape = in_spec.shape
    if len(in_shape) > 3:
      raise ValueError('Need to `jax.vmap` in order to batch')
    in_shape = (1,) + in_shape
    dims = (1,) + window_shape + (1,)  # NHWC or NHC
    non_spatial_axes = 0, len(window_shape) + 1
    strides = strides or (1,) * len(window_shape)
    for i in sorted(non_spatial_axes):
      window_shape = window_shape[:i] + (1,) + window_shape[i:]
      strides = strides[:i] + (1,) + strides[i:]
    padding = lax.padtype_to_pads(in_shape, window_shape, strides, padding)
    out_shape = lax.reduce_window_shape_tuple(in_shape, dims, strides, padding)
    out_shape = out_shape[1:]
    return state.Shape(out_shape, dtype=in_spec.dtype)

  def _call(self, x):
    if len(x.shape) > 3:
      raise ValueError('Need to `jax.vmap` in order to batch')
    result = self._call_batched(x[np.newaxis])
    return result[0]

  def _call_and_update_batched(self, x):
    return self._call_batched(x), self

  @abc.abstractmethod
  def _call_batched(self, x) -> Any:
    raise NotImplementedError


class MaxPooling(Pooling):
  """Max pooling layer, computes the maximum within the window."""

  def _call_batched(self, x):
    info = self.info
    return lax.reduce_window(x, -np.inf, lax.max,
                             info.dims, info.strides, info.padding)


class SumPooling(Pooling):
  """Sum pooling layer, computes the sum within the window."""

  def _call_batched(self, x):
    info = self.info
    return lax.reduce_window(x, 0., lax.add,
                             info.dims, info.strides, info.padding)


class AvgPooling(Pooling):
  """Average pooling layer, computes the average within the window."""

  def _call_batched(self, x):
    info = self.info
    one = np.ones(x.shape[1:-1], dtype=x.dtype)
    window_strides = info.strides[1:-1]
    window_sizes = lax.reduce_window(one, 0., lax.add, info.window_shape,
                                     window_strides, info.padding)
    outputs = lax.reduce_window(x, 0., lax.add,
                                info.dims, info.strides, info.padding)
    return outputs / window_sizes[..., np.newaxis]
