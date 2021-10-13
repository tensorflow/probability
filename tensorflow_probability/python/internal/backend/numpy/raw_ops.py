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
"""Numpy implementations of TensorFlow general top-level functions."""

import collections

# Dependency imports
import numpy as np
import numpy as onp  # Disable JAX rewrite.  # pylint: disable=reimported

from tensorflow_probability.python.internal.backend.numpy import _utils as utils


__all__ = [
    'BroadcastGradientArgs',
    'IgammaGradA',
    'MatrixDiagPartV2',
    'RandomGammaGrad',
]


JAX_MODE = False


_BroadcastGradientArgs = collections.namedtuple(
    'BroadcastGradientArgs', ['r0', 'r1'])


def _broadcast_gradient_args(s0, s1, name=None):  # pylint: disable=unused-argument
  bc_shp = onp.array(
      (onp.zeros(tuple(s0) + (0,)) + onp.zeros(tuple(s1) + (0,))).shape[:-1],
      dtype=onp.int32)
  pad_s0 = onp.pad(s0, [[len(bc_shp) - len(s0), 0]],
                   mode='constant', constant_values=-1)
  pad_s1 = onp.pad(s1, [[len(bc_shp) - len(s1), 0]],
                   mode='constant', constant_values=-1)
  return _BroadcastGradientArgs(
      onp.where((bc_shp != pad_s0) | (pad_s0 == 1))[0].astype(onp.int32),
      onp.where((bc_shp != pad_s1) | (pad_s1 == 1))[0].astype(onp.int32))


def _matrix_diag_part_v2(input, k, padding_value, name=None):  # pylint: disable=redefined-builtin,unused-argument
  """Implements tf.raw_ops.MatrixDiagPartV2, for scalar k."""
  if np.array(k).ndim > 0:
    raise NotImplementedError
  shp = np.shape(input)

  if JAX_MODE:
    if len(shp) > 2:
      from jax import vmap  # pylint: disable=g-import-not-at-top
      return vmap(_matrix_diag_part_v2, (0, None, None))(
          input, k, padding_value)
    return np.diag(input, k=k)

  input = np.reshape(input, (-1, shp[-2], shp[-1]))
  output = np.array([np.diag(arr, k=k) for arr in input])
  return output.reshape(*(shp[:-2] + output.shape[1:]))


def _random_gamma_grad(alpha, sample, name=None):  # pylint: disable=unused-argument
  if JAX_MODE:
    import jax.lax  # pylint: disable=g-import-not-at-top
    # Broadcast alpha and samples.
    broadcast_shape = (alpha + sample).shape
    alpha = np.broadcast_to(alpha, broadcast_shape)
    sample = np.broadcast_to(sample, broadcast_shape)
    return jax.lax.random_gamma_grad(alpha, sample)
  raise NotImplementedError


def _igamma_grad_a(a, x, name=None):  # pylint: disable=unused-argument
  if JAX_MODE:
    import jax.lax  # pylint: disable=g-import-not-at-top
    # Broadcast a and x.

    a, x = np.broadcast_arrays(a, x)
    return jax.lax.igamma_grad_a(a, x)
  raise NotImplementedError


BroadcastGradientArgs = utils.copy_docstring(  # pylint: disable=invalid-name
    'tf.raw_ops.BroadcastGradientArgs',
    _broadcast_gradient_args)

IgammaGradA = utils.copy_docstring(  # pylint: disable=invalid-name
    'tf.raw_ops.IgammaGradA',
    _igamma_grad_a)

MatrixDiagPartV2 = utils.copy_docstring(  # pylint: disable=invalid-name
    'tf.raw_ops.MatrixDiagPartV2',
    _matrix_diag_part_v2)

RandomGammaGrad = utils.copy_docstring(  # pylint: disable=invalid-name
    'tf.raw_ops.RandomGammaGrad',
    _random_gamma_grad)
