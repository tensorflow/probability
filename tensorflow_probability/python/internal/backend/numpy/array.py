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
"""Numpy implementations of TensorFlow general top-level functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
import numpy as np

import tensorflow as tf

from tensorflow_probability.python.internal.backend.numpy.internal import utils


__all__ = [
    'concat',
    'expand_dims',
    'fill',
    'linspace',
    'ones',
    'ones_like',
    'range',
    'rank',
    'reshape',
    'reverse',
    'roll',
    'shape',
    'size',
    'split',
    'squeeze',
    'stack',
    'tile',
    'transpose',
    'where',
    'zeros',
    'zeros_like',
    # 'boolean_mask',
    # 'einsum',
    # 'foldl',
    # 'foldr',
    # 'gather',
    # 'gather_nd',
    # 'one_hot',
    # 'pad',
    # 'tensordot',
    # 'unstack',
]


def _ones_like(input, dtype=None, name=None):  # pylint: disable=redefined-builtin
  s = _shape(input)
  if isinstance(s, (np.ndarray, np.generic)):
    return np.ones(s, utils.numpy_dtype(dtype or input.dtype))
  return tf.ones(s, dtype or s.dtype, name)


def _shape(input, out_type=tf.int32, name=None):  # pylint: disable=redefined-builtin,unused-argument
  return np.array(np.array(input).shape).astype(utils.numpy_dtype(out_type))


def _size(input, out_type=tf.int32, name=None):  # pylint: disable=redefined-builtin, unused-argument
  return np.prod(np.array(input).shape).astype(utils.numpy_dtype(out_type))


def _transpose(a, perm=None, conjugate=False, name='transpose'):  # pylint: disable=unused-argument
  x = np.transpose(a, perm)
  return np.conjugate(x) if conjugate else x


def _zeros_like(input, dtype=None, name=None):  # pylint: disable=redefined-builtin
  s = _shape(input)
  if isinstance(s, (np.ndarray, np.generic)):
    return np.zeros(s, utils.numpy_dtype(dtype or input.dtype))
  return tf.zeros(s, dtype or s.dtype, name)


# --- Begin Public Functions --------------------------------------------------


concat = utils.copy_docstring(
    tf.concat,
    lambda values, axis, name='concat': np.concatenate(values, axis))

expand_dims = utils.copy_docstring(
    tf.expand_dims,
    lambda input, axis, name=None: np.expand_dims(input, axis))

fill = utils.copy_docstring(
    tf.fill,
    lambda dims, value, name=None: value * np.ones(dims, np.array(value).dtype))

reverse = utils.copy_docstring(
    tf.reverse,
    lambda tensor, axis, name=None: np.flip(tensor, axis))

linspace = utils.copy_docstring(
    tf.linspace,
    lambda start, stop, num, name=None: (  # pylint: disable=g-long-lambda
        np.linspace(start, stop, num).astype(np.array(start).dtype)))

ones = utils.copy_docstring(
    tf.ones,
    lambda shape, dtype=tf.float32, name=None: np.ones(  # pylint: disable=g-long-lambda
        shape, utils.numpy_dtype(dtype)))

ones_like = utils.copy_docstring(
    tf.ones_like,
    _ones_like)

range = utils.copy_docstring(  # pylint: disable=redefined-builtin
    tf.range,
    lambda start, limit=None, delta=1, dtype=None, name='range': (  # pylint: disable=g-long-lambda
        np.arange(start, limit, delta, utils.numpy_dtype(dtype))))

rank = utils.copy_docstring(
    tf.rank,
    lambda input, name=None: len(np.array(input).shape))  # pylint: disable=redefined-builtin,g-long-lambda

reshape = utils.copy_docstring(
    tf.reshape,
    lambda tensor, shape, name=None: np.reshape(tensor, shape))

roll = utils.copy_docstring(
    tf.roll,
    lambda input, shift, axis: np.roll(input, shift, axis))  # pylint: disable=unnecessary-lambda

shape = utils.copy_docstring(
    tf.shape,
    _shape)

size = utils.copy_docstring(
    tf.size,
    _size)

split = utils.copy_docstring(
    tf.split,
    lambda value, num_or_size_splits, axis=0, num=None, name='split': (  # pylint: disable=g-long-lambda
        np.split(value, num_or_size_splits, axis)))

squeeze = utils.copy_docstring(
    tf.squeeze,
    lambda input, axis=None, name=None: np.squeeze(input, axis))

stack = utils.copy_docstring(
    tf.stack,
    lambda values, axis, name=None: np.stack(values, axis))

tile = utils.copy_docstring(
    tf.tile,
    lambda input, multiples, name=None: np.tile(input, multiples))

transpose = utils.copy_docstring(
    tf.transpose,
    _transpose)

where = utils.copy_docstring(
    tf.where,
    lambda condition, x=None, y=None, name=None: np.where(condition, x, y))

zeros = utils.copy_docstring(
    tf.zeros,
    lambda shape, dtype=tf.float32, name=None: np.zeros(  # pylint: disable=g-long-lambda
        shape, utils.numpy_dtype(dtype)))

zeros_like = utils.copy_docstring(
    tf.zeros_like,
    _zeros_like)
