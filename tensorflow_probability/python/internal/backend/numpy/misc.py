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
"""Numpy implementations of TensorFlow functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
import numpy as np

import tensorflow as tf

from tensorflow_probability.python.internal.backend.numpy import _utils as utils
from tensorflow_probability.python.internal.backend.numpy.ops import is_tensor


__all__ = [
    'argsort',
    'sort',
    'is_tensor',
    # 'clip_by_norm',
    # 'floormod',
    # 'meshgrid',
    # 'mod',
    # 'norm',
    # 'realdiv',
    # 'scatter_nd',
    # 'searchsorted',
    # 'strided_slice',
    # 'truediv',
    # 'truncatediv',
    # 'truncatemod',
    # 'unique',
    # 'unique_with_counts',
]


def _argsort(values, axis=-1, direction='ASCENDING', stable=False, name=None):  # pylint: disable=unused-argument
  """Numpy implementation of `tf.argsort`."""
  if direction == 'ASCENDING':
    pass
  elif direction == 'DESCENDING':
    values = np.negative(values)
  else:
    raise ValueError('Unrecognized direction: {}.'.format(direction))
  return np.argsort(values, axis, kind='stable' if stable else 'quicksort')


def _sort(values, axis=-1, direction='ASCENDING', stable=False, name=None):  # pylint: disable=unused-argument
  """Numpy implementation of `tf.sort`."""
  if direction == 'ASCENDING':
    pass
  elif direction == 'DESCENDING':
    values = np.negative(values)
  else:
    raise ValueError('Unrecognized direction: {}.'.format(direction))
  result = np.sort(values, axis, kind='stable' if stable else 'quicksort')
  if direction == 'DESCENDING':
    return np.negative(result)
  return result


# --- Begin Public Functions --------------------------------------------------

argsort = utils.copy_docstring(
    tf.argsort,
    _argsort)

sort = utils.copy_docstring(
    tf.sort,
    _sort)
