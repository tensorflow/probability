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
"""Utility functions for dtypes."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


__all__ = [
    'common_dtype',
]


def common_dtype(args_list, preferred_dtype=None):
  """Returns explict dtype from `args_list` if there is one."""
  dtype = None
  # Make a copy so as to not modify arguments.
  args_list = list(args_list)
  while args_list:
    a = args_list.pop()
    if hasattr(a, 'dtype'):
      dt = tf.as_dtype(getattr(a, 'dtype')).base_dtype.as_numpy_dtype
    else:
      if isinstance(a, list):
        # Allows for nested types, e.g. Normal([np.float16(1.0)], [2.0])
        args_list.extend(a)
      continue
    if dtype is None:
      dtype = dt
    elif dtype != dt:
      raise TypeError('Found incompatible dtypes, {} and {}.'.format(dtype, dt))
  return preferred_dtype if dtype is None else tf.as_dtype(dtype)
