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
"""Utility functions for GLM module."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.contrib import framework as contrib_framework


__all__ = [
    'common_dtype',
]


def common_dtype(args_list, preferred_dtype=None):
  """Returns explict dtype from `args_list` if there is one."""
  dtype = None
  for a in args_list:
    if isinstance(a, (np.ndarray, np.generic)):
      dt = a.dtype.type
    elif contrib_framework.is_tensor(a):
      dt = a.dtype.as_numpy_dtype
    else:
      continue
    if dtype is None:
      dtype = dt
    elif dtype != dt:
      raise TypeError('Found incompatible dtypes, {} and {}.'.format(dtype, dt))
  return preferred_dtype if dtype is None else dtype
