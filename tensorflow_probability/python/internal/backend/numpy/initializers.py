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
"""Experimental Numpy backend."""

import numpy as np


__all__ = [
    'constant',
]


class constant(object):  # pylint: disable=invalid-name
  """Constant initializer."""

  def __init__(self, value=0., dtype=np.float32, verify_shape=False):
    del verify_shape
    self.value = value
    self.dtype = dtype

  def __call__(self, shape, dtype=None, partition_info=None,
               verify_shape=None):
    del partition_info, verify_shape
    dtype = dtype or self.dtype
    return np.full(shape, self.value, dtype)
