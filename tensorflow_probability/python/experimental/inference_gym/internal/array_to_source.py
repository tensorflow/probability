# Lint as: python2, python3
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
"""Utility to convert arrays to Python source files."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

__all__ = [
    'array_to_source',
]


def array_to_source(name, array):
  """Encodes a numpy array to a Python source string.

  The generated source creates a top-level Python variable with name `name` that
  is assigned the contents of `array`. It assumes that NumPy is imported with an
  `np` alias.

  Args:
    name: Python `str`. Name of the generated variable.
    array: Numpy array to encode.

  Returns:
    array_str: The encoded array.
  """
  ret = '{} = np.array([\n'.format(name)
  array = np.asarray(array)
  array_flat = array.reshape([-1])
  for e in array_flat:
    e_str = np.array2string(e, floatmode='unique')
    ret += '    {},\n'.format(e_str)
  ret += ']).reshape({})\n'.format(array.shape)
  return ret
