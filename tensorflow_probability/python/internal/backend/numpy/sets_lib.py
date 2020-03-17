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
"""Numpy implementations of sets functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
import numpy as np

from tensorflow_probability.python.internal.backend.numpy import _utils as utils


__all__ = [
    'difference',
]


def _difference(a, b, aminusb=True, validate_indices=True):
  if not aminusb:
    raise NotImplementedError(
        'Argument `aminusb != True` is currently unimplemented.')
  if not validate_indices:
    raise NotImplementedError(
        'Argument `validate_indices != True` is currently unimplemented.')
  return np.setdiff1d(a, b)


# --- Begin Public Functions --------------------------------------------------


# TODO(b/136555907): Add unit test.
difference = utils.copy_docstring(
    'tf.sets.difference',
    _difference)
