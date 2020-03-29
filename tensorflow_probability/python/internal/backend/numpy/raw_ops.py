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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
import numpy as np

from tensorflow_probability.python.internal.backend.numpy import _utils as utils


__all__ = [
    'MatrixDiagPartV2',
]


JAX_MODE = False


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


MatrixDiagPartV2 = utils.copy_docstring(  # pylint: disable=invalid-name
    'tf.raw_ops.MatrixDiagPartV2',
    _matrix_diag_part_v2)
