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
"""Numpy implementations of TensorFlow dtype related."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow_probability.python.internal.backend.numpy import _utils as utils


__all__ = [
    'as_dtype',
    'bool',
    'complex',
    'complex128',
    'complex64',
    'double',
    'DType',
    'float16',
    'float32',
    'float64',
    'int16',
    'int32',
    'int64',
    'int8',
    'string',
    'uint16',
    'uint32',
    'uint64',
    'uint8',
    # 'as_string',
    # 'bfloat16',
    # 'dtypes',
    # 'qint16',
    # 'qint32',
    # 'qint8',
    # 'quint16',
    # 'quint8',
]


JAX_MODE = False

DType = np.dtype  # pylint: disable=invalid-name


def _complex(real, imag, name=None):  # pylint: disable=unused-argument
  dtype = utils.common_dtype([real, imag], dtype_hint=float32)
  real = np.array(real, dtype=dtype)
  imag = np.array(imag, dtype=dtype)
  if as_dtype(dtype) == float32:
    complex_dtype = complex64
  else:
    complex_dtype = complex128
  return real + imag * complex_dtype(1j)


# --- Begin Public Functions --------------------------------------------------

as_dtype = utils.copy_docstring(
    'tf.as_dtype',
    lambda type_value: np.dtype(  # pylint: disable=g-long-lambda
        type_value.name if hasattr(type_value, 'name') else type_value).type)

real_dtype = lambda dtype: np.real(np.zeros((0,), dtype=as_dtype(dtype))).dtype

bool = np.bool  # pylint: disable=redefined-builtin

complex = utils.copy_docstring('tf.complex', _complex)  # pylint: disable=redefined-builtin

complex128 = np.complex128

complex64 = np.complex64

double = np.double


if JAX_MODE:
  bfloat16 = np.bfloat16
  __all__.append('bfloat16')

float16 = np.float16

float32 = np.float32

float64 = np.float64

int16 = np.int16

int32 = np.int32

int64 = np.int64

int8 = np.int8

# Handle version drift between internal/external/jax numpy.
string = getattr(np, 'str', getattr(np, 'string', None))

uint16 = np.uint16

uint32 = np.uint32

uint64 = np.uint64

uint8 = np.uint8
