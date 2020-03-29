# Copyright 2019 The TensorFlow Probability Authors.
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
"""Numpy implementations of `tf.signal` functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from tensorflow_probability.python.internal.backend.numpy import _utils as utils

__all__ = [
    'fft',
    'fft2d',
    'fft3d',
    'ifft',
    'ifft2d',
    'ifft3d',
    'irfft',
    'irfft2d',
    'irfft3d',
    'rfft',
    'rfft2d',
    'rfft3d',
]


fft = utils.copy_docstring(
    'tf.signal.fft',
    lambda input, name=None: np.fft.fftn(input, axes=[-1]))

fft2d = utils.copy_docstring(
    'tf.signal.fft2d',
    lambda input, name=None: np.fft.fftn(input, axes=[-2, -1]))

fft3d = utils.copy_docstring(
    'tf.signal.fft3d',
    lambda input, name=None: np.fft.fftn(input, axes=[-3, -2, -1]))

ifft = utils.copy_docstring(
    'tf.signal.ifft',
    lambda input, name=None: np.fft.ifftn(input, axes=[-1]))

ifft2d = utils.copy_docstring(
    'tf.signal.ifft2d',
    lambda input, name=None: np.fft.ifftn(input, axes=[-2, -1]))

ifft3d = utils.copy_docstring(
    'tf.signal.ifft3d',
    lambda input, name=None: np.fft.ifftn(input, axes=[-3, -2, -1]))

rfft = utils.copy_docstring(
    'tf.signal.rfft',
    lambda input, fft_length=None, name=None: np.fft.rfftn(  # pylint:disable=g-long-lambda
        input, s=fft_length, axes=[-1]))

rfft2d = utils.copy_docstring(
    'tf.signal.rfft2d',
    lambda input, fft_length=None, name=None: np.fft.rfftn(  # pylint:disable=g-long-lambda
        input, s=fft_length, axes=[-2, -1]))

rfft3d = utils.copy_docstring(
    'tf.signal.rfft3d',
    lambda input, fft_length=None, name=None: np.fft.rfftn(  # pylint:disable=g-long-lambda
        input, s=fft_length, axes=[-3, -2, -1]))

irfft = utils.copy_docstring(
    'tf.signal.irfft',
    lambda input, fft_length=None, name=None: np.fft.irfftn(  # pylint:disable=g-long-lambda
        input, s=fft_length, axes=[-1]))

irfft2d = utils.copy_docstring(
    'tf.signal.irfft2d',
    lambda input, fft_length=None, name=None: np.fft.irfftn(  # pylint:disable=g-long-lambda
        input, s=fft_length, axes=[-2, -1]))

irfft3d = utils.copy_docstring(
    'tf.signal.irfft3d',
    lambda input, fft_length=None, name=None: np.fft.irfftn(  # pylint:disable=g-long-lambda
        input, s=fft_length, axes=[-3, -2, -1]))
