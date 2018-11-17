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
"""Numerically stable variants of common mathematical expressions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf


def log1psquare(x):
  """A numerically stable implementation of log(1 + x**2)."""
  # We use the following identity:
  #
  # log(1 + x**2) = 2 log(|x|) + log(1 / x**2 + 1)
  #
  # Where the last term is 0 up to the numerical precision when 1 / x**2 is
  # small enough compared to machine epsilon.
  is_small = tf.abs(x) * np.sqrt(np.finfo(x.dtype.as_numpy_dtype).eps) <= 1.
  ones = tf.ones_like(x)

  # Also mask out the large/small x's, so the gradients are propagated
  # correctly.
  small_x = tf.where(is_small, x, ones)
  large_x = tf.where(is_small, ones, x)

  return tf.where(is_small, tf.log1p(small_x**2.), 2. * tf.log(tf.abs(large_x)))
