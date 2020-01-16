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
import tensorflow.compat.v2 as tf

from tensorflow_probability.python.internal import dtype_util


__all__ = [
    'clip_by_value_preserve_gradient',
    'log1psquare',
]


def log1psquare(x, name=None):
  """Numerically stable calculation of `log(1 + x**2)` for small or large `|x|`.

  For sufficiently large `x` we use the following observation:

  ```none
  log(1 + x**2) =   2 log(|x|) + log(1 + 1 / x**2)
                --> 2 log(|x|)  as x --> inf
  ```

  Numerically, `log(1 + 1 / x**2)` is `0` when `1 / x**2` is small relative to
  machine epsilon.

  Args:
    x: Float `Tensor` input.
    name: Python string indicating the name of the TensorFlow operation.
      Default value: `'log1psquare'`.

  Returns:
    log1psq: Float `Tensor` representing `log(1. + x**2.)`.
  """
  with tf.name_scope(name or 'log1psquare'):
    x = tf.convert_to_tensor(x, dtype_hint=tf.float32, name='x')
    dtype = dtype_util.as_numpy_dtype(x.dtype)

    eps = np.finfo(dtype).eps.astype(np.float64)
    is_large = tf.abs(x) > (eps**-0.5).astype(dtype)

    # Mask out small x's so the gradient correctly propagates.
    abs_large_x = tf.where(is_large, tf.abs(x), tf.ones([], x.dtype))
    return tf.where(is_large,
                    2. * tf.math.log(abs_large_x),
                    tf.math.log1p(tf.square(x)))


def clip_by_value_preserve_gradient(t, clip_value_min, clip_value_max,
                                    name=None):
  """Clips values to a specified min and max while leaving gradient unaltered.

  Like `tf.clip_by_value`, this function returns a tensor of the same type and
  shape as input `t` but with values clamped to be no smaller than to
  `clip_value_min` and no larger than `clip_value_max`. Unlike
  `tf.clip_by_value`, the gradient is unaffected by this op, i.e.,

  ```python
  tf.gradients(tfp.math.clip_by_value_preserve_gradient(x), x)[0]
  # ==> ones_like(x)
  ```

  Note: `clip_value_min` needs to be smaller or equal to `clip_value_max` for
  correct results.

  Args:
    t: A `Tensor`.
    clip_value_min: A scalar `Tensor`, or a `Tensor` with the same shape
      as `t`. The minimum value to clip by.
    clip_value_max: A scalar `Tensor`, or a `Tensor` with the same shape
      as `t`. The maximum value to clip by.
    name: A name for the operation (optional).
      Default value: `'clip_by_value_preserve_gradient'`.

  Returns:
    clipped_t: A clipped `Tensor`.
  """
  with tf.name_scope(name or 'clip_by_value_preserve_gradient'):
    t = tf.convert_to_tensor(t, name='t')
    clip_t = tf.clip_by_value(t, clip_value_min, clip_value_max)
    return t + tf.stop_gradient(clip_t - t)
