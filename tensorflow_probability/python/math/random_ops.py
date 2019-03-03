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
"""Functions for generating random samples.

Note: Many of these functions will eventually be migrated to core TensorFlow.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


__all__ = [
    'random_rademacher',
    'random_rayleigh',
]


def random_rademacher(shape, dtype=tf.float32, seed=None, name=None):
  """Generates `Tensor` consisting of `-1` or `+1`, chosen uniformly at random.

  For more details, see [Rademacher distribution](
  https://en.wikipedia.org/wiki/Rademacher_distribution).

  Args:
    shape: Vector-shaped, `int` `Tensor` representing shape of output.
    dtype: (Optional) TF `dtype` representing `dtype` of output.
    seed: (Optional) Python integer to seed the random number generator.
    name: Python `str` name prefixed to Ops created by this function.
      Default value: `None` (i.e., 'random_rademacher').

  Returns:
    rademacher: `Tensor` with specified `shape` and `dtype` consisting of `-1`
      or `+1` chosen uniformly-at-random.
  """
  with tf.compat.v1.name_scope(name, 'random_rademacher', [shape, seed]):
    random_bernoulli = tf.random.uniform(
        shape, minval=0, maxval=2, dtype=tf.int32, seed=seed)
    return tf.cast(2 * random_bernoulli - 1, dtype)


def random_rayleigh(shape, scale=None, dtype=tf.float32, seed=None, name=None):
  """Generates `Tensor` of positive reals drawn from a Rayleigh distributions.

  The probability density function of a Rayleigh distribution with `scale`
  parameter is given by:

  ```none
  f(x) = x scale**-2 exp(-x**2 0.5 scale**-2)
  ```

  For more details, see [Rayleigh distribution](
  https://en.wikipedia.org/wiki/Rayleigh_distribution)

  Args:
    shape: Vector-shaped, `int` `Tensor` representing shape of output.
    scale: (Optional) Positive `float` `Tensor` representing `Rayleigh` scale.
      Default value: `None` (i.e., `scale = 1.`).
    dtype: (Optional) TF `dtype` representing `dtype` of output.
      Default value: `tf.float32`.
    seed: (Optional) Python integer to seed the random number generator.
      Default value: `None` (i.e., no seed).
    name: Python `str` name prefixed to Ops created by this function.
      Default value: `None` (i.e., 'random_rayleigh').

  Returns:
    rayleigh: `Tensor` with specified `shape` and `dtype` consisting of positive
      real values drawn from a Rayleigh distribution with specified `scale`.
  """
  with tf.compat.v1.name_scope(name, 'random_rayleigh', [shape, scale, seed]):
    if scale is not None:
      # Its important to expand the shape to match scale's, otherwise we won't
      # have independent draws.
      scale = tf.convert_to_tensor(value=scale, dtype=dtype, name='scale')
      shape = tf.broadcast_dynamic_shape(shape, tf.shape(input=scale))
    x = tf.sqrt(-2. * tf.math.log(
        tf.random.uniform(shape, minval=0, maxval=1, dtype=dtype, seed=seed)))
    if scale is None:
      return x
    return x * scale
