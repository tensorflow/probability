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

import numpy as np

import tensorflow.compat.v2 as tf

from tensorflow_probability.python.internal import prefer_static as ps
from tensorflow_probability.python.internal import samplers

__all__ = [
    'rademacher',
    'rayleigh',
    'spherical_uniform',
]


def rademacher(shape, dtype=tf.float32, seed=None, name=None):
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
  with tf.name_scope(name or 'rademacher'):
    # Choose the dtype to cause `2 * random_bernoulli - 1` to run in the same
    # memory (host or device) as the downstream cast will want to put it.  The
    # convention on GPU is that int32 are in host memory and int64 are in device
    # memory.
    shape = ps.convert_to_shape_tensor(shape)
    generation_dtype = tf.int64 if tf.as_dtype(dtype) != tf.int32 else tf.int32
    random_bernoulli = samplers.uniform(
        shape, minval=0, maxval=2, dtype=generation_dtype, seed=seed)
    return tf.cast(2 * random_bernoulli - 1, dtype)


def rayleigh(shape, scale=None, dtype=tf.float32, seed=None, name=None):
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
  with tf.name_scope(name or 'rayleigh'):
    if scale is not None:
      # Its important to expand the shape to match scale's, otherwise we won't
      # have independent draws.
      scale = tf.convert_to_tensor(scale, dtype=dtype, name='scale')
      shape = tf.broadcast_dynamic_shape(shape, tf.shape(scale))
    x = tf.sqrt(-2. * tf.math.log(
        samplers.uniform(shape, minval=0, maxval=1, dtype=dtype, seed=seed)))
    if scale is None:
      return x
    return x * scale


def spherical_uniform(
    shape,
    dimension,
    dtype=tf.float32,
    seed=None,
    name=None):
  """Generates `Tensor` drawn from a uniform distribution on the sphere.

  Args:
    shape: Vector-shaped, `int` `Tensor` representing shape of output.
    dimension: Scalar `int` `Tensor`, representing the dimensionality of the
      space where the sphere is embedded.
    dtype: (Optional) TF `dtype` representing `dtype` of output.
      Default value: `tf.float32`.
    seed: (Optional) Python integer to seed the random number generator.
      Default value: `None` (i.e., no seed).
    name: Python `str` name prefixed to Ops created by this function.
      Default value: `None` (i.e., 'random_rayleigh').

  Returns:
    spherical_uniform: `Tensor` with specified `shape` and `dtype` consisting
      of positive real values drawn from a Rayleigh distribution with specified
      `scale`.
  """
  with tf.name_scope(name or 'spherical_uniform'):
    seed = samplers.sanitize_seed(seed)
    dimension = ps.convert_to_shape_tensor(ps.cast(dimension, dtype=tf.int32))
    shape = ps.convert_to_shape_tensor(shape, dtype=tf.int32)
    dimension_static = tf.get_static_value(dimension)
    sample_shape = ps.concat([shape, [dimension]], axis=0)
    sample_shape = ps.convert_to_shape_tensor(sample_shape)
    # Special case one and two dimensions. This is to guard against the case
    # where the normal samples are zero. This can happen in dimensions 1 and 2.
    if dimension_static is not None:
      # This is equivalent to sampling Rademacher random variables.
      if dimension_static == 1:
        return rademacher(sample_shape, dtype=dtype, seed=seed)
      elif dimension_static == 2:
        u = samplers.uniform(
            shape, minval=0, maxval=2 * np.pi, dtype=dtype, seed=seed)
        return tf.stack([tf.math.cos(u), tf.math.sin(u)], axis=-1)
      else:
        normal_samples = samplers.normal(
            shape=ps.concat([shape, [dimension_static]], axis=0),
            seed=seed,
            dtype=dtype)
        unit_norm = normal_samples / tf.norm(
            normal_samples, ord=2, axis=-1)[..., tf.newaxis]
        return unit_norm

    # If we can't determine the dimension statically, tf.where between the
    # different options.
    r_seed, u_seed, n_seed = samplers.split_seed(
        seed, n=3, salt='spherical_uniform_dynamic_shape')
    rademacher_samples = rademacher(sample_shape, dtype=dtype, seed=r_seed)
    u = samplers.uniform(
        shape, minval=0, maxval=2 * np.pi, dtype=dtype, seed=u_seed)
    twod_samples = tf.concat(
        [tf.math.cos(u)[..., tf.newaxis],
         tf.math.sin(u)[..., tf.newaxis] * tf.ones(
             [dimension - 1], dtype=dtype)], axis=-1)

    normal_samples = samplers.normal(
        shape=ps.concat([shape, [dimension]], axis=0),
        seed=n_seed,
        dtype=dtype)
    nd_samples = normal_samples / tf.norm(
        normal_samples, ord=2, axis=-1)[..., tf.newaxis]

    return tf.where(
        tf.math.equal(dimension, 1),
        rademacher_samples,
        tf.where(
            tf.math.equal(dimension, 2),
            twod_samples,
            nd_samples))

