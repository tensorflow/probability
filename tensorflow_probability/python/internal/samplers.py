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
"""Random samplers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import hashlib

# Dependency imports

import numpy as np
import six

import tensorflow.compat.v2 as tf

__all__ = [
    'categorical',
    'gamma',
    'normal',
    'poisson',
    'sanitize_seed',
    'split_seed',
    'shuffle',
    'uniform',
]


JAX_MODE = False

SEED_DTYPE = np.uint32 if JAX_MODE else np.int32


def sanitize_seed(seed, salt=None):
  """Map various types to a seed `Tensor`."""
  if salt is not None and not isinstance(salt, str):
    raise TypeError('`salt` must be a python `str`, got {}'.format(repr(salt)))
  if seed is None or isinstance(seed, six.integer_types):
    if JAX_MODE:
      raise ValueError('TFP-on-JAX requires a `jax.random.PRNGKey` `seed` arg.')
    # TODO(b/147874898): Do we deprecate `int` seeds, migrate ints to stateless?
    if salt is not None:
      # Prefer to incorporate salt as a constant.
      if seed is not None:
        seed = int(hashlib.sha512(
            str((seed, salt)).encode('utf-8')).hexdigest(), 16) % (2**31 - 1)
      salt = None
    # Convert "stateful-indicating" `int`/`None` seed to stateless Tensor seed,
    # by way of a stateful sampler.
    seed = tf.random.uniform([2], seed=seed, minval=np.iinfo(SEED_DTYPE).min,
                             maxval=np.iinfo(SEED_DTYPE).max, dtype=SEED_DTYPE,
                             name='seed')
  if salt is not None:
    salt = int(hashlib.sha512(str(salt).encode('utf-8')).hexdigest(), 16)
    if JAX_MODE:
      from jax import random as jaxrand  # pylint: disable=g-import-not-at-top
      seed = jaxrand.fold_in(seed, salt & (2**32 - 1))
    else:
      seed = tf.bitwise.bitwise_xor(
          seed, np.uint64([salt & (2**64 - 1)]).view(np.int32))
  return tf.convert_to_tensor(seed, dtype=SEED_DTYPE, name='seed')


def split_seed(seed, n=2, salt=None, name=None):
  """Splits a seed deterministically into derived seeds."""
  if not (isinstance(n, int) or tf.is_tensor(n)):  # avoid confusion with salt.
    raise TypeError(
        '`n` must be a python `int` or an int Tensor, got {}'.format(repr(n)))
  with tf.name_scope(name or 'split'):
    seed = sanitize_seed(seed, salt=salt)
    if JAX_MODE:
      from jax import random as jaxrand  # pylint: disable=g-import-not-at-top
      return jaxrand.split(seed, n)
    seeds = tf.random.stateless_uniform(
        [n, 2], seed=seed, minval=None, maxval=None, dtype=SEED_DTYPE)
    if isinstance(n, six.integer_types):
      seeds = tf.unstack(seeds)
    return seeds


def categorical(
    logits,
    num_samples,
    dtype=None,
    seed=None,
    name=None):
  """As `tf.random.categorical`, but handling stateful/stateless `seed`s."""
  with tf.name_scope(name or 'categorical'):
    seed = sanitize_seed(seed)
    return tf.random.stateless_categorical(
        logits=logits, num_samples=num_samples, seed=seed, dtype=dtype)


def gamma(
    shape,
    alpha,
    beta=None,
    dtype=tf.float32,
    seed=None,
    name=None):
  """As `tf.random.gamma`, but handling stateful/stateless `seed`s."""
  with tf.name_scope(name or 'gamma'):
    seed = sanitize_seed(seed)
    alpha = tf.convert_to_tensor(alpha, dtype=dtype)
    beta = None if beta is None else tf.convert_to_tensor(beta, dtype=dtype)
    params_shape = tf.shape(alpha)
    if beta is not None:
      params_shape = tf.broadcast_dynamic_shape(params_shape, tf.shape(beta))
    shape = tf.convert_to_tensor(shape, dtype=params_shape.dtype)
    samples_shape = tf.concat([shape, params_shape], axis=0)
    return tf.random.stateless_gamma(
        shape=samples_shape, seed=seed, alpha=alpha, beta=beta, dtype=dtype)


def normal(
    shape,
    mean=0.0,
    stddev=1.0,
    dtype=tf.float32,
    seed=None,
    name=None):
  """As `tf.random.normal`, but handling stateful/stateless `seed`s."""
  with tf.name_scope(name or 'normal'):
    # TODO(b/147874898): Remove workaround for seed-sensitive tests.
    if seed is None or isinstance(seed, six.integer_types):
      return tf.random.normal(
          shape=shape, seed=seed, mean=mean, stddev=stddev, dtype=dtype)

    seed = sanitize_seed(seed)
    return tf.random.stateless_normal(
        shape=shape, seed=seed, mean=mean, stddev=stddev, dtype=dtype)


def poisson(
    shape,
    lam,
    dtype=tf.float32,
    seed=None,
    name=None):
  """As `tf.random.poisson`, but handling stateful/stateless `seed`s."""
  with tf.name_scope(name or 'poisson'):
    seed = sanitize_seed(seed)
    lam_shape = tf.shape(lam)
    shape = tf.convert_to_tensor(shape, dtype=lam_shape.dtype)
    sample_shape = tf.concat([shape, lam_shape], axis=0)
    return tf.random.stateless_poisson(
        shape=sample_shape, seed=seed, lam=lam, dtype=dtype)


def shuffle(
    value,
    seed=None,
    name=None):
  """As `tf.random.shuffle`, but handling stateful/stateless `seed`s."""
  with tf.name_scope(name or 'shuffle'):
    seed = sanitize_seed(seed)
    sortkey = tf.random.stateless_uniform(shape=[tf.shape(value)[0]], seed=seed)
    return tf.gather(value, tf.argsort(sortkey))


def uniform(
    shape,
    minval=0,
    maxval=None,
    dtype=tf.float32,
    seed=None,
    name=None):
  """As `tf.random.uniform`, but handling stateful/stateless `seed`s."""
  with tf.name_scope(name or 'uniform'):
    seed = sanitize_seed(seed)
    return tf.random.stateless_uniform(
        shape=shape, seed=seed, minval=minval, maxval=maxval, dtype=dtype)
