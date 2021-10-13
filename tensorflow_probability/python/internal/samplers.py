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

import hashlib
import warnings

# Dependency imports

import numpy as np
import six

import tensorflow.compat.v2 as tf

from tensorflow_probability.python.internal import prefer_static as ps


# ** See PRNGS.md for more detailed discussion about this packge. **

__all__ = [
    'categorical',
    'fold_in',
    'gamma',
    'is_stateful_seed',
    'normal',
    'poisson',
    'sanitize_seed',
    'split_seed',
    'shuffle',
    'uniform',
    'zeros_seed',
]

JAX_MODE = False

SEED_DTYPE = np.uint32 if JAX_MODE else np.int32


def zeros_seed():
  return tf.constant([0, 0], dtype=SEED_DTYPE)


def is_stateful_seed(seed):
  return seed is None or isinstance(seed, six.integer_types)


def sanitize_seed(seed, salt=None, name=None):
  """Map various PRNG seed flavors to a seed `Tensor`.

  This function implements TFP's standard PRNG seeding semantics.
  See https://github.com/tensorflow/probability/blob/main/PRNGS.md
  for details.

  Operationally, `sanitize_seed` maps any seed flavor to a
  "stateless-compatible" seed, namely a `int32[2]` Tensor.  To wit:
  - If the `seed` argument is an `int` or `None`, we use `tf.random.uniform`
    to _statefully_ draw a pair of unbounded `int32`s and wrap them into a
    Tensor.
  - If the `seed` argument is a stateless-compatible seed already, we
    just cast it to an `int32[2]` Tensor.

  This, any function that accepts a `seed` argument can be written in
  stateless-seed style internally, and acquires TFP's
  seed-type-directed stateless/stateful switching behavior by just
  running the input seed through `sanitize_seed` on entry.

  The `sanitize_seed` function also allows salting the seed: if a user
  accidentally passes the same stateful seed to two different calls to
  `sanitize_seed` with different salts, they will get independent
  randomness.  We may micro-optimize by removing salting from
  `sanitize_seed` of already-stateless seeds in the future, as using a
  stateless seed already requires seed uniqueness discipline.

  Args:
    seed: An `int32[2]` Tensor or a Python list of 2 `ints`, which
      will be treated as stateless seeds; or a Python `int` or `None`,
      which will be treated as stateful seeds.
    salt: An optional Python string.
    name: An optional Python string, name to add to TF ops created by
      this function.

  Returns:
    seed: An `int32[2]` Tensor suitable for use as a stateless PRNG
      seed.

  """
  if callable(seed):  # e.g. SeedStream.
    seed = seed()
  if salt is not None and not isinstance(salt, str):
    raise TypeError('`salt` must be a python `str`, got {}'.format(repr(salt)))
  with tf.name_scope(name or 'sanitize_seed'):
    if is_stateful_seed(seed):
      if JAX_MODE:
        raise ValueError(
            'TFP-on-JAX requires a `jax.random.PRNGKey` `seed` arg.')
      elif (tf.distribute.get_replica_context() is not None and
            tf.distribute.get_replica_context().num_replicas_in_sync > 1):
        warnings.warn(
            'Using stateful random seeds in replicated context can yield '
            'unreproducible results. For more details, see '
            'https://github.com/tensorflow/probability/blob/main/PRNGS.md')

      # TODO(b/147874898): Deprecate `int` seeds, migrate ints to stateless?
      if salt is not None:
        # Prefer to incorporate salt as a constant.
        if seed is not None:
          seed = int(hashlib.sha512(
              str((seed, salt)).encode('utf-8')).hexdigest(), 16) % (2**31 - 1)
        salt = None
      # Convert "stateful-indicating" `int`/`None` seed to stateless Tensor seed
      # by way of a stateful sampler.
      seed = tf.random.uniform([2], seed=seed, minval=np.iinfo(SEED_DTYPE).min,
                               maxval=np.iinfo(SEED_DTYPE).max,
                               dtype=SEED_DTYPE, name='seed')

    # TODO(b/159209541): Consider ignoring salts for stateless seeds, for
    # performance and because using stateless seeds already requires the
    # discipline of splitting.

    if salt is not None:
      salt = int(hashlib.sha512(str(salt).encode('utf-8')).hexdigest(), 16)
      seed = fold_in(seed, salt)

    return tf.convert_to_tensor(seed, dtype=SEED_DTYPE, name='seed')


def fold_in(seed, salt):
  """Folds salt into seed to form a new seed."""
  if JAX_MODE:
    from jax import random as jaxrand  # pylint: disable=g-import-not-at-top
    import jax.numpy as jnp  # pylint: disable=g-import-not-at-top
    return jaxrand.fold_in(
        seed, jnp.asarray(salt & np.uint32(2**32 - 1), dtype=SEED_DTYPE))
  if isinstance(salt, (six.integer_types)):
    seed = tf.bitwise.bitwise_xor(
        seed, np.uint64([salt & (2**64 - 1)]).view(np.int32))
  else:
    seed = tf.random.experimental.stateless_fold_in(seed, salt)
  return seed


def split_seed(seed, n=2, salt=None, name=None):
  """Splits a seed into `n` derived seeds.

  See https://github.com/tensorflow/probability/blob/main/PRNGS.md
  for details.
  Args:
    seed: The seed to split; may be an `int`, an `(int, int) tuple`, or a
      `Tensor`. `int` seeds are converted to `Tensor` seeds using
      `tf.random.uniform` stateful sampling. Tuples are converted to `Tensor`.
    n: The number of splits to return. In TensorFlow, if `n` is an integer, this
      function returns a list of seeds and otherwise returns a `Tensor` of
      seeds.  In JAX, this function always returns an array of seeds.
    salt: Optional `str` salt to mix with the seed.
    name: Optional name to scope related ops.

  Returns:
    seeds: If `n` is a Python `int`, a `tuple` of seed values is returned. If
      `n` is an int `Tensor`, a single `Tensor` of shape `[n, 2]` is returned. A
      single such seed is suitable to pass as the `seed` argument of the
      `tf.random.stateless_*` ops.
  """
  if not (isinstance(n, int)
          or isinstance(n, np.ndarray)
          or tf.is_tensor(n)):  # avoid confusion with salt.
    raise TypeError(
        '`n` must be a python `int` or an int Tensor, got {}'.format(repr(n)))
  with tf.name_scope(name or 'split_seed'):
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
    params_shape = ps.shape(alpha)
    if beta is not None:
      params_shape = ps.broadcast_shape(params_shape, ps.shape(beta))
    shape = ps.convert_to_shape_tensor(
        shape,
        dtype=getattr(params_shape, 'dtype', np.int32))  # May be TensorShape.
    samples_shape = ps.concat([shape, params_shape], axis=0)
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
    if is_stateful_seed(seed):
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
    lam_shape = ps.shape(lam)
    sample_shape = ps.concat([shape, lam_shape], axis=0)
    return tf.random.stateless_poisson(
        shape=sample_shape, seed=seed, lam=lam, dtype=dtype)


def shuffle(
    value,
    seed=None,
    name=None):
  """As `tf.random.shuffle`, but handling stateful/stateless `seed`s."""
  with tf.name_scope(name or 'shuffle'):
    seed = sanitize_seed(seed)
    sortkey = tf.random.stateless_uniform(shape=[ps.shape(value)[0]], seed=seed)
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
