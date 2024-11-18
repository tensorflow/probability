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

import collections
import hashlib
import warnings

import numpy as np
import six

import tensorflow.compat.v2 as tf

from tensorflow_probability.python.internal import prefer_static as ps


# ** See PRNGS.md for more detailed discussion about this package. **

__all__ = [
    'categorical',
    'clone_seed',
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


_OldSaltSeed = collections.namedtuple('_OldSaltSeed', ['seed'])


def enable_old_salt_for_seed(seed, enable):
  if enable:
    return _OldSaltSeed(seed)
  else:
    return seed


def _get_seed_and_old_salt(seed):
  if isinstance(seed, _OldSaltSeed):
    return seed.seed, True
  else:
    return seed, False


def zeros_seed():
  if JAX_MODE:
    import jax  # pylint: disable=g-import-not-at-top
    return jax.random.key(0)
  return tf.constant([0, 0], dtype=SEED_DTYPE)


def is_stateful_seed(seed):
  return seed is None or isinstance(seed, six.integer_types)


def sanitize_seed(seed, salt=None, name=None):
  """Map various PRNG seed flavors to a seed `Tensor`.

  This function implements TFP's standard PRNG seeding semantics.
  See https://github.com/tensorflow/probability/blob/main/PRNGS.md
  for details.

  Operationally, `sanitize_seed` maps any seed flavor to a
  "stateless-compatible" seed.  Under TensorFlow and NumPy this means:
  - If the `seed` argument is an `int` or `None`, we use `tf.random.uniform`
    to _statefully_ draw a pair of unbounded `int32`s and wrap them into a
    Tensor.
  - If the `seed` argument is a stateless-compatible seed already, we
    just cast it to an `int32[2]` Tensor.

  Under JAX, this function only accepts outputs from `jax.random.PRNGKey`, being
  a no-op except for the salting behavior described below.

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
    seed: An `int32[2]` Tensor or a Python list or tuple of 2 `ints`,
      which will be treated as stateless seeds; or a Python `int` or
      `None`, which will be treated as stateful seeds.
    salt: An optional Python string.
    name: An optional Python string, name to add to TF ops created by
      this function.

  Returns:
    seed: An `int32[2]` Tensor suitable for use as a stateless PRNG
      seed.

  """
  seed, old_salt = _get_seed_and_old_salt(seed)
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

    # TODO(b/223267515): In JAX mode, raise a user-friendly error if seed is
    # not a PRNGKey.

    # TODO(b/159209541): Consider ignoring salts for stateless seeds, for
    # performance and because using stateless seeds already requires the
    # discipline of splitting.

    if salt is not None:
      salt = int(hashlib.sha512(str(salt).encode('utf-8')).hexdigest(), 16)
      if not old_salt:
        salt = salt % (2**31 - 1)
      seed = fold_in(seed, salt)

    if JAX_MODE:
      import jax  # pylint: disable=g-import-not-at-top
      # Typed keys are returned as is, otherwise wrap them.
      if not jax.dtypes.issubdtype(seed.dtype, jax.dtypes.prng_key):
        seed = jax.random.wrap_key_data(seed)
    else:
      seed = tf.convert_to_tensor(seed, dtype=SEED_DTYPE, name='seed')
    return enable_old_salt_for_seed(seed, old_salt)


def get_integer_seed(seed):
  """Returns an integer seed in [0, 2**31).

  Args:
    seed: A seed suitable to be passed to `sanitize_seed`.

  Returns:
    integer_seed: A Python integer (if seed was a Python integer or we're in
    JAX) or an integer Tensor.
  """
  if isinstance(seed, six.integer_types):
    return seed % (2**31)
  seed = sanitize_seed(seed)
  seed, _ = _get_seed_and_old_salt(seed)
  # maxval is exclusive, so technically this doesn't generate all possible
  # non-negative integers, but it's good enough for our purposes.
  integer_seed = tf.random.stateless_uniform(
      shape=[], seed=seed, minval=0, maxval=2**31 - 1, dtype=tf.int32)
  if JAX_MODE:
    # This function isn't ever used in a jit context, so we can eagerly convert
    # it to an integer to simplify caller's code.
    integer_seed = int(integer_seed)
  return integer_seed


def fold_in(seed, salt):
  """Folds salt into seed to form a new seed."""
  seed, old_salt = _get_seed_and_old_salt(seed)
  if JAX_MODE:
    from jax import random as jaxrand  # pylint: disable=g-import-not-at-top
    import jax.numpy as jnp  # pylint: disable=g-import-not-at-top
    seed = jaxrand.fold_in(
        seed, jnp.asarray(salt & np.uint32(2**32 - 1), dtype=SEED_DTYPE))
  else:
    if isinstance(salt, (six.integer_types)):
      seed = tf.bitwise.bitwise_xor(
          seed, np.uint64([salt & (2**64 - 1)]).view(np.int32))
    else:
      seed = tf.random.experimental.stateless_fold_in(seed, salt)
  return enable_old_salt_for_seed(seed, old_salt)


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
    seed, old_salt = _get_seed_and_old_salt(seed)
    if JAX_MODE:
      from jax import random as jaxrand  # pylint: disable=g-import-not-at-top
      return jaxrand.split(seed, int(n))
    seeds = tf.random.stateless_uniform(
        [n, 2], seed=seed, minval=None, maxval=None, dtype=SEED_DTYPE)
    if isinstance(n, six.integer_types):
      seeds = tf.unstack(seeds)
      seeds = [enable_old_salt_for_seed(seed, old_salt) for seed in seeds]
    else:
      seeds = enable_old_salt_for_seed(seeds, old_salt)
    return seeds


def clone_seed(seed):
  """Clones a seed so it can be reused without causing a JAX KeyReuseError."""
  seed, old_salt = _get_seed_and_old_salt(seed)
  if JAX_MODE:
    from jax import random as jaxrand  # pylint: disable=g-import-not-at-top
    if hasattr(jaxrand, 'clone'):
      # JAX v0.4.26+
      seed = jaxrand.clone(seed)
  return enable_old_salt_for_seed(seed, old_salt)


def categorical(
    logits,
    num_samples,
    dtype=None,
    seed=None,
    name=None):
  """As `tf.random.categorical`, but handling stateful/stateless `seed`s."""
  with tf.name_scope(name or 'categorical'):
    seed = sanitize_seed(seed)
    seed, _ = _get_seed_and_old_salt(seed)
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
    seed, _ = _get_seed_and_old_salt(seed)
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
    seed, _ = _get_seed_and_old_salt(seed)
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
    seed, _ = _get_seed_and_old_salt(seed)
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
    seed, _ = _get_seed_and_old_salt(seed)
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
    seed, _ = _get_seed_and_old_salt(seed)
    return tf.random.stateless_uniform(
        shape=shape, seed=seed, minval=minval, maxval=maxval, dtype=dtype)
