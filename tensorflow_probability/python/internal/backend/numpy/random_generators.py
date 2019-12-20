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
"""Numpy implementations of TensorFlow functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

# Dependency imports
import numpy as np

import tensorflow.compat.v2 as tf

from tensorflow_probability.python.internal.backend.numpy import _utils as utils
from tensorflow_probability.python.internal.backend.numpy.numpy_math import softmax as _softmax


__all__ = [
    'categorical',
    'gamma',
    'normal',
    'poisson',
    'uniform',
    'set_seed',
    # 'all_candidate_sampler',
    # 'experimental',
    # 'fixed_unigram_candidate_sampler',
    # 'learned_unigram_candidate_sampler',
    # 'log_uniform_candidate_sampler',
    # 'shuffle',
    # 'stateless_categorical',
    # 'stateless_normal',
    # 'stateless_truncated_normal',
    # 'stateless_uniform',
    # 'truncated_normal',
    # 'uniform_candidate_sampler',
]


JAX_MODE = False


def _shape(args, size):
  try:
    size = tuple(size)
  except TypeError:
    size = (size,)
  if not args:
    return size
  if len(args) == 1:
    return size + np.array(args[0]).shape
  return size + functools.reduce(np.broadcast, args).shape


def _categorical(logits, num_samples, dtype=None, seed=None, name=None):  # pylint: disable=unused-argument
  rng = np.random if seed is None else np.random.RandomState(seed & 0xffffffff)
  dtype = utils.numpy_dtype(dtype or np.int64)
  if not hasattr(logits, 'shape'):
    logits = np.array(logits, np.float32)
  probs = _softmax(logits)
  n = logits.shape[-1]
  return np.apply_along_axis(lambda p: rng.choice(n, p=p, size=num_samples), 1,
                             probs)


def _categorical_jax(logits, num_samples, dtype=None, seed=None, name=None):  # pylint: disable=unused-argument
  dtype = utils.numpy_dtype(dtype or np.int64)
  if not hasattr(logits, 'shape') or not hasattr(logits, 'dtype'):
    logits = np.array(logits, np.float32)
  import jax.random as jaxrand  # pylint: disable=g-import-not-at-top
  if seed is None:
    raise ValueError('Must provide PRNGKey to sample in JAX.')
  z = jaxrand.gumbel(
      key=seed, shape=logits.shape + (num_samples,), dtype=logits.dtype)
  return np.argmax(np.expand_dims(logits, -1) + z, axis=-2).astype(dtype)


def _gamma(shape, alpha, beta=None, dtype=tf.float32, seed=None,
           name=None):  # pylint: disable=unused-argument
  rng = np.random if seed is None else np.random.RandomState(seed & 0xffffffff)
  dtype = utils.common_dtype([alpha, beta], dtype_hint=dtype)
  scale = 1. if beta is None else (1. / beta)
  shape = _shape([alpha, scale], shape)
  return rng.gamma(shape=alpha, scale=scale, size=shape).astype(dtype)


def _gamma_jax(shape, alpha, beta=None, dtype=tf.float32, seed=None, name=None):  # pylint: disable=unused-argument
  dtype = utils.common_dtype([alpha, beta], dtype_hint=dtype)
  shape = _shape([alpha, beta], shape)
  import jax.random as jaxrand  # pylint: disable=g-import-not-at-top
  if seed is None:
    raise ValueError('Must provide PRNGKey to sample in JAX.')
  samps = jaxrand.gamma(key=seed, a=alpha, shape=shape, dtype=dtype)
  return samps if beta is None else samps / beta


def _normal(shape, mean=0.0, stddev=1.0, dtype=tf.float32, seed=None,
            name=None):  # pylint: disable=unused-argument
  rng = np.random if seed is None else np.random.RandomState(seed & 0xffffffff)
  dtype = utils.common_dtype([mean, stddev], dtype_hint=dtype)
  shape = _shape([mean, stddev], shape)
  return rng.normal(loc=mean, scale=stddev, size=shape).astype(dtype)


def _normal_jax(shape, mean=0.0, stddev=1.0, dtype=tf.float32, seed=None,
                name=None):  # pylint: disable=unused-argument
  dtype = utils.common_dtype([mean, stddev], dtype_hint=dtype)
  shape = _shape([mean, stddev], shape)
  import jax.random as jaxrand  # pylint: disable=g-import-not-at-top
  if seed is None:
    raise ValueError('Must provide PRNGKey to sample in JAX.')
  return jaxrand.normal(key=seed, shape=shape, dtype=dtype) * stddev + mean


def _poisson(shape, lam, dtype=tf.float32, seed=None,
             name=None):  # pylint: disable=unused-argument
  rng = np.random if seed is None else np.random.RandomState(seed & 0xffffffff)
  dtype = utils.common_dtype([lam], dtype_hint=dtype)
  shape = _shape([lam], shape)
  return rng.poisson(lam=lam, size=shape).astype(dtype)


if JAX_MODE:
  from jax import jit  # pylint: disable=g-import-not-at-top
  from jax import lax  # pylint: disable=g-import-not-at-top
  from jax import random  # pylint: disable=g-import-not-at-top
  from jax.util import partial  # pylint: disable=g-import-not-at-top

  # Jitting the implementation because
  # sampling is very slow outside of JIT
  # and causes tests to timeout.
  @partial(jit, static_argnums=(2, 3, 4, 5))
  def _poisson_jax_impl(lam, seed, shape, dtype, name, max_iters):  # pylint: disable=unused-argument
    """Jit-able implementation of Knuth Poisson random sampler."""
    # Based on the TF implementation for lam < 10 in
    # https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/kernels/random_poisson_op.cc
    # which uses the Knuth algorithm.
    # Reference:
    # https://en.wikipedia.org/wiki/Poisson_distribution#Generating_Poisson-distributed_random_variables
    # This implementation can be improved using the
    # transformed rejection sampling algorithm for
    # lam > 10. A reference implementation can be found here:
    # https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/kernels/random_poisson_op.cc#L159-L239

    shape = _shape([lam], shape)
    max_iters = (max_iters
                 if max_iters is not None
                 else np.iinfo(np.int32).max)

    def body_fn(carry):
      """Inner loop of Knuth algorithm."""
      i, k, rng, log_prod = carry
      rng, subkey = random.split(rng)
      k = np.where(log_prod > -lam, k + 1, k)
      return i + 1, k, rng, log_prod + np.log(random.uniform(subkey, shape))

    def cond_fn(carry):
      i, log_prod = carry[0], carry[3]
      return np.any(log_prod > -lam) & (i < max_iters)

    k = lax.while_loop(cond_fn, body_fn,
                       (0, np.zeros(shape, dtype=np.int32),
                        seed, np.zeros(shape)))[1]
    return (k - 1).astype(dtype)


def _poisson_jax(shape, lam, dtype=tf.float32, seed=None,
                 name=None, max_iters=None):  # pylint: disable=unused-argument
  """Jax Poisson random sampler."""
  # TODO(b/146674643): use transformed rejection sampling with lam > 10.
  return _poisson_jax_impl(lam, seed, shape, dtype, name, max_iters)


def _uniform(shape, minval=0, maxval=None, dtype=tf.float32, seed=None,
             name=None):  # pylint: disable=unused-argument
  rng = np.random if seed is None else np.random.RandomState(seed & 0xffffffff)
  dtype = utils.common_dtype([minval, maxval], dtype_hint=dtype)
  maxval = 1 if maxval is None else maxval
  shape = _shape([minval, maxval], shape)
  return rng.uniform(low=minval, high=maxval, size=shape).astype(dtype)


def _uniform_jax(shape, minval=0, maxval=None, dtype=tf.float32, seed=None,
                 name=None):  # pylint: disable=unused-argument
  """Jax uniform random sampler."""
  import jax.random as jaxrand  # pylint: disable=g-import-not-at-top
  if seed is None:
    raise ValueError('Must provide PRNGKey to sample in JAX.')
  dtype = utils.common_dtype([minval, maxval], dtype_hint=dtype)
  if np.issubdtype(dtype, np.integer):
    if maxval is None:
      raise ValueError(
          'Must specify maxval for integer dtype {}.'.format(dtype))
    shape = _shape([], shape)
    return jaxrand.randint(key=seed, shape=shape, minval=minval, maxval=maxval,
                           dtype=dtype)
  else:
    maxval = 1 if maxval is None else maxval
    shape = _shape([], shape)
    return jaxrand.uniform(key=seed, shape=shape, dtype=dtype, minval=minval,
                           maxval=maxval)


# --- Begin Public Functions --------------------------------------------------


categorical = utils.copy_docstring(
    tf.random.categorical,
    _categorical_jax if JAX_MODE else _categorical)

gamma = utils.copy_docstring(
    tf.random.gamma,
    _gamma_jax if JAX_MODE else _gamma)

normal = utils.copy_docstring(
    tf.random.normal,
    _normal_jax if JAX_MODE else _normal)

poisson = utils.copy_docstring(
    tf.random.poisson,
    _poisson_jax if JAX_MODE else _poisson)

uniform = utils.copy_docstring(
    tf.random.uniform,
    _uniform_jax if JAX_MODE else _uniform)

set_seed = utils.copy_docstring(
    tf.random.set_seed,
    (lambda seed: None if JAX_MODE  # pylint: disable=g-long-lambda
     else lambda seed: np.random.seed(seed % (2**32 - 1))))
