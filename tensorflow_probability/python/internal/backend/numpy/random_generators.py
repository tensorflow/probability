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

import numpy as np
import numpy as onp  # pylint: disable=reimported

from tensorflow_probability.python.internal.backend.numpy import _utils as utils
from tensorflow_probability.python.internal.backend.numpy import ops
from tensorflow_probability.python.internal.backend.numpy.numpy_math import softmax as _softmax


__all__ = [
    'stateless_binomial',
    'stateless_categorical',
    'gamma',
    'stateless_gamma',
    'stateless_normal',
    'stateless_parameterized_truncated_normal',
    'stateless_poisson',
    'stateless_shuffle',
    'stateless_uniform',
    'set_seed',
    # 'all_candidate_sampler',
    # 'experimental',
    # 'fixed_unigram_candidate_sampler',
    # 'learned_unigram_candidate_sampler',
    # 'log_uniform_candidate_sampler',
    # 'stateless_truncated_normal',
    # 'truncated_normal',
    # 'uniform_candidate_sampler',
]


JAX_MODE = False


def _ensure_shape_tuple(t):
  try:
    return tuple(int(x) for x in onp.array(t))
  except TypeError:
    pass
  try:
    return (int(onp.array(t)),)
  except TypeError:
    pass
  raise TypeError('Non-shape-like value: {} (type {})'.format(t, type(t)))


def _bcast_shape(base_shape, args):
  bcast_shape = _ensure_shape_tuple(base_shape)
  for arg in args:
    bcast_shape = ops.broadcast_shape(bcast_shape, np.asarray(arg).shape)
  return bcast_shape


def _binomial(shape, seed, counts, probs, output_dtype=np.int32, name=None):  # pylint: disable=unused-argument
  """Massaging dtype and nan handling of np.random.binomial."""
  rng = np.random if seed is None else np.random.RandomState(seed & 0xffffffff)
  invalid_count = (np.int64(counts) < 0) != (counts < 0)
  if np.any(invalid_count):
    raise ValueError('int64 overflow: {} -> {}'.format(
        counts[np.where(invalid_count)],
        np.int64(counts)[np.where(invalid_count)]))
  probs = np.where(counts > 0, probs, 0)
  nans = np.isnan(probs)
  probs_for_np = np.where(nans, 0, probs)
  samps = rng.binomial(np.int64(counts), np.float64(probs_for_np), shape)
  return np.where(nans, np.nan, samps.astype(utils.numpy_dtype(output_dtype)))


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
  """Jax implementation of `tf.random.stateless_categorical`."""
  dtype = utils.numpy_dtype(dtype or np.int64)
  if not hasattr(logits, 'shape') or not hasattr(logits, 'dtype'):
    logits = np.array(logits, np.float32)
  import jax.random as jaxrand  # pylint: disable=g-import-not-at-top
  if seed is None:
    raise ValueError('Must provide PRNGKey to sample in JAX.')
  z = jaxrand.gumbel(
      key=seed, shape=logits.shape + (num_samples,), dtype=logits.dtype)
  return np.argmax(np.expand_dims(logits, -1) + z, axis=-2).astype(dtype)


def _gamma(shape, alpha, beta=None, dtype=np.float32, seed=None,
           name=None):  # pylint: disable=unused-argument
  rng = np.random if seed is None else np.random.RandomState(seed & 0xffffffff)
  scale = 1. if beta is None else (1. / beta)
  shape = _ensure_shape_tuple(shape)
  return rng.gamma(shape=alpha, scale=scale, size=shape).astype(dtype)


def _gamma_jax(shape, alpha, beta=None, dtype=np.float32, seed=None, name=None):  # pylint: disable=unused-argument
  """JAX-based reparameterized gamma sampler."""
  dtype = utils.common_dtype([alpha, beta], dtype_hint=dtype)
  alpha = np.array(alpha, dtype=dtype)
  beta = None if beta is None else np.array(beta, dtype=dtype)
  shape = _ensure_shape_tuple(shape)
  import jax.random as jaxrand  # pylint: disable=g-import-not-at-top
  if seed is None:
    raise ValueError('Must provide PRNGKey to sample in JAX.')
  # TODO(srvasude): Sample in the given dtype once
  # https://github.com/google/jax/issues/2130 is fixed.
  samps = jaxrand.gamma(
      key=seed, a=alpha, shape=shape, dtype=np.float64).astype(dtype)
  # Match the 0->tiny behavior of tf.random.gamma.
  return np.maximum(np.finfo(dtype).tiny,
                    samps if beta is None else samps / beta)


def _normal(shape, mean=0.0, stddev=1.0, dtype=np.float32, seed=None,
            name=None):  # pylint: disable=unused-argument
  rng = np.random if seed is None else np.random.RandomState(seed & 0xffffffff)
  dtype = utils.common_dtype([mean, stddev], dtype_hint=dtype)
  shape = _bcast_shape(shape, [mean, stddev])
  return rng.normal(loc=mean, scale=stddev, size=shape).astype(dtype)


def _normal_jax(shape, mean=0.0, stddev=1.0, dtype=np.float32, seed=None,
                name=None):  # pylint: disable=unused-argument
  dtype = utils.common_dtype([mean, stddev], dtype_hint=dtype)
  shape = _bcast_shape(shape, [mean, stddev])
  import jax.random as jaxrand  # pylint: disable=g-import-not-at-top
  if seed is None:
    raise ValueError('Must provide PRNGKey to sample in JAX.')
  return jaxrand.normal(key=seed, shape=shape, dtype=dtype) * stddev + mean


def _poisson(shape, lam, dtype=np.float32, seed=None,
             name=None):  # pylint: disable=unused-argument
  rng = np.random if seed is None else np.random.RandomState(seed & 0xffffffff)
  dtype = utils.common_dtype([lam], dtype_hint=dtype)
  shape = _ensure_shape_tuple(shape)
  return rng.poisson(lam=lam, size=shape).astype(dtype)


if JAX_MODE:
  from functools import partial  # pylint: disable=g-import-not-at-top
  from jax import jit  # pylint: disable=g-import-not-at-top
  from jax import lax  # pylint: disable=g-import-not-at-top
  from jax import random  # pylint: disable=g-import-not-at-top

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


def _poisson_jax(shape, lam, dtype=np.float32, seed=None,
                 name=None, max_iters=None):  # pylint: disable=unused-argument
  """Jax Poisson random sampler."""
  # TODO(b/146674643): use transformed rejection sampling with lam > 10.
  lam = np.array(lam)
  return _poisson_jax_impl(lam, seed, _ensure_shape_tuple(shape), dtype, name,
                           max_iters)


def _shuffle(value, seed=None, name=None):  # pylint: disable=unused-argument
  rng = np.random if seed is None else np.random.RandomState(seed & 0xffffffff)
  ret = np.array(value)
  rng.shuffle(ret)
  return ret


def _shuffle_jax(value, seed=None, name=None):  # pylint: disable=unused-argument
  import jax.random as jaxrand  # pylint: disable=g-import-not-at-top
  if seed is None:
    raise ValueError('Must provide PRNGKey to sample in JAX.')
  return jaxrand.permutation(seed, value, axis=0, independent=True)


def _truncated_normal(
    shape, seed, means=0.0, stddevs=1.0, minvals=-2.0, maxvals=2.0, name=None):  # pylint: disable=unused-argument
  from scipy import stats  # pylint: disable=g-import-not-at-top
  rng = np.random if seed is None else np.random.RandomState(seed & 0xffffffff)
  std_low = (minvals - means) / stddevs
  std_high = (maxvals - means) / stddevs
  std_samps = stats.truncnorm.rvs(
      std_low, std_high, size=shape, random_state=rng)
  return std_samps * stddevs + means


def _truncated_normal_jax(
    shape, seed, means=0.0, stddevs=1.0, minvals=-2.0, maxvals=2.0, name=None):  # pylint: disable=unused-argument
  import jax.random as jaxrand  # pylint: disable=g-import-not-at-top
  if seed is None:
    raise ValueError('Must provide PRNGKey to sample in JAX.')
  dtype = utils.common_dtype([means, stddevs, minvals, maxvals])
  std_low = (minvals - means) / stddevs
  std_high = (maxvals - means) / stddevs
  std_samps = jaxrand.truncated_normal(seed, std_low, std_high, shape, dtype)
  return std_samps * stddevs + means


def _uniform(shape, minval=0, maxval=None, dtype=np.float32, seed=None,
             name=None):  # pylint: disable=unused-argument
  """Numpy uniform random sampler."""
  rng = np.random if seed is None else np.random.RandomState(seed & 0xffffffff)
  if minval is not None:
    minval = ops.convert_to_tensor(minval, dtype=dtype)
  if maxval is not None:
    maxval = ops.convert_to_tensor(maxval, dtype=dtype)
  if np.issubdtype(dtype, np.integer):
    if maxval is None:
      if minval is None:
        return rng.randint(
            np.iinfo(dtype).min, np.iinfo(dtype).max + 1, size=shape
            ).astype(dtype)
      raise ValueError('Must provide maxval for integer sampling')
    return rng.randint(low=minval, high=maxval, size=shape, dtype=dtype)
  maxval = 1 if maxval is None else maxval
  shape = _bcast_shape(shape, [minval, maxval])
  return rng.uniform(low=minval, high=maxval, size=shape).astype(dtype)


def _uniform_jax(shape, minval=0, maxval=None, dtype=np.float32, seed=None,
                 name=None):  # pylint: disable=unused-argument
  """Jax uniform random sampler."""
  import jax.random as jaxrand  # pylint: disable=g-import-not-at-top
  if seed is None:
    raise ValueError('Must provide PRNGKey to sample in JAX.')
  dtype = utils.common_dtype([minval, maxval], dtype_hint=dtype)
  final_rank = max([len(shape), len(np.shape(minval)), len(np.shape(maxval))])
  if np.issubdtype(dtype, np.integer):
    if maxval is None:
      raise ValueError(
          'Must specify maxval for integer dtype {}.'.format(dtype))
    shape = _bcast_shape(shape, [minval, maxval])
    # We must match ranks, as lax.max refuses to broadcast different-rank args.
    minval = minval + np.zeros([1] * final_rank, dtype=dtype)
    return jaxrand.randint(key=seed, shape=shape, minval=minval, maxval=maxval,
                           dtype=dtype)
  else:
    maxval = ops.convert_to_tensor(1, dtype) if maxval is None else maxval
    shape = _bcast_shape(shape, [minval, maxval])
    # We must match ranks, as lax.max refuses to broadcast different-rank args.
    minval = minval + np.zeros([1] * final_rank, dtype=dtype)
    maxval = maxval + np.zeros([1] * final_rank, dtype=dtype)
    return jaxrand.uniform(key=seed, shape=shape, dtype=dtype, minval=minval,
                           maxval=maxval)


# --- Begin Public Functions --------------------------------------------------

stateless_binomial = utils.copy_docstring(
    'tf.random.stateless_binomial',
    _binomial)

# TODO(b/147874898): rewrite samplers to use stateless signature. In the
# meantime, we copy docstrings from stateful random samplers.
stateless_categorical = utils.copy_docstring(
    'tf.random.categorical',
    _categorical_jax if JAX_MODE else _categorical)

stateless_gamma = utils.copy_docstring(
    'tf.random.gamma',
    _gamma_jax if JAX_MODE else _gamma)


# TODO(b/147874898): Delete this method.
def gamma(shape, alpha, beta=None, dtype=np.float32, seed=None, name=None):
  """Handles the difference in shape parameter interpretation."""
  # While we still have usages of tf.random.gamma and tf.random.stateless_gamma,
  # we must handle the different interpretation of the shape argument
  # between the two. `tf.random.gamma` interprets shape as a prefix.
  # `tf.random.stateless_gamma` interprets shape as the full output shape,
  # including as suffix the broadcast of alpha and beta shapes.
  scale = 1 if beta is None else beta
  shape = _ensure_shape_tuple(shape) + _bcast_shape((), [alpha, scale])
  return stateless_gamma(shape=shape, alpha=alpha, beta=beta, dtype=dtype,
                         seed=seed, name=name)


stateless_normal = utils.copy_docstring(
    'tf.random.normal',
    _normal_jax if JAX_MODE else _normal)

stateless_parameterized_truncated_normal = utils.copy_docstring(
    'tf.random.stateless_parameterized_truncated_normal',
    _truncated_normal_jax if JAX_MODE else _truncated_normal)

stateless_poisson = utils.copy_docstring(
    'tf.random.poisson',
    _poisson_jax if JAX_MODE else _poisson)

# TODO(b/147874898): Delete this method in favor of using `samplers.shuffle`.
stateless_shuffle = (_shuffle_jax if JAX_MODE else _shuffle)

stateless_uniform = utils.copy_docstring(
    'tf.random.uniform',
    _uniform_jax if JAX_MODE else _uniform)

set_seed = utils.copy_docstring(
    'tf.random.set_seed',
    (lambda seed: None if JAX_MODE  # pylint: disable=g-long-lambda
     else lambda seed: np.random.seed(seed % (2**32 - 1))))
