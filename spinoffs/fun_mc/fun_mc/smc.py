# Copyright 2024 The TensorFlow Probability Authors.
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Implementation of Sequential Monte Carlo."""

from typing import Protocol, TypeVar, runtime_checkable

from fun_mc import backend
from fun_mc import types

jax = backend.jax
jnp = backend.jnp
tfp = backend.tfp
util = backend.util
distribute_lib = backend.distribute_lib

Array = types.Array
Seed = types.Seed
Float = types.Float
Int = types.Int
Bool = types.Bool
DType = types.DType
BoolScalar = types.BoolScalar
IntScalar = types.IntScalar
FloatScalar = types.FloatScalar
State = TypeVar('State')
Extra = TypeVar('Extra')
T = TypeVar('T')

__all__ = [
    'conditional_systematic_resampling',
    'SampleAncestorsFn',
    'systematic_resampling',
]


@runtime_checkable
class SampleAncestorsFn(Protocol):
  """Function that generates ancestor indices for resampling."""

  def __call__(
      self,
      log_weights: Float[Array, 'num_particles'],
      seed: Seed,
  ) -> Int[Array, 'num_particles']:
    """Generate a set of ancestor indices from particle weights."""


@types.runtime_typed
def systematic_resampling(
    log_weights: Float[Array, 'num_particles'],
    seed: Seed,
    permute: bool = False,
) -> Int[Array, 'num_particles']:
  """Generate parent indices via systematic resampling.

  Args:
    log_weights: Unnormalized log-scale weights.
    seed: PRNG seed.
    permute: Whether to permute the parent indices. Otherwise, they are sorted
      in an ascending order.

  Returns:
    parent_idxs: parent indices such that the marginal probability that a
      randomly chosen element will be `i` is equal to `softmax(log_weights)[i]`.
  """
  shift_seed, permute_seed = util.split_seed(seed, 2)
  log_weights = jnp.where(
      jnp.isnan(log_weights),
      jnp.array(-float('inf'), log_weights.dtype),
      log_weights,
  )
  probs = jax.nn.softmax(log_weights)
  # A common situation is all -inf log_weights that creats a NaN vector.
  probs = jnp.where(
      jnp.all(jnp.isfinite(probs)), probs, jnp.ones_like(probs) / probs.shape[0]
  )
  num_particles = probs.shape[0]

  shift = util.random_uniform([], log_weights.dtype, shift_seed)
  pie = jnp.cumsum(probs) * num_particles + shift
  repeats = jnp.array(util.diff(jnp.floor(pie), prepend=0), jnp.int32)
  parent_idxs = util.repeat(
      jnp.arange(num_particles), repeats, total_repeat_length=num_particles
  )
  if permute:
    parent_idxs = util.random_permutation(parent_idxs, permute_seed)
  return parent_idxs


@types.runtime_typed
def conditional_systematic_resampling(
    log_weights: Float[Array, 'num_particles'],
    seed: Seed,
) -> Int[Array, 'num_particles']:
  """Apply conditional systematic resampling to `softmax(log_weights)`.

  Equivalent to (but typically much more efficient than) the following
  rejection sampler:

  ```python
  for i in count():
    parents = systematic_resampling(log_weights, seed=i, permute=True)
    if parents[0] == 0:
      break
  return parents
  ```

  The algorithm used is from [1].

  Args:
    log_weights: Unnormalized log-scale weights.
    seed: PRNG seed.

  Returns:
    parent_idxs: A sample from the posterior over the output of
      `systematic_resampling`, conditioned on parent_idxs[0] == 0.

  #### References

  [1]: Chopin and Singh, 'On Particle Gibbs Sampling,' Bernoulli, 2015
      https://www.jstor.org/stable/43590414
  """
  mixture_seed, shift_seed, permute_seed = util.split_seed(seed, 3)
  log_weights = jnp.where(
      jnp.isnan(log_weights),
      jnp.array(-float('inf'), log_weights.dtype),
      log_weights,
  )
  probs = jax.nn.softmax(log_weights)
  num_particles = log_weights.shape[0]

  # Sample from the posterior over shift given that parents[0] == 0. This turns
  # out to be a mixture of non-overlapping uniforms.
  scaled_w1 = num_particles * probs[0]
  r = scaled_w1 % 1.0
  prob_shift_less_than_r = r * jnp.ceil(scaled_w1) / scaled_w1
  shift = util.random_uniform(
      shape=[], dtype=log_weights.dtype, seed=shift_seed
  )
  shift = jnp.where(
      util.random_uniform(shape=[], dtype=log_weights.dtype, seed=mixture_seed)
      < prob_shift_less_than_r,
      shift * r,
      r + shift * (1 - r),
  )
  # Proceed as usual once we've figured out the shift.
  pie = jnp.cumsum(probs) * num_particles + shift
  repeats = jnp.array(util.diff(jnp.floor(pie), prepend=0), jnp.int32)
  parent_idxs = util.repeat(
      jnp.arange(num_particles), repeats, total_repeat_length=num_particles
  )
  # Permute parents[1:].
  permuted_parents = util.random_permutation(parent_idxs[1:], permute_seed)
  parents = jnp.concatenate([parent_idxs[:1], permuted_parents])
  return parents
