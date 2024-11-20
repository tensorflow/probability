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

from typing import Any, Callable, Generic, Protocol, TypeVar, runtime_checkable

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
    'effective_sample_size_predicate',
    'ParticleGatherFn',
    'ResamplingPredicate',
    'SampleAncestorsFn',
    'sequential_monte_carlo_init',
    'sequential_monte_carlo_step',
    'SequentialMonteCarloKernel',
    'SequentialMonteCarloState',
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


@util.dataclass
class SequentialMonteCarloState(Generic[State]):
  """State of sequential Monte Carlo.

  Attributes:
    state: The particles.
    log_weights: Unnormalized log weight of the particles.
    step: Current timestep.
  """

  state: State
  log_weights: Float[Array, 'num_particles']
  step: IntScalar

  def log_normalizing_constant(self) -> FloatScalar:
    """Log of the unbiased normalizing constant estimator."""
    return tfp.math.reduce_logmeanexp(self.log_weights, axis=-1)

  def effective_sample_size(self) -> FloatScalar:
    """Estimates the effective sample size."""
    norm_weights = jax.nn.softmax(self.log_weights)
    return 1.0 / jnp.sum(norm_weights**2)


@util.dataclass
class SequentialMonteCarloExtra(Generic[State, Extra]):
  """Extra outputs from sequential Monte Carlo.

  Attributes:
    incremental_log_weights: Incremental log weights for this timestep.
    kernel_extra: Extra outputs from the kernel operator.
    resampled: Whether resampling happened or not.
    ancestor_idxs: Ancestor indices.
    state_after_resampling: State after resampling but before running the SMC
      kernel.
    log_weights_after_resampling: Log weights of particles after resampling but
      before running the SMC kernel.
  """

  incremental_log_weights: Float[Array, 'num_particles']
  kernel_extra: Extra
  resampled: BoolScalar
  ancestor_idxs: Int[Array, 'num_particles']
  state_after_resampling: State
  log_weights_after_resampling: Float[Array, 'num_particles']


@runtime_checkable
class SequentialMonteCarloKernel(Protocol[State, Extra]):
  """SMC kernel, a function that proposes new states and produces the incremental log weights."""

  def __call__(
      self,
      state: State,
      step: IntScalar,
      seed: Seed,
  ) -> tuple[
      State,
      tuple[Float[Array, 'num_particles'], Extra],
  ]:
    """Perform an SMC kernel step.

    Given a previous state `x_{t - 1}^{1:K}` (`K` = number of particles) at
    timestep `(t - 1)`, an SequentialMonteCarloKernel returns:

    1. A new set of particles `x_t^{1:K}` at timestep t.
    2. The incremental log weights `iw_t^{1:K}` at step `t`.

    SMC is commonly performed on states whose dimension increases with each
    timestep (see Section 3.5 of [1]), e.g. `len(x_t) = t` and
    `len(x_{t - 1}) = t - 1`. Then, for every particle `k` in `{1, ..., K}`, the
    new states are obtained as

    ```none
    z_t^k ~ q_t(. | x_{t - 1}^k)
    x_t^k = append(x_{t - 1}^k, z_t^k)
    ```

    the incremental log weight is computed as

    ```none
    iw_t^k = log p_t(x_t^k) - log p_{t - 1}(x_{t - 1}^k)
         - log q_t(x_t[-1]^k | x_{t - 1}^k)
    ```

    where `q_t` is the proposal and `{p_t(x_t); t = 1, ..., T}` is the sequence
    of unnormalized target distributions.

    Alternatively, SMC can also be performed on states that live in the same
    space, as in the SMC samplers [2]. Then, for every particle k in
    `{1, ..., K}`, the new states are obtained as

    ```none
    x_t^k ~ q_t(x_t | x_{t - 1}^k)
    ```

    and the incremental log weight is computed as

    ```none
    iw_t^k = log p_t(x_t^k) + log r_{t - 1}(x_{t - 1}^k | x_t^k)
        - log p_{t - 1}(x_{t - 1}^k) - log q_t(x_t^k | x_{t - 1}^k)

    where `q_t`, `r_{t - 1}` are the forward and reverse kernels respectively
    and `{p_t(x_t); t = 1, ..., T}` is the sequence of unnormalized target
    distributions.

    In the most general case, the kernel should maintain a 'proper weighting'
    invariant. A set of weighted particles `x^{1:K}`, w^{1:K} is properly
    weighted w.r.t. an unnormalized target `p(x)` if

    ```none
    E[1 / K sum_k w^k f(x^k)] = c E_{pi(x)}[f(x)] for any f,
    ```

    where c is a constant and pi(x) is the normalized p(x). Commonly, c the
    normalization constant of p(x), i.e. c = int p(x) dx and pi(x) = p(x) / c.

    The SequentialMonteCarloKernel maintains the 'proper weighting invariant' in
    the sense that
    if `x_{t - 1}^{1:K}`, `w_{t - 1}^{1:K}` is properly weighted w.r.t. an
    unnormalized target `p_{t - 1}(x_{t - 1})`, then `x_t^{1:K}`, `w_t^{1:K}` is
    properly weighted w.r.t. `p_t(x_t)`, where `w_t^k = w_{t - 1}^k * iw_t^k`.

    In this setting, the unnormalized targets are only defined implicitly.

    Args:
      state: The previous particle state, `x_{t - 1}^{1:K}`.
      step: The previous timestep, `t - 1`.
      seed: A PRNG key.

    Returns:
      state: The new particles, `x_t^{1:K}`.
      extra: A 2-tuple of:
        incremental_log_weights: The incremental log weight at timestep t,
          `iw_t^{1:K}`.
        kernel_extra: Extra information returned by the kernel.

    #### References:

    [1]: Doucet, Arnaud, and Adam M. Johansen. 'A tutorial on particle filtering
         and smoothing: Fifteen years later.' Handbook of nonlinear filtering
         12.656-704 (2009): 3.
         https://www.stats.ox.ac.uk/~doucet/doucet_johansen_tutorialPF2011.pdf
    [2]: Del Moral, Pierre, Arnaud Doucet, and Ajay Jasra. 'Sequential monte
         carlo samplers.' Journal of the Royal Statistical Society Series B:
         Statistical Methodology 68.3 (2006): 411-436.
         https://academic.oup.com/jrsssb/article/68/3/411/7110641
    """


@runtime_checkable
class ResamplingPredicate(Protocol):
  """Function that decides whether to resample."""

  def __call__(self, state: SequentialMonteCarloState) -> BoolScalar:
    """Return boolean indicating whether to resample.

    Note that resampling happens before stepping the kernel.

    Args:
      state: State step `t - 1`.

    Returns:
      Whether resampling happens during the SMC step at step `t`.
    """


@types.runtime_typed
def effective_sample_size_predicate(
    state: SequentialMonteCarloState,
) -> BoolScalar:
  """A resampling predicate that uses effective sample size.

  Args:
    state: SMC state.

  Returns:
    True if the effective sample size is less than half the number of particles,
    False otherwise.
  """
  num_particles = state.log_weights.shape[0]
  return state.effective_sample_size() < num_particles / 2


@runtime_checkable
class ParticleGatherFn(Protocol[State]):
  """Function that indexes into a batch of states."""

  def __call__(
      self,
      state: State,
      indices: Int[Array, 'num_particles'],
  ) -> State:
    """Gather states at the given indices."""


@types.runtime_typed
def _defalt_pytree_gather(
    state: State,
    indices: Int[Array, 'num_particles'],
) -> State:
  """Indexes into states using the default gather.

  Assumes `state` is a pytree of arrays with a single leading batch dimension.

  Args:
    state: The particles.
    indices: The gather indices.

  Returns:
    new_state: Gathered state (with the same leading dimension).
  """
  return util.map_tree(lambda x: x[indices], state)


@types.runtime_typed
def sequential_monte_carlo_init(
    state: State,
    num_particles: int | None = None,
    initial_step: IntScalar = 0,
    weight_dtype: DType = jnp.float32,
) -> SequentialMonteCarloState[State]:
  """Initializes the sequential Monte Carlo state.

  Args:
    state: Initial state representing the SMC particles.
    num_particles: Number of particles, if `None`, it is inferred from the first
      element of `state`.
    initial_step: Initial step number.
    weight_dtype: DType of the `log_weights`.

  Returns:
    `SequentialMonteCarloState`.
  """
  if num_particles is None:
    num_particles = util.flatten_tree(state)[0].shape[0]
  return SequentialMonteCarloState(
      state=state,
      log_weights=jnp.zeros([num_particles], dtype=weight_dtype),
      step=jnp.asarray(initial_step, dtype=jnp.int32),
  )


@types.runtime_typed
def sequential_monte_carlo_step(
    smc_state: SequentialMonteCarloState[State],
    kernel: SequentialMonteCarloKernel[State, Extra],
    seed: Seed,
    resampling_pred: ResamplingPredicate = effective_sample_size_predicate,
    sample_ancestors_fn: SampleAncestorsFn = systematic_resampling,
    state_gather_fn: ParticleGatherFn[State] = _defalt_pytree_gather,
) -> tuple[
    SequentialMonteCarloState[State], SequentialMonteCarloExtra[State, Extra]
]:
  """Take a step of sequential Monte Carlo.

  Given a previous SMC state `x_{t - 1}^{1:K}`, `w_{t - 1}^{1:K}` (where `K` is
  the number of particles) at timestep `t - 1` that is properly weighted w.r.t.
  an unnormalized target `p_{t - 1}(x_{t - 1})`, returns a new SMC state
  `x_t^{1:K}`, `w_t^{1:K}` at timestep `t` that is properly weighted w.r.t.
  `p_t(x_t)`.

  Note that the unnormalized target is implicitly defined by the kernel.

  This implementation first resamples, then steps the kernel.

  Args:
    smc_state: SMC state at timestep `t - 1`, `x_{t - 1}^{1:K}`, `w_{t -
      1}^{1:K}`.
    kernel: SMC kernel.
    seed: Random seed.
    resampling_pred: Resampling predicate.
    sample_ancestors_fn: Ancestor index sampling function.
    state_gather_fn: State gather function.

  Returns:
    smc_state: SMC state at timestep t, x_t^{1:K}, w_t^{1:K}.
    smc_extra: Extra information for the SMC step.
  """
  resample_seed, kernel_seed = util.split_seed(seed, 2)

  def do_resample(
      state,
      log_weights,
      seed,
  ):
    ancestor_idxs = sample_ancestors_fn(log_weights, seed)
    new_state = state_gather_fn(state, ancestor_idxs)
    num_particles = log_weights.shape[0]
    new_log_weights = jnp.full(
        (num_particles,), tfp.math.reduce_logmeanexp(log_weights)
    )
    return (new_state, ancestor_idxs, new_log_weights)

  def dont_resample(
      state,
      log_weights,
      seed,
  ):
    del seed
    num_particles = log_weights.shape[0]
    return state, jnp.arange(num_particles), log_weights

  # NOTE: We don't explicitly disable resampling at the first step. However, if
  # we initialize the log weights to zeros, either of
  # 1. resampling according to the effective sample size criterion and
  # 2. using systematic resampling effectively disables resampling at the first
  #    step.
  # First-step resampling can always be forced via the `resampling_pred`.
  should_resample = resampling_pred(smc_state)
  state_after_resampling, ancestor_idxs, log_weights_after_resampling = (
      _smart_cond(
          should_resample,
          do_resample,
          dont_resample,
          smc_state.state,
          smc_state.log_weights,
          resample_seed,
      )
  )

  # Step kernel
  state, (incremental_log_weights, kernel_extra) = kernel(
      state_after_resampling, smc_state.step, kernel_seed
  )

  new_log_weights = log_weights_after_resampling + incremental_log_weights

  smc_state = smc_state.replace(  # pytype: disable=attribute-error
      state=state,
      log_weights=new_log_weights,
      step=smc_state.step + 1,
  )
  smc_extra = SequentialMonteCarloExtra(
      incremental_log_weights=incremental_log_weights,
      kernel_extra=kernel_extra,
      resampled=should_resample,
      ancestor_idxs=ancestor_idxs,
      state_after_resampling=state_after_resampling,
      log_weights_after_resampling=log_weights_after_resampling,
  )
  return smc_state, smc_extra


def _smart_cond(
    pred: BoolScalar,
    true_fn: Callable[..., T],
    false_fn: Callable[..., T],
    *args: Any
) -> T:
  """Like lax.cond, but shortcircuiting for static predicates."""
  static_pred = util.get_static_value(pred)
  if static_pred is None:
    return jax.lax.cond(pred, true_fn, false_fn, *args)
  elif static_pred:
    return true_fn(*args)
  else:
    return false_fn(*args)
