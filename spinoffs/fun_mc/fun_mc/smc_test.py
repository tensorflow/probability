# Copyright 2024 The TensorFlow Probability Authors.
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
import functools

# Dependency imports

from absl.testing import parameterized
import jax as real_jax
import mock
import tensorflow.compat.v2 as real_tf
from tensorflow_probability.python.internal import test_util as tfp_test_util
from fun_mc import backend
from fun_mc import fun_mc_lib as fun_mc
from fun_mc import smc
from fun_mc import test_util
from fun_mc import types

jax = backend.jax
jnp = backend.jnp
tfp = backend.tfp
util = backend.util
tfd = tfp.distributions
distribute_lib = backend.distribute_lib
Root = tfd.JointDistributionCoroutine.Root
Array = types.Array
Seed = types.Seed
Float = types.Float
Int = types.Int
Bool = types.Bool
BoolScalar = types.BoolScalar
IntScalar = types.IntScalar
FloatScalar = types.FloatScalar


real_tf.enable_v2_behavior()
real_tf.experimental.numpy.experimental_enable_numpy_behavior()
real_jax.config.update('jax_enable_x64', True)

BACKEND = None  # Rewritten by backends/rewrite.py.


def _test_seed() -> Seed:
  seed = tfp_test_util.test_seed() % (2**32 - 1)
  if BACKEND == 'backend_jax':
    return jax.random.PRNGKey(seed)
  else:
    return util.make_tensor_seed([seed, 0])


@types.runtime_typed
def basic_kernel(
    state: Float[Array, 'num_particles'],
    step: IntScalar,
    seed: Seed,
) -> tuple[
    Float[Array, 'num_particles'],
    tuple[Float[Array, 'num_particles'], tuple[()]],
]:
  del step
  random_weights = util.random_uniform(state.shape, state.dtype, seed)
  log_weights = jnp.log(random_weights)
  return state, (log_weights, ())


@types.runtime_typed
def ess_kernel(
    state: Float[Array, 'num_particles'],
    step: IntScalar,
    seed: Seed,
) -> tuple[
    Float[Array, 'num_particles'],
    tuple[Float[Array, 'num_particles'], tuple[()]],
]:
  """Make the ESS low only for first 3 timesteps."""
  del seed
  num_particles = state.shape[0]
  high_ess_log_weights = jnp.zeros([num_particles], state.dtype)
  low_ess_log_weights = jnp.stack(
      [jnp.zeros([], dtype=state.dtype)]
      + [jnp.array(-jnp.inf, dtype=state.dtype)] * (num_particles - 1),
      0,
  )
  log_weights = jnp.where(step < 3, low_ess_log_weights, high_ess_log_weights)
  return state, (log_weights, ())


@types.runtime_typed
def kernel_log_weights_eq_two(
    state: Float[Array, 'num_particles'],
    step: IntScalar,
    seed: Seed,
) -> tuple[
    Float[Array, 'num_particles'],
    tuple[Float[Array, 'num_particles'], tuple[()]],
]:
  """Always returns log weight equals 2."""
  del seed, step
  num_particles = state.shape[0]
  log_weights = jnp.full([num_particles], 2, dtype=jnp.float32)
  return state, (log_weights, ())


@types.runtime_typed
def kernel_log_weights_eq_step(
    state: Float[Array, 'num_particles'],
    step: IntScalar,
    seed: Seed,
) -> tuple[
    Float[Array, 'num_particles'],
    tuple[Float[Array, 'num_particles'], tuple[()]],
]:
  """Log weight equals timestep."""
  del seed
  num_particles = state.shape[0]
  log_weights = jnp.full([num_particles], step, dtype=jnp.float32)
  return state, (log_weights, ())


@types.runtime_typed
def kernel_extra_is_finished(
    state: Float[Array, 'num_particles'],
    step: IntScalar,
    seed: Seed,
    num_timesteps: int,
) -> tuple[
    Float[Array, 'num_particles'],
    tuple[Float[Array, 'num_particles'], BoolScalar],
]:
  """Returns an extra `is_finished` value based on `num_timesteps`."""
  del seed
  num_particles = state.shape[0]
  log_weights = jnp.zeros([num_particles], state.dtype)
  is_finished = step >= num_timesteps - 1
  return state, (log_weights, is_finished)


@types.runtime_typed
def kernel_log_weights_eq_neg_inf_if_state_lt_zero(
    state: Float[Array, 'num_particles'],
    step: IntScalar,
    seed: Seed,
) -> tuple[
    Float[Array, 'num_particles'],
    tuple[Float[Array, 'num_particles'], tuple[()]],
]:
  """Decrements the state and returns -inf weight when it dips below zero."""
  del seed, step
  new_state = state - 1
  num_particles = state.shape[0]
  neg_infs = jnp.full([num_particles], -jnp.inf, dtype=state.dtype)
  zeros = jnp.zeros([num_particles], dtype=state.dtype)
  log_weights = jnp.where(new_state < 0, neg_infs, zeros)
  return new_state, (log_weights, ())


@types.runtime_typed
def always_predicate(
    state: smc.SequentialMonteCarloState,
) -> BoolScalar:
  del state
  return True


@types.runtime_typed
def never_predicate(
    state: smc.SequentialMonteCarloState,
) -> BoolScalar:
  del state
  return False


class SMCTest(tfp_test_util.TestCase):

  @property
  def _dtype(self):
    raise NotImplementedError()

  def _constant(self, value):
    return jnp.array(value, self._dtype)

  @parameterized.parameters(True, False)
  def test_systematic_resampling(self, permute):
    seed = _test_seed()

    num_replications = 10000
    weights = self._constant([0.0, 0.5, 0.2, 0.3, 0.0])
    log_weights = jnp.log(weights)

    def kernel(seed):
      seed, sample_seed = util.split_seed(seed, 2)
      parents = smc.systematic_resampling(
          log_weights, seed=sample_seed, permute=permute
      )
      return seed, parents

    _, parents = jax.jit(
        lambda seed: fun_mc.trace(seed, kernel, num_replications)
    )(seed)

    # [num_samples, parents, parents]
    freqs = jnp.mean(
        jnp.array(
            parents[..., jnp.newaxis] == jnp.arange(len(weights)), jnp.float32
        ),
        (0, 1),
    )

    self.assertAllClose(freqs, weights, atol=0.05)

    if permute:
      mean_index = jnp.sum(weights * jnp.arange(len(weights)))
      self.assertAllClose(
          jnp.mean(parents, 0), [mean_index] * len(weights), atol=0.05
      )

  def test_conditional_systematic_resampling(self):
    seed = _test_seed()

    num_replications = 10000
    weights = self._constant([0.2, 0.5, 0.2, 0.1, 0.0])
    log_weights = jnp.log(weights)

    def kernel(seed):
      seed, systematic_seed, cond_seed = util.split_seed(seed, 3)
      systematic_parents = smc.systematic_resampling(
          log_weights, seed=systematic_seed, permute=True
      )
      conditional_parents = smc.systematic_resampling(
          log_weights,
          seed=cond_seed,
      )
      return seed, (systematic_parents, conditional_parents)

    _, (systematic_parents, conditional_parents) = jax.jit(
        lambda seed: fun_mc.trace(seed, kernel, num_replications)
    )(seed)

    self.assertFalse(jnp.all(systematic_parents[:, 0] == 0))
    self.assertTrue(jnp.all(conditional_parents[:, 0] == 0))

    accepted_samples = jnp.array(systematic_parents[:, 0] == 0, jnp.float32)
    rejection_freqs = jnp.sum(
        jnp.mean(
            accepted_samples[:, jnp.newaxis, jnp.newaxis]
            * jnp.array(
                systematic_parents[..., jnp.newaxis]
                == jnp.arange(len(weights)),
                jnp.float32,
            ),
            1,
        ),
        0,
    ) / jnp.sum(accepted_samples)
    conditional_freqs = jnp.mean(
        jnp.array(
            conditional_parents[..., jnp.newaxis] == jnp.arange(len(weights)),
            jnp.float32,
        ),
        (0, 1),
    )
    self.assertAllClose(rejection_freqs, conditional_freqs, atol=0.05)

  def test_smc_runs_and_shapes_correct(self):
    num_particles = 3
    num_timesteps = 20

    def kernel(smc_state, seed):
      step_seed, seed = util.split_seed(seed, 2)
      smc_state, _ = smc.sequential_monte_carlo_step(
          smc_state,
          kernel=basic_kernel,
          seed=step_seed,
      )
      return (smc_state, seed), ()

    @jax.jit
    def run_smc(seed):
      (smc_state, _), _ = fun_mc.trace(
          (
              smc.sequential_monte_carlo_init(
                  jnp.zeros((num_particles,), dtype=self._dtype),
                  weight_dtype=self._dtype,
              ),
              seed,
          ),
          kernel,
          num_timesteps,
      )
      return smc_state

    smc_state = run_smc(_test_seed())
    self.assertEqual(smc_state.state.shape, (num_particles,))
    self.assertEqual(smc_state.log_weights.shape, (num_particles,))
    self.assertEqual(smc_state.step.shape, ())
    self.assertEqual(smc_state.log_normalizing_constant().shape, ())
    self.assertEqual(smc_state.effective_sample_size().shape, ())

  @parameterized.parameters(True, False)
  def test_static_resampling(self, resample):
    num_particles = 3
    num_timesteps = 20

    patch_cond = self.enter_context(
        mock.patch.object(
            jax.lax, 'cond', autospec=True, side_effect=jax.lax.cond
        )
    )

    def kernel(smc_state, seed):
      step_seed, seed = util.split_seed(seed, 2)
      smc_state, extra = smc.sequential_monte_carlo_step(
          smc_state,
          kernel=basic_kernel,
          seed=step_seed,
          resampling_pred=lambda _: resample,
      )
      return (smc_state, seed), extra.resampled

    @jax.jit
    def run_smc(seed):
      _, resampled_trace = fun_mc.trace(
          (
              smc.sequential_monte_carlo_init(
                  state=jnp.zeros((num_particles,), dtype=self._dtype),
                  weight_dtype=self._dtype,
              ),
              seed,
          ),
          kernel,
          num_timesteps,
      )
      return resampled_trace

    resampled_trace = run_smc(_test_seed())
    self.assertFalse(patch_cond.called)
    self.assertAllEqual(
        jnp.full(resampled_trace.shape, resample), resampled_trace
    )

  def test_ess_resampling(self):
    num_particles = 3
    num_timesteps = 20

    patch_cond = self.enter_context(
        mock.patch.object(
            jax.lax, 'cond', autospec=True, side_effect=jax.lax.cond
        )
    )

    def kernel(smc_state, seed):
      step_seed, seed = util.split_seed(seed, 2)
      smc_state, extra = smc.sequential_monte_carlo_step(
          smc_state,
          kernel=ess_kernel,
          seed=step_seed,
          resampling_pred=smc.effective_sample_size_predicate,
      )
      return (smc_state, seed), extra.resampled

    @jax.jit
    def run_smc(seed):
      _, resampled_trace = fun_mc.trace(
          (
              smc.sequential_monte_carlo_init(
                  state=jnp.zeros((num_particles,), dtype=self._dtype),
                  weight_dtype=self._dtype,
              ),
              seed,
          ),
          kernel,
          num_timesteps,
      )
      return resampled_trace

    resampled_trace = run_smc(_test_seed())
    self.assertTrue(patch_cond.called)

    # The initial weights are zero so ESS is high so there is no resampling.
    self.assertTrue(~resampled_trace[0])
    # The next weights for the next 3 steps have low ESS by design so there is
    # resampling.
    self.assertTrue(jnp.all(resampled_trace[1:4]))
    # The weights for the rest of the steps have high ESS by design so there is
    # no resampling.
    self.assertTrue(jnp.all(~resampled_trace[4:]))

  @parameterized.product(
      [
          dict(
              max_num_timesteps=12,
              num_timesteps=12,
          ),
          dict(
              max_num_timesteps=12,
              num_timesteps=10,
          ),
          dict(
              max_num_timesteps=12,
              num_timesteps=1,
          ),
          dict(
              max_num_timesteps=1,
              num_timesteps=1,
          ),
      ],
      stop_early=[True, False],
      resampling_pred=[
          always_predicate,
          never_predicate,
      ],
  )
  def test_log_normalizing_constant(
      self,
      max_num_timesteps,
      num_timesteps,
      stop_early,
      resampling_pred,
  ):
    num_particles = 3

    def kernel(smc_state, seed):
      step_seed, seed = util.split_seed(seed, 2)
      smc_state, _ = smc.sequential_monte_carlo_step(
          smc_state,
          kernel=kernel_log_weights_eq_two,
          resampling_pred=resampling_pred,
          seed=step_seed,
      )
      if stop_early:  # pylint: disable=cell-var-from-loop
        return (smc_state, seed), ()
      else:
        return (smc_state, seed), smc_state.log_normalizing_constant()

    @jax.jit
    def run_smc(seed):
      def stop_fn(smc_state_and_seed, _):
        smc_state, _ = smc_state_and_seed
        return smc_state.step >= num_timesteps

      (smc_state, _), log_normalizing_constants = fun_mc.trace(
          (
              smc.sequential_monte_carlo_init(
                  state=jnp.zeros((num_particles,), dtype=self._dtype),
                  weight_dtype=self._dtype,
              ),
              seed,
          ),
          kernel,  # pylint: disable=cell-var-from-loop
          max_num_timesteps,
          stop_fn=stop_fn if stop_early else None,  # pylint: disable=cell-var-from-loop
      )
      if stop_early:  # pylint: disable=cell-var-from-loop
        return smc_state.log_normalizing_constant()
      else:
        return log_normalizing_constants[num_timesteps - 1]

    log_normalizing_constant = run_smc(_test_seed())
    # log_weights are 2 at each step and n step are taken, so total is n * 2.
    self.assertAllClose(log_normalizing_constant, 2.0 * num_timesteps)

  @parameterized.product(
      [
          dict(
              max_num_timesteps=12,
              num_timesteps=12,
          ),
          dict(
              max_num_timesteps=12,
              num_timesteps=10,
          ),
          dict(
              max_num_timesteps=12,
              num_timesteps=1,
          ),
          dict(
              max_num_timesteps=1,
              num_timesteps=1,
          ),
      ],
      stop_early=[True, False],
      resampling_pred=[
          always_predicate,
          never_predicate,
      ],
  )
  def test_log_normalizing_constant_time_dependent(
      self,
      max_num_timesteps,
      num_timesteps,
      stop_early,
      resampling_pred,
  ):
    num_particles = 3

    def kernel(smc_state, seed):
      step_seed, seed = util.split_seed(seed, 2)
      smc_state, _ = smc.sequential_monte_carlo_step(
          smc_state,
          kernel=kernel_log_weights_eq_step,
          resampling_pred=resampling_pred,
          seed=step_seed,
      )
      if stop_early:  # pylint: disable=cell-var-from-loop
        return (smc_state, seed), ()
      else:
        return (smc_state, seed), smc_state.log_normalizing_constant()

    @jax.jit
    def run_smc(seed):
      def stop_fn(smc_state_and_seed, _):
        smc_state, _ = smc_state_and_seed
        return smc_state.step >= num_timesteps

      (smc_state, _), log_normalizing_constants = fun_mc.trace(
          (
              smc.sequential_monte_carlo_init(
                  state=jnp.zeros((num_particles,), dtype=self._dtype),
                  weight_dtype=self._dtype,
              ),
              seed,
          ),
          kernel,  # pylint: disable=cell-var-from-loop
          max_num_timesteps,
          stop_fn=stop_fn if stop_early else None,  # pylint: disable=cell-var-from-loop
      )
      if stop_early:  # pylint: disable=cell-var-from-loop
        return smc_state.log_normalizing_constant()
      else:
        return log_normalizing_constants[num_timesteps - 1]

    log_normalizing_constant = run_smc(_test_seed())
    # log_weights are 2 at each step and n step are taken, so total is n * 2.
    self.assertAllClose(
        log_normalizing_constant, jnp.sum(jnp.arange(num_timesteps))
    )

  @parameterized.named_parameters(
      ('12_steps_12_max_stop_early', 12, 12, True),
      ('10_steps_12_max_stop_early', 10, 12, True),
      ('1_step_10_max_stop_early', 1, 10, True),
      ('1_step_1_max_stop_early', 1, 1, True),
      ('12_steps_12_max', 12, 12, False),
      ('10_steps_12_max', 10, 12, False),
      ('1_step_10_max', 1, 10, False),
      ('1_step_1_max', 1, 1, False),
  )
  def test_log_normalizing_constant_no_resampling_last_step(
      self,
      num_timesteps,
      max_num_timesteps,
      stop_early,
  ):
    """Check log normalizing constant when not resampling only on the last step."""
    num_particles = 3

    def resampling_pred(state):
      return state.step < num_timesteps - 1

    def kernel(smc_state, seed):
      step_seed, seed = util.split_seed(seed, 2)
      smc_state, extra = smc.sequential_monte_carlo_step(
          smc_state,
          kernel=kernel_log_weights_eq_two,
          resampling_pred=resampling_pred,
          seed=step_seed,
      )
      return (smc_state, seed), (
          extra.resampled,
          smc_state.log_normalizing_constant(),
      )

    @jax.jit
    def run_smc(seed):
      def stop_fn(smc_state_and_seed, _):
        smc_state, _ = smc_state_and_seed
        return smc_state.step >= num_timesteps

      (smc_state, _), (resampled_trace, log_normalizing_constants) = (
          fun_mc.trace(
              (
                  smc.sequential_monte_carlo_init(
                      state=jnp.zeros((num_particles,), dtype=self._dtype),
                      weight_dtype=self._dtype,
                  ),
                  seed,
              ),
              kernel,  # pylint: disable=cell-var-from-loop
              max_num_timesteps,
              stop_fn=stop_fn if stop_early else None,  # pylint: disable=cell-var-from-loop
          )
      )
      if stop_early:  # pylint: disable=cell-var-from-loop
        log_normalizing_constant = smc_state.log_normalizing_constant()
      else:
        log_normalizing_constant = log_normalizing_constants[num_timesteps - 1]
      return log_normalizing_constant, resampled_trace

    log_normaling_constant, resampled_trace = run_smc(_test_seed())

    self.assertAllClose(log_normaling_constant, 2.0 * num_timesteps)
    # Resampling never happens on the last step.
    self.assertAllTrue(~resampled_trace[num_timesteps - 1])
    # Resampling happens on all but the last step.
    self.assertAllTrue(jnp.all(resampled_trace[: num_timesteps - 1]))

  @parameterized.product(
      num_timesteps=[0, 1, 9, 10, 11, 100],
      max_num_timesteps=[1, 10],
  )
  def test_final_timestep(self, num_timesteps, max_num_timesteps):
    """Test returning the index of the first is_finished occurrence."""
    num_particles = 3

    def kernel(smc_state, seed):
      step_seed, seed = util.split_seed(seed, 2)
      smc_state, extra = smc.sequential_monte_carlo_step(
          smc_state,
          kernel=functools.partial(
              kernel_extra_is_finished, num_timesteps=num_timesteps
          ),
          seed=step_seed,
      )
      is_finished = extra.kernel_extra
      return (smc_state, seed), is_finished

    @jax.jit
    def run_smc(seed):
      _, is_finished_trace = fun_mc.trace(
          (
              smc.sequential_monte_carlo_init(
                  state=jnp.zeros((num_particles,), dtype=self._dtype),
                  weight_dtype=self._dtype,
              ),
              seed,
          ),
          kernel,
          max_num_timesteps,
      )

      # Find the index of the first occurence of `is_finished==True`. If all
      # are `False`, return `max_num_timesteps - 1`.
      final_timestep = jnp.where(
          jnp.all(~is_finished_trace),
          max_num_timesteps - 1,
          jnp.argmax(is_finished_trace),
      )
      return final_timestep

    final_timestep = run_smc(_test_seed())

    if num_timesteps >= max_num_timesteps:
      self.assertEqual(final_timestep, max_num_timesteps - 1)
    else:
      self.assertEqual(final_timestep, max(num_timesteps - 1, 0))

  @parameterized.parameters(True, False)
  def test_some_neg_inf_weights(self, stop_early):
    """Test that SMC correctly handles some weights of negative infinity."""
    num_particles = 4
    num_timesteps = 3
    max_num_timesteps = 3

    def kernel(smc_state, seed):
      step_seed, seed = util.split_seed(seed, 2)
      smc_state, extra = smc.sequential_monte_carlo_step(
          smc_state,
          kernel=kernel_log_weights_eq_neg_inf_if_state_lt_zero,
          resampling_pred=always_predicate,
          seed=step_seed,
      )
      return (smc_state, seed), {
          'state': smc_state.state,
          'log_weights': smc_state.log_weights,
          'ancestor_idxs': extra.ancestor_idxs,
          'resampled': extra.resampled,
          'log_normalizing_constant': smc_state.log_normalizing_constant(),
      }

    @jax.jit
    def run_smc(seed):
      def stop_fn(smc_state_and_seed, _):
        smc_state, _ = smc_state_and_seed
        return smc_state.step >= num_timesteps

      _, trace = fun_mc.trace(
          (
              smc.sequential_monte_carlo_init(
                  state=jnp.arange(num_particles, dtype=self._dtype),
                  weight_dtype=self._dtype,
              ),
              seed,
          ),
          kernel,
          max_num_timesteps,
          stop_fn=stop_fn if stop_early else None,
      )
      return trace

    trace = run_smc(_test_seed())

    # assert that the weights of the non-negative particles are finite
    self.assertTrue(
        jnp.all(
            jnp.equal(
                jnp.isfinite(trace['log_weights'][:num_timesteps]),
                jnp.greater_equal(trace['state'][:num_timesteps], 0.0),
            )
        )
    )

    # Assert that the weights of the negative particles are -inf
    self.assertTrue(
        jnp.all(
            jnp.equal(
                jnp.isneginf(trace['log_weights'][:num_timesteps]),
                jnp.less(trace['state'][:num_timesteps], 0.0),
            )
        )
    )

    # Assert that the weights don't all become negative inf at once.
    for i in range(num_timesteps):
      self.assertTrue(
          jnp.any(jnp.isfinite(trace['log_weights'][i])), msg=f'step={i}'
      )

    # Assert that -inf weights are never selected for resampling
    for i in range(1, num_timesteps):
      ancestor_inds = trace['ancestor_idxs'][i]
      ancestor_weights = trace['log_weights'][i - 1][ancestor_inds]
      self.assertTrue(
          jnp.all(ancestor_weights > -float('inf')), msg=f'step={i}'
      )

    # Assert that the log normalizing constant is finite
    self.assertTrue(
        jnp.isfinite(trace['log_normalizing_constant'][num_timesteps - 1])
    )

  @parameterized.parameters(True, False)
  def test_all_neg_inf_weights(self, stop_early):
    """Test handling of weights *all* becoming negative infinity."""
    num_particles = 4
    num_timesteps = 5
    max_num_timesteps = 7

    def kernel(smc_state, seed):
      step_seed, seed = util.split_seed(seed, 2)
      smc_state, extra = smc.sequential_monte_carlo_step(
          smc_state,
          kernel=kernel_log_weights_eq_neg_inf_if_state_lt_zero,
          resampling_pred=always_predicate,
          seed=step_seed,
      )
      return (smc_state, seed), {
          'state': smc_state.state,
          'log_weights': smc_state.log_weights,
          'ancestor_idxs': extra.ancestor_idxs,
          'resampled': extra.resampled,
          'log_normalizing_constant': smc_state.log_normalizing_constant(),
      }

    @jax.jit
    def run_smc(seed):
      def stop_fn(smc_state_and_seed, _):
        smc_state, _ = smc_state_and_seed
        return smc_state.step >= num_timesteps

      _, trace = fun_mc.trace(
          (
              smc.sequential_monte_carlo_init(
                  state=jnp.arange(num_particles, dtype=self._dtype),
                  weight_dtype=self._dtype,
              ),
              seed,
          ),
          kernel,
          max_num_timesteps,
          stop_fn=stop_fn if stop_early else None,
      )
      return trace

    trace = run_smc(_test_seed())

    # Assert that the weights of the non-negative states are finite
    self.assertTrue(
        jnp.all(
            jnp.equal(
                jnp.isfinite(trace['log_weights'][:num_timesteps]),
                jnp.greater_equal(trace['state'][:num_timesteps], 0.0),
            )
        )
    )

    # Assert that the weights of the negative states are -inf
    self.assertTrue(
        jnp.all(
            jnp.equal(
                jnp.isneginf(trace['log_weights'][:num_timesteps]),
                jnp.less(trace['state'][:num_timesteps], 0.0),
            )
        )
    )

    # Assert that the weights eventually all become negative inf at once.
    self.assertTrue(
        jnp.any(
            jnp.logical_not(
                jnp.any(
                    jnp.isfinite(trace['log_weights'][:num_timesteps]), axis=1
                )
            )
        )
    )

    # Assert that -inf weights are never selected for resampling, if there's a
    # choice not to pick them.
    for i in range(1, num_timesteps):
      ancestor_inds = trace['ancestor_idxs'][i]
      if not jnp.any(trace['log_weights'][i - 1] > -float('inf')):
        continue
      ancestor_weights = trace['log_weights'][i - 1][ancestor_inds]
      self.assertTrue(
          jnp.all(ancestor_weights > -float('inf')), msg=f'step={i}'
      )

    # Assert that the bound is negative inf
    self.assertTrue(
        jnp.isneginf(trace['log_normalizing_constant'][num_timesteps - 1])
    )

  @parameterized.parameters(True, False)
  def test_some_neg_inf_weights_no_resampling(self, stop_early):
    """Test handling some negative inf weights without resampling."""
    num_particles = 4
    num_timesteps = 3
    max_num_timesteps = 3

    def kernel(smc_state, seed):
      step_seed, seed = util.split_seed(seed, 2)
      smc_state, _ = smc.sequential_monte_carlo_step(
          smc_state,
          kernel=kernel_log_weights_eq_neg_inf_if_state_lt_zero,
          resampling_pred=never_predicate,
          seed=step_seed,
      )
      return (smc_state, seed), {
          'state': smc_state.state,
          'log_weights': smc_state.log_weights,
          'log_normalizing_constant': smc_state.log_normalizing_constant(),
      }

    @jax.jit
    def run_smc(seed):
      def stop_fn(smc_state_and_seed, _):
        smc_state, _ = smc_state_and_seed
        return smc_state.step >= num_timesteps

      _, trace = fun_mc.trace(
          (
              smc.sequential_monte_carlo_init(
                  state=jnp.arange(num_particles, dtype=self._dtype),
                  weight_dtype=self._dtype,
              ),
              seed,
          ),
          kernel,
          max_num_timesteps,
          stop_fn=stop_fn if stop_early else None,
      )
      return trace

    trace = run_smc(_test_seed())

    # Assert that the weights of the non-negative states are finite
    self.assertTrue(
        jnp.all(
            jnp.equal(
                jnp.isfinite(trace['log_weights'][:num_timesteps]),
                jnp.greater_equal(trace['state'][:num_timesteps], 0.0),
            )
        )
    )

    # Assert that the weights of the negative states are -inf
    self.assertTrue(
        jnp.all(
            jnp.equal(
                jnp.isneginf(trace['log_weights'][:num_timesteps]),
                jnp.less(trace['state'][:num_timesteps], 0.0),
            )
        )
    )

    # Assert that the weights don't all become negative inf at once.
    for i in range(num_timesteps):
      self.assertTrue(
          jnp.any(jnp.isfinite(trace['log_weights'][i])), msg=f'step={i}'
      )

    # Assert that the log normalizing constant is finite
    self.assertTrue(
        jnp.isfinite(trace['log_normalizing_constant'][num_timesteps - 1])
    )

  @parameterized.parameters(True, False)
  def test_all_neg_inf_weights_no_resampling(self, stop_early):
    """Test handling *all* negative inf weights without resampling."""
    num_particles = 4
    num_timesteps = 5
    max_num_timesteps = 5

    def kernel(smc_state, seed):
      step_seed, seed = util.split_seed(seed, 2)
      smc_state, _ = smc.sequential_monte_carlo_step(
          smc_state,
          kernel=kernel_log_weights_eq_neg_inf_if_state_lt_zero,
          resampling_pred=never_predicate,
          seed=step_seed,
      )
      return (smc_state, seed), {
          'state': smc_state.state,
          'log_weights': smc_state.log_weights,
          'log_normalizing_constant': smc_state.log_normalizing_constant(),
      }

    @jax.jit
    def run_smc(seed):
      def stop_fn(smc_state_and_seed, _):
        smc_state, _ = smc_state_and_seed
        return smc_state.step >= num_timesteps

      _, trace = fun_mc.trace(
          (
              smc.sequential_monte_carlo_init(
                  state=jnp.arange(num_particles, dtype=self._dtype),
                  weight_dtype=self._dtype,
              ),
              seed,
          ),
          kernel,
          max_num_timesteps,
          stop_fn=stop_fn if stop_early else None,
      )
      return trace

    trace = run_smc(_test_seed())

    # Assert that the weights of the non-negative states are finite
    self.assertTrue(
        jnp.all(
            jnp.equal(
                jnp.isfinite(trace['log_weights'][:num_timesteps]),
                jnp.greater_equal(trace['state'][:num_timesteps], 0.0),
            )
        )
    )

    # Assert that the weights of the negative states are -inf
    self.assertTrue(
        jnp.all(
            jnp.equal(
                jnp.isneginf(trace['log_weights'][:num_timesteps]),
                jnp.less(trace['state'][:num_timesteps], 0.0),
            )
        )
    )

    # Assert that the weights eventually all become negative inf at once.
    self.assertTrue(
        jnp.any(
            jnp.logical_not(
                jnp.any(
                    jnp.isfinite(trace['log_weights'][:num_timesteps]), axis=1
                )
            )
        )
    )

    # Assert that the bound is negative inf
    self.assertTrue(
        jnp.isneginf(trace['log_normalizing_constant'][num_timesteps - 1])
    )

  @parameterized.product(
      resampling_pred=[
          always_predicate,
          never_predicate,
          smc.effective_sample_size_predicate,
      ],
      num_timesteps=[1, 5],
  )
  def test_normalizing_const_gaussian(self, resampling_pred, num_timesteps):
    """Check that the normalizing constant estimate of SMC is correct.

    This test chooses the unnormalized targets, s_t, as

    s_t(x) = prod_{r=1}^t exp(-(x_r^2)/2)

    i.e. the unnormalized targets are a sequence of independent Gaussian
    potentials with mean 0 and variance 1. Specifically, each potential does not
    include the 1/sqrt(2*pi) term that would be required to normalize a standard
    Gaussian. Those terms end up being collected into the computed normalizing
    constant, i.e. Z should be

    (2 * pi)^(t/2)

    for each timestep t.

    The chosen sequence of unnormalized targets implies that the incremental
    weight should be

      s_t(x_{1:t}) / (s_{t-1}(x_{1:t-1}) * q_t(x_t))
      = exp(-(x_t^2)/2) / q_t(x_t).

    In log space this is

      - x_t^2 / 2  - log q_t(x_t).

    Args:
      resampling_pred: Resampling predicate.
      num_timesteps: The number of steps to run SMC for.
    """
    num_particles = 2_000 * num_timesteps
    q_std = 1.25

    def smc_kernel(state, step, seed):
      del state, step
      q_dist = tfd.Normal(
          jnp.zeros([num_particles], self._dtype),
          jnp.full([num_particles], q_std, dtype=self._dtype),
      )
      new_x = q_dist.sample(seed=seed)
      log_q = q_dist.log_prob(new_x)
      log_p = -jnp.square(new_x) / 2
      return new_x, (log_p - log_q, ())

    def kernel(smc_state, seed):
      step_seed, seed = util.split_seed(seed, 2)
      smc_state, _ = smc.sequential_monte_carlo_step(
          smc_state,
          kernel=smc_kernel,
          resampling_pred=resampling_pred,
          seed=step_seed,
      )
      return (smc_state, seed), ()

    @jax.jit
    def run_smc(seed):
      (smc_state, _), _ = fun_mc.trace(
          (
              smc.sequential_monte_carlo_init(
                  state=jnp.zeros((num_particles,), self._dtype),
                  weight_dtype=self._dtype,
              ),
              seed,
          ),
          kernel,
          num_timesteps,
      )
      return smc_state.log_normalizing_constant()

    log_normalizing_constant = run_smc(_test_seed())
    log_normalizing_constant_expected = (num_timesteps / 2.0) * (
        jnp.log(2) + jnp.log(jnp.pi)
    )
    self.assertAllClose(
        log_normalizing_constant, log_normalizing_constant_expected, atol=1e-2
    )

  def test_lgssm(self):
    """Check filtering distributions and the normalizing constant of a linear Gaussian state-space model."""
    num_timesteps = 10
    init_loc = 0.0
    init_scale = 1.0
    transition_mult = 1.0
    transition_add = 0.0
    transition_scale = 1.0
    obs_mult = 1.0
    obs_add = 0.0
    obs_scale = 1.0

    obs_seed, smc_seed = util.split_seed(_test_seed(), 2)
    obs = util.random_normal((num_timesteps,), self._dtype, obs_seed)

    def smc_kernel(state, step, seed):
      # Sample from the initial distribution (when prev_timestep = 0)
      init = tfd.Normal(
          jnp.full_like(state, init_loc, dtype=self._dtype),
          jnp.full_like(state, init_scale, dtype=self._dtype),
      ).sample(seed=seed)

      # Sample from the transition distribution (when step > 0)
      not_init = tfd.Normal(
          state * transition_mult + transition_add,
          transition_scale,
      ).sample(seed=seed)
      new_particles = jnp.where(step == 0, init, not_init)
      incremental_log_weights = tfd.Normal(
          new_particles * obs_mult + obs_add, obs_scale
      ).log_prob(obs[step])
      return new_particles, (incremental_log_weights, ())

    @jax.jit
    def particle_expectation(state, log_weights):
      # Assumes state's shape is [num_particles, ...]
      return jnp.einsum('i...,i->...', state, jax.nn.softmax(log_weights))

    def kernel(smc_state, seed):
      step_seed, seed = util.split_seed(seed, 2)
      smc_state, _ = smc.sequential_monte_carlo_step(
          smc_state,
          kernel=smc_kernel,
          resampling_pred=smc.effective_sample_size_predicate,
          seed=step_seed,
      )
      state = smc_state.state
      log_weights = smc_state.log_weights
      filtering_mean = particle_expectation(state, log_weights)
      filtering_std = jnp.sqrt(
          particle_expectation(state**2, log_weights) - filtering_mean**2
      )

      return (smc_state, seed), (filtering_mean, filtering_std)

    # SMC bootstrap filtering posterior
    num_particles = 500

    (smc_state, _), (filtering_means, filtering_stds) = fun_mc.trace(
        (
            smc.sequential_monte_carlo_init(
                state=jnp.full(
                    [num_particles], float('NaN'), dtype=self._dtype
                ),
                weight_dtype=self._dtype,
            ),
            smc_seed,
        ),
        kernel,
        num_timesteps,
    )

    # Ground truth filtering posterior
    tfp_lgssm = tfd.LinearGaussianStateSpaceModel(
        num_timesteps=num_timesteps,
        transition_matrix=jnp.array([[transition_mult]]),
        transition_noise=tfd.MultivariateNormalDiag(
            jnp.array([transition_add]),
            scale_diag=jnp.array([transition_scale]),
        ),
        observation_matrix=jnp.array([[obs_mult]]),
        observation_noise=tfd.MultivariateNormalDiag(
            jnp.array([obs_add]), jnp.array([obs_scale])
        ),
        initial_state_prior=tfd.MultivariateNormalDiag(
            jnp.array([init_loc]), jnp.array([init_scale])
        ),
    )

    gt_filter_results = tfp_lgssm.forward_filter(obs[..., jnp.newaxis])
    gt_filtering_means = gt_filter_results.filtered_means[:, 0]
    gt_filtering_stds = jnp.sqrt(gt_filter_results.filtered_covs[:, 0, 0])

    # SMC log marginal likelihood
    log_evidence = smc_state.log_normalizing_constant()

    # Ground truth log evidence
    gt_log_evidence = jnp.sum(gt_filter_results.log_likelihoods)

    self.assertAllClose(gt_filtering_means, filtering_means, atol=0.2)
    self.assertAllClose(gt_filtering_stds, filtering_stds, atol=0.2)
    self.assertAllClose(gt_log_evidence, log_evidence, rtol=0.01)
    self.assertAllClose(gt_log_evidence, log_evidence, atol=0.2)


@test_util.multi_backend_test(globals(), 'smc_test')
class SMCTest32(SMCTest):

  @property
  def _dtype(self):
    return jnp.float32


@test_util.multi_backend_test(globals(), 'smc_test')
class SMCTest64(SMCTest):

  @property
  def _dtype(self):
    return jnp.float64


del SMCTest

if __name__ == '__main__':
  tfp_test_util.main()
