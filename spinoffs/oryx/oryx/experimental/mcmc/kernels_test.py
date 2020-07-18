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
# Lint as: python3
"""Tests for tensorflow_probability.spinoffs.oryx.experimental.mcmc.kernels."""
from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import random
import jax.numpy as np
import numpy as onp

from oryx.core.interpreters import harvest
from oryx.experimental.mcmc import kernels
from oryx.experimental.mcmc import utils
from tensorflow_probability.substrates import jax as tfp


tf = tfp.tf2jax
inference_gym = tfp.experimental.inference_gym

MAKE_KERNELS = [
    ('metropolis',
     lambda log_prob: kernels.metropolis(log_prob, kernels.random_walk(1.)),
     0.64),
    ('metropolis_hastings',
     lambda log_prob: kernels.metropolis_hastings(  # pylint: disable=g-long-lambda
         log_prob, kernels.random_walk(1.)), 0.64),
    ('mala', lambda log_prob: kernels.mala(log_prob, step_size=1.), 0.93),
    ('hmc', lambda log_prob: kernels.hmc(log_prob, step_size=1.), 0.93),
]


class KernelsTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.model = inference_gym.targets.IllConditionedGaussian(ndims=2)
    self._seed = random.PRNGKey(0)

  def _make_unconstrained_log_prob(self):
    def _constrain(state):
      return tf.nest.map_structure(self.model.default_event_space_bijector,
                                   state)

    return utils.constrain(_constrain)(self.model.unnormalized_log_prob)

  def _initialize_state(self, key):
    num_events = len(tf.nest.flatten(self.model.event_shape))
    init_keys = tf.nest.pack_sequence_as(
        self.model.event_shape, list(random.split(key, num_events)))

    def _init(key, shape):
      return random.normal(key, shape)

    initial_state = tf.nest.map_structure(_init, init_keys,
                                          self.model.event_shape)
    return initial_state

  @parameterized.named_parameters(MAKE_KERNELS)
  def test_single_chain(self, make_kernel, target_accept_rate):
    num_samples = 20000

    sample_key, chain_key, init_key = random.split(self._seed, 3)
    unconstrained_log_prob = self._make_unconstrained_log_prob()
    initial_state = self._initialize_state(init_key)
    kernel = make_kernel(unconstrained_log_prob)
    sample_chain = jax.jit(
        harvest.harvest(
            kernels.sample_chain(kernel, num_samples),
            tag=kernels.MCMC_METRICS))

    true_samples = self.model.sample(sample_shape=4096, seed=sample_key)
    samples, metrics = sample_chain({}, chain_key, initial_state)

    onp.testing.assert_allclose(
        true_samples.mean(axis=0), samples.mean(axis=0), rtol=0.5, atol=0.1)
    onp.testing.assert_allclose(
        np.cov(true_samples.T), np.cov(samples.T), rtol=0.5, atol=0.1)
    onp.testing.assert_allclose(target_accept_rate,
                                metrics['kernel']['accept_prob'].mean(),
                                atol=1e-2, rtol=1e-2)

  @parameterized.named_parameters(MAKE_KERNELS)
  def test_multiple_chains(self, make_kernel, target_accept_rate):
    num_chains = 16
    num_samples = 4000

    sample_key, chain_key, init_key = random.split(self._seed, 3)
    unconstrained_log_prob = self._make_unconstrained_log_prob()
    initial_states = jax.vmap(self._initialize_state)(
        random.split(init_key, num_chains))

    kernel = make_kernel(unconstrained_log_prob)
    sample_chain = jax.jit(
        jax.vmap(harvest.harvest(
            kernels.sample_chain(kernel, num_samples),
            tag=kernels.MCMC_METRICS)))

    true_samples = self.model.sample(sample_shape=4096, seed=sample_key)
    samples, metrics = sample_chain({}, random.split(chain_key, num_chains),
                                    initial_states)
    samples = tf.nest.map_structure(
        lambda s, shape: s.reshape([num_chains * num_samples] + list(shape)),
        samples, self.model.event_shape)

    onp.testing.assert_allclose(
        true_samples.mean(axis=0), samples.mean(axis=0), rtol=0.1, atol=0.1)
    onp.testing.assert_allclose(
        np.cov(true_samples.T), np.cov(samples.T), rtol=0.1, atol=0.1)
    onp.testing.assert_allclose(target_accept_rate,
                                metrics['kernel']['accept_prob'].mean(),
                                atol=1e-2, rtol=1e-2)


if __name__ == '__main__':
  absltest.main()
