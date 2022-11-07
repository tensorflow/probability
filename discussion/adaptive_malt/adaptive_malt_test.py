# Copyright 2022 The TensorFlow Probability Authors.
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
"""Tests for adaptive_malt."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
from discussion.adaptive_malt import adaptive_malt


class AdaptiveMALTTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('_malt', 'malt'),
      ('_hmc', 'hmc'),
  )
  def test_adaptive_smoke(self, method):
    """Smoke test."""
    target = adaptive_malt.TestGaussian(ndims=5)
    _ = jax.jit(lambda seed: adaptive_malt.run_adaptive_mcmc_on_target(  # pylint: disable=g-long-lambda
        target=target,
        method=method,
        num_chains=1,
        init_step_size=1e-2,
        num_adaptation_steps=2,
        num_results=8,
        seed=seed))(
            jax.random.PRNGKey(0))

  @parameterized.named_parameters(
      ('_malt', 'malt'),
      ('_hmc', 'hmc'),
  )
  def test_fixed_smoke(self, method):
    """Smoke test."""
    target = adaptive_malt.TestGaussian(ndims=5)
    _ = jax.jit(lambda seed: adaptive_malt.run_fixed_mcmc_on_target(  # pylint: disable=g-long-lambda
        target=target,
        method=method,
        init_x=target.sample((1,), seed=jax.random.PRNGKey(1)),
        scalar_step_size=1e-2,
        vector_step_size=jnp.ones(5),
        damping=0.5,
        num_warmup_steps=2,
        num_results=8,
        num_integrator_steps=1,
        seed=seed))(
            jax.random.PRNGKey(0))


if __name__ == '__main__':
  absltest.main()
