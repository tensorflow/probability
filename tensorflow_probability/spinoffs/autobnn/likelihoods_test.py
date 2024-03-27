# Copyright 2023 The TensorFlow Probability Authors.
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
"""Tests for bnn.py."""

from absl.testing import parameterized
import jax.numpy as jnp
from tensorflow_probability.spinoffs.autobnn import likelihoods
from absl.testing import absltest


class LikelihoodTests(parameterized.TestCase):

  @parameterized.parameters(
      likelihoods.NormalLikelihoodLogisticNoise(),
      likelihoods.NormalLikelihoodLogNormalNoise(),
      likelihoods.NormalLikelihoodVaryingNoise(),
      likelihoods.NegativeBinomial(),
      likelihoods.ZeroInflatedNegativeBinomial(),
  )
  def test_likelihoods(self, likelihood_model):
    lp = likelihood_model.log_likelihood(
        params={'noise_scale': 0.4},
        nn_out=jnp.ones(shape=(10, likelihood_model.num_outputs())),
        observations=jnp.zeros(shape=(10, 1)),
    )
    self.assertEqual(lp.shape, (10, 1))

  @parameterized.parameters(list(likelihoods.NAME_TO_LIKELIHOOD_MODEL.keys()))
  def test_get_likelihood_model(self, likelihood_model):
    m = likelihoods.get_likelihood_model(likelihood_model, {})
    lp = m.log_likelihood(
        params={'noise_scale': 0.4},
        nn_out=jnp.ones(shape=(10, m.num_outputs())),
        observations=jnp.zeros(shape=(10, 1)),
    )
    self.assertEqual(lp.shape, (10, 1))

    m2 = likelihoods.get_likelihood_model(
        likelihood_model,
        {'noise_min': 0.1, 'log_noise_scale': 0.5, 'log_noise_mean': -1.0},
    )
    lp2 = m2.log_likelihood(
        params={'noise_scale': 0.4},
        nn_out=jnp.ones(shape=(10, m2.num_outputs())),
        observations=jnp.zeros(shape=(10, 1)),
    )
    self.assertEqual(lp2.shape, (10, 1))


if __name__ == '__main__':
  absltest.main()
