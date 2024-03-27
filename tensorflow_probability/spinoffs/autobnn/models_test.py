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
"""Tests for models.py."""

from absl.testing import parameterized
import jax
import jax.numpy as jnp
from tensorflow_probability.spinoffs.autobnn import likelihoods
from tensorflow_probability.spinoffs.autobnn import models
from absl.testing import absltest


MODELS = list(models.MODEL_NAME_TO_MAKE_FUNCTION.keys())


class ModelsTest(parameterized.TestCase):

  @parameterized.parameters(MODELS)
  def test_make_model(self, model_name):
    m = models.make_model(
        model_name,
        likelihoods.NormalLikelihoodLogisticNoise(),
        time_series_xs=jnp.linspace(0.0, 1.0, 50),
        width=5,
        periods=[0.2],
    )
    params = m.init(jax.random.PRNGKey(0), jnp.zeros(5))
    lp = m.log_prior(params)
    self.assertTrue((lp < 0.0) or (lp > 0.0))

  @parameterized.product(
      model_name=MODELS,
      # It takes too long to test all of the likelihoods, so just test a
      # couple to make sure each model correctly handles num_outputs > 1.
      likelihood_name=[
          'normal_likelihood_varying_noise',
          'zero_inflated_negative_binomial',
      ],
  )
  def test_make_model_and_likelihood(self, model_name, likelihood_name):
    ll = likelihoods.get_likelihood_model(likelihood_name, {})
    m = models.make_model(
        model_name,
        ll,
        time_series_xs=jnp.linspace(0.0, 1.0, 50),
        width=5,
        periods=[0.2],
    )
    params = m.init(jax.random.PRNGKey(0), jnp.zeros(5))
    lp = m.log_prior(params)
    self.assertTrue((lp < 0.0) or (lp > 0.0))


if __name__ == '__main__':
  absltest.main()
