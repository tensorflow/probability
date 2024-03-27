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

from flax import linen as nn
import jax
import jax.numpy as jnp
from tensorflow_probability.spinoffs.autobnn import bnn
from tensorflow_probability.substrates.jax.distributions import lognormal as lognormal_lib
from tensorflow_probability.substrates.jax.distributions import normal as normal_lib
from absl.testing import absltest


class MyBNN(bnn.BNN):

  def distributions(self):
    return super().distributions() | {
        'dense': {
            'kernel': normal_lib.Normal(loc=0, scale=1),
            'bias': normal_lib.Normal(loc=0, scale=1),
        },
        'amplitude': lognormal_lib.LogNormal(loc=0, scale=1),
    }

  def setup(self):
    self.dense = nn.Dense(50)
    super().setup()

  def __call__(self, inputs):
    return self.amplitude * jnp.sum(self.dense(inputs))


class BnnTests(absltest.TestCase):

  def test_mybnn(self):
    my_bnn = MyBNN()
    d = my_bnn.distributions()
    self.assertIn('noise_scale', d)
    sample_noise = d['noise_scale'].sample(1, seed=jax.random.PRNGKey(0))
    self.assertEqual((1,), sample_noise.shape)

    params = my_bnn.init(jax.random.PRNGKey(0), jnp.zeros(1))
    lp1 = my_bnn.log_prior(params)
    params['params']['amplitude'] += 50
    lp2 = my_bnn.log_prior(params)
    self.assertLess(lp2, lp1)

    data = jnp.array([[0], [1], [2], [3], [4], [5]], dtype=jnp.float32)
    obs = jnp.array([1, 0, 1, 0, 1, 0], dtype=jnp.float32)
    ll = my_bnn.log_likelihood(params, data, obs)
    lp = my_bnn.log_prob(params, data, obs)
    self.assertLess(jnp.sum(lp), jnp.sum(ll))


if __name__ == '__main__':
  absltest.main()
