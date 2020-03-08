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
"""Tests TFP distribution compositionality with JAX transformations."""
from absl.testing import parameterized
import hypothesis as hp
from hypothesis import strategies as hps
import jax
from jax import random

# pylint: disable=no-name-in-module

from tensorflow_probability.python.distributions._jax import hypothesis_testlib as dhps
from tensorflow_probability.python.experimental.substrates.jax import tf2jax as tf
from tensorflow_probability.python.internal._jax import hypothesis_testlib as tfp_hps
from tensorflow_probability.python.internal._jax import test_util

JIT_SAMPLE_BLACKLIST = set()
JIT_LOGPROB_BLACKLIST = set()

VMAP_SAMPLE_BLACKLIST = set()
VMAP_LOGPROB_BLACKLIST = set()

test_all_distributions = parameterized.named_parameters(
    {'testcase_name': dname, 'dist_name': dname} for dname in
    sorted(list(dhps.INSTANTIABLE_BASE_DISTS.keys())))


class JitTest(test_util.TestCase):

  @test_all_distributions
  @hp.given(hps.data())
  @tfp_hps.tfp_hp_settings()
  def testSample(self, dist_name, data):
    if dist_name in JIT_SAMPLE_BLACKLIST:
      self.skipTest('Distribution currently broken.')
    dist = data.draw(dhps.distributions(enable_vars=False,
                                        dist_name=dist_name))
    def _sample(seed):
      return dist.sample(seed=seed)
    seed = test_util.test_seed()
    self.assertAllClose(_sample(seed), jax.jit(_sample)(seed), rtol=1e-6,
                        atol=1e-6)

  @test_all_distributions
  @hp.given(hps.data())
  @tfp_hps.tfp_hp_settings()
  def testLogProb(self, dist_name, data):
    if dist_name in JIT_LOGPROB_BLACKLIST:
      self.skipTest('Distribution currently broken.')
    dist = data.draw(dhps.distributions(enable_vars=False,
                                        dist_name=dist_name))
    sample = dist.sample(seed=test_util.test_seed())
    self.assertAllClose(dist.log_prob(sample), jax.jit(dist.log_prob)(sample),
                        rtol=1e-6, atol=1e-6)


class VmapTest(test_util.TestCase):

  @test_all_distributions
  @hp.given(hps.data())
  @tfp_hps.tfp_hp_settings()
  def testSample(self, dist_name, data):
    if dist_name in VMAP_SAMPLE_BLACKLIST:
      self.skipTest('Distribution currently broken.')
    dist = data.draw(dhps.distributions(enable_vars=False,
                                        dist_name=dist_name))
    def _sample(seed):
      return dist.sample(seed=seed)
    seed = test_util.test_seed()
    jax.vmap(_sample)(random.split(seed, 10))

  @test_all_distributions
  @hp.given(hps.data())
  @tfp_hps.tfp_hp_settings()
  def testLogProb(self, dist_name, data):
    if dist_name in VMAP_LOGPROB_BLACKLIST:
      self.skipTest('Distribution currently broken.')
    dist = data.draw(dhps.distributions(enable_vars=False,
                                        dist_name=dist_name))
    sample = dist.sample(seed=test_util.test_seed(), sample_shape=10)
    self.assertAllClose(jax.vmap(dist.log_prob)(sample), dist.log_prob(sample),
                        rtol=1e-6, atol=1e-6)


if __name__ == '__main__':
  tf.test.main()
