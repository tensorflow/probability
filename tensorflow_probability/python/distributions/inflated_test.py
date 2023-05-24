# Copyright 2022 The TensorFlow Probability Authors. All Rights Reserved.
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
# ==============================================================================

import numpy as np
import tensorflow.compat.v2 as tf

from tensorflow_probability.python.distributions import gamma
from tensorflow_probability.python.distributions import inflated
from tensorflow_probability.python.distributions import lognormal
from tensorflow_probability.python.distributions import negative_binomial
from tensorflow_probability.python.distributions import normal
from tensorflow_probability.python.experimental.util import trainable
from tensorflow_probability.python.internal import test_util
from tensorflow_probability.python.math import gradient


class DistributionsTest(test_util.TestCase):

  def test_inflated(self):
    zinb = inflated.Inflated(
        negative_binomial.NegativeBinomial(5.0, probs=0.1),
        inflated_loc_probs=0.2)
    samples = zinb.sample(sample_shape=10, seed=test_util.test_seed())
    self.assertEqual((10,), samples.shape)

    spike_and_slab = inflated.Inflated(
        normal.Normal(loc=1.0, scale=2.0), inflated_loc_probs=0.5)
    lprob = self.evaluate(spike_and_slab.log_prob(99.0))
    self.assertLess(lprob, 0.0)

    param_props = inflated.Inflated.parameter_properties(dtype=tf.float32)
    self.assertFalse(param_props['distribution'].is_tensor)
    self.assertTrue(param_props['inflated_loc_logits'].is_preferred)
    self.assertFalse(param_props['inflated_loc_probs'].is_preferred)
    self.assertTrue(param_props['inflated_loc'].is_tensor)

  def test_inflated_batched(self):
    nb = negative_binomial.NegativeBinomial(
        total_count=np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32),
        logits=np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32))
    zinb = inflated.Inflated(
        nb,
        inflated_loc_probs=np.array([0.2, 0.4, 0.6, 0.8, 1.0],
                                    dtype=np.float32))

    lprob = zinb.log_prob([0, 1, 2, 3, 4])
    self.assertEqual((5,), lprob.shape)

    samples = zinb.sample(seed=test_util.test_seed())
    self.assertEqual((5,), samples.shape)

  def test_inflated_continuous_log_prob(self):
    spike_and_slab = inflated.Inflated(
        normal.Normal(loc=1.0, scale=2.0), inflated_loc_probs=0.1)
    self.assertEqual(self.evaluate(tf.math.log(0.1)),
                     self.evaluate(spike_and_slab.log_prob(0.0)))
    self.assertNear(
        self.evaluate(tf.math.log(0.9) + normal.Normal(
            loc=1.0, scale=2.0).log_prob(2.0)),
        self.evaluate(spike_and_slab.log_prob(2.0)),
        1e-6
    )

  def test_inflated_factory(self):
    spike_and_slab_class = inflated.inflated_factory('SpikeAndSlab',
                                                     normal.Normal, 0.0)
    spike_and_slab = spike_and_slab_class(
        inflated_loc_probs=0.3, loc=5.0, scale=2.0)
    spike_and_slab2 = inflated.Inflated(
        normal.Normal(loc=5.0, scale=2.0), inflated_loc_probs=0.3)
    self.assertEqual(
        self.evaluate(spike_and_slab.log_prob(7.0)),
        self.evaluate(spike_and_slab2.log_prob(7.0)))

  def test_zero_inflated_negative_binomial(self):
    zinb = inflated.ZeroInflatedNegativeBinomial(
        inflated_loc_probs=0.2, probs=0.5, total_count=10.0)
    self.assertEqual('ZeroInflatedNegativeBinomial', zinb.name)

  def test_zinb_is_trainable(self):
    init_fn, apply_fn = trainable.make_trainable_stateless(
        inflated.ZeroInflatedNegativeBinomial,
        batch_and_event_shape=[5],
        parameter_dtype=tf.float32)
    init_obj = init_fn(seed=test_util.test_seed())
    # ZeroInflatedNegativeBinomial should have three parameters per scalar
    # distribution:  two from NegativeBinomial, and one for the mixture weight.
    self.assertEqual(3, len(init_obj))
    init_dist = apply_fn(init_obj)

    lprob = init_dist.log_prob(np.array([0, 1, 2, 3, 4], dtype=np.float32))
    self.assertEqual((5,), lprob.shape)

    samples = init_dist.sample(seed=test_util.test_seed())
    self.assertEqual((5,), samples.shape)

  @test_util.disable_test_for_backend(
      disable_numpy=True,
      reason='Only TF has gradient tape')
  def test_safe_value_for_distribution(self):
    x = self.evaluate(inflated._safe_value_for_distribution(
        gamma.Gamma(concentration=3.0, rate=2.0)))
    lp, grad = gradient.value_and_gradient(
        lambda p: gamma.Gamma(concentration=p, rate=2.0).log_prob(x),
        3.0)
    self.assertAllFinite(lp)
    self.assertAllFinite(grad)

  @test_util.disable_test_for_backend(
      disable_numpy=True,
      reason='Only TF has gradient tape')
  def test_log_prob_for_inflated_lognormal_is_diffable(self):
    x = tf.constant([0.0, 1.0])

    # pylint: disable=g-long-lambda
    lp, grad = gradient.value_and_gradient(
        lambda loc: inflated.Inflated(
            lognormal.LogNormal(loc=loc, scale=1.0),
            inflated_loc_probs=0.5,
        ).log_prob(x),
        5.0,
    )
    # pylint: enable=g-long-lambda
    self.assertAllFinite(lp)
    self.assertAllFinite(grad)


if __name__ == '__main__':
  test_util.main()
