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
import itertools
# Dependency imports
from absl.testing import parameterized
import numpy as np
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp

from tensorflow_probability.python.internal import test_util

tfd = tfp.distributions
tfb = tfp.bijectors


@test_util.test_all_tf_execution_regimes
class StoppingRatioLogisticTest(test_util.TestCase):

  def _random_cutpoints(self, shape):
    return self._ordered.inverse(self._rng.randn(*shape))

  def _random_location(self, shape):
    return self._rng.randn(*shape)

  def _random_rvs(self, shape):
    return self._rng.multinomial(1, *shape)

  def setUp(self):
    self._ordered = tfb.Ordered()
    self._rng = test_util.test_np_rng()
    super(StoppingRatioLogisticTest, self).setUp()

  @parameterized.parameters(
      itertools.product(['cutpoints', 'loc', 'both'], [[], [1], [1, 2, 3]])
  )
  def testBatchShapes(self, test, batch_shape):
    if test == 'cutpoints':
      cutpoints = self._random_cutpoints(batch_shape + [2])
      loc = tf.constant(0., dtype=tf.float64)
    elif test == 'loc':
      cutpoints = tf.constant([1., 2.], dtype=tf.float64)
      loc = self._random_location(batch_shape)
    elif test == 'both':
      cutpoints = self._random_cutpoints(batch_shape + [2])
      loc = self._random_location(batch_shape)

    dist = tfd.StoppingRatioLogistic(cutpoints=cutpoints, loc=loc)

    self.assertAllEqual(dist.batch_shape, batch_shape)
    self.assertAllEqual(
        self.evaluate(dist.batch_shape_tensor()), batch_shape)

    self.assertAllEqual(dist.event_shape, [])
    self.assertAllEqual(self.evaluate(dist.event_shape_tensor()), [])

    categorical_probs = dist.categorical_probs()
    categorical_probs_shape = tf.shape(categorical_probs)
    self.assertAllEqual(
        self.evaluate(categorical_probs_shape), batch_shape + [3])

    samples = dist.sample(seed=test_util.test_seed())
    sample_shape = tf.shape(samples)
    self.assertAllEqual(self.evaluate(sample_shape), batch_shape)

    probs = dist.prob(samples)
    probs_shape = tf.shape(probs)
    self.assertAllEqual(self.evaluate(probs_shape), batch_shape)

    samples = dist.sample([4, 5], seed=test_util.test_seed())
    sample_shape_n = tf.shape(samples)
    self.assertAllEqual(self.evaluate(sample_shape_n), [4, 5] + batch_shape)

    probs = dist.prob(samples)
    probs_shape = tf.shape(probs)
    self.assertAllEqual(self.evaluate(probs_shape), [4, 5] + batch_shape)

    mode = dist.mode()
    mode_shape = tf.shape(mode)
    self.assertAllEqual(self.evaluate(mode_shape), batch_shape)

  def testProbs(self):
    expected_probs = [0.11920291, 0.44039854, 0.38790172, 0.05249681]
    dist = tfd.StoppingRatioLogistic(cutpoints=[-2., 0., 2.], loc=0.)

    categorical_probs = self.evaluate(dist.categorical_probs())
    self.assertAllClose(expected_probs, categorical_probs, atol=1e-4)

    probs = self.evaluate(dist.prob([0, 1, 2, 3]))
    self.assertAllClose(expected_probs, probs, atol=1e-4)

  def testMode(self):
    dist = tfd.StoppingRatioLogistic(cutpoints=[-10., 10.], loc=[-20., 0., 20.])
    mode = self.evaluate(dist.mode())
    self.assertAllEqual([0, 1, 2], mode)

  def testSample(self):
    dist = tfd.StoppingRatioLogistic(cutpoints=[-1., 0., 1.], loc=0.)
    samples = self.evaluate(dist.sample(int(1e5), seed=test_util.test_seed()))
    expected_probs = [0.2689414, 0.3655293, 0.26722333, 0.09830596]
    for k, p in enumerate(expected_probs):
      self.assertAllClose(np.mean(samples == k), p, atol=0.01)

  def testKLAgainstSampling(self):
    a_cutpoints = self._random_cutpoints([4])
    b_cutpoints = self._random_cutpoints([4])
    loc = self._random_location([])

    a = tfd.StoppingRatioLogistic(cutpoints=a_cutpoints, loc=loc)
    b = tfd.StoppingRatioLogistic(cutpoints=b_cutpoints, loc=loc)

    samples = a.sample(int(1e5), seed=test_util.test_seed())
    kl_samples = self.evaluate(a.log_prob(samples) - b.log_prob(samples))
    kl = self.evaluate(tfd.kl_divergence(a, b))

    self.assertAllMeansClose(kl_samples, kl, axis=0, atol=2e-2)

  def testUnorderedCutpointsFails(self):
    with self.assertRaisesRegexp(
        ValueError, 'Argument `cutpoints` must be non-decreasing.'):
      dist = tfd.StoppingRatioLogistic(
          cutpoints=[1., 0.9], loc=0.0, validate_args=True)
      self.evaluate(dist.mode())

if __name__ == '__main__':
  test_util.main()
