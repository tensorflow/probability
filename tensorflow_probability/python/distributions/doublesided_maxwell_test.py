# Copyright 2018 The TensorFlow Probability Authors.
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
"""Tests for Double-sided Maxwell Distribution."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized

import numpy as np
from scipy import stats

import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp

from tensorflow_probability.python.internal import test_util

tfd = tfp.distributions


@test_util.test_all_tf_execution_regimes
class DoublesidedMaxwellTest(test_util.TestCase):

  def setUp(self):
    self._rng = np.random.RandomState(123)

  def _testParamShapes(self, sample_shape, expected):
    param_shapes = tfd.DoublesidedMaxwell.param_shapes(sample_shape)
    mu_shape, sigma_shape = param_shapes['loc'], param_shapes['scale']

    self.assertAllEqual(expected, self.evaluate(mu_shape))
    self.assertAllEqual(expected, self.evaluate(sigma_shape))
    mu = tf.zeros(mu_shape)
    sigma = tf.ones(sigma_shape)
    dsmaxwell = tfd.DoublesidedMaxwell(mu, sigma, validate_args=True)
    self.assertAllEqual(expected, self.evaluate(
        tf.shape(dsmaxwell.sample(seed=test_util.test_seed()))))

  def _testParamStaticShapes(self, sample_shape, expected):
    param_shapes = tfd.DoublesidedMaxwell.param_static_shapes(sample_shape)
    mu_shape, sigma_shape = param_shapes['loc'], param_shapes['scale']
    self.assertEqual(expected, mu_shape)
    self.assertEqual(expected, sigma_shape)

  def testParamShapes(self):
    sample_shape = [10, 3, 4]
    self._testParamShapes(sample_shape, sample_shape)
    self._testParamShapes(tf.constant(sample_shape), sample_shape)
    self._testParamStaticShapes(sample_shape, sample_shape)

  def testDoublesidedMaxwellPDF(self):
    # Use only positive values since we will use the one sided maxwell
    # as a test.
    x = np.arange(1, 5, dtype=np.float64)
    loc = 1.
    scale = 1.

    # Test the pdf value by using its relationship to the 1-sided Maxwell.
    dsmaxwell = tfd.DoublesidedMaxwell(loc=loc, scale=scale, validate_args=True)
    log_prob = dsmaxwell.log_prob(x)
    expected_log_prob = tf.identity(
        stats.maxwell.logpdf(np.abs(x), loc, scale) - np.log(2))
    self.assertAllClose(expected_log_prob, log_prob)

  def testInvalidScale(self):
    scale = [-.01, 0., 2.]
    with self.assertRaisesOpError('Argument `scale` must be positive.'):
      dsmaxwell = tfd.DoublesidedMaxwell(
          loc=0., scale=scale, validate_args=True)
      self.evaluate(dsmaxwell.scale)

  @parameterized.named_parameters(
      ('test1', 4.0, 1.0),
      ('test2', 2.0, 3.0))
  def testDoublesidedMaxwellSample(self, loc, scale):
    n = int(100e3)
    dsmaxwell = tfd.DoublesidedMaxwell(loc=loc, scale=scale, validate_args=True)
    samples = dsmaxwell.sample(n, seed=test_util.test_seed())
    mean = dsmaxwell.mean()
    variance = dsmaxwell.variance()

    [samples_, mean_, variance_] = self.evaluate([samples, mean, variance])

    # Check first and second moments.
    self.assertEqual((n,), samples_.shape)
    self.assertAllClose(np.mean(samples_), mean_, atol=0., rtol=0.1)
    self.assertAllClose(np.var(samples_), variance_, atol=0., rtol=0.1)

  def testDoublesidedMaxwellMean(self):
    # loc will be broadcast to [7, 7, 7]
    loc = [7.]
    sigma = [1., 2., 3.]

    dsmaxwell = tfd.DoublesidedMaxwell(loc=loc, scale=sigma, validate_args=True)

    self.assertAllEqual((3,), dsmaxwell.mean().shape)
    self.assertAllEqual([7., 7., 7.], self.evaluate(dsmaxwell.mean()))

  def testDoublesidedMaxwellVariance(self):
    # sigma will be broadcast to [7, 7, 7]
    loc = [1., 2., 3.]
    sigma = [7.]

    dsmaxwell = tfd.DoublesidedMaxwell(loc=loc, scale=sigma, validate_args=True)

    self.assertAllEqual((3,), dsmaxwell.variance().shape)
    self.assertAllClose([147., 147, 147], self.evaluate(dsmaxwell.variance()))

  def testDoublesidedMaxwellStandardDeviation(self):
    # sigma will be broadcast to [7, 7, 7]
    loc = [1., 2., 3.]
    sigma = [7.]

    dsmaxwell = tfd.DoublesidedMaxwell(loc=loc, scale=sigma, validate_args=True)

    self.assertAllEqual((3,), dsmaxwell.stddev().shape)
    std = np.sqrt(147)
    self.assertAllClose(
        [std, std, std], self.evaluate(dsmaxwell.stddev()))

  def testDoublesidedMaxwellStandardSampleShape(self):
    # sigma will be broadcast to [7, 7, 7]
    loc = [1., 2., 3.]
    sigma = [7.]

    dsmaxwell = tfd.DoublesidedMaxwell(loc=loc, scale=sigma, validate_args=True)

    n = 10
    samples = dsmaxwell.sample(n, seed=test_util.test_seed())
    self.assertAllEqual((n, 3,), samples.shape)

if __name__ == '__main__':
  tf.test.main()
