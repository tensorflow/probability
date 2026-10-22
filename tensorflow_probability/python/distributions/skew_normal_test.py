# Copyright 2026 The TensorFlow Probability Authors.
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
"""Tests for SkewNormal."""

import math
import numpy as np
from scipy import stats as sp_stats

import tensorflow.compat.v2 as tf

from tensorflow_probability.python.distributions import skew_normal
from tensorflow_probability.python.internal import test_util


@test_util.test_all_tf_execution_regimes
class SkewNormalTest(test_util.TestCase):

  def setUp(self):
    self._rng = np.random.RandomState(123)
    super(SkewNormalTest, self).setUp()

  def testSkewNormalShape(self):
    loc = tf.constant([3.0] * 5)
    scale = tf.constant(11.0)
    skewness = tf.constant([2.0] * 5)
    dist = skew_normal.SkewNormal(loc=loc, scale=scale, skewness=skewness)
    
    self.assertEqual(self.evaluate(dist.batch_shape_tensor()), (5,))
    self.assertEqual(dist.batch_shape, tf.TensorShape([5]))
    self.assertAllEqual(self.evaluate(dist.event_shape_tensor()), [])
    self.assertEqual(dist.event_shape, tf.TensorShape([]))

  def testSkewNormalLogPDF(self):
    batch_size = 6
    loc = tf.constant([2.0] * batch_size)
    scale = tf.constant([3.0] * batch_size)
    skewness = tf.constant([0.0, 1.0, -1.0, 2.0, -2.0, 5.0])
    x = np.array([2.5, 2.5, 4.0, -1.0, 5.0, 2.0], dtype=np.float32)

    dist = skew_normal.SkewNormal(loc=loc, scale=scale, skewness=skewness)
    log_pdf = dist.log_prob(x)
    
    expected_log_pdf = sp_stats.skewnorm.logpdf(
        x, a=self.evaluate(skewness), loc=self.evaluate(loc), scale=self.evaluate(scale))
    
    self.assertAllClose(self.evaluate(log_pdf), expected_log_pdf, rtol=1e-4)

  def testSkewNormalCDF(self):
    batch_size = 6
    loc = tf.constant([2.0] * batch_size)
    scale = tf.constant([3.0] * batch_size)
    skewness = tf.constant([0.0, 1.0, -1.0, 2.0, -2.0, 5.0])
    x = np.array([2.5, 2.5, 4.0, -1.0, 5.0, 2.0], dtype=np.float32)

    dist = skew_normal.SkewNormal(loc=loc, scale=scale, skewness=skewness)
    cdf = dist.cdf(x)
    
    expected_cdf = sp_stats.skewnorm.cdf(
        x, a=self.evaluate(skewness), loc=self.evaluate(loc), scale=self.evaluate(scale))
    
    self.assertAllClose(self.evaluate(cdf), expected_cdf, rtol=1e-4)

  def testSkewNormalMean(self):
    loc = np.array([2.0, -1.0, 0.0])
    scale = np.array([3.0, 0.5, 1.0])
    skewness = np.array([1.0, -2.0, 0.0])

    dist = skew_normal.SkewNormal(loc=loc, scale=scale, skewness=skewness)
    expected_mean = sp_stats.skewnorm.mean(a=skewness, loc=loc, scale=scale)
    
    self.assertAllClose(self.evaluate(dist.mean()), expected_mean, rtol=1e-4)

  def testSkewNormalVariance(self):
    loc = np.array([2.0, -1.0, 0.0])
    scale = np.array([3.0, 0.5, 1.0])
    skewness = np.array([1.0, -2.0, 0.0])

    dist = skew_normal.SkewNormal(loc=loc, scale=scale, skewness=skewness)
    expected_var = sp_stats.skewnorm.var(a=skewness, loc=loc, scale=scale)
    
    self.assertAllClose(self.evaluate(dist.variance()), expected_var, rtol=1e-4)

  def testSkewNormalSample(self):
    loc = tf.constant(2.0)
    scale = tf.constant(3.0)
    skewness = tf.constant(1.5)
    
    dist = skew_normal.SkewNormal(loc=loc, scale=scale, skewness=skewness)
    
    n = 100000
    samples = dist.sample(n, seed=test_util.test_seed())
    sample_values = self.evaluate(samples)
    
    self.assertEqual(sample_values.shape, (n,))
    self.assertAllClose(
        sample_values.mean(),
        self.evaluate(dist.mean()),
        rtol=0.02,
        atol=0.02)
    self.assertAllClose(
        sample_values.var(),
        self.evaluate(dist.variance()),
        rtol=0.02,
        atol=0.02)

  def testSkewNormalFullyReparameterized(self):
    loc = tf.constant(2.0)
    scale = tf.constant(3.0)
    skewness = tf.constant(1.5)
    
    # We do a simpler reparameterization check instead of `compute_gradient`
    # because tf.test.compute_gradient expects scalar output or specific shapes.
    # Instead, we just check if the gradient w.r.t parameters is not None.
    with tf.GradientTape() as tape:
      tape.watch([loc, scale, skewness])
      dist = skew_normal.SkewNormal(loc=loc, scale=scale, skewness=skewness)
      samples = dist.sample(100, seed=test_util.test_seed())
      loss = tf.reduce_mean(samples)
      
    grad_loc, grad_scale, grad_skewness = tape.gradient(loss, [loc, scale, skewness])
    self.assertIsNotNone(grad_loc)
    self.assertIsNotNone(grad_scale)
    self.assertIsNotNone(grad_skewness)

if __name__ == '__main__':
  test_util.main()
