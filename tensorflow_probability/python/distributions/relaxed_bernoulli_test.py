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
"""Tests for the RelaxedBernoulli distribution."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
import numpy as np
import scipy.special
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp

from tensorflow_probability.python.internal import tensorshape_util
from tensorflow_probability.python.internal import test_util as tfp_test_util
from tensorflow.python.framework import test_util  # pylint: disable=g-direct-tensorflow-import

tfd = tfp.distributions


@test_util.run_all_in_graph_and_eager_modes
class RelaxedBernoulliTest(tf.test.TestCase):

  def testP(self):
    """Tests that parameter P is set correctly. Note that dist.p != dist.pdf."""
    temperature = 1.0
    p = [0.1, 0.4]
    dist = tfd.RelaxedBernoulli(temperature, probs=p)
    self.assertAllClose(p, self.evaluate(dist.probs))

  def testLogits(self):
    temperature = 2.0
    logits = [-42., 42.]
    dist = tfd.RelaxedBernoulli(temperature, logits=logits)
    self.assertAllClose(logits, self.evaluate(dist.logits))

    self.assertAllClose(scipy.special.expit(logits), self.evaluate(dist.probs))

    p = [0.01, 0.99, 0.42]
    dist = tfd.RelaxedBernoulli(temperature, probs=p)
    self.assertAllClose(scipy.special.logit(p), self.evaluate(dist.logits))

  def testInvalidP(self):
    temperature = 1.0
    invalid_ps = [1.01, 2.]
    for p in invalid_ps:
      with self.assertRaisesOpError("probs has components greater than 1"):
        dist = tfd.RelaxedBernoulli(temperature, probs=p, validate_args=True)
        self.evaluate(dist.probs)

    invalid_ps = [-0.01, -3.]
    for p in invalid_ps:
      with self.assertRaisesOpError("Condition x >= 0"):
        dist = tfd.RelaxedBernoulli(temperature, probs=p, validate_args=True)
        self.evaluate(dist.probs)

    valid_ps = [0.0, 0.5, 1.0]
    for p in valid_ps:
      dist = tfd.RelaxedBernoulli(temperature, probs=p)
      self.assertEqual(p, self.evaluate(dist.probs))

  def testShapes(self):
    for batch_shape in ([], [1], [2, 3, 4]):
      temperature = 1.0
      p = np.random.random(batch_shape).astype(np.float32)
      dist = tfd.RelaxedBernoulli(temperature, probs=p)
      self.assertAllEqual(batch_shape,
                          tensorshape_util.as_list(dist.batch_shape))
      self.assertAllEqual(batch_shape, self.evaluate(dist.batch_shape_tensor()))
      self.assertAllEqual([], tensorshape_util.as_list(dist.event_shape))
      self.assertAllEqual([], self.evaluate(dist.event_shape_tensor()))

  def testZeroTemperature(self):
    """If validate_args, raises InvalidArgumentError when temperature is 0."""
    temperature = tf.constant(0.0)
    p = tf.constant([0.1, 0.4])
    with self.assertRaises(tf.errors.InvalidArgumentError):
      dist = tfd.RelaxedBernoulli(temperature, probs=p, validate_args=True)
      sample = dist.sample()
      self.evaluate(sample)

  def testDtype(self):
    temperature = tf.constant(1.0, dtype=tf.float32)
    p = tf.constant([0.1, 0.4], dtype=tf.float32)
    dist = tfd.RelaxedBernoulli(temperature, probs=p)
    self.assertEqual(dist.dtype, tf.float32)
    self.assertEqual(dist.dtype, dist.sample(5).dtype)
    self.assertEqual(dist.probs.dtype, dist.prob([0.0]).dtype)
    self.assertEqual(dist.probs.dtype, dist.log_prob([0.0]).dtype)

    temperature = tf.constant(1.0, dtype=tf.float64)
    p = tf.constant([0.1, 0.4], dtype=tf.float64)
    dist64 = tfd.RelaxedBernoulli(temperature, probs=p)
    self.assertEqual(dist64.dtype, tf.float64)
    self.assertEqual(dist64.dtype, dist64.sample(5).dtype)

  def testLogProb(self):
    t = np.array(1.0, dtype=np.float64)
    p = np.array(0.1, dtype=np.float64)  # P(x=1)
    dist = tfd.RelaxedBernoulli(t, probs=p)
    xs = np.array([0.1, 0.3, 0.5, 0.9], dtype=np.float64)
    # analytical density from Maddison et al. 2016
    alpha = np.array(p / (1 - p), dtype=np.float64)
    expected_log_pdf = (
        np.log(t) + np.log(alpha) + (-t - 1) * (np.log(xs) + np.log(1 - xs)) -
        2 * np.log(alpha * np.power(xs, -t) + np.power(1 - xs, -t)))
    log_pdf = self.evaluate(dist.log_prob(xs))
    self.assertAllClose(expected_log_pdf, log_pdf)

  def testBoundaryConditions(self):
    temperature = 1e-2
    dist = tfd.RelaxedBernoulli(temperature, probs=1.0)
    self.assertAllClose(np.nan, self.evaluate(dist.log_prob(0.0)))
    self.assertAllClose([np.nan], [self.evaluate(dist.log_prob(1.0))])

  def testSampleN(self):
    """mean of quantized samples still approximates the Bernoulli mean."""
    temperature = 1e-2
    p = [0.2, 0.6, 0.5]
    dist = tfd.RelaxedBernoulli(temperature, probs=p)
    n = 10000
    samples = dist.sample(n, seed=tfp_test_util.test_seed())
    self.assertEqual(samples.dtype, tf.float32)
    sample_values = self.evaluate(samples)
    self.assertTrue(np.all(sample_values >= 0))
    self.assertTrue(np.all(sample_values <= 1))

    frac_ones_like = np.sum(sample_values >= 0.5, axis=0) / n
    self.assertAllClose(p, frac_ones_like, atol=1e-2)

  def testParamTensorFromLogits(self):
    x = tf.constant([-1., 0.5, 1.])
    d = tfd.RelaxedBernoulli(temperature=1., logits=x, validate_args=True)
    self.assertAllClose(
        *self.evaluate([x, d.logits_parameter()]),
        atol=0, rtol=1e-4)
    self.assertAllClose(
        *self.evaluate([tf.nn.sigmoid(x), d.probs_parameter()]),
        atol=0, rtol=1e-4)

  def testParamTensorFromProbs(self):
    x = tf.constant([0.1, 0.5, 0.4])
    d = tfd.RelaxedBernoulli(temperature=1., probs=x, validate_args=True)
    logit = lambda x: tf.math.log(x) - tf.math.log1p(-x)
    self.assertAllClose(
        *self.evaluate([logit(x), d.logits_parameter()]),
        atol=0, rtol=1e-4)
    self.assertAllClose(
        *self.evaluate([x, d.probs_parameter()]),
        atol=0, rtol=1e-4)


if __name__ == "__main__":
  tf.test.main()
