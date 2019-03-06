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
"""Tests for Categorical distribution."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
from absl.testing import parameterized
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability.python.internal import test_util as tfp_test_util
from tensorflow.python.framework import test_util  # pylint: disable=g-direct-tensorflow-import

tfd = tfp.distributions


def make_categorical(batch_shape, num_classes, dtype=tf.int32):
  logits = tf.random.uniform(
      list(batch_shape) + [num_classes], -10, 10, dtype=tf.float32) - 50.
  return tfd.Categorical(logits, dtype=dtype)


@test_util.run_all_in_graph_and_eager_modes
class CategoricalTest(tf.test.TestCase, parameterized.TestCase):

  def testP(self):
    p = [0.2, 0.8]
    dist = tfd.Categorical(probs=p)
    self.assertAllClose(p, self.evaluate(dist.probs))
    self.assertAllEqual([2], dist.logits.shape)

  def testLogits(self):
    p = np.array([0.2, 0.8], dtype=np.float32)
    logits = np.log(p) - 50.
    dist = tfd.Categorical(logits=logits)
    self.assertAllEqual([2], dist.probs.shape)
    self.assertAllEqual([2], dist.logits.shape)
    self.assertAllClose(p, self.evaluate(dist.probs))
    self.assertAllClose(logits, self.evaluate(dist.logits))

  def testShapes(self):
    for batch_shape in ([], [1], [2, 3, 4]):
      dist = make_categorical(batch_shape, 10)
      self.assertAllEqual(batch_shape, dist.batch_shape)
      self.assertAllEqual(batch_shape,
                          self.evaluate(dist.batch_shape_tensor()))
      self.assertAllEqual([], dist.event_shape)
      self.assertAllEqual([], self.evaluate(dist.event_shape_tensor()))
      self.assertEqual(10, self.evaluate(dist.num_categories))
      # num_categories is available as a constant because the shape is
      # known at graph build time.
      self.assertEqual(10, tf.get_static_value(dist.num_categories))

    for batch_shape in ([], [1], [2, 3, 4]):
      dist = make_categorical(
          batch_shape, tf.constant(
              10, dtype=tf.int32))
      self.assertAllEqual(len(batch_shape), dist.batch_shape.ndims)
      self.assertAllEqual(batch_shape,
                          self.evaluate(dist.batch_shape_tensor()))
      self.assertAllEqual([], dist.event_shape)
      self.assertAllEqual([], self.evaluate(dist.event_shape_tensor()))
      self.assertEqual(10, self.evaluate(dist.num_categories))

  def testDtype(self):
    dist = make_categorical([], 5, dtype=tf.int32)
    self.assertEqual(dist.dtype, tf.int32)
    self.assertEqual(dist.dtype, dist.sample(5).dtype)
    self.assertEqual(dist.dtype, dist.mode().dtype)
    dist = make_categorical([], 5, dtype=tf.int64)
    self.assertEqual(dist.dtype, tf.int64)
    self.assertEqual(dist.dtype, dist.sample(5).dtype)
    self.assertEqual(dist.dtype, dist.mode().dtype)
    self.assertEqual(dist.probs.dtype, tf.float32)
    self.assertEqual(dist.logits.dtype, tf.float32)
    self.assertEqual(dist.logits.dtype, dist.entropy().dtype)
    self.assertEqual(
        dist.logits.dtype, dist.prob(np.array(
            0, dtype=np.int64)).dtype)
    self.assertEqual(
        dist.logits.dtype, dist.log_prob(np.array(
            0, dtype=np.int64)).dtype)
    for dtype in [tf.float16, tf.float32, tf.float64]:
      dist = make_categorical([], 5, dtype=dtype)
      self.assertEqual(dist.dtype, dtype)
      self.assertEqual(dist.dtype, dist.sample(5).dtype)

  def testUnknownShape(self):
    logits = lambda l: tf.compat.v1.placeholder_with_default(
        np.float32(l), shape=None)
    sample = lambda l: tfd.Categorical(logits=logits(l)).sample()
    # Will sample class 1.
    sample_value = self.evaluate(sample([-1000.0, 1000.0]))
    self.assertEqual(1, sample_value)

    # Batch entry 0 will sample class 1, batch entry 1 will sample class 0.
    sample_value_batch = self.evaluate(
        sample([[-1000.0, 1000.0], [1000.0, -1000.0]]))
    self.assertAllEqual([1, 0], sample_value_batch)

  def testPMFWithBatch(self):
    histograms = [[0.2, 0.8], [0.6, 0.4]]
    dist = tfd.Categorical(tf.math.log(histograms) - 50.)
    self.assertAllClose([0.2, 0.4], self.evaluate(dist.prob([0, 1])))

  def testPMFNoBatch(self):
    histograms = [0.2, 0.8]
    dist = tfd.Categorical(tf.math.log(histograms) - 50.)
    self.assertAllClose(0.2, self.evaluate(dist.prob(0)))

  def testCDFWithDynamicEventShapeKnownNdims(self):
    """Test that dynamically-sized events with unknown shape work."""
    batch_size = 2
    make_ph = tf.compat.v1.placeholder_with_default
    histograms = lambda h: make_ph(np.float32(h), shape=(batch_size, None))
    event = lambda e: make_ph(np.float32(e), shape=(batch_size,))
    dist = lambda h: tfd.Categorical(probs=histograms(h))
    cdf_op = lambda h, e: dist(h).cdf(event(e))

    # Feed values in with different shapes...
    # three classes.
    event_feed_one = [0, 1]
    histograms_feed_one = [[0.5, 0.3, 0.2], [1.0, 0.0, 0.0]]
    expected_cdf_one = [0.5, 1.0]

    # six classes.
    event_feed_two = [2, 5]
    histograms_feed_two = [[0.9, 0.0, 0.0, 0.0, 0.0, 0.1],
                           [0.15, 0.2, 0.05, 0.35, 0.13, 0.12]]
    expected_cdf_two = [0.9, 1.0]

    actual_cdf_one = self.evaluate(
        cdf_op(histograms_feed_one, event_feed_one))
    actual_cdf_two = self.evaluate(
        cdf_op(histograms_feed_two, event_feed_two))

    self.assertAllClose(expected_cdf_one, actual_cdf_one)
    self.assertAllClose(expected_cdf_two, actual_cdf_two)

  @parameterized.named_parameters(
      ("test1", [0, 1], [[0.5, 0.3, 0.2], [1.0, 0.0, 0.0]], [0.5, 1.0]),
      ("test2", [2, 5], [[0.9, 0.0, 0.0, 0.0, 0.0, 0.1],
                         [0.15, 0.2, 0.05, 0.35, 0.13, 0.12]], [0.9, 1.0]))
  def testCDFWithDynamicEventShapeUnknownNdims(
      self, events, histograms, expected_cdf):
    """Test that dynamically-sized events with unknown shape work."""
    event_ph = tf.compat.v1.placeholder_with_default(events, shape=None)
    histograms_ph = tf.compat.v1.placeholder_with_default(
        histograms, shape=None)
    dist = tfd.Categorical(probs=histograms_ph)
    cdf_op = dist.cdf(event_ph)

    actual_cdf = self.evaluate(cdf_op)
    self.assertAllClose(actual_cdf, expected_cdf)

  def testCDFWithBatch(self):
    histograms = [[0.2, 0.1, 0.3, 0.25, 0.15],
                  [0.1, 0.2, 0.3, 0.2, 0.2],
                  [0.1, 0.2, 0.3, 0.2, 0.2],
                  [0.1, 0.2, 0.3, 0.2, 0.2]]
    # Note we're testing events outside [0, K-1].
    event = [0, 3, -1, 10]
    expected_cdf = [0.2, 0.8, 0.0, 1.0]
    dist = tfd.Categorical(probs=histograms)
    cdf_op = dist.cdf(event)

    self.assertAllClose(expected_cdf, self.evaluate(cdf_op))

  def testCDFWithBatchAndFloatDtype(self):
    histograms = [[0.1, 0.2, 0.3, 0.2, 0.2],
                  [0.1, 0.2, 0.3, 0.2, 0.2],
                  [0.1, 0.2, 0.3, 0.2, 0.2],
                  [0.1, 0.2, 0.3, 0.2, 0.2]]
    # Note we're testing events outside [0, K-1].
    event = [-1., 10., 2.0, 2.5]
    expected_cdf = [0.0, 1.0, 0.6, 0.6]
    dist = tfd.Categorical(probs=histograms, dtype=tf.float32)
    cdf_op = dist.cdf(event)

    self.assertAllClose(expected_cdf, self.evaluate(cdf_op))

  def testCDFNoBatch(self):
    histogram = [0.1, 0.2, 0.3, 0.4]
    event = 2
    expected_cdf = 0.6
    dist = tfd.Categorical(probs=histogram)
    cdf_op = dist.cdf(event)

    self.assertAlmostEqual(expected_cdf, self.evaluate(cdf_op))

  def testCDFBroadcasting(self):
    # shape: [batch=2, n_bins=3]
    histograms = [[0.2, 0.1, 0.7],
                  [0.3, 0.45, 0.25]]

    # shape: [batch=3, batch=2]
    devent = [
        [0, 0],
        [1, 1],
        [2, 2]
    ]
    dist = tfd.Categorical(probs=histograms)

    # We test that the probabilities are correctly broadcasted over the
    # additional leading batch dimension of size 3.
    expected_cdf_result = np.zeros((3, 2))
    expected_cdf_result[0, 0] = 0.2
    expected_cdf_result[0, 1] = 0.3
    expected_cdf_result[1, 0] = 0.3
    expected_cdf_result[1, 1] = 0.3 + 0.45
    expected_cdf_result[2, 0] = 1.0
    expected_cdf_result[2, 1] = 1.0

    self.assertAllClose(expected_cdf_result, self.evaluate(dist.cdf(devent)))

  def testBroadcastWithBatchParamsAndBiggerEvent(self):
    ## The parameters have a single batch dimension, and the event has two.

    # param shape is [3 x 4], where 4 is the number of bins (non-batch dim).
    cat_params_py = [
        [0.2, 0.15, 0.35, 0.3],
        [0.1, 0.05, 0.68, 0.17],
        [0.1, 0.05, 0.68, 0.17]
    ]

    # event shape = [5, 3], both are "batch" dimensions.
    disc_event_py = [
        [0, 1, 2],
        [1, 2, 3],
        [0, 0, 0],
        [1, 1, 1],
        [2, 1, 0]
    ]

    # shape is [3]
    normal_params_py = [
        -10.0,
        120.0,
        50.0
    ]

    # shape is [5, 3]
    real_event_py = [
        [-1.0, 0.0, 1.0],
        [100.0, 101, -50],
        [90, 90, 90],
        [-4, -400, 20.0],
        [0.0, 0.0, 0.0]
    ]

    cat_params_tf = tf.constant(cat_params_py)
    disc_event_tf = tf.constant(disc_event_py)
    cat = tfd.Categorical(probs=cat_params_tf)

    normal_params_tf = tf.constant(normal_params_py)
    real_event_tf = tf.constant(real_event_py)
    norm = tfd.Normal(loc=normal_params_tf, scale=1.0)

    # Check that normal and categorical have the same broadcasting behaviour.
    to_run = {
        "cat_prob": cat.prob(disc_event_tf),
        "cat_log_prob": cat.log_prob(disc_event_tf),
        "cat_cdf": cat.cdf(disc_event_tf),
        "cat_log_cdf": cat.log_cdf(disc_event_tf),
        "norm_prob": norm.prob(real_event_tf),
        "norm_log_prob": norm.log_prob(real_event_tf),
        "norm_cdf": norm.cdf(real_event_tf),
        "norm_log_cdf": norm.log_cdf(real_event_tf),
    }

    run_result = self.evaluate(to_run)

    self.assertAllEqual(run_result["cat_prob"].shape,
                        run_result["norm_prob"].shape)
    self.assertAllEqual(run_result["cat_log_prob"].shape,
                        run_result["norm_log_prob"].shape)
    self.assertAllEqual(run_result["cat_cdf"].shape,
                        run_result["norm_cdf"].shape)
    self.assertAllEqual(run_result["cat_log_cdf"].shape,
                        run_result["norm_log_cdf"].shape)

  def testLogPMF(self):
    logits = np.log([[0.2, 0.8], [0.6, 0.4]]) - 50.
    dist = tfd.Categorical(logits)
    self.assertAllClose(np.log([0.2, 0.4]),
                        self.evaluate(dist.log_prob([0, 1])))
    self.assertAllClose(np.log([0.2, 0.4]),
                        self.evaluate(dist.log_prob([0.0, 1.0])))

  def testEntropyNoBatch(self):
    logits = np.log([0.2, 0.8]) - 50.
    dist = tfd.Categorical(logits)
    self.assertAllClose(-(0.2 * np.log(0.2) + 0.8 * np.log(0.8)),
                        self.evaluate(dist.entropy()))

  def testEntropyWithBatch(self):
    logits = np.log([[0.2, 0.8], [0.6, 0.4]]) - 50.
    dist = tfd.Categorical(logits)
    self.assertAllClose([-(0.2 * np.log(0.2) + 0.8 * np.log(0.8)),
                         -(0.6 * np.log(0.6) + 0.4 * np.log(0.4))],
                        self.evaluate(dist.entropy()))

  def testEntropyGradient(self):
    with tf.GradientTape(persistent=True) as tape:
      logits = tf.constant([[1., 2., 3.], [2., 5., 1.]])
      tape.watch(logits)

      probabilities = tf.nn.softmax(logits)
      log_probabilities = tf.nn.log_softmax(logits)
      true_entropy = -tf.reduce_sum(
          input_tensor=probabilities * log_probabilities, axis=-1)

      categorical_distribution = tfd.Categorical(probs=probabilities)
      categorical_entropy = categorical_distribution.entropy()

    # works
    true_entropy_g = tape.gradient(true_entropy, logits)
    categorical_entropy_g = tape.gradient(categorical_entropy, logits)

    res = self.evaluate({"true_entropy": true_entropy,
                         "categorical_entropy": categorical_entropy,
                         "true_entropy_g": true_entropy_g,
                         "categorical_entropy_g": categorical_entropy_g})
    self.assertAllClose(res["true_entropy"],
                        res["categorical_entropy"])
    self.assertAllClose(res["true_entropy_g"],
                        res["categorical_entropy_g"])

  def testSample(self):
    histograms = [[[0.2, 0.8], [0.4, 0.6]]]
    dist = tfd.Categorical(tf.math.log(histograms) - 50.)
    n = 10000
    samples = dist.sample(n, seed=tfp_test_util.test_seed())
    samples.set_shape([n, 1, 2])
    self.assertEqual(samples.dtype, tf.int32)
    sample_values = self.evaluate(samples)
    self.assertFalse(np.any(sample_values < 0))
    self.assertFalse(np.any(sample_values > 1))
    self.assertAllClose(
        [[0.2, 0.4]], np.mean(sample_values == 0, axis=0), atol=1e-2)
    self.assertAllClose(
        [[0.8, 0.6]], np.mean(sample_values == 1, axis=0), atol=1e-2)

  def testSampleWithSampleShape(self):
    histograms = [[[0.2, 0.8], [0.4, 0.6]]]
    dist = tfd.Categorical(tf.math.log(histograms) - 50.)
    samples = dist.sample((100, 100), seed=tfp_test_util.test_seed())
    prob = dist.prob(samples)
    prob_val = self.evaluate(prob)
    self.assertAllClose(
        [0.2**2 + 0.8**2], [prob_val[:, :, :, 0].mean()], atol=1e-2)
    self.assertAllClose(
        [0.4**2 + 0.6**2], [prob_val[:, :, :, 1].mean()], atol=1e-2)

  def testNotReparameterized(self):
    p = tf.constant([0.3, 0.3, 0.4])
    _, grad_p = tfp.math.value_and_gradient(
        lambda x: tfd.Categorical(x).sample(100), p)
    self.assertIsNone(grad_p)

  def testLogPMFBroadcasting(self):
    # 1 x 2 x 2
    histograms = [[[0.2, 0.8], [0.4, 0.6]]]
    dist = tfd.Categorical(tf.math.log(histograms) - 50.)

    prob = dist.prob(1)
    self.assertAllClose([[0.8, 0.6]], self.evaluate(prob))

    prob = dist.prob([1])
    self.assertAllClose([[0.8, 0.6]], self.evaluate(prob))

    prob = dist.prob([0, 1])
    self.assertAllClose([[0.2, 0.6]], self.evaluate(prob))

    prob = dist.prob([[0, 1]])
    self.assertAllClose([[0.2, 0.6]], self.evaluate(prob))

    prob = dist.prob([[[0, 1]]])
    self.assertAllClose([[[0.2, 0.6]]], self.evaluate(prob))

    prob = dist.prob([[1, 0], [0, 1]])
    self.assertAllClose([[0.8, 0.4], [0.2, 0.6]], self.evaluate(prob))

    prob = dist.prob([[[1, 1], [1, 0]], [[1, 0], [0, 1]]])
    self.assertAllClose([[[0.8, 0.6], [0.8, 0.4]], [[0.8, 0.4], [0.2, 0.6]]],
                        self.evaluate(prob))

  def testLogPMFShape(self):
    # shape [1, 2, 2]
    histograms = [[[0.2, 0.8], [0.4, 0.6]]]
    dist = tfd.Categorical(tf.math.log(histograms))

    log_prob = dist.log_prob([0, 1])
    self.assertEqual(2, log_prob.shape.ndims)
    self.assertAllEqual([1, 2], log_prob.shape)

    log_prob = dist.log_prob([[[1, 1], [1, 0]], [[1, 0], [0, 1]]])
    self.assertEqual(3, log_prob.shape.ndims)
    self.assertAllEqual([2, 2, 2], log_prob.shape)

  def testLogPMFShapeNoBatch(self):
    histograms = [0.2, 0.8]
    dist = tfd.Categorical(tf.math.log(histograms))

    log_prob = dist.log_prob(0)
    self.assertEqual(0, log_prob.shape.ndims)
    self.assertAllEqual([], log_prob.shape)

    log_prob = dist.log_prob([[[1, 1], [1, 0]], [[1, 0], [0, 1]]])
    self.assertEqual(3, log_prob.shape.ndims)
    self.assertAllEqual([2, 2, 2], log_prob.shape)

  def testMode(self):
    histograms = [[[0.2, 0.8], [0.6, 0.4]]]
    dist = tfd.Categorical(tf.math.log(histograms) - 50.)
    self.assertAllEqual([[1, 0]], self.evaluate(dist.mode()))

  def testCategoricalCategoricalKL(self):

    def np_softmax(logits):
      exp_logits = np.exp(logits)
      return exp_logits / exp_logits.sum(axis=-1, keepdims=True)

    for categories in [2, 4]:
      for batch_size in [1, 10]:
        a_logits = np.random.randn(batch_size, categories)
        b_logits = np.random.randn(batch_size, categories)

        a = tfd.Categorical(logits=a_logits)
        b = tfd.Categorical(logits=b_logits)

        kl = tfd.kl_divergence(a, b)
        kl_val = self.evaluate(kl)
        # Make sure KL(a||a) is 0
        kl_same = self.evaluate(tfd.kl_divergence(a, a))

        prob_a = np_softmax(a_logits)
        prob_b = np_softmax(b_logits)
        kl_expected = np.sum(prob_a * (np.log(prob_a) - np.log(prob_b)),
                             axis=-1)

        self.assertEqual(kl.shape, (batch_size,))
        self.assertAllClose(kl_val, kl_expected)
        self.assertAllClose(kl_same, np.zeros_like(kl_expected))


if __name__ == "__main__":
  tf.test.main()
