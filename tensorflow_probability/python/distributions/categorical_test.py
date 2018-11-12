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

from tensorflow_probability.python.distributions import categorical
from tensorflow_probability.python.distributions import kullback_leibler
from tensorflow_probability.python.distributions import normal
from tensorflow.python.eager import backprop
from tensorflow.python.framework import tensor_util


def make_categorical(batch_shape, num_classes, dtype=tf.int32):
  logits = tf.random_uniform(
      list(batch_shape) + [num_classes], -10, 10, dtype=tf.float32) - 50.
  return categorical.Categorical(logits, dtype=dtype)


class CategoricalTest(tf.test.TestCase, parameterized.TestCase):

  def testP(self):
    p = [0.2, 0.8]
    dist = categorical.Categorical(probs=p)
    with self.cached_session():
      self.assertAllClose(p, dist.probs.eval())
      self.assertAllEqual([2], dist.logits.shape)

  def testLogits(self):
    p = np.array([0.2, 0.8], dtype=np.float32)
    logits = np.log(p) - 50.
    dist = categorical.Categorical(logits=logits)
    with self.cached_session():
      self.assertAllEqual([2], dist.probs.shape)
      self.assertAllEqual([2], dist.logits.shape)
      self.assertAllClose(dist.probs.eval(), p)
      self.assertAllClose(dist.logits.eval(), logits)

  def testShapes(self):
    with self.cached_session():
      for batch_shape in ([], [1], [2, 3, 4]):
        dist = make_categorical(batch_shape, 10)
        self.assertAllEqual(batch_shape, dist.batch_shape)
        self.assertAllEqual(batch_shape, dist.batch_shape_tensor().eval())
        self.assertAllEqual([], dist.event_shape)
        self.assertAllEqual([], dist.event_shape_tensor().eval())
        self.assertEqual(10, dist.event_size.eval())
        # event_size is available as a constant because the shape is
        # known at graph build time.
        self.assertEqual(10, tensor_util.constant_value(dist.event_size))

      for batch_shape in ([], [1], [2, 3, 4]):
        dist = make_categorical(
            batch_shape, tf.constant(
                10, dtype=tf.int32))
        self.assertAllEqual(len(batch_shape), dist.batch_shape.ndims)
        self.assertAllEqual(batch_shape, dist.batch_shape_tensor().eval())
        self.assertAllEqual([], dist.event_shape)
        self.assertAllEqual([], dist.event_shape_tensor().eval())
        self.assertEqual(10, dist.event_size.eval())

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
    with self.cached_session():
      logits = tf.placeholder(dtype=tf.float32)
      dist = categorical.Categorical(logits)
      sample = dist.sample()
      # Will sample class 1.
      sample_value = sample.eval(feed_dict={logits: [-1000.0, 1000.0]})
      self.assertEqual(1, sample_value)

      # Batch entry 0 will sample class 1, batch entry 1 will sample class 0.
      sample_value_batch = sample.eval(
          feed_dict={logits: [[-1000.0, 1000.0], [1000.0, -1000.0]]})
      self.assertAllEqual([1, 0], sample_value_batch)

  def testPMFWithBatch(self):
    histograms = [[0.2, 0.8], [0.6, 0.4]]
    dist = categorical.Categorical(tf.log(histograms) - 50.)
    with self.cached_session():
      self.assertAllClose(dist.prob([0, 1]).eval(), [0.2, 0.4])

  def testPMFNoBatch(self):
    histograms = [0.2, 0.8]
    dist = categorical.Categorical(tf.log(histograms) - 50.)
    with self.cached_session():
      self.assertAllClose(dist.prob(0).eval(), 0.2)

  def testCDFWithDynamicEventShapeKnownNdims(self):
    """Test that dynamically-sized events with unknown shape work."""
    batch_size = 2
    histograms = tf.placeholder(dtype=tf.float32,
                                shape=(batch_size, None))
    event = tf.placeholder(dtype=tf.float32, shape=(batch_size,))
    dist = categorical.Categorical(probs=histograms)
    cdf_op = dist.cdf(event)

    # Feed values into the placeholder with different shapes
    # three classes.
    event_feed_one = [0, 1]
    histograms_feed_one = [[0.5, 0.3, 0.2], [1.0, 0.0, 0.0]]
    expected_cdf_one = [0.0, 1.0]
    feed_dict_one = {
        histograms: histograms_feed_one,
        event: event_feed_one
    }

    # six classes.
    event_feed_two = [2, 5]
    histograms_feed_two = [[0.9, 0.0, 0.0, 0.0, 0.0, 0.1],
                           [0.15, 0.2, 0.05, 0.35, 0.13, 0.12]]
    expected_cdf_two = [0.9, 0.88]
    feed_dict_two = {
        histograms: histograms_feed_two,
        event: event_feed_two
    }

    with self.cached_session() as sess:
      actual_cdf_one = sess.run(cdf_op, feed_dict=feed_dict_one)
      actual_cdf_two = sess.run(cdf_op, feed_dict=feed_dict_two)

    self.assertAllClose(actual_cdf_one, expected_cdf_one)
    self.assertAllClose(actual_cdf_two, expected_cdf_two)

  @parameterized.named_parameters(
      ("test1", [0, 1], [[0.5, 0.3, 0.2], [1.0, 0.0, 0.0]], [0.0, 1.0]),
      ("test2", [2, 5], [[0.9, 0.0, 0.0, 0.0, 0.0, 0.1],
                         [0.15, 0.2, 0.05, 0.35, 0.13, 0.12]], [0.9, 0.88]))
  def testCDFWithDynamicEventShapeUnknownNdims(
      self, events, histograms, expected_cdf):
    """Test that dynamically-sized events with unknown shape work."""
    event_ph = tf.placeholder_with_default(events, shape=None)
    histograms_ph = tf.placeholder_with_default(histograms, shape=None)
    dist = categorical.Categorical(probs=histograms_ph)
    cdf_op = dist.cdf(event_ph)

    actual_cdf = self.evaluate(cdf_op)
    self.assertAllClose(actual_cdf, expected_cdf)

  def testCDFWithBatch(self):
    histograms = [[0.1, 0.2, 0.3, 0.25, 0.15],
                  [0.0, 0.75, 0.2, 0.05, 0.0]]
    event = [0, 3]
    expected_cdf = [0.0, 0.95]
    dist = categorical.Categorical(probs=histograms)
    cdf_op = dist.cdf(event)

    with self.cached_session():
      self.assertAllClose(cdf_op.eval(), expected_cdf)

  def testCDFNoBatch(self):
    histogram = [0.1, 0.2, 0.3, 0.4]
    event = 2
    expected_cdf = 0.3
    dist = categorical.Categorical(probs=histogram)
    cdf_op = dist.cdf(event)

    with self.cached_session():
      self.assertAlmostEqual(cdf_op.eval(), expected_cdf)

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
    dist = categorical.Categorical(probs=histograms)

    # We test that the probabilities are correctly broadcasted over the
    # additional leading batch dimension of size 3.
    expected_cdf_result = np.zeros((3, 2))
    expected_cdf_result[0, 0] = 0
    expected_cdf_result[0, 1] = 0
    expected_cdf_result[1, 0] = 0.2
    expected_cdf_result[1, 1] = 0.3
    expected_cdf_result[2, 0] = 0.3
    expected_cdf_result[2, 1] = 0.75

    with self.cached_session():
      self.assertAllClose(dist.cdf(devent).eval(), expected_cdf_result)

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
    cat = categorical.Categorical(probs=cat_params_tf)

    normal_params_tf = tf.constant(normal_params_py)
    real_event_tf = tf.constant(real_event_py)
    norm = normal.Normal(loc=normal_params_tf, scale=1.0)

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

    with self.cached_session() as sess:
      run_result = sess.run(to_run)

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
    dist = categorical.Categorical(logits)
    with self.cached_session():
      self.assertAllClose(dist.log_prob([0, 1]).eval(), np.log([0.2, 0.4]))
      self.assertAllClose(dist.log_prob([0.0, 1.0]).eval(), np.log([0.2, 0.4]))

  def testEntropyNoBatch(self):
    logits = np.log([0.2, 0.8]) - 50.
    dist = categorical.Categorical(logits)
    with self.cached_session():
      self.assertAllClose(dist.entropy().eval(),
                          -(0.2 * np.log(0.2) + 0.8 * np.log(0.8)))

  def testEntropyWithBatch(self):
    logits = np.log([[0.2, 0.8], [0.6, 0.4]]) - 50.
    dist = categorical.Categorical(logits)
    with self.cached_session():
      self.assertAllClose(dist.entropy().eval(), [
          -(0.2 * np.log(0.2) + 0.8 * np.log(0.8)),
          -(0.6 * np.log(0.6) + 0.4 * np.log(0.4))
      ])

  def testEntropyGradient(self):
    with self.cached_session() as sess:
      logits = tf.constant([[1., 2., 3.], [2., 5., 1.]])

      probabilities = tf.nn.softmax(logits)
      log_probabilities = tf.nn.log_softmax(logits)
      true_entropy = - tf.reduce_sum(
          probabilities * log_probabilities, axis=-1)

      categorical_distribution = categorical.Categorical(probs=probabilities)
      categorical_entropy = categorical_distribution.entropy()

      # works
      true_entropy_g = tf.gradients(true_entropy, [logits])
      categorical_entropy_g = tf.gradients(
          categorical_entropy, [logits])

      res = sess.run({"true_entropy": true_entropy,
                      "categorical_entropy": categorical_entropy,
                      "true_entropy_g": true_entropy_g,
                      "categorical_entropy_g": categorical_entropy_g})
      self.assertAllClose(res["true_entropy"],
                          res["categorical_entropy"])
      self.assertAllClose(res["true_entropy_g"],
                          res["categorical_entropy_g"])

  def testSample(self):
    with self.cached_session():
      histograms = [[[0.2, 0.8], [0.4, 0.6]]]
      dist = categorical.Categorical(tf.log(histograms) - 50.)
      n = 10000
      samples = dist.sample(n, seed=123)
      samples.set_shape([n, 1, 2])
      self.assertEqual(samples.dtype, tf.int32)
      sample_values = samples.eval()
      self.assertFalse(np.any(sample_values < 0))
      self.assertFalse(np.any(sample_values > 1))
      self.assertAllClose(
          [[0.2, 0.4]], np.mean(
              sample_values == 0, axis=0), atol=1e-2)
      self.assertAllClose(
          [[0.8, 0.6]], np.mean(
              sample_values == 1, axis=0), atol=1e-2)

  def testSampleWithSampleShape(self):
    with self.cached_session():
      histograms = [[[0.2, 0.8], [0.4, 0.6]]]
      dist = categorical.Categorical(tf.log(histograms) - 50.)
      samples = dist.sample((100, 100), seed=123)
      prob = dist.prob(samples)
      prob_val = prob.eval()
      self.assertAllClose(
          [0.2**2 + 0.8**2], [prob_val[:, :, :, 0].mean()], atol=1e-2)
      self.assertAllClose(
          [0.4**2 + 0.6**2], [prob_val[:, :, :, 1].mean()], atol=1e-2)

  def testNotReparameterized(self):
    p = tf.constant([0.3, 0.3, 0.4])
    with backprop.GradientTape() as tape:
      tape.watch(p)
      dist = categorical.Categorical(p)
      samples = dist.sample(100)
    grad_p = tape.gradient(samples, p)
    self.assertIsNone(grad_p)

  def testLogPMFBroadcasting(self):
    with self.cached_session():
      # 1 x 2 x 2
      histograms = [[[0.2, 0.8], [0.4, 0.6]]]
      dist = categorical.Categorical(tf.log(histograms) - 50.)

      prob = dist.prob(1)
      self.assertAllClose([[0.8, 0.6]], prob.eval())

      prob = dist.prob([1])
      self.assertAllClose([[0.8, 0.6]], prob.eval())

      prob = dist.prob([0, 1])
      self.assertAllClose([[0.2, 0.6]], prob.eval())

      prob = dist.prob([[0, 1]])
      self.assertAllClose([[0.2, 0.6]], prob.eval())

      prob = dist.prob([[[0, 1]]])
      self.assertAllClose([[[0.2, 0.6]]], prob.eval())

      prob = dist.prob([[1, 0], [0, 1]])
      self.assertAllClose([[0.8, 0.4], [0.2, 0.6]], prob.eval())

      prob = dist.prob([[[1, 1], [1, 0]], [[1, 0], [0, 1]]])
      self.assertAllClose([[[0.8, 0.6], [0.8, 0.4]], [[0.8, 0.4], [0.2, 0.6]]],
                          prob.eval())

  def testLogPMFShape(self):
    with self.cached_session():
      # shape [1, 2, 2]
      histograms = [[[0.2, 0.8], [0.4, 0.6]]]
      dist = categorical.Categorical(tf.log(histograms))

      log_prob = dist.log_prob([0, 1])
      self.assertEqual(2, log_prob.shape.ndims)
      self.assertAllEqual([1, 2], log_prob.shape)

      log_prob = dist.log_prob([[[1, 1], [1, 0]], [[1, 0], [0, 1]]])
      self.assertEqual(3, log_prob.shape.ndims)
      self.assertAllEqual([2, 2, 2], log_prob.shape)

  def testLogPMFShapeNoBatch(self):
    histograms = [0.2, 0.8]
    dist = categorical.Categorical(tf.log(histograms))

    log_prob = dist.log_prob(0)
    self.assertEqual(0, log_prob.shape.ndims)
    self.assertAllEqual([], log_prob.shape)

    log_prob = dist.log_prob([[[1, 1], [1, 0]], [[1, 0], [0, 1]]])
    self.assertEqual(3, log_prob.shape.ndims)
    self.assertAllEqual([2, 2, 2], log_prob.shape)

  def testMode(self):
    with self.cached_session():
      histograms = [[[0.2, 0.8], [0.6, 0.4]]]
      dist = categorical.Categorical(tf.log(histograms) - 50.)
      self.assertAllEqual(dist.mode().eval(), [[1, 0]])

  def testCategoricalCategoricalKL(self):

    def np_softmax(logits):
      exp_logits = np.exp(logits)
      return exp_logits / exp_logits.sum(axis=-1, keepdims=True)

    with self.cached_session() as sess:
      for categories in [2, 4]:
        for batch_size in [1, 10]:
          a_logits = np.random.randn(batch_size, categories)
          b_logits = np.random.randn(batch_size, categories)

          a = categorical.Categorical(logits=a_logits)
          b = categorical.Categorical(logits=b_logits)

          kl = kullback_leibler.kl_divergence(a, b)
          kl_val = sess.run(kl)
          # Make sure KL(a||a) is 0
          kl_same = sess.run(kullback_leibler.kl_divergence(a, a))

          prob_a = np_softmax(a_logits)
          prob_b = np_softmax(b_logits)
          kl_expected = np.sum(prob_a * (np.log(prob_a) - np.log(prob_b)),
                               axis=-1)

          self.assertEqual(kl.shape, (batch_size,))
          self.assertAllClose(kl_val, kl_expected)
          self.assertAllClose(kl_same, np.zeros_like(kl_expected))


if __name__ == "__main__":
  tf.test.main()
