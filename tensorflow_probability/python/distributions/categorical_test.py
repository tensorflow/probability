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

# Dependency imports
from absl.testing import parameterized
import numpy as np

import tensorflow.compat.v1 as tf1
import tensorflow.compat.v2 as tf

from tensorflow_probability.python.bijectors import identity
from tensorflow_probability.python.distributions import categorical
from tensorflow_probability.python.distributions import kullback_leibler
from tensorflow_probability.python.distributions import normal
from tensorflow_probability.python.internal import prefer_static
from tensorflow_probability.python.internal import tensorshape_util
from tensorflow_probability.python.internal import test_util
from tensorflow_probability.python.math import gradient
from tensorflow_probability.python.util import deferred_tensor


def make_categorical(batch_shape, num_classes, dtype=tf.int32):
  logits = -50. + tf.random.uniform(
      list(batch_shape) + [num_classes], minval=-10, maxval=10,
      dtype=tf.float32, seed=test_util.test_seed())
  return categorical.Categorical(logits, dtype=dtype, validate_args=True)


@test_util.test_all_tf_execution_regimes
class CategoricalTest(test_util.TestCase):

  def testP(self):
    p = [0.2, 0.8]
    dist = categorical.Categorical(probs=p, validate_args=True)
    self.assertAllClose(p, self.evaluate(dist.probs))
    self.assertAllEqual([2], dist.probs.shape)

  def testLogits(self):
    p = np.array([0.2, 0.8], dtype=np.float32)
    logits = np.log(p) - 50.
    dist = categorical.Categorical(logits=logits, validate_args=True)
    self.assertAllEqual([2], dist.logits.shape)
    self.assertAllClose(logits, self.evaluate(dist.logits))

  def testShapes(self):
    for batch_shape in ([], [1], [2, 3, 4]):
      dist = make_categorical(batch_shape, 10)
      self.assertAllEqual(batch_shape, dist.batch_shape)
      self.assertAllEqual(batch_shape,
                          self.evaluate(dist.batch_shape_tensor()))
      self.assertAllEqual([], dist.event_shape)
      self.assertAllEqual([], self.evaluate(dist.event_shape_tensor()))
      num_categories = tf.shape(
          dist.probs if dist.logits is None else dist.logits)[-1]
      self.assertEqual(10, self.evaluate(num_categories))
      # The number of categories is available as a constant because the shape is
      # known at graph build time.
      num_categories = prefer_static.shape(
          dist.probs if dist.logits is None else dist.logits)[-1]
      self.assertEqual(10, tf.get_static_value(num_categories))

    for batch_shape in ([], [1], [2, 3, 4]):
      dist = make_categorical(
          batch_shape, tf.constant(
              10, dtype=tf.int32))
      self.assertAllEqual(
          len(batch_shape), tensorshape_util.rank(dist.batch_shape))
      self.assertAllEqual(batch_shape,
                          self.evaluate(dist.batch_shape_tensor()))
      self.assertAllEqual([], dist.event_shape)
      self.assertAllEqual([], self.evaluate(dist.event_shape_tensor()))
      num_categories = tf.shape(
          dist.probs if dist.logits is None else dist.logits)[-1]
      self.assertEqual(10, self.evaluate(num_categories))

  def testDtype(self):
    dist = make_categorical([], 5, dtype=tf.int32)
    self.assertEqual(dist.dtype, tf.int32)
    self.assertEqual(
        dist.dtype, dist.sample(5, seed=test_util.test_seed()).dtype)
    self.assertEqual(dist.dtype, dist.mode().dtype)
    dist = make_categorical([], 5, dtype=tf.int64)
    self.assertEqual(dist.dtype, tf.int64)
    self.assertEqual(
        dist.dtype, dist.sample(5, seed=test_util.test_seed()).dtype)
    self.assertEqual(dist.dtype, dist.mode().dtype)
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
      self.assertEqual(
          dist.dtype, dist.sample(5, seed=test_util.test_seed()).dtype)

  def testUnknownShape(self):
    logits = lambda l: tf1.placeholder_with_default(  # pylint: disable=g-long-lambda
        np.float32(l), shape=None)
    sample = lambda l: categorical.Categorical(  # pylint: disable=g-long-lambda
        logits=logits(l), validate_args=True).sample(seed=test_util.test_seed())
    # Will sample class 1.
    sample_value = self.evaluate(sample([-1000.0, 1000.0]))
    self.assertEqual(1, sample_value)

    # Batch entry 0 will sample class 1, batch entry 1 will sample class 0.
    sample_value_batch = self.evaluate(
        sample([[-1000.0, 1000.0], [1000.0, -1000.0]]))
    self.assertAllEqual([1, 0], sample_value_batch)

  def testPMFWithBatch(self):
    histograms = np.array([[0.2, 0.8], [0.6, 0.4]])
    dist = categorical.Categorical(
        tf.math.log(histograms) - 50., validate_args=True)
    self.assertAllClose([0.2, 0.4], self.evaluate(dist.prob([0, 1])))

  def testPMFNoBatch(self):
    histograms = np.array([0.2, 0.8])
    dist = categorical.Categorical(
        tf.math.log(histograms) - 50., validate_args=True)
    self.assertAllClose(0.2, self.evaluate(dist.prob(0)))

  def testCDFWithDynamicEventShapeKnownNdims(self):
    """Test that dynamically-sized events with unknown shape work."""
    batch_size = 2
    make_ph = tf1.placeholder_with_default
    histograms = lambda h: make_ph(np.float32(h), shape=(batch_size, None))
    event = lambda e: make_ph(np.float32(e), shape=(batch_size,))
    dist = lambda h: categorical.Categorical(  # pylint: disable=g-long-lambda
        probs=histograms(h),
        validate_args=True)
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
      ('test1', [0, 1], [[0.5, 0.3, 0.2], [1.0, 0.0, 0.0]], [0.5, 1.0]),
      ('test2', [2, 5], [[0.9, 0.0, 0.0, 0.0, 0.0, 0.1],
                         [0.15, 0.2, 0.05, 0.35, 0.13, 0.12]], [0.9, 1.0]))
  def testCDFWithDynamicEventShapeUnknownNdims(
      self, events, histograms, expected_cdf):
    """Test that dynamically-sized events with unknown shape work."""
    event_ph = tf1.placeholder_with_default(events, shape=None)
    histograms_ph = tf1.placeholder_with_default(
        histograms, shape=None)
    dist = categorical.Categorical(probs=histograms_ph, validate_args=True)
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
    dist = categorical.Categorical(probs=histograms, validate_args=False)
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
    dist = categorical.Categorical(
        probs=histograms, dtype=tf.float32, validate_args=False)
    cdf_op = dist.cdf(event)

    self.assertAllClose(expected_cdf, self.evaluate(cdf_op))

  def testCDFNoBatch(self):
    histogram = [0.1, 0.2, 0.3, 0.4]
    event = 2
    expected_cdf = 0.6
    dist = categorical.Categorical(probs=histogram, validate_args=True)
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
    dist = categorical.Categorical(probs=histograms, validate_args=True)

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
    cat = categorical.Categorical(probs=cat_params_tf, validate_args=True)

    normal_params_tf = tf.constant(normal_params_py)
    real_event_tf = tf.constant(real_event_py)
    norm = normal.Normal(loc=normal_params_tf, scale=1.0)

    # Check that normal and categorical have the same broadcasting behaviour.
    to_run = {
        'cat_prob': cat.prob(disc_event_tf),
        'cat_log_prob': cat.log_prob(disc_event_tf),
        'cat_cdf': cat.cdf(disc_event_tf),
        'cat_log_cdf': cat.log_cdf(disc_event_tf),
        'norm_prob': norm.prob(real_event_tf),
        'norm_log_prob': norm.log_prob(real_event_tf),
        'norm_cdf': norm.cdf(real_event_tf),
        'norm_log_cdf': norm.log_cdf(real_event_tf),
    }

    run_result = self.evaluate(to_run)

    self.assertAllEqual(run_result['cat_prob'].shape,
                        run_result['norm_prob'].shape)
    self.assertAllEqual(run_result['cat_log_prob'].shape,
                        run_result['norm_log_prob'].shape)
    self.assertAllEqual(run_result['cat_cdf'].shape,
                        run_result['norm_cdf'].shape)
    self.assertAllEqual(run_result['cat_log_cdf'].shape,
                        run_result['norm_log_cdf'].shape)

  def testLogPMF(self):
    logits = np.log(np.array([[0.2, 0.8], [0.6, 0.4]])) - 50.
    dist = categorical.Categorical(logits, validate_args=True)
    self.assertAllClose(np.log(np.array([0.2, 0.4])),
                        self.evaluate(dist.log_prob([0, 1])))
    self.assertAllClose(np.log(np.array([0.2, 0.4])),
                        self.evaluate(dist.log_prob([0.0, 1.0])))

  def testEntropyNoBatch(self):
    logits = np.log(np.array([0.2, 0.8])) - 50.
    dist = categorical.Categorical(logits, validate_args=True)
    self.assertAllClose(-(0.2 * np.log(0.2) + 0.8 * np.log(0.8)),
                        self.evaluate(dist.entropy()),
                        atol=0, rtol=1e-5)

  def testEntropyWithBatch(self):
    logits = np.log(np.array([[0.2, 0.8], [0.6, 0.4]])) - 50.
    dist = categorical.Categorical(logits, validate_args=True)
    self.assertAllClose([-(0.2 * np.log(0.2) + 0.8 * np.log(0.8)),
                         -(0.6 * np.log(0.6) + 0.4 * np.log(0.4))],
                        self.evaluate(dist.entropy()),
                        atol=0, rtol=1e-5)

  @test_util.numpy_disable_gradient_test
  def testEntropyGradient(self):

    def true_entropy_fn(logits):
      probabilities = tf.math.softmax(logits)
      log_probabilities = tf.math.log_softmax(logits)
      return -tf.reduce_sum(probabilities * log_probabilities, axis=-1)

    def categorical_entropy_fn(logits):
      categorical_distribution = categorical.Categorical(
          logits=logits, validate_args=True)
      return categorical_distribution.entropy()

    # works
    logits = tf.constant([[1., 2., 3.], [2., 5., 1.]])
    true_entropy, true_entropy_g = gradient.value_and_gradient(
        true_entropy_fn, [logits])
    categorical_entropy, categorical_entropy_g = gradient.value_and_gradient(
        categorical_entropy_fn, [logits])

    res = self.evaluate({'true_entropy': true_entropy,
                         'categorical_entropy': categorical_entropy,
                         'true_entropy_g': true_entropy_g,
                         'categorical_entropy_g': categorical_entropy_g})
    self.assertAllClose(res['true_entropy'],
                        res['categorical_entropy'])
    self.assertAllClose(res['true_entropy_g'],
                        res['categorical_entropy_g'])

  @test_util.numpy_disable_gradient_test
  def testEntropyGradientNegInfLogits(self):

    def get_entropy_fn(logit):
      logits = [logit, tf.constant(-np.inf, dtype=tf.float32)]
      categorical_distribution = categorical.Categorical(
          logits=logits, validate_args=True)
      return categorical_distribution.entropy()

    logit = 0.

    categorical_entropy, categorical_entropy_g = gradient.value_and_gradient(
        get_entropy_fn, logit)
    self.assertAllClose(categorical_entropy, 0.)
    self.assertAllClose(categorical_entropy_g, 0.)

  def testEntropyWithZeroProbabilities(self):
    probs = np.array([[0, 0.5, 0.5], [0, 1, 0]])
    dist = categorical.Categorical(probs=probs, validate_args=True)
    dist_entropy = dist.entropy()

    ans = [-(0.5*np.log(0.5) + 0.5*np.log(0.5)), -(np.log(1))]
    self.assertAllClose(self.evaluate(dist_entropy), ans)

  def testEntropyWithNegInfLogits(self):
    probs = np.array([[0, 0.5, 0.5], [0, 1, 0]])
    dist = categorical.Categorical(logits=np.log(probs), validate_args=True)
    dist_entropy = dist.entropy()

    ans = [-(0.5*np.log(0.5) + 0.5*np.log(0.5)), -(np.log(1))]
    self.assertAllClose(self.evaluate(dist_entropy), ans)

  def testEntropyWithLargeNegLogits(self):
    num_categories = 11
    logits = np.array([
        [-1e7] * num_categories,
        [-1e8] * num_categories,
        [-1e9] * num_categories], dtype=np.float32)
    dist = categorical.Categorical(logits=logits, validate_args=True)
    dist_entropy = dist.entropy()

    ans = [np.log(num_categories)] * 3
    self.assertAllClose(self.evaluate(dist_entropy), ans)

  def testSample(self):
    histograms = np.array([[[0.2, 0.8], [0.4, 0.6]]])
    dist = categorical.Categorical(
        tf.math.log(histograms) - 50., validate_args=True)
    n = 10000
    samples = dist.sample(n, seed=test_util.test_seed())
    tensorshape_util.set_shape(samples, [n, 1, 2])
    self.assertEqual(samples.dtype, tf.int32)
    sample_values = self.evaluate(samples)
    self.assertFalse(np.any(sample_values < 0))
    self.assertFalse(np.any(sample_values > 1))
    self.assertAllClose(
        [[0.2, 0.4]], np.mean(sample_values == 0, axis=0), atol=1e-2)
    self.assertAllClose(
        [[0.8, 0.6]], np.mean(sample_values == 1, axis=0), atol=1e-2)

  def testSampleWithSampleShape(self):
    histograms = np.array([[[0.2, 0.8], [0.4, 0.6]]])
    dist = categorical.Categorical(
        tf.math.log(histograms) - 50., validate_args=True)
    samples = dist.sample((100, 100), seed=test_util.test_seed())
    prob = dist.prob(samples)
    prob_val = self.evaluate(prob)
    self.assertAllClose(
        [0.2**2 + 0.8**2], [prob_val[:, :, :, 0].mean()], atol=1e-2)
    self.assertAllClose(
        [0.4**2 + 0.6**2], [prob_val[:, :, :, 1].mean()], atol=1e-2)

  @test_util.numpy_disable_gradient_test
  @test_util.jax_disable_test_missing_functionality(
      'JAX does not return None for gradients.')
  def testNotReparameterized(self):
    p = tf.constant([0.3, 0.3, 0.4])
    _, grad_p = gradient.value_and_gradient(
        # pylint: disable=g-long-lambda
        lambda x: categorical.Categorical(x, validate_args=True).sample(
            100, seed=test_util.test_seed()),
        p)
    self.assertIsNone(grad_p)

  def testLogPMFBroadcasting(self):
    # 1 x 2 x 2
    histograms = np.array([[[0.2, 0.8], [0.4, 0.6]]])
    dist = categorical.Categorical(
        tf.math.log(histograms) - 50., validate_args=True)

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
    histograms = np.array([[[0.2, 0.8], [0.4, 0.6]]])
    dist = categorical.Categorical(tf.math.log(histograms), validate_args=True)

    log_prob = dist.log_prob([0, 1])
    self.assertEqual(2, tensorshape_util.rank(log_prob.shape))
    self.assertAllEqual([1, 2], log_prob.shape)

    log_prob = dist.log_prob([[[1, 1], [1, 0]], [[1, 0], [0, 1]]])
    self.assertEqual(3, tensorshape_util.rank(log_prob.shape))
    self.assertAllEqual([2, 2, 2], log_prob.shape)

  def testLogPMFShapeNoBatch(self):
    histograms = np.array([0.2, 0.8])
    dist = categorical.Categorical(tf.math.log(histograms), validate_args=True)

    log_prob = dist.log_prob(0)
    self.assertEqual(0, tensorshape_util.rank(log_prob.shape))
    self.assertAllEqual([], log_prob.shape)

    log_prob = dist.log_prob([[[1, 1], [1, 0]], [[1, 0], [0, 1]]])
    self.assertEqual(3, tensorshape_util.rank(log_prob.shape))
    self.assertAllEqual([2, 2, 2], log_prob.shape)

  def testMode(self):
    histograms = np.array([[[0.2, 0.8], [0.6, 0.4]]])
    dist = categorical.Categorical(
        tf.math.log(histograms) - 50., validate_args=True)
    self.assertAllEqual([[1, 0]], self.evaluate(dist.mode()))

  def testCategoricalCategoricalKL(self):

    def np_softmax(logits):
      exp_logits = np.exp(logits)
      return exp_logits / exp_logits.sum(axis=-1, keepdims=True)

    for categories in [2, 4]:
      for batch_size in [1, 10]:
        a_logits = np.random.randn(batch_size, categories)
        b_logits = np.random.randn(batch_size, categories)

        a = categorical.Categorical(logits=a_logits, validate_args=True)
        b = categorical.Categorical(logits=b_logits, validate_args=True)

        kl = kullback_leibler.kl_divergence(a, b)
        kl_val = self.evaluate(kl)
        # Make sure KL(a||a) is 0
        kl_same = self.evaluate(kullback_leibler.kl_divergence(a, a))

        prob_a = np_softmax(a_logits)
        prob_b = np_softmax(b_logits)
        kl_expected = np.sum(prob_a * (np.log(prob_a) - np.log(prob_b)),
                             axis=-1)

        self.assertEqual(kl.shape, (batch_size,))
        self.assertAllClose(kl_val, kl_expected)
        self.assertAllClose(kl_same, np.zeros_like(kl_expected))

  @parameterized.parameters(
      # Test categorical distributions from probs.
      (dict(probs=[0.0, 0.5, 0.5]), dict(probs=[0.2, 0.2, 0.6])),
      # Test categorical distributions from logits.
      (dict(logits=[-np.inf, np.log(0.5), np.log(0.5)]),
       dict(logits=[np.log(0.2), np.log(0.2), np.log(0.6)])),
  )
  def testCategoricalCategoricalKLExtreme(self, kwargs1, kwargs2):
    a = categorical.Categorical(**kwargs1)
    kl = kullback_leibler.kl_divergence(a, a)
    kl_val = self.evaluate(kl)
    self.assertEqual(kl_val, 0.0)

    b = categorical.Categorical(**kwargs2)
    kl = kullback_leibler.kl_divergence(a, b)
    kl_val = self.evaluate(kl)
    self.assertAllClose(kl_val, np.log(0.5) - 0.5 * (np.log(0.2) + np.log(0.6)))

    kl = kullback_leibler.kl_divergence(b, a)
    kl_val = self.evaluate(kl)
    self.assertEqual(kl_val, np.inf)

  @test_util.numpy_disable_gradient_test
  @parameterized.parameters(
      (np.array([-np.inf, np.log(0.5), np.log(0.5)]),
       np.array([np.log(0.2), np.log(0.2), np.log(0.6)])),
      (np.array([-np.inf, np.log(0.5), np.log(0.5)]),
       np.array([-np.inf, np.log(0.2), np.log(0.6)])),
  )
  def testCategoricalCategoricalKLExtremeGrad(self, a_logits, b_logits):
    def get_kl_fn(a_logits, b_logits):
      a = categorical.Categorical(logits=a_logits)
      b = categorical.Categorical(logits=b_logits)
      return a.kl_divergence(b)

    kl, grads = gradient.value_and_gradient(
        get_kl_fn, a_logits, b_logits)
    # We get the expected answer when clipping the -inf.
    expected_kl, expected_grads = gradient.value_and_gradient(
        get_kl_fn, np.maximum(-10000, a_logits), np.maximum(-10000, b_logits))
    self.assertAllClose(kl, expected_kl)
    self.assertAllClose(grads, expected_grads)

  def testParamTensorFromLogits(self):
    x = tf.constant([-1., 0.5, 1.])
    d = categorical.Categorical(logits=x, validate_args=True)
    self.assertAllClose(
        *self.evaluate([x, d.logits_parameter()]),
        atol=0, rtol=1e-4)
    self.assertAllClose(
        *self.evaluate([tf.math.softmax(x),
                        d.probs_parameter()]),
        atol=0,
        rtol=1e-4)

  def testParamTensorFromProbs(self):
    x = tf.constant([0.1, 0.5, 0.4])
    d = categorical.Categorical(probs=x, validate_args=True)
    self.assertAllClose(
        *self.evaluate([tf.math.log(x), d.logits_parameter()]),
        atol=0, rtol=1e-4)
    self.assertAllClose(
        *self.evaluate([x, d.probs_parameter()]),
        atol=0, rtol=1e-4)


@test_util.test_all_tf_execution_regimes
class ForceProbsToZeroOutsideSupportTest(test_util.TestCase):

  def testPMFFloats(self):
    logits = np.log(np.array([0.2, 0.8]))
    dist = categorical.Categorical(logits=logits,
                                   force_probs_to_zero_outside_support=True)
    inputs = [0., 0.5, 1., 1.5]
    # Should be 0 for non-integers.
    expected_outputs = np.array([0.2, 0., 0.8, 0.])
    self.assertAllClose(self.evaluate(dist.prob(inputs)), expected_outputs)

  def testPMFOutOfBounds(self):
    logits = np.log(np.array([0.2, 0.8]))
    dist = categorical.Categorical(logits=logits,
                                   force_probs_to_zero_outside_support=True)
    inputs = [-2, -1, 0, 1, 2, 3]
    # Should be 0 for out of bound inputs.
    expected_outputs = np.array([0., 0., 0.2, 0.8, 0., 0.])
    self.assertAllClose(self.evaluate(dist.prob(inputs)), expected_outputs)


@test_util.test_all_tf_execution_regimes
class CategoricalFromVariableTest(test_util.TestCase):

  @test_util.tf_tape_safety_test
  def testGradientLogits(self):
    x = tf.Variable([-1., 0., 1])
    d = categorical.Categorical(logits=x, validate_args=True)
    with tf.GradientTape() as tape:
      loss = -d.log_prob([0, 2])
    g = tape.gradient(loss, d.trainable_variables)
    self.assertLen(g, 1)
    self.assertAllNotNone(g)

  @test_util.tf_tape_safety_test
  def testGradientProbs(self):
    x = tf.Variable([0.1, 0.7, 0.2])
    d = categorical.Categorical(probs=x, validate_args=True)
    with tf.GradientTape() as tape:
      loss = -d.log_prob([0, 2])
    g = tape.gradient(loss, d.trainable_variables)
    self.assertLen(g, 1)
    self.assertAllNotNone(g)

  def testAssertionsProbs(self):
    x = tf.Variable([0.1, 0.7, 0.0])
    with self.assertRaisesOpError('Argument `probs` must sum to 1.'):
      d = categorical.Categorical(probs=x, validate_args=True)
      self.evaluate([v.initializer for v in d.variables])
      self.evaluate(d.entropy())

  def testAssertionsLogits(self):
    x = deferred_tensor.TransformedVariable(0., identity.Identity(), shape=None)
    with self.assertRaisesRegexp(
        ValueError, 'Argument `logits` must have rank at least 1.'):
      d = categorical.Categorical(logits=x, validate_args=True)
      self.evaluate([v.initializer for v in d.variables])
      self.evaluate(d.entropy())


if __name__ == '__main__':
  test_util.main()
