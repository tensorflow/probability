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
"""Tests for Mixture distribution."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import contextlib

# Dependency imports
import numpy as np
from scipy import stats
import tensorflow.compat.v1 as tf1
import tensorflow.compat.v2 as tf

from tensorflow_probability.python import distributions as tfd
from tensorflow_probability.python.internal import hypothesis_testlib as tfp_hps
from tensorflow_probability.python.internal import tensorshape_util
from tensorflow_probability.python.internal import test_util


def _swap_first_last_axes(array):
  rank = len(array.shape)
  transpose = [rank - 1] + list(range(0, rank - 1))
  return array.transpose(transpose)


def _mixture_stddev_np(pi_vector, mu_vector, sigma_vector):
  """Computes the standard deviation of a univariate mixture distribution.

  Acts upon `np.array`s (not `tf.Tensor`s).

  Args:
    pi_vector: A `np.array` of mixture weights. Shape `[batch, components]`.
    mu_vector: A `np.array` of means. Shape `[batch, components]`
    sigma_vector: A `np.array` of stddevs. Shape `[batch, components]`.

  Returns:
    A `np.array` containing the batch of standard deviations.
  """
  pi_vector = np.expand_dims(pi_vector, axis=1)
  mean_wa = np.matmul(pi_vector, np.expand_dims(mu_vector, axis=2))
  var_wa = np.matmul(pi_vector, np.expand_dims(sigma_vector**2, axis=2))
  mid_term = np.matmul(pi_vector, np.expand_dims(mu_vector**2, axis=2))
  mixture_variance = (
      np.squeeze(var_wa) + np.squeeze(mid_term) - np.squeeze(mean_wa**2))
  return np.sqrt(mixture_variance)


@contextlib.contextmanager
def _test_capture_mvndiag_sample_outputs():
  """Use monkey-patching to capture the output of an MVNDiag sample."""
  data_container = []
  true_mvndiag_sample = tfd.MultivariateNormalDiag.sample

  def _capturing_mvndiag_sample(
      self, sample_shape=(), seed=None, name='sample', **kwargs):
    samples = true_mvndiag_sample(self, sample_shape, seed, name, **kwargs)
    data_container.append(samples)
    return samples

  tfd.MultivariateNormalDiag.sample = _capturing_mvndiag_sample
  yield data_container
  tfd.MultivariateNormalDiag.sample = true_mvndiag_sample


@contextlib.contextmanager
def _test_capture_normal_sample_outputs():
  """Use monkey-patching to capture the output of an Normal sample."""
  data_container = []
  true_normal_sample = tfd.Normal.sample

  def _capturing_normal_sample(
      self, sample_shape=(), seed=None, name='sample', **kwargs):
    samples = true_normal_sample(self, sample_shape, seed, name, **kwargs)
    data_container.append(samples)
    return samples

  tfd.Normal.sample = _capturing_normal_sample
  yield data_container
  tfd.Normal.sample = true_normal_sample


def make_univariate_mixture(batch_shape, num_components, use_static_graph):
  batch_shape = tf.convert_to_tensor(value=batch_shape, dtype=tf.int32)
  seed_stream = test_util.test_seed_stream('univariate_mixture')
  logits = -50. + tf.random.uniform(
      tf.concat((batch_shape, [num_components]), axis=0),
      -1, 1, dtype=tf.float32, seed=seed_stream())
  components = [
      tfd.Normal(loc=tf.random.normal(batch_shape, seed=seed_stream()),
                 scale=10 * tf.random.uniform(batch_shape, seed=seed_stream()))
      for _ in range(num_components)
  ]
  cat = tfd.Categorical(logits, dtype=tf.int32)
  return tfd.Mixture(
      cat, components, use_static_graph=use_static_graph, validate_args=True)


def make_multivariate_mixture(batch_shape, num_components, event_shape,
                              use_static_graph, batch_shape_tensor=None):
  if batch_shape_tensor is None:
    batch_shape_tensor = batch_shape
  batch_shape_tensor = tf.convert_to_tensor(
      value=batch_shape_tensor, dtype=tf.int32)
  seed_stream = test_util.test_seed_stream('multivariate_mixture')
  logits = -50. + tf.random.uniform(
      tf.concat((batch_shape_tensor, [num_components]), 0),
      -1, 1, dtype=tf.float32, seed=seed_stream())
  tensorshape_util.set_shape(
      logits, tensorshape_util.concatenate(batch_shape, num_components))
  static_batch_and_event_shape = (
      tf.TensorShape(batch_shape).concatenate(event_shape))
  event_shape = tf.convert_to_tensor(value=event_shape, dtype=tf.int32)
  batch_and_event_shape = tf.concat((batch_shape_tensor, event_shape), 0)

  def create_component():
    loc = tf.random.normal(batch_and_event_shape, seed=seed_stream())
    scale_diag = 10 * tf.random.uniform(
        batch_and_event_shape, seed=seed_stream())
    tensorshape_util.set_shape(loc, static_batch_and_event_shape)
    tensorshape_util.set_shape(scale_diag, static_batch_and_event_shape)
    return tfd.MultivariateNormalDiag(
        loc=loc, scale_diag=scale_diag, validate_args=True)

  components = [create_component() for _ in range(num_components)]
  cat = tfd.Categorical(logits, dtype=tf.int32)
  return tfd.Mixture(
      cat, components, use_static_graph=use_static_graph, validate_args=True)


@test_util.test_all_tf_execution_regimes
class MixtureTest(test_util.TestCase):
  use_static_graph = False

  def testShapes(self):
    for batch_shape in ([], [1], [2, 3, 4]):
      dist = make_univariate_mixture(
          batch_shape,
          num_components=10,
          use_static_graph=self.use_static_graph)
      self.assertAllEqual(batch_shape, dist.batch_shape)
      self.assertAllEqual(batch_shape, self.evaluate(dist.batch_shape_tensor()))
      self.assertAllEqual([], dist.event_shape)
      self.assertAllEqual([], self.evaluate(dist.event_shape_tensor()))

      for event_shape in ([1], [2]):
        dist = make_multivariate_mixture(
            batch_shape,
            num_components=10,
            event_shape=event_shape,
            use_static_graph=self.use_static_graph)
        self.assertAllEqual(batch_shape, dist.batch_shape)
        self.assertAllEqual(
            batch_shape, self.evaluate(dist.batch_shape_tensor()))
        self.assertAllEqual(event_shape, dist.event_shape)
        self.assertAllEqual(event_shape,
                            self.evaluate(dist.event_shape_tensor()))

  def testBrokenShapesStatic(self):
    with self.assertRaisesWithPredicateMatch(ValueError,
                                             r'cat.num_classes != len'):
      tfd.Mixture(
          tfd.Categorical([0.1, 0.5]),  # 2 classes
          [tfd.Normal(loc=1.0, scale=2.0)],
          use_static_graph=self.use_static_graph,
          validate_args=True)
    with self.assertRaisesWithPredicateMatch(
        ValueError, r'components\[1\] batch shape must be compatible'):
      # The value error is raised because the batch shapes of the
      # Normals are not equal.  One is a scalar, the other is a
      # vector of size (2,).
      tfd.Mixture(
          tfd.Categorical([-0.5, 0.5]),  # scalar batch
          [
              tfd.Normal(loc=1.0, scale=2.0),  # scalar dist
              tfd.Normal(loc=[1.0, 1.0], scale=[2.0, 2.0])
          ],
          use_static_graph=self.use_static_graph,
          validate_args=True)

  @test_util.jax_disable_test_missing_functionality(
      'Shapes are statically known in JAX.')
  def testBrokenShapeUnknownCategories(self):
    with self.assertRaisesWithPredicateMatch(ValueError, r'Could not infer'):
      cat_logits = tf.Variable([[13., 19.]], shape=[1, None], dtype=tf.float32)
      tfd.Mixture(
          tfd.Categorical(cat_logits), [tfd.Normal(loc=[1.0], scale=[2.0])],
          use_static_graph=self.use_static_graph,
          validate_args=True)

  @test_util.jax_disable_test_missing_functionality(
      'Shapes are statically known in JAX.')
  def testBrokenShapesDynamic(self):
    d0_param = tf.Variable([2., 3], shape=tf.TensorShape(None))
    d1_param = tf.Variable([1.], shape=tf.TensorShape(None))

    self.evaluate([d0_param.initializer, d1_param.initializer])

    with self.assertRaisesWithPredicateMatch(
        tf.errors.InvalidArgumentError, 'batch shape must match cat'):
      d = tfd.Mixture(
          tfd.Categorical([0.1, 0.2]), [
              tfd.Normal(loc=d0_param, scale=d0_param),
              tfd.Normal(loc=d1_param, scale=d1_param)
          ],
          validate_args=True,
          use_static_graph=self.use_static_graph)
      self.evaluate([d.sample(seed=test_util.test_seed())])

  def testBrokenTypes(self):
    with self.assertRaisesWithPredicateMatch(TypeError, 'Categorical'):
      tfd.Mixture(
          None, [], use_static_graph=self.use_static_graph, validate_args=True)
    cat = tfd.Categorical([0.3, 0.2])
    # components must be a list of distributions
    with self.assertRaisesWithPredicateMatch(
        TypeError, 'all .* must be Distribution instances'):
      tfd.Mixture(
          cat, [None],
          use_static_graph=self.use_static_graph,
          validate_args=True)
    with self.assertRaisesWithPredicateMatch(TypeError, 'same dtype'):
      tfd.Mixture(
          cat, [
              tfd.Normal(loc=[1.0], scale=[2.0]),
              tfd.Normal(loc=[np.float16(1.0)], scale=[np.float16(2.0)]),
          ],
          use_static_graph=self.use_static_graph,
          validate_args=True)
    with self.assertRaisesWithPredicateMatch(ValueError, 'non-empty list'):
      tfd.Mixture(
          tfd.Categorical([0.3, 0.2]),
          None,
          use_static_graph=self.use_static_graph,
          validate_args=True)

    # TODO(ebrevdo): once distribution Domains have been added, add a
    # test to ensure that the domains of the distributions in a
    # mixture are checked for equivalence.

  def testMeanUnivariate(self):
    for batch_shape in ((), (2,), (2, 3)):
      dist = make_univariate_mixture(
          batch_shape=batch_shape,
          num_components=2,
          use_static_graph=self.use_static_graph)
      mean = dist.mean()
      self.assertEqual(batch_shape, mean.shape)

      cat_probs = tf.math.softmax(dist.cat.logits)
      dist_means = [d.mean() for d in dist.components]

      mean_value, cat_probs_value, dist_means_value = self.evaluate(
          [mean, cat_probs, dist_means])
      self.assertEqual(batch_shape, mean_value.shape)

      cat_probs_value = _swap_first_last_axes(cat_probs_value)
      true_mean = sum(
          [c_p * m for (c_p, m) in zip(cat_probs_value, dist_means_value)])

      self.assertAllClose(true_mean, mean_value)

  def testMeanMultivariate(self):
    for batch_shape in ((), (2,), (2, 3)):
      dist = make_multivariate_mixture(
          batch_shape=batch_shape,
          num_components=2,
          event_shape=(4,),
          use_static_graph=self.use_static_graph)
      mean = dist.mean()
      self.assertEqual(batch_shape + (4,), mean.shape)

      cat_probs = tf.math.softmax(dist.cat.logits)
      dist_means = [d.mean() for d in dist.components]

      mean_value, cat_probs_value, dist_means_value = self.evaluate(
          [mean, cat_probs, dist_means])
      self.assertEqual(batch_shape + (4,), mean_value.shape)

      cat_probs_value = _swap_first_last_axes(cat_probs_value)

      # Add a new innermost dimension for broadcasting to mvn vector shape
      cat_probs_value = [np.expand_dims(c_p, -1) for c_p in cat_probs_value]

      true_mean = sum(
          [c_p * m for (c_p, m) in zip(cat_probs_value, dist_means_value)])

      self.assertAllClose(true_mean, mean_value)

  def testStddevShapeUnivariate(self):
    num_components = 2
    # This is the same shape test which is done in 'testMeanUnivariate'.
    for batch_shape in ((), (2,), (2, 3)):
      dist = make_univariate_mixture(
          batch_shape=batch_shape,
          num_components=num_components,
          use_static_graph=self.use_static_graph)
      dev = dist.stddev()
      self.assertEqual(batch_shape, dev.shape)

      cat_probs = tf.math.softmax(dist.cat.logits)
      dist_devs = [d.stddev() for d in dist.components]
      dist_means = [d.mean() for d in dist.components]

      res = self.evaluate([dev, cat_probs, dist_devs, dist_means])
      dev_value, cat_probs_values, dist_devs_values, dist_means_values = res
      # Manual computation of stddev.
      batch_shape_res = cat_probs_values.shape[:-1]
      event_shape_res = dist_devs_values[0].shape[len(batch_shape_res):]
      stacked_mean_res = np.stack(dist_means_values, -1)
      stacked_dev_res = np.stack(dist_devs_values, -1)

      # Broadcast cat probs over event dimensions.
      for _ in range(len(event_shape_res)):
        cat_probs_values = np.expand_dims(cat_probs_values, len(batch_shape))
      cat_probs_values = cat_probs_values + np.zeros_like(stacked_dev_res)  # pylint: disable=g-no-augmented-assignment

      # Perform stddev computation on a flattened batch.
      flat_batch_manual_dev = _mixture_stddev_np(
          np.reshape(cat_probs_values, [-1, num_components]),
          np.reshape(stacked_mean_res, [-1, num_components]),
          np.reshape(stacked_dev_res, [-1, num_components]))

      # Reshape to full shape.
      full_shape_res = list(batch_shape_res) + list(event_shape_res)
      manual_dev = np.reshape(flat_batch_manual_dev, full_shape_res)
      self.assertEqual(batch_shape, dev_value.shape)
      self.assertAllClose(manual_dev, dev_value)

  def testStddevShapeMultivariate(self):
    num_components = 2

    # This is the same shape test which is done in 'testMeanMultivariate'.
    for batch_shape in ((), (2,), (2, 3)):
      dist = make_multivariate_mixture(
          batch_shape=batch_shape,
          num_components=num_components,
          event_shape=(4,),
          use_static_graph=self.use_static_graph)
      dev = dist.stddev()
      self.assertEqual(batch_shape + (4,), dev.shape)

      cat_probs = tf.math.softmax(dist.cat.logits)
      dist_devs = [d.stddev() for d in dist.components]
      dist_means = [d.mean() for d in dist.components]

      res = self.evaluate([dev, cat_probs, dist_devs, dist_means])
      dev_value, cat_probs_values, dist_devs_values, dist_means_values = res
      # Manual computation of stddev.
      batch_shape_res = cat_probs_values.shape[:-1]
      event_shape_res = dist_devs_values[0].shape[len(batch_shape_res):]
      stacked_mean_res = np.stack(dist_means_values, -1)
      stacked_dev_res = np.stack(dist_devs_values, -1)

      # Broadcast cat probs over event dimensions.
      for _ in range(len(event_shape_res)):
        cat_probs_values = np.expand_dims(cat_probs_values, len(batch_shape))
      cat_probs_values = cat_probs_values + np.zeros_like(stacked_dev_res)  # pylint: disable=g-no-augmented-assignment

      # Perform stddev computation on a flattened batch.
      flat_batch_manual_dev = _mixture_stddev_np(
          np.reshape(cat_probs_values, [-1, num_components]),
          np.reshape(stacked_mean_res, [-1, num_components]),
          np.reshape(stacked_dev_res, [-1, num_components]))

      # Reshape to full shape.
      full_shape_res = list(batch_shape_res) + list(event_shape_res)
      manual_dev = np.reshape(flat_batch_manual_dev, full_shape_res)
      self.assertEqual(tuple(full_shape_res), dev_value.shape)
      self.assertAllClose(manual_dev, dev_value)

  def testSpecificStddevValue(self):
    # TODO(b/135281612): Remove explicit float32 casting.
    cat_probs = np.float32([0.5, 0.5])
    component_means = np.float32([-10, 0.1])
    component_devs = np.float32([0.05, 2.33])
    ground_truth_stddev = 5.3120805

    mixture_dist = tfd.Mixture(
        cat=tfd.Categorical(probs=cat_probs),
        components=[
            tfd.Normal(loc=component_means[0], scale=component_devs[0]),
            tfd.Normal(loc=component_means[1], scale=component_devs[1]),
        ],
        use_static_graph=self.use_static_graph,
        validate_args=True)
    mix_dev = mixture_dist.stddev()
    actual_stddev = self.evaluate(mix_dev)
    self.assertAllClose(actual_stddev, ground_truth_stddev)

  def testProbScalarUnivariate(self):
    dist = make_univariate_mixture(
        batch_shape=[],
        num_components=2,
        use_static_graph=self.use_static_graph)
    for x in [
        np.array([1.0, 2.0], dtype=np.float32),
        np.array(1.0, dtype=np.float32),
        np.random.randn(3, 4).astype(np.float32)
    ]:
      p_x = dist.prob(x)

      self.assertEqual(x.shape, p_x.shape)
      cat_probs = tf.math.softmax([dist.cat.logits])[0]
      dist_probs = [d.prob(x) for d in dist.components]

      p_x_value, cat_probs_value, dist_probs_value = self.evaluate(
          [p_x, cat_probs, dist_probs])
      self.assertEqual(x.shape, p_x_value.shape)

      total_prob = sum(
          c_p_value * d_p_value
          for (c_p_value, d_p_value) in zip(cat_probs_value, dist_probs_value))

      self.assertAllClose(total_prob, p_x_value)

  def testProbScalarMultivariate(self):
    dist = make_multivariate_mixture(
        batch_shape=[],
        num_components=2,
        event_shape=[3],
        use_static_graph=self.use_static_graph)
    for x in [
        np.array([[-1.0, 0.0, 1.0], [0.5, 1.0, -0.3]], dtype=np.float32),
        np.array([-1.0, 0.0, 1.0], dtype=np.float32),
        np.random.randn(2, 2, 3).astype(np.float32)
    ]:
      p_x = dist.prob(x)

      self.assertEqual(x.shape[:-1], p_x.shape)

      cat_probs = tf.math.softmax([dist.cat.logits])[0]
      dist_probs = [d.prob(x) for d in dist.components]

      p_x_value, cat_probs_value, dist_probs_value = self.evaluate(
          [p_x, cat_probs, dist_probs])

      self.assertEqual(x.shape[:-1], p_x_value.shape)

      total_prob = sum(
          c_p_value * d_p_value
          for (c_p_value, d_p_value) in zip(cat_probs_value, dist_probs_value))

      self.assertAllClose(total_prob, p_x_value)

  def testProbBatchUnivariate(self):
    dist = make_univariate_mixture(
        batch_shape=[2, 3],
        num_components=2,
        use_static_graph=self.use_static_graph)

    for x in [
        np.random.randn(2, 3).astype(np.float32),
        np.random.randn(4, 2, 3).astype(np.float32)
    ]:
      p_x = dist.prob(x)
      self.assertEqual(x.shape, p_x.shape)

      cat_probs = tf.math.softmax(dist.cat.logits)
      dist_probs = [d.prob(x) for d in dist.components]

      p_x_value, cat_probs_value, dist_probs_value = self.evaluate(
          [p_x, cat_probs, dist_probs])
      self.assertEqual(x.shape, p_x_value.shape)

      cat_probs_value = _swap_first_last_axes(cat_probs_value)

      total_prob = sum(
          c_p_value * d_p_value
          for (c_p_value, d_p_value) in zip(cat_probs_value, dist_probs_value))

      self.assertAllClose(total_prob, p_x_value)

  def testProbBatchMultivariate(self):
    dist = make_multivariate_mixture(
        batch_shape=[2, 3],
        num_components=2,
        event_shape=[4],
        use_static_graph=self.use_static_graph)

    for x in [
        np.random.randn(2, 3, 4).astype(np.float32),
        np.random.randn(4, 2, 3, 4).astype(np.float32)
    ]:
      p_x = dist.prob(x)
      self.assertEqual(x.shape[:-1], p_x.shape)

      cat_probs = tf.math.softmax(dist.cat.logits)
      dist_probs = [d.prob(x) for d in dist.components]

      p_x_value, cat_probs_value, dist_probs_value = self.evaluate(
          [p_x, cat_probs, dist_probs])
      self.assertEqual(x.shape[:-1], p_x_value.shape)

      cat_probs_value = _swap_first_last_axes(cat_probs_value)
      total_prob = sum(
          c_p_value * d_p_value
          for (c_p_value, d_p_value) in zip(cat_probs_value, dist_probs_value))

      self.assertAllClose(total_prob, p_x_value)

  def testSampleScalarBatchUnivariate(self):
    num_components = 3
    batch_shape = []
    dist = make_univariate_mixture(
        batch_shape=batch_shape,
        num_components=num_components,
        use_static_graph=self.use_static_graph)
    n = 4
    with _test_capture_normal_sample_outputs() as component_samples:
      samples = dist.sample(n, seed=test_util.test_seed())
    self.assertEqual(samples.dtype, tf.float32)
    self.assertEqual((4,), samples.shape)
    cat_samples = dist.cat.sample(n, seed=test_util.test_seed())
    sample_values, cat_sample_values, dist_sample_values = self.evaluate(
        [samples, cat_samples, component_samples])
    self.assertEqual((4,), sample_values.shape)

    for c in range(num_components):
      which_c = np.where(cat_sample_values == c)[0]
      size_c = which_c.size
      # Scalar Batch univariate case: batch_size == 1, rank 1
      if self.use_static_graph:
        which_dist_samples = dist_sample_values[c][which_c]
      else:
        which_dist_samples = dist_sample_values[c][:size_c]
      self.assertAllClose(which_dist_samples, sample_values[which_c])

  # Test that sampling with the same seed twice gives the same results.
  def testSampleMultipleTimes(self):
    # 5 component mixture.
    logits = [-10.0, -5.0, 0.0, 5.0, 10.0]
    mus = [-5.0, 0.0, 5.0, 4.0, 20.0]
    sigmas = [0.1, 5.0, 3.0, 0.2, 4.0]

    n = 100
    seed = test_util.test_seed()
    components = [
        tfd.Normal(loc=mu, scale=sigma) for mu, sigma in zip(mus, sigmas)
    ]
    cat = tfd.Categorical(logits, dtype=tf.int32, name='cat1')
    dist1 = tfd.Mixture(
        cat,
        components,
        name='mixture1',
        use_static_graph=self.use_static_graph,
        validate_args=True)
    tf.random.set_seed(seed)
    samples1 = self.evaluate(dist1.sample(n, seed=seed))

    components2 = [
        tfd.Normal(loc=mu, scale=sigma) for mu, sigma in zip(mus, sigmas)
    ]
    cat2 = tfd.Categorical(logits, dtype=tf.int32, name='cat2')
    dist2 = tfd.Mixture(
        cat2,
        components2,
        name='mixture2',
        use_static_graph=self.use_static_graph,
        validate_args=True)
    tf.random.set_seed(seed)
    samples2 = self.evaluate(dist2.sample(n, seed=seed))

    self.assertAllClose(samples1, samples2)

  def testSampleScalarBatchMultivariate(self):
    num_components = 3
    dist = make_multivariate_mixture(
        batch_shape=[],
        num_components=num_components,
        event_shape=[2],
        use_static_graph=self.use_static_graph)
    n = 4
    seed = test_util.test_seed()
    with _test_capture_mvndiag_sample_outputs() as component_samples:
      tf.random.set_seed(seed)
      samples = dist.sample(n, seed=seed)
    self.assertEqual(samples.dtype, tf.float32)
    self.assertEqual((4, 2), samples.shape)
    tf.random.set_seed(seed)
    cat_samples = dist.cat.sample(n, seed=seed)
    sample_values, cat_sample_values, dist_sample_values = self.evaluate(
        [samples, cat_samples, component_samples])
    self.assertEqual((4, 2), sample_values.shape)
    for c in range(num_components):
      which_c = np.where(cat_sample_values == c)[0]
      size_c = which_c.size
      # Scalar Batch multivariate case: batch_size == 1, rank 2
      if self.use_static_graph:
        which_dist_samples = dist_sample_values[c][which_c, :]
      else:
        which_dist_samples = dist_sample_values[c][:size_c, :]
      self.assertAllClose(which_dist_samples, sample_values[which_c, :])

  def testSampleBatchUnivariate(self):
    num_components = 3
    dist = make_univariate_mixture(
        batch_shape=[2, 3],
        num_components=num_components,
        use_static_graph=self.use_static_graph)
    n = 4
    seed = test_util.test_seed()
    with _test_capture_normal_sample_outputs() as component_samples:
      tf.random.set_seed(seed)
      samples = dist.sample(n, seed=seed)
    self.assertEqual(samples.dtype, tf.float32)
    self.assertEqual((4, 2, 3), samples.shape)
    tf.random.set_seed(seed)
    cat_samples = dist.cat.sample(n, seed=seed)
    sample_values, cat_sample_values, dist_sample_values = self.evaluate(
        [samples, cat_samples, component_samples])
    self.assertEqual((4, 2, 3), sample_values.shape)
    for c in range(num_components):
      which_c_s, which_c_b0, which_c_b1 = np.where(cat_sample_values == c)
      size_c = which_c_s.size
      # Batch univariate case: batch_size == [2, 3], rank 3
      if self.use_static_graph:
        which_dist_samples = dist_sample_values[c][which_c_s, which_c_b0,
                                                   which_c_b1]
      else:
        which_dist_samples = dist_sample_values[c][range(size_c), which_c_b0,
                                                   which_c_b1]
      self.assertAllClose(which_dist_samples,
                          sample_values[which_c_s, which_c_b0, which_c_b1])

  def _testSampleBatchMultivariate(self, fully_known_batch_shape):
    num_components = 3
    if fully_known_batch_shape:
      batch_shape = [2, 3]
      batch_shape_tensor = [2, 3]
    else:
      batch_shape = [None, 3]
      batch_shape_tensor = tf1.placeholder_with_default([2, 3], shape=[2])

    dist = make_multivariate_mixture(
        batch_shape=batch_shape,
        num_components=num_components,
        event_shape=[4],
        batch_shape_tensor=batch_shape_tensor,
        use_static_graph=self.use_static_graph)
    n = 5
    seed = test_util.test_seed()
    with _test_capture_mvndiag_sample_outputs() as component_samples:
      tf.random.set_seed(seed)
      samples = dist.sample(n, seed=seed)
    self.assertEqual(samples.dtype, tf.float32)
    if fully_known_batch_shape:
      self.assertEqual((5, 2, 3, 4), samples.shape)
    else:
      self.assertEqual([5, None, 3, 4], tensorshape_util.as_list(samples.shape))
    tf.random.set_seed(seed)
    cat_samples = dist.cat.sample(n, seed=seed)
    sample_values, cat_sample_values, dist_sample_values = self.evaluate(
        [samples, cat_samples, component_samples])
    self.assertEqual((5, 2, 3, 4), sample_values.shape)

    for c in range(num_components):
      which_c_s, which_c_b0, which_c_b1 = np.where(cat_sample_values == c)
      size_c = which_c_s.size
      # Batch univariate case: batch_size == [2, 3], rank 4 (multivariate)
      if self.use_static_graph:
        which_dist_samples = dist_sample_values[c][which_c_s, which_c_b0,
                                                   which_c_b1, :]
      else:
        which_dist_samples = dist_sample_values[c][range(size_c), which_c_b0,
                                                   which_c_b1, :]
      self.assertAllClose(which_dist_samples,
                          sample_values[which_c_s, which_c_b0, which_c_b1, :])

  def testSampleBatchMultivariateFullyKnownBatchShape(self):
    self._testSampleBatchMultivariate(fully_known_batch_shape=True)

  def testSampleBatchMultivariateNotFullyKnownBatchShape(self):
    # In eager mode the batch shape is always known so we
    # can return immediately
    if tf.executing_eagerly():
      return
    self._testSampleBatchMultivariate(fully_known_batch_shape=False)

  def testEntropyLowerBoundMultivariate(self):
    for batch_shape in ((), (2,), (2, 3)):
      dist = make_multivariate_mixture(
          batch_shape=batch_shape,
          num_components=2,
          event_shape=(4,),
          use_static_graph=self.use_static_graph)
      entropy_lower_bound = dist.entropy_lower_bound()
      self.assertEqual(batch_shape, entropy_lower_bound.shape)

      cat_probs = tf.math.softmax(dist.cat.logits)
      dist_entropy = [d.entropy() for d in dist.components]

      entropy_lower_bound_value, cat_probs_value, dist_entropy_value = (
          self.evaluate([entropy_lower_bound, cat_probs, dist_entropy]))
      self.assertEqual(batch_shape, entropy_lower_bound_value.shape)

      cat_probs_value = _swap_first_last_axes(cat_probs_value)

      # entropy_lower_bound = sum_i pi_i entropy_i
      # for i in num_components, batchwise.
      true_entropy_lower_bound = sum(
          [c_p * m for (c_p, m) in zip(cat_probs_value, dist_entropy_value)])

      self.assertAllClose(true_entropy_lower_bound, entropy_lower_bound_value)

  def testCdfScalarUnivariate(self):
    """Tests CDF against scipy for a mixture of seven gaussians."""
    # Construct a mixture of gaussians with seven components.
    n_components = 7

    # pre-softmax mixture probabilities.
    mixture_weight_logits = np.random.uniform(
        low=-1, high=1, size=(n_components,)).astype(np.float32)

    def _scalar_univariate_softmax(x):
      e_x = np.exp(x - np.max(x))
      return e_x / e_x.sum()

    # Construct the tfd.Mixture object.
    mixture_weights = _scalar_univariate_softmax(mixture_weight_logits)
    means = [np.random.uniform(low=-10, high=10, size=()).astype(np.float32)
             for _ in range(n_components)]
    sigmas = [np.ones(shape=(), dtype=np.float32) for _ in range(n_components)]
    cat_tf = tfd.Categorical(probs=mixture_weights)
    components_tf = [
        tfd.Normal(loc=mu, scale=sigma) for (mu, sigma) in zip(means, sigmas)
    ]
    mixture_tf = tfd.Mixture(
        cat=cat_tf,
        components=components_tf,
        use_static_graph=self.use_static_graph,
        validate_args=True)

    # These are two test cases to verify.
    xs_to_check = [
        np.array(1.0, dtype=np.float32),
        np.array(np.random.randn()).astype(np.float32)
    ]

    for x_tensor in xs_to_check:
      with self.cached_session() as sess:
        x_cdf_tf_result, x_log_cdf_tf_result = sess.run(
            [mixture_tf.cdf(x_tensor), mixture_tf.log_cdf(x_tensor)])

        # Compute the cdf with scipy.
        scipy_component_cdfs = [stats.norm.cdf(x=x_tensor, loc=mu, scale=sigma)
                                for (mu, sigma) in zip(means, sigmas)]
        scipy_cdf_result = np.dot(mixture_weights,
                                  np.array(scipy_component_cdfs))
        self.assertAllClose(x_cdf_tf_result, scipy_cdf_result)
        self.assertAllClose(np.exp(x_log_cdf_tf_result), scipy_cdf_result)

  def testCdfBatchUnivariate(self):
    """Tests against scipy for a (batch of) mixture(s) of seven gaussians."""
    n_components = 7
    batch_size = 5
    mixture_weight_logits = np.random.uniform(
        low=-1, high=1, size=(batch_size, n_components)).astype(np.float32)

    def _batch_univariate_softmax(x):
      e_x = np.exp(x)
      e_x_sum = np.expand_dims(np.sum(e_x, axis=1), axis=1)
      return e_x / np.tile(e_x_sum, reps=[1, x.shape[1]])

    psize = (batch_size,)
    mixture_weights = _batch_univariate_softmax(mixture_weight_logits)
    means = [np.random.uniform(low=-10, high=10, size=psize).astype(np.float32)
             for _ in range(n_components)]
    sigmas = [np.ones(shape=psize, dtype=np.float32)
              for _ in range(n_components)]
    cat_tf = tfd.Categorical(probs=mixture_weights)
    components_tf = [
        tfd.Normal(loc=mu, scale=sigma) for (mu, sigma) in zip(means, sigmas)
    ]
    mixture_tf = tfd.Mixture(
        cat=cat_tf,
        components=components_tf,
        use_static_graph=self.use_static_graph,
        validate_args=True)

    xs_to_check = [
        np.array([1.0, 5.9, -3, 0.0, 0.0], dtype=np.float32),
        np.random.randn(batch_size).astype(np.float32)
    ]

    for x_tensor in xs_to_check:
      with self.cached_session() as sess:
        x_cdf_tf_result, x_log_cdf_tf_result = sess.run(
            [mixture_tf.cdf(x_tensor), mixture_tf.log_cdf(x_tensor)])

        # Compute the cdf with scipy.
        scipy_component_cdfs = [stats.norm.cdf(x=x_tensor, loc=mu, scale=sigma)
                                for (mu, sigma) in zip(means, sigmas)]
        weights_and_cdfs = zip(np.transpose(mixture_weights, axes=[1, 0]),
                               scipy_component_cdfs)
        final_cdf_probs_per_component = [
            np.multiply(c_p_value, d_cdf_value)
            for (c_p_value, d_cdf_value) in weights_and_cdfs]
        scipy_cdf_result = np.sum(final_cdf_probs_per_component, axis=0)
        self.assertAllClose(x_cdf_tf_result, scipy_cdf_result)
        self.assertAllClose(np.exp(x_log_cdf_tf_result), scipy_cdf_result)

  def testSampleBimixGamma(self):
    """Tests a bug in the underlying tfd.Gamma op.

    Mixture's use of dynamic partition requires `random_gamma` correctly returns
    an empty `Tensor`.
    """
    gm = tfd.Mixture(
        cat=tfd.Categorical(probs=[.3, .7]),
        components=[tfd.Gamma(1., 2.), tfd.Gamma(2., 1.)],
        use_static_graph=self.use_static_graph,
        validate_args=True)
    x_ = self.evaluate(gm.sample(seed=test_util.test_seed()))
    self.assertAllEqual([], x_.shape)

  @test_util.tf_tape_safety_test
  def testGradientsThroughParams(self):
    logits = tf.Variable(np.zeros((3, 5, 2)), dtype=tf.float32,
                         shape=tf.TensorShape([None, None, 2]))
    concentration = tf.Variable(np.ones((3, 5, 4)), dtype=tf.float32,
                                shape=tf.TensorShape(None))
    loc = tf.Variable(np.zeros((3, 5, 4)), dtype=tf.float32,
                      shape=tf.TensorShape(None))
    scale = tf.Variable(1., dtype=tf.float32, shape=tf.TensorShape(None))

    dist = tfd.Mixture(
        tfd.Categorical(logits=logits),
        components=[tfd.Dirichlet(concentration),
                    tfd.MultivariateNormalDiag(
                        loc=loc, scale_identity_multiplier=scale)],
        use_static_graph=self.use_static_graph,
        validate_args=True)

    with tf.GradientTape() as tape:
      loss = tf.reduce_sum(dist.log_prob(tf.ones((3, 5, 4)) / 4.))
    grad = tape.gradient(loss, dist.trainable_variables)
    self.assertLen(grad, 4)
    self.assertAllNotNone(grad)

  def testExcessiveConcretizationOfParams(self):
    logits = tfp_hps.defer_and_count_usage(
        tf.Variable(np.zeros((3, 5, 2)), dtype=tf.float32,
                    shape=tf.TensorShape([None, None, 2]), name='logits'))
    concentration = tfp_hps.defer_and_count_usage(
        tf.Variable(np.ones((3, 5, 4)), dtype=tf.float32,
                    shape=tf.TensorShape(None), name='concentration'))
    loc = tfp_hps.defer_and_count_usage(
        tf.Variable(np.zeros((3, 5, 4)), dtype=tf.float32,
                    shape=tf.TensorShape(None), name='loc'))
    scale = tfp_hps.defer_and_count_usage(
        tf.Variable(1., dtype=tf.float32,
                    shape=tf.TensorShape(None), name='scale'))

    dist = tfd.Mixture(
        tfd.Categorical(logits=logits),
        components=[tfd.Dirichlet(concentration),
                    tfd.Independent(tfd.Normal(loc=loc, scale=scale),
                                    reinterpreted_batch_ndims=1)],
        use_static_graph=self.use_static_graph,
        validate_args=True)

    for method in ('batch_shape_tensor', 'event_shape_tensor',
                   'entropy_lower_bound'):
      with tfp_hps.assert_no_excessive_var_usage(method, max_permissible=2):
        getattr(dist, method)()

    with tfp_hps.assert_no_excessive_var_usage('sample', max_permissible=2):
      dist.sample(seed=test_util.test_seed())

    for method in ('prob', 'log_prob'):
      with tfp_hps.assert_no_excessive_var_usage('method', max_permissible=2):
        getattr(dist, method)(tf.ones((3, 5, 4)) / 4.)

    # TODO(b/140579567): The `stddev()` and `variance()` methods require
    # calling both:
    #  - `self.components[i].mean()`
    #  - `self.components[i].stddev()`
    # Thus, these methods incur an additional concretization (or two if
    # `validate_args=True` for `self.components[i]`).

    for method in ('stddev', 'variance'):
      with tfp_hps.assert_no_excessive_var_usage(method, max_permissible=3):
        getattr(dist, method)()


class MixtureStaticSampleTest(MixtureTest):
  use_static_graph = True


class MixtureBenchmark(tf.test.Benchmark):
  use_static_graph = False

  def _runSamplingBenchmark(self, name, create_distribution, use_gpu,
                            num_components, batch_size, num_features,
                            sample_size):
    config = tf1.ConfigProto()
    config.allow_soft_placement = True
    np.random.seed(127)
    with tf1.Session(config=config, graph=tf.Graph()) as sess:
      tf.random.set_seed(0)
      with tf.device('/device:GPU:0' if use_gpu else '/cpu:0'):
        mixture = create_distribution(
            num_components=num_components,
            batch_size=batch_size,
            num_features=num_features)
        sample_op = mixture.sample(sample_size, seed=test_util.test_seed()).op
        sess.run(tf1.global_variables_initializer())
        reported = self.run_op_benchmark(
            sess,
            sample_op,
            min_iters=10,
            name=('%s_%s_components_%d_batch_%d_features_%d_sample_%d' %
                  (name, use_gpu, num_components, batch_size, num_features,
                   sample_size)))
        tf1.logging.vlog(
            2, '\t'.join(['%s', '%d', '%d', '%d', '%d', '%g']) %
            (use_gpu, num_components, batch_size, num_features, sample_size,
             reported['wall_time']))

  def benchmarkSamplingMVNDiag(self):
    tf1.logging.vlog(
        2, 'mvn_diag\tuse_gpu\tcomponents\tbatch\tfeatures\tsample\twall_time')

    def create_distribution(batch_size, num_components, num_features):
      cat = tfd.Categorical(logits=np.random.randn(batch_size, num_components))
      mus = [
          tf.Variable(np.random.randn(batch_size, num_features))
          for _ in range(num_components)
      ]
      sigmas = [
          tf.Variable(np.random.rand(batch_size, num_features))
          for _ in range(num_components)
      ]
      components = list(
          tfd.MultivariateNormalDiag(loc=mu, scale_diag=sigma)
          for (mu, sigma) in zip(mus, sigmas))
      return tfd.Mixture(
          cat,
          components,
          use_static_graph=self.use_static_graph,
          validate_args=True)

    for use_gpu in False, True:
      if use_gpu and not tf.test.is_gpu_available():
        continue
      for num_components in 1, 8, 16:
        for batch_size in 1, 32:
          for num_features in 1, 64, 512:
            for sample_size in 1, 32, 128:
              self._runSamplingBenchmark(
                  'mvn_diag',
                  create_distribution=create_distribution,
                  use_gpu=use_gpu,
                  num_components=num_components,
                  batch_size=batch_size,
                  num_features=num_features,
                  sample_size=sample_size)

  def benchmarkSamplingMVNFull(self):
    tf1.logging.vlog(
        2, 'mvn_full\tuse_gpu\tcomponents\tbatch\tfeatures\tsample\twall_time')

    def psd(x):
      """Construct batch-wise PSD matrices."""
      return np.stack([np.dot(np.transpose(z), z) for z in x])

    def create_distribution(batch_size, num_components, num_features):
      cat = tfd.Categorical(logits=np.random.randn(batch_size, num_components))
      mus = [
          tf.Variable(np.random.randn(batch_size, num_features))
          for _ in range(num_components)
      ]
      sigmas = [
          tf.Variable(
              psd(np.random.rand(batch_size, num_features, num_features)))
          for _ in range(num_components)
      ]
      components = list(
          tfd.MultivariateNormalTriL(
              loc=mu, scale_tril=tf.linalg.cholesky(sigma))
          for (mu, sigma) in zip(mus, sigmas))
      return tfd.Mixture(
          cat,
          components,
          use_static_graph=self.use_static_graph,
          validate_args=True)

    for use_gpu in False, True:
      if use_gpu and not tf.test.is_gpu_available():
        continue
      for num_components in 1, 8, 16:
        for batch_size in 1, 32:
          for num_features in 1, 64, 512:
            for sample_size in 1, 32, 128:
              self._runSamplingBenchmark(
                  'mvn_full',
                  create_distribution=create_distribution,
                  use_gpu=use_gpu,
                  num_components=num_components,
                  batch_size=batch_size,
                  num_features=num_features,
                  sample_size=sample_size)


class MixtureStaticSampleBenchmark(MixtureBenchmark):
  use_static_graph = True


if __name__ == '__main__':
  tf.test.main()
