# Copyright 2019 The TensorFlow Probability Authors.
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
"""Tests for PlackettLuce distribution."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import scipy
import tensorflow.compat.v1 as tf1
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp

from tensorflow_probability.python.internal import tensorshape_util
from tensorflow_probability.python.internal import test_util

tfb = tfp.bijectors
tfd = tfp.distributions


def make_plackett_luce(batch_shape,
                       num_elements,
                       dtype=tf.int32,
                       scores_dtype=tf.float32):
  scores = tf.random.uniform(
      list(batch_shape) + [num_elements], 0.1, 10,
      dtype=scores_dtype, seed=test_util.test_seed())
  return tfd.PlackettLuce(scores, dtype=dtype, validate_args=True)


@test_util.test_all_tf_execution_regimes
class PlackettLuceTest(test_util.TestCase):

  def setUp(self):
    super(PlackettLuceTest, self).setUp()
    self._rng = np.random.RandomState(42)

  def assertRaises(self, error_class, msg):
    if tf.executing_eagerly():
      return self.assertRaisesRegexp(error_class, msg)
    return self.assertRaisesOpError(msg)

  def testScores(self):
    s = [0.1, 2., 5.]
    dist = tfd.PlackettLuce(scores=s, validate_args=True)
    self.assertAllClose(s, self.evaluate(dist.scores))
    self.assertAllEqual([3], dist.scores.shape)

  def testShapes(self):
    for batch_shape in ([], [1], [2, 3, 4]):
      dist = make_plackett_luce(batch_shape, 10)
      self.assertAllEqual(batch_shape,
                          tensorshape_util.as_list(dist.batch_shape))
      self.assertAllEqual(batch_shape, self.evaluate(dist.batch_shape_tensor()))
      self.assertAllEqual([10], tensorshape_util.as_list(dist.event_shape))
      self.assertAllEqual([10], self.evaluate(dist.event_shape_tensor()))
      # event_shape is available as a constant because the shape is
      # known at graph build time.
      self.assertEqual(10, dist.event_shape)

    for batch_shape in ([], [1], [2, 3, 4]):
      dist = make_plackett_luce(batch_shape, tf.constant(10, dtype=tf.int32))
      self.assertAllEqual(
          len(batch_shape), tensorshape_util.rank(dist.batch_shape))
      self.assertAllEqual(batch_shape, self.evaluate(dist.batch_shape_tensor()))
      self.assertAllEqual([10], tensorshape_util.as_list(dist.event_shape))
      self.assertEqual(10, self.evaluate(dist.event_shape_tensor()))

  def testDtype(self):
    dist = make_plackett_luce([], 5, dtype=tf.int32)
    self.assertEqual(dist.dtype, tf.int32)
    self.assertEqual(dist.dtype, dist.sample(
        5, seed=test_util.test_seed()).dtype)
    self.assertEqual(dist.dtype, dist.mode().dtype)
    dist = make_plackett_luce([], 5, dtype=tf.int64)
    self.assertEqual(dist.dtype, tf.int64)
    self.assertEqual(dist.dtype, dist.sample(
        5, seed=test_util.test_seed()).dtype)
    self.assertEqual(dist.dtype, dist.mode().dtype)
    self.assertEqual(dist.scores.dtype, tf.float32)
    self.assertEqual(dist.scores.dtype, dist.prob(
        np.array([1, 0, 2, 4, 3], dtype=np.int64)).dtype)
    self.assertEqual(dist.scores.dtype, dist.log_prob(
        np.array([1, 0, 2, 4, 3], dtype=np.int64)).dtype)
    dist = make_plackett_luce([], 5, dtype=tf.int64, scores_dtype=tf.float64)
    self.assertEqual(dist.scores.dtype, tf.float64)
    self.assertEqual(dist.scores.dtype, dist.prob(
        np.array([1, 0, 2, 4, 3], dtype=np.int64)).dtype)
    self.assertEqual(dist.scores.dtype, dist.log_prob(
        np.array([1, 0, 2, 4, 3], dtype=np.int64)).dtype)

  def testUnknownShape(self):
    scores = tf1.placeholder_with_default(
        [[1e-6, 1000.0], [1000.0, 1e-6]], shape=None)
    dist = tfd.PlackettLuce(scores, validate_args=True)
    sample = dist.sample(seed=test_util.test_seed())
    # Batch entry 0, 1 will sample permutations [1, 0], [0, 1].
    sample_value_batch = self.evaluate(sample)
    self.assertAllEqual([[1, 0], [0, 1]], sample_value_batch)

  def testPmfUniform(self):
    # Check that distribution with equal scores gives uniform probability.
    scores = self._rng.uniform() * np.ones(shape=(8, 2, 10), dtype=np.float32)
    dist = tfd.PlackettLuce(scores=scores, validate_args=True)
    np_sample = self.evaluate(dist.sample(seed=test_util.test_seed()))
    np_prob = self.evaluate(dist.log_prob(np_sample))
    n_fac = scipy.special.factorial(scores.shape[-1]).astype(np.float32)
    expected_prob = self.evaluate(-tf.math.log(n_fac) * tf.ones_like(np_prob))
    self.assertAllClose(np_prob, expected_prob)

  def testPmfMode(self):
    # Check that mode has higher probability than any other random sample.
    scores = self._rng.uniform(low=0.1, size=(10)).astype(np.float32)
    dist = tfd.PlackettLuce(scores=scores, validate_args=True)
    np_sample = self.evaluate(dist.sample(seed=test_util.test_seed()))
    np_prob = self.evaluate(dist.log_prob(np_sample))
    mode = self.evaluate(dist.mode())
    mode_prob = self.evaluate(dist.log_prob(mode))
    self.assertLessEqual(np_prob, mode_prob)

  def testSample(self):
    scores = np.array([[[0.1, 2.3, 5.], [4.2, 0.5, 3.1]]])
    dist = tfd.PlackettLuce(scores, validate_args=True)
    n = 100
    k = scores.shape[-1]
    samples = dist.sample(n, seed=test_util.test_seed())
    self.assertEqual(samples.dtype, tf.int32)
    sample_values = self.evaluate(samples)
    self.assertAllEqual([n, 1, 2, k], sample_values.shape)
    self.assertFalse(np.any(sample_values < 0))
    self.assertFalse(np.any(sample_values > k))
    self.assertTrue(np.all(np.sum(sample_values, axis=-1) == (k-1)*k//2))

  def testAssertValidSample(self):
    scores = np.array([[[0.1, 2.3, 5.], [4.2, 0.5, 3.1]]])
    dist = tfd.PlackettLuce(scores, validate_args=True)
    with self.assertRaisesOpError('Sample must be a permutation'):
      self.evaluate(dist.log_prob([1, 0, 1]))


@test_util.test_all_tf_execution_regimes
class PlackettLuceFromVariableTest(test_util.TestCase):

  def testAssertionsProbsAfterMutation(self):
    x = tf.Variable([0.25, 0.25, 0.5])
    d = tfd.PlackettLuce(scores=x, validate_args=True)
    with self.assertRaisesOpError('Condition x > 0 did not hold element-wise'):
      self.evaluate([v.initializer for v in d.variables])
      with tf.control_dependencies([x.assign([-0.25, 0.75, 0.5])]):
        self.evaluate(d.scores_parameter())

  def testAssertionsScores(self):
    x = tfp.util.TransformedVariable(0., tfb.Identity(), shape=None)
    with self.assertRaisesRegexp(
        ValueError, 'Argument `scores` must have rank at least 1.'):
      d = tfd.PlackettLuce(scores=x, validate_args=True)
      self.evaluate([v.initializer for v in d.variables])

if __name__ == '__main__':
  tf.test.main()
