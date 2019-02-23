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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from tensorflow.python.framework import test_util  # pylint:disable=g-direct-tensorflow-import


tfd = tfp.distributions


@test_util.run_all_in_graph_and_eager_modes
class MultinomialTest(tf.test.TestCase):

  def setUp(self):
    self._rng = np.random.RandomState(42)

  def testSimpleShapes(self):
    p = [.1, .3, .6]
    dist = tfd.Multinomial(total_count=1., probs=p)
    self.assertEqual(3, self.evaluate(dist.event_shape_tensor()))
    self.assertAllEqual([], self.evaluate(dist.batch_shape_tensor()))
    self.assertEqual(tf.TensorShape([3]), dist.event_shape)
    self.assertEqual(tf.TensorShape([]), dist.batch_shape)

  def testComplexShapes(self):
    p = 0.5 * np.ones([3, 2, 2], dtype=np.float32)
    n = [[3., 2], [4, 5], [6, 7]]
    dist = tfd.Multinomial(total_count=n, probs=p)
    self.assertEqual(2, self.evaluate(dist.event_shape_tensor()))
    self.assertAllEqual([3, 2], self.evaluate(dist.batch_shape_tensor()))
    self.assertEqual(tf.TensorShape([2]), dist.event_shape)
    self.assertEqual(tf.TensorShape([3, 2]), dist.batch_shape)
    self.assertEqual(tf.TensorShape([17, 3, 2, 2]), dist.sample(17).shape)

  def testN(self):
    p = [[0.1, 0.2, 0.7], [0.2, 0.3, 0.5]]
    n = [[3.], [4]]
    dist = tfd.Multinomial(total_count=n, probs=p)
    self.assertEqual((2, 1), dist.total_count.shape)
    self.assertAllClose(n, self.evaluate(dist.total_count))

  def testP(self):
    p = [[0.1, 0.2, 0.7]]
    dist = tfd.Multinomial(total_count=3., probs=p)
    self.assertEqual((1, 3), dist.probs.shape)
    self.assertEqual((1, 3), dist.logits.shape)
    self.assertAllClose(p, self.evaluate(dist.probs))

  def testLogits(self):
    p = np.array([[0.1, 0.2, 0.7]], dtype=np.float32)
    logits = np.log(p) - 50.
    multinom = tfd.Multinomial(total_count=3., logits=logits)
    self.assertEqual((1, 3), multinom.probs.shape)
    self.assertEqual((1, 3), multinom.logits.shape)
    self.assertAllClose(p, self.evaluate(multinom.probs))
    self.assertAllClose(logits, self.evaluate(multinom.logits))

  def testPmfUnderflow(self):
    logits = np.array([[-200, 0]], dtype=np.float32)
    dist = tfd.Multinomial(total_count=1., logits=logits)
    lp = self.evaluate(dist.log_prob([1., 0.]))[0]
    self.assertAllClose(-200, lp, atol=0, rtol=1e-6)

  def testPmfandCountsAgree(self):
    p = [[0.1, 0.2, 0.7]]
    n = [[5.]]
    dist = tfd.Multinomial(total_count=n, probs=p, validate_args=True)
    self.evaluate(dist.prob([2., 3, 0]))
    self.evaluate(dist.prob([3., 0, 2]))
    with self.assertRaisesOpError("must be non-negative"):
      self.evaluate(dist.prob([-1., 4, 2]))
    with self.assertRaisesOpError("counts must sum to `self.total_count`"):
      self.evaluate(dist.prob([3., 3, 0]))

  def testPmfNonIntegerCounts(self):
    p = [[0.1, 0.2, 0.7]]
    n = [[5.]]
    # No errors with integer n.
    multinom = tfd.Multinomial(
        total_count=n, probs=p, validate_args=True)
    self.evaluate(multinom.prob([2., 1, 2]))
    self.evaluate(multinom.prob([3., 0, 2]))
    # Counts don't sum to n.
    with self.assertRaisesOpError("counts must sum to `self.total_count`"):
      self.evaluate(multinom.prob([2., 3, 2]))
    # Counts are non-integers.
    x = tf.compat.v1.placeholder_with_default([1., 2.5, 1.5], shape=None)
    with self.assertRaisesOpError(
        "cannot contain fractional components."):
      self.evaluate(multinom.prob(x))

    multinom = tfd.Multinomial(
        total_count=n, probs=p, validate_args=False)
    self.evaluate(multinom.prob([1., 2., 2.]))
    # Non-integer arguments work.
    self.evaluate(multinom.prob([1.0, 2.5, 1.5]))

  def testPmfBothZeroBatches(self):
    # Both zero-batches.  No broadcast
    p = [0.5, 0.5]
    counts = [1., 0]
    pmf = tfd.Multinomial(total_count=1., probs=p).prob(counts)
    self.assertAllClose(0.5, self.evaluate(pmf))
    self.assertEqual((), pmf.shape)

  def testPmfBothZeroBatchesNontrivialN(self):
    # Both zero-batches.  No broadcast
    p = [0.1, 0.9]
    counts = [3., 2]
    dist = tfd.Multinomial(total_count=5., probs=p)
    pmf = dist.prob(counts)
    # 5 choose 3 = 5 choose 2 = 10. 10 * (.9)^2 * (.1)^3 = 81/10000.
    self.assertAllClose(81. / 10000, self.evaluate(pmf))
    self.assertEqual((), pmf.shape)

  def testPmfPStretchedInBroadcastWhenSameRank(self):
    p = [[0.1, 0.9]]
    counts = [[1., 0], [0, 1]]
    pmf = tfd.Multinomial(total_count=1., probs=p).prob(counts)
    self.assertAllClose([0.1, 0.9], self.evaluate(pmf))
    self.assertEqual((2), pmf.shape)

  def testPmfPStretchedInBroadcastWhenLowerRank(self):
    p = [0.1, 0.9]
    counts = [[1., 0], [0, 1]]
    pmf = tfd.Multinomial(total_count=1., probs=p).prob(counts)
    self.assertAllClose([0.1, 0.9], self.evaluate(pmf))
    self.assertEqual((2), pmf.shape)

  def testPmfCountsStretchedInBroadcastWhenSameRank(self):
    p = [[0.1, 0.9], [0.7, 0.3]]
    counts = [[1., 0]]
    pmf = tfd.Multinomial(total_count=1., probs=p).prob(counts)
    self.assertAllClose(self.evaluate(pmf), [0.1, 0.7])
    self.assertEqual((2), pmf.shape)

  def testPmfCountsStretchedInBroadcastWhenLowerRank(self):
    p = [[0.1, 0.9], [0.7, 0.3]]
    counts = [1., 0]
    pmf = tfd.Multinomial(total_count=1., probs=p).prob(counts)
    self.assertAllClose(self.evaluate(pmf), [0.1, 0.7])
    self.assertEqual(pmf.shape, (2))

  def testPmfShapeCountsStretchedN(self):
    # [2, 2, 2]
    p = [[[0.1, 0.9], [0.1, 0.9]], [[0.7, 0.3], [0.7, 0.3]]]
    # [2, 2]
    n = [[3., 3], [3, 3]]
    # [2]
    counts = [2., 1]
    pmf = tfd.Multinomial(total_count=n, probs=p).prob(counts)
    self.evaluate(pmf)
    self.assertEqual(pmf.shape, (2, 2))

  def testPmfShapeCountsPStretchedN(self):
    p = [0.1, 0.9]
    counts = [3., 2]
    n = np.full([4, 3], 5., dtype=np.float32)
    pmf = tfd.Multinomial(total_count=n, probs=p).prob(counts)
    self.evaluate(pmf)
    self.assertEqual((4, 3), pmf.shape)

  def testMultinomialMean(self):
    n = 5.
    p = [0.1, 0.2, 0.7]
    dist = tfd.Multinomial(total_count=n, probs=p)
    expected_means = 5 * np.array(p, dtype=np.float32)
    self.assertEqual((3,), dist.mean().shape)
    self.assertAllClose(expected_means, self.evaluate(dist.mean()))

  def testMultinomialCovariance(self):
    n = 5.
    p = [0.1, 0.2, 0.7]
    dist = tfd.Multinomial(total_count=n, probs=p)
    expected_covariances = [[9. / 20, -1 / 10, -7 / 20],
                            [-1 / 10, 4 / 5, -7 / 10],
                            [-7 / 20, -7 / 10, 21 / 20]]
    self.assertEqual((3, 3), dist.covariance().shape)
    self.assertAllClose(
        expected_covariances, self.evaluate(dist.covariance()))

  def testMultinomialCovarianceBatch(self):
    # Shape [2]
    n = [5.] * 2
    # Shape [4, 1, 2]
    p = [[[0.1, 0.9]], [[0.1, 0.9]]] * 2
    dist = tfd.Multinomial(total_count=n, probs=p)
    # Shape [2, 2]
    inner_var = [[9. / 20, -9 / 20], [-9 / 20, 9 / 20]]
    # Shape [4, 2, 2, 2]
    expected_covariances = [[inner_var, inner_var]] * 4
    self.assertEqual((4, 2, 2, 2), dist.covariance().shape)
    self.assertAllClose(
        expected_covariances, self.evaluate(dist.covariance()))

  def testCovarianceMultidimensional(self):
    # Shape [3, 5, 4]
    p = np.random.dirichlet([.25, .25, .25, .25], [3, 5]).astype(np.float32)
    # Shape [6, 3, 3]
    p2 = np.random.dirichlet([.3, .3, .4], [6, 3]).astype(np.float32)

    ns = np.random.randint(low=1, high=11, size=[3, 5]).astype(np.float32)
    ns2 = np.random.randint(low=1, high=11, size=[6, 1]).astype(np.float32)

    dist = tfd.Multinomial(ns, p)
    dist2 = tfd.Multinomial(ns2, p2)

    covariance = dist.covariance()
    covariance2 = dist2.covariance()
    self.assertEqual((3, 5, 4, 4), covariance.shape)
    self.assertEqual((6, 3, 3, 3), covariance2.shape)

  def testCovarianceFromSampling(self):
    # We will test mean, cov, var, stddev on a Multinomial constructed via
    # broadcast between alpha, n.
    theta = np.array([[1., 2, 3],
                      [2.5, 4, 0.01]], dtype=np.float32)
    theta /= np.sum(theta, 1)[..., tf.newaxis]
    n = np.array([[10., 9.], [8., 7.], [6., 5.]], dtype=np.float32)
    # batch_shape=[3, 2], event_shape=[3]
    dist = tfd.Multinomial(n, theta)
    x = dist.sample(int(1000e3), seed=1)
    sample_mean = tf.reduce_mean(input_tensor=x, axis=0)
    x_centered = x - sample_mean[tf.newaxis, ...]
    sample_cov = tf.reduce_mean(
        input_tensor=tf.matmul(x_centered[..., tf.newaxis],
                               x_centered[..., tf.newaxis, :]),
        axis=0)
    sample_var = tf.linalg.diag_part(sample_cov)
    sample_stddev = tf.sqrt(sample_var)
    [
        sample_mean_,
        sample_cov_,
        sample_var_,
        sample_stddev_,
        analytic_mean,
        analytic_cov,
        analytic_var,
        analytic_stddev,
    ] = self.evaluate([
        sample_mean,
        sample_cov,
        sample_var,
        sample_stddev,
        dist.mean(),
        dist.covariance(),
        dist.variance(),
        dist.stddev(),
    ])
    self.assertAllClose(sample_mean_, analytic_mean, atol=0.01, rtol=0.01)
    self.assertAllClose(sample_cov_, analytic_cov, atol=0.01, rtol=0.01)
    self.assertAllClose(sample_var_, analytic_var, atol=0.01, rtol=0.01)
    self.assertAllClose(sample_stddev_, analytic_stddev, atol=0.01, rtol=0.01)

  def testSampleUnbiasedNonScalarBatch(self):
    dist = tfd.Multinomial(
        total_count=[7., 6., 5.],
        logits=tf.math.log(2. * self._rng.rand(4, 3, 2).astype(np.float32)))
    n = int(3e4)
    x = dist.sample(n, seed=0)
    sample_mean = tf.reduce_mean(input_tensor=x, axis=0)
    # Cyclically rotate event dims left.
    x_centered = tf.transpose(a=x - sample_mean, perm=[1, 2, 3, 0])
    sample_covariance = tf.matmul(
        x_centered, x_centered, adjoint_b=True) / n
    [
        sample_mean_,
        sample_covariance_,
        actual_mean_,
        actual_covariance_,
    ] = self.evaluate([
        sample_mean,
        sample_covariance,
        dist.mean(),
        dist.covariance(),
    ])
    self.assertAllEqual([4, 3, 2], sample_mean.shape)
    self.assertAllClose(actual_mean_, sample_mean_, atol=0., rtol=0.10)
    self.assertAllEqual([4, 3, 2, 2], sample_covariance.shape)
    self.assertAllClose(
        actual_covariance_, sample_covariance_, atol=0., rtol=0.20)

  def testSampleUnbiasedScalarBatch(self):
    dist = tfd.Multinomial(
        total_count=5.,
        logits=tf.math.log(2. * self._rng.rand(4).astype(np.float32)))
    n = int(5e3)
    x = dist.sample(n, seed=0)
    sample_mean = tf.reduce_mean(input_tensor=x, axis=0)
    x_centered = x - sample_mean  # Already transposed to [n, 2].
    sample_covariance = tf.matmul(
        x_centered, x_centered, adjoint_a=True) / n
    [
        sample_mean_,
        sample_covariance_,
        actual_mean_,
        actual_covariance_,
    ] = self.evaluate([
        sample_mean,
        sample_covariance,
        dist.mean(),
        dist.covariance(),
    ])
    self.assertAllEqual([4], sample_mean.shape)
    self.assertAllClose(actual_mean_, sample_mean_, atol=0., rtol=0.10)
    self.assertAllEqual([4, 4], sample_covariance.shape)
    self.assertAllClose(
        actual_covariance_, sample_covariance_, atol=0., rtol=0.20)

  def testNotReparameterized(self):
    total_count = tf.constant(5.0)
    probs = tf.constant([0.2, 0.6])
    _, [grad_total_count, grad_probs] = tfp.math.value_and_gradient(
        lambda n, p: tfd.Multinomial(total_count=n, probs=p).sample(100),
        [total_count, probs])
    self.assertIsNone(grad_total_count)
    self.assertIsNone(grad_probs)


if __name__ == "__main__":
  tf.test.main()
