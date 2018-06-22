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
"""Tests for MultivariateNormal."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
import numpy as np
from scipy import stats
import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions


class MultivariateNormalDiagTest(tf.test.TestCase):
  """Well tested because this is a simple override of the base class."""

  def setUp(self):
    self._rng = np.random.RandomState(42)

  def testScalarParams(self):
    mu = -1.
    diag = -5.
    with self.test_session():
      with self.assertRaisesRegexp(ValueError, "at least 1 dimension"):
        tfd.MultivariateNormalDiag(mu, diag)

  def testVectorParams(self):
    mu = [-1.]
    diag = [-5.]
    with self.test_session():
      dist = tfd.MultivariateNormalDiag(mu, diag, validate_args=True)
      self.assertAllEqual([3, 1], dist.sample(3).get_shape())

  def testDistWithBatchShapeOneThenTransformedThroughSoftplus(self):
    # This complex combination of events resulted in a loss of static shape
    # information when tensor_util.constant_value(self._needs_rotation) was
    # being used incorrectly (resulting in always rotating).
    # Batch shape = [1], event shape = [3]
    mu = tf.zeros((1, 3))
    diag = tf.ones((1, 3))
    with self.test_session():
      base_dist = tfd.MultivariateNormalDiag(mu, diag, validate_args=True)
      dist = tfd.TransformedDistribution(
          base_dist, validate_args=True, bijector=tfp.bijectors.Softplus())
      samps = dist.sample(5)  # Shape [5, 1, 3].
      self.assertAllEqual([5, 1], dist.log_prob(samps).get_shape())

  def testMean(self):
    mu = [-1., 1]
    diag = [1., -5]
    with self.test_session():
      dist = tfd.MultivariateNormalDiag(mu, diag, validate_args=True)
      self.assertAllEqual(mu, dist.mean().eval())

  def testMeanWithBroadcastLoc(self):
    mu = [-1.]
    diag = [1., -5]
    with self.test_session():
      dist = tfd.MultivariateNormalDiag(mu, diag, validate_args=True)
      self.assertAllEqual([-1., -1.], dist.mean().eval())

  def testEntropy(self):
    mu = [-1., 1]
    diag = [-1., 5]
    diag_mat = np.diag(diag)
    scipy_mvn = stats.multivariate_normal(mean=mu, cov=diag_mat**2)
    with self.test_session():
      dist = tfd.MultivariateNormalDiag(mu, diag, validate_args=True)
      self.assertAllClose(scipy_mvn.entropy(), dist.entropy().eval(), atol=1e-4)

  def testSample(self):
    mu = [-1., 1]
    diag = [1., -2]
    with self.test_session():
      dist = tfd.MultivariateNormalDiag(mu, diag, validate_args=True)
      samps = dist.sample(int(1e3), seed=0).eval()
      cov_mat = tf.matrix_diag(diag).eval()**2

      self.assertAllClose(mu, samps.mean(axis=0),
                          atol=0., rtol=0.05)
      self.assertAllClose(cov_mat, np.cov(samps.T),
                          atol=0.05, rtol=0.05)

  def testSingularScaleRaises(self):
    mu = [-1., 1]
    diag = [1., 0]
    with self.test_session():
      dist = tfd.MultivariateNormalDiag(mu, diag, validate_args=True)
      with self.assertRaisesOpError("Singular"):
        dist.sample().eval()

  def testSampleWithBroadcastScale(self):
    # mu corresponds to a 2-batch of 3-variate normals
    mu = np.zeros([2, 3])

    # diag corresponds to no batches of 3-variate normals
    diag = np.ones([3])

    with self.test_session():
      dist = tfd.MultivariateNormalDiag(mu, diag, validate_args=True)

      mean = dist.mean()
      self.assertAllEqual([2, 3], mean.get_shape())
      self.assertAllClose(mu, mean.eval())

      n = int(1e3)
      samps = dist.sample(n, seed=0).eval()
      cov_mat = tf.matrix_diag(diag).eval()**2
      sample_cov = np.matmul(samps.transpose([1, 2, 0]),
                             samps.transpose([1, 0, 2])) / n

      self.assertAllClose(mu, samps.mean(axis=0),
                          atol=0.10, rtol=0.05)
      self.assertAllClose([cov_mat, cov_mat], sample_cov,
                          atol=0.10, rtol=0.05)

  def testCovariance(self):
    with self.test_session():
      mvn = tfd.MultivariateNormalDiag(loc=tf.zeros([2, 3], dtype=tf.float32))
      self.assertAllClose(
          np.diag(np.ones([3], dtype=np.float32)),
          mvn.covariance().eval())

      mvn = tfd.MultivariateNormalDiag(
          loc=tf.zeros([3], dtype=tf.float32),
          scale_identity_multiplier=[3., 2.])
      self.assertAllEqual([2], mvn.batch_shape)
      self.assertAllEqual([3], mvn.event_shape)
      self.assertAllClose(
          np.array([[[3., 0, 0],
                     [0, 3, 0],
                     [0, 0, 3]],
                    [[2, 0, 0],
                     [0, 2, 0],
                     [0, 0, 2]]])**2.,
          mvn.covariance().eval())

      mvn = tfd.MultivariateNormalDiag(
          loc=tf.zeros([3], dtype=tf.float32),
          scale_diag=[[3., 2, 1], [4, 5, 6]])
      self.assertAllEqual([2], mvn.batch_shape)
      self.assertAllEqual([3], mvn.event_shape)
      self.assertAllClose(
          np.array([[[3., 0, 0],
                     [0, 2, 0],
                     [0, 0, 1]],
                    [[4, 0, 0],
                     [0, 5, 0],
                     [0, 0, 6]]])**2.,
          mvn.covariance().eval())

  def testVariance(self):
    with self.test_session():
      mvn = tfd.MultivariateNormalDiag(loc=tf.zeros([2, 3], dtype=tf.float32))
      self.assertAllClose(
          np.ones([3], dtype=np.float32),
          mvn.variance().eval())

      mvn = tfd.MultivariateNormalDiag(
          loc=tf.zeros([3], dtype=tf.float32),
          scale_identity_multiplier=[3., 2.])
      self.assertAllClose(
          np.array([[3., 3, 3],
                    [2, 2, 2]])**2.,
          mvn.variance().eval())

      mvn = tfd.MultivariateNormalDiag(
          loc=tf.zeros([3], dtype=tf.float32),
          scale_diag=[[3., 2, 1], [4, 5, 6]])
      self.assertAllClose(
          np.array([[3., 2, 1],
                    [4, 5, 6]])**2.,
          mvn.variance().eval())

  def testStddev(self):
    with self.test_session():
      mvn = tfd.MultivariateNormalDiag(loc=tf.zeros([2, 3], dtype=tf.float32))
      self.assertAllClose(
          np.ones([3], dtype=np.float32),
          mvn.stddev().eval())

      mvn = tfd.MultivariateNormalDiag(
          loc=tf.zeros([3], dtype=tf.float32),
          scale_identity_multiplier=[3., 2.])
      self.assertAllClose(
          np.array([[3., 3, 3],
                    [2, 2, 2]]),
          mvn.stddev().eval())

      mvn = tfd.MultivariateNormalDiag(
          loc=tf.zeros([3], dtype=tf.float32),
          scale_diag=[[3., 2, 1], [4, 5, 6]])
      self.assertAllClose(
          np.array([[3., 2, 1],
                    [4, 5, 6]]),
          mvn.stddev().eval())

  def testMultivariateNormalDiagWithSoftplusScale(self):
    mu = [-1.0, 1.0]
    diag = [-1.0, -2.0]
    with self.test_session():
      dist = tfd.MultivariateNormalDiagWithSoftplusScale(
          mu, diag, validate_args=True)
      samps = dist.sample(1000, seed=0).eval()
      cov_mat = tf.matrix_diag(tf.nn.softplus(diag)).eval()**2

      self.assertAllClose(mu, samps.mean(axis=0), atol=0.1)
      self.assertAllClose(cov_mat, np.cov(samps.T), atol=0.1)

  def testMultivariateNormalDiagNegLogLikelihood(self):
    num_draws = 50
    dims = 3
    with self.test_session() as sess:
      x_pl = tf.placeholder(dtype=tf.float32, shape=[None, dims], name="x")
      mu_var = tf.get_variable(
          name="mu",
          shape=[dims],
          dtype=tf.float32,
          initializer=tf.constant_initializer(1.))
      sess.run([tf.global_variables_initializer()])

      mvn = tfd.MultivariateNormalDiag(
          loc=mu_var, scale_diag=tf.ones(shape=[dims], dtype=tf.float32))

      # Typically you'd use `mvn.log_prob(x_pl)` which is always at least as
      # numerically stable as `tf.log(mvn.prob(x_pl))`. However in this test
      # we're testing a bug specific to `prob` and not `log_prob`;
      # http://stackoverflow.com/q/45109305. (The underlying issue was not
      # related to `Distributions` but that `reduce_prod` didn't correctly
      # handle negative indexes.)
      neg_log_likelihood = -tf.reduce_sum(tf.log(mvn.prob(x_pl)))
      grad_neg_log_likelihood = tf.gradients(neg_log_likelihood,
                                             tf.trainable_variables())

      x = np.zeros([num_draws, dims], dtype=np.float32)
      grad_neg_log_likelihood_ = sess.run(
          grad_neg_log_likelihood,
          feed_dict={x_pl: x})
      self.assertEqual(1, len(grad_neg_log_likelihood_))
      self.assertAllClose(grad_neg_log_likelihood_[0],
                          np.tile(num_draws, dims),
                          rtol=1e-6, atol=0.)

  def testDynamicBatchShape(self):
    mvn = tfd.MultivariateNormalDiag(
        loc=tf.placeholder(tf.float32, shape=[None, None, 2]),
        scale_diag=tf.placeholder(tf.float32, shape=[None, None, 2]))
    self.assertListEqual(mvn.batch_shape.as_list(), [None, None])
    self.assertListEqual(mvn.event_shape.as_list(), [2])

  def testDynamicEventShape(self):
    mvn = tfd.MultivariateNormalDiag(
        loc=tf.placeholder(tf.float32, shape=[2, 3, None]),
        scale_diag=tf.placeholder(tf.float32, shape=[2, 3, None]))
    self.assertListEqual(mvn.batch_shape.as_list(), [2, 3])
    self.assertListEqual(mvn.event_shape.as_list(), [None])

  def testKLDivIdenticalGradientDefined(self):
    dims = 3
    with self.test_session() as sess:
      loc = tf.zeros([dims], dtype=tf.float32)
      mvn = tfd.MultivariateNormalDiag(
          loc=loc, scale_diag=np.ones([dims], dtype=np.float32))
      g = tf.gradients(tfd.kl_divergence(mvn, mvn), loc)
      g_ = sess.run(g)
      self.assertAllEqual(np.ones_like(g_, dtype=np.bool),
                          np.isfinite(g_))


if __name__ == "__main__":
  tf.test.main()
