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

from tensorflow_probability.python.internal import tensorshape_util
from tensorflow_probability.python.internal import test_case
from tensorflow_probability.python.internal import test_util as tfp_test_util
from tensorflow.python.framework import test_util  # pylint: disable=g-direct-tensorflow-import,g-import-not-at-top

tfd = tfp.distributions


@test_util.run_all_in_graph_and_eager_modes
class MultivariateNormalDiagTest(test_case.TestCase):
  """Well tested because this is a simple override of the base class."""

  def setUp(self):
    self._rng = np.random.RandomState(42)

  def testScalarParams(self):
    mu = -1.
    diag = -5.
    with self.assertRaisesRegexp(ValueError, "at least 1 dimension"):
      tfd.MultivariateNormalDiag(mu, diag)

  def testVectorParams(self):
    mu = [-1.]
    diag = [-5.]
    dist = tfd.MultivariateNormalDiag(mu, diag, validate_args=True)
    self.assertAllEqual([3, 1], dist.sample(3).shape)

  def testDistWithBatchShapeOneThenTransformedThroughSoftplus(self):
    # This complex combination of events resulted in a loss of static shape
    # information when tf.get_static_value(self._needs_rotation) was
    # being used incorrectly (resulting in always rotating).
    # Batch shape = [1], event shape = [3]
    mu = tf.zeros((1, 3))
    diag = tf.ones((1, 3))
    base_dist = tfd.MultivariateNormalDiag(mu, diag, validate_args=True)
    dist = tfd.TransformedDistribution(
        base_dist, validate_args=True, bijector=tfp.bijectors.Softplus())
    samps = dist.sample(5)  # Shape [5, 1, 3].
    self.assertAllEqual([5, 1], dist.log_prob(samps).shape)

  def testMean(self):
    mu = [-1., 1]
    diag = [1., -5]
    dist = tfd.MultivariateNormalDiag(mu, diag, validate_args=True)
    self.assertAllEqual(mu, self.evaluate(dist.mean()))

  def testMeanWithBroadcastLoc(self):
    mu = [-1.]
    diag = [1., -5]
    dist = tfd.MultivariateNormalDiag(mu, diag, validate_args=True)
    self.assertAllEqual([-1., -1.], self.evaluate(dist.mean()))

  def testEntropy(self):
    mu = [-1., 1]
    diag = [-1., 5]
    diag_mat = np.diag(diag)
    scipy_mvn = stats.multivariate_normal(mean=mu, cov=diag_mat**2)
    dist = tfd.MultivariateNormalDiag(mu, diag, validate_args=True)
    self.assertAllClose(
        scipy_mvn.entropy(), self.evaluate(dist.entropy()), atol=1e-4)

  def testSample(self):
    mu = [-1., 1]
    diag = [1., -2]
    dist = tfd.MultivariateNormalDiag(mu, diag, validate_args=True)
    samps = self.evaluate(
        dist.sample(int(1e3), seed=tfp_test_util.test_seed(hardcoded_seed=0)))
    cov_mat = self.evaluate(tf.linalg.diag(diag))**2

    self.assertAllClose(mu, samps.mean(axis=0), atol=0., rtol=0.05)
    self.assertAllClose(cov_mat, np.cov(samps.T), atol=0.05, rtol=0.05)

  def testSingularScaleRaises(self):
    mu = [-1., 1]
    diag = [1., 0]
    dist = tfd.MultivariateNormalDiag(mu, diag, validate_args=True)
    with self.assertRaisesOpError("Singular"):
      self.evaluate(dist.sample())

  def testSampleWithBroadcastScale(self):
    # mu corresponds to a 2-batch of 3-variate normals
    mu = np.zeros([2, 3])

    # diag corresponds to no batches of 3-variate normals
    diag = np.ones([3])

    dist = tfd.MultivariateNormalDiag(mu, diag, validate_args=True)

    mean = dist.mean()
    self.assertAllEqual([2, 3], mean.shape)
    self.assertAllClose(mu, self.evaluate(mean))

    n = int(1e3)
    samps = self.evaluate(dist.sample(n, seed=tfp_test_util.test_seed()))
    cov_mat = self.evaluate(tf.linalg.diag(diag))**2
    sample_cov = np.matmul(
        samps.transpose([1, 2, 0]), samps.transpose([1, 0, 2])) / n

    self.assertAllClose(mu, samps.mean(axis=0), atol=0.10, rtol=0.05)
    self.assertAllClose([cov_mat, cov_mat], sample_cov, atol=0.10, rtol=0.05)

  def testCovariance(self):
    mvn = tfd.MultivariateNormalDiag(loc=tf.zeros([2, 3], dtype=tf.float32))
    self.assertAllClose(
        np.diag(np.ones([3], dtype=np.float32)),
        self.evaluate(mvn.covariance()))

    mvn = tfd.MultivariateNormalDiag(
        loc=tf.zeros([3], dtype=tf.float32), scale_identity_multiplier=[3., 2.])
    self.assertAllEqual([2], mvn.batch_shape)
    self.assertAllEqual([3], mvn.event_shape)
    self.assertAllClose(
        np.array([[[3., 0, 0], [0, 3, 0], [0, 0, 3]], [[2, 0, 0], [0, 2, 0],
                                                       [0, 0, 2]]])**2.,
        self.evaluate(mvn.covariance()))

    mvn = tfd.MultivariateNormalDiag(
        loc=tf.zeros([3], dtype=tf.float32), scale_diag=[[3., 2, 1], [4, 5, 6]])
    self.assertAllEqual([2], mvn.batch_shape)
    self.assertAllEqual([3], mvn.event_shape)
    self.assertAllClose(
        np.array([[[3., 0, 0], [0, 2, 0], [0, 0, 1]], [[4, 0, 0], [0, 5, 0],
                                                       [0, 0, 6]]])**2.,
        self.evaluate(mvn.covariance()))

  def testVariance(self):
    mvn = tfd.MultivariateNormalDiag(loc=tf.zeros([2, 3], dtype=tf.float32))
    self.assertAllClose(
        np.ones([3], dtype=np.float32), self.evaluate(mvn.variance()))

    mvn = tfd.MultivariateNormalDiag(
        loc=tf.zeros([3], dtype=tf.float32), scale_identity_multiplier=[3., 2.])
    self.assertAllClose(
        np.array([[3., 3, 3], [2, 2, 2]])**2., self.evaluate(mvn.variance()))

    mvn = tfd.MultivariateNormalDiag(
        loc=tf.zeros([3], dtype=tf.float32), scale_diag=[[3., 2, 1], [4, 5, 6]])
    self.assertAllClose(
        np.array([[3., 2, 1], [4, 5, 6]])**2., self.evaluate(mvn.variance()))

  def testStddev(self):
    mvn = tfd.MultivariateNormalDiag(loc=tf.zeros([2, 3], dtype=tf.float32))
    self.assertAllClose(
        np.ones([3], dtype=np.float32), self.evaluate(mvn.stddev()))

    mvn = tfd.MultivariateNormalDiag(
        loc=tf.zeros([3], dtype=tf.float32), scale_identity_multiplier=[3., 2.])
    self.assertAllClose(
        np.array([[3., 3, 3], [2, 2, 2]]), self.evaluate(mvn.stddev()))

    mvn = tfd.MultivariateNormalDiag(
        loc=tf.zeros([3], dtype=tf.float32), scale_diag=[[3., 2, 1], [4, 5, 6]])
    self.assertAllClose(
        np.array([[3., 2, 1], [4, 5, 6]]), self.evaluate(mvn.stddev()))

  def testMultivariateNormalDiagWithSoftplusScale(self):
    mu = [-1.0, 1.0]
    diag = [-1.0, -2.0]
    dist = tfd.MultivariateNormalDiagWithSoftplusScale(
        mu, diag, validate_args=True)
    samps = self.evaluate(dist.sample(1000, seed=tfp_test_util.test_seed()))
    cov_mat = self.evaluate(tf.linalg.diag(tf.nn.softplus(diag))**2)

    self.assertAllClose(mu, samps.mean(axis=0), atol=0.1)
    self.assertAllClose(cov_mat, np.cov(samps.T), atol=0.1)

  def testMultivariateNormalDiagNegLogLikelihood(self):
    num_draws = 50
    dims = 3
    x = np.zeros([num_draws, dims], dtype=np.float32)
    x_pl = tf.compat.v1.placeholder_with_default(
        input=x, shape=[None, dims], name="x")
    mu_var = tf.compat.v1.get_variable(
        name="mu",
        shape=[dims],
        dtype=tf.float32,
        initializer=tf.compat.v1.initializers.constant(1.))
    self.evaluate([tf.compat.v1.global_variables_initializer()])

    def neg_log_likelihood(mu):
      mvn = tfd.MultivariateNormalDiag(
          loc=mu, scale_diag=tf.ones(shape=[dims], dtype=tf.float32))

      # Typically you'd use `mvn.log_prob(x_pl)` which is always at least as
      # numerically stable as `tf.log(mvn.prob(x_pl))`. However in this test
      # we're testing a bug specific to `prob` and not `log_prob`;
      # http://stackoverflow.com/q/45109305. (The underlying issue was not
      # related to `Distributions` but that `reduce_prod` didn't correctly
      # handle negative indexes.)
      return -tf.reduce_sum(input_tensor=tf.math.log(mvn.prob(x_pl)))

    _, grad_neg_log_likelihood = tfp.math.value_and_gradient(
        neg_log_likelihood, mu_var)

    self.assertAllClose(
        self.evaluate(grad_neg_log_likelihood),
        np.tile(num_draws, dims),
        rtol=1e-6,
        atol=0.)

  def testDynamicBatchShape(self):
    if tf.executing_eagerly():
      return
    loc = np.float32(self._rng.rand(1, 1, 2))
    scale_diag = np.float32(self._rng.rand(1, 1, 2))
    mvn = tfd.MultivariateNormalDiag(
        loc=tf.compat.v1.placeholder_with_default(
            input=loc, shape=[None, None, 2]),
        scale_diag=tf.compat.v1.placeholder_with_default(
            input=scale_diag, shape=[None, None, 2]))
    self.assertListEqual(
        tensorshape_util.as_list(mvn.batch_shape), [None, None])
    self.assertListEqual(tensorshape_util.as_list(mvn.event_shape), [2])

  def testDynamicEventShape(self):
    if tf.executing_eagerly():
      return
    loc = np.float32(self._rng.rand(2, 3, 2))
    scale_diag = np.float32(self._rng.rand(2, 3, 2))
    mvn = tfd.MultivariateNormalDiag(
        loc=tf.compat.v1.placeholder_with_default(
            input=loc, shape=[2, 3, None]),
        scale_diag=tf.compat.v1.placeholder_with_default(
            input=scale_diag, shape=[2, 3, None]))
    self.assertListEqual(tensorshape_util.as_list(mvn.batch_shape), [2, 3])
    self.assertListEqual(tensorshape_util.as_list(mvn.event_shape), [None])

  def testKLDivIdenticalGradientDefined(self):
    dims = 3
    loc = tf.zeros([dims], dtype=tf.float32)
    def self_kl_divergence(loc):
      mvn = tfd.MultivariateNormalDiag(
          loc=loc, scale_diag=np.ones([dims], dtype=np.float32))
      return tfd.kl_divergence(mvn, mvn)
    _, gradients = self.evaluate(tfp.math.value_and_gradient(
        self_kl_divergence, loc))
    self.assertAllEqual(
        np.ones_like(gradients, dtype=np.bool),
        np.isfinite(gradients))

  def testProbForLargeDimIsNotNan(self):
    # Verifies a fix for GitHub issue #223
    # (https://github.com/tensorflow/probability/issues/223)
    loc_ = np.tile([0.], 1000)
    scale_diag_ = np.tile([.1], 1000)
    dist_test = tfp.distributions.MultivariateNormalDiag(loc_, scale_diag_)

    x_ = np.tile([1.], 1000)
    p_ = self.evaluate(dist_test.prob(x_))
    self.assertFalse(np.isnan(p_))


if __name__ == "__main__":
  tf.test.main()
