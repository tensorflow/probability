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
"""Tests for VectorLaplaceLinearOperator."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from tensorflow_probability.python.internal import test_util as tfp_test_util
from tensorflow.python.framework import test_util  # pylint: disable=g-direct-tensorflow-import,g-import-not-at-top

tfd = tfp.distributions


@test_util.run_all_in_graph_and_eager_modes
class VectorLaplaceDiagTest(tf.test.TestCase):
  """Well tested because this is a simple override of the base class."""

  def setUp(self):
    super(VectorLaplaceDiagTest, self).setUp()
    self._rng = np.random.RandomState(42)

  def testScalarParams(self):
    mu = -1.
    diag = -5.
    with self.assertRaisesRegexp(ValueError, "at least 1 dimension"):
      tfd.VectorLaplaceDiag(mu, diag)

  def testVectorParams(self):
    mu = [-1.]
    diag = [-5.]
    dist = tfd.VectorLaplaceDiag(mu, diag, validate_args=True)
    self.assertAllEqual([3, 1], dist.sample(3).shape)

  def testDistWithBatchShapeOneThenTransformedThroughSoftplus(self):
    # This complex combination of events resulted in a loss of static shape
    # information when tf.get_static_value(self._needs_rotation) was
    # being used incorrectly (resulting in always rotating).
    # Batch shape = [1], event shape = [3]
    mu = tf.zeros((1, 3))
    diag = tf.ones((1, 3))
    base_dist = tfd.VectorLaplaceDiag(mu, diag, validate_args=True)
    dist = tfd.TransformedDistribution(
        base_dist, validate_args=True, bijector=tfp.bijectors.Softplus())
    samps = dist.sample(5)  # Shape [5, 1, 3].
    self.assertAllEqual([5, 1], dist.log_prob(samps).shape)

  def testMean(self):
    mu = [-1., 1]
    diag = [1., -5]
    dist = tfd.VectorLaplaceDiag(mu, diag, validate_args=True)
    self.assertAllEqual(mu, self.evaluate(dist.mean()))

  def testMeanWithBroadcastLoc(self):
    mu = [-1.]
    diag = [1., -5]
    dist = tfd.VectorLaplaceDiag(mu, diag, validate_args=True)
    self.assertAllEqual([-1., -1.], self.evaluate(dist.mean()))

  def testSample(self):
    mu = [-1., 1]
    diag = [1., -2]
    dist = tfd.VectorLaplaceDiag(mu, diag, validate_args=True)
    seed = tfp_test_util.test_seed(hardcoded_seed=0, set_eager_seed=False)
    samps = self.evaluate(dist.sample(int(1e4), seed=seed))
    cov_mat = 2. * self.evaluate(tf.linalg.diag(diag))**2

    self.assertAllClose(mu, samps.mean(axis=0), atol=0., rtol=0.05)
    self.assertAllClose(cov_mat, np.cov(samps.T), atol=0.05, rtol=0.05)

  def testSingularScaleRaises(self):
    mu = [-1., 1]
    diag = [1., 0]
    dist = tfd.VectorLaplaceDiag(mu, diag, validate_args=True)
    with self.assertRaisesOpError("Singular"):
      self.evaluate(dist.sample())

  def testSampleWithBroadcastScale(self):
    # mu corresponds to a 2-batch of 3-variate normals
    mu = np.zeros([2, 3])

    # diag corresponds to no batches of 3-variate normals
    diag = np.ones([3])

    dist = tfd.VectorLaplaceDiag(mu, diag, validate_args=True)

    mean = dist.mean()
    self.assertAllEqual([2, 3], mean.shape)
    self.assertAllClose(mu, self.evaluate(mean))

    n = int(1e4)
    samps = self.evaluate(dist.sample(n, seed=tfp_test_util.test_seed()))
    cov_mat = 2. * self.evaluate(tf.linalg.diag(diag))**2
    sample_cov = np.matmul(
        samps.transpose([1, 2, 0]), samps.transpose([1, 0, 2])) / n

    self.assertAllClose(mu, samps.mean(axis=0), atol=0.10, rtol=0.05)
    self.assertAllClose([cov_mat, cov_mat], sample_cov, atol=0.10, rtol=0.05)

  def testCovariance(self):
    vla = tfd.VectorLaplaceDiag(loc=tf.zeros([2, 3], dtype=tf.float32))
    self.assertAllClose(2. * np.diag(np.ones([3], dtype=np.float32)),
                        self.evaluate(vla.covariance()))

    vla = tfd.VectorLaplaceDiag(
        loc=tf.zeros([3], dtype=tf.float32), scale_identity_multiplier=[3., 2.])
    self.assertAllEqual([2], vla.batch_shape)
    self.assertAllEqual([3], vla.event_shape)
    self.assertAllClose(
        2. * np.array([[[3., 0, 0], [0, 3, 0], [0, 0, 3]],
                       [[2, 0, 0], [0, 2, 0], [0, 0, 2]]])**2.,
        self.evaluate(vla.covariance()))

    vla = tfd.VectorLaplaceDiag(
        loc=tf.zeros([3], dtype=tf.float32), scale_diag=[[3., 2, 1], [4, 5, 6]])
    self.assertAllEqual([2], vla.batch_shape)
    self.assertAllEqual([3], vla.event_shape)
    self.assertAllClose(
        2. * np.array([[[3., 0, 0], [0, 2, 0], [0, 0, 1]],
                       [[4, 0, 0], [0, 5, 0], [0, 0, 6]]])**2.,
        self.evaluate(vla.covariance()))

  def testVariance(self):
    vla = tfd.VectorLaplaceDiag(loc=tf.zeros([2, 3], dtype=tf.float32))
    self.assertAllClose(2. * np.ones([3], dtype=np.float32),
                        self.evaluate(vla.variance()))

    vla = tfd.VectorLaplaceDiag(
        loc=tf.zeros([3], dtype=tf.float32), scale_identity_multiplier=[3., 2.])
    self.assertAllClose(2. * np.array([[3., 3, 3], [2, 2, 2]])**2.,
                        self.evaluate(vla.variance()))

    vla = tfd.VectorLaplaceDiag(
        loc=tf.zeros([3], dtype=tf.float32), scale_diag=[[3., 2, 1], [4, 5, 6]])
    self.assertAllClose(2. * np.array([[3., 2, 1], [4, 5, 6]])**2.,
                        self.evaluate(vla.variance()))

  def testStddev(self):
    vla = tfd.VectorLaplaceDiag(loc=tf.zeros([2, 3], dtype=tf.float32))
    self.assertAllClose(
        np.sqrt(2) * np.ones([3], dtype=np.float32),
        self.evaluate(vla.stddev()))

    vla = tfd.VectorLaplaceDiag(
        loc=tf.zeros([3], dtype=tf.float32), scale_identity_multiplier=[3., 2.])
    self.assertAllClose(
        np.sqrt(2) * np.array([[3., 3, 3], [2, 2, 2]]),
        self.evaluate(vla.stddev()))

    vla = tfd.VectorLaplaceDiag(
        loc=tf.zeros([3], dtype=tf.float32), scale_diag=[[3., 2, 1], [4, 5, 6]])
    self.assertAllClose(
        np.sqrt(2) * np.array([[3., 2, 1], [4, 5, 6]]),
        self.evaluate(vla.stddev()))


if __name__ == "__main__":
  tf.test.main()
