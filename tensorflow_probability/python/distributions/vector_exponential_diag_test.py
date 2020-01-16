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
"""Tests for VectorExponentialLinearOperator."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports

import numpy as np
import tensorflow.compat.v2 as tf
from tensorflow_probability.python import distributions as tfd
from tensorflow_probability.python.internal import test_util


@test_util.test_all_tf_execution_regimes
class VectorExponentialDiagTest(test_util.TestCase):
  """Well tested because this is a simple override of the base class."""

  def setUp(self):
    self._rng = np.random.RandomState(42)

  def testScalarParams(self):
    mu = -1.
    diag = -5.
    with self.assertRaisesRegexp(ValueError, 'at least 1 dimension'):
      tfd.VectorExponentialDiag(mu, diag)

  def testVectorParams(self):
    mu = [-1.]
    diag = [-5.]
    dist = tfd.VectorExponentialDiag(mu, diag, validate_args=True)
    self.assertAllEqual([3, 1], dist.sample(
        3, seed=test_util.test_seed()).shape)

  def testMean(self):
    mu = [-1., 1]
    diag = [1., -5]
    dist = tfd.VectorExponentialDiag(mu, diag, validate_args=True)
    self.assertAllEqual([-1. + 1., 1. - 5.], self.evaluate(dist.mean()))

  def testMode(self):
    mu = [-1.]
    diag = [1., -5]
    dist = tfd.VectorExponentialDiag(mu, diag, validate_args=True)
    self.assertAllEqual([-1., -1.], self.evaluate(dist.mode()))

  def testMeanWithBroadcastLoc(self):
    mu = [-1.]
    diag = [1., -5]
    dist = tfd.VectorExponentialDiag(mu, diag, validate_args=True)
    self.assertAllEqual([-1. + 1, -1. - 5], self.evaluate(dist.mean()))

  def testSample(self):
    mu = np.array([-2., 1])
    diag = np.array([1., -2])
    dist = tfd.VectorExponentialDiag(mu, diag, validate_args=True)
    samps = self.evaluate(dist.sample(int(1e4), seed=test_util.test_seed()))
    cov_mat = self.evaluate(tf.linalg.diag(diag))**2

    self.assertAllClose(
        [-2 + 1, 1. - 2], samps.mean(axis=0), atol=0., rtol=0.05)
    self.assertAllClose(cov_mat, np.cov(samps.T), atol=0.05, rtol=0.05)

  def testAssertValidSample(self):
    v = tfd.VectorExponentialDiag(loc=[4., 5.], validate_args=True)
    with self.assertRaisesOpError('Sample is not contained in the support.'):
      self.evaluate(v.log_prob([3., 4.]))

  def testSingularScaleRaises(self):
    mu = [-1., 1]
    diag = [1., 0]
    dist = tfd.VectorExponentialDiag(mu, diag, validate_args=True)
    with self.assertRaisesOpError('Singular'):
      self.evaluate(dist.sample(seed=test_util.test_seed()))

  def testSampleWithBroadcastScale(self):
    # mu corresponds to a 2-batch of 3-variate normals
    mu = np.zeros([2, 3])

    # diag corresponds to no batches of 3-variate normals
    diag = np.ones([3])

    dist = tfd.VectorExponentialDiag(mu, diag, validate_args=True)

    mean = dist.mean()
    self.assertAllEqual([2, 3], mean.shape)
    self.assertAllClose(mu + diag, self.evaluate(mean))

    n = int(1e4)
    samps = self.evaluate(dist.sample(n, seed=test_util.test_seed()))
    samps_centered = samps - samps.mean(axis=0)
    cov_mat = self.evaluate(tf.linalg.diag(diag))**2
    sample_cov = np.matmul(
        samps_centered.transpose([1, 2, 0]), samps_centered.transpose([1, 0, 2
                                                                      ])) / n

    self.assertAllClose(mu + diag, samps.mean(axis=0), atol=0.10, rtol=0.05)
    self.assertAllClose([cov_mat, cov_mat], sample_cov, atol=0.10, rtol=0.05)

  def testCovariance(self):
    vex = tfd.VectorExponentialDiag(
        loc=tf.ones([2, 3], dtype=tf.float32), validate_args=True)
    self.assertAllClose(
        np.diag(np.ones([3], dtype=np.float32)),
        self.evaluate(vex.covariance()))

    vex = tfd.VectorExponentialDiag(
        loc=tf.ones([3], dtype=tf.float32),
        scale_identity_multiplier=[3., 2.],
        validate_args=True)
    self.assertAllEqual([2], vex.batch_shape)
    self.assertAllEqual([3], vex.event_shape)
    self.assertAllClose(
        np.array([[[3., 0, 0], [0, 3, 0], [0, 0, 3]], [[2, 0, 0], [0, 2, 0],
                                                       [0, 0, 2]]])**2.,
        self.evaluate(vex.covariance()))

    vex = tfd.VectorExponentialDiag(
        loc=tf.ones([3], dtype=tf.float32),
        scale_diag=[[3., 2, 1], [4, 5, 6]],
        validate_args=True)
    self.assertAllEqual([2], vex.batch_shape)
    self.assertAllEqual([3], vex.event_shape)
    self.assertAllClose(
        np.array([[[3., 0, 0], [0, 2, 0], [0, 0, 1]], [[4, 0, 0], [0, 5, 0],
                                                       [0, 0, 6]]])**2.,
        self.evaluate(vex.covariance()))

  def testVariance(self):
    vex = tfd.VectorExponentialDiag(
        loc=tf.zeros([2, 3], dtype=tf.float32), validate_args=True)
    self.assertAllClose(
        np.ones([3], dtype=np.float32), self.evaluate(vex.variance()))

    vex = tfd.VectorExponentialDiag(
        loc=tf.ones([3], dtype=tf.float32),
        scale_identity_multiplier=[3., 2.],
        validate_args=True)
    self.assertAllClose(
        np.array([[3., 3, 3], [2., 2, 2]])**2., self.evaluate(vex.variance()))

    vex = tfd.VectorExponentialDiag(
        loc=tf.ones([3], dtype=tf.float32),
        scale_diag=[[3., 2, 1], [4., 5, 6]],
        validate_args=True)
    self.assertAllClose(
        np.array([[3., 2, 1], [4., 5, 6]])**2., self.evaluate(vex.variance()))

  def testStddev(self):
    vex = tfd.VectorExponentialDiag(
        loc=tf.zeros([2, 3], dtype=tf.float32), validate_args=True)
    self.assertAllClose(
        np.ones([3], dtype=np.float32), self.evaluate(vex.stddev()))

    vex = tfd.VectorExponentialDiag(
        loc=tf.zeros([3], dtype=tf.float32),
        scale_identity_multiplier=[3., 2.],
        validate_args=True)
    self.assertAllClose(
        np.array([[3., 3, 3], [2., 2, 2]]), self.evaluate(vex.stddev()))

    vex = tfd.VectorExponentialDiag(
        loc=tf.zeros([3], dtype=tf.float32),
        scale_diag=[[3., 2, 1], [4, 5, 6]],
        validate_args=True)
    self.assertAllClose(
        np.array([[3., 2, 1], [4., 5, 6]]), self.evaluate(vex.stddev()))

  def testSupportBijectorOutsideRange(self):
    vex = tfd.VectorExponentialDiag(
        loc=tf.ones([3], dtype=tf.float32),
        scale_diag=[[3., 2., 1.], [4., 5., 6.]],
        validate_args=True)
    x = np.array([[-8.3, -0.4, -1e-6]])
    bijector_inverse_x = vex._experimental_default_event_space_bijector(
        ).inverse(x)
    self.assertAllNan(self.evaluate(bijector_inverse_x))


if __name__ == '__main__':
  tf.test.main()
