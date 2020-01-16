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
"""Tests for MultivariateNormalFullCovariance."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports

import numpy as np
from scipy import stats
import tensorflow.compat.v2 as tf
from tensorflow_probability.python import bijectors as tfb
from tensorflow_probability.python import distributions as tfd
from tensorflow_probability.python.internal import tensorshape_util
from tensorflow_probability.python.internal import test_util


rng = np.random.RandomState(42)


@test_util.test_all_tf_execution_regimes
class MultivariateNormalFullCovarianceTest(test_util.TestCase):

  def _random_pd_matrix(self, *shape):
    mat = rng.rand(*shape)
    chol = tfb.TransformDiagonal(tfb.Softplus())(mat)
    chol = tf.linalg.band_part(chol, -1, 0)
    return self.evaluate(tf.matmul(chol, chol, adjoint_b=True))

  def testRaisesIfInitializedWithNonSymmetricMatrix(self):
    mu = [1., 2.]
    sigma = [[1., 0.], [1., 1.]]  # Nonsingular, but not symmetric
    with self.assertRaisesOpError("not symmetric"):
      mvn = tfd.MultivariateNormalFullCovariance(mu, sigma, validate_args=True)
      self.evaluate(mvn.covariance())

  def testNamePropertyIsSetByInitArg(self):
    mu = [1., 2.]
    sigma = [[1., 0.], [0., 1.]]
    mvn = tfd.MultivariateNormalFullCovariance(
        mu, sigma, name="Billy", validate_args=True)
    self.assertStartsWith(mvn.name, "Billy")

  def testDoesNotRaiseIfInitializedWithSymmetricMatrix(self):
    mu = rng.rand(10)
    sigma = self._random_pd_matrix(10, 10)
    mvn = tfd.MultivariateNormalFullCovariance(mu, sigma, validate_args=True)
    # Should not raise
    self.evaluate(mvn.covariance())

  def testLogPDFScalarBatch(self):
    mu = rng.rand(2)
    sigma = self._random_pd_matrix(2, 2)
    mvn = tfd.MultivariateNormalFullCovariance(mu, sigma, validate_args=True)
    x = rng.rand(2)

    log_pdf = mvn.log_prob(x)
    pdf = mvn.prob(x)

    scipy_mvn = stats.multivariate_normal(mean=mu, cov=sigma)

    expected_log_pdf = scipy_mvn.logpdf(x)
    expected_pdf = scipy_mvn.pdf(x)
    self.assertEqual((), log_pdf.shape)
    self.assertEqual((), pdf.shape)
    self.assertAllClose(expected_log_pdf, self.evaluate(log_pdf))
    self.assertAllClose(expected_pdf, self.evaluate(pdf))

  def testLogPDFScalarBatchCovarianceNotProvided(self):
    mu = rng.rand(2)
    mvn = tfd.MultivariateNormalFullCovariance(
        mu, covariance_matrix=None, validate_args=True)
    x = rng.rand(2)

    log_pdf = mvn.log_prob(x)
    pdf = mvn.prob(x)

    # Initialize a scipy_mvn with the default covariance.
    scipy_mvn = stats.multivariate_normal(mean=mu, cov=np.eye(2))

    expected_log_pdf = scipy_mvn.logpdf(x)
    expected_pdf = scipy_mvn.pdf(x)
    self.assertEqual((), log_pdf.shape)
    self.assertEqual((), pdf.shape)
    self.assertAllClose(expected_log_pdf, self.evaluate(log_pdf))
    self.assertAllClose(expected_pdf, self.evaluate(pdf))

  def testShapes(self):
    mu = rng.rand(3, 5, 2)
    covariance = self._random_pd_matrix(3, 5, 2, 2)

    mvn = tfd.MultivariateNormalFullCovariance(
        mu, covariance, validate_args=True)

    # Shapes known at graph construction time.
    self.assertEqual((2,), tuple(tensorshape_util.as_list(mvn.event_shape)))
    self.assertEqual((3, 5), tuple(tensorshape_util.as_list(mvn.batch_shape)))

    # Shapes known at runtime.
    self.assertEqual((2,), tuple(self.evaluate(mvn.event_shape_tensor())))
    self.assertEqual((3, 5), tuple(self.evaluate(mvn.batch_shape_tensor())))

  def _random_mu_and_sigma(self, batch_shape, event_shape):
    # This ensures sigma is positive def.
    mat_shape = batch_shape + event_shape + event_shape
    mat = rng.randn(*mat_shape)
    perm = np.arange(mat.ndim)
    perm[-2:] = [perm[-1], perm[-2]]
    sigma = np.matmul(mat, np.transpose(mat, perm))

    mu_shape = batch_shape + event_shape
    mu = rng.randn(*mu_shape)

    return mu, sigma

  def testKLBatch(self):
    batch_shape = [2]
    event_shape = [3]
    mu_a, sigma_a = self._random_mu_and_sigma(batch_shape, event_shape)
    mu_b, sigma_b = self._random_mu_and_sigma(batch_shape, event_shape)
    mvn_a = tfd.MultivariateNormalFullCovariance(
        loc=mu_a, covariance_matrix=sigma_a, validate_args=True)
    mvn_b = tfd.MultivariateNormalFullCovariance(
        loc=mu_b, covariance_matrix=sigma_b, validate_args=True)

    kl = tfd.kl_divergence(mvn_a, mvn_b)
    self.assertEqual(batch_shape, kl.shape)

    kl_v = self.evaluate(kl)
    expected_kl_0 = _compute_non_batch_kl(mu_a[0, :], sigma_a[0, :, :],
                                          mu_b[0, :], sigma_b[0, :])
    expected_kl_1 = _compute_non_batch_kl(mu_a[1, :], sigma_a[1, :, :],
                                          mu_b[1, :], sigma_b[1, :])
    self.assertAllClose(expected_kl_0, kl_v[0])
    self.assertAllClose(expected_kl_1, kl_v[1])

  def testKLBatchBroadcast(self):
    batch_shape = [2]
    event_shape = [3]
    mu_a, sigma_a = self._random_mu_and_sigma(batch_shape, event_shape)
    # No batch shape.
    mu_b, sigma_b = self._random_mu_and_sigma([], event_shape)
    mvn_a = tfd.MultivariateNormalFullCovariance(
        loc=mu_a, covariance_matrix=sigma_a, validate_args=True)
    mvn_b = tfd.MultivariateNormalFullCovariance(
        loc=mu_b, covariance_matrix=sigma_b, validate_args=True)

    kl = tfd.kl_divergence(mvn_a, mvn_b)
    self.assertEqual(batch_shape, kl.shape)

    kl_v = self.evaluate(kl)
    expected_kl_0 = _compute_non_batch_kl(mu_a[0, :], sigma_a[0, :, :], mu_b,
                                          sigma_b)
    expected_kl_1 = _compute_non_batch_kl(mu_a[1, :], sigma_a[1, :, :], mu_b,
                                          sigma_b)
    self.assertAllClose(expected_kl_0, kl_v[0])
    self.assertAllClose(expected_kl_1, kl_v[1])


def _compute_non_batch_kl(mu_a, sigma_a, mu_b, sigma_b):
  """Non-batch KL for N(mu_a, sigma_a), N(mu_b, sigma_b)."""
  # Check using numpy operations
  # This mostly repeats the tensorflow code _kl_mvn_mvn(), but in numpy.
  # So it is important to also check that KL(mvn, mvn) = 0.
  sigma_b_inv = np.linalg.inv(sigma_b)

  t = np.trace(sigma_b_inv.dot(sigma_a))
  q = (mu_b - mu_a).dot(sigma_b_inv).dot(mu_b - mu_a)
  k = mu_a.shape[0]
  l = np.log(np.linalg.det(sigma_b) / np.linalg.det(sigma_a))

  return 0.5 * (t + q - k + l)


if __name__ == "__main__":
  tf.test.main()
