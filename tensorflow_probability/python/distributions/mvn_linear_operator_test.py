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
"""Tests for MultivariateNormalLinearOperator."""

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


@test_util.test_all_tf_execution_regimes
class MultivariateNormalLinearOperatorTest(test_util.TestCase):

  def setUp(self):
    self.rng = np.random.RandomState(42)
    super(MultivariateNormalLinearOperatorTest, self).setUp()

  def _random_tril_matrix(self, shape):
    mat = self.rng.rand(*shape)
    chol = tfb.TransformDiagonal(tfb.Softplus())(mat)
    return tf.linalg.band_part(chol, -1, 0)

  def _random_loc_and_scale(self, batch_shape, event_shape):
    # This ensures covariance is positive def.
    mat_shape = batch_shape + event_shape + event_shape
    scale = tf.linalg.LinearOperatorLowerTriangular(
        self._random_tril_matrix(mat_shape),
        is_non_singular=True)
    loc_shape = batch_shape + event_shape
    loc = self.rng.randn(*loc_shape)
    return loc, scale

  def testNamePropertyIsSetByInitArg(self):
    loc = [1., 2.]
    scale = tf.linalg.LinearOperatorIdentity(2)
    mvn = tfd.MultivariateNormalLinearOperator(
        loc, scale, name='Billy', validate_args=True)
    self.assertStartsWith(mvn.name, 'Billy')

  def testLogPDFScalarBatch(self):
    loc = self.rng.rand(2)
    scale = tf.linalg.LinearOperatorLowerTriangular(
        self._random_tril_matrix([2, 2]), is_non_singular=True)
    mvn = tfd.MultivariateNormalLinearOperator(loc, scale, validate_args=True)
    x = self.rng.rand(2)

    log_pdf = mvn.log_prob(x)
    pdf = mvn.prob(x)

    covariance = self.evaluate(
        tf.matmul(scale.to_dense(), scale.to_dense(), adjoint_b=True))
    scipy_mvn = stats.multivariate_normal(mean=loc, cov=covariance)

    expected_log_pdf = scipy_mvn.logpdf(x)
    expected_pdf = scipy_mvn.pdf(x)
    self.assertEqual((), log_pdf.shape)
    self.assertEqual((), pdf.shape)
    self.assertAllClose(expected_log_pdf, self.evaluate(log_pdf))
    self.assertAllClose(expected_pdf, self.evaluate(pdf))

  def testRaisesIfScaleNotProvided(self):
    loc = self.rng.rand(2)
    with self.assertRaises(ValueError):
      tfd.MultivariateNormalLinearOperator(loc, scale=None, validate_args=True)

  def testShapes(self):
    loc = self.rng.rand(3, 5, 2)
    scale = tf.linalg.LinearOperatorLowerTriangular(
        self._random_tril_matrix([3, 5, 2, 2]), is_non_singular=True)

    mvn = tfd.MultivariateNormalLinearOperator(loc, scale, validate_args=True)

    # Shapes known at graph construction time.
    self.assertEqual((2,), tuple(tensorshape_util.as_list(mvn.event_shape)))
    self.assertEqual((3, 5), tuple(tensorshape_util.as_list(mvn.batch_shape)))

    # Shapes known at runtime.
    self.assertEqual((2,), tuple(self.evaluate(mvn.event_shape_tensor())))
    self.assertEqual((3, 5), tuple(self.evaluate(mvn.batch_shape_tensor())))

  def testMeanAndCovariance(self):
    loc, scale = self._random_loc_and_scale(
        batch_shape=[3, 4], event_shape=[5])
    mvn = tfd.MultivariateNormalLinearOperator(loc, scale, validate_args=True)

    self.assertAllEqual(self.evaluate(mvn.mean()), loc)
    self.assertAllClose(
        self.evaluate(mvn.covariance()),
        np.matmul(
            self.evaluate(scale.to_dense()),
            np.transpose(self.evaluate(scale.to_dense()), [0, 1, 3, 2])))

  @test_util.tf_tape_safety_test
  def testVariableLocation(self):
    loc = tf.Variable([1., 1.])
    scale = tf.linalg.LinearOperatorLowerTriangular(
        tf.eye(2), is_non_singular=True)
    d = tfd.MultivariateNormalLinearOperator(loc, scale, validate_args=True)
    self.evaluate(loc.initializer)
    with tf.GradientTape() as tape:
      lp = d.log_prob([0., 0.])
    self.assertIsNotNone(tape.gradient(lp, loc))

  @test_util.jax_disable_variable_test
  def testVariableScaleAssertions(self):
    # We test that changing the scale to be non-invertible raises an exception
    # when validate_args is True. This is really just testing the underlying
    # AffineLinearOperator instance, but we include it to demonstrate that it
    # works as expected.
    loc = tf.constant([1., 1.])
    scale_tensor = tf.Variable(np.eye(2, dtype=np.float32))
    scale = tf.linalg.LinearOperatorLowerTriangular(
        scale_tensor,
        is_non_singular=True)
    d = tfd.MultivariateNormalLinearOperator(loc, scale, validate_args=True)
    self.evaluate(scale_tensor.initializer)
    with self.assertRaises(Exception):
      with tf.control_dependencies([scale_tensor.assign([[1., 0.], [1., 0.]])]):
        self.evaluate(d.sample(seed=test_util.test_seed()))

  def testKLBatch(self):
    batch_shape = [2]
    event_shape = [3]
    loc_a, scale_a = self._random_loc_and_scale(batch_shape, event_shape)
    loc_b, scale_b = self._random_loc_and_scale(batch_shape, event_shape)
    mvn_a = tfd.MultivariateNormalLinearOperator(
        loc=loc_a, scale=scale_a, validate_args=True)
    mvn_b = tfd.MultivariateNormalLinearOperator(
        loc=loc_b, scale=scale_b, validate_args=True)

    kl = tfd.kl_divergence(mvn_a, mvn_b)
    self.assertEqual(batch_shape, kl.shape)

    kl_v = self.evaluate(kl)
    expected_kl_0 = self._compute_non_batch_kl(
        loc_a[0, :],
        self.evaluate(scale_a.to_dense())[0, :, :], loc_b[0, :],
        self.evaluate(scale_b.to_dense())[0, :])
    expected_kl_1 = self._compute_non_batch_kl(
        loc_a[1, :],
        self.evaluate(scale_a.to_dense())[1, :, :], loc_b[1, :],
        self.evaluate(scale_b.to_dense())[1, :])
    self.assertAllClose(expected_kl_0, kl_v[0])
    self.assertAllClose(expected_kl_1, kl_v[1])

  def testKLBatchBroadcast(self):
    batch_shape = [2]
    event_shape = [3]
    loc_a, scale_a = self._random_loc_and_scale(batch_shape, event_shape)
    # No batch shape.
    loc_b, scale_b = self._random_loc_and_scale([], event_shape)
    mvn_a = tfd.MultivariateNormalLinearOperator(
        loc=loc_a, scale=scale_a, validate_args=True)
    mvn_b = tfd.MultivariateNormalLinearOperator(
        loc=loc_b, scale=scale_b, validate_args=True)

    kl = tfd.kl_divergence(mvn_a, mvn_b)
    self.assertEqual(batch_shape, kl.shape)

    kl_v = self.evaluate(kl)
    expected_kl_0 = self._compute_non_batch_kl(
        loc_a[0, :],
        self.evaluate(scale_a.to_dense())[0, :, :], loc_b,
        self.evaluate(scale_b.to_dense()))
    expected_kl_1 = self._compute_non_batch_kl(
        loc_a[1, :],
        self.evaluate(scale_a.to_dense())[1, :, :], loc_b,
        self.evaluate(scale_b.to_dense()))
    self.assertAllClose(expected_kl_0, kl_v[0])
    self.assertAllClose(expected_kl_1, kl_v[1])

  def _compute_non_batch_kl(self, loc_a, scale_a, loc_b, scale_b):
    """Non-batch KL for N(loc_a, scale_a), N(loc_b, scale_b)."""
    # Check using numpy operations
    # This mostly repeats the tensorflow code _kl_mvn_mvn(), but in numpy.
    # So it is important to also check that KL(mvn, mvn) = 0.
    covariance_a = np.dot(scale_a, scale_a.T)
    covariance_b = np.dot(scale_b, scale_b.T)
    covariance_b_inv = np.linalg.inv(covariance_b)

    t = np.trace(covariance_b_inv.dot(covariance_a))
    q = (loc_b - loc_a).dot(covariance_b_inv).dot(loc_b - loc_a)
    k = loc_a.shape[0]
    l = np.log(np.linalg.det(covariance_b) / np.linalg.det(covariance_a))

    return 0.5 * (t + q - k + l)


if __name__ == '__main__':
  tf.test.main()
