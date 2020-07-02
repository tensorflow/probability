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
"""Tests for the Cholesky LKJ distribution."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
from absl.testing import parameterized
import numpy as np
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp

from tensorflow_probability.python.internal import test_util


tfb = tfp.bijectors
tfd = tfp.distributions


@test_util.test_all_tf_execution_regimes
@parameterized.parameters(np.float32, np.float64)
class CholeskyLKJTest(test_util.TestCase):

  def testLogProbMatchesTransformedDistribution(self, dtype):

    # In this test, we start with a distribution supported on N x N SPD matrices
    # which factorizes as an LKJ-supported correlation matrix and a diagonal
    # of exponential random variables. The total number of independent
    # parameters is N(N-1)/2 + N = N(N+1)/2. Using the CholeskyOuterProduct
    # bijector (which requires N(N+1)/2 independent parameters), we map it to
    # the space of lower triangular Cholesky factors. We show that the resulting
    # distribution factorizes as the product of CholeskyLKJ and Rayleigh
    # distributions.

    # Given a sample from the 'LKJ + Exponential' distribution on SPD matrices,
    # and the corresponding log prob, returns the transformed sample and log
    # prob in Cholesky-space; which is furthermore factored into a diagonal
    # matrix and a Cholesky factor of a correlation matrix.
    def _get_transformed_sample_and_log_prob(lkj_exp_sample, lkj_exp_log_prob,
                                             dimension):
      # We change variables to the space of SPD matrices parameterized by the
      # lower triangular entries (including the diagonal) of \Sigma.
      # The transformation is given by \Sigma = \sqrt{S} P \sqrt{S}.
      #
      # The jacobian matrix J of the forward transform has the block form
      # [[I, 0]; [*, D]]; where I is the N x N identity matrix; and D is an
      # N*(N - 1) square diagonal matrix and * need not be computed.
      # Here, N is the dimension.  D_ij (0<=i<j<n) equals:
      #  d/d(P_ij) \Sigma_ij = d/d(P_ij) \sqrt{S_i S_j} P_ij = \sqrt{S_i S_j}.
      # Hence, detJ = \prod_ij (i < j) \sqrt{S_i S_j}  [N(N-1)/2 terms]
      #             = \prod_i S_i^{.5 * (N - 1)}
      # [1] Zhenxun Wang and Yunan Wu and Haitao Chu
      # 'On equivalence of the LKJ distribution and the restricted Wishart
      # distribution'. 2018
      exp_variance, lkj_corr = lkj_exp_sample
      sqrt_exp_variance = tf.math.sqrt(exp_variance[..., tf.newaxis])
      sigma_sample = (tf.linalg.matrix_transpose(sqrt_exp_variance) *
                      lkj_corr * sqrt_exp_variance)
      sigma_log_prob = lkj_exp_log_prob - .5 * (dimension - 1) * tf.reduce_sum(
          tf.math.log(tf.linalg.diag_part(sigma_sample)), axis=-1)

      # We change variables again to lower triangular L; where LL^T = \Sigma.
      # This is just inverse of the tfb.CholeskyOuterProduct bijector.
      cholesky_sigma_sample = tf.linalg.cholesky(sigma_sample)
      cholesky_sigma_log_prob = sigma_log_prob + tfb.Invert(
          tfb.CholeskyOuterProduct()).inverse_log_det_jacobian(
              cholesky_sigma_sample, event_ndims=2)

      # Change of variables to R, A; where L = RA; R is diagonal matrix
      # with each dimension's standard deviation and A is a Cholesky factor of a
      # correlation matrix. A is parameterized by its strictly lower triangular
      # entries; i.e. N(N-1)/2 entries.
      #
      # The jacobian determinant is the product of each row's transformation, as
      # each row is transformed independently as R_ii = ||L_i|| and
      # A_ij = L_ij/R_ii. Here ||...|| denotes Euclidean norm. Direct
      # computation shows that the jacobian determinant for the ith row is
      # R^{i-1} / A_ii.
      std_dev_sample = tf.linalg.norm(cholesky_sigma_sample, axis=-1)
      cholesky_corr_sample = (
          cholesky_sigma_sample / std_dev_sample[..., tf.newaxis])

      cholesky_corr_std_dev_sample = (std_dev_sample, cholesky_corr_sample)
      cholesky_corr_std_dev_log_prob = (
          cholesky_sigma_log_prob + tf.reduce_sum(
              tf.range(dimension, dtype=dtype) * tf.math.log(std_dev_sample) -
              tf.math.log(tf.linalg.diag_part(cholesky_corr_sample)),
              axis=-1))

      return cholesky_corr_std_dev_sample, cholesky_corr_std_dev_log_prob

    for dimension in (2, 3, 4, 5):
      rate = np.linspace(.5, 2., 10, dtype=dtype)
      concentration = np.linspace(2., 5., 10, dtype=dtype)

      # We start with a distribution on SPD matrices given by the product of
      # LKJ and Exponential random variables.
      lkj_exponential_covariance_dist = tfd.JointDistributionSequential([
          tfd.Sample(tfd.Exponential(rate=rate), sample_shape=dimension),
          tfd.LKJ(dimension=dimension, concentration=concentration)
      ])
      x = self.evaluate(
          lkj_exponential_covariance_dist.sample(
              10, seed=test_util.test_seed()))

      # We transform a sample from the space of SPD matrices to the space of its
      # lower triangular Cholesky factors. We decompose it into the product of a
      # diagonal matrix and a Cholesky factor of a correlation matrix.
      transformed_x, transformed_log_prob = (
          _get_transformed_sample_and_log_prob(
              x, lkj_exponential_covariance_dist.log_prob(x), dimension))

      # We now show that the transformation resulted in a distribution which
      # factors as the product of a rayleigh (the square root of an exponential)
      # and a CholeskyLKJ distribution with the same parameters as the LKJ.
      rayleigh_dist = tfd.TransformedDistribution(
          bijector=tfb.Invert(tfb.Square()),
          distribution=tfd.Exponential(rate=rate))
      cholesky_lkj_rayleigh_dist = tfd.JointDistributionSequential([
          tfd.Sample(rayleigh_dist, sample_shape=dimension),
          tfd.CholeskyLKJ(dimension=dimension, concentration=concentration)
      ])
      self.assertAllClose(
          self.evaluate(transformed_log_prob),
          self.evaluate(cholesky_lkj_rayleigh_dist.log_prob(transformed_x)),
          rtol=1e-3 if dtype == np.float32 else 1e-6)

  def testDimensionGuard(self, dtype):
    testee_lkj = tfd.CholeskyLKJ(
        dimension=3, concentration=dtype([1., 4.]))
    with self.assertRaisesRegexp(ValueError, 'dimension mismatch'):
      testee_lkj.log_prob(tf.eye(4))

  def testZeroDimension(self, dtype):
    testee_lkj = tfd.CholeskyLKJ(
        dimension=0, concentration=dtype([1., 4.]), validate_args=True)
    results = testee_lkj.sample(sample_shape=[4, 3], seed=test_util.test_seed())
    self.assertEqual(results.shape, [4, 3, 2, 0, 0])

  def testOneDimension(self, dtype):
    testee_lkj = tfd.CholeskyLKJ(
        dimension=1, concentration=dtype([1., 4.]), validate_args=True)
    results = testee_lkj.sample(sample_shape=[4, 3], seed=test_util.test_seed())
    self.assertEqual(results.shape, [4, 3, 2, 1, 1])

  def testValidateLowerTriangularInput(self, dtype):
    testee_lkj = tfd.CholeskyLKJ(
        dimension=2, concentration=dtype(4.), validate_args=True)
    with self.assertRaisesOpError('must be lower triangular'):
      self.evaluate(testee_lkj.log_prob(dtype([[1., 1.], [1., 1.]])))

  def testValidateConcentration(self, dtype):
    dimension = 3
    concentration = tf.Variable(0.5, dtype=dtype)
    d = tfd.CholeskyLKJ(dimension, concentration, validate_args=True)
    with self.assertRaisesOpError('Argument `concentration` must be >= 1.'):
      self.evaluate([v.initializer for v in d.variables])
      self.evaluate(d.sample(seed=test_util.test_seed()))

  def testValidateConcentrationAfterMutation(self, dtype):
    dimension = 3
    concentration = tf.Variable(1.5, dtype=dtype)
    d = tfd.CholeskyLKJ(dimension, concentration, validate_args=True)
    self.evaluate([v.initializer for v in d.variables])
    with self.assertRaisesOpError('Argument `concentration` must be >= 1.'):
      with tf.control_dependencies([concentration.assign(0.5)]):
        self.evaluate(d.sample(seed=test_util.test_seed()))


if __name__ == '__main__':
  tf.test.main()
