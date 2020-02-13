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
from scipy import special as sp_special
from scipy import stats as sp_stats

import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp

from tensorflow_probability.python.internal import test_util

tfb = tfp.bijectors
tfd = tfp.distributions


@test_util.test_all_tf_execution_regimes
class DirichletTest(test_util.TestCase):

  def testSimpleShapes(self):
    alpha = np.random.rand(3)
    dist = tfd.Dirichlet(alpha)
    self.assertEqual(3, self.evaluate(dist.event_shape_tensor()))
    self.assertAllEqual([], self.evaluate(dist.batch_shape_tensor()))
    self.assertEqual(tf.TensorShape([3]), dist.event_shape)
    self.assertEqual(tf.TensorShape([]), dist.batch_shape)

  def testComplexShapes(self):
    alpha = np.random.rand(3, 2, 2)
    dist = tfd.Dirichlet(alpha)
    self.assertEqual(2, self.evaluate(dist.event_shape_tensor()))
    self.assertAllEqual([3, 2], self.evaluate(dist.batch_shape_tensor()))
    self.assertEqual(tf.TensorShape([2]), dist.event_shape)
    self.assertEqual(tf.TensorShape([3, 2]), dist.batch_shape)

  def testConcentrationProperty(self):
    alpha = [[1., 2, 3]]
    dist = tfd.Dirichlet(alpha)
    self.assertEqual([1, 3], dist.concentration.shape)
    self.assertAllClose(alpha, self.evaluate(dist.concentration))

  def testPdfXProper(self):
    alpha = [[1., 2, 3]]
    dist = tfd.Dirichlet(alpha, validate_args=True)
    self.evaluate(dist.prob([.1, .3, .6]))
    self.evaluate(dist.prob([.2, .3, .5]))
    # Either condition can trigger.
    with self.assertRaisesOpError('Samples must be non-negative'):
      self.evaluate(dist.prob([-1., 1.5, 0.5]))
    with self.assertRaisesOpError('Sample last-dimension must sum to `1`'):
      self.evaluate(dist.prob([.1, .2, .8]))

  def testLogPdfOnBoundaryIsFiniteWhenAlphaIsOne(self):
    # Test concentration = 1. for each dimension.
    concentration = 3 * np.ones((10, 10)).astype(np.float32)
    concentration[range(10), range(10)] = 1.
    x = 1 / 9. * np.ones((10, 10)).astype(np.float32)
    x[range(10), range(10)] = 0.
    dist = tfd.Dirichlet(concentration, validate_args=True)
    log_prob = self.evaluate(dist.log_prob(x))
    self.assertAllEqual(
        np.ones_like(log_prob, dtype=np.bool), np.isfinite(log_prob))

    # Test when concentration[k] = 1., and x is zero at various dimensions.
    dist = tfd.Dirichlet(10 * [1.])
    log_prob = self.evaluate(dist.log_prob(x))
    self.assertAllEqual(
        np.ones_like(log_prob, dtype=np.bool), np.isfinite(log_prob))

  def testPdfOnBoundaryWhenAlphaGreaterThanOne(self):
    dist = tfd.Dirichlet(9 * [2.])
    x = 0.125 * np.ones(9).astype(np.float32)
    x[3] = 0.
    log_pdf = self.evaluate(dist.log_prob(x))
    self.assertAllNegativeInf(log_pdf)

    pdf = self.evaluate(dist.prob(x))
    self.assertAllEqual(pdf, np.zeros_like(pdf))

  def testPdfZeroBatches(self):
    alpha = [1., 2]
    x = [.5, .5]
    dist = tfd.Dirichlet(alpha)
    pdf = dist.prob(x)
    self.assertAllClose(1., self.evaluate(pdf))
    self.assertEqual((), pdf.shape)

  def testPdfZeroBatchesNontrivialX(self):
    alpha = [1., 2]
    x = [.3, .7]
    dist = tfd.Dirichlet(alpha)
    pdf = dist.prob(x)
    self.assertAllClose(7. / 5, self.evaluate(pdf))
    self.assertEqual((), pdf.shape)

  def testPdfUniformZeroBatches(self):
    # Corresponds to a uniform distribution
    alpha = [1., 1, 1]
    x = [[.2, .5, .3], [.3, .4, .3]]
    dist = tfd.Dirichlet(alpha)
    pdf = dist.prob(x)
    self.assertAllClose([2., 2.], self.evaluate(pdf))
    self.assertAllEqual([2], pdf.shape)

  def testPdfAlphaStretchedInBroadcastWhenSameRank(self):
    alpha = [[1., 2]]
    x = [[.5, .5], [.3, .7]]
    dist = tfd.Dirichlet(alpha)
    pdf = dist.prob(x)
    self.assertAllClose([1., 7. / 5], self.evaluate(pdf))
    self.assertAllEqual([2], pdf.shape)

  def testPdfAlphaStretchedInBroadcastWhenLowerRank(self):
    alpha = [1., 2]
    x = [[.5, .5], [.2, .8]]
    pdf = tfd.Dirichlet(alpha).prob(x)
    self.assertAllClose([1., 8. / 5], self.evaluate(pdf))
    self.assertAllEqual([2], pdf.shape)

  def testPdfXStretchedInBroadcastWhenSameRank(self):
    alpha = [[1., 2], [2., 3]]
    x = [[.5, .5]]
    pdf = tfd.Dirichlet(alpha).prob(x)
    self.assertAllClose([1., 3. / 2], self.evaluate(pdf))
    self.assertAllEqual([2], pdf.shape)

  def testPdfXStretchedInBroadcastWhenLowerRank(self):
    alpha = [[1., 2], [2., 3]]
    x = [.5, .5]
    pdf = tfd.Dirichlet(alpha).prob(x)
    self.assertAllClose([1., 3. / 2], self.evaluate(pdf))
    self.assertAllEqual([2], pdf.shape)

  def testMean(self):
    alpha = [1., 2, 3]
    dirichlet = tfd.Dirichlet(concentration=alpha)
    self.assertAllEqual([3], dirichlet.mean().shape)
    expected_mean = sp_stats.dirichlet.mean(alpha)
    self.assertAllClose(expected_mean, self.evaluate(dirichlet.mean()))

  def testCovarianceFromSampling(self):
    alpha = np.array([[1., 2, 3],
                      [2.5, 4, 0.01]], dtype=np.float32)
    dist = tfd.Dirichlet(alpha)  # batch_shape=[2], event_shape=[3]
    x = dist.sample(int(250e3), seed=test_util.test_seed())
    sample_mean = tf.reduce_mean(x, axis=0)
    x_centered = x - sample_mean[None, ...]
    sample_cov = tf.reduce_mean(
        tf.matmul(x_centered[..., None], x_centered[..., None, :]), axis=0)
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

    self.assertAllClose(sample_mean_, analytic_mean, atol=0.04, rtol=0.)
    self.assertAllClose(sample_cov_, analytic_cov, atol=0.06, rtol=0.)
    self.assertAllClose(sample_var_, analytic_var, atol=0.03, rtol=0.)
    self.assertAllClose(sample_stddev_, analytic_stddev, atol=0.02, rtol=0.)

  def testVariance(self):
    alpha = [1., 2, 3]
    denominator = np.sum(alpha)**2 * (np.sum(alpha) + 1)
    dirichlet = tfd.Dirichlet(concentration=alpha)
    self.assertEqual(dirichlet.covariance().shape, (3, 3))
    expected_covariance = np.diag(sp_stats.dirichlet.var(alpha))
    expected_covariance += [[0., -2, -3], [-2, 0, -6], [-3, -6, 0]
                           ] / denominator
    self.assertAllClose(
        self.evaluate(dirichlet.covariance()), expected_covariance)

  def testMode(self):
    alpha = np.array([1.1, 2, 3])
    expected_mode = (alpha - 1) / (np.sum(alpha) - 3)
    dirichlet = tfd.Dirichlet(concentration=alpha)
    self.assertAllEqual([3], dirichlet.mode().shape)
    self.assertAllClose(expected_mode, self.evaluate(dirichlet.mode()))

  def testModeInvalid(self):
    alpha = np.array([1., 2, 3])
    dirichlet = tfd.Dirichlet(
        concentration=alpha, allow_nan_stats=False)
    with self.assertRaisesOpError('Condition x < y.*'):
      self.evaluate(dirichlet.mode())

  def testModeEnableAllowNanStats(self):
    alpha = np.array([1., 2, 3])
    dirichlet = tfd.Dirichlet(
        concentration=alpha, allow_nan_stats=True)
    expected_mode = np.zeros_like(alpha) + np.nan

    self.assertAllEqual([3], dirichlet.mode().shape)
    self.assertAllClose(expected_mode, self.evaluate(dirichlet.mode()))

  def testEntropy(self):
    alpha = [1., 2, 3]
    dirichlet = tfd.Dirichlet(concentration=alpha)
    self.assertAllEqual([], dirichlet.entropy().shape)
    expected_entropy = sp_stats.dirichlet.entropy(alpha)
    self.assertAllClose(expected_entropy, self.evaluate(dirichlet.entropy()))

  def testSample(self):
    alpha = [1., 2]
    dirichlet = tfd.Dirichlet(alpha)
    n = tf.constant(100000)
    samples = dirichlet.sample(n, seed=test_util.test_seed())
    sample_values = self.evaluate(samples)
    self.assertEqual(sample_values.shape, (100000, 2))
    self.assertTrue(np.all(sample_values > 0.0))
    self.assertLess(
        sp_stats.kstest(
            # Beta is a univariate distribution.
            sample_values[:, 0],
            sp_stats.beta(a=1., b=2.).cdf)[0],
        0.01)

  def testDirichletFullyReparameterized(self):
    alpha = tf.constant([1.0, 2.0, 3.0])
    _, grad_alpha = tfp.math.value_and_gradient(
        lambda a: tfd.Dirichlet(a).sample(  # pylint: disable=g-long-lambda
            100, seed=test_util.test_seed()), alpha)
    self.assertIsNotNone(grad_alpha)

  def testDirichletDirichletKL(self):
    conc1 = np.array([[1., 2., 3., 1.5, 2.5, 3.5],
                      [1.5, 2.5, 3.5, 4.5, 5.5, 6.5]])
    conc2 = np.array([[0.5, 1., 1.5, 2., 2.5, 3.]])

    d1 = tfd.Dirichlet(conc1)
    d2 = tfd.Dirichlet(conc2)
    x = d1.sample(int(1e4), seed=test_util.test_seed())
    kl_sample = tf.reduce_mean(d1.log_prob(x) - d2.log_prob(x), axis=0)
    kl_actual = tfd.kl_divergence(d1, d2)

    kl_sample_val = self.evaluate(kl_sample)
    kl_actual_val = self.evaluate(kl_actual)

    self.assertEqual(conc1.shape[:-1], kl_actual.shape)

    kl_expected = (
        sp_special.gammaln(np.sum(conc1, -1))
        - sp_special.gammaln(np.sum(conc2, -1))
        - np.sum(sp_special.gammaln(conc1) - sp_special.gammaln(conc2), -1)
        + np.sum((conc1 - conc2) * (sp_special.digamma(conc1) -
                                    sp_special.digamma(
                                        np.sum(conc1, -1, keepdims=True))), -1))

    self.assertAllClose(kl_expected, kl_actual_val, atol=0., rtol=1e-5)
    self.assertAllClose(kl_sample_val, kl_actual_val, atol=0., rtol=1e-1)

    # Make sure KL(d1||d1) is 0
    kl_same = self.evaluate(tfd.kl_divergence(d1, d1))
    self.assertAllClose(kl_same, np.zeros_like(kl_expected))

  def testDegenerateAlignedStridedSlice(self):
    # Corresponds to the TF fix in tensorflow/tensorflow#d9b3db0
    d = tfd.Dirichlet(tf.math.softplus(tf.zeros([2, 2, 2])))
    batch_shape = [2, 2]
    self.assertAllEqual(batch_shape, d.batch_shape)
    self.assertAllEqual(np.zeros(batch_shape)[1:0].shape,
                        d[1:0].batch_shape)

  def testSupportBijectorOutsideRange(self):
    conc = np.array([2., 4, 5])
    dist = tfd.Dirichlet(conc, validate_args=True)
    eps = 1e-5
    with self.assertRaisesOpError('must sum to `1`'):
      self.evaluate(dist._experimental_default_event_space_bijector(
          ).inverse([0.2, 0.5 + eps, 0.3]))

    with self.assertRaisesOpError('must be non-negative|must sum to `1`'):
      self.evaluate(dist._experimental_default_event_space_bijector(
          ).inverse([0.7, 0.3, -eps]))


@test_util.test_all_tf_execution_regimes
class DirichletFromVariableTest(test_util.TestCase):

  @test_util.tf_tape_safety_test
  def testGradients(self):
    x = tf.Variable([1., 1.1, 1.2])
    d = tfd.Dirichlet(concentration=x, validate_args=True)
    with tf.GradientTape() as tape:
      loss = -d.log_prob([0.1, 0.2, 0.7])
    g = tape.gradient(loss, d.trainable_variables)
    self.assertLen(g, 1)
    self.assertAllNotNone(g)

  def testAssertions(self):
    x = tfp.util.TransformedVariable(0.3679, tfb.Exp(), shape=None)
    with self.assertRaisesRegexp(
        ValueError, 'Argument `concentration` must have rank at least 1.'):
      d = tfd.Dirichlet(concentration=x, validate_args=True)
      self.evaluate([v.initializer for v in d.variables])
      self.evaluate(d.entropy())


if __name__ == '__main__':
  tf.test.main()
