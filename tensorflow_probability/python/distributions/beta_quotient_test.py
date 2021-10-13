# Copyright 2020 The TensorFlow Probability Authors.
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

# Dependency imports
import numpy as np
from scipy import special as sp_special
from scipy import stats as sp_stats

import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp

from tensorflow_probability.python.distributions.internal import statistical_testing as st
from tensorflow_probability.python.internal import test_util

tfd = tfp.distributions


class _BetaQuotientTest(object):

  # Since the BetaQuotient distribution is the ratio distribution of two Betas,
  # we should be able to approximate the density through quadrature.
  def _compute_logpdf_quadrature(self, alpha0, beta0, alpha1, beta1, z):
    def _log_integrand(y):
      # Pad the axes to allow for vectorized computation
      return (
          np.log(y) + sp_stats.beta.logpdf(
              y * z[..., np.newaxis],
              alpha0[..., np.newaxis],
              beta0[..., np.newaxis]) +
          sp_stats.beta.logpdf(
              y,
              alpha1[..., np.newaxis],
              beta1[..., np.newaxis]))
    roots, weights = sp_special.roots_legendre(8000)
    # We need to account for the change of interval from [-1, 1] to [0, 1]
    shifted_roots = 0.5 * roots + 0.5
    return -np.log(2.) + sp_special.logsumexp(
        _log_integrand(shifted_roots) + np.log(weights), axis=-1)

  def testBetaQuotientShape(self):
    a = tf.ones([5], dtype=self.dtype)
    b = tf.ones([5], dtype=self.dtype)
    c = tf.ones([5], dtype=self.dtype)
    d = tf.ones([5], dtype=self.dtype)
    beta_quotient = tfd.BetaQuotient(a, b, c, d, validate_args=True)

    self.assertEqual(self.evaluate(beta_quotient.batch_shape_tensor()), (5,))
    self.assertEqual(beta_quotient.batch_shape, tf.TensorShape([5]))
    self.assertAllEqual(self.evaluate(beta_quotient.event_shape_tensor()),
                        [])
    self.assertEqual(beta_quotient.event_shape, tf.TensorShape([]))

  def testBetaQuotientShapeBroadcast(self):
    a = tf.ones([3, 1, 1, 1], dtype=self.dtype)
    b = tf.ones([1, 2, 1, 1], dtype=self.dtype)
    c = tf.ones([1, 1, 5, 1], dtype=self.dtype)
    d = tf.ones([1, 1, 1, 7], dtype=self.dtype)
    beta_quotient = tfd.BetaQuotient(a, b, c, d, validate_args=True)

    self.assertAllEqual(
        self.evaluate(beta_quotient.batch_shape_tensor()), (3, 2, 5, 7))
    self.assertEqual(beta_quotient.batch_shape, tf.TensorShape([3, 2, 5, 7]))
    self.assertAllEqual(
        self.evaluate(beta_quotient.event_shape_tensor()), [])
    self.assertEqual(beta_quotient.event_shape, tf.TensorShape([]))

  def testInvalidConcentration(self):
    with self.assertRaisesOpError('`concentration` must be positive'):
      beta_quotient = tfd.BetaQuotient(-1., 1., 1., 1., validate_args=True)
      self.evaluate(beta_quotient.sample())

    with self.assertRaisesOpError('`concentration` must be positive'):
      beta_quotient = tfd.BetaQuotient(1., -1., 1., 1., validate_args=True)
      self.evaluate(beta_quotient.sample())

    with self.assertRaisesOpError('`concentration` must be positive'):
      beta_quotient = tfd.BetaQuotient(1., 1., -1., 1., validate_args=True)
      self.evaluate(beta_quotient.sample())

    with self.assertRaisesOpError('`concentration` must be positive'):
      beta_quotient = tfd.BetaQuotient(1., 1., 1., -1., validate_args=True)
      self.evaluate(beta_quotient.sample())

  def testLogPdf(self):
    # Keep the `concentration`'s above 1 since quadrature has problems
    # otherwise.
    a = np.array([3., 2., 8.], dtype=self.dtype)[..., np.newaxis]
    b = np.array([1.8, 2.4, 3.2], dtype=self.dtype)[..., np.newaxis]
    c = np.array([5.5, 2., 4.3], dtype=self.dtype)[..., np.newaxis]
    d = np.array([1.6, 2.9, 6.4], dtype=self.dtype)[..., np.newaxis]
    beta_quotient = tfd.BetaQuotient(a, b, c, d, validate_args=True)
    x = np.linspace(0.1, 10., 50).astype(self.dtype)

    self.assertAllClose(
        self._compute_logpdf_quadrature(a, b, c, d, x),
        self.evaluate(beta_quotient.log_prob(x)), rtol=1e-4)

  def testLogPdfBroadcast(self):
    # Keep the `concentration`'s above 1 since quadrature has problems
    # otherwise.
    a = tf.random.uniform(
        shape=[2, 1, 1, 1],
        minval=1., maxval=5., seed=test_util.test_seed(), dtype=self.dtype)
    b = tf.random.uniform(
        shape=[1, 3, 1, 1],
        minval=1., maxval=5., seed=test_util.test_seed(), dtype=self.dtype)
    c = tf.random.uniform(
        shape=[1, 1, 5, 1],
        minval=1., maxval=5., seed=test_util.test_seed(), dtype=self.dtype)
    d = tf.random.uniform(
        shape=[1, 1, 1, 7],
        minval=1., maxval=5., seed=test_util.test_seed(), dtype=self.dtype)
    beta_quotient = tfd.BetaQuotient(a, b, c, d, validate_args=True)
    x = np.linspace(0.1, 5., 7).astype(self.dtype)
    log_prob, a, b, c, d = self.evaluate(
        [beta_quotient.log_prob(x), a, b, c, d])
    self.assertAllClose(
        self._compute_logpdf_quadrature(a, b, c, d, x),
        log_prob, rtol=4e-4)

  def testBetaQuotientSample(self):
    a = tf.random.uniform(
        shape=[2, 1, 1, 1],
        minval=1., maxval=5., seed=test_util.test_seed(), dtype=self.dtype)
    b = tf.random.uniform(
        shape=[1, 3, 1, 1],
        minval=1., maxval=5., seed=test_util.test_seed(), dtype=self.dtype)
    c = tf.random.uniform(
        shape=[1, 1, 5, 1],
        minval=1., maxval=5., seed=test_util.test_seed(), dtype=self.dtype)
    d = tf.random.uniform(
        shape=[1, 1, 1, 7],
        minval=1., maxval=5., seed=test_util.test_seed(), dtype=self.dtype)
    beta_quotient = tfd.BetaQuotient(a, b, c, d, validate_args=True)
    # TODO(b/179283344): Increase this to 3e5 when CPU-only gamma sampler is
    # fixed.
    n = int(3e4)
    samples = beta_quotient.sample(n, seed=test_util.test_seed())
    sample_values = self.evaluate(samples)
    self.assertEqual(sample_values.shape, (n, 2, 3, 5, 7))
    self.assertFalse(np.any(sample_values < 0.0))
    self.evaluate(
        st.assert_true_mean_equal_by_dkwm(
            samples,
            low=self.dtype(0.),
            high=self.dtype(np.inf),
            expected=beta_quotient.mean(),
            false_fail_rate=self.dtype(1e-6)))

  @test_util.numpy_disable_gradient_test
  def testBetaQuotientFullyReparameterized(self):
    a = tf.constant(1.0)
    b = tf.constant(2.0)
    c = tf.constant(3.0)
    d = tf.constant(4.0)
    _, [grad_a, grad_b, grad_c, grad_d] = tfp.math.value_and_gradient(
        lambda a_, b_, c_, d_: tfd.BetaQuotient(  # pylint: disable=g-long-lambda
            a_, b_, c_, d_, validate_args=True).sample(
                10, seed=test_util.test_seed()), [a, b, c, d])
    self.assertIsNotNone(grad_a)
    self.assertIsNotNone(grad_b)
    self.assertIsNotNone(grad_c)
    self.assertIsNotNone(grad_d)
    self.assertNotAllZero(grad_a)
    self.assertNotAllZero(grad_b)
    self.assertNotAllZero(grad_c)
    self.assertNotAllZero(grad_d)

  def testBetaQuotientMeanNoNanStats(self):
    # Mean will not be defined for the first entry.
    a = np.array([2.0, 3.0, 2.5])
    b = np.array([2.0, 4.0, 5.0])
    c = np.array([1.0, 3.0, 2.5])
    d = np.array([3.0, 4.0, 5.0])
    beta_quotient = tfd.BetaQuotient(
        a, b, c, d, allow_nan_stats=False, validate_args=True)
    with self.assertRaisesOpError('mean undefined'):
      self.evaluate(beta_quotient.mean())

  def testBetaQuotientMeanAllowNanStats(self):
    # Mean will not be defined for the first entry.
    a = np.array([2.0, 3.0, 2.5])
    b = np.array([2.0, 4.0, 5.0])
    c = np.array([1.0, 3.0, 2.5])
    d = np.array([3.0, 4.0, 5.0])
    beta_quotient = tfd.BetaQuotient(
        a, b, c, d, allow_nan_stats=True, validate_args=True)
    self.assertEqual(beta_quotient.mean().shape, (3,))
    self.assertAllNan(self.evaluate(beta_quotient.mean())[0])


@test_util.test_all_tf_execution_regimes
class BetaQuotientTestFloat32(test_util.TestCase, _BetaQuotientTest):
  dtype = np.float32


@test_util.test_all_tf_execution_regimes
class BetaQuotientTestFloat64(test_util.TestCase, _BetaQuotientTest):
  dtype = np.float64


if __name__ == '__main__':
  test_util.main()
