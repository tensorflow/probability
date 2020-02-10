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
"""Tests for Generalized Pareto distribution."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
import hypothesis as hp
import hypothesis.strategies as hps
import numpy as np
from scipy import stats as sp_stats

import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp

from tensorflow_probability.python.internal import hypothesis_testlib as tfp_hps
from tensorflow_probability.python.internal import test_util

tfd = tfp.distributions


# Pylint doesn't understand hps.composite.
# pylint: disable=no-value-for-parameter


@hps.composite
def generalized_paretos(draw, batch_shape=None):
  if batch_shape is None:
    batch_shape = draw(tfp_hps.shapes())

  constraints = dict(
      loc=tfp_hps.identity_fn,
      scale=tfp_hps.softplus_plus_eps(),
      concentration=lambda x: tf.math.tanh(x) * 0.24)  # <.25==safe for variance

  params = draw(
      tfp_hps.broadcasting_params(
          batch_shape,
          params_event_ndims=dict(loc=0, scale=0, concentration=0),
          constraint_fn_for=constraints.get))
  dist = tfd.GeneralizedPareto(validate_args=draw(hps.booleans()), **params)
  if dist.batch_shape != batch_shape:
    raise AssertionError('batch_shape mismatch: expect {} but got {}'.format(
        batch_shape, dist))
  return dist


@test_util.test_all_tf_execution_regimes
class GeneralizedParetoTest(test_util.TestCase):

  @hp.given(generalized_paretos())
  @tfp_hps.tfp_hp_settings()
  def testShape(self, dist):
    # batch_shape == dist.batch_shape asserted in generalized_paretos()
    self.assertEqual(dist.batch_shape, self.evaluate(dist.batch_shape_tensor()))
    self.assertEqual(tf.TensorShape([]), dist.event_shape)
    self.assertAllEqual([], self.evaluate(dist.event_shape_tensor()))

  @hp.given(generalized_paretos(batch_shape=[]))
  @tfp_hps.tfp_hp_settings()
  def testLogPDF(self, dist):
    loc, scale, conc = self.evaluate([dist.loc, dist.scale, dist.concentration])
    hp.assume(abs(loc / scale) < 1e7)
    xs = self.evaluate(dist.sample(seed=test_util.test_seed()))

    logp = dist.log_prob(xs)
    self.assertEqual(dist.batch_shape, logp.shape)
    p = dist.prob(xs)
    self.assertEqual(dist.batch_shape, p.shape)

    expected_logp = sp_stats.genpareto(conc, loc=loc, scale=scale).logpdf(xs)
    actual_logp = self.evaluate(logp)
    self.assertAllClose(expected_logp, actual_logp, rtol=1e-5)
    self.assertAllClose(np.exp(expected_logp), self.evaluate(p), rtol=1e-5)

  def testLogPDFBoundary(self):
    # When loc = concentration = 0, we have an exponential distribution. Check
    # that at 0 we have finite log prob.
    scale = np.array([0.1, 0.5, 1., 2., 5., 10.], dtype=np.float32)
    dist = tfd.GeneralizedPareto(
        loc=0, scale=scale, concentration=0, validate_args=True)
    log_pdf = self.evaluate(dist.log_prob(0.))
    self.assertAllClose(-np.log(scale), log_pdf, rtol=1e-5)

    # Log prob should be finite on the boundary regardless of parameters.
    loc = np.array([1., 2., 5.]).astype(np.float32)
    scale = 2.
    concentration = np.array([-5., -3.4, -1.]).astype(np.float32)
    dist = tfd.GeneralizedPareto(
        loc=loc, scale=scale, concentration=concentration, validate_args=True)
    log_pdf_at_loc = self.evaluate(dist.log_prob(loc))
    self.assertAllFinite(log_pdf_at_loc)

    # TODO(b/144948687) Avoid `nan` at boundary. Ideally we'd do this test:
    # boundary = loc - scale / concentration
    # log_pdf_at_boundary = dist.log_prob(boundary)
    # self.assertAllFinite(log_pdf_at_boundary)

  def testAssertValidSample(self):
    loc = np.array([1., 2., 5.]).astype(np.float32)
    scale = 2.
    concentration = np.array([-5., -3.4, 1.]).astype(np.float32)
    dist = tfd.GeneralizedPareto(
        loc=loc, scale=scale, concentration=concentration, validate_args=True)

    with self.assertRaisesOpError('must be greater than or equal to `loc`'):
      self.evaluate(dist.prob([1.3, 1.3, 6.]))

    with self.assertRaisesOpError(
        'less than or equal to `loc - scale / concentration`'):
      self.evaluate(dist.cdf([1.5, 2.3, 6.]))

  def testSupportBijectorOutsideRange(self):
    loc = np.array([1., 2., 5.]).astype(np.float32)
    scale = 2.
    concentration = np.array([-5., -2., 1.]).astype(np.float32)
    dist = tfd.GeneralizedPareto(
        loc=loc, scale=scale, concentration=concentration, validate_args=True)

    x = np.array([1. - 1e-6, 3.1, 4.9]).astype(np.float32)
    bijector_inverse_x = dist._experimental_default_event_space_bijector(
        ).inverse(x)
    self.assertAllNan(self.evaluate(bijector_inverse_x))

  @hp.given(generalized_paretos(batch_shape=[]))
  @tfp_hps.tfp_hp_settings()
  def testCDF(self, dist):
    xs = self.evaluate(dist.sample(seed=test_util.test_seed()))
    cdf = dist.cdf(xs)
    self.assertEqual(dist.batch_shape, cdf.shape)

    loc, scale, conc = self.evaluate([dist.loc, dist.scale, dist.concentration])
    hp.assume(abs(loc / scale) < 1e7)
    expected_cdf = sp_stats.genpareto(conc, loc=loc, scale=scale).cdf(xs)
    actual_cdf = self.evaluate(cdf)
    msg = ('Location: {}, scale: {}, concentration: {}, xs: {} '
           'scipy cdf: {}, tfp cdf: {}')
    hp.note(msg.format(loc, scale, conc, xs, expected_cdf, actual_cdf))
    self.assertAllClose(expected_cdf, actual_cdf, rtol=5e-5)

  @hp.given(generalized_paretos(batch_shape=[]))
  @tfp_hps.tfp_hp_settings()
  def testMean(self, dist):
    loc, scale, conc = self.evaluate([dist.loc, dist.scale, dist.concentration])
    hp.note('Location: {}, scale: {}, concentration: {}'.format(
        loc, scale, conc))
    self.assertEqual(dist.batch_shape, dist.mean().shape)
    # scipy doesn't seem to be very accurate for small concentrations, so use
    # higher precision.
    expected = sp_stats.genpareto(np.float64(conc), loc=np.float64(loc),
                                  scale=np.float64(scale)).mean()
    actual = self.evaluate(dist.mean())
    # There is an unavoidable catastropic cancellation for means near 0
    self.assertAllClose(expected, actual, rtol=5e-4, atol=1e-4)

  @hp.given(generalized_paretos(batch_shape=[]))
  @tfp_hps.tfp_hp_settings()
  def testVariance(self, dist):
    loc, scale, conc = self.evaluate([dist.loc, dist.scale, dist.concentration])
    # scipy doesn't seem to be very accurate for small concentrations, so use
    # higher precision.
    expected = sp_stats.genpareto(np.float64(conc), loc=np.float64(loc),
                                  scale=np.float64(scale)).var()
    # scipy sometimes returns nonsense zero or negative variances.
    hp.assume(expected > 0)
    # scipy gets bad answers for very small concentrations even in 64-bit.
    # https://github.com/scipy/scipy/issues/11168
    hp.assume(conc > 1e-4)
    self.assertEqual(dist.batch_shape, dist.variance().shape)
    actual = self.evaluate(dist.variance())
    msg = ('Location: {}, scale: {}, concentration: {}, '
           'scipy variance: {}, tfp variance: {}')
    hp.note(msg.format(loc, scale, conc, expected, actual))
    self.assertAllClose(expected, actual)

  @hp.given(generalized_paretos(batch_shape=[]))
  @tfp_hps.tfp_hp_settings()
  def testEntropy(self, dist):
    loc, scale, conc = self.evaluate([dist.loc, dist.scale, dist.concentration])
    self.assertEqual(dist.batch_shape, dist.entropy().shape)
    expected = sp_stats.genpareto.entropy(conc, loc=loc, scale=scale)
    actual = self.evaluate(dist.entropy())
    self.assertAllClose(expected, actual)

  def testSample(self):
    loc = np.float32(-7.5)
    scale = np.float32(3.5)
    conc = np.float32(0.07)
    n = 10**5
    dist = tfd.GeneralizedPareto(
        loc=loc, scale=scale, concentration=conc, validate_args=True)
    samples = dist.sample(n, seed=test_util.test_seed())
    sample_values = self.evaluate(samples)
    self.assertEqual((n,), samples.shape)
    self.assertEqual((n,), sample_values.shape)
    self.assertTrue(self._kstest(loc, scale, conc, sample_values))
    self.assertAllClose(
        sp_stats.genpareto.mean(conc, loc=loc, scale=scale),
        sample_values.mean(),
        rtol=.02)
    self.assertAllClose(
        sp_stats.genpareto.var(conc, loc=loc, scale=scale),
        sample_values.var(),
        rtol=.08)

  def testFullyReparameterized(self):
    loc = tf.constant(4.0)
    scale = tf.constant(3.0)
    conc = tf.constant(2.0)
    _, grads = tfp.math.value_and_gradient(
        lambda *args: tfd.GeneralizedPareto(*args, validate_args=True).sample(  # pylint: disable=g-long-lambda
            100, seed=test_util.test_seed()), [loc, scale, conc])
    self.assertLen(grads, 3)
    self.assertAllNotNone(grads)

  def testSampleKolmogorovSmirnovMultiDimensional(self):
    loc = np.linspace(-10, 10, 3).reshape(3, 1, 1)
    scale = np.linspace(1e-6, 7, 5).reshape(5, 1)
    conc = np.linspace(-1.3, 1.3, 7)

    dist = tfd.GeneralizedPareto(
        loc=loc, scale=scale, concentration=conc, validate_args=True)
    n = 10**4
    samples = dist.sample(n, seed=test_util.test_seed())
    sample_values = self.evaluate(samples)
    self.assertEqual((n, 3, 5, 7), samples.shape)
    self.assertEqual((n, 3, 5, 7), sample_values.shape)

    fails = 0
    trials = 0
    for li, l in enumerate(loc.reshape(-1)):
      for si, s in enumerate(scale.reshape(-1)):
        for ci, c in enumerate(conc.reshape(-1)):
          samps = sample_values[:, li, si, ci]
          trials += 1
          fails += 0 if self._kstest(l, s, c, samps) else 1
    self.assertLess(fails, trials * 0.03)

  def _kstest(self, loc, scale, conc, samples):
    # Uses the Kolmogorov-Smirnov test for goodness of fit.
    ks, _ = sp_stats.kstest(samples,
                            sp_stats.genpareto(conc, loc=loc, scale=scale).cdf)
    # Return True when the test passes.
    return ks < 0.02

  def testPdfOfSampleMultiDims(self):
    dist = tfd.GeneralizedPareto(
        loc=0,
        scale=[[2.], [3.]],
        concentration=[-.37, .11],
        validate_args=True)
    num = 50000
    samples = dist.sample(num, seed=test_util.test_seed())
    pdfs = dist.prob(samples)
    sample_vals, pdf_vals = self.evaluate([samples, pdfs])
    self.assertEqual((num, 2, 2), samples.shape)
    self.assertEqual((num, 2, 2), pdfs.shape)
    self._assertIntegral(sample_vals[:, 0, 0], pdf_vals[:, 0, 0], err=0.02)
    self._assertIntegral(sample_vals[:, 0, 1], pdf_vals[:, 0, 1], err=0.02)
    self._assertIntegral(sample_vals[:, 1, 0], pdf_vals[:, 1, 0], err=0.02)
    self._assertIntegral(sample_vals[:, 1, 1], pdf_vals[:, 1, 1], err=0.02)

  def _assertIntegral(self, sample_vals, pdf_vals, err=1e-3):
    s_p = zip(sample_vals, pdf_vals)
    prev = (0, 0)
    total = 0
    for k in sorted(s_p, key=lambda x: x[0]):
      pair_pdf = (k[1] + prev[1]) / 2
      total += (k[0] - prev[0]) * pair_pdf
      prev = k
    self.assertNear(1., total, err=err)

  def testNonPositiveInitializationParamsRaises(self):
    scale = tf.constant(0.0, name='scale')
    with self.assertRaisesOpError('Argument `scale` must be positive.'):
      dist = tfd.GeneralizedPareto(
          loc=0, scale=scale, concentration=1, validate_args=True)
      self.evaluate(dist.mean())

  @test_util.tf_tape_safety_test
  def testGradientThroughConcentration(self):
    concentration = tf.Variable(3.)
    d = tfd.GeneralizedPareto(
        loc=0, scale=1, concentration=concentration, validate_args=True)
    with tf.GradientTape() as tape:
      loss = -d.log_prob([1., 2., 4.])
    grad = tape.gradient(loss, d.trainable_variables)
    self.assertLen(grad, 1)
    self.assertAllNotNone(grad)

  def testAssertsPositiveScale(self):
    scale = tf.Variable([1., 2., -3.])
    self.evaluate(scale.initializer)
    with self.assertRaisesOpError('Argument `scale` must be positive.'):
      d = tfd.GeneralizedPareto(
          loc=0, scale=scale, concentration=1, validate_args=True)
      self.evaluate(d.sample(seed=test_util.test_seed()))

  def testAssertsPositiveScaleAfterMutation(self):
    scale = tf.Variable([1., 2., 3.])
    self.evaluate(scale.initializer)
    d = tfd.GeneralizedPareto(
        loc=0, scale=scale, concentration=0.25, validate_args=True)
    self.evaluate(d.mean())
    with self.assertRaisesOpError('Argument `scale` must be positive.'):
      with tf.control_dependencies([scale.assign([1., 2., -3.])]):
        self.evaluate(d.sample(seed=test_util.test_seed()))

  @test_util.tf_tape_safety_test
  def testGradientThroughLocScale(self):
    loc = tf.Variable(1.)
    scale = tf.Variable(2.5)
    d = tfd.GeneralizedPareto(
        loc=loc, scale=scale, concentration=.15, validate_args=True)
    with tf.GradientTape() as tape:
      loss = -d.log_prob([1., 2., 4.])
    grads = tape.gradient(loss, d.trainable_variables)
    self.assertLen(grads, 2)
    self.assertAllNotNone(grads)


if __name__ == '__main__':
  tf.test.main()
