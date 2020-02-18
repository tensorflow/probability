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
"""Tests for half-Student's t distribution."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

# Dependency imports
import numpy as np
from scipy import stats as sp_stats
from scipy.special import gamma

import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp

from tensorflow_probability.python.distributions.internal import statistical_testing as st
from tensorflow_probability.python.internal import assert_util
from tensorflow_probability.python.internal import test_util


tfd = tfp.distributions


def _true_mean(df, loc, scale):
  """Calculate the true mean in numpy for testing.

  Reference implementation, using
  https://en.wikipedia.org/wiki/Folded-t_and_half-t_distributions

  Not careful about numerical accuracy. Don't use for large df.

  Args:
    df: Positive float.
    loc: float.
    scale: float.

  Returns:
    The mean for a half normal.
  """
  df = np.array(df)
  loc = np.array(loc)
  scale = np.array(scale)
  return loc + 2 * scale * np.sqrt(df / np.pi) * gamma(0.5 * (df + 1)) / (
      gamma(0.5 * df) * (df - 1))


def _true_variance(df, scale):
  """Calculate the true variance in numpy for testing.

  Reference implementation, using
  https://en.wikipedia.org/wiki/Folded-t_and_half-t_distributions

  Not careful about numerical accuracy. Don't use for large df.

  Args:
    df: Positive float.
    scale: float.

  Returns:
    The variance for a half normal.
  """
  df = np.array(df)
  scale = np.array(scale)
  return scale**2 * (
      df / (df - 2.) - (4 * df) / (np.pi * (df - 1.) ** 2) *
      (gamma(0.5 * (df + 1)) / gamma(0.5 * df))**2)


@test_util.test_all_tf_execution_regimes
class HalfStudentTTest(test_util.TestCase):

  def testPDFAndLogPDF(self):
    batch_size = 6
    df_v = 3.
    loc_v = -7.
    sigma_v = 8.
    df = tf.constant([df_v] * batch_size)
    loc = tf.constant([loc_v] * batch_size)
    sigma = tf.constant([sigma_v] * batch_size)
    t = np.array([-2.5, 2.5, 8., 0., -1., 2.], dtype=np.float32)
    half_student = tfd.HalfStudentT(
        df, loc=loc, scale=sigma, validate_args=True)

    log_pdf = half_student.log_prob(t)
    self.assertEqual(log_pdf.shape, (batch_size,))
    log_pdf_values = self.evaluate(log_pdf)
    pdf = half_student.prob(t)
    self.assertEqual(pdf.shape, (batch_size,))
    pdf_values = self.evaluate(pdf)

    expected_log_pdf = (
        np.log(2.) + sp_stats.t.logpdf(t, df_v, loc=loc_v, scale=sigma_v))
    expected_pdf = 2. * sp_stats.t.pdf(t, df_v, loc=loc_v, scale=sigma_v)
    self.assertAllClose(expected_log_pdf, log_pdf_values)
    self.assertAllClose(np.log(expected_pdf), log_pdf_values)
    self.assertAllClose(expected_pdf, pdf_values)
    self.assertAllClose(np.exp(expected_log_pdf), pdf_values)

  def testLogPDFMultidimensional(self):
    batch_size = 6
    df_v = np.array([1.5, 7.2])
    loc_v = np.array([-4., -3.])
    sigma_v = np.array([np.sqrt(10.), np.sqrt(15.)])
    df = tf.constant([df_v.tolist()] * batch_size)
    loc = tf.constant([loc_v.tolist()] * batch_size)
    sigma = tf.constant([sigma_v.tolist()] * batch_size)
    t = np.array([[-2.5, 2.5, 4., 0., -1., 2.]], dtype=np.float32).T
    half_student = tfd.HalfStudentT(
        df, loc=loc, scale=sigma, validate_args=True)

    log_pdf = half_student.log_prob(t)
    log_pdf_values = self.evaluate(log_pdf)
    self.assertEqual(log_pdf.shape, (6, 2))
    pdf = half_student.prob(t)
    pdf_values = self.evaluate(pdf)
    self.assertEqual(pdf.shape, (6, 2))

    expected_log_pdf = (
        np.log(2.) + sp_stats.t.logpdf(t, df_v, loc=loc_v, scale=sigma_v))
    expected_pdf = (
        2. * sp_stats.t.pdf(t, df_v, loc=loc_v, scale=sigma_v))
    self.assertAllClose(expected_log_pdf, log_pdf_values)
    self.assertAllClose(np.log(expected_pdf), log_pdf_values)
    self.assertAllClose(expected_pdf, pdf_values)
    self.assertAllClose(np.exp(expected_log_pdf), pdf_values)

  def testCDFAndLogCDF(self):
    batch_size = 6
    df_v = 3.
    loc_v = -7.
    sigma_v = 8.
    df = tf.constant([df_v] * batch_size)
    loc = tf.constant([loc_v] * batch_size)
    sigma = tf.constant([sigma_v] * batch_size)
    t = np.array([-2.5, 2.5, 8., 0., -1., 2.], dtype=np.float32)
    half_student = tfd.HalfStudentT(
        df, loc=loc, scale=sigma, validate_args=True)

    log_cdf = half_student.log_cdf(t)
    self.assertEqual(log_cdf.shape, (6,))
    log_cdf_values = self.evaluate(log_cdf)
    cdf = half_student.cdf(t)
    self.assertEqual(cdf.shape, (6,))
    cdf_values = self.evaluate(cdf)

    # no reference implementation in numpy/scipy, so use
    # CDF(|X|) = 2 CDF(X) - 1, and just take the log of that
    # for the log_cdf
    expected_cdf = 2. * sp_stats.t.cdf(t, df_v, loc=loc_v, scale=sigma_v) - 1.
    expected_log_cdf = np.log(expected_cdf)
    self.assertAllClose(expected_log_cdf, log_cdf_values, atol=0., rtol=1e-5)
    self.assertAllClose(
        np.log(expected_cdf), log_cdf_values, atol=0., rtol=1e-5)
    self.assertAllClose(expected_cdf, cdf_values, atol=0., rtol=1e-5)
    self.assertAllClose(
        np.exp(expected_log_cdf), cdf_values, atol=0., rtol=1e-5)

  def testEntropy(self):
    df_v = np.array([[2., 3., 7.]])  # 1x3
    loc_v = np.array([[1., -1, 0]])  # 1x3
    sigma_v = np.array([[1., 2., 3.]]).T  # transposed => 3x1
    half_student = tfd.HalfStudentT(
        df=df_v, loc=loc_v, scale=sigma_v, validate_args=True)
    ent = half_student.entropy()
    ent_values = self.evaluate(ent)

    # Help scipy broadcast to 3x3
    ones = np.array([[1, 1, 1]])
    sigma_bc = np.abs(sigma_v) * ones
    loc_bc = ones.T * loc_v
    df_bc = ones.T * df_v
    expected_entropy = sp_stats.t.entropy(
        np.reshape(df_bc, [-1]),
        loc=np.reshape(loc_bc, [-1]),
        scale=np.reshape(sigma_bc, [-1])) - np.log(2)
    expected_entropy = np.reshape(expected_entropy, df_bc.shape)
    self.assertAllClose(expected_entropy, ent_values)

  def testSample(self):
    df_v = 4.
    loc_v = 3.
    scale_v = math.sqrt(10.)
    df = tf.constant(df_v)
    loc = tf.constant(loc_v)
    scale = tf.constant(scale_v)
    n = tf.constant(200000)
    half_student = tfd.HalfStudentT(
        df=df, loc=loc, scale=scale, validate_args=True)
    samples = half_student.sample(n, seed=test_util.test_seed())
    sample_values = self.evaluate(samples)
    expected_mean = self.evaluate(half_student.mean())
    expected_var = self.evaluate(half_student.variance())
    n_val = 200000
    self.assertEqual(sample_values.shape, (n_val,))
    self.assertAllClose(sample_values.mean(), expected_mean, rtol=0.1, atol=0)
    self.assertAllClose(
        sample_values.var(), expected_var, rtol=0.1, atol=0)
    self._checkKLApprox(df_v, loc_v, scale_v, sample_values)

  # Test that sampling with the same seed twice gives the same results.
  def testSampleMultipleTimes(self):
    df = tf.constant(4.)
    loc = tf.constant(3.)
    sigma = tf.constant(math.sqrt(10.))
    n = tf.constant(100)
    seed = test_util.test_seed()

    tf.random.set_seed(seed)
    half_student = tfd.HalfStudentT(
        df=df, loc=loc, scale=sigma, name='half_student_t1', validate_args=True)
    samples1 = self.evaluate(half_student.sample(n, seed=seed))

    tf.random.set_seed(seed)
    half_student2 = tfd.HalfStudentT(
        df=df, loc=loc, scale=sigma, name='half_student_t2', validate_args=True)
    samples2 = self.evaluate(half_student2.sample(n, seed=seed))

    self.assertAllClose(samples1, samples2)

  def testSampleSmallDfNoNan(self):
    df_v = [1e-1, 1e-5, 1e-10, 1e-20]
    df = tf.constant(df_v)
    n = tf.constant(200000)
    half_student = tfd.HalfStudentT(df=df, loc=1., scale=1., validate_args=True)
    samples = half_student.sample(n, seed=test_util.test_seed())
    sample_values = self.evaluate(samples)
    n_val = 200000
    self.assertEqual(sample_values.shape, (n_val, 4))
    self.assertTrue(np.all(np.logical_not(np.isnan(sample_values))))

  def testSampleMultiDimensional(self):
    batch_size = 7
    df_v = [5., 7.]
    loc_v = [3., -3.]
    sigma_v = [math.sqrt(10.), math.sqrt(15.)]
    df = tf.constant([df_v] * batch_size)
    loc = tf.constant([loc_v] * batch_size)
    sigma = tf.constant([sigma_v] * batch_size)
    n = tf.constant(200000)
    half_student = tfd.HalfStudentT(
        df=df, loc=loc, scale=sigma, validate_args=True)
    samples = half_student.sample(n, seed=test_util.test_seed())
    sample_values = self.evaluate(samples)
    expected_mean = self.evaluate(half_student.mean()).mean(axis=0)
    expected_var = self.evaluate(half_student.variance()).mean(axis=0)
    self.assertEqual(samples.shape, (200000, batch_size, 2))
    self.assertAllClose(
        sample_values[:, 0, 0].mean(), expected_mean[0], rtol=0.1, atol=0)
    self.assertAllClose(
        sample_values[:, 0, 0].var(),
        expected_var[0],
        rtol=0.2,
        atol=0)
    self._checkKLApprox(
        df_v[0], loc_v[0], sigma_v[0], sample_values[:, 0, 0])
    self.assertAllClose(
        sample_values[:, 0, 1].mean(), expected_mean[1], rtol=0.1, atol=0)
    self.assertAllClose(
        sample_values[:, 0, 1].var(),
        expected_var[1],
        rtol=0.2,
        atol=0)
    self._checkKLApprox(
        df_v[1], loc_v[1], sigma_v[1], sample_values[:, 0, 1])

  def _checkKLApprox(self, df, loc, sigma, samples):
    n = samples.size
    np.random.seed(137)
    sample_scipy = np.abs(sp_stats.t.rvs(df, loc=0., scale=sigma, size=n)) + loc
    covg = 0.99
    _, right = sp_stats.t.interval(0.5 * (1 + covg), df, loc=loc, scale=sigma)
    r = (loc, right)
    bins = 100
    hist, _ = np.histogram(samples, bins=bins, range=r)
    hist_scipy, _ = np.histogram(sample_scipy, bins=bins, range=r)
    self.assertGreater(hist.sum(), n * (covg - .01))
    self.assertGreater(hist_scipy.sum(), n * (covg - .01))
    hist_min1 = hist + 1.  # put at least one item in each bucket
    hist_norm = hist_min1 / hist_min1.sum()
    hist_scipy_min1 = hist_scipy + 1.  # put at least one item in each bucket
    hist_scipy_norm = hist_scipy_min1 / hist_scipy_min1.sum()
    kl_appx = np.sum(np.log(hist_scipy_norm / hist_norm) * hist_scipy_norm)
    self.assertLess(kl_appx, 1)

  def testBroadcastingParams(self):

    def _check(half_student):
      self.assertEqual(half_student.mean().shape, (3,))
      self.assertEqual(half_student.variance().shape, (3,))
      self.assertEqual(half_student.entropy().shape, (3,))
      self.assertEqual(half_student.log_prob(6.).shape, (3,))
      self.assertEqual(half_student.prob(6.).shape, (3,))
      self.assertEqual(half_student.sample(
          37, seed=test_util.test_seed()).shape, (37, 3,))

    _check(
        tfd.HalfStudentT(df=[
            2.,
            3.,
            4.,
        ], loc=2., scale=1., validate_args=True))
    _check(
        tfd.HalfStudentT(df=7., loc=[
            2.,
            3.,
            4.,
        ], scale=1., validate_args=True))
    _check(
        tfd.HalfStudentT(df=7., loc=3., scale=[
            2.,
            3.,
            4.,
        ], validate_args=True))

  def testBroadcastingPdfArgs(self):

    def _assert_shape(half_student, arg, shape):
      self.assertEqual(half_student.log_prob(arg).shape, shape)
      self.assertEqual(half_student.prob(arg).shape, shape)

    def _check(half_student):
      _assert_shape(half_student, 5., (3,))
      xs = np.array([5., 6., 7.], dtype=np.float32)
      _assert_shape(half_student, xs, (3,))
      xs = np.array([xs])
      _assert_shape(half_student, xs, (1, 3))
      xs = xs.T
      _assert_shape(half_student, xs, (3, 3))

    _check(
        tfd.HalfStudentT(df=[
            2.,
            3.,
            4.,
        ], loc=2., scale=1., validate_args=True))
    _check(
        tfd.HalfStudentT(df=7., loc=[
            2.,
            3.,
            4.,
        ], scale=1., validate_args=True))
    _check(
        tfd.HalfStudentT(df=7., loc=3., scale=[
            2.,
            3.,
            4.,
        ], validate_args=True))

    def _check2d(half_student):
      _assert_shape(half_student, 5., (1, 3))
      xs = np.array([5., 6., 7.], dtype=np.float32)
      _assert_shape(half_student, xs, (1, 3))
      xs = np.array([xs])
      _assert_shape(half_student, xs, (1, 3))
      xs = xs.T
      _assert_shape(half_student, xs, (3, 3))

    _check2d(
        tfd.HalfStudentT(df=[[
            2.,
            3.,
            4.,
        ]], loc=2., scale=1., validate_args=True))
    _check2d(
        tfd.HalfStudentT(df=7., loc=[[
            2.,
            3.,
            4.,
        ]], scale=1., validate_args=True))
    _check2d(
        tfd.HalfStudentT(df=7., loc=3., scale=[[
            2.,
            3.,
            4.,
        ]], validate_args=True))

    def _check2d_rows(half_student):
      _assert_shape(half_student, 5., (3, 1))
      xs = np.array([5., 6., 7.], dtype=np.float32)  # (3,)
      _assert_shape(half_student, xs, (3, 3))
      xs = np.array([xs])  # (1,3)
      _assert_shape(half_student, xs, (3, 3))
      xs = xs.T  # (3,1)
      _assert_shape(half_student, xs, (3, 1))

    _check2d_rows(
        tfd.HalfStudentT(
            df=[[2.], [3.], [4.]], loc=2., scale=1., validate_args=True))
    _check2d_rows(
        tfd.HalfStudentT(
            df=7., loc=[[2.], [3.], [4.]], scale=1., validate_args=True))
    _check2d_rows(
        tfd.HalfStudentT(
            df=7., loc=3., scale=[[2.], [3.], [4.]], validate_args=True))

  def testMeanAllowNanStatsIsFalseWorksWhenAllBatchMembersAreDefined(self):
    loc = [1., 3.3, 4.4]
    half_student = tfd.HalfStudentT(
        df=[3., 5., 7.], loc=loc, scale=[3., 2., 1.], validate_args=True)
    mean = self.evaluate(half_student.mean())
    self.assertEqual((3,), mean.shape)

  def testMeanAllowNanStatsIsFalseRaisesWhenBatchMemberIsUndefined(self):
    loc = [1., 3.3, 4.4]
    half_student = tfd.HalfStudentT(
        df=[0.5, 5., 7.],
        loc=loc,
        scale=[3., 2., 1.],
        allow_nan_stats=False,
        validate_args=True)
    with self.assertRaisesOpError('x < y'):
      self.evaluate(half_student.mean())

  def testMeanAllowNanStatsIsTrueReturnsNaNForUndefinedBatchMembers(self):
    loc = [-2, 0., 1., 3.3, 4.4]
    sigma = [5., 4., 3., 2., 1.]
    half_student = tfd.HalfStudentT(
        df=[0.5, 1., 3., 5., 7.],
        loc=loc,
        scale=sigma,
        allow_nan_stats=True,
        validate_args=True)
    mean = self.evaluate(half_student.mean())

    # confirm two NaNs, where they are expected
    self.assertEqual(np.isnan(mean).sum(), 2)
    self.assertAllNan(mean[:2])

  def testVarianceAllowNanStatsTrueReturnsNaNforUndefinedBatchMembers(self):
    # df = 0.5 ==> undefined mean ==> undefined variance.
    # df = 1.5 ==> infinite variance.
    df = [0.5, 1.5, 3., 5., 7.]
    loc = [-2, 0., 1., 3.3, 4.4]
    sigma = [5., 4., 3., 2., 1.]
    half_student = tfd.HalfStudentT(
        df=df, loc=loc, scale=sigma, allow_nan_stats=True, validate_args=True)
    var = self.evaluate(half_student.variance())
    # Verify edge cases work as intended.
    expected_var = _true_variance(df, sigma)
    expected_var[0] = np.nan
    expected_var[1] = np.inf
    self.assertAllClose(expected_var, var)

  def testVarianceAllowNanStatsFalseGivesCorrectValueForDefinedBatchMembers(
      self):
    # df = 1.5 ==> infinite variance.
    df = [1.5, 3., 5., 7.]
    loc = [0., 1., 3.3, 4.4]
    sigma = [4., 3., 2., 1.]
    half_student = tfd.HalfStudentT(
        df=df, loc=loc, scale=sigma, validate_args=True)
    var = self.evaluate(half_student.variance())

    expected_var = _true_variance(df, sigma)
    expected_var[0] = np.inf
    self.assertAllClose(expected_var, var)

  def testVarianceAllowNanStatsFalseRaisesForUndefinedBatchMembers(self):
    # df <= 1 ==> variance not defined
    half_student = tfd.HalfStudentT(
        df=1., loc=0., scale=1., allow_nan_stats=False, validate_args=True)
    with self.assertRaisesOpError('x < y'):
      self.evaluate(half_student.variance())

    # df <= 1 ==> variance not defined
    half_student = tfd.HalfStudentT(
        df=0.5, loc=0., scale=1., allow_nan_stats=False, validate_args=True)
    with self.assertRaisesOpError('x < y'):
      self.evaluate(half_student.variance())

  def testStd(self):
    # Defined for all batch members.
    df = [3.5, 5., 3., 5., 7.]
    loc = [-2.2]
    sigma = [5., 4., 3., 2., 1.]
    half_student = tfd.HalfStudentT(
        df=df, loc=loc, scale=sigma, validate_args=True)
    # Test broadcast of loc across shape of df/sigma
    stddev = self.evaluate(half_student.stddev())
    loc *= len(df)

    expected_var = _true_variance(df, sigma)
    self.assertAllClose(expected_var ** 0.5, stddev)

  def testPdfOfSample(self):
    half_student = tfd.HalfStudentT(
        df=3., loc=np.pi, scale=1., validate_args=True)
    num = 20000
    samples = half_student.sample(num, seed=test_util.test_seed())
    pdfs = half_student.prob(samples)
    mean = half_student.mean()
    mean_pdf = half_student.prob(half_student.mean())
    sample_vals, pdf_vals, mean_val, mean_pdf_val = self.evaluate(
        [samples, pdfs, half_student.mean(), mean_pdf])
    self.assertEqual(samples.shape, (num,))
    self.assertEqual(pdfs.shape, (num,))
    self.assertEqual(mean.shape, ())
    true_mean = _true_mean(3., np.pi, 1.)
    self.assertNear(true_mean, np.mean(sample_vals), err=0.1)
    self.assertNear(true_mean, mean_val, err=1e-6)
    # Verify integral over sample*pdf ~= 1.
    # Tolerance increased since eager was getting a value of 1.002041.
    self._assertIntegral(sample_vals, pdf_vals, err=5e-2)
    self.assertNear(
        2 * sp_stats.t.pdf(true_mean, 3., loc=np.pi), mean_pdf_val, err=1e-6)

  def testFullyReparameterized(self):
    df = tf.constant(2.0)
    loc = tf.constant(1.0)
    sigma = tf.constant(3.0)
    _, [grad_df, grad_loc, grad_sigma] = tfp.math.value_and_gradient(
        lambda d, m, s: tfd.HalfStudentT(  # pylint: disable=g-long-lambda
            df=d, loc=m, scale=s, validate_args=True).sample(
                100, seed=test_util.test_seed()), [df, loc, sigma])
    self.assertIsNotNone(grad_df)
    self.assertIsNotNone(grad_loc)
    self.assertIsNotNone(grad_sigma)

  def testPdfOfSampleMultiDims(self):
    half_student = tfd.HalfStudentT(
        df=[7., 11.], loc=[[5.], [6.]], scale=3., validate_args=True)
    self.assertAllEqual([], half_student.event_shape)
    self.assertAllEqual([], self.evaluate(half_student.event_shape_tensor()))
    self.assertAllEqual([2, 2], half_student.batch_shape)
    self.assertAllEqual([2, 2], self.evaluate(
        half_student.batch_shape_tensor()))
    num = 50000
    samples = half_student.sample(num, seed=test_util.test_seed())
    pdfs = half_student.prob(samples)
    sample_vals, pdf_vals = self.evaluate([samples, pdfs])
    self.assertEqual(samples.shape, (num, 2, 2))
    self.assertEqual(pdfs.shape, (num, 2, 2))
    self.assertNear(_true_mean(7, 5, 3), np.mean(sample_vals[:, 0, :]), err=0.1)
    self.assertNear(
        _true_mean(11, 6, 3), np.mean(sample_vals[:, 1, :]), err=0.1)
    self._assertIntegral(sample_vals[:, 0, 0], pdf_vals[:, 0, 0], err=0.05)
    self._assertIntegral(sample_vals[:, 0, 1], pdf_vals[:, 0, 1], err=0.05)
    self._assertIntegral(sample_vals[:, 1, 0], pdf_vals[:, 1, 0], err=0.05)
    self._assertIntegral(sample_vals[:, 1, 1], pdf_vals[:, 1, 1], err=0.05)
    self.assertNear(
        _true_variance(7, 3),
        np.var(sample_vals[:, :, 0]),
        err=1.0)
    self.assertNear(
        _true_variance(11, 3),
        np.var(sample_vals[:, :, 1]),
        err=1.0)

  def _assertIntegral(self, sample_vals, pdf_vals, err=1.5e-3):
    s_p = zip(sample_vals, pdf_vals)
    prev = (sample_vals.min(), 0)
    total = 0
    for k in sorted(s_p, key=lambda x: x[0]):
      pair_pdf = (k[1] + prev[1]) / 2
      total += (k[0] - prev[0]) * pair_pdf
      prev = k
    self.assertNear(1., total, err=err)

  def testNegativeDofFails(self):
    with self.assertRaisesOpError(r'`df` must be positive'):
      half_student = tfd.HalfStudentT(
          df=[2, -5.], loc=0., scale=1., validate_args=True, name='S')
      self.evaluate(half_student.mean())

  @test_util.tf_tape_safety_test
  def testGradientThroughParams(self):
    df = tf.Variable([[17.3], [14.]])
    loc = tf.Variable([[-5., 0., 0.5]])
    scale = tf.Variable(2.)
    d = tfd.HalfStudentT(df=df, loc=loc, scale=scale, validate_args=True)
    with tf.GradientTape() as tape:
      loss = -d.log_prob(np.ones((2, 3)))
    grad = tape.gradient(loss, d.trainable_variables)
    self.assertLen(grad, 3)
    self.assertAllNotNone(grad)

  def testAssertParamsAreFloats(self):
    df = tf.Variable(14, dtype=tf.int32)
    loc = tf.Variable(0, dtype=tf.int32)
    scale = tf.Variable(1, dtype=tf.int32)
    with self.assertRaisesRegexp(ValueError, 'Expected floating point'):
      tfd.HalfStudentT(df=df, loc=loc, scale=scale, validate_args=True)

  # Sample testing
  def testSampleEmpiricalCDF(self):
    num_samples = 300000
    dist = tfd.HalfStudentT(df=5., loc=10., scale=2., validate_args=True)
    samples = dist.sample(num_samples, seed=test_util.test_seed())

    check_cdf_agrees = st.assert_true_cdf_equal_by_dkwm(
        samples, dist.cdf, false_fail_rate=1e-6)
    check_enough_power = assert_util.assert_less(
        st.min_discrepancy_of_true_cdfs_detectable_by_dkwm(
            num_samples, false_fail_rate=1e-6, false_pass_rate=1e-6), 0.01)
    self.evaluate([check_cdf_agrees, check_enough_power])


if __name__ == '__main__':
  tf.test.main()
