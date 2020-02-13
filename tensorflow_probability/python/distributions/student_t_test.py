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
"""Tests for Student t distribution."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

# Dependency imports
import numpy as np
from scipy import stats as sp_stats

import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp

from tensorflow_probability.python.internal import test_util

tfd = tfp.distributions


@test_util.test_all_tf_execution_regimes
class StudentTTest(test_util.TestCase):

  def testStudentPDFAndLogPDF(self):
    batch_size = 6
    df = tf.constant([3.] * batch_size)
    mu = tf.constant([7.] * batch_size)
    sigma = tf.constant([8.] * batch_size)
    df_v = 3.
    mu_v = 7.
    sigma_v = 8.
    t = np.array([-2.5, 2.5, 8., 0., -1., 2.], dtype=np.float32)
    student = tfd.StudentT(df, loc=mu, scale=-sigma, validate_args=True)

    log_pdf = student.log_prob(t)
    self.assertEquals(log_pdf.shape, (6,))
    log_pdf_values = self.evaluate(log_pdf)
    pdf = student.prob(t)
    self.assertEquals(pdf.shape, (6,))
    pdf_values = self.evaluate(pdf)

    expected_log_pdf = sp_stats.t.logpdf(t, df_v, loc=mu_v, scale=sigma_v)
    expected_pdf = sp_stats.t.pdf(t, df_v, loc=mu_v, scale=sigma_v)
    self.assertAllClose(expected_log_pdf, log_pdf_values)
    self.assertAllClose(np.log(expected_pdf), log_pdf_values)
    self.assertAllClose(expected_pdf, pdf_values)
    self.assertAllClose(np.exp(expected_log_pdf), pdf_values)

  def testStudentLogPDFMultidimensional(self):
    batch_size = 6
    df = tf.constant([[1.5, 7.2]] * batch_size)
    mu = tf.constant([[3., -3.]] * batch_size)
    sigma = tf.constant(
        [[-math.sqrt(10.), math.sqrt(15.)]] * batch_size)
    df_v = np.array([1.5, 7.2])
    mu_v = np.array([3., -3.])
    sigma_v = np.array([np.sqrt(10.), np.sqrt(15.)])
    t = np.array([[-2.5, 2.5, 4., 0., -1., 2.]], dtype=np.float32).T
    student = tfd.StudentT(df, loc=mu, scale=sigma, validate_args=True)
    log_pdf = student.log_prob(t)
    log_pdf_values = self.evaluate(log_pdf)
    self.assertEqual(log_pdf.shape, (6, 2))
    pdf = student.prob(t)
    pdf_values = self.evaluate(pdf)
    self.assertEqual(pdf.shape, (6, 2))

    expected_log_pdf = sp_stats.t.logpdf(t, df_v, loc=mu_v, scale=sigma_v)
    expected_pdf = sp_stats.t.pdf(t, df_v, loc=mu_v, scale=sigma_v)
    self.assertAllClose(expected_log_pdf, log_pdf_values)
    self.assertAllClose(np.log(expected_pdf), log_pdf_values)
    self.assertAllClose(expected_pdf, pdf_values)
    self.assertAllClose(np.exp(expected_log_pdf), pdf_values)

  def testStudentCDFAndLogCDF(self):
    batch_size = 6
    df = tf.constant([3.] * batch_size)
    mu = tf.constant([7.] * batch_size)
    sigma = tf.constant([-8.] * batch_size)
    df_v = 3.
    mu_v = 7.
    sigma_v = 8.
    t = np.array([-2.5, 2.5, 8., 0., -1., 2.], dtype=np.float32)
    student = tfd.StudentT(df, loc=mu, scale=sigma, validate_args=True)

    log_cdf = student.log_cdf(t)
    self.assertEquals(log_cdf.shape, (6,))
    log_cdf_values = self.evaluate(log_cdf)
    cdf = student.cdf(t)
    self.assertEquals(cdf.shape, (6,))
    cdf_values = self.evaluate(cdf)

    expected_log_cdf = sp_stats.t.logcdf(t, df_v, loc=mu_v, scale=sigma_v)
    expected_cdf = sp_stats.t.cdf(t, df_v, loc=mu_v, scale=sigma_v)
    self.assertAllClose(expected_log_cdf, log_cdf_values, atol=0., rtol=1e-5)
    self.assertAllClose(
        np.log(expected_cdf), log_cdf_values, atol=0., rtol=1e-5)
    self.assertAllClose(expected_cdf, cdf_values, atol=0., rtol=1e-5)
    self.assertAllClose(
        np.exp(expected_log_cdf), cdf_values, atol=0., rtol=1e-5)

  def testStudentEntropy(self):
    df_v = np.array([[2., 3., 7.]])  # 1x3
    mu_v = np.array([[1., -1, 0]])  # 1x3
    sigma_v = np.array([[1., -2., 3.]]).T  # transposed => 3x1
    student = tfd.StudentT(df=df_v, loc=mu_v, scale=sigma_v, validate_args=True)
    ent = student.entropy()
    ent_values = self.evaluate(ent)

    # Help scipy broadcast to 3x3
    ones = np.array([[1, 1, 1]])
    sigma_bc = np.abs(sigma_v) * ones
    mu_bc = ones.T * mu_v
    df_bc = ones.T * df_v
    expected_entropy = sp_stats.t.entropy(
        np.reshape(df_bc, [-1]),
        loc=np.reshape(mu_bc, [-1]),
        scale=np.reshape(sigma_bc, [-1]))
    expected_entropy = np.reshape(expected_entropy, df_bc.shape)
    self.assertAllClose(expected_entropy, ent_values)

  def testStudentSample(self):
    df = tf.constant(4.)
    mu = tf.constant(3.)
    sigma = tf.constant(-math.sqrt(10.))
    df_v = 4.
    mu_v = 3.
    sigma_v = np.sqrt(10.)
    n = tf.constant(200000)
    student = tfd.StudentT(df=df, loc=mu, scale=sigma, validate_args=True)
    samples = student.sample(n, seed=test_util.test_seed())
    sample_values = self.evaluate(samples)
    n_val = 200000
    self.assertEqual(sample_values.shape, (n_val,))
    self.assertAllClose(sample_values.mean(), mu_v, rtol=0.1, atol=0)
    self.assertAllClose(
        sample_values.var(), sigma_v**2 * df_v / (df_v - 2), rtol=0.1, atol=0)
    self._checkKLApprox(df_v, mu_v, sigma_v, sample_values)

  # Test that sampling with the same seed twice gives the same results.
  def testStudentSampleMultipleTimes(self):
    df = tf.constant(4.)
    mu = tf.constant(3.)
    sigma = tf.constant(math.sqrt(10.))
    n = tf.constant(100)
    seed = test_util.test_seed()

    tf.random.set_seed(seed)
    student = tfd.StudentT(
        df=df, loc=mu, scale=sigma, name='student_t1', validate_args=True)
    samples1 = self.evaluate(student.sample(n, seed=seed))

    tf.random.set_seed(seed)
    student2 = tfd.StudentT(
        df=df, loc=mu, scale=sigma, name='student_t2', validate_args=True)
    samples2 = self.evaluate(student2.sample(n, seed=seed))

    self.assertAllClose(samples1, samples2)

  def testStudentSampleSmallDfNoNan(self):
    df_v = [1e-1, 1e-5, 1e-10, 1e-20]
    df = tf.constant(df_v)
    n = tf.constant(200000)
    student = tfd.StudentT(df=df, loc=1., scale=1., validate_args=True)
    samples = student.sample(n, seed=test_util.test_seed())
    sample_values = self.evaluate(samples)
    n_val = 200000
    self.assertEqual(sample_values.shape, (n_val, 4))
    self.assertTrue(np.all(np.logical_not(np.isnan(sample_values))))

  def testStudentSampleMultiDimensional(self):
    batch_size = 7
    df = tf.constant([[5., 7.]] * batch_size)
    mu = tf.constant([[3., -3.]] * batch_size)
    sigma = tf.constant(
        [[math.sqrt(10.), math.sqrt(15.)]] * batch_size)
    df_v = [5., 7.]
    mu_v = [3., -3.]
    sigma_v = [np.sqrt(10.), np.sqrt(15.)]
    n = tf.constant(200000)
    student = tfd.StudentT(df=df, loc=mu, scale=sigma, validate_args=True)
    samples = student.sample(n, seed=test_util.test_seed())
    sample_values = self.evaluate(samples)
    self.assertEqual(samples.shape, (200000, batch_size, 2))
    self.assertAllClose(
        sample_values[:, 0, 0].mean(), mu_v[0], rtol=0.1, atol=0)
    self.assertAllClose(
        sample_values[:, 0, 0].var(),
        sigma_v[0]**2 * df_v[0] / (df_v[0] - 2),
        rtol=0.2,
        atol=0)
    self._checkKLApprox(df_v[0], mu_v[0], sigma_v[0], sample_values[:, 0, 0])
    self.assertAllClose(
        sample_values[:, 0, 1].mean(), mu_v[1], rtol=0.1, atol=0)
    self.assertAllClose(
        sample_values[:, 0, 1].var(),
        sigma_v[1]**2 * df_v[1] / (df_v[1] - 2),
        rtol=0.2,
        atol=0)
    self._checkKLApprox(df_v[1], mu_v[1], sigma_v[1], sample_values[:, 0, 1])

  def _checkKLApprox(self, df, mu, sigma, samples):
    n = samples.size
    np.random.seed(137)
    sample_scipy = sp_stats.t.rvs(df, loc=mu, scale=sigma, size=n)
    covg = 0.99
    r = sp_stats.t.interval(covg, df, loc=mu, scale=sigma)
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

    def _check(student):
      self.assertEqual(student.mean().shape, (3,))
      self.assertEqual(student.variance().shape, (3,))
      self.assertEqual(student.entropy().shape, (3,))
      self.assertEqual(student.log_prob(2.).shape, (3,))
      self.assertEqual(student.prob(2.).shape, (3,))
      self.assertEqual(student.sample(
          37, seed=test_util.test_seed()).shape, (37, 3,))

    _check(
        tfd.StudentT(df=[
            2.,
            3.,
            4.,
        ], loc=2., scale=1., validate_args=True))
    _check(
        tfd.StudentT(df=7., loc=[
            2.,
            3.,
            4.,
        ], scale=1., validate_args=True))
    _check(
        tfd.StudentT(df=7., loc=3., scale=[
            2.,
            3.,
            4.,
        ], validate_args=True))

  def testBroadcastingPdfArgs(self):

    def _assert_shape(student, arg, shape):
      self.assertEqual(student.log_prob(arg).shape, shape)
      self.assertEqual(student.prob(arg).shape, shape)

    def _check(student):
      _assert_shape(student, 2., (3,))
      xs = np.array([2., 3., 4.], dtype=np.float32)
      _assert_shape(student, xs, (3,))
      xs = np.array([xs])
      _assert_shape(student, xs, (1, 3))
      xs = xs.T
      _assert_shape(student, xs, (3, 3))

    _check(
        tfd.StudentT(df=[
            2.,
            3.,
            4.,
        ], loc=2., scale=1., validate_args=True))
    _check(
        tfd.StudentT(df=7., loc=[
            2.,
            3.,
            4.,
        ], scale=1., validate_args=True))
    _check(
        tfd.StudentT(df=7., loc=3., scale=[
            2.,
            3.,
            4.,
        ], validate_args=True))

    def _check2d(student):
      _assert_shape(student, 2., (1, 3))
      xs = np.array([2., 3., 4.], dtype=np.float32)
      _assert_shape(student, xs, (1, 3))
      xs = np.array([xs])
      _assert_shape(student, xs, (1, 3))
      xs = xs.T
      _assert_shape(student, xs, (3, 3))

    _check2d(
        tfd.StudentT(df=[[
            2.,
            3.,
            4.,
        ]], loc=2., scale=1., validate_args=True))
    _check2d(
        tfd.StudentT(df=7., loc=[[
            2.,
            3.,
            4.,
        ]], scale=1., validate_args=True))
    _check2d(
        tfd.StudentT(df=7., loc=3., scale=[[
            2.,
            3.,
            4.,
        ]], validate_args=True))

    def _check2d_rows(student):
      _assert_shape(student, 2., (3, 1))
      xs = np.array([2., 3., 4.], dtype=np.float32)  # (3,)
      _assert_shape(student, xs, (3, 3))
      xs = np.array([xs])  # (1,3)
      _assert_shape(student, xs, (3, 3))
      xs = xs.T  # (3,1)
      _assert_shape(student, xs, (3, 1))

    _check2d_rows(
        tfd.StudentT(
            df=[[2.], [3.], [4.]], loc=2., scale=1., validate_args=True))
    _check2d_rows(
        tfd.StudentT(
            df=7., loc=[[2.], [3.], [4.]], scale=1., validate_args=True))
    _check2d_rows(
        tfd.StudentT(
            df=7., loc=3., scale=[[2.], [3.], [4.]], validate_args=True))

  def testMeanAllowNanStatsIsFalseWorksWhenAllBatchMembersAreDefined(self):
    mu = [1., 3.3, 4.4]
    student = tfd.StudentT(
        df=[3., 5., 7.], loc=mu, scale=[3., 2., 1.], validate_args=True)
    mean = self.evaluate(student.mean())
    self.assertAllClose([1., 3.3, 4.4], mean)

  def testMeanAllowNanStatsIsFalseRaisesWhenBatchMemberIsUndefined(self):
    mu = [1., 3.3, 4.4]
    student = tfd.StudentT(
        df=[0.5, 5., 7.],
        loc=mu,
        scale=[3., 2., 1.],
        allow_nan_stats=False,
        validate_args=True)
    with self.assertRaisesOpError('x < y'):
      self.evaluate(student.mean())

  def testMeanAllowNanStatsIsTrueReturnsNaNForUndefinedBatchMembers(self):
    mu = [-2, 0., 1., 3.3, 4.4]
    sigma = [5., 4., 3., 2., 1.]
    student = tfd.StudentT(
        df=[0.5, 1., 3., 5., 7.],
        loc=mu,
        scale=sigma,
        allow_nan_stats=True,
        validate_args=True)
    mean = self.evaluate(student.mean())
    self.assertAllClose([np.nan, np.nan, 1., 3.3, 4.4], mean)

  def testVarianceAllowNanStatsTrueReturnsNaNforUndefinedBatchMembers(self):
    # df = 0.5 ==> undefined mean ==> undefined variance.
    # df = 1.5 ==> infinite variance.
    df = [0.5, 1.5, 3., 5., 7.]
    mu = [-2, 0., 1., 3.3, 4.4]
    sigma = [5., 4., 3., 2., 1.]
    student = tfd.StudentT(
        df=df, loc=mu, scale=sigma, allow_nan_stats=True, validate_args=True)
    var = self.evaluate(student.variance())
    # Past versions of scipy differed from our preferred behavior for undefined
    # variance; newer versions changed this, but in order not to deal with
    # straddling both versions, we just override our expectation for the
    # undefined case.
    expected_var = [
        sp_stats.t.var(d, loc=m, scale=s) for (d, m, s) in zip(df, mu, sigma)
    ]
    expected_var[0] = np.nan
    self.assertAllClose(expected_var, var)

  def testVarianceAllowNanStatsFalseGivesCorrectValueForDefinedBatchMembers(
      self):
    # df = 1.5 ==> infinite variance.
    df = [1.5, 3., 5., 7.]
    mu = [0., 1., 3.3, 4.4]
    sigma = [4., 3., 2., 1.]
    student = tfd.StudentT(df=df, loc=mu, scale=sigma, validate_args=True)
    var = self.evaluate(student.variance())

    expected_var = [
        sp_stats.t.var(d, loc=m, scale=s) for (d, m, s) in zip(df, mu, sigma)
    ]
    self.assertAllClose(expected_var, var)

  def testVarianceAllowNanStatsFalseRaisesForUndefinedBatchMembers(self):
    # df <= 1 ==> variance not defined
    student = tfd.StudentT(
        df=1., loc=0., scale=1., allow_nan_stats=False, validate_args=True)
    with self.assertRaisesOpError('x < y'):
      self.evaluate(student.variance())

    # df <= 1 ==> variance not defined
    student = tfd.StudentT(
        df=0.5, loc=0., scale=1., allow_nan_stats=False, validate_args=True)
    with self.assertRaisesOpError('x < y'):
      self.evaluate(student.variance())

  def testStd(self):
    # Defined for all batch members.
    df = [3.5, 5., 3., 5., 7.]
    mu = [-2.2]
    sigma = [5., 4., 3., 2., 1.]
    student = tfd.StudentT(df=df, loc=mu, scale=sigma, validate_args=True)
    # Test broadcast of mu across shape of df/sigma
    stddev = self.evaluate(student.stddev())
    mu *= len(df)

    expected_stddev = [
        sp_stats.t.std(d, loc=m, scale=s) for (d, m, s) in zip(df, mu, sigma)
    ]
    self.assertAllClose(expected_stddev, stddev)

  def testMode(self):
    df = [0.5, 1., 3]
    mu = [-1, 0., 1]
    sigma = [5., 4., 3.]
    student = tfd.StudentT(df=df, loc=mu, scale=sigma, validate_args=True)
    # Test broadcast of mu across shape of df/sigma
    mode = self.evaluate(student.mode())
    self.assertAllClose([-1., 0, 1], mode)

  def testPdfOfSample(self):
    student = tfd.StudentT(df=3., loc=np.pi, scale=1., validate_args=True)
    num = 20000
    samples = student.sample(num, seed=test_util.test_seed())
    pdfs = student.prob(samples)
    mean = student.mean()
    mean_pdf = student.prob(student.mean())
    sample_vals, pdf_vals, mean_val, mean_pdf_val = self.evaluate(
        [samples, pdfs, student.mean(), mean_pdf])
    self.assertEqual(samples.shape, (num,))
    self.assertEqual(pdfs.shape, (num,))
    self.assertEqual(mean.shape, ())
    self.assertNear(np.pi, np.mean(sample_vals), err=0.1)
    self.assertNear(np.pi, mean_val, err=1e-6)
    # Verify integral over sample*pdf ~= 1.
    # Tolerance increased since eager was getting a value of 1.002041.
    self._assertIntegral(sample_vals, pdf_vals, err=5e-2)
    self.assertNear(
        sp_stats.t.pdf(np.pi, 3., loc=np.pi), mean_pdf_val, err=1e-6)

  def testFullyReparameterized(self):
    df = tf.constant(2.0)
    mu = tf.constant(1.0)
    sigma = tf.constant(3.0)
    _, [grad_df, grad_mu, grad_sigma] = tfp.math.value_and_gradient(
        lambda d, m, s: tfd.StudentT(df=d, loc=m, scale=s, validate_args=True).  # pylint: disable=g-long-lambda
        sample(100, seed=test_util.test_seed()), [df, mu, sigma])
    self.assertIsNotNone(grad_df)
    self.assertIsNotNone(grad_mu)
    self.assertIsNotNone(grad_sigma)

  def testPdfOfSampleMultiDims(self):
    student = tfd.StudentT(
        df=[7., 11.], loc=[[5.], [6.]], scale=3., validate_args=True)
    self.assertAllEqual([], student.event_shape)
    self.assertAllEqual([], self.evaluate(student.event_shape_tensor()))
    self.assertAllEqual([2, 2], student.batch_shape)
    self.assertAllEqual([2, 2], self.evaluate(student.batch_shape_tensor()))
    num = 50000
    samples = student.sample(num, seed=test_util.test_seed())
    pdfs = student.prob(samples)
    sample_vals, pdf_vals = self.evaluate([samples, pdfs])
    self.assertEqual(samples.shape, (num, 2, 2))
    self.assertEqual(pdfs.shape, (num, 2, 2))
    self.assertNear(5., np.mean(sample_vals[:, 0, :]), err=0.1)
    self.assertNear(6., np.mean(sample_vals[:, 1, :]), err=0.1)
    self._assertIntegral(sample_vals[:, 0, 0], pdf_vals[:, 0, 0], err=0.05)
    self._assertIntegral(sample_vals[:, 0, 1], pdf_vals[:, 0, 1], err=0.05)
    self._assertIntegral(sample_vals[:, 1, 0], pdf_vals[:, 1, 0], err=0.05)
    self._assertIntegral(sample_vals[:, 1, 1], pdf_vals[:, 1, 1], err=0.05)
    self.assertNear(
        sp_stats.t.var(7., loc=0., scale=3.),  # loc d.n. effect var
        np.var(sample_vals[:, :, 0]),
        err=1.0)
    self.assertNear(
        sp_stats.t.var(11., loc=0., scale=3.),  # loc d.n. effect var
        np.var(sample_vals[:, :, 1]),
        err=1.0)

  def _assertIntegral(self, sample_vals, pdf_vals, err=1.5e-3):
    s_p = zip(sample_vals, pdf_vals)
    prev = (sample_vals.min() - 1000, 0)
    total = 0
    for k in sorted(s_p, key=lambda x: x[0]):
      pair_pdf = (k[1] + prev[1]) / 2
      total += (k[0] - prev[0]) * pair_pdf
      prev = k
    self.assertNear(1., total, err=err)

  def testNegativeDofFails(self):
    with self.assertRaisesOpError(r'`df` must be positive'):
      student = tfd.StudentT(
          df=[2, -5.], loc=0., scale=1., validate_args=True, name='S')
      self.evaluate(student.mean())

  @test_util.tf_tape_safety_test
  def testGradientThroughParams(self):
    df = tf.Variable([[17.3], [14.]])
    loc = tf.Variable([[-5., 0., 5.]])
    scale = tf.Variable(2.)
    d = tfd.StudentT(df=df, loc=loc, scale=scale, validate_args=True)
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
      tfd.StudentT(df=df, loc=loc, scale=scale, validate_args=True)


if __name__ == '__main__':
  tf.test.main()
