# Copyright 2018 The TensorFlow Probability Authors.
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Tests for Student t distribution."""

import itertools
import math

# Dependency imports
from absl.testing import parameterized
import numpy as np
from scipy import special as sp_special
from scipy import stats as sp_stats

import tensorflow.compat.v2 as tf

from tensorflow_probability.python.distributions import student_t
from tensorflow_probability.python.distributions import uniform
from tensorflow_probability.python.internal import test_util
from tensorflow_probability.python.math import gradient


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
    student = student_t.StudentT(df, loc=mu, scale=-sigma, validate_args=True)

    log_pdf = student.log_prob(t)
    self.assertAllEqual(log_pdf.shape, (6,))
    log_pdf_values = self.evaluate(log_pdf)
    pdf = student.prob(t)
    self.assertAllEqual(pdf.shape, (6,))
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
    student = student_t.StudentT(df, loc=mu, scale=sigma, validate_args=True)
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
    student = student_t.StudentT(df, loc=mu, scale=sigma, validate_args=True)

    log_cdf = student.log_cdf(t)
    self.assertAllEqual(log_cdf.shape, (6,))
    log_cdf_values = self.evaluate(log_cdf)
    cdf = student.cdf(t)
    self.assertAllEqual(cdf.shape, (6,))
    cdf_values = self.evaluate(cdf)

    expected_log_cdf = sp_stats.t.logcdf(t, df_v, loc=mu_v, scale=sigma_v)
    expected_cdf = sp_stats.t.cdf(t, df_v, loc=mu_v, scale=sigma_v)
    self.assertAllClose(expected_log_cdf, log_cdf_values, atol=0.)
    self.assertAllClose(np.log(expected_cdf), log_cdf_values, atol=0.)
    self.assertAllClose(expected_cdf, cdf_values, atol=0.)
    self.assertAllClose(np.exp(expected_log_cdf), cdf_values, atol=0.)

  def testStudentSurvivalFunctionAndLogSurvivalFunction(self):
    batch_size = 6
    df = tf.constant([3.] * batch_size)
    mu = tf.constant([7.] * batch_size)
    sigma = tf.constant([-8.] * batch_size)
    df_v = 3.
    mu_v = 7.
    sigma_v = 8.
    t = np.array([-2.5, 2.5, 8., 0., -1., 2.], dtype=np.float32)
    student = student_t.StudentT(df, loc=mu, scale=sigma, validate_args=True)

    log_sf = student.log_survival_function(t)
    self.assertAllEqual(log_sf.shape, (6,))
    log_sf_values = self.evaluate(log_sf)
    sf = student.survival_function(t)
    self.assertAllEqual(sf.shape, (6,))
    sf_values = self.evaluate(sf)

    expected_log_sf = sp_stats.t.logsf(t, df_v, loc=mu_v, scale=sigma_v)
    expected_sf = sp_stats.t.sf(t, df_v, loc=mu_v, scale=sigma_v)
    self.assertAllClose(expected_log_sf, log_sf_values, atol=0.)
    self.assertAllClose(np.log(expected_sf), log_sf_values, atol=0.)
    self.assertAllClose(expected_sf, sf_values, atol=0.)
    self.assertAllClose(np.exp(expected_log_sf), sf_values, atol=0.)

  def testStudentQuantile(self):
    # The SciPy implementation of quantile function is not very accurate. E.g.:
    #
    #   numpy_dtype = np.float64
    #   df = numpy_dtype(9.)
    #   p = np.finfo(numpy_dtype).eps
    #   t = sp_special.stdtrit(df, p)
    #   relerr = np.abs((p - sp_special.stdtr(df, t)) / p)
    #   relerr > 1e-8
    #
    # So instead of comparing expected_quantile and quantile_values, we compare
    # p (or expected_cdf) and cdf_values, where cdf_values is computed applying
    # the SciPy implementation of cdf function to quantile_values.
    batch_shape = (40, 1)
    df = np.random.uniform(
        low=0.5, high=10., size=batch_shape).astype(np.float32)
    mu = 7. * np.ones(batch_shape, dtype=np.float32)
    sigma = -8.
    p = np.logspace(-4., -0.01, 20).astype(np.float32)
    student = student_t.StudentT(
        df, loc=mu, scale=sigma, validate_args=True)

    quantile = student.quantile(p)
    self.assertAllEqual(quantile.shape, (40, 20))
    quantile_values = self.evaluate(quantile)
    cdf_values = sp_stats.t.cdf(
        quantile_values, df, loc=mu, scale=np.abs(sigma))

    expected_cdf = np.broadcast_to(p, shape=(40, 20))
    self.assertAllClose(expected_cdf, cdf_values, atol=0., rtol=5e-5)

  def testStudentEntropy(self):
    df_v = np.array([[2., 3., 7.]])  # 1x3
    mu_v = np.array([[1., -1, 0]])  # 1x3
    sigma_v = np.array([[1., -2., 3.]]).T  # transposed => 3x1
    student = student_t.StudentT(
        df=df_v, loc=mu_v, scale=sigma_v, validate_args=True)
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
    student = student_t.StudentT(df=df, loc=mu, scale=sigma, validate_args=True)
    samples = student.sample(n, seed=test_util.test_seed())
    sample_values = self.evaluate(samples)
    n_val = 200000
    self.assertEqual(sample_values.shape, (n_val,))
    self.assertAllClose(sample_values.mean(), mu_v, rtol=0.1, atol=0)
    self.assertAllClose(
        sample_values.var(), sigma_v**2 * df_v / (df_v - 2), rtol=0.1, atol=0)
    self._check_kl_approx(df_v, mu_v, sigma_v, sample_values)

  # Test that sampling with the same seed twice gives the same results.
  def testStudentSampleMultipleTimes(self):
    df = tf.constant(4.)
    mu = tf.constant(3.)
    sigma = tf.constant(math.sqrt(10.))
    n = tf.constant(100)
    seed = test_util.test_seed()

    tf.random.set_seed(seed)
    student = student_t.StudentT(
        df=df, loc=mu, scale=sigma, name='student_t1', validate_args=True)
    samples1 = self.evaluate(student.sample(n, seed=seed))

    tf.random.set_seed(seed)
    student2 = student_t.StudentT(
        df=df, loc=mu, scale=sigma, name='student_t2', validate_args=True)
    samples2 = self.evaluate(student2.sample(n, seed=seed))

    self.assertAllClose(samples1, samples2)

  def testStudentSampleSmallDfNoNan(self):
    df_v = [1e-1, 1e-5, 1e-10, 1e-20]
    df = tf.constant(df_v)
    n = tf.constant(200000)
    student = student_t.StudentT(df=df, loc=1., scale=1., validate_args=True)
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
    student = student_t.StudentT(df=df, loc=mu, scale=sigma, validate_args=True)
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
    self._check_kl_approx(df_v[0], mu_v[0], sigma_v[0], sample_values[:, 0, 0])
    self.assertAllClose(
        sample_values[:, 0, 1].mean(), mu_v[1], rtol=0.1, atol=0)
    self.assertAllClose(
        sample_values[:, 0, 1].var(),
        sigma_v[1]**2 * df_v[1] / (df_v[1] - 2),
        rtol=0.2,
        atol=0)
    self._check_kl_approx(df_v[1], mu_v[1], sigma_v[1], sample_values[:, 0, 1])

  def _check_kl_approx(self, df, mu, sigma, samples):
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
        student_t.StudentT(
            df=[
                2.,
                3.,
                4.,
            ], loc=2., scale=1., validate_args=True))
    _check(
        student_t.StudentT(
            df=7., loc=[
                2.,
                3.,
                4.,
            ], scale=1., validate_args=True))
    _check(
        student_t.StudentT(
            df=7., loc=3., scale=[
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
        student_t.StudentT(
            df=[
                2.,
                3.,
                4.,
            ], loc=2., scale=1., validate_args=True))
    _check(
        student_t.StudentT(
            df=7., loc=[
                2.,
                3.,
                4.,
            ], scale=1., validate_args=True))
    _check(
        student_t.StudentT(
            df=7., loc=3., scale=[
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
        student_t.StudentT(
            df=[[
                2.,
                3.,
                4.,
            ]], loc=2., scale=1., validate_args=True))
    _check2d(
        student_t.StudentT(
            df=7., loc=[[
                2.,
                3.,
                4.,
            ]], scale=1., validate_args=True))
    _check2d(
        student_t.StudentT(
            df=7., loc=3., scale=[[
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
        student_t.StudentT(
            df=[[2.], [3.], [4.]], loc=2., scale=1., validate_args=True))
    _check2d_rows(
        student_t.StudentT(
            df=7., loc=[[2.], [3.], [4.]], scale=1., validate_args=True))
    _check2d_rows(
        student_t.StudentT(
            df=7., loc=3., scale=[[2.], [3.], [4.]], validate_args=True))

  def testMeanAllowNanStatsIsFalseWorksWhenAllBatchMembersAreDefined(self):
    mu = [1., 3.3, 4.4]
    student = student_t.StudentT(
        df=[3., 5., 7.], loc=mu, scale=[3., 2., 1.], validate_args=True)
    mean = self.evaluate(student.mean())
    self.assertAllClose([1., 3.3, 4.4], mean)

  def testMeanAllowNanStatsIsFalseRaisesWhenBatchMemberIsUndefined(self):
    mu = [1., 3.3, 4.4]
    student = student_t.StudentT(
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
    student = student_t.StudentT(
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
    student = student_t.StudentT(
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
    student = student_t.StudentT(df=df, loc=mu, scale=sigma, validate_args=True)
    var = self.evaluate(student.variance())

    expected_var = [
        sp_stats.t.var(d, loc=m, scale=s) for (d, m, s) in zip(df, mu, sigma)
    ]
    self.assertAllClose(expected_var, var)

  def testVarianceAllowNanStatsFalseRaisesForUndefinedBatchMembers(self):
    # df <= 1 ==> variance not defined
    student = student_t.StudentT(
        df=1., loc=0., scale=1., allow_nan_stats=False, validate_args=True)
    with self.assertRaisesOpError('x < y'):
      self.evaluate(student.variance())

    # df <= 1 ==> variance not defined
    student = student_t.StudentT(
        df=0.5, loc=0., scale=1., allow_nan_stats=False, validate_args=True)
    with self.assertRaisesOpError('x < y'):
      self.evaluate(student.variance())

  def testStd(self):
    # Defined for all batch members.
    df = [3.5, 5., 3., 5., 7.]
    mu = [-2.2]
    sigma = [5., 4., 3., 2., 1.]
    student = student_t.StudentT(df=df, loc=mu, scale=sigma, validate_args=True)
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
    student = student_t.StudentT(df=df, loc=mu, scale=sigma, validate_args=True)
    # Test broadcast of mu across shape of df/sigma
    mode = self.evaluate(student.mode())
    self.assertAllClose([-1., 0, 1], mode)

  def testPdfOfSample(self):
    student = student_t.StudentT(df=3., loc=np.pi, scale=1., validate_args=True)
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

  @test_util.numpy_disable_gradient_test
  def testFullyReparameterized(self):
    df = tf.constant(2.0)
    mu = tf.constant(1.0)
    sigma = tf.constant(3.0)
    grad_df, grad_mu, grad_sigma = gradient.value_and_gradient(
        lambda d, m, s: student_t.StudentT(  # pylint: disable=g-long-lambda
            df=d,
            loc=m,
            scale=s,
            validate_args=True).sample(100, seed=test_util.test_seed()),
        [df, mu, sigma])[1]
    self.assertIsNotNone(grad_df)
    self.assertIsNotNone(grad_mu)
    self.assertIsNotNone(grad_sigma)

  def testPdfOfSampleMultiDims(self):
    student = student_t.StudentT(
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
      student = student_t.StudentT(
          df=[2, -5.], loc=0., scale=1., validate_args=True, name='S')
      self.evaluate(student.mean())

  @test_util.tf_tape_safety_test
  def testGradientThroughParams(self):
    df = tf.Variable([[17.3], [14.]])
    loc = tf.Variable([[-5., 0., 5.]])
    scale = tf.Variable(2.)
    d = student_t.StudentT(df=df, loc=loc, scale=scale, validate_args=True)
    with tf.GradientTape() as tape:
      loss = -d.log_prob(np.ones((2, 3)))
    grad = tape.gradient(loss, d.trainable_variables)
    self.assertLen(grad, 3)
    self.assertAllNotNone(grad)

  def testAssertParamsAreFloats(self):
    df = tf.Variable(14, dtype=tf.int32)
    loc = tf.Variable(0, dtype=tf.int32)
    scale = tf.Variable(1, dtype=tf.int32)
    with self.assertRaisesRegex(ValueError, 'Expected floating point'):
      student_t.StudentT(df=df, loc=loc, scale=scale, validate_args=True)


@test_util.test_graph_and_eager_modes
class StdtrTest(test_util.TestCase):

  def testStdtrBroadcast(self):
    df = np.ones([3, 2], dtype=np.float32)
    t = np.ones([4, 1, 1], dtype=np.float32)
    self.assertAllEqual([4, 3, 2], student_t.stdtr(df, t).shape)

  def testStdtrDtype(self):
    df = np.ones(1, dtype=np.float16)
    t = np.ones(1, dtype=np.float16)

    with self.assertRaisesRegex(TypeError, 'df.dtype=(.*?) is not handled'):
      student_t.stdtr(df, t)

  def _test_stdtr_value(self, df_low, df_high, use_log10_scale, dtype, rtol):
    tiny = np.finfo(dtype).tiny
    n = [int(5e2)]
    strm = test_util.test_seed_stream()

    df = uniform.Uniform(
        low=dtype(df_low), high=dtype(df_high)).sample(n, strm())
    df = tf.math.pow(dtype(10.), df) if use_log10_scale else df
    p = uniform.Uniform(low=tiny, high=dtype(1.)).sample(n, strm())
    df, p = self.evaluate([df, p])
    t = sp_special.stdtrit(df, p)

    # Wrap in tf.function for faster computations.
    stdtr = tf.function(student_t.stdtr, autograph=False)

    cdf_values = self.evaluate(stdtr(df, t))
    self.assertDTypeEqual(cdf_values, dtype)

    expected_cdf = sp_special.stdtr(df, t)
    self.assertAllClose(expected_cdf, cdf_values, atol=0., rtol=rtol)

  @parameterized.named_parameters(
      {'testcase_name': 'float32',
       'dtype': np.float32,
       'rtol': 5e-5},
      {'testcase_name': 'float64',
       'dtype': np.float64,
       'rtol': 5e-13})
  def testStdtrSmall(self, dtype, rtol):
    self._test_stdtr_value(
        df_low=0.5, df_high=10., use_log10_scale=False, dtype=dtype, rtol=rtol)

  @parameterized.named_parameters(
      {'testcase_name': 'float32',
       'dtype': np.float32,
       'rtol': 1e-5},
      {'testcase_name': 'float64',
       'dtype': np.float64,
       'rtol': 1e-12})
  def testStdtrMedium(self, dtype, rtol):
    self._test_stdtr_value(
        df_low=10., df_high=1e2, use_log10_scale=False, dtype=dtype, rtol=rtol)

  @parameterized.named_parameters(
      {'testcase_name': 'float32',
       'dtype': np.float32,
       'rtol': 1e-5},
      {'testcase_name': 'float64',
       'dtype': np.float64,
       'rtol': 1e-12})
  def testStdtrLarge(self, dtype, rtol):
    self._test_stdtr_value(
        df_low=1e2, df_high=1e4, use_log10_scale=False, dtype=dtype, rtol=rtol)

  @parameterized.named_parameters(
      {'testcase_name': 'float64',
       'dtype': np.float64,
       'rtol': 5e-14})
  def testStdtrVeryLarge(self, dtype, rtol):
    self._test_stdtr_value(
        df_low=4., df_high=8., use_log10_scale=True, dtype=dtype, rtol=rtol)

  @parameterized.parameters(np.float32, np.float64)
  def testStdtrBounds(self, dtype):
    # Test out-of-range values (should return NaN output).
    df = np.array([-1., 0.], dtype=dtype)
    t = np.array([0.5, 0.5], dtype=dtype)

    result = self.evaluate(student_t.stdtr(df, t))
    self.assertDTypeEqual(result, dtype)
    self.assertAllNan(result)

  @test_util.numpy_disable_gradient_test
  def testStdtrGradient(self):
    space_df = np.logspace(np.log10(0.5), 8., num=15).tolist()
    space_p = np.linspace(0.01, 0.99, num=15).tolist()
    df, p = zip(*list(itertools.product(space_df, space_p)))
    t = sp_special.stdtrit(df, p)
    df, t = [tf.constant(z, dtype='float64') for z in (df, t)]

    # Wrap in tf.function for faster computations.
    stdtr = tf.function(student_t.stdtr, autograph=False)

    err = self.compute_max_gradient_error(
        lambda z: stdtr(z, t), [df], delta=1e-5)
    self.assertLess(err, 2e-10)

    err = self.compute_max_gradient_error(
        lambda z: stdtr(df, z), [t], delta=1e-5)
    self.assertLess(err, 7e-11)

  @parameterized.parameters(np.float32, np.float64)
  @test_util.numpy_disable_gradient_test
  def testStdtrGradientFinite(self, dtype):
    eps = np.finfo(dtype).eps
    log_df_max = 4. if dtype == np.float32 else 8.

    space_df = np.logspace(np.log10(0.5), log_df_max, num=15).tolist()
    # An odd num > 1 ensures that 0.5 is in np.linspace(eps, 1. - eps, num).
    space_p = np.linspace(eps, 1. - eps, num=13).tolist()
    space_p += [0.5 - eps, 0.5 + eps]
    df, p = zip(*list(itertools.product(space_df, space_p)))
    t = sp_special.stdtrit(df, p)
    df, t = [tf.constant(z, dtype=dtype) for z in (df, t)]
    # The result of sp_special.stdtrit(df, 0.5) can be different from 0.
    t = tf.where(tf.math.equal(p, 0.5), dtype(0.), t)

    # Wrap in tf.function for faster computations.
    @tf.function(autograph=False)
    def stdtr_partials(df, t):
      return gradient.value_and_gradient(student_t.stdtr, [df, t])[1]

    partial_df, partial_t = self.evaluate(stdtr_partials(df, t))

    self.assertDTypeEqual(partial_df, dtype)
    self.assertDTypeEqual(partial_t, dtype)
    self.assertAllFinite([partial_df, partial_t])

  @parameterized.parameters(np.float32, np.float64)
  @test_util.numpy_disable_gradient_test
  def testStdtrGradientBounds(self, dtype):
    # Test out-of-range values (should return NaN output).
    df = tf.constant([-1., 0.], dtype=dtype)
    t = tf.constant([0.5, 0.5], dtype=dtype)

    partial_df, partial_t = self.evaluate(
        gradient.value_and_gradient(student_t.stdtr, [df, t])[1])

    self.assertDTypeEqual(partial_df, dtype)
    self.assertDTypeEqual(partial_t, dtype)
    self.assertAllNan([partial_df, partial_t])

  @test_util.numpy_disable_gradient_test
  def testStdtrGradientBroadcast(self):
    df = np.ones([3, 2], dtype=np.float32)
    t = np.zeros([4, 1, 1], dtype=np.float32)

    def simple_binary_operator(df, t):
      return df + t

    simple_partials = gradient.value_and_gradient(
        simple_binary_operator, [df, t])[1]
    stdtr_partials = gradient.value_and_gradient(
        student_t.stdtr, [df, t])[1]
    df_partials, t_partials = zip([*simple_partials], [*stdtr_partials])
    self.assertShapeEqual(df_partials[0], df_partials[1])
    self.assertShapeEqual(t_partials[0], t_partials[1])

  @parameterized.parameters(np.float32, np.float64)
  @test_util.numpy_disable_gradient_test
  def testStdtrSecondDerivativeFinite(self, dtype):
    eps = np.finfo(dtype).eps

    space_df = np.logspace(np.log10(0.5), 4., num=7).tolist()
    # An odd num > 1 ensures that 0.5 is in np.linspace(eps, 1. - eps, num).
    space_p = np.linspace(eps, 1. - eps, num=5).tolist()
    space_p += [0.5 - eps, 0.5 + eps]
    df, p = zip(*list(itertools.product(space_df, space_p)))
    t = sp_special.stdtrit(df, p)
    df, t = [tf.constant(z, dtype=dtype) for z in (df, t)]
    # The result of sp_special.stdtrit(df, 0.5) can be different from 0.
    t = tf.where(tf.math.equal(p, 0.5), dtype(0.), t)

    def stdtr_partials(df, t):
      return gradient.value_and_gradient(student_t.stdtr, [df, t])[1]

    # Wrap in tf.function for faster computations.
    @tf.function(autograph=False)
    def stdtr_partials_of_partials(df, t):
      return gradient.value_and_gradient(stdtr_partials, [df, t])[1]

    partials_of_partials = stdtr_partials_of_partials(df, t)
    self.assertAllFinite(self.evaluate(partials_of_partials))


@test_util.test_graph_and_eager_modes
class StdtritTest(test_util.TestCase):

  def testStdtritBroadcast(self):
    df = np.ones([3, 2], dtype=np.float32)
    p = np.full([4, 1, 1], fill_value=0.5, dtype=np.float32)
    self.assertAllEqual([4, 3, 2], student_t.stdtrit(df, p).shape)

  def testStdtritDtype(self):
    df = np.ones(1, dtype=np.float16)
    p = np.array([0.5], dtype=np.float16)

    with self.assertRaisesRegex(TypeError, 'df.dtype=(.*?) is not handled'):
      student_t.stdtrit(df, p)

  def _test_stdtrit_value(self, df_low, df_high, use_log10_scale, dtype, rtol):
    # The SciPy implementation of quantile function is not very accurate. E.g.:
    #
    #   numpy_dtype = np.float64
    #   df = numpy_dtype(9.)
    #   p = np.finfo(numpy_dtype).eps
    #   t = sp_special.stdtrit(df, p)
    #   relerr = np.abs((p - sp_special.stdtr(df, t)) / p)
    #   relerr > 1e-8
    #
    # So instead of comparing expected_quantile and quantile_values, we compare
    # p (or expected_cdf) and cdf_values, where cdf_values is computed applying
    # the SciPy implementation of cdf function to quantile_values.
    tiny = np.finfo(dtype).tiny
    n = [int(5e2)]
    strm = test_util.test_seed_stream()

    df = uniform.Uniform(
        low=dtype(df_low), high=dtype(df_high)).sample(n, strm())
    df = tf.math.pow(dtype(10.), df) if use_log10_scale else df
    p = uniform.Uniform(low=tiny, high=dtype(1.)).sample(n, strm())

    # Wrap in tf.function for faster computations.
    stdtrit = tf.function(student_t.stdtrit, autograph=False)

    quantile_values, df, expected_cdf = self.evaluate([stdtrit(df, p), df, p])
    self.assertDTypeEqual(quantile_values, dtype)

    cdf_values = sp_special.stdtr(df, quantile_values)
    self.assertAllClose(expected_cdf, cdf_values, atol=0., rtol=rtol)

  @parameterized.named_parameters(
      {'testcase_name': 'float32',
       'dtype': np.float32,
       'rtol': 5e-5},
      {'testcase_name': 'float64',
       'dtype': np.float64,
       'rtol': 5e-13})
  def testStdtritSmall(self, dtype, rtol):
    self._test_stdtrit_value(
        df_low=0.5, df_high=10., use_log10_scale=False, dtype=dtype, rtol=rtol)

  @parameterized.named_parameters(
      {'testcase_name': 'float32',
       'dtype': np.float32,
       'rtol': 1e-5},
      {'testcase_name': 'float64',
       'dtype': np.float64,
       'rtol': 1e-12})
  def testStdtritMedium(self, dtype, rtol):
    self._test_stdtrit_value(
        df_low=10., df_high=1e2, use_log10_scale=False, dtype=dtype, rtol=rtol)

  @parameterized.named_parameters(
      {'testcase_name': 'float32',
       'dtype': np.float32,
       'rtol': 1e-5},
      {'testcase_name': 'float64',
       'dtype': np.float64,
       'rtol': 1e-12})
  def testStdtritLarge(self, dtype, rtol):
    self._test_stdtrit_value(
        df_low=1e2, df_high=1e4, use_log10_scale=False, dtype=dtype, rtol=rtol)

  @parameterized.named_parameters(
      {'testcase_name': 'float64',
       'dtype': np.float64,
       'rtol': 5e-14})
  def testStdtritVeryLarge(self, dtype, rtol):
    self._test_stdtrit_value(
        df_low=4., df_high=8., use_log10_scale=True, dtype=dtype, rtol=rtol)

  @parameterized.parameters(np.float32, np.float64)
  def testStdtritBounds(self, dtype):
    # Test out-of-range values (should return NaN output).
    df = np.array([-1., 0., 3., 3., 3., 3.], dtype=dtype)
    p = np.array([0.4, 0.4, -1., 0., 1., 2.], dtype=dtype)

    result = self.evaluate(student_t.stdtrit(df, p))
    self.assertDTypeEqual(result, dtype)
    self.assertAllNan(result)

  @test_util.numpy_disable_gradient_test
  def testStdtritGradient(self):
    space_df = np.logspace(np.log10(0.5), 8., num=15).tolist()
    space_p = np.linspace(0.01, 0.99, num=15).tolist()
    df, p = [
        tf.constant(z, dtype='float64')
        for z in zip(*list(itertools.product(space_df, space_p)))]

    # Wrap in tf.function for faster computations.
    stdtrit = tf.function(student_t.stdtrit, autograph=False)

    err = self.compute_max_gradient_error(
        lambda z: stdtrit(z, p), [df], delta=1e-6)
    self.assertLess(err, 5e-6)

    err = self.compute_max_gradient_error(
        lambda z: stdtrit(df, z), [p], delta=1e-7)
    self.assertLess(err, 1e-4)

  @parameterized.parameters(np.float32, np.float64)
  @test_util.numpy_disable_gradient_test
  def testStdtritGradientFinite(self, dtype):
    eps = np.finfo(dtype).eps
    log_df_max = 4. if dtype == np.float32 else 8.

    space_df = np.logspace(np.log10(0.5), log_df_max, num=15).tolist()
    # An odd num > 1 ensures that 0.5 is in np.linspace(eps, 1. - eps, num).
    space_p = np.linspace(eps, 1. - eps, num=13).tolist()
    space_p += [0.5 - eps, 0.5 + eps]
    df, p = [
        tf.constant(z, dtype=dtype)
        for z in zip(*list(itertools.product(space_df, space_p)))]

    # Wrap in tf.function for faster computations.
    @tf.function(autograph=False)
    def stdtrit_partials(df, p):
      return gradient.value_and_gradient(student_t.stdtrit, [df, p])[1]

    partial_df, partial_p = self.evaluate(stdtrit_partials(df, p))

    self.assertDTypeEqual(partial_df, dtype)
    self.assertDTypeEqual(partial_p, dtype)
    self.assertAllFinite([partial_df, partial_p])

  @parameterized.parameters(np.float32, np.float64)
  @test_util.numpy_disable_gradient_test
  def testStdtritGradientBounds(self, dtype):
    # Test out-of-range values (should return NaN output).
    df = tf.constant([-1., 0., 3., 3., 3., 3.], dtype=dtype)
    p = tf.constant([0.4, 0.4, -1., 0., 1., 2.], dtype=dtype)

    partial_df, partial_p = self.evaluate(
        gradient.value_and_gradient(student_t.stdtrit, [df, p])[1])

    self.assertDTypeEqual(partial_df, dtype)
    self.assertDTypeEqual(partial_p, dtype)
    self.assertAllNan([partial_df, partial_p])

  @test_util.numpy_disable_gradient_test
  def testStdtritGradientBroadcast(self):
    df = np.ones([3, 2], dtype=np.float32)
    p = np.full([4, 1, 1], fill_value=0.5, dtype=np.float32)

    def simple_binary_operator(df, p):
      return df + p

    simple_partials = gradient.value_and_gradient(
        simple_binary_operator, [df, p])[1]
    stdtrit_partials = gradient.value_and_gradient(
        student_t.stdtrit, [df, p])[1]
    df_partials, p_partials = zip([*simple_partials], [*stdtrit_partials])
    self.assertShapeEqual(df_partials[0], df_partials[1])
    self.assertShapeEqual(p_partials[0], p_partials[1])

  @parameterized.parameters(np.float32, np.float64)
  @test_util.numpy_disable_gradient_test
  def testStdtritSecondDerivativeFinite(self, dtype):
    eps = np.finfo(dtype).eps
    # Avoid small values for df: the gradients can veer off to infinity when p
    # is close to eps or (1 - eps).
    log_df_min = 0. if dtype == np.float32 else np.log10(0.5)

    space_df = np.logspace(log_df_min, 4., num=7).tolist()
    # An odd num > 1 ensures that 0.5 is in np.linspace(eps, 1. - eps, num).
    space_p = np.linspace(eps, 1. - eps, num=5).tolist()
    space_p += [0.5 - eps, 0.5 + eps]

    df, p = [
        tf.constant(z, dtype=dtype)
        for z in zip(*list(itertools.product(space_df, space_p)))]

    def stdtrit_partials(df, p):
      return gradient.value_and_gradient(student_t.stdtrit, [df, p])[1]

    # Wrap in tf.function for faster computations.
    @tf.function(autograph=False)
    def stdtrit_partials_of_partials(df, p):
      return gradient.value_and_gradient(stdtrit_partials, [df, p])[1]

    partials_of_partials = stdtrit_partials_of_partials(df, p)
    self.assertAllFinite(self.evaluate(partials_of_partials))


if __name__ == '__main__':
  test_util.main()
