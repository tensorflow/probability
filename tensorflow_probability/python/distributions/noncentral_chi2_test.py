# Copyright 2022 The TensorFlow Probability Authors.
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
"""Tests for noncentral_chi2."""

from absl.testing import parameterized

import numpy as np
from scipy import stats
import tensorflow as tf

from tensorflow_probability.python.distributions import noncentral_chi2
from tensorflow_probability.python.distributions.internal import statistical_testing as st
from tensorflow_probability.python.internal import test_util

JAX_MODE = False


@test_util.test_all_tf_execution_regimes
class NoncentralChi2Test(test_util.TestCase):

  def test_noncentral_chi2_log_pdf(self):
    batch_size = 7
    df_v = 2.
    nc_v = 3.
    df = tf.constant([df_v] * batch_size, dtype=tf.float64)
    nc = tf.constant([nc_v] * batch_size, dtype=tf.float64)

    x = np.array([2.5, 2.5, 4., 0.1, 1., 2., 10.], dtype=np.float64)

    nc_chi2 = noncentral_chi2.NoncentralChi2(df, nc)

    expected_log_pdf = stats.ncx2.logpdf(x, df=df_v, nc=nc_v)

    log_pdf = nc_chi2.log_prob(x)

    self.assertEqual(log_pdf.shape, (7,))
    self.assertAllClose(self.evaluate(log_pdf), expected_log_pdf)

    pdf = nc_chi2.prob(x)
    self.assertEqual(pdf.shape, (7,))
    self.assertAllClose(self.evaluate(pdf), np.exp(expected_log_pdf))

  def test_log_pdf_asserts_on_invalid_sample(self):
    d = noncentral_chi2.NoncentralChi2(
        df=13.37, noncentrality=42., validate_args=True)

    with self.assertRaisesOpError('Sample must be non-negative.'):
      self.evaluate(d.log_prob([14.2, -5.3]))

  def test_pdf_on_boundary(self):
    d = noncentral_chi2.NoncentralChi2(
        df=[2., 4., 1.], noncentrality=[0., 1., 2.], validate_args=True)
    log_prob_boundary = self.evaluate(d.log_prob(0.))

    self.assertAllFinite(log_prob_boundary[0])
    self.assertAllNegativeInf(log_prob_boundary[1])
    self.assertAllPositiveInf(log_prob_boundary[2])

    prob_boundary = self.evaluate(d.prob(0.))
    self.assertAllFinite(prob_boundary[:1])
    self.assertAllPositiveInf(prob_boundary[2])

  def test_noncentral_chi2_log_cdf(self):
    batch_size = 7
    df_v = 2.
    nc_v = 3.
    df = tf.constant([df_v] * batch_size, dtype=tf.float64)
    nc = tf.constant([nc_v] * batch_size, dtype=tf.float64)

    x = np.array([2.5, 2.5, 4., 0.1, 1., 2., 10.], dtype=np.float64)

    nc_chi2 = noncentral_chi2.NoncentralChi2(df, nc)

    expected_log_cdf = stats.ncx2.logcdf(x, df=df_v, nc=nc_v)

    log_cdf_fn = tf.function(nc_chi2.log_cdf, autograph=False)
    log_cdf = log_cdf_fn(x)

    self.assertEqual(log_cdf.shape, (7,))
    self.assertAllClose(self.evaluate(log_cdf), expected_log_cdf)

    cdf_fn = tf.function(nc_chi2.cdf, autograph=False)
    cdf = cdf_fn(x)

    self.assertEqual(cdf.shape, (7,))
    self.assertAllClose(self.evaluate(cdf), np.exp(expected_log_cdf))

  def test_noncentral_chi2_log_cdf_hard(self):
    dfs = np.array([1., 1., 0.5, 3.7, 2., 10., 1000., 1e7, 1e2, 1e2])
    ncs = np.array([0., 10., 1., 2., 1000., 0., 100., 1e3, 1e6, 1e1])

    df = tf.convert_to_tensor(dfs, dtype=tf.float64)
    nc = tf.convert_to_tensor(ncs, dtype=tf.float64)

    x = np.array([0.1, 12., 2.5, 5.5, 1002., 20., 1100., 1e7, 1e6 + 1e2, 1e3],
                 dtype=np.float64)

    nc_chi2 = noncentral_chi2.NoncentralChi2(df, nc)

    expected_log_cdf = stats.ncx2.logcdf(x, df=dfs, nc=ncs)

    log_cdf_fn = tf.function(nc_chi2.log_cdf, autograph=False)
    log_cdf = log_cdf_fn(x)

    self.assertEqual(log_cdf.shape, (10,))
    self.assertAllClose(self.evaluate(log_cdf), expected_log_cdf)

    cdf_fn = tf.function(nc_chi2.cdf, autograph=False)
    cdf = cdf_fn(x)

    self.assertEqual(cdf.shape, (10,))
    self.assertAllClose(self.evaluate(cdf), np.exp(expected_log_cdf))

  def test_noncentral_chi2_mean(self):
    df_v = np.array([1., 3., 5.], dtype=np.float64)
    nc_v = np.array([0., 2., 4.], dtype=np.float64)
    expected_mean = stats.ncx2.mean(df=df_v, nc=nc_v)
    nc_chi2 = noncentral_chi2.NoncentralChi2(
        df=df_v, noncentrality=nc_v, validate_args=True)

    self.assertEqual(nc_chi2.mean().shape, (3,))
    self.assertAllClose(nc_chi2.mean(), expected_mean)

  def test_noncentral_chi2_variance(self):
    df_v = np.array([1., 3., 5.], dtype=np.float64)
    nc_v = np.array([0., 2., 4.], dtype=np.float64)
    expected_variances = stats.ncx2.var(df=df_v, nc=nc_v)
    nc_chi2 = noncentral_chi2.NoncentralChi2(
        df=df_v, noncentrality=nc_v, validate_args=True)

    self.assertEqual(nc_chi2.variance().shape, (3,))
    self.assertAllClose(nc_chi2.variance(), expected_variances)

  @test_util.tf_tape_safety_test
  def test_gradient_through_params(self):
    df = tf.Variable([19.43, 1.], dtype=tf.float64)
    nc = tf.Variable([5.12, 0.], dtype=tf.float64)

    d = noncentral_chi2.NoncentralChi2(
        df=df, noncentrality=nc, validate_args=True)

    with tf.GradientTape() as tape:
      loss = -d.log_prob([1., 2.])

    grad = tape.gradient(loss, d.trainable_variables)

    self.assertLen(grad, 2)
    self.assertAllNotNone(grad)

  @test_util.tf_tape_safety_test
  def test_gradient_through_non_variable_params(self):
    df = tf.convert_to_tensor([19.43, 1.], dtype=tf.float64)
    noncentrality = tf.convert_to_tensor([5.12, 0.], dtype=tf.float64)

    d = noncentral_chi2.NoncentralChi2(
        df=df, noncentrality=noncentrality, validate_args=True)

    with tf.GradientTape() as tape:
      tape.watch([d.df, d.noncentrality])
      loss = -d.log_prob([1., 2.])

    grad = tape.gradient(loss, [d.df, d.noncentrality])
    grad = self.evaluate(grad)

    self.assertLen(grad, 2)
    self.assertAllNotNone(grad)
    self.assertAllNotNan(grad)

  @test_util.numpy_disable_gradient_test
  @parameterized.named_parameters(('float32', np.float32, 5e-4),
                                  ('float64', np.float64, 2e-4))
  def test_cdf_gradient_with_respect_to_x(self, dtype, max_err):
    df = tf.constant([np.e, 0.01, 0.1, 0.5, 1., 1.5, 3.1, 4., 10., 20., 100.3],
                     dtype=dtype)[..., tf.newaxis]
    nc = tf.constant([0., 0.01, 0.1, 0.5, 1., 1.5, 3.1, 4., 10., 20., 100.2],
                     dtype=dtype)[..., tf.newaxis]

    xs = tf.constant([0.1, 0.5, 0.9, 1., 2.9, 5., 7., 8.1, 21., 200.],
                     dtype=dtype)

    cdf_fn = tf.function(
        noncentral_chi2.NoncentralChi2(df, nc).cdf, autograph=False)

    err = self.compute_max_gradient_error(cdf_fn, [xs])
    self.assertLess(err, max_err)

  @test_util.numpy_disable_gradient_test
  @parameterized.named_parameters(('float32', np.float32, 7e-3),
                                  ('float64', np.float64, 2e-4))
  def test_cdf_gradient_with_respect_to_noncentrality(self, dtype, max_err):
    df = tf.constant([np.e, 0.01, 0.5, 1., 1.5, 3.1, 4., 10., 20., 100.3],
                     dtype=dtype)[..., tf.newaxis]
    nc = tf.constant([0.01, 0.1, 0.5, 1., 1.5, 3.1, 4., 10., 20., 100.2],
                     dtype=dtype)[..., tf.newaxis]

    xs = tf.constant([0., 0.1, 0.5, 0.9, 1., 2.9, 5., 4., 7., 8.1, 21., 200.],
                     dtype=dtype)

    cdf_fn = tf.function(
        lambda nc: noncentral_chi2.NoncentralChi2(df, nc).cdf(xs),
        autograph=False)

    err = self.compute_max_gradient_error(cdf_fn, [nc])
    self.assertLess(err, max_err)

  def test_asserts_positive_df(self):
    df = tf.Variable([1., 2., -3.])
    noncentrality = tf.Variable([0., 2., 1.])

    with self.assertRaisesOpError('Argument `df` must be positive.'):
      d = noncentral_chi2.NoncentralChi2(
          df=df, noncentrality=noncentrality, validate_args=True)
      self.evaluate([v.initializer for v in d.variables])
      self.evaluate(d.sample(seed=test_util.test_seed()))

  def test_asserts_nonnegative_nc(self):
    df = tf.Variable([1., 2., 3.])
    noncentrality = tf.Variable([0., 2., -1.])

    with self.assertRaisesOpError(
        'Argument `noncentrality` must be non-negative.'):
      d = noncentral_chi2.NoncentralChi2(
          df=df, noncentrality=noncentrality, validate_args=True)
      self.evaluate([v.initializer for v in d.variables])
      self.evaluate(d.sample(seed=test_util.test_seed()))

  def test_approx_quantile(self):
    batch_size = 6
    df = np.linspace(1., 20., batch_size).astype(np.float64)[..., np.newaxis]
    nc = np.linspace(0., 20., batch_size).astype(np.float64)[..., np.newaxis]
    x = np.linspace(0., 1., 13).astype(np.float64)
    dist = noncentral_chi2.NoncentralChi2(
        df=df, noncentrality=nc, validate_args=True)

    expected_quantile = stats.ncx2.ppf(x, df, nc)

    quantile = dist.quantile_approx(x)
    self.assertEqual(quantile.shape, (batch_size, 13))
    self.assertAllClose(self.evaluate(quantile), expected_quantile)

  def test_approx_quantile_close_to_boundary(self):
    batch_size = 6
    df = np.linspace(0.1, 20., batch_size).astype(np.float64)[..., np.newaxis]
    nc = np.linspace(0., 20., batch_size).astype(np.float64)[..., np.newaxis]
    x = np.linspace(0., 0.01, 10).astype(np.float64)
    dist = noncentral_chi2.NoncentralChi2(
        df=df, noncentrality=nc, validate_args=True)

    expected_quantile = stats.ncx2.ppf(x, df, nc)

    quantile = dist.quantile_approx(x)
    self.assertEqual(quantile.shape, (batch_size, 10))
    self.assertAllClose(self.evaluate(quantile), expected_quantile)

  @test_util.numpy_disable_gradient_test
  @parameterized.named_parameters(
      # Gradients are on the order of 1e2 to 1e3, so this corresponds to less
      # than 1% relative error.
      ('float32', np.float32, 5e-1),
      ('float64', np.float64, 2e-2))
  def test_approx_quantile_gradient(self, dtype, max_err):
    if tf.executing_eagerly() and not JAX_MODE:
      self.skipTest('Eager test is too slow.')

    df = tf.constant([2., np.e, 0.01, 0.5, 1., 1.5, 3.1, 4., 10., 20., 100.3],
                     dtype=dtype)[..., tf.newaxis]
    nc = tf.constant([0., 0.01, 0.1, 0.5, 1., 1.5, 3.1, 4., 10., 20., 100.2],
                     dtype=dtype)[..., tf.newaxis]

    ps = tf.constant([0.1, 0.3, 0.5, 0.7, 0.9], dtype=dtype)

    quantile_fn = tf.function(
        noncentral_chi2.NoncentralChi2(df, nc).quantile_approx, autograph=False)

    err = self.compute_max_gradient_error(quantile_fn, [ps])
    self.assertLess(err, max_err)


@test_util.test_graph_and_eager_modes
class NoncentralChi2SamplingTest(test_util.TestCase):

  def test_sample(self):
    if not tf.test.is_gpu_available():
      self.skipTest('no GPU')
    with tf.device('GPU'):
      self.evaluate(
          noncentral_chi2.NoncentralChi2(
              df=2., noncentrality=1.,
              validate_args=True).sample(seed=test_util.test_seed()))

  def test_sample_xla(self):
    self.skip_if_no_xla()
    if not tf.executing_eagerly():
      return  # jit_compile is eager-only

    df = np.random.rand(4, 3).astype(np.float32)
    nc = np.random.rand(4, 3).astype(np.float32)

    nc_chi2 = noncentral_chi2.NoncentralChi2(
        df=df, noncentrality=nc, validate_args=True)

    self.evaluate(
        tf.function(
            lambda: nc_chi2.sample(seed=test_util.test_seed()),
            jit_compile=True)())

  def test_sample_low_df(self):
    df = np.linspace(0.1, 1., 10)
    nc = np.float64(10.)
    num_samples = int(1e5)

    self.assertLess(
        self.evaluate(
            st.min_num_samples_for_dkwm_cdf_test(
                discrepancy=0.04, false_fail_rate=1e-9, false_pass_rate=1e-9)),
        num_samples)

    nc_chi2 = noncentral_chi2.NoncentralChi2(
        df=df, noncentrality=nc, validate_args=True)

    samples = nc_chi2.sample(num_samples, seed=test_util.test_seed())

    self.evaluate(
        st.assert_true_cdf_equal_by_dkwm(
            samples, nc_chi2.cdf, false_fail_rate=1e-9))

    self.assertAllMeansClose(
        self.evaluate(samples), nc_chi2.mean(), axis=0, rtol=0.03)

    self.assertAllClose(
        self.evaluate(tf.math.reduce_variance(samples, axis=0)),
        nc_chi2.variance(),
        rtol=0.05)

  def test_sample_high_df(self):
    df = np.linspace(10., 20., 10)
    nc = np.float64(10.)
    num_samples = int(1e5)

    self.assertLess(
        self.evaluate(
            st.min_num_samples_for_dkwm_cdf_test(
                discrepancy=0.04, false_fail_rate=1e-9, false_pass_rate=1e-9)),
        num_samples)

    nc_chi2 = noncentral_chi2.NoncentralChi2(
        df=df, noncentrality=nc, validate_args=True)

    samples = nc_chi2.sample(num_samples, seed=test_util.test_seed())

    self.evaluate(
        st.assert_true_cdf_equal_by_dkwm(
            samples, nc_chi2.cdf, false_fail_rate=1e-9))

    self.assertAllMeansClose(
        self.evaluate(samples), nc_chi2.mean(), axis=0, rtol=0.01)

    self.assertAllClose(
        self.evaluate(tf.math.reduce_variance(samples, axis=0)),
        nc_chi2.variance(),
        rtol=0.05)


if __name__ == '__main__':
  test_util.main()
