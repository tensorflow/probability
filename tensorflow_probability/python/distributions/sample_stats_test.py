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
"""Tests for Sample Stats Ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from tensorflow.python.ops import spectral_ops_test_util

tfe = tf.contrib.eager
tfd = tfp.distributions
rng = np.random.RandomState(0)


@tfe.run_all_tests_in_graph_and_eager_modes
class _AutoCorrelationTest(object):

  @property
  def use_static_shape(self):
    raise NotImplementedError("Subclass failed to implement `use_static_shape`")

  @property
  def dtype(self):
    raise NotImplementedError("Subclass failed to implement `dtype`.")

  def test_constant_sequence_axis_0_max_lags_none_center_false(self):
    x_ = np.array([[0., 0., 0.],
                   [1., 1., 1.]]).astype(self.dtype)
    x_ph = tf.placeholder_with_default(
        input=x_, shape=x_.shape if self.use_static_shape else None)
    with spectral_ops_test_util.fft_kernel_label_map():
      # Setting normalize = True means we divide by zero.
      auto_corr = tfd.auto_correlation(
          x_ph, axis=1, center=False, normalize=False)
      if self.use_static_shape:
        self.assertEqual((2, 3), auto_corr.shape)
      auto_corr_ = self.evaluate(auto_corr)
      self.assertAllClose([[0., 0., 0.], [1., 1., 1.]], auto_corr_)

  def test_constant_sequence_axis_0_max_lags_none_center_true(self):
    x_ = np.array([[0., 0., 0.],
                   [1., 1., 1.]]).astype(self.dtype)
    x_ph = tf.placeholder_with_default(
        input=x_, shape=x_.shape if self.use_static_shape else None)
    with spectral_ops_test_util.fft_kernel_label_map():
      # Setting normalize = True means we divide by zero.
      auto_corr = tfd.auto_correlation(
          x_ph, axis=1, normalize=False, center=True)
      if self.use_static_shape:
        self.assertEqual((2, 3), auto_corr.shape)
      auto_corr_ = self.evaluate(auto_corr)
      self.assertAllClose([[0., 0., 0.], [0., 0., 0.]], auto_corr_)

  def check_results_versus_brute_force(
      self, x, axis, max_lags, center, normalize):
    """Compute auto-correlation by brute force, then compare to tf result."""
    # Brute for auto-corr -- avoiding fft and transpositions.
    axis_len = x.shape[axis]
    if max_lags is None:
      max_lags = axis_len - 1
    else:
      max_lags = min(axis_len - 1, max_lags)
    auto_corr_at_lag = []
    if center:
      x -= x.mean(axis=axis, keepdims=True)
    for m in range(max_lags + 1):
      auto_corr_at_lag.append((
          np.take(x, indices=range(0, axis_len - m), axis=axis) *
          np.conj(np.take(x, indices=range(m, axis_len), axis=axis))
      ).mean(axis=axis, keepdims=True))
    rxx = np.concatenate(auto_corr_at_lag, axis=axis)
    if normalize:
      rxx /= np.take(rxx, [0], axis=axis)

    x_ph = tf.placeholder_with_default(
        x, shape=x.shape if self.use_static_shape else None)
    with spectral_ops_test_util.fft_kernel_label_map():
      auto_corr = tfd.auto_correlation(
          x_ph,
          axis=axis,
          max_lags=max_lags,
          center=center,
          normalize=normalize)
      if self.use_static_shape:
        output_shape = list(x.shape)
        output_shape[axis] = max_lags + 1
        self.assertAllEqual(output_shape, auto_corr.shape)
      self.assertAllClose(rxx, self.evaluate(auto_corr), rtol=1e-5, atol=1e-5)

  def test_axis_n1_center_false_max_lags_none(self):
    x = rng.randn(2, 3, 4).astype(self.dtype)
    if self.dtype in [np.complex64]:
      x = 1j * rng.randn(2, 3, 4).astype(self.dtype)
    self.check_results_versus_brute_force(
        x, axis=-1, max_lags=None, center=False, normalize=False)

  def test_axis_n2_center_false_max_lags_none(self):
    x = rng.randn(3, 4, 5).astype(self.dtype)
    if self.dtype in [np.complex64]:
      x = 1j * rng.randn(3, 4, 5).astype(self.dtype)
    self.check_results_versus_brute_force(
        x, axis=-2, max_lags=None, center=False, normalize=False)

  def test_axis_n1_center_false_max_lags_none_normalize_true(self):
    x = rng.randn(2, 3, 4).astype(self.dtype)
    if self.dtype in [np.complex64]:
      x = 1j * rng.randn(2, 3, 4).astype(self.dtype)
    self.check_results_versus_brute_force(
        x, axis=-1, max_lags=None, center=False, normalize=True)

  def test_axis_n2_center_false_max_lags_none_normalize_true(self):
    x = rng.randn(3, 4, 5).astype(self.dtype)
    if self.dtype in [np.complex64]:
      x = 1j * rng.randn(3, 4, 5).astype(self.dtype)
    self.check_results_versus_brute_force(
        x, axis=-2, max_lags=None, center=False, normalize=True)

  def test_axis_0_center_true_max_lags_none(self):
    x = rng.randn(3, 4, 5).astype(self.dtype)
    if self.dtype in [np.complex64]:
      x = 1j * rng.randn(3, 4, 5).astype(self.dtype)
    self.check_results_versus_brute_force(
        x, axis=0, max_lags=None, center=True, normalize=False)

  def test_axis_2_center_true_max_lags_1(self):
    x = rng.randn(3, 4, 5).astype(self.dtype)
    if self.dtype in [np.complex64]:
      x = 1j * rng.randn(3, 4, 5).astype(self.dtype)
    self.check_results_versus_brute_force(
        x, axis=2, max_lags=1, center=True, normalize=False)

  def test_axis_2_center_true_max_lags_100(self):
    # There are less than 100 elements in axis 2, so expect we get back an array
    # the same size as x, despite having asked for 100 lags.
    x = rng.randn(3, 4, 5).astype(self.dtype)
    if self.dtype in [np.complex64]:
      x = 1j * rng.randn(3, 4, 5).astype(self.dtype)
    self.check_results_versus_brute_force(
        x, axis=2, max_lags=100, center=True, normalize=False)

  def test_long_orthonormal_sequence_has_corr_length_0(self):
    l = 10000
    x = rng.randn(l).astype(self.dtype)
    x_ph = tf.placeholder_with_default(
        x, shape=(l,) if self.use_static_shape else None)
    with spectral_ops_test_util.fft_kernel_label_map():
      rxx = tfd.auto_correlation(
          x_ph, max_lags=l // 2, center=True, normalize=False)
      if self.use_static_shape:
        self.assertAllEqual((l // 2 + 1,), rxx.shape)
      rxx_ = self.evaluate(rxx)
      # OSS CPU FFT has some accuracy issues is not the most accurate.
      # So this tolerance is a bit bad.
      self.assertAllClose(1., rxx_[0], rtol=0.05)
      # The maximal error in the rest of the sequence is not great.
      self.assertAllClose(np.zeros(l // 2), rxx_[1:], atol=0.1)
      # The mean error in the rest is ok, actually 0.008 when I tested it.
      self.assertLess(np.abs(rxx_[1:]).mean(), 0.02)

  def test_step_function_sequence(self):
    if tf.executing_eagerly() and not self.use_static_shape:
      # TODO(b/122840816): Modify this test so that it runs in eager mode with
      # dynamic shapes, or document that this is the intended behavior.
      return

    # x jumps to new random value every 10 steps.  So correlation length = 10.
    x = (rng.randint(-10, 10, size=(1000, 1))
         * np.ones((1, 10))).ravel().astype(self.dtype)
    x_ph = tf.placeholder_with_default(
        x, shape=(1000 * 10,) if self.use_static_shape else None)
    with spectral_ops_test_util.fft_kernel_label_map():
      rxx = tfd.auto_correlation(
          x_ph, max_lags=1000 * 10 // 2, center=True, normalize=False)
      if self.use_static_shape:
        self.assertAllEqual((1000 * 10 // 2 + 1,), rxx.shape)
      rxx_ = self.evaluate(rxx)
      rxx_ /= rxx_[0]
      # Expect positive correlation for the first 10 lags, then significantly
      # smaller negative.
      self.assertGreater(rxx_[:10].min(), 0)
      self.assertGreater(rxx_[9], 5 * rxx_[10:20].mean())
      # RXX should be decreasing for the first 10 lags.
      diff = np.diff(rxx_)
      self.assertLess(diff[:10].max(), 0)

  def test_normalization(self):
    l = 10000
    x = 3 * rng.randn(l).astype(self.dtype)
    x_ph = tf.placeholder_with_default(
        x, shape=(l,) if self.use_static_shape else None)
    with spectral_ops_test_util.fft_kernel_label_map():
      rxx = tfd.auto_correlation(
          x_ph, max_lags=l // 2, center=True, normalize=True)
      if self.use_static_shape:
        self.assertAllEqual((l // 2 + 1,), rxx.shape)
      rxx_ = self.evaluate(rxx)
      # Note that RXX[0] = 1, despite the fact that E[X^2] = 9, and this is
      # due to normalize=True.
      # OSS CPU FFT has some accuracy issues is not the most accurate.
      # So this tolerance is a bit bad.
      self.assertAllClose(1., rxx_[0], rtol=0.05)
      # The maximal error in the rest of the sequence is not great.
      self.assertAllClose(np.zeros(l // 2), rxx_[1:], atol=0.1)
      # The mean error in the rest is ok, actually 0.008 when I tested it.
      self.assertLess(np.abs(rxx_[1:]).mean(), 0.02)


@tfe.run_all_tests_in_graph_and_eager_modes
class AutoCorrelationTestStaticShapeFloat32(tf.test.TestCase,
                                            _AutoCorrelationTest):

  @property
  def dtype(self):
    return np.float32

  @property
  def use_static_shape(self):
    return True


@tfe.run_all_tests_in_graph_and_eager_modes
class AutoCorrelationTestStaticShapeComplex64(tf.test.TestCase,
                                              _AutoCorrelationTest):

  @property
  def dtype(self):
    return np.complex64

  @property
  def use_static_shape(self):
    return True


@tfe.run_all_tests_in_graph_and_eager_modes
class AutoCorrelationTestDynamicShapeFloat32(tf.test.TestCase,
                                             _AutoCorrelationTest):

  @property
  def dtype(self):
    return np.float32

  @property
  def use_static_shape(self):
    return False


@tfe.run_all_tests_in_graph_and_eager_modes
class PercentileTestWithLowerInterpolation(tf.test.TestCase):

  _interpolation = "lower"

  def test_one_dim_odd_input(self):
    x = [1., 5., 3., 2., 4.]
    for q in [0, 10, 25, 49.9, 50, 50.01, 90, 95, 100]:
      expected_percentile = np.percentile(
          x, q=q, interpolation=self._interpolation, axis=0)
      pct = tfd.percentile(x, q=q, interpolation=self._interpolation, axis=[0])
      self.assertAllEqual((), pct.shape)
      self.assertAllClose(expected_percentile, self.evaluate(pct))

  def test_one_dim_odd_input_vector_q(self):
    x = [1., 5., 3., 2., 4.]
    q = np.array([0, 10, 25, 49.9, 50, 50.01, 90, 95, 100])
    expected_percentile = np.percentile(
        x, q=q, interpolation=self._interpolation, axis=0)
    pct = tfd.percentile(x, q=q, interpolation=self._interpolation, axis=[0])
    self.assertAllEqual(q.shape, pct.shape)
    self.assertAllClose(expected_percentile, self.evaluate(pct))

  def test_one_dim_even_input(self):
    x = [1., 5., 3., 2., 4., 5.]
    for q in [0, 10, 25, 49.9, 50, 50.01, 90, 95, 100]:
      expected_percentile = np.percentile(
          x, q=q, interpolation=self._interpolation)
      pct = tfd.percentile(x, q=q, interpolation=self._interpolation)
      self.assertAllEqual((), pct.shape)
      self.assertAllClose(expected_percentile, self.evaluate(pct))

  def test_two_dim_odd_input_axis_0(self):
    x = np.array([[-1., 50., -3.5, 2., -1], [0., 0., 3., 2., 4.]]).T
    for q in [0, 10, 25, 49.9, 50, 50.01, 90, 95, 100]:
      expected_percentile = np.percentile(
          x, q=q, interpolation=self._interpolation, axis=0)
      # Get dim 1 with negative and positive indices.
      pct_neg_index = tfd.percentile(
          x, q=q, interpolation=self._interpolation, axis=[0])
      pct_pos_index = tfd.percentile(
          x, q=q, interpolation=self._interpolation, axis=[0])
      self.assertAllEqual((2,), pct_neg_index.shape)
      self.assertAllEqual((2,), pct_pos_index.shape)
      self.assertAllClose(expected_percentile, self.evaluate(pct_neg_index))
      self.assertAllClose(expected_percentile, self.evaluate(pct_pos_index))

  def test_two_dim_even_axis_0(self):
    x = np.array([[1., 2., 4., 50.], [1., 2., -4., 5.]]).T
    for q in [0, 10, 25, 49.9, 50, 50.01, 90, 95, 100]:
      expected_percentile = np.percentile(
          x, q=q, interpolation=self._interpolation, axis=0)
      pct = tfd.percentile(x, q=q, interpolation=self._interpolation, axis=[0])
      self.assertAllEqual((2,), pct.shape)
      self.assertAllClose(expected_percentile, self.evaluate(pct))

  def test_two_dim_even_input_and_keep_dims_true(self):
    x = np.array([[1., 2., 4., 50.], [1., 2., -4., 5.]]).T
    for q in [0, 10, 25, 49.9, 50, 50.01, 90, 95, 100]:
      expected_percentile = np.percentile(
          x, q=q, interpolation=self._interpolation, keepdims=True, axis=0)
      pct = tfd.percentile(
          x, q=q, interpolation=self._interpolation, keep_dims=True, axis=[0])
      self.assertAllEqual((1, 2), pct.shape)
      self.assertAllClose(expected_percentile, self.evaluate(pct))

  def test_four_dimensional_input(self):
    x = rng.rand(2, 3, 4, 5)
    for axis in [None, 0, 1, -2, (0,), (-1,), (-1, 1), (3, 1), (-3, 0)]:
      expected_percentile = np.percentile(
          x, q=0.77, interpolation=self._interpolation, axis=axis)
      pct = tfd.percentile(
          x, q=0.77, interpolation=self._interpolation, axis=axis)
      self.assertAllEqual(expected_percentile.shape, pct.shape)
      self.assertAllClose(expected_percentile, self.evaluate(pct))

  def test_four_dimensional_input_q_vector(self):
    x = rng.rand(3, 4, 5, 6)
    q = [0.25, 0.75]
    for axis in [None, 0, (-1, 1)]:
      expected_percentile = np.percentile(
          x, q=q, interpolation=self._interpolation, axis=axis)
      pct = tfd.percentile(
          x, q=q, interpolation=self._interpolation, axis=axis)
      self.assertAllEqual(expected_percentile.shape, pct.shape)
      self.assertAllClose(expected_percentile, self.evaluate(pct))

  def test_four_dimensional_input_q_vector_and_keepdims(self):
    x = rng.rand(3, 4, 5, 6)
    q = [0.25, 0.75]
    for axis in [None, 0, (-1, 1)]:
      expected_percentile = np.percentile(
          x, q=q, interpolation=self._interpolation, axis=axis, keepdims=True)
      pct = tfd.percentile(
          x, q=q, interpolation=self._interpolation, axis=axis, keep_dims=True)
      self.assertAllEqual(expected_percentile.shape, pct.shape)
      self.assertAllClose(expected_percentile, self.evaluate(pct))

  def test_four_dimensional_input_and_keepdims(self):
    x = rng.rand(2, 3, 4, 5)
    for axis in [None, 0, 1, -2, (0,), (-1,), (-1, 1), (3, 1), (-3, 0)]:
      expected_percentile = np.percentile(
          x,
          q=0.77,
          interpolation=self._interpolation,
          axis=axis,
          keepdims=True)
      pct = tfd.percentile(
          x,
          q=0.77,
          interpolation=self._interpolation,
          axis=axis,
          keep_dims=True)
      self.assertAllEqual(expected_percentile.shape, pct.shape)
      self.assertAllClose(expected_percentile, self.evaluate(pct))

  def test_four_dimensional_input_x_static_ndims_but_dynamic_sizes(self):
    x = rng.rand(2, 3, 4, 5)
    x_ph = tf.placeholder_with_default(input=x, shape=[None, None, None, None])
    for axis in [None, 0, 1, -2, (0,), (-1,), (-1, 1), (3, 1), (-3, 0)]:
      expected_percentile = np.percentile(
          x, q=0.77, interpolation=self._interpolation, axis=axis)
      pct = tfd.percentile(
          x_ph, q=0.77, interpolation=self._interpolation, axis=axis)
      self.assertAllClose(expected_percentile, self.evaluate(pct))

  def test_four_dimensional_input_and_keepdims_x_static_ndims_dynamic_sz(self):
    x = rng.rand(2, 3, 4, 5)
    x_ph = tf.placeholder_with_default(input=x, shape=[None, None, None, None])
    for axis in [None, 0, 1, -2, (0,), (-1,), (-1, 1), (3, 1), (-3, 0)]:
      expected_percentile = np.percentile(
          x,
          q=0.77,
          interpolation=self._interpolation,
          axis=axis,
          keepdims=True)
      pct = tfd.percentile(
          x_ph,
          q=0.77,
          interpolation=self._interpolation,
          axis=axis,
          keep_dims=True)
      self.assertAllClose(expected_percentile, self.evaluate(pct))

  def test_with_integer_dtype(self):
    x = [1, 5, 3, 2, 4]
    for q in [0, 10, 25, 49.9, 50, 50.01, 90, 95, 100]:
      expected_percentile = np.percentile(
          x, q=q, interpolation=self._interpolation)
      pct = tfd.percentile(x, q=q, interpolation=self._interpolation)
      self.assertEqual(tf.int32, pct.dtype)
      self.assertAllEqual((), pct.shape)
      self.assertAllClose(expected_percentile, self.evaluate(pct))


@tfe.run_all_tests_in_graph_and_eager_modes
class PercentileTestWithHigherInterpolation(
    PercentileTestWithLowerInterpolation):

  _interpolation = "higher"


@tfe.run_all_tests_in_graph_and_eager_modes
class PercentileTestWithNearestInterpolation(tf.test.TestCase):
  """Test separately because np.round and tf.round make different choices."""

  _interpolation = "nearest"

  def test_one_dim_odd_input(self):
    x = [1., 5., 3., 2., 4.]
    for q in [0, 10.1, 25.1, 49.9, 50.1, 50.01, 89, 100]:
      expected_percentile = np.percentile(
          x, q=q, interpolation=self._interpolation)
      pct = tfd.percentile(x, q=q, interpolation=self._interpolation)
      self.assertAllEqual((), pct.shape)
      self.assertAllClose(expected_percentile, self.evaluate(pct))

  def test_one_dim_even_input(self):
    x = [1., 5., 3., 2., 4., 5.]
    for q in [0, 10.1, 25.1, 49.9, 50.1, 50.01, 89, 100]:
      expected_percentile = np.percentile(
          x, q=q, interpolation=self._interpolation)
      pct = tfd.percentile(x, q=q, interpolation=self._interpolation)
      self.assertAllEqual((), pct.shape)
      self.assertAllClose(expected_percentile, self.evaluate(pct))

  def test_invalid_interpolation_raises(self):
    x = [1., 5., 3., 2., 4.]
    with self.assertRaisesRegexp(ValueError, "interpolation"):
      tfd.percentile(x, q=0.5, interpolation="bad")

  def test_2d_q_raises_static(self):
    x = [1., 5., 3., 2., 4.]
    with self.assertRaisesRegexp(ValueError, "Expected.*ndims"):
      tfd.percentile(x, q=[[0.5]])

  def test_2d_q_raises_dynamic(self):
    x = [1., 5., 3., 2., 4.]
    q_ph = tf.placeholder_with_default(input=[[0.5]], shape=None)
    if tf.executing_eagerly():
      with self.assertRaisesRegexp(ValueError, "Expected.*ndims"):
        pct = tfd.percentile(x, q=q_ph, validate_args=True)
    else:
      pct = tfd.percentile(x, q=q_ph, validate_args=True)
      with self.assertRaisesOpError("rank"):
        self.evaluate(pct)

  def test_finds_max_of_long_array(self):
    # d - 1 == d in float32 and d = 3e7.
    # So this test only passes if we use double for the percentile indices.
    # If float is used, it fails with InvalidArgumentError about an index out of
    # bounds.
    x = tf.linspace(0., 3e7, num=int(3e7))
    minval = tfd.percentile(x, q=0, validate_args=True)
    self.assertAllEqual(0, self.evaluate(minval))


if __name__ == "__main__":
  tf.test.main()
