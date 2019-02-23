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
"""Tests for quantiles.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from tensorflow.python.framework import test_util  # pylint: disable=g-direct-tensorflow-import
rng = np.random.RandomState(0)


@test_util.run_all_in_graph_and_eager_modes
class FindBinsTest(tf.test.TestCase):

  def test_1d_array_no_extend_lower_and_upper_dtype_int64(self):
    x = [-1., 0., 4., 5., 10., 20.]
    edges = [0., 5., 10.]
    bins = tfp.stats.find_bins(x, edges, dtype=tf.int64)
    self.assertDTypeEqual(bins, np.int64)
    self.assertAllEqual((6,), bins.shape)
    bins_ = self.evaluate(bins)
    self.assertAllEqual([-1, 0, 0, 1, 1, 2], bins_)

  def test_1d_array_extend_lower_and_upper(self):
    x = [-1., 0., 4., 5., 10., 20.]
    edges = [0., 5., 10.]
    bins = tfp.stats.find_bins(
        x, edges, extend_lower_interval=True, extend_upper_interval=True)
    self.assertDTypeEqual(bins, np.float32)
    self.assertAllEqual((6,), bins.shape)
    bins_ = self.evaluate(bins)
    self.assertAllEqual([0, 0, 0, 1, 1, 1], bins_)

  def test_1d_array_no_extend_lower_and_upper(self):
    x = [-1., 0., 4., 5., 10., 20.]
    edges = [0., 5., 10.]
    bins = tfp.stats.find_bins(
        x, edges, extend_lower_interval=False, extend_upper_interval=False)
    self.assertDTypeEqual(bins, np.float32)
    self.assertAllEqual((6,), bins.shape)
    bins_ = self.evaluate(bins)
    self.assertAllEqual([np.nan, 0, 0, 1, 1, np.nan], bins_)

  def test_x_is_2d_array_dtype_int32(self):
    x = [[0., 8., 60.],
         [10., 20., 3.]]
    edges = [[0., 5., 10.],
             [5., 7., 11.],
             [10., 50., 100.]]

    # The intervals for the first column are
    #  [0, 5), [5, 10]
    # and for the second column
    #  [5, 7), [7, 50]
    # and for the third column
    #  [10, 11), [11, 100]
    expected_bins = [[0, 1, 1],
                     [1, 1, -1]]

    bins = tfp.stats.find_bins(x, edges, dtype=tf.int32)
    self.assertDTypeEqual(bins, np.int32)
    self.assertAllEqual((2, 3), bins.shape)
    bins_ = self.evaluate(bins)
    self.assertAllEqual(expected_bins, bins_)

  def test_3d_array_has_expected_bins(self):
    x = np.linspace(0., 1000, 1000, dtype=np.float32).reshape(10, 10, 10)
    edges = [0., 500., 1000.]
    bins = tfp.stats.find_bins(x, edges)
    self.assertAllEqual(x.shape, bins.shape)
    self.assertDTypeEqual(bins, np.float32)
    flat_bins_ = np.ravel(self.evaluate(bins))

    # Demonstrate that x crosses the 500 threshold at index 500
    self.assertLess(x.ravel()[499], 500)
    self.assertGreater(x.ravel()[500], 500)
    self.assertAllEqual(np.zeros((500,)), flat_bins_[:500])
    self.assertAllEqual(np.ones((500,)), flat_bins_[500:])

  def test_large_random_array_has_expected_bin_fractions(self):
    x = rng.rand(100, 99, 98)
    edges = np.linspace(0., 1., 11)  # Deciles
    edges = edges.reshape(11, 1, 1) + np.zeros((99, 98))
    bins = tfp.stats.find_bins(x, edges)

    self.assertAllEqual(x.shape, bins.shape)
    self.assertDTypeEqual(bins, np.float64)
    bins_ = self.evaluate(bins)
    self.assertAllClose((bins_ == 0).mean(), 0.1, rtol=0.05)
    self.assertAllClose((bins_ == 1).mean(), 0.1, rtol=0.05)
    self.assertAllClose((bins_ == 2).mean(), 0.1, rtol=0.05)

    mask = (0.3 <= x) & (x < 0.4)
    self.assertAllEqual(3. * np.ones((mask.sum(),)), bins_[mask])

  def test_large_random_array_has_expected_bin_fractions_with_broadcast(self):
    x = rng.rand(100, 99, 98)
    # rank(edges) < rank(x), so it will broadcast.
    edges = np.linspace(0., 1., 11)  # Deciles
    bins = tfp.stats.find_bins(x, edges)

    self.assertAllEqual(x.shape, bins.shape)
    self.assertDTypeEqual(bins, np.float64)
    bins_ = self.evaluate(bins)
    self.assertAllClose((bins_ == 0).mean(), 0.1, rtol=0.05)
    self.assertAllClose((bins_ == 1).mean(), 0.1, rtol=0.05)
    self.assertAllClose((bins_ == 2).mean(), 0.1, rtol=0.05)

    mask = (0.3 <= x) & (x < 0.4)
    self.assertAllEqual(3. * np.ones((mask.sum(),)), bins_[mask])

  def test_too_few_edges_raises(self):
    x = [1., 2., 3., 4.]
    edges = [2.]
    with self.assertRaisesRegexp(ValueError, '1 or more bin'):
      tfp.stats.find_bins(x, edges)


@test_util.run_all_in_graph_and_eager_modes
class PercentileTestWithLowerInterpolation(tf.test.TestCase):

  _interpolation = 'lower'

  def test_one_dim_odd_input(self):
    x = [1., 5., 3., 2., 4.]
    for q in [0, 10, 25, 49.9, 50, 50.01, 90, 95, 100]:
      expected_percentile = np.percentile(
          x, q=q, interpolation=self._interpolation, axis=0)
      pct = tfp.stats.percentile(
          x, q=q, interpolation=self._interpolation, axis=[0])
      self.assertAllEqual((), pct.shape)
      self.assertAllClose(expected_percentile, self.evaluate(pct))

  def test_one_dim_odd_input_vector_q(self):
    x = [1., 5., 3., 2., 4.]
    q = np.array([0, 10, 25, 49.9, 50, 50.01, 90, 95, 100])
    expected_percentile = np.percentile(
        x, q=q, interpolation=self._interpolation, axis=0)
    pct = tfp.stats.percentile(
        x, q=q, interpolation=self._interpolation, axis=[0])
    self.assertAllEqual(q.shape, pct.shape)
    self.assertAllClose(expected_percentile, self.evaluate(pct))

  def test_one_dim_even_input(self):
    x = [1., 5., 3., 2., 4., 5.]
    for q in [0, 10, 25, 49.9, 50, 50.01, 90, 95, 100]:
      expected_percentile = np.percentile(
          x, q=q, interpolation=self._interpolation)
      pct = tfp.stats.percentile(x, q=q, interpolation=self._interpolation)
      self.assertAllEqual((), pct.shape)
      self.assertAllClose(expected_percentile, self.evaluate(pct))

  def test_two_dim_odd_input_axis_0(self):
    x = np.array([[-1., 50., -3.5, 2., -1], [0., 0., 3., 2., 4.]]).T
    for q in [0, 10, 25, 49.9, 50, 50.01, 90, 95, 100]:
      expected_percentile = np.percentile(
          x, q=q, interpolation=self._interpolation, axis=0)
      # Get dim 1 with negative and positive indices.
      pct_neg_index = tfp.stats.percentile(
          x, q=q, interpolation=self._interpolation, axis=[0])
      pct_pos_index = tfp.stats.percentile(
          x, q=q, interpolation=self._interpolation, axis=[0])
      self.assertAllEqual((2,), pct_neg_index.shape)
      self.assertAllEqual((2,), pct_pos_index.shape)
      self.assertAllClose(expected_percentile, self.evaluate(pct_neg_index))
      self.assertAllClose(expected_percentile, self.evaluate(pct_pos_index))

  def test_simple(self):
    # Simple test that exposed something the other 1-D tests didn't.
    x = np.array([1., 2., 4., 50.])
    q = 10
    expected_percentile = np.percentile(
        x, q=q, interpolation=self._interpolation, axis=0)
    pct = tfp.stats.percentile(
        x, q=q, interpolation=self._interpolation, axis=[0])
    self.assertAllClose(expected_percentile, self.evaluate(pct))

  def test_two_dim_even_axis_0(self):
    x = np.array([[1., 2., 4., 50.], [1., 2., -4., 5.]]).T
    for q in [0, 10, 25, 49.9, 50, 50.01, 90, 95, 100]:
      expected_percentile = np.percentile(
          x, q=q, interpolation=self._interpolation, axis=0)
      pct = tfp.stats.percentile(
          x, q=q, interpolation=self._interpolation, axis=[0])
      self.assertAllEqual((2,), pct.shape)
      self.assertAllClose(expected_percentile, self.evaluate(pct))

  def test_two_dim_even_input_and_keep_dims_true(self):
    x = np.array([[1., 2., 4., 50.], [1., 2., -4., 5.]]).T
    for q in [0, 10, 25, 49.9, 50, 50.01, 90, 95, 100]:
      expected_percentile = np.percentile(
          x, q=q, interpolation=self._interpolation, keepdims=True, axis=0)
      pct = tfp.stats.percentile(
          x, q=q, interpolation=self._interpolation, keep_dims=True, axis=[0])
      self.assertAllEqual((1, 2), pct.shape)
      self.assertAllClose(expected_percentile, self.evaluate(pct))

  def test_four_dimensional_input(self):
    x = rng.rand(2, 3, 4, 5)
    for axis in [None, 0, 1, -2, (0,), (-1,), (-1, 1), (3, 1), (-3, 0)]:
      expected_percentile = np.percentile(
          x, q=0.77, interpolation=self._interpolation, axis=axis)
      pct = tfp.stats.percentile(
          x, q=0.77, interpolation=self._interpolation, axis=axis)
      self.assertAllEqual(expected_percentile.shape, pct.shape)
      self.assertAllClose(expected_percentile, self.evaluate(pct))

  def test_four_dimensional_input_q_vector(self):
    x = rng.rand(3, 4, 5, 6)
    q = [0.25, 0.75]
    for axis in [None, 0, (-1, 1)]:
      expected_percentile = np.percentile(
          x, q=q, interpolation=self._interpolation, axis=axis)
      pct = tfp.stats.percentile(
          x, q=q, interpolation=self._interpolation, axis=axis)
      self.assertAllEqual(expected_percentile.shape, pct.shape)
      self.assertAllClose(expected_percentile, self.evaluate(pct))

  def test_four_dimensional_input_q_vector_and_keepdims(self):
    x = rng.rand(3, 4, 5, 6)
    q = [0.25, 0.75]
    for axis in [None, 0, (-1, 1)]:
      expected_percentile = np.percentile(
          x, q=q, interpolation=self._interpolation, axis=axis, keepdims=True)
      pct = tfp.stats.percentile(
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
      pct = tfp.stats.percentile(
          x,
          q=0.77,
          interpolation=self._interpolation,
          axis=axis,
          keep_dims=True)
      self.assertAllEqual(expected_percentile.shape, pct.shape)
      self.assertAllClose(expected_percentile, self.evaluate(pct))

  def test_four_dimensional_input_x_static_ndims_but_dynamic_sizes(self):
    x = rng.rand(2, 3, 4, 5)
    x_ph = tf.compat.v1.placeholder_with_default(
        input=x, shape=[None, None, None, None])
    for axis in [None, 0, 1, -2, (0,), (-1,), (-1, 1), (3, 1), (-3, 0)]:
      expected_percentile = np.percentile(
          x, q=0.77, interpolation=self._interpolation, axis=axis)
      pct = tfp.stats.percentile(
          x_ph, q=0.77, interpolation=self._interpolation, axis=axis)
      self.assertAllClose(expected_percentile, self.evaluate(pct))

  def test_four_dimensional_input_and_keepdims_x_static_ndims_dynamic_sz(self):
    x = rng.rand(2, 3, 4, 5)
    x_ph = tf.compat.v1.placeholder_with_default(
        input=x, shape=[None, None, None, None])
    for axis in [None, 0, 1, -2, (0,), (-1,), (-1, 1), (3, 1), (-3, 0)]:
      expected_percentile = np.percentile(
          x,
          q=0.77,
          interpolation=self._interpolation,
          axis=axis,
          keepdims=True)
      pct = tfp.stats.percentile(
          x_ph,
          q=0.77,
          interpolation=self._interpolation,
          axis=axis,
          keep_dims=True)
      self.assertAllClose(expected_percentile, self.evaluate(pct))

  def test_with_integer_dtype(self):
    if self._interpolation in {'linear', 'midpoint'}:
      self.skipTest('Skipping integer dtype test for interpolation {}'.format(
          self._interpolation))
    x = [1, 5, 3, 2, 4]
    for q in [0, 10, 25, 49.9, 50, 50.01, 90, 95, 100]:
      expected_percentile = np.percentile(
          x, q=q, interpolation=self._interpolation)
      pct = tfp.stats.percentile(x, q=q, interpolation=self._interpolation)
      self.assertEqual(tf.int32, pct.dtype)
      self.assertAllEqual((), pct.shape)
      self.assertAllClose(expected_percentile, self.evaluate(pct))


class PercentileTestWithLinearInterpolation(
    PercentileTestWithLowerInterpolation):

  _interpolation = 'linear'

  def test_integer_dtype_raises(self):
    with self.assertRaisesRegexp(TypeError, 'not allowed with dtype'):
      tfp.stats.percentile(x=[1, 2], q=30, interpolation='linear')

  def test_grads_at_sample_pts_with_no_preserve_gradients(self):
    dist = tfp.distributions.Normal(np.float64(0), np.float64(1))
    x = dist.sample(10001, seed=0)
    # 50th quantile will lie exactly on a data point.
    # 49.123... will not
    q = tf.constant(np.array([50, 49.123456789]))  # Percentiles, in [0, 100]

    analytic_pct, grad_analytic_pct = tfp.math.value_and_gradient(
        lambda q_: dist.quantile(q_ / 100.), q)
    sample_pct, grad_sample_pct = tfp.math.value_and_gradient(
        lambda q_: tfp.stats.percentile(  # pylint: disable=g-long-lambda
            x, q_, interpolation='linear', preserve_gradients=False),
        q)

    [
        analytic_pct,
        d_analytic_pct_dq,
        sample_pct,
        d_sample_pct_dq,
    ] = self.evaluate([
        analytic_pct,
        grad_analytic_pct,
        sample_pct,
        grad_sample_pct,
    ])

    self.assertAllClose(analytic_pct, sample_pct, atol=0.05)

    # Near the median, the normal PDF is approximately constant C, with
    # C = 1 / sqrt(2 * pi).  So the cdf is approximately F(x) = x / C.
    # Thus the quantile function is approximately F^{-1}(y) = C * y.
    self.assertAllClose(np.sqrt(2 * np.pi) / 100 * np.ones([2]),
                        d_analytic_pct_dq, atol=1e-4)

    # At the 50th percentile exactly, the sample gradient is exactly zero!
    # This is due to preserve_gradient == False.
    self.assertAllEqual(0., d_sample_pct_dq[0])

    # Tolerance at the other point is terrible (2x), but this is a sample
    # quantile based gradient.
    self.assertAllClose(
        d_analytic_pct_dq[1], d_sample_pct_dq[1], atol=0, rtol=2)
    # The absolute values are close though (but tiny).
    self.assertAllClose(
        d_analytic_pct_dq[1], d_sample_pct_dq[1], atol=0.05, rtol=0)

  def test_grads_at_sample_pts_with_yes_preserve_gradients(self):
    dist = tfp.distributions.Normal(np.float64(0), np.float64(1))
    x = dist.sample(10001, seed=0)
    # 50th quantile will lie exactly on a data point.
    # 49.123... will not
    q = tf.constant(np.array([50, 49.123456789]))  # Percentiles, in [0, 100]
    analytic_pct, grad_analytic_pct = tfp.math.value_and_gradient(
        lambda q_: dist.quantile(q_ / 100.), q)
    sample_pct, grad_sample_pct = tfp.math.value_and_gradient(
        lambda q_: tfp.stats.percentile(  # pylint: disable=g-long-lambda
            x, q_, interpolation='linear', preserve_gradients=True),
        q)
    [
        analytic_pct,
        d_analytic_pct_dq,
        sample_pct,
        d_sample_pct_dq,
    ] = self.evaluate([
        analytic_pct,
        grad_analytic_pct,
        sample_pct,
        grad_sample_pct,
    ])

    self.assertAllClose(analytic_pct, sample_pct, atol=0.05)

    # Near the median, the normal PDF is approximately constant C, with
    # C = 1 / sqrt(2 * pi).  So the cdf is approximately F(x) = x / C.
    # Thus the quantile function is approximately F^{-1}(y) = C * y.
    self.assertAllClose(np.sqrt(2 * np.pi) / 100 * np.ones([2]),
                        d_analytic_pct_dq, atol=1e-4)

    # At the 50th percentile exactly, the sample gradient not exactly zero!
    # This is due to preserve_gradient == True.
    self.assertNotEqual(0., d_sample_pct_dq[0])

    # Tolerance is terrible (2x), but this is a sample quantile based gradient.
    self.assertAllClose(d_analytic_pct_dq, d_sample_pct_dq, atol=0, rtol=2)
    # The absolute values are close though (but tiny).
    self.assertAllClose(d_analytic_pct_dq, d_sample_pct_dq, atol=0.1, rtol=0)


class PercentileTestWithMidpointInterpolation(
    PercentileTestWithLowerInterpolation):

  _interpolation = 'midpoint'

  def test_integer_dtype_raises(self):
    with self.assertRaisesRegexp(TypeError, 'not allowed with dtype'):
      tfp.stats.percentile(x=[1, 2], q=30, interpolation='midpoint')


class PercentileTestWithHigherInterpolation(
    PercentileTestWithLowerInterpolation):

  _interpolation = 'higher'


class PercentileTestWithNearestInterpolation(tf.test.TestCase):
  """Test separately because np.round and tf.round make different choices."""

  _interpolation = 'nearest'

  def test_one_dim_odd_input(self):
    x = [1., 5., 3., 2., 4.]
    for q in [0, 10.1, 25.1, 49.9, 50.1, 50.01, 89, 100]:
      expected_percentile = np.percentile(
          x, q=q, interpolation=self._interpolation)
      pct = tfp.stats.percentile(x, q=q, interpolation=self._interpolation)
      self.assertAllEqual((), pct.shape)
      self.assertAllClose(expected_percentile, self.evaluate(pct))

  def test_one_dim_even_input(self):
    x = [1., 5., 3., 2., 4., 5.]
    for q in [0, 10.1, 25.1, 49.9, 50.1, 50.01, 89, 100]:
      expected_percentile = np.percentile(
          x, q=q, interpolation=self._interpolation)
      pct = tfp.stats.percentile(x, q=q, interpolation=self._interpolation)
      self.assertAllEqual((), pct.shape)
      self.assertAllClose(expected_percentile, self.evaluate(pct))

  def test_invalid_interpolation_raises(self):
    x = [1., 5., 3., 2., 4.]
    with self.assertRaisesRegexp(ValueError, 'interpolation'):
      tfp.stats.percentile(x, q=0.5, interpolation='bad')

  def test_2d_q_raises_static(self):
    x = [1., 5., 3., 2., 4.]
    with self.assertRaisesRegexp(ValueError, 'Expected.*ndims'):
      tfp.stats.percentile(x, q=[[0.5]])

  def test_2d_q_raises_dynamic(self):
    if tf.executing_eagerly(): return
    x = [1., 5., 3., 2., 4.]
    q_ph = tf.compat.v1.placeholder_with_default(input=[[0.5]], shape=None)
    pct = tfp.stats.percentile(x, q=q_ph, validate_args=True,
                               interpolation=self._interpolation)
    with self.assertRaisesOpError('rank'):
      self.evaluate(pct)

  def test_finds_max_of_long_array(self):
    # d - 1 == d in float32 and d = 3e7.
    # So this test only passes if we use double for the percentile indices.
    # If float is used, it fails with InvalidArgumentError about an index out of
    # bounds.
    x = tf.linspace(0., 3e7, num=int(3e7))
    minval = tfp.stats.percentile(x, q=0, validate_args=True,
                                  interpolation=self._interpolation)
    self.assertAllEqual(0, self.evaluate(minval))


@test_util.run_all_in_graph_and_eager_modes
class QuantilesTest(tf.test.TestCase):
  """Test for quantiles. Most functionality tested implicitly via percentile."""

  def test_quartiles_of_vector(self):
    x = tf.linspace(0., 1000., 10000)
    cut_points = tfp.stats.quantiles(x, num_quantiles=4)
    self.assertAllEqual((5,), cut_points.shape)
    cut_points_ = self.evaluate(cut_points)
    self.assertAllClose([0., 250., 500., 750., 1000.], cut_points_, rtol=0.002)

  def test_deciles_of_rank_3_tensor(self):
    x = rng.rand(3, 100000, 2)
    cut_points = tfp.stats.quantiles(x, num_quantiles=10, axis=1)
    self.assertAllEqual((11, 3, 2), cut_points.shape)
    cut_points_ = self.evaluate(cut_points)

    # cut_points_[:, i, j] should all be about the same.
    self.assertAllClose(np.linspace(0, 1, 11), cut_points_[:, 0, 0], atol=0.03)
    self.assertAllClose(np.linspace(0, 1, 11), cut_points_[:, 1, 1], atol=0.03)


if __name__ == '__main__':
  tf.test.main()
