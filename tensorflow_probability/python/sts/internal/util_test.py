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
"""Tests for StructuralTimeSeries utilities."""

from absl.testing import parameterized

import numpy as np
import pandas as pd
import tensorflow.compat.v1 as tf1
import tensorflow.compat.v2 as tf

from tensorflow_probability.python.distributions import mvn_diag
from tensorflow_probability.python.distributions import mvn_tril
from tensorflow_probability.python.distributions import normal
from tensorflow_probability.python.internal import tensorshape_util
from tensorflow_probability.python.internal import test_util
from tensorflow_probability.python.sts.components import local_level
from tensorflow_probability.python.sts.internal import missing_values_util
from tensorflow_probability.python.sts.internal import util as sts_util


@test_util.test_all_tf_execution_regimes
class MultivariateNormalUtilsTest(test_util.TestCase):

  def test_factored_joint_mvn_diag_full(self):
    batch_shape = [3, 2]

    mvn1 = mvn_diag.MultivariateNormalDiag(
        loc=tf.zeros(batch_shape + [3]), scale_diag=tf.ones(batch_shape + [3]))

    mvn2 = mvn_tril.MultivariateNormalTriL(
        loc=tf.ones(batch_shape + [2]),
        scale_tril=(tf.ones(batch_shape + [2, 2]) *
                    tf.linalg.cholesky([[5., -2], [-2, 3.1]])))

    joint = sts_util.factored_joint_mvn([mvn1, mvn2])
    self.assertEqual(self.evaluate(joint.event_shape_tensor()),
                     self.evaluate(mvn1.event_shape_tensor() +
                                   mvn2.event_shape_tensor()))

    joint_mean_ = self.evaluate(joint.mean())
    self.assertAllEqual(joint_mean_[..., :3], self.evaluate(mvn1.mean()))
    self.assertAllEqual(joint_mean_[..., 3:], self.evaluate(mvn2.mean()))

    joint_cov_ = self.evaluate(joint.covariance())
    self.assertAllEqual(joint_cov_[..., :3, :3],
                        self.evaluate(mvn1.covariance()))
    self.assertAllEqual(joint_cov_[..., 3:, 3:],
                        self.evaluate(mvn2.covariance()))

  def test_factored_joint_mvn_broadcast_batch_shape(self):
    # Test that combining MVNs with different but broadcast-compatible
    # batch shapes yields an MVN with the correct broadcast batch shape.
    random_with_shape = (
        lambda shape: np.random.standard_normal(shape).astype(np.float32))

    event_shape = [3]
    # mvn with batch shape [2]
    mvn1 = mvn_diag.MultivariateNormalDiag(
        loc=random_with_shape([2] + event_shape),
        scale_diag=tf.exp(random_with_shape([2] + event_shape)))

    # mvn with batch shape [3, 2]
    mvn2 = mvn_diag.MultivariateNormalDiag(
        loc=random_with_shape([3, 2] + event_shape),
        scale_diag=tf.exp(random_with_shape([1, 2] + event_shape)))

    # mvn with batch shape [1, 2]
    mvn3 = mvn_diag.MultivariateNormalDiag(
        loc=random_with_shape([1, 2] + event_shape),
        scale_diag=tf.exp(random_with_shape([2] + event_shape)))

    joint = sts_util.factored_joint_mvn([mvn1, mvn2, mvn3])
    self.assertAllEqual(self.evaluate(joint.batch_shape_tensor()), [3, 2])

    joint_mean_ = self.evaluate(joint.mean())
    broadcast_means = tf.ones_like(joint.mean()[..., 0:1])
    self.assertAllEqual(joint_mean_[..., :3],
                        self.evaluate(broadcast_means * mvn1.mean()))
    self.assertAllEqual(joint_mean_[..., 3:6],
                        self.evaluate(broadcast_means * mvn2.mean()))
    self.assertAllEqual(joint_mean_[..., 6:9],
                        self.evaluate(broadcast_means * mvn3.mean()))

    joint_cov_ = self.evaluate(joint.covariance())
    broadcast_covs = tf.ones_like(joint.covariance()[..., :1, :1])
    self.assertAllEqual(joint_cov_[..., :3, :3],
                        self.evaluate(broadcast_covs * mvn1.covariance()))
    self.assertAllEqual(joint_cov_[..., 3:6, 3:6],
                        self.evaluate(broadcast_covs * mvn2.covariance()))
    self.assertAllEqual(joint_cov_[..., 6:9, 6:9],
                        self.evaluate(broadcast_covs * mvn3.covariance()))

  def test_sum_mvns(self):
    batch_shape = [4, 2]
    random_with_shape = (
        lambda shape: np.random.standard_normal(shape).astype(np.float32))

    mvn1 = mvn_diag.MultivariateNormalDiag(
        loc=random_with_shape(batch_shape + [3]),
        scale_diag=np.exp(random_with_shape(batch_shape + [3])))
    mvn2 = mvn_diag.MultivariateNormalDiag(
        loc=random_with_shape(batch_shape + [3]),
        scale_diag=np.exp(random_with_shape(batch_shape + [3])))

    sum_mvn = sts_util.sum_mvns([mvn1, mvn2])
    self.assertAllClose(self.evaluate(sum_mvn.mean()),
                        self.evaluate(mvn1.mean() + mvn2.mean()))
    self.assertAllClose(self.evaluate(sum_mvn.covariance()),
                        self.evaluate(mvn1.covariance() + mvn2.covariance()))

  def test_sum_mvns_broadcast_batch_shape(self):
    random_with_shape = (
        lambda shape: np.random.standard_normal(shape).astype(np.float32))

    event_shape = [3]
    mvn1 = mvn_diag.MultivariateNormalDiag(
        loc=random_with_shape([2] + event_shape),
        scale_diag=np.exp(random_with_shape([2] + event_shape)))
    mvn2 = mvn_diag.MultivariateNormalDiag(
        loc=random_with_shape([1, 2] + event_shape),
        scale_diag=np.exp(random_with_shape([3, 2] + event_shape)))
    mvn3 = mvn_diag.MultivariateNormalDiag(
        loc=random_with_shape([3, 2] + event_shape),
        scale_diag=np.exp(random_with_shape([2] + event_shape)))

    sum_mvn = sts_util.sum_mvns([mvn1, mvn2, mvn3])
    self.assertAllClose(self.evaluate(sum_mvn.mean()),
                        self.evaluate(mvn1.mean() + mvn2.mean() + mvn3.mean()))
    self.assertAllClose(self.evaluate(sum_mvn.covariance()),
                        self.evaluate(mvn1.covariance() +
                                      mvn2.covariance() +
                                      mvn3.covariance()))


class _UtilityTests(test_util.TestCase):

  def test_broadcast_batch_shape(self):
    batch_shapes = ([2], [3, 2], [1, 2])
    distributions = [
        normal.Normal(  # pylint:disable=g-complex-comprehension
            loc=self._build_tensor(np.zeros(batch_shape)),
            scale=self._build_tensor(np.ones(batch_shape)))
        for batch_shape in batch_shapes]
    if self.use_static_shape:
      self.assertEqual(
          [3, 2], sts_util.broadcast_batch_shape(distributions))
    else:
      broadcast_batch_shape = sts_util.broadcast_batch_shape(distributions)
      # Broadcast shape in Eager can contain Python `int`s, so we need to
      # explicitly convert to Tensor.
      self.assertAllEqual([3, 2], self.evaluate(tf.convert_to_tensor(
          value=broadcast_batch_shape)))

  def test_maybe_expand_trailing_dim(self):
    for shape_in, expected_shape_out in [
        # pyformat: disable
        ([4, 3], [4, 3, 1]),
        ([4, 3, 1], [4, 3, 1]),
        ([4], [4, 1]),
        ([1], [1]),
        ([4, 1], [4, 1])
        # pyformat: enable
    ]:
      shape_out = self._shape_as_list(
          sts_util._maybe_expand_trailing_dim(
              self._build_tensor(np.zeros(shape_in))))
      self.assertAllEqual(shape_out, expected_shape_out)

  def test_empirical_statistics_accepts_masked_values(self):

    # Ensure that masks broadcast over batch shape by creating a batch of
    # time series.
    time_series = np.random.randn(3, 2, 5)
    mask = np.array([[True, False, False, True, False],
                     [True, True, True, True, True]])

    masked_series = missing_values_util.MaskedTimeSeries(
        time_series=time_series, is_missing=mask)
    mean, stddev, initial = self.evaluate(
        sts_util.empirical_statistics(masked_series))

    # Should return default values when the series is completely masked.
    self.assertAllClose(mean[:, 1], tf.zeros_like(mean[:, 1]))
    self.assertAllClose(stddev[:, 1], tf.ones_like(stddev[:, 1]))
    self.assertAllClose(initial[:, 1], tf.zeros_like(initial[:, 1]))

    # Otherwise, should return the actual mean/stddev/initial values.
    time_series = time_series[:, 0, :]
    mask = mask[0, :]
    broadcast_mask = np.broadcast_to(mask, time_series.shape)
    unmasked_series = time_series[~broadcast_mask].reshape([3, 3])
    unmasked_mean, unmasked_stddev, unmasked_initial = self.evaluate(
        sts_util.empirical_statistics(unmasked_series))
    self.assertAllClose(mean[:, 0], unmasked_mean)
    self.assertAllClose(stddev[:, 0], unmasked_stddev)
    self.assertAllClose(initial[:, 0], unmasked_initial)

    # Run the same tests without batch shape.
    unbatched_time_series = time_series[0, :]
    masked_series = missing_values_util.MaskedTimeSeries(
        time_series=unbatched_time_series, is_missing=mask)
    mean, stddev, initial = self.evaluate(
        sts_util.empirical_statistics(masked_series))
    unmasked_mean, unmasked_stddev, unmasked_initial = self.evaluate(
        sts_util.empirical_statistics(unbatched_time_series[~mask]))
    self.assertAllClose(mean, unmasked_mean)
    self.assertAllClose(stddev, unmasked_stddev)
    self.assertAllClose(initial, unmasked_initial)

  def test_mix_over_posterior_draws(self):
    num_posterior_draws = 3
    batch_shape = [2, 1]
    means = np.random.randn(*np.concatenate([[num_posterior_draws],
                                             batch_shape]))
    variances = np.exp(np.random.randn(*np.concatenate(
        [[num_posterior_draws], batch_shape])))

    posterior_mixture_dist = sts_util.mix_over_posterior_draws(
        self._build_tensor(means),
        self._build_tensor(variances))

    # Compute the true statistics of the mixture distribution.
    mixture_mean = np.mean(means, axis=0)
    mixture_variance = np.mean(variances + means**2, axis=0) - mixture_mean**2

    self.assertAllClose(mixture_mean,
                        self.evaluate(posterior_mixture_dist.mean()))
    self.assertAllClose(mixture_variance,
                        self.evaluate(posterior_mixture_dist.variance()))

  def test_pad_batch_dimension_when_input_has_sample_shape(self):

    model_batch_shape = [3, 2]
    model = local_level.LocalLevel(
        level_scale_prior=normal.Normal(
            loc=self._build_tensor(np.random.randn(*model_batch_shape)),
            scale=1.))

    num_timesteps = 5
    sample_shape = [4]
    observed_time_series = self._build_tensor(np.random.randn(
        *(sample_shape + model_batch_shape + [num_timesteps, 1])))

    padded_observed_time_series = (
        sts_util.pad_batch_dimension_for_multiple_chains(
            observed_time_series, model=model, chain_batch_shape=[8, 2]))
    self.assertAllEqual(
        self._shape_as_list(padded_observed_time_series.time_series),
        sample_shape + [1, 1] + model_batch_shape + [num_timesteps, 1])

  def test_dont_pad_batch_dimension_when_input_has_no_sample_shape(self):

    model_batch_shape = [3, 2]
    model = local_level.LocalLevel(
        level_scale_prior=normal.Normal(
            loc=self._build_tensor(np.random.randn(*model_batch_shape)),
            scale=1.))

    num_timesteps = 5
    observed_time_series = self._build_tensor(
        np.random.randn(num_timesteps, 1))

    padded_observed_time_series = (
        sts_util.pad_batch_dimension_for_multiple_chains(
            observed_time_series, model=model, chain_batch_shape=[8, 2]))
    self.assertAllEqual(
        self._shape_as_list(padded_observed_time_series.time_series),
        self._shape_as_list(observed_time_series))

  @parameterized.named_parameters(
      ('array_with_unit_dim', np.array([[3.], [4.], [5.]]), [3, 1], None),
      ('array_with_nans', np.array([3., 4., np.nan]), [3, 1
                                                      ], [False, False, True]),
      ('tensor_with_nans', lambda: tf.constant([3., 4., np.nan]), [3, 1],
       [False, False, True]),
      ('masked_time_series',
       missing_values_util.MaskedTimeSeries(
           [1., 2., 3.], [False, True, False]), [3, 1], [False, True, False]),
      (
          'masked_time_series_tensor',
          lambda: missing_values_util.MaskedTimeSeries(  # pylint: disable=g-long-lambda
              tf.constant([1., 2., 3.]),
              tf.constant([False, True, False])),
          [3, 1],
          [False, True, False]),
      ('series_fully_observed',
       pd.Series([1., 2., 3.], index=pd.date_range(
           '2014-01-01', '2014-01-03')), [3, 1], None),
      ('series_partially_observed',
       pd.Series(
           [1., np.nan, 3.], index=pd.date_range(
               '2014-01-01', '2014-01-03')), [3, 1], [False, True, False]),
      ('series_indexed_by_range_only', pd.Series([1., 2., 3.]), [3, 1], None),
      ('dataframe_single_column',
       pd.DataFrame([1., np.nan, 3.],
                    columns=['value'],
                    index=pd.date_range('2014-01-01', '2014-01-03')), [3, 1],
       [False, True, False]),
      ('dataframe_multi_column',
       pd.DataFrame([[1., 4.], [np.nan, 2.], [3., np.nan]],
                    columns=['value1', 'value2'],
                    index=pd.date_range('2014-01-01', '2014-01-03')), [2, 3, 1],
       [[False, True, False], [False, False, True]]))
  def test_canonicalizes_observed_time_series(
      self, observed_time_series, expected_shape, expected_is_missing):
    if callable(observed_time_series):
      observed_time_series = observed_time_series()
    observed_time_series, is_missing = (
        sts_util.canonicalize_observed_time_series_with_mask(
            observed_time_series))
    # Evaluate with explicit identity ops to avoid TF1 error
    # `RuntimeError: The Session graph is empty.`
    observed_time_series, is_missing = self.evaluate(
        (observed_time_series, is_missing))
    self.assertAllEqual(observed_time_series.shape, expected_shape)
    if is_missing is None:
      self.assertIsNone(expected_is_missing)
    elif expected_is_missing is None:
      expected_is_missing = np.zeros(is_missing.shape, dtype=bool)
    self.assertAllEqual(expected_is_missing, is_missing)

  def test_series_with_no_fixed_frequency_raises_error(self):
    with self.assertRaisesRegex(ValueError, 'no set frequency'):
      observed_time_series = pd.Series(
          [1., 2., 4.],
          index=pd.to_datetime(['2014-01-01', '2014-01-02', '2014-01-04']))
      sts_util.canonicalize_observed_time_series_with_mask(observed_time_series)

  def _shape_as_list(self, tensor):
    if self.use_static_shape:
      return tensorshape_util.as_list(tensor.shape)
    else:
      return list(self.evaluate(tf.shape(tensor)))

  def _build_tensor(self, ndarray):
    """Convert a numpy array to a TF placeholder.

    Args:
      ndarray: any object convertible to a numpy array via `np.asarray()`.

    Returns:
      placeholder: a TensorFlow `placeholder` with default value given by the
      provided `ndarray`, dtype given by `self.dtype`, and shape specified
      statically only if `self.use_static_shape` is `True`.
    """

    ndarray = np.asarray(ndarray).astype(self.dtype)
    return tf1.placeholder_with_default(
        ndarray, shape=ndarray.shape if self.use_static_shape else None)


@test_util.test_all_tf_execution_regimes
class UtilityTestsDynamicFloat32(_UtilityTests):
  use_static_shape = False
  dtype = np.float32


class UtilityTestsStaticFloat64(_UtilityTests):
  use_static_shape = True
  dtype = np.float64

del _UtilityTests

if __name__ == '__main__':
  test_util.main()
