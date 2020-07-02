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
"""Tests for the statistical testing library."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

# Dependency imports
from absl.testing import parameterized
import numpy as np
import tensorflow.compat.v1 as tf1
import tensorflow.compat.v2 as tf

from tensorflow_probability.python.distributions.internal import statistical_testing as st
from tensorflow_probability.python.internal import test_util
# pylint: disable=g-error-prone-assert-raises

# This file is testing new assertions, which must of necessity appear in
# assertRaises blocks to check that they do, in fact, raise when their terms are
# violated.


@test_util.test_all_tf_execution_regimes
class StatisticalTestingTest(test_util.TestCase):

  def assert_design_soundness(self, dtype, min_num_samples, min_discrepancy):
    thresholds = [1e-5, 1e-2, 1.1e-1, 0.9, 1., 1.02, 2., 10., 1e2, 1e5, 1e10]
    rates = [1e-6, 1e-3, 1e-2, 1.1e-1, 0.2, 0.5, 0.7, 1.]
    false_fail_rates, false_pass_rates = np.meshgrid(rates, rates)
    false_fail_rates = false_fail_rates.flatten().astype(dtype=dtype)
    false_pass_rates = false_pass_rates.flatten().astype(dtype=dtype)

    detectable_discrepancies = []
    for false_pass_rate, false_fail_rate in zip(
        false_pass_rates, false_fail_rates):
      sufficient_n = min_num_samples(
          thresholds, false_fail_rate=false_fail_rate,
          false_pass_rate=false_pass_rate)
      detectable_discrepancies.append(min_discrepancy(
          sufficient_n, false_fail_rate=false_fail_rate,
          false_pass_rate=false_pass_rate))

    detectable_discrepancies_ = self.evaluate(detectable_discrepancies)
    for discrepancies, false_pass_rate, false_fail_rate in zip(
        detectable_discrepancies_, false_pass_rates, false_fail_rates):
      below_threshold = discrepancies <= thresholds
      self.assertAllEqual(
          np.ones_like(below_threshold, np.bool), below_threshold,
          msg='false_pass_rate({}), false_fail_rate({})'.format(
              false_pass_rate, false_fail_rate))
      # Could probably bring this factor of 2 down by cleverly allocating
      # envelopes in the one-sample min_num_samples computations.
      bound_tight = thresholds <= discrepancies * 2
      self.assertAllEqual(
          np.ones_like(bound_tight, np.bool), bound_tight,
          msg='false_pass_rate({}), false_fail_rate({})'.format(
              false_pass_rate, false_fail_rate))

  @parameterized.parameters(np.float32, np.float64)
  def test_dkwm_design_cdf_one_sample_soundness(self, dtype):
    self.assert_design_soundness(
        dtype, st.min_num_samples_for_dkwm_cdf_test,
        st.min_discrepancy_of_true_cdfs_detectable_by_dkwm)

  @parameterized.parameters(np.float32, np.float64)
  def test_dkwm_cdf_one_sample_assertion(self, dtype):
    rng = np.random.RandomState(seed=0)
    num_samples = 13000

    d = st.min_discrepancy_of_true_cdfs_detectable_by_dkwm(
        num_samples, false_fail_rate=1e-6, false_pass_rate=1e-6)
    d = self.evaluate(d)
    self.assertLess(d, 0.05)

    # Test that the test assertion agrees that the cdf of the standard
    # uniform distribution is the identity.
    samples = rng.uniform(size=num_samples).astype(dtype=dtype)
    self.evaluate(st.assert_true_cdf_equal_by_dkwm(
        samples, lambda x: x, false_fail_rate=1e-6))

    # Test that the test assertion confirms that the cdf of a
    # scaled uniform distribution is not the identity.
    with self.assertRaisesOpError('Empirical CDF outside K-S envelope'):
      samples = rng.uniform(
          low=0., high=0.9, size=num_samples).astype(dtype=dtype)
      self.evaluate(st.assert_true_cdf_equal_by_dkwm(
          samples, lambda x: x, false_fail_rate=1e-6))

    # Test that the test assertion confirms that the cdf of a
    # shifted uniform distribution is not the identity.
    with self.assertRaisesOpError('Empirical CDF outside K-S envelope'):
      samples = rng.uniform(
          low=0.1, high=1.1, size=num_samples).astype(dtype=dtype)
      self.evaluate(st.assert_true_cdf_equal_by_dkwm(
          samples, lambda x: x, false_fail_rate=1e-6))

  def test_empirical_cdfs_with_duplicates(self):
    # A batch of sorted samples with nontrivial repetition.
    # There are repetitions across batch members, which should not
    # confuse the CDF computation.
    # Shape: [num_samples, batch_size]
    samples = np.array([[1, 1, 1],
                        [1, 2, 3],
                        [2, 3, 3],
                        [2, 4, 3]])
    expected_low_cdf_values = np.array(
        [[0, 0, 0.5, 0.5],
         [0, 0.25, 0.5, 0.75],
         [0, 0.25, 0.25, 0.25]])
    expected_low_cdf_values = np.array(
        [[0, 0, 0],
         [0, 0.25, 0.25],
         [0.5, 0.5, 0.25],
         [0.5, 0.75, 0.25]])
    low_empirical_cdfs = self.evaluate(
        st.empirical_cdfs(samples, samples, continuity='left'))
    self.assertAllEqual(expected_low_cdf_values, low_empirical_cdfs)
    expected_high_cdf_values = np.array(
        [[0.5, 0.5, 1, 1],
         [0.25, 0.5, 0.75, 1],
         [0.25, 1, 1, 1]])
    expected_high_cdf_values = np.array(
        [[0.5, 0.25, 0.25],
         [0.5, 0.5, 1],
         [1, 0.75, 1],
         [1, 1, 1]])
    high_empirical_cdfs = self.evaluate(
        st.empirical_cdfs(samples, samples, continuity='right'))
    self.assertAllEqual(expected_high_cdf_values, high_empirical_cdfs)

  @parameterized.parameters(np.float32, np.float64)
  def test_kolmogorov_smirnov_distance(self, dtype):
    samples = np.array([[1, 1, 1],
                        [1, 2, 3],
                        [2, 3, 3],
                        [2, 4, 3]])
    samples = tf.convert_to_tensor(value=samples, dtype=dtype)
    def cdf(x):
      ones = tf.ones_like(x)
      answer = tf1.where(x < 3, 0.6 * ones, ones)
      answer = tf1.where(x < 2, 0.3 * ones, answer)
      answer = tf1.where(x < 1, 0.1 * ones, answer)
      return tf1.where(x < 0, 0 * ones, answer)

    def left_continuous_cdf(x):
      ones = tf.ones_like(x)
      answer = tf1.where(x <= 3, 0.6 * ones, ones)
      answer = tf1.where(x <= 2, 0.3 * ones, answer)
      answer = tf1.where(x <= 1, 0.1 * ones, answer)
      return tf1.where(x <= 0, 0 * ones, answer)
    obtained = self.evaluate(st.kolmogorov_smirnov_distance(
        samples, cdf, left_continuous_cdf=left_continuous_cdf))
    self.assertAllClose([0.4, 0.25, 0.35], obtained, atol=1e-7)

  @parameterized.parameters(np.float32, np.float64)
  def test_dkwm_cdf_one_sample_batch_discrete_assertion(self, dtype):
    rng = np.random.RandomState(seed=0)
    num_samples = 13000
    batch_shape = [3, 2]
    shape = [num_samples] + batch_shape

    probs = [0.1, 0.2, 0.3, 0.4]
    samples = rng.choice(4, size=shape, p=probs).astype(dtype=dtype)
    def cdf(x):
      ones = tf.ones_like(x)
      answer = tf1.where(x < 3, 0.6 * ones, ones)
      answer = tf1.where(x < 2, 0.3 * ones, answer)
      answer = tf1.where(x < 1, 0.1 * ones, answer)
      return tf1.where(x < 0, 0 * ones, answer)

    def left_continuous_cdf(x):
      ones = tf.ones_like(x)
      answer = tf1.where(x <= 3, 0.6 * ones, ones)
      answer = tf1.where(x <= 2, 0.3 * ones, answer)
      answer = tf1.where(x <= 1, 0.1 * ones, answer)
      return tf1.where(x <= 0, 0 * ones, answer)

    self.evaluate(st.assert_true_cdf_equal_by_dkwm(
        samples, cdf, left_continuous_cdf=left_continuous_cdf,
        false_fail_rate=1e-6))
    d = st.min_discrepancy_of_true_cdfs_detectable_by_dkwm(
        tf.ones(batch_shape) * num_samples,
        false_fail_rate=1e-6, false_pass_rate=1e-6)
    self.assertTrue(np.all(self.evaluate(d) < 0.05))

    def check_catches_mistake(wrong_probs):
      wrong_samples = rng.choice(
          len(wrong_probs), size=shape, p=wrong_probs).astype(dtype=dtype)
      with self.assertRaisesOpError('Empirical CDF outside K-S envelope'):
        self.evaluate(st.assert_true_cdf_equal_by_dkwm(
            wrong_samples, cdf, left_continuous_cdf=left_continuous_cdf,
            false_fail_rate=1e-6))

    check_catches_mistake([0.1, 0.2, 0.3, 0.3, 0.1])
    check_catches_mistake([0.2, 0.2, 0.3, 0.3])

  @parameterized.parameters(np.float32, np.float64)
  def test_kolmogorov_smirnov_distance_two_sample(self, dtype):
    del dtype
    samples1 = np.array([[1, 1, 2, 2],
                         [1, 2, 3, 4],
                         [1, 3, 3, 3]])
    samples2 = np.array([[1, 2, 2, 1, 1, 2],
                         [4, 4, 3, 2, 1, 1],
                         [2, 2, 2, 1, -4, 7]])
    # Unlike empirical_cdfs, the samples Tensors must come in iid across the
    # leading dimension.
    obtained = self.evaluate(st.kolmogorov_smirnov_distance_two_sample(
        tf.transpose(a=samples1), tf.transpose(a=samples2)))
    self.assertAllClose([0.0, 1 / 12., 0.75 - 1 / 6.], obtained, atol=1e-7)

  @parameterized.parameters(np.float32, np.float64)
  def test_dkwm_cdf_two_sample_batch_discrete_assertion(self, dtype):
    rng = np.random.RandomState(seed=0)
    num_samples = 52000
    batch_shape = [3, 2]
    shape = [num_samples] + batch_shape

    probs = [0.1, 0.2, 0.3, 0.4]
    samples1 = rng.choice(4, size=shape, p=probs).astype(dtype=dtype)
    samples2 = rng.choice(4, size=shape, p=probs).astype(dtype=dtype)
    self.evaluate(st.assert_true_cdf_equal_by_dkwm_two_sample(
        samples1, samples2, false_fail_rate=1e-6))

    def check_catches_mistake(wrong_probs):
      wrong_samples = rng.choice(
          len(wrong_probs), size=shape, p=wrong_probs).astype(dtype=dtype)
      with self.assertRaisesOpError(
          'Empirical CDFs outside joint K-S envelope'):
        self.evaluate(st.assert_true_cdf_equal_by_dkwm_two_sample(
            samples1, wrong_samples, false_fail_rate=1e-6))

    n = tf.ones(batch_shape) * num_samples
    d = st.min_discrepancy_of_true_cdfs_detectable_by_dkwm_two_sample(
        n, n, false_fail_rate=1e-6, false_pass_rate=1e-6)
    self.assertTrue(np.all(self.evaluate(d) < 0.05))

    check_catches_mistake([0.1, 0.2, 0.3, 0.3, 0.1])
    check_catches_mistake([0.2, 0.2, 0.3, 0.3])

  @parameterized.parameters(np.float32, np.float64)
  def test_dkwm_design_mean_one_sample_soundness(self, dtype):
    self.assert_design_soundness(
        dtype,
        functools.partial(
            st.min_num_samples_for_dkwm_mean_test, low=0., high=1.),
        functools.partial(
            st.min_discrepancy_of_true_means_detectable_by_dkwm,
            low=0., high=1.))

  def assert_design_soundness_two_sample(
      self, dtype, min_num_samples, min_discrepancy):
    thresholds = [1e-5, 1e-2, 1.1e-1, 0.9, 1., 1.02, 2., 10., 1e2, 1e5, 1e10]
    rates = [1e-6, 1e-3, 1e-2, 1.1e-1, 0.2, 0.5, 0.7, 1.]
    false_fail_rates, false_pass_rates = np.meshgrid(rates, rates)
    false_fail_rates = false_fail_rates.flatten().astype(dtype=dtype)
    false_pass_rates = false_pass_rates.flatten().astype(dtype=dtype)

    detectable_discrepancies = []
    for false_pass_rate, false_fail_rate in zip(
        false_pass_rates, false_fail_rates):
      [
          sufficient_n1,
          sufficient_n2
      ] = min_num_samples(
          thresholds,
          false_fail_rate=false_fail_rate,
          false_pass_rate=false_pass_rate)

      detectable_discrepancies.append(
          min_discrepancy(
              n1=sufficient_n1,
              n2=sufficient_n2,
              false_fail_rate=false_fail_rate,
              false_pass_rate=false_pass_rate))

    detectable_discrepancies_ = self.evaluate(detectable_discrepancies)
    for discrepancies, false_pass_rate, false_fail_rate in zip(
        detectable_discrepancies_, false_pass_rates, false_fail_rates):
      below_threshold = discrepancies <= thresholds
      self.assertAllEqual(
          np.ones_like(below_threshold, np.bool), below_threshold,
          msg='false_pass_rate({}), false_fail_rate({})'.format(
              false_pass_rate, false_fail_rate))
      bound_tight = thresholds <= discrepancies * 2
      self.assertAllEqual(
          np.ones_like(bound_tight, np.bool), bound_tight,
          msg='false_pass_rate({}), false_fail_rate({})'.format(
              false_pass_rate, false_fail_rate))

  @parameterized.parameters(np.float32, np.float64)
  def test_dkwm_design_cdf_two_sample_soundness(self, dtype):
    self.assert_design_soundness_two_sample(
        dtype, st.min_num_samples_for_dkwm_cdf_two_sample_test,
        st.min_discrepancy_of_true_cdfs_detectable_by_dkwm_two_sample)

  @parameterized.parameters(np.float32, np.float64)
  def test_dkwm_design_mean_two_sample_soundness(self, dtype):
    min_num_samples = functools.partial(
        st.min_num_samples_for_dkwm_mean_two_sample_test,
        low1=0., high1=1., low2=0., high2=1.)
    min_discrepancy = functools.partial(
        st.min_discrepancy_of_true_means_detectable_by_dkwm_two_sample,
        low1=0., high1=1., low2=0., high2=1.)
    self.assert_design_soundness_two_sample(
        dtype, min_num_samples, min_discrepancy)

  @parameterized.parameters(np.float32, np.float64)
  def test_true_mean_confidence_interval_by_dkwm_one_sample(self, dtype):
    rng = np.random.RandomState(seed=0)

    num_samples = 5000
    # 5000 samples is chosen to be enough to find discrepancies of
    # size 0.1 or more with assurance 1e-6, as confirmed here:
    d = st.min_discrepancy_of_true_means_detectable_by_dkwm(
        num_samples, 0., 1., false_fail_rate=1e-6, false_pass_rate=1e-6)
    d = self.evaluate(d)
    self.assertLess(d, 0.1)

    # Test that the confidence interval computed for the mean includes
    # 0.5 and excludes 0.4 and 0.6.
    samples = rng.uniform(size=num_samples).astype(dtype=dtype)
    (low, high) = st.true_mean_confidence_interval_by_dkwm(
        samples, 0., 1., error_rate=1e-6)
    low, high = self.evaluate([low, high])
    self.assertGreater(low, 0.4)
    self.assertLess(low, 0.5)
    self.assertGreater(high, 0.5)
    self.assertLess(high, 0.6)

  @parameterized.parameters(np.float32, np.float64)
  def test_dkwm_mean_one_sample_assertion(self, dtype):
    rng = np.random.RandomState(seed=0)
    num_samples = 5000

    # Test that the test assertion agrees that the mean of the standard
    # uniform distribution is 0.5.
    samples = rng.uniform(size=num_samples).astype(dtype=dtype)
    self.evaluate(st.assert_true_mean_equal_by_dkwm(
        samples, 0., 1., 0.5, false_fail_rate=1e-6))

    # Test that the test assertion confirms that the mean of the
    # standard uniform distribution is not 0.4.
    with self.assertRaisesOpError('true mean greater than expected'):
      self.evaluate(st.assert_true_mean_equal_by_dkwm(
          samples, 0., 1., 0.4, false_fail_rate=1e-6))

    # Test that the test assertion confirms that the mean of the
    # standard uniform distribution is not 0.6.
    with self.assertRaisesOpError('true mean smaller than expected'):
      self.evaluate(st.assert_true_mean_equal_by_dkwm(
          samples, 0., 1., 0.6, false_fail_rate=1e-6))

  @parameterized.parameters(np.float32, np.float64)
  def test_dkwm_mean_in_interval_one_sample_assertion(self, dtype):
    rng = np.random.RandomState(seed=0)
    num_samples = 5000

    # Test that the test assertion agrees that the mean of the standard
    # uniform distribution is between 0.4 and 0.6.
    samples = rng.uniform(size=num_samples).astype(dtype=dtype)
    self.evaluate(st.assert_true_mean_in_interval_by_dkwm(
        samples, 0., 1.,
        expected_low=0.4, expected_high=0.6, false_fail_rate=1e-6))

    # Test that the test assertion confirms that the mean of the
    # standard uniform distribution is not between 0.2 and 0.4.
    with self.assertRaisesOpError('true mean greater than expected'):
      self.evaluate(st.assert_true_mean_in_interval_by_dkwm(
          samples, 0., 1.,
          expected_low=0.2, expected_high=0.4, false_fail_rate=1e-6))

    # Test that the test assertion confirms that the mean of the
    # standard uniform distribution is not between 0.6 and 0.8.
    with self.assertRaisesOpError('true mean smaller than expected'):
      self.evaluate(st.assert_true_mean_in_interval_by_dkwm(
          samples, 0., 1.,
          expected_low=0.6, expected_high=0.8, false_fail_rate=1e-6))

  @parameterized.parameters(np.float32, np.float64)
  def test_dkwm_mean_two_sample_assertion(self, dtype):
    rng = np.random.RandomState(seed=0)
    num_samples = 4000

    # 4000 samples is chosen to be enough to find discrepancies of
    # size 0.2 or more with assurance 1e-6, as confirmed here:
    d = st.min_discrepancy_of_true_means_detectable_by_dkwm_two_sample(
        num_samples, 0., 1., num_samples, 0., 1.,
        false_fail_rate=1e-6, false_pass_rate=1e-6)
    d = self.evaluate(d)
    self.assertLess(d, 0.2)

    # Test that the test assertion agrees that the standard
    # uniform distribution has the same mean as itself.
    samples1 = rng.uniform(size=num_samples).astype(dtype=dtype)
    samples2 = rng.uniform(size=num_samples).astype(dtype=dtype)
    self.evaluate(st.assert_true_mean_equal_by_dkwm_two_sample(
        samples1, 0., 1., samples2, 0., 1., false_fail_rate=1e-6))

  @parameterized.parameters(np.float32, np.float64)
  def test_dkwm_mean_two_sample_assertion_beta_2_1_false(self, dtype):
    rng = np.random.RandomState(seed=0)
    num_samples = 4000
    samples1 = rng.uniform(size=num_samples).astype(dtype=dtype)

    # As established above, 4000 samples is enough to find discrepancies
    # of size 0.2 or more with assurance 1e-6.

    # Test that the test assertion confirms that the mean of the
    # standard uniform distribution is different from the mean of beta(2, 1).
    beta_high_samples = rng.beta(2, 1, size=num_samples).astype(dtype=dtype)
    with self.assertRaisesOpError('true mean smaller than expected'):
      self.evaluate(st.assert_true_mean_equal_by_dkwm_two_sample(
          samples1, 0., 1.,
          beta_high_samples, 0., 1.,
          false_fail_rate=1e-6))

  @parameterized.parameters(np.float32, np.float64)
  def test_dkwm_mean_two_sample_assertion_beta_1_2_false(self, dtype):
    rng = np.random.RandomState(seed=0)
    num_samples = 4000
    samples1 = rng.uniform(size=num_samples).astype(dtype=dtype)

    # As established above, 4000 samples is enough to find discrepancies
    # of size 0.2 or more with assurance 1e-6.

    # Test that the test assertion confirms that the mean of the
    # standard uniform distribution is different from the mean of beta(1, 2).
    beta_low_samples = rng.beta(1, 2, size=num_samples).astype(dtype=dtype)
    with self.assertRaisesOpError('true mean greater than expected'):
      self.evaluate(st.assert_true_mean_equal_by_dkwm_two_sample(
          samples1, 0., 1.,
          beta_low_samples, 0., 1.,
          false_fail_rate=1e-6))

  @parameterized.parameters(np.float32, np.float64)
  def test_dkwm_argument_validity_checking(self, dtype):
    rng = np.random.RandomState(seed=0)
    samples = rng.uniform(
        low=[0., 1.], high=[1., 2.], size=(2500, 1, 2)).astype(dtype=dtype)

    # Test that the test library complains if the given samples fall
    # outside the purported bounds.
    with self.assertRaisesOpError('maximum value exceeds expectations'):
      self.evaluate(st.true_mean_confidence_interval_by_dkwm(
          samples, [[0., 1.]], [[0.5, 1.5]], error_rate=0.5))
    with self.assertRaisesOpError('minimum value falls below expectations'):
      self.evaluate(st.true_mean_confidence_interval_by_dkwm(
          samples, [[0.5, 1.5]], [[1., 2.]], error_rate=0.5))

    # But doesn't complain if they don't.
    op = st.true_mean_confidence_interval_by_dkwm(
        samples, [[0., 1.]], [[1., 2.]], error_rate=0.5)
    _ = self.evaluate(op)

  @parameterized.parameters(np.float32, np.float64)
  def test_do_maximum_mean(self, dtype):
    n = 117
    envelope = 0.02  # > 2 / n, but < 3 / n
    rng = np.random.RandomState(seed=8)
    samples = rng.uniform(size=n).astype(dtype=dtype)

    # Compute the answer in TF using the code under test
    envelope_t = tf.convert_to_tensor(value=envelope, dtype=dtype)
    max_mean = st._do_maximum_mean(samples, envelope_t, 1)
    max_mean = self.evaluate(max_mean)

    # Compute the correct answer for this case in numpy.  In this
    # example, `n` and `envelope` are such that `samples[2]` is the
    # element that should be taken partially, regardless of the
    # content of the `samples` array (see algorithm description in
    # `../ops/statistical_testing.py`).
    samples = sorted(samples)
    weight = 1. / n - (envelope - 2. / n)
    answer = samples[2] * weight + sum(samples[3:]) / n + envelope * 1.
    self.assertAllClose(max_mean, answer, rtol=1e-9)

  @parameterized.parameters(np.float32, np.float64)
  def test_random_projections(self, dtype):
    strm = test_util.test_seed_stream()
    rng = np.random.RandomState(seed=strm() % 2**31)
    num_samples = 57000

    # Validate experiment design

    # False fail rate here is the target rate of 1e-6 divided by the number of
    # projections.
    d = st.min_discrepancy_of_true_cdfs_detectable_by_dkwm_two_sample(
        num_samples, num_samples, false_fail_rate=1e-8, false_pass_rate=1e-6)
    # Choose num_samples so the discrepancy is below 0.05, which should be
    # enough to detect a mean shift of around 1/8 of a standard deviation, or a
    # scale increase of around 25% (in any particular projection).
    self.assertLess(self.evaluate(d), 0.05)

    ground_truth = rng.multivariate_normal(
        mean=[0, 0], cov=[[1, 0.5], [0.5, 1]], size=num_samples).astype(dtype)
    more_samples = rng.multivariate_normal(
        mean=[0, 0], cov=[[1, 0.5], [0.5, 1]], size=num_samples).astype(dtype)
    self.evaluate(
        st.assert_multivariate_true_cdf_equal_on_projections_two_sample(
            ground_truth, more_samples, num_projections=100,
            false_fail_rate=1e-6, seed=strm()))

    def assert_catches_mistake(mean, cov):
      wrong_samples = rng.multivariate_normal(
          mean=mean, cov=cov, size=num_samples).astype(dtype=dtype)
      msg = 'Empirical CDFs outside joint K-S envelope'
      with self.assertRaisesOpError(msg):
        self.evaluate(
            st.assert_multivariate_true_cdf_equal_on_projections_two_sample(
                ground_truth, wrong_samples, num_projections=100,
                false_fail_rate=1e-6, seed=strm()))

    assert_catches_mistake([0, 1], [[1, 0.5], [0.5, 1]])
    assert_catches_mistake([0, 0], [[1, 0.7], [0.7, 1]])
    assert_catches_mistake([0, 0], [[1, 0.3], [0.3, 1]])

if __name__ == '__main__':
  tf.test.main()
