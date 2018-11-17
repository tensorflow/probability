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

# Dependency imports
from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from tensorflow_probability.python.distributions.internal import statistical_testing as st


@parameterized.parameters(np.float32, np.float64)
class StatisticalTestingTest(tf.test.TestCase):

  def test_dkwm_design_mean_one_sample_soundness(self, dtype):
    thresholds = [1e-5, 1e-2, 1.1e-1, 0.9, 1., 1.02, 2., 10., 1e2, 1e5, 1e10]
    rates = [1e-6, 1e-3, 1e-2, 1.1e-1, 0.2, 0.5, 0.7, 1.]
    false_fail_rates, false_pass_rates = np.meshgrid(rates, rates)
    false_fail_rates = false_fail_rates.flatten().astype(dtype=dtype)
    false_pass_rates = false_pass_rates.flatten().astype(dtype=dtype)

    detectable_discrepancies = []
    for false_pass_rate, false_fail_rate in zip(
        false_pass_rates, false_fail_rates):
      sufficient_n = st.min_num_samples_for_dkwm_mean_test(
          thresholds, low=0., high=1., false_fail_rate=false_fail_rate,
          false_pass_rate=false_pass_rate)
      detectable_discrepancies.append(
          st.min_discrepancy_of_true_means_detectable_by_dkwm(
              sufficient_n, low=0., high=1., false_fail_rate=false_fail_rate,
              false_pass_rate=false_pass_rate))

    detectable_discrepancies_ = self.evaluate(detectable_discrepancies)
    for discrepancies, false_pass_rate, false_fail_rate in zip(
        detectable_discrepancies_, false_pass_rates, false_fail_rates):
      below_threshold = discrepancies <= thresholds
      self.assertAllEqual(
          np.ones_like(below_threshold, np.bool), below_threshold,
          msg='false_pass_rate({}), false_fail_rate({})'.format(
              false_pass_rate, false_fail_rate))

  def test_dkwm_design_mean_two_sample_soundness(self, dtype):
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
      ] = st.min_num_samples_for_dkwm_mean_two_sample_test(
          thresholds, low1=0., high1=1., low2=0., high2=1.,
          false_fail_rate=false_fail_rate,
          false_pass_rate=false_pass_rate)

      detectable_discrepancies.append(
          st.min_discrepancy_of_true_means_detectable_by_dkwm_two_sample(
              n1=sufficient_n1,
              low1=0.,
              high1=1.,
              n2=sufficient_n2,
              low2=0.,
              high2=1.,
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
    with self.assertRaisesOpError("true mean greater than expected"):
      self.evaluate(st.assert_true_mean_equal_by_dkwm(
          samples, 0., 1., 0.4, false_fail_rate=1e-6))

    # Test that the test assertion confirms that the mean of the
    # standard uniform distribution is not 0.6.
    with self.assertRaisesOpError("true mean smaller than expected"):
      self.evaluate(st.assert_true_mean_equal_by_dkwm(
          samples, 0., 1., 0.6, false_fail_rate=1e-6))

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
    with self.assertRaisesOpError("true mean greater than expected"):
      self.evaluate(st.assert_true_mean_in_interval_by_dkwm(
          samples, 0., 1.,
          expected_low=0.2, expected_high=0.4, false_fail_rate=1e-6))

    # Test that the test assertion confirms that the mean of the
    # standard uniform distribution is not between 0.6 and 0.8.
    with self.assertRaisesOpError("true mean smaller than expected"):
      self.evaluate(st.assert_true_mean_in_interval_by_dkwm(
          samples, 0., 1.,
          expected_low=0.6, expected_high=0.8, false_fail_rate=1e-6))

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

  def test_dkwm_mean_two_sample_assertion_beta_2_1_false(self, dtype):
    rng = np.random.RandomState(seed=0)
    num_samples = 4000
    samples1 = rng.uniform(size=num_samples).astype(dtype=dtype)

    # As established above, 4000 samples is enough to find discrepancies
    # of size 0.2 or more with assurance 1e-6.

    # Test that the test assertion confirms that the mean of the
    # standard uniform distribution is different from the mean of beta(2, 1).
    beta_high_samples = rng.beta(2, 1, size=num_samples).astype(dtype=dtype)
    with self.assertRaisesOpError("true mean smaller than expected"):
      self.evaluate(st.assert_true_mean_equal_by_dkwm_two_sample(
          samples1, 0., 1.,
          beta_high_samples, 0., 1.,
          false_fail_rate=1e-6))

  def test_dkwm_mean_two_sample_assertion_beta_1_2_false(self, dtype):
    rng = np.random.RandomState(seed=0)
    num_samples = 4000
    samples1 = rng.uniform(size=num_samples).astype(dtype=dtype)

    # As established above, 4000 samples is enough to find discrepancies
    # of size 0.2 or more with assurance 1e-6.

    # Test that the test assertion confirms that the mean of the
    # standard uniform distribution is different from the mean of beta(1, 2).
    beta_low_samples = rng.beta(1, 2, size=num_samples).astype(dtype=dtype)
    with self.assertRaisesOpError("true mean greater than expected"):
      self.evaluate(st.assert_true_mean_equal_by_dkwm_two_sample(
          samples1, 0., 1.,
          beta_low_samples, 0., 1.,
          false_fail_rate=1e-6))

  def test_dkwm_argument_validity_checking(self, dtype):
    rng = np.random.RandomState(seed=0)
    samples = rng.uniform(
        low=[0., 1.], high=[1., 2.], size=(2500, 1, 2)).astype(dtype=dtype)

    # Test that the test library complains if the given samples fall
    # outside the purported bounds.
    with self.assertRaisesOpError("maximum value exceeds expectations"):
      self.evaluate(st.true_mean_confidence_interval_by_dkwm(
          samples, [[0., 1.]], [[0.5, 1.5]], error_rate=0.5))
    with self.assertRaisesOpError("minimum value falls below expectations"):
      self.evaluate(st.true_mean_confidence_interval_by_dkwm(
          samples, [[0.5, 1.5]], [[1., 2.]], error_rate=0.5))

    # But doesn't complain if they don't.
    op = st.true_mean_confidence_interval_by_dkwm(
        samples, [[0., 1.]], [[1., 2.]], error_rate=0.5)
    _ = self.evaluate(op)

  def test_do_maximum_mean(self, dtype):
    n = 117
    envelope = 0.02  # > 2 / n, but < 3 / n
    rng = np.random.RandomState(seed=8)
    samples = rng.uniform(size=n).astype(dtype=dtype)

    # Compute the answer in TF using the code under test
    envelope_t = tf.convert_to_tensor(envelope, dtype=dtype)
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


if __name__ == '__main__':
  tf.test.main()
