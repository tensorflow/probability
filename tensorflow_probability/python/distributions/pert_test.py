# Copyright 2019 The TensorFlow Probability Authors.
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
"""Tests for the PERT distribution."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
import numpy as np
from scipy import stats as sp_stats

import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp

from tensorflow_probability.python.distributions.internal import statistical_testing as st
from tensorflow_probability.python.internal import assert_util
from tensorflow_probability.python.internal import test_util

tfd = tfp.distributions


@test_util.test_all_tf_execution_regimes
class PERTTest(test_util.TestCase):

  def _generate_boilerplate_param(self):
    temperature = np.array([1., 2., 4., 10.])
    low = np.array(len(temperature) * [1.])
    peak = np.array(len(temperature) * [7.])
    high = np.array(len(temperature) * [10.])
    a = 1. + temperature * (peak - low) / (high - low)
    b = 1. + temperature * (high - peak) / (high - low)
    return temperature, low, peak, high, a, b

  # Shape and broadcast testing
  def testPertShape(self):
    dist = tfd.PERT(
        low=[3.0],
        peak=[10.0],
        high=[11.0],
        temperature=[4.0],
        validate_args=True)
    self.assertEqual(([1]), self.evaluate(dist.batch_shape_tensor()))
    self.assertEqual(tf.TensorShape([1]), dist.batch_shape)
    self.assertAllEqual([], self.evaluate(dist.event_shape_tensor()))
    self.assertEqual(tf.TensorShape([]), dist.event_shape)

  def testBroadcastingtemperature(self):
    dist = tfd.PERT(
        low=1., peak=2., high=3., temperature=[1., 4., 10.], validate_args=True)
    self.assertEqual([3], self.evaluate(dist.batch_shape_tensor()))
    self.assertEqual(tf.TensorShape([3]), dist.batch_shape)
    self.assertAllEqual([], self.evaluate(dist.event_shape_tensor()))
    self.assertEqual(tf.TensorShape([]), dist.event_shape)

  def testBroadcastingParam(self):
    dist = tfd.PERT(
        low=1.,
        peak=[2., 3., 4., 5., 6., 7., 8., 9.],
        high=10.,
        temperature=4.,
        validate_args=True)
    self.assertEqual([8], self.evaluate(dist.batch_shape_tensor()))
    self.assertEqual(tf.TensorShape([8]), dist.batch_shape)
    self.assertAllEqual([], self.evaluate(dist.event_shape_tensor()))
    self.assertEqual(tf.TensorShape([]), dist.event_shape)

  def testBroadcastingHigherDimParam(self):
    dist = tfd.PERT(
        low=[1., 2.],
        peak=[[2., 3.], [4., 5.]],
        high=10.,
        temperature=4.,
        validate_args=True)
    self.assertAllEqual([2, 2], self.evaluate(dist.batch_shape_tensor()))
    self.assertEqual(tf.TensorShape([2, 2]), dist.batch_shape)
    self.assertAllEqual([], self.evaluate(dist.event_shape_tensor()))
    self.assertEqual(tf.TensorShape([]), dist.event_shape)

  def testEdgeRangeOutput(self):
    dist = tfd.PERT(
        low=3.0, peak=10.0, high=11.0, temperature=4.0, validate_args=False)
    self.assertEqual(True, self.evaluate(tf.math.is_nan(dist.prob(1.))))
    self.assertEqual(True, self.evaluate(tf.math.is_nan(dist.log_prob(1.))))
    self.assertEqual(1., self.evaluate(dist.cdf(11.)))
    self.assertEqual(0., self.evaluate(dist.cdf(3.)))
    self.assertEqual(1.,
                     self.evaluate(dist.cdf(3.) + dist.survival_function(3.)))
    self.assertEqual(1.,
                     self.evaluate(dist.cdf(11.) + dist.survival_function(11.)))
    self.assertEqual(1.,
                     self.evaluate(dist.cdf(5.) + dist.survival_function(5.)))
    self.assertEqual(-float('inf'), self.evaluate(dist.log_cdf(3.)))
    self.assertEqual(0., self.evaluate(dist.log_cdf(11.)))

  # Statistical property testing
  def testMean(self):
    temperature, low, peak, high, a, b = self._generate_boilerplate_param()
    dist = tfd.PERT(low, peak, high, temperature, validate_args=True)
    expected_mean = sp_stats.beta.mean(a, b, low, high - low)
    self.assertAllClose(expected_mean, self.evaluate(dist.mean()))

  def testVariance(self):
    temperature, low, peak, high, a, b = self._generate_boilerplate_param()
    dist = tfd.PERT(low, peak, high, temperature, validate_args=True)
    expected_var = sp_stats.beta.var(a, b, low, high - low)
    self.assertAllClose(expected_var, self.evaluate(dist.variance()))

  # Sample testing
  def testSampleEmpiricalCDF(self):
    num_samples = 300000
    temperature, low, peak, high = 2., 1., 7., 10.
    dist = tfd.PERT(low, peak, high, temperature, validate_args=True)
    samples = dist.sample(num_samples, seed=test_util.test_seed())

    check_cdf_agrees = st.assert_true_cdf_equal_by_dkwm(
        samples, dist.cdf, false_fail_rate=1e-6)
    check_enough_power = assert_util.assert_less(
        st.min_discrepancy_of_true_cdfs_detectable_by_dkwm(
            num_samples, false_fail_rate=1e-6, false_pass_rate=1e-6), 0.01)
    self.evaluate([check_cdf_agrees, check_enough_power])

  # Parameter restriction testing
  def testTemperaturePositive(self):
    temperature = tf.Variable(0.)
    low = [1., 2., 3.]
    peak = [2., 3., 4.]
    high = [3., 4., 5.]
    self.evaluate(temperature.initializer)
    with self.assertRaisesOpError('`temperature` must be positive.'):
      dist = tfd.PERT(low, peak, high, temperature, validate_args=True)
      self.evaluate(dist.sample(1, seed=test_util.test_seed()))

  def testTemperaturePositiveAfterMutation(self):
    temperature = tf.Variable(4.)
    low = [1., 2., 3.]
    peak = [2., 3., 4.]
    high = [3., 4., 5.]
    self.evaluate(temperature.initializer)
    dist = tfd.PERT(low, peak, high, temperature, validate_args=True)
    with self.assertRaisesOpError('`temperature` must be positive.'):
      with tf.control_dependencies([temperature.assign(-1.)]):
        self.evaluate(dist.sample(1, seed=test_util.test_seed()))

  def testPeakLowInequality(self):
    low = tf.Variable([1., 2., 3.])
    peak = tf.Variable([1., 2., 3.])
    high = [5., 5., 5.]
    self.evaluate(low.initializer)
    self.evaluate(peak.initializer)
    with self.assertRaisesOpError('`peak` must be greater than `low`.'):
      dist = tfd.PERT(low, peak, high, validate_args=True)
      self.evaluate(dist.sample(1, seed=test_util.test_seed()))

  def testPeakLowInequalityAfterMutation(self):
    low = tf.Variable([1., 2., 3.])
    peak = tf.Variable([2., 3., 4.])
    high = [5., 5., 5.]
    self.evaluate(low.initializer)
    self.evaluate(peak.initializer)
    dist = tfd.PERT(low, peak, high, validate_args=True)
    with self.assertRaisesOpError('`peak` must be greater than `low`.'):
      with tf.control_dependencies([peak.assign([0., 0., 0.])]):
        self.evaluate(dist.sample(1, seed=test_util.test_seed()))

  def testHighPeakInequality(self):
    low = [1., 2., 3.]
    peak = tf.Variable([2., 3., 4.])
    high = tf.Variable([5., 5., 4.])
    self.evaluate(high.initializer)
    self.evaluate(peak.initializer)
    with self.assertRaisesOpError('`high` must be greater than `peak`.'):
      dist = tfd.PERT(low, peak, high, validate_args=True)
      self.evaluate(dist.sample(1, seed=test_util.test_seed()))

  def testHighPeakInequalityAfterMutation(self):
    low = [1., 2., 3.]
    peak = tf.Variable([2., 3., 4.])
    high = tf.Variable([5., 5., 5.])
    self.evaluate(high.initializer)
    self.evaluate(peak.initializer)
    dist = tfd.PERT(low, peak, high, validate_args=True)
    with self.assertRaisesOpError('`high` must be greater than `peak`.'):
      with tf.control_dependencies([high.assign([0., 0., 0.])]):
        self.evaluate(dist.sample(1, seed=test_util.test_seed()))

  def testAssertValidSample(self):
    low = [1., 2., 3.]
    peak = [2., 3., 4.]
    high = [3., 4., 5.]
    dist = tfd.PERT(low, peak, high, validate_args=True)
    with self.assertRaisesOpError('must be greater than or equal to `low`.'):
      self.evaluate(dist.prob([1.3, 1., 3.5]))
    with self.assertRaisesOpError('must be less than or equal to `high`.'):
      self.evaluate(dist.prob([2.1, 3.2, 5.2]))

  def testPdfAtBoundary(self):
    low = [1., 2., 3.]
    peak = [2., 3., 4.]
    high = [3., 4., 5.]
    dist = tfd.PERT(low, peak, high, validate_args=True)
    pdf = self.evaluate(dist.prob([low, high]))
    log_pdf = self.evaluate(dist.log_prob([low, high]))
    self.assertAllEqual(pdf, np.zeros_like(pdf))
    self.assertAllNegativeInf(log_pdf)

  def testSupportBijectorOutsideRange(self):
    low = np.array([1., 2., 3.])
    peak = np.array([4., 4., 4.])
    high = np.array([6., 7., 6.])
    dist = tfd.PERT(low, peak, high, validate_args=True)
    eps = 1e-6
    x = np.array([1. - eps, 1.5, 6. + eps])
    bijector_inverse_x = dist._experimental_default_event_space_bijector(
        ).inverse(x)
    self.assertAllNan(self.evaluate(bijector_inverse_x))

if __name__ == '__main__':
  tf.test.main()
