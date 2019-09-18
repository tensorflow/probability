"""Tests for PERT distribution."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


# Dependency imports
import numpy as np
from scipy import stats as sp_stats

import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp

from tensorflow_probability.python.internal import test_case
from tensorflow_probability.python.internal import test_util as tfp_test_util
from tensorflow.python.framework import test_util  # pylint: disable=g-direct-tensorflow-import

tfd = tfp.distributions

@test_util.run_all_in_graph_and_eager_modes
class PERTTest(test_case.TestCase):
  def _generate_boilerplate_param(self):
    temperature = np.array([1., 2., 4., 10.])
    low = np.array(len(temperature)*[1.])
    peak = np.array(len(temperature)*[7.])
    high = np.array(len(temperature)*[10.])
    a = 1. + temperature * (peak - low) / (high - low)
    b = 1. + temperature * (high - peak) / (high - low)
    return temperature, low, peak, high, a, b

  # Shape and broadcast testing
  def testPertShape(self):
    dist = tfd.PERT(low=[3.0], peak=[10.0], high=[11.0], temperature=[4.0])
    self.assertEqual(([1]), self.evaluate(dist.batch_shape_tensor()))
    self.assertEqual(tf.TensorShape([1]), dist.batch_shape)
    self.assertAllEqual([], self.evaluate(dist.event_shape_tensor()))
    self.assertEqual(tf.TensorShape([]), dist.event_shape)

  def testBroadcastingtemperature(self):
    dist = tfd.PERT(
        low=1.,
        peak=2.,
        high=3.,
        temperature=[1., 4., 10.])
    self.assertEqual([3], self.evaluate(dist.batch_shape_tensor()))
    self.assertEqual(tf.TensorShape([3]), dist.batch_shape)
    self.assertAllEqual([], self.evaluate(dist.event_shape_tensor()))
    self.assertEqual(tf.TensorShape([]), dist.event_shape)

  def testBroadcastingParam(self):
    dist = tfd.PERT(
        low=1.,
        peak=[2., 3., 4., 5., 6., 7., 8., 9.],
        high=10.,
        temperature=4.)
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
    dist = tfd.PERT(low=3.0, peak=10.0, high=11.0, temperature=4.0)
    self.assertEqual(True, self.evaluate(tf.math.is_nan(dist.prob(1.))))
    self.assertEqual(True, self.evaluate(tf.math.is_nan(dist.log_prob(1.))))
    self.assertEqual(1., self.evaluate(dist.cdf(11.)))
    self.assertEqual(0., self.evaluate(dist.cdf(3.)))
    self.assertEqual(
        1.,
        self.evaluate(dist.cdf(3.) + dist.survival_function(3.)))
    self.assertEqual(
        1.,
        self.evaluate(dist.cdf(11.) + dist.survival_function(11.)))
    self.assertEqual(
        1.,
        self.evaluate(dist.cdf(5.) + dist.survival_function(5.)))
    self.assertEqual(-float("inf"), self.evaluate(dist.log_cdf(3.)))
    self.assertEqual(0., self.evaluate(dist.log_cdf(11.)))


  # Statistical property testing
  def testMean(self):
    temperature, low, peak, high, a, b = self._generate_boilerplate_param()
    dist = tfd.PERT(low, peak, high, temperature)
    expected_mean = sp_stats.beta.mean(a, b, low, high-low)
    self.assertAllClose(expected_mean, self.evaluate(dist.mean()))

  def testVariance(self):
    temperature, low, peak, high, a, b = self._generate_boilerplate_param()
    dist = tfd.PERT(low, peak, high, temperature)
    expected_var = sp_stats.beta.var(a, b, low, high-low)
    self.assertAllClose(expected_var, self.evaluate(dist.variance()))

  # Sample testing
  def testSampleStatistics(self):
    n = 1 << 19
    temperature, low, peak, high, a, b = self._generate_boilerplate_param()
    dist = (tfd.PERT(low, peak, high, temperature)
                  .sample(n, seed=tfp_test_util.test_seed()))
    samples = self.evaluate(dist)
    expected_mean = sp_stats.beta.mean(a, b, low, high-low)
    expected_var = sp_stats.beta.var(a, b, low, high-low)
    self.assertAllClose(
        samples.mean(axis=0),
        expected_mean,
        rtol=0.01,
        atol=0.01,
        msg="Sample mean is highly off of expected.")
    self.assertAllClose(
        samples.var(axis=0),
        expected_var,
        rtol=0.01,
        atol=0.01,
        msg="Sample variance is highly off of expected.")

  # Parameter restriction testing
  def testtemperaturePositive(self):
    temperature = tf.Variable(0.)
    low = [1., 2., 3.]
    peak = [2., 3., 4.]
    high = [3., 4., 5.]
    self.evaluate(temperature.initializer)
    with self.assertRaisesRegex(
        tf.errors.InvalidArgumentError,
        "`temperature` must be positive."):
      dist = tfd.PERT(low, peak, high, temperature, validate_args=True)
      self.evaluate(dist.sample(1))

  def testtemperaturePositiveAfterMutation(self):
    temperature = tf.Variable(4.)
    low = [1., 2., 3.]
    peak = [2., 3., 4.]
    high = [3., 4., 5.]
    self.evaluate(temperature.initializer)
    dist = tfd.PERT(low, peak, high, temperature, validate_args=True)
    with self.assertRaisesRegex(
        tf.errors.InvalidArgumentError,
        "`temperature` must be positive."):
      with tf.control_dependencies([temperature.assign(-1.)]):
        self.evaluate(dist.sample(1))

  def testpeaklownequality(self):
    low = tf.Variable([1., 2., 3.])
    peak = tf.Variable([1., 2., 3.])
    high = [5., 5., 5.]
    self.evaluate(low.initializer)
    self.evaluate(peak.initializer)
    with self.assertRaisesRegex(
        tf.errors.InvalidArgumentError,
        "`peak` must be greater than `low`."):
      dist = tfd.PERT(low, peak, high, validate_args=True)
      self.evaluate(dist.sample(1))

  def testpeaklownequalityAfterMutation(self):
    low = tf.Variable([1., 2., 3.])
    peak = tf.Variable([2., 3., 4.])
    high = [5., 5., 5.]
    self.evaluate(low.initializer)
    self.evaluate(peak.initializer)
    dist = tfd.PERT(low, peak, high, validate_args=True)
    with self.assertRaisesRegex(
        tf.errors.InvalidArgumentError,
        "`peak` must be greater than `low`."):
      with tf.control_dependencies([peak.assign([0., 0., 0.])]):
        self.evaluate(dist.sample(1))

  def testMaxpeakInequality(self):
    low = [1., 2., 3.]
    peak = tf.Variable([2., 3., 4.])
    high = tf.Variable([5., 5., 4.])
    self.evaluate(high.initializer)
    self.evaluate(peak.initializer)
    with self.assertRaisesRegex(
        tf.errors.InvalidArgumentError,
        "`high` must be greater than `peak`."):
      dist = tfd.PERT(low, peak, high, validate_args=True)
      self.evaluate(dist.sample(1))

  def testMaxpeakInequalityAfterMutation(self):
    low = [1., 2., 3.]
    peak = tf.Variable([2., 3., 4.])
    high = tf.Variable([5., 5., 5.])
    self.evaluate(high.initializer)
    self.evaluate(peak.initializer)
    dist = tfd.PERT(low, peak, high, validate_args=True)
    with self.assertRaisesRegex(
        tf.errors.InvalidArgumentError,
        "`high` must be greater than `peak`."):
      with tf.control_dependencies([high.assign([0., 0., 0.])]):
        self.evaluate(dist.sample(1))

if __name__ == "__main__":
  tf.test.main()
