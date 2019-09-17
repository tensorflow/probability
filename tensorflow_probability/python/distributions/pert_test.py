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
    smoothness = np.array([1., 2., 4., 10.])
    mini = np.array(len(smoothness)*[1.])
    mode = np.array(len(smoothness)*[7.])
    maxi = np.array(len(smoothness)*[10.])
    a = 1. + smoothness * (mode - mini) / (maxi - mini)
    b = 1. + smoothness * (maxi - mode) / (maxi - mini)
    return smoothness, mini, mode, maxi, a, b

  # Shape and broadcast testing
  def testPertShape(self):
    dist = tfd.PERT(mini=[3.0], mode=[10.0], maxi=[11.0], smoothness=[4.0])
    self.assertEqual(([1]), self.evaluate(dist.batch_shape_tensor()))
    self.assertEqual(tf.TensorShape([1]), dist.batch_shape)
    self.assertAllEqual([], self.evaluate(dist.event_shape_tensor()))
    self.assertEqual(tf.TensorShape([]), dist.event_shape)

  def testBroadcastingSmoothness(self):
    dist = tfd.PERT(
        mini=1.,
        mode=2.,
        maxi=3.,
        smoothness=[1., 4., 10.])
    self.assertEqual([3], self.evaluate(dist.batch_shape_tensor()))
    self.assertEqual(tf.TensorShape([3]), dist.batch_shape)
    self.assertAllEqual([], self.evaluate(dist.event_shape_tensor()))
    self.assertEqual(tf.TensorShape([]), dist.event_shape)

  def testBroadcastingParam(self):
    dist = tfd.PERT(
        mini=1.,
        mode=[2., 3., 4., 5., 6., 7., 8., 9.],
        maxi=10.,
        smoothness=4.)
    self.assertEqual([8], self.evaluate(dist.batch_shape_tensor()))
    self.assertEqual(tf.TensorShape([8]), dist.batch_shape)
    self.assertAllEqual([], self.evaluate(dist.event_shape_tensor()))
    self.assertEqual(tf.TensorShape([]), dist.event_shape)

  def testBroadcastingHigherDimParam(self):
    dist = tfd.PERT(
        mini=[1., 2.],
        mode=[[2., 3.], [4., 5.]],
        maxi=10.,
        smoothness=4.,
        validate_args=True)
    self.assertAllEqual([2, 2], self.evaluate(dist.batch_shape_tensor()))
    self.assertEqual(tf.TensorShape([2, 2]), dist.batch_shape)
    self.assertAllEqual([], self.evaluate(dist.event_shape_tensor()))
    self.assertEqual(tf.TensorShape([]), dist.event_shape)

  def testEdgeRangeOutput(self):
    dist = tfd.PERT(mini=3.0, mode=10.0, maxi=11.0, smoothness=4.0)
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
    smoothness, mini, mode, maxi, a, b = self._generate_boilerplate_param()
    dist = tfd.PERT(mini, mode, maxi, smoothness)
    expected_mean = sp_stats.beta.mean(a, b, mini, maxi-mini)
    self.assertAllClose(expected_mean, self.evaluate(dist.mean()))

  def testVariance(self):
    smoothness, mini, mode, maxi, a, b = self._generate_boilerplate_param()
    dist = tfd.PERT(mini, mode, maxi, smoothness)
    expected_var = sp_stats.beta.var(a, b, mini, maxi-mini)
    self.assertAllClose(expected_var, self.evaluate(dist.variance()))

  # Sample testing
  def testSampleStatistics(self):
    n = 1 << 19
    smoothness, mini, mode, maxi, a, b = self._generate_boilerplate_param()
    dist = tfd.PERT(mini, mode, maxi, smoothness)\
              .sample(n, seed=tfp_test_util.test_seed())
    samples = self.evaluate(dist)
    expected_mean = sp_stats.beta.mean(a, b, mini, maxi-mini)
    expected_var = sp_stats.beta.var(a, b, mini, maxi-mini)
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
  def testSmoothnessPositive(self):
    smoothness = tf.Variable(0.)
    mini = [1., 2., 3.]
    mode = [2., 3., 4.]
    maxi = [3., 4., 5.]
    self.evaluate(smoothness.initializer)
    with self.assertRaisesRegex(
        tf.errors.InvalidArgumentError,
        "Smoothness parameter must be positive."):
      dist = tfd.PERT(mini, mode, maxi, smoothness, validate_args=True)
      self.evaluate(dist.sample(1))

  def testSmoothnessPositiveAfterMutation(self):
    smoothness = tf.Variable(4.)
    mini = [1., 2., 3.]
    mode = [2., 3., 4.]
    maxi = [3., 4., 5.]
    self.evaluate(smoothness.initializer)
    dist = tfd.PERT(mini, mode, maxi, smoothness, validate_args=True)
    with self.assertRaisesRegex(
        tf.errors.InvalidArgumentError,
        "Smoothness parameter must be positive."):
      with tf.control_dependencies([smoothness.assign(-1.)]):
        self.evaluate(dist.sample(1))

  def testModeMinInequality(self):
    mini = tf.Variable([1., 2., 3.])
    mode = tf.Variable([1., 2., 3.])
    maxi = [5., 5., 5.]
    self.evaluate(mini.initializer)
    self.evaluate(mode.initializer)
    with self.assertRaisesRegex(
        tf.errors.InvalidArgumentError,
        "Mode must be greater than minimum."):
      dist = tfd.PERT(mini, mode, maxi, validate_args=True)
      self.evaluate(dist.sample(1))

  def testModeMinInequalityAfterMutation(self):
    mini = tf.Variable([1., 2., 3.])
    mode = tf.Variable([2., 3., 4.])
    maxi = [5., 5., 5.]
    self.evaluate(mini.initializer)
    self.evaluate(mode.initializer)
    dist = tfd.PERT(mini, mode, maxi, validate_args=True)
    with self.assertRaisesRegex(
        tf.errors.InvalidArgumentError,
        "Mode must be greater than minimum."):
      with tf.control_dependencies([mode.assign([0., 0., 0.])]):
        self.evaluate(dist.sample(1))

  def testMaxModeInequality(self):
    mini = [1., 2., 3.]
    mode = tf.Variable([2., 3., 4.])
    maxi = tf.Variable([5., 5., 4.])
    self.evaluate(maxi.initializer)
    self.evaluate(mode.initializer)
    with self.assertRaisesRegex(
        tf.errors.InvalidArgumentError,
        "Maximum must be greater than mode."):
      dist = tfd.PERT(mini, mode, maxi, validate_args=True)
      self.evaluate(dist.sample(1))

  def testMaxModeInequalityAfterMutation(self):
    mini = [1., 2., 3.]
    mode = tf.Variable([2., 3., 4.])
    maxi = tf.Variable([5., 5., 5.])
    self.evaluate(maxi.initializer)
    self.evaluate(mode.initializer)
    dist = tfd.PERT(mini, mode, maxi, validate_args=True)
    with self.assertRaisesRegex(
        tf.errors.InvalidArgumentError,
        "Maximum must be greater than mode."):
      with tf.control_dependencies([maxi.assign([0., 0., 0.])]):
        self.evaluate(dist.sample(1))

if __name__ == "__main__":
  tf.test.main()
