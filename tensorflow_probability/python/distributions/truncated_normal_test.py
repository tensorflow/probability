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
"""Tests for TruncatedNormal distribution."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import importlib
import itertools

# Dependency imports
from absl.testing import parameterized
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp


def try_import(name):  # pylint: disable=invalid-name
  module = None
  try:
    module = importlib.import_module(name)
  except ImportError as e:
    tf.logging.warning("Could not import %s: %s" % (name, str(e)))
  return module

stats = try_import("scipy.stats")

tfd = tfp.distributions

EPSILON = 1e-5


def scipy_trunc_norm_dist(loc, scale, low, high):
  """Construct a scipy.stats.truncnorm for the (scalar) parameters given.

  Note: scipy's definition of the parameters is slightly different.
  https://github.com/scipy/scipy/issues/7591

  Args:
    loc: Params describing distribution (doesn't support batch)
    scale:
    low:
    high:

  Returns:
    scipy frozen distribution.
  """
  a = (low - loc) / scale
  b = (high - loc) / scale
  return stats.truncnorm(a, b, loc=loc, scale=scale)


class _TruncatedNormalTestCase(tf.test.TestCase):

  def setUp(self):
    self._rng = np.random.RandomState(42)

  def assertAllFinite(self, a):
    is_finite = np.isfinite(a)
    all_true = np.ones_like(is_finite, dtype=np.bool)
    self.assertAllEqual(all_true, is_finite)

  def assertAllGreaterEqual(self, a, b):
    comparison = a >= b
    all_true = np.ones_like(comparison, dtype=np.bool)
    self.assertAllEqual(comparison, all_true)

  def assertAllLessEqual(self, a, b):
    comparison = a <= b
    all_true = np.ones_like(comparison, dtype=np.bool)
    self.assertAllEqual(comparison, all_true)

  def assertEmpiricalDistributionsEqual(self, sample_a, sample_b, rtol=1e-6,
                                        atol=1e-6):
    """Assert the empirical distribution of two set of samples is similar.

    Args:
      sample_a: Flat numpy array of samples from dist a.
      sample_b: Flat numpy array of samples from dist b.
      rtol: Relative tolerances in the histogram comparison.
      atol: Absolute tolerances in the histogram comparison.
    """
    self.assertAllFinite(sample_a)
    self.assertAllFinite(sample_b)

    lb = np.min([sample_a, sample_b])
    ub = np.max([sample_a, sample_b])

    hist_a = np.histogram(sample_a, range=(lb, ub), bins=30, density=True)[0]
    hist_b = np.histogram(sample_b, range=(lb, ub), bins=30, density=True)[0]

    self.assertAllClose(hist_a, hist_b, rtol=rtol, atol=atol)


class TruncatedNormalStandaloneTestCase(_TruncatedNormalTestCase,
                                        parameterized.TestCase):

  def _testParamShapes(self, desired_shape):
    with self.test_session():
      tn_param_shapes = tfd.TruncatedNormal.param_shapes(desired_shape)
      # Check the shapes by comparison with the untruncated Normal.
      n_param_shapes = tf.distributions.Normal.param_shapes(desired_shape)
      self.assertAllEqual(tn_param_shapes["loc"].eval(),
                          n_param_shapes["loc"].eval())
      self.assertAllEqual(tn_param_shapes["scale"].eval(),
                          n_param_shapes["scale"].eval())
      self.assertAllEqual(tn_param_shapes["low"].eval(),
                          n_param_shapes["loc"].eval())
      self.assertAllEqual(tn_param_shapes["high"].eval(),
                          n_param_shapes["loc"].eval())

      loc = tf.zeros(tn_param_shapes["loc"])
      scale = tf.ones(tn_param_shapes["scale"])
      high = tf.ones(tn_param_shapes["high"])
      low = tf.ones(tn_param_shapes["low"])
      sample_shape = tf.shape(tfd.TruncatedNormal(
          loc=loc, scale=scale, low=low,
          high=high).sample()).eval()
      self.assertAllEqual(desired_shape, sample_shape)

  def testParamShapes(self):
    desired_shape = [10, 3, 4]
    self._testParamShapes(desired_shape)
    self._testParamShapes(tf.constant(desired_shape))

  def testParamStaticShapes(self):
    sample_shape = [7]
    self._testParamShapes(sample_shape)
    self._testParamShapes(tf.TensorShape(sample_shape))

  def testShapeWithPlaceholders(self):
    loc = tf.placeholder(dtype=tf.float32)
    scale = tf.placeholder(dtype=tf.float32)
    ub = tf.placeholder(dtype=tf.float32)
    lb = tf.placeholder(dtype=tf.float32)
    dist = tfd.TruncatedNormal(loc, scale, lb, ub)

    with self.test_session() as sess:
      self.assertEqual(dist.batch_shape, tf.TensorShape(None))
      self.assertEqual(dist.event_shape, ())
      self.assertAllEqual(dist.event_shape_tensor().eval(), [])
      self.assertAllEqual(
          sess.run(
              dist.batch_shape_tensor(),
              feed_dict={
                  loc: 5.0,
                  scale: [1.0, 2.0],
                  lb: [-1.0],
                  ub: [10.0, 11.0],
              }), [2])
    self.assertAllEqual(
        sess.run(
            dist.sample(5),
            feed_dict={
                loc: 0.0,
                scale: 1.0,
                lb: -1.0,
                ub: [[1.0, 2.0]]
            }
        ).shape,
        [5, 1, 2]
    )

  def testBatchSampling(self):
    """Check (empirically) the different parameters in a batch are respected.
    """
    with self.test_session():
      n = int(1e5)
      lb = [[-1.0, 9.0], [0., 8.]]
      ub = [[1.0, 11.0], [5., 20.]]
      dist = tfd.TruncatedNormal(loc=[[0., 10.], [0., 10.]],
                                 scale=[[1., 1.], [5., 5.]],
                                 low=lb, high=ub)
      x = dist.sample(n, seed=4).eval()
      self.assertEqual(x.shape, (n, 2, 2))

      means = np.mean(x, axis=0)
      var = np.var(x, axis=0)
      self.assertAllClose(means, [[0., 10.], [2.299, 12.48]],
                          rtol=1e-2, atol=1e-2)
      self.assertAllClose(var, [[0.29, 0.29], [1.99, 8.74]],
                          rtol=1e-2, atol=1e-2)

      empirical_lb = np.min(x, axis=0)
      self.assertAllClose(empirical_lb, lb, atol=0.1)
      empirical_ub = np.max(x, axis=0)
      self.assertAllClose(empirical_ub, ub, atol=0.1)

  @parameterized.parameters(
      (0., 1., -1., 1.),
      (1., 1., 0., 2.),
      (-0.5, 0.5, -0.9, -0.4),
      (10., 3.0, 9.9, 25.),
      (2., 1.5, 0.1, 1.9))
  def testMomentsEmpirically(self, loc, scale, low, high):
    with self.test_session():
      n = int(2e5)
      dist = tfd.TruncatedNormal(loc=loc, scale=scale,
                                 low=low,
                                 high=high)
      x = dist.sample(n, seed=1).eval()
      empirical_mean = np.mean(x)
      empirical_var = np.var(x)
      expected_mean = dist.mean().eval()
      expected_var = dist.variance().eval()
      self.assertAlmostEqual(expected_mean, empirical_mean, places=1)
      self.assertAlmostEqual(expected_var, empirical_var, places=1)

  def testNegativeSigmaFails(self):
    with self.test_session():
      dist = tfd.TruncatedNormal(loc=0., scale=-0.1, low=-1.0,
                                 high=1.0, validate_args=True)
      with self.assertRaisesOpError("Condition x > 0 did not hold"):
        dist.mean().eval()

  def testIncorrectBoundsFails(self):
    with self.test_session():
      dist = tfd.TruncatedNormal(loc=0., scale=-0.1, low=1.0,
                                 high=-1.0, validate_args=True)
      with self.assertRaisesOpError("Condition x > 0 did not hold"):
        dist.mean().eval()

      dist = tfd.TruncatedNormal(loc=0., scale=-0.1, low=1.0,
                                 high=1.0, validate_args=True)
      with self.assertRaisesOpError("Condition x > 0 did not hold"):
        dist.mean().eval()

  @parameterized.parameters(
      (0., 1., -1., 1.),
      (1., 1., 0., 2.),
      (-0.5, 0.5, -0.9, -0.4),
      (10., 3.0, 9.9, 25.),
      (2., 1.5, 0.1, 1.9),
      (-2., 0.2, -1.5, -0.5))
  def testMode(self, loc, scale, low, high):
    with self.test_session():
      dist = tfd.TruncatedNormal(loc=loc, scale=scale,
                                 low=low,
                                 high=high)
      mode = np.asscalar(dist.mode().eval())
      if loc < low:
        expected_mode = low
      elif loc > high:
        expected_mode = high
      else:
        expected_mode = loc
      self.assertAlmostEqual(mode, expected_mode)

  @parameterized.parameters((np.float32), (np.float64))
  def testReparametrizable(self, dtype=np.float32):
    loc = tf.Variable(dtype(0.1))
    scale = tf.Variable(dtype(1.1))
    low = tf.Variable(dtype(-10.0))
    high = tf.Variable(dtype(5.0))
    dist = tfd.TruncatedNormal(loc=loc, scale=scale,
                               low=low,
                               high=high)

    with self.test_session() as sess:
      n = int(2e5)
      sess.run(tf.global_variables_initializer())
      empirical_abs_mean = tf.reduce_mean(tf.abs(dist.sample(n, seed=6)))

      loc_err = tf.test.compute_gradient_error(
          loc, loc.shape, empirical_abs_mean, [1], x_init_value=loc.eval(),
          init_targets=[tf.global_variables_initializer()], delta=0.1)
      scale_err = tf.test.compute_gradient_error(
          scale, scale.shape, empirical_abs_mean, [1],
          x_init_value=scale.eval(),
          init_targets=[tf.global_variables_initializer()], delta=0.1)
      low_err = tf.test.compute_gradient_error(
          low, low.shape, empirical_abs_mean, [1],
          x_init_value=low.eval(),
          init_targets=[tf.global_variables_initializer()], delta=0.1)
      high_err = tf.test.compute_gradient_error(
          high, high.shape, empirical_abs_mean, [1],
          x_init_value=high.eval(),
          init_targets=[tf.global_variables_initializer()], delta=0.1)
      # These gradients are noisy due to sampling.
      self.assertLess(loc_err, 0.05)
      self.assertLess(scale_err, 0.05)
      self.assertLess(low_err, 0.05)
      self.assertLess(high_err, 0.05)

  @parameterized.parameters(
      itertools.product((np.float32, np.float64),
                        ("prob", "log_prob", "cdf", "log_cdf",
                         "survival_function", "log_survival_function"))
  )
  def testGradientsFx(self, dtype, fn_name):
    loc = tf.Variable(dtype(0.1))
    scale = tf.Variable(dtype(3.0))
    low = tf.Variable(dtype(-10.0))
    high = tf.Variable(dtype(5.0))
    dist = tfd.TruncatedNormal(loc=loc, scale=scale,
                               low=low,
                               high=high)
    x = np.array([-1.0, 0.01, 0.1, 1., 4.9]).astype(dtype)
    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      func = getattr(dist, fn_name)
      mean_value = tf.reduce_mean(func(x))
      loc_err = tf.test.compute_gradient_error(
          loc, loc.shape, mean_value, [1], x_init_value=loc.eval(),
          init_targets=[tf.global_variables_initializer()])
      scale_err = tf.test.compute_gradient_error(
          scale, scale.shape, mean_value, [1], x_init_value=scale.eval(),
          init_targets=[tf.global_variables_initializer()])
      self.assertLess(loc_err, 1e-2)
      self.assertLess(scale_err, 1e-2)

  @parameterized.parameters(
      itertools.product((np.float32, np.float64),
                        ("entropy", "mean", "variance", "mode"))
  )
  def testGradientsNx(self, dtype, fn_name):
    loc = tf.Variable(dtype(0.1))
    scale = tf.Variable(dtype(3.0))
    low = tf.Variable(dtype(-10.0))
    high = tf.Variable(dtype(5.0))
    dist = tfd.TruncatedNormal(loc=loc, scale=scale,
                               low=low,
                               high=high)
    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      func = getattr(dist, fn_name)
      v = func()
      loc_err = tf.test.compute_gradient_error(
          loc, loc.shape, v, [1], x_init_value=loc.eval(),
          init_targets=[tf.global_variables_initializer()])
      self.assertLess(loc_err, 0.005)

      if fn_name not in ["mode"]:
        scale_err = tf.test.compute_gradient_error(
            scale, scale.shape, v, [1], x_init_value=scale.eval(),
            init_targets=[tf.global_variables_initializer()])
        self.assertLess(scale_err, 0.01)


@parameterized.parameters(
    (0.0, 1.0),
    (10.0, 1.0),
    (-0.3, 2.0),
    (100., 5.0),
    )
class TruncatedNormalTestCompareWithNormal(_TruncatedNormalTestCase,
                                           parameterized.TestCase):
  """Test by comparing TruncatedNormals with wide bounds and unbounded Normal.
  """

  def constructDists(self, loc, scale):
    truncated_dist = tfd.TruncatedNormal(loc=loc, scale=scale,
                                         low=loc - (10. * scale),
                                         high=loc + (10. * scale))
    normal_dist = tf.distributions.Normal(loc=loc, scale=scale)
    return truncated_dist, normal_dist

  def testEntropy(self, loc, scale):
    with self.test_session():
      truncated_dist, normal_dist = self.constructDists(loc, scale)
      self.assertEqual(truncated_dist.entropy().eval(),
                       normal_dist.entropy().eval())

  def testSampling(self, loc, scale):
    with self.test_session():
      n = 1000000
      truncated_dist, normal_dist = self.constructDists(loc, scale)
      truncated_samples = truncated_dist.sample(n).eval().flatten()
      lb = truncated_dist.low.eval()
      ub = truncated_dist.high.eval()
      self.assertAllGreaterEqual(truncated_samples, lb)
      self.assertAllLessEqual(truncated_samples, ub)

      normal_samples = normal_dist.sample(n, seed=2).eval().flatten()
      # Rejection sample the normal distribution
      rejection_samples = normal_samples[normal_samples >= lb]
      rejection_samples = rejection_samples[rejection_samples <= ub]

      self.assertEmpiricalDistributionsEqual(truncated_samples,
                                             rejection_samples,
                                             rtol=1e-2, atol=1e-1)

  def testLogProb(self, loc, scale):
    with self.test_session():
      truncated_dist, normal_dist = self.constructDists(loc, scale)
      low = truncated_dist.low.eval()
      high = truncated_dist.high.eval()
      test_x = list(np.random.uniform(low, high, 10))
      test_x += [low, high, low + EPSILON,
                 high - EPSILON]
      tr_log_prob = truncated_dist.log_prob(test_x).eval()
      n_log_prob = normal_dist.log_prob(test_x).eval()
      self.assertAllClose(tr_log_prob, n_log_prob, rtol=1e-4, atol=1e-4)

      no_support_log_prob = truncated_dist.log_prob(
          [low - EPSILON, high + EPSILON,
           low - 100., high + 100.]).eval()
      self.assertAllEqual(no_support_log_prob,
                          [np.log(0.)] * len(no_support_log_prob))

  def testCDF(self, loc, scale):
    with self.test_session():
      truncated_dist, normal_dist = self.constructDists(loc, scale)
      low = truncated_dist.low.eval()
      high = truncated_dist.high.eval()
      test_x = list(np.random.uniform(low, high, 10))
      test_x += [low, high, low + EPSILON,
                 high - EPSILON]
      tr_cdf = truncated_dist.cdf(test_x).eval()
      n_cdf = normal_dist.cdf(test_x).eval()
      self.assertAllClose(tr_cdf, n_cdf, rtol=1e-4, atol=1e-4)


if stats:
  @parameterized.parameters(
      (0., 1., -1., 1.),
      (1., 1., 0., 2.),
      (-0.5, 0.5, -0.9, -0.4),
      (10., 3.0, 9.9, 25.),
      (2., 1.5, 0.1, 1.9),
      (-2., 0.2, -1.5, -0.5))
  class TruncatedNormalTestCompareWithScipy(_TruncatedNormalTestCase,
                                            parameterized.TestCase):

    def constructDists(self, loc, scale, low, high):
      tf_dist = tfd.TruncatedNormal(loc=loc, scale=scale,
                                    low=low,
                                    high=high)
      sp_dist = scipy_trunc_norm_dist(loc, scale, low, high)
      return tf_dist, sp_dist

    def testSampling(self, loc, scale, low, high):
      with self.test_session():
        n = int(1000000)
        tf_dist, sp_dist = self.constructDists(loc, scale,
                                               low, high)
        tf_samples = tf_dist.sample(n, seed=3).eval().flatten()
        self.assertAllGreaterEqual(tf_samples, low)
        self.assertAllLessEqual(tf_samples, high)

        sp_samples = sp_dist.rvs(size=n)
        self.assertEmpiricalDistributionsEqual(tf_samples, sp_samples,
                                               atol=0.05, rtol=0.05)

    def testEntropy(self, loc, scale, low, high):
      with self.test_session():
        tf_dist, sp_dist = self.constructDists(loc, scale,
                                               low, high)
        self.assertAlmostEqual(tf_dist.entropy().eval(), sp_dist.entropy(),
                               places=2)

    def testLogProb(self, loc, scale, low, high):
      with self.test_session():
        test_x = list(np.random.uniform(low, high, 10))
        test_x += [low, high, low + EPSILON,
                   low - EPSILON, high + EPSILON,
                   high - EPSILON]

        tf_dist, sp_dist = self.constructDists(loc, scale,
                                               low, high)

        tf_log_prob = tf_dist.log_prob(test_x).eval()
        sp_log_prob = sp_dist.logpdf(test_x)
        self.assertAllClose(tf_log_prob, sp_log_prob, rtol=1e-4, atol=1e-4)

    def testCDF(self, loc, scale, low, high):
      with self.test_session():
        test_x = list(np.random.uniform(low, high, 10))
        test_x += [low, high, low + EPSILON,
                   low - EPSILON, high + EPSILON,
                   high - EPSILON, low - 100.,
                   high + 100.]

        tf_dist, sp_dist = self.constructDists(loc, scale,
                                               low, high)

        tf_cdf = tf_dist.cdf(test_x).eval()
        sp_cdf = sp_dist.cdf(test_x)
        self.assertAllClose(tf_cdf, sp_cdf, rtol=1e-4, atol=1e-4)

    def testMoments(self, loc, scale, low, high):
      with self.test_session():
        tf_dist, sp_dist = self.constructDists(loc, scale,
                                               low, high)
        self.assertAlmostEqual(tf_dist.mean().eval(), sp_dist.mean(), places=3)
        self.assertAlmostEqual(tf_dist.variance().eval(), sp_dist.var(),
                               places=3)


if __name__ == "__main__":
  tf.test.main()
