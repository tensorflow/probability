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

from tensorflow_probability.python.internal import test_case
from tensorflow_probability.python.internal import test_util as tfp_test_util
from tensorflow.python.framework import test_util  # pylint: disable=g-direct-tensorflow-import,g-import-not-at-top
from tensorflow.python.ops import gradient_checker_v2  # pylint: disable=g-direct-tensorflow-import


def try_import(name):  # pylint: disable=invalid-name
  module = None
  try:
    module = importlib.import_module(name)
  except ImportError as e:
    tf.compat.v1.logging.warning("Could not import %s: %s" % (name, str(e)))
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


class _TruncatedNormalTestCase(test_case.TestCase):

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


@test_util.run_all_in_graph_and_eager_modes
class TruncatedNormalStandaloneTestCase(_TruncatedNormalTestCase,
                                        parameterized.TestCase):

  def _testParamShapes(self, desired_shape):
    tn_param_shapes = tfd.TruncatedNormal.param_shapes(desired_shape)
    # Check the shapes by comparison with the untruncated Normal.
    n_param_shapes = tfd.Normal.param_shapes(desired_shape)
    self.assertAllEqual(
        self.evaluate(tn_param_shapes["loc"]),
        self.evaluate(n_param_shapes["loc"]))
    self.assertAllEqual(
        self.evaluate(tn_param_shapes["scale"]),
        self.evaluate(n_param_shapes["scale"]))
    self.assertAllEqual(
        self.evaluate(tn_param_shapes["low"]),
        self.evaluate(n_param_shapes["loc"]))
    self.assertAllEqual(
        self.evaluate(tn_param_shapes["high"]),
        self.evaluate(n_param_shapes["loc"]))

    loc = tf.zeros(tn_param_shapes["loc"])
    scale = tf.ones(tn_param_shapes["scale"])
    high = tf.ones(tn_param_shapes["high"])
    low = tf.zeros(tn_param_shapes["low"])
    sample_shape = self.evaluate(
        tf.shape(
            input=tfd.TruncatedNormal(loc=loc, scale=scale, low=low,
                                      high=high).sample()))
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
    if tf.executing_eagerly():
      return
    loc = tf.compat.v1.placeholder_with_default(input=5., shape=None)
    scale = tf.compat.v1.placeholder_with_default(input=[1., 2], shape=None)
    ub = tf.compat.v1.placeholder_with_default(input=[10., 11.], shape=None)
    lb = tf.compat.v1.placeholder_with_default(input=[-1.], shape=None)
    dist = tfd.TruncatedNormal(loc, scale, lb, ub)

    self.assertEqual(dist.batch_shape, tf.TensorShape(None))
    self.assertEqual(dist.event_shape, ())
    self.assertAllEqual(self.evaluate(dist.event_shape_tensor()), [])
    self.assertAllEqual(self.evaluate(dist.batch_shape_tensor()), [2])
    self.assertAllEqual(self.evaluate(dist.sample(5)).shape, [5, 2])

    ub = tf.compat.v1.placeholder_with_default(input=[[5., 11.]], shape=None)
    dist = tfd.TruncatedNormal(loc, scale, lb, ub)
    self.assertAllEqual(self.evaluate(dist.sample(5)).shape, [5, 1, 2])

  def testBatchSampling(self):
    """Check (empirically) the different parameters in a batch are respected.
    """
    n = int(1e5)
    lb = [[-1.0, 9.0], [0., 8.]]
    ub = [[1.0, 11.0], [5., 20.]]
    dist = tfd.TruncatedNormal(
        loc=[[0., 10.], [0., 10.]], scale=[[1., 1.], [5., 5.]], low=lb, high=ub)
    x = self.evaluate(dist.sample(n, seed=tfp_test_util.test_seed()))
    self.assertEqual(x.shape, (n, 2, 2))

    means = np.mean(x, axis=0)
    var = np.var(x, axis=0)
    self.assertAllClose(
        means, [[0., 10.], [2.299, 12.48]], rtol=1e-2, atol=1e-2)
    self.assertAllClose(var, [[0.29, 0.29], [1.99, 8.74]], rtol=1e-2, atol=1e-2)

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
    n = int(2e5)
    dist = tfd.TruncatedNormal(loc=loc, scale=scale, low=low, high=high)
    x = self.evaluate(dist.sample(n, seed=tfp_test_util.test_seed()))
    empirical_mean = np.mean(x)
    empirical_var = np.var(x)
    expected_mean = self.evaluate(dist.mean())
    expected_var = self.evaluate(dist.variance())
    self.assertAlmostEqual(expected_mean, empirical_mean, places=1)
    self.assertAlmostEqual(expected_var, empirical_var, places=1)

  def testNegativeSigmaFails(self):
    with self.assertRaisesOpError("Condition x > 0 did not hold"):
      dist = tfd.TruncatedNormal(
          loc=0., scale=-0.1, low=-1.0, high=1.0, validate_args=True)
      self.evaluate(dist.mean())

  def testIncorrectBoundsFails(self):
    with self.assertRaisesOpError("Condition x > 0 did not hold"):
      dist = tfd.TruncatedNormal(
          loc=0., scale=-0.1, low=1.0, high=-1.0, validate_args=True)
      self.evaluate(dist.mean())

    with self.assertRaisesOpError("Condition x > 0 did not hold"):
      dist = tfd.TruncatedNormal(
          loc=0., scale=-0.1, low=1.0, high=1.0, validate_args=True)
      self.evaluate(dist.mean())

  @parameterized.parameters(
      (0., 1., -1., 1.),
      (1., 1., 0., 2.),
      (-0.5, 0.5, -0.9, -0.4),
      (10., 3.0, 9.9, 25.),
      (2., 1.5, 0.1, 1.9),
      (-2., 0.2, -1.5, -0.5))
  def testMode(self, loc, scale, low, high):
    dist = tfd.TruncatedNormal(loc=loc, scale=scale, low=low, high=high)
    mode = np.asscalar(self.evaluate(dist.mode()))
    if loc < low:
      expected_mode = low
    elif loc > high:
      expected_mode = high
    else:
      expected_mode = loc
    self.assertAlmostEqual(mode, expected_mode)

  @parameterized.parameters((np.float32), (np.float64))
  def testReparametrizable(self, dtype=np.float32):
    loc = tf.compat.v2.Variable(dtype(0.1))
    scale = tf.compat.v2.Variable(dtype(1.1))
    low = tf.compat.v2.Variable(dtype(-10.0))
    high = tf.compat.v2.Variable(dtype(5.0))
    self.evaluate(tf.compat.v1.global_variables_initializer())

    def f(loc, scale, low, high):
      dist = tfd.TruncatedNormal(loc=loc, scale=scale, low=low, high=high)

      n = int(2e5)
      return tf.reduce_mean(
          input_tensor=tf.abs(dist.sample(n, seed=tfp_test_util.test_seed())))

    err = gradient_checker_v2.max_error(
        *gradient_checker_v2.compute_gradient(
            f, [loc, scale, low, high], delta=0.1))

    # These gradients are noisy due to sampling.
    self.assertLess(err, 0.05)

  def testReparametrizableBatch(self):
    def samples_sum(loc):
      dist = tfp.distributions.TruncatedNormal(
          loc=loc, scale=1., low=-1., high=1.)
      return tf.reduce_sum(input_tensor=dist.sample(100))

    loc = tf.constant([0., 1.])
    _, dy_loc = self.evaluate(tfp.math.value_and_gradient(samples_sum, loc))
    self.assertAllGreaterEqual(dy_loc, 0.)

  @parameterized.parameters(
      itertools.product((np.float32, np.float64),
                        ("prob", "log_prob", "cdf", "log_cdf",
                         "survival_function", "log_survival_function"))
  )
  def testGradientsFx(self, dtype, fn_name):
    loc = tf.compat.v2.Variable(dtype(0.1))
    scale = tf.compat.v2.Variable(dtype(3.0))
    low = tf.compat.v2.Variable(dtype(-10.0))
    high = tf.compat.v2.Variable(dtype(5.0))

    x = np.array([-1.0, 0.01, 0.1, 1., 4.9]).astype(dtype)
    self.evaluate(tf.compat.v1.global_variables_initializer())

    def f(loc, scale):
      dist = tfd.TruncatedNormal(loc=loc, scale=scale, low=low, high=high)
      func = getattr(dist, fn_name)
      return tf.reduce_mean(input_tensor=func(x))

    err = gradient_checker_v2.max_error(
        *gradient_checker_v2.compute_gradient(f, [loc, scale]))
    self.assertLess(err, 1e-2)

  @parameterized.parameters(
      itertools.product((np.float32, np.float64),
                        ("entropy", "mean", "variance", "mode"))
  )
  def testGradientsNx(self, dtype, fn_name):
    loc = tf.compat.v2.Variable(dtype(0.1))
    scale = tf.compat.v2.Variable(dtype(3.0))
    low = tf.compat.v2.Variable(dtype(-10.0))
    high = tf.compat.v2.Variable(dtype(5.0))

    self.evaluate(tf.compat.v1.global_variables_initializer())

    def f(loc, scale):
      dist = tfd.TruncatedNormal(loc=loc, scale=scale, low=low, high=high)
      func = getattr(dist, fn_name)
      return func()

    if fn_name not in ["mode"]:
      err = gradient_checker_v2.max_error(
          *gradient_checker_v2.compute_gradient(f, [loc, scale]))
      self.assertLess(err, 0.005)
    else:
      err = gradient_checker_v2.max_error(
          *gradient_checker_v2.compute_gradient(
              lambda x: f(x, scale), [loc]))
      self.assertLess(err, 0.005)


@test_util.run_all_in_graph_and_eager_modes
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
    normal_dist = tfd.Normal(loc=loc, scale=scale)
    return truncated_dist, normal_dist

  def testEntropy(self, loc, scale):
    truncated_dist, normal_dist = self.constructDists(loc, scale)
    self.assertEqual(
        self.evaluate(truncated_dist.entropy()),
        self.evaluate(normal_dist.entropy()))

  def testSampling(self, loc, scale):
    n = 1000000
    truncated_dist, normal_dist = self.constructDists(loc, scale)
    truncated_samples = self.evaluate(truncated_dist.sample(n)).flatten()
    lb = self.evaluate(truncated_dist.low)
    ub = self.evaluate(truncated_dist.high)
    self.assertAllGreaterEqual(truncated_samples, lb)
    self.assertAllLessEqual(truncated_samples, ub)

    normal_samples = self.evaluate(normal_dist.sample(
        n, seed=tfp_test_util.test_seed())).flatten()
    # Rejection sample the normal distribution
    rejection_samples = normal_samples[normal_samples >= lb]
    rejection_samples = rejection_samples[rejection_samples <= ub]

    self.assertEmpiricalDistributionsEqual(
        truncated_samples, rejection_samples, rtol=1e-2, atol=1e-1)

  def testLogProb(self, loc, scale):
    truncated_dist, normal_dist = self.constructDists(loc, scale)
    low = self.evaluate(truncated_dist.low)
    high = self.evaluate(truncated_dist.high)
    test_x = list(np.float32(
        np.random.uniform(low, high, 10)))
    test_x += [low, high, low + EPSILON, high - EPSILON]
    tr_log_prob = self.evaluate(truncated_dist.log_prob(test_x))
    n_log_prob = self.evaluate(normal_dist.log_prob(test_x))
    self.assertAllClose(tr_log_prob, n_log_prob, rtol=1e-4, atol=1e-4)

    no_support_log_prob = self.evaluate(
        truncated_dist.log_prob(
            np.float32(
                [low - EPSILON, high + EPSILON, low - 100., high + 100.]
            )))
    self.assertAllEqual(no_support_log_prob,
                        [np.log(0.)] * len(no_support_log_prob))

  def testCDF(self, loc, scale):
    truncated_dist, normal_dist = self.constructDists(loc, scale)
    low = self.evaluate(truncated_dist.low)
    high = self.evaluate(truncated_dist.high)
    test_x = list(
        np.float32(np.random.uniform(low, high, 10)))
    test_x += [low, high, low + EPSILON, high - EPSILON]
    tr_cdf = self.evaluate(truncated_dist.cdf(test_x))
    n_cdf = self.evaluate(normal_dist.cdf(test_x))
    self.assertAllClose(tr_cdf, n_cdf, rtol=1e-4, atol=1e-4)


if stats:
  @test_util.run_all_in_graph_and_eager_modes
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
      n = int(1000000)
      tf_dist, sp_dist = self.constructDists(loc, scale, low, high)
      tf_samples = self.evaluate(tf_dist.sample(
          n, seed=tfp_test_util.test_seed())).flatten()
      self.assertAllGreaterEqual(tf_samples, low)
      self.assertAllLessEqual(tf_samples, high)

      sp_samples = sp_dist.rvs(size=n)
      self.assertEmpiricalDistributionsEqual(
          tf_samples, sp_samples, atol=0.05, rtol=0.05)

    def testEntropy(self, loc, scale, low, high):
      tf_dist, sp_dist = self.constructDists(loc, scale, low, high)
      self.assertAlmostEqual(
          self.evaluate(tf_dist.entropy()), sp_dist.entropy(), places=2)

    def testLogProb(self, loc, scale, low, high):
      test_x = list(np.float32(np.random.uniform(low, high, 10)))
      test_x += [
          low, high, low + EPSILON, low - EPSILON, high + EPSILON,
          high - EPSILON
      ]

      tf_dist, sp_dist = self.constructDists(loc, scale, low, high)

      tf_log_prob = self.evaluate(tf_dist.log_prob(test_x))
      sp_log_prob = sp_dist.logpdf(test_x)
      self.assertAllClose(tf_log_prob, sp_log_prob, rtol=1e-4, atol=1e-4)

    def testCDF(self, loc, scale, low, high):
      test_x = list(np.float32(np.random.uniform(low, high, 10)))
      test_x += [
          low, high, low + EPSILON, low - EPSILON, high + EPSILON,
          high - EPSILON, low - 100., high + 100.
      ]

      tf_dist, sp_dist = self.constructDists(loc, scale, low, high)

      tf_cdf = self.evaluate(tf_dist.cdf(test_x))
      sp_cdf = sp_dist.cdf(test_x)
      self.assertAllClose(tf_cdf, sp_cdf, rtol=1e-4, atol=1e-4)

    def testMoments(self, loc, scale, low, high):
      tf_dist, sp_dist = self.constructDists(loc, scale, low, high)
      self.assertAlmostEqual(
          self.evaluate(tf_dist.mean()), sp_dist.mean(), places=3)
      self.assertAlmostEqual(
          self.evaluate(tf_dist.variance()), sp_dist.var(), places=3)


if __name__ == "__main__":
  tf.test.main()
