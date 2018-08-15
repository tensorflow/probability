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
"""Tests for initializers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import importlib
# Dependency imports
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


class HalfNormalTest(tf.test.TestCase):

  def setUp(self):
    self._rng = np.random.RandomState(123)

  def assertAllFinite(self, tensor):
    is_finite = np.isfinite(self.evaluate(tensor))
    all_true = np.ones_like(is_finite, dtype=np.bool)
    self.assertAllEqual(all_true, is_finite)

  def _testParamShapes(self, sample_shape, expected):
    with self.test_session():
      param_shapes = tfd.HalfNormal.param_shapes(sample_shape)
      scale_shape = param_shapes["scale"]
      self.assertAllEqual(expected, self.evaluate(scale_shape))
      scale = tf.ones(scale_shape)
      self.assertAllEqual(expected,
                          self.evaluate(
                              tf.shape(tfd.HalfNormal(scale).sample())))

  def _testParamStaticShapes(self, sample_shape, expected):
    param_shapes = tfd.HalfNormal.param_static_shapes(sample_shape)
    scale_shape = param_shapes["scale"]
    self.assertEqual(expected, scale_shape)

  def _testBatchShapes(self, dist, tensor):
    self.assertAllEqual(self.evaluate(dist.batch_shape_tensor()), tensor.shape)
    self.assertAllEqual(
        self.evaluate(dist.batch_shape_tensor()), self.evaluate(tensor).shape)
    self.assertAllEqual(dist.batch_shape, tensor.shape)
    self.assertAllEqual(dist.batch_shape, self.evaluate(tensor).shape)

  def testParamShapes(self):
    sample_shape = [10, 3, 4]
    self._testParamShapes(sample_shape, sample_shape)
    self._testParamShapes(tf.constant(sample_shape), sample_shape)

  def testParamStaticShapes(self):
    sample_shape = [10, 3, 4]
    self._testParamStaticShapes(sample_shape, sample_shape)
    self._testParamStaticShapes(tf.TensorShape(sample_shape), sample_shape)

  def testHalfNormalLogPDF(self):
    with self.test_session():
      batch_size = 6
      scale = tf.constant([3.0] * batch_size)
      x = np.array([-2.5, 2.5, 4.0, 0.0, -1.0, 2.0], dtype=np.float32)
      halfnorm = tfd.HalfNormal(scale=scale)

      log_pdf = halfnorm.log_prob(x)
      self._testBatchShapes(halfnorm, log_pdf)

      pdf = halfnorm.prob(x)
      self._testBatchShapes(halfnorm, pdf)

      if not stats:
        return
      expected_log_pdf = stats.halfnorm(scale=self.evaluate(scale)).logpdf(x)
      self.assertAllClose(expected_log_pdf, self.evaluate(log_pdf))
      self.assertAllClose(np.exp(expected_log_pdf), self.evaluate(pdf))

  def testHalfNormalLogPDFMultidimensional(self):
    with self.test_session():
      batch_size = 6
      scale = tf.constant([[3.0, 1.0]] * batch_size)
      x = np.array([[-2.5, 2.5, 4.0, 0.0, -1.0, 2.0]], dtype=np.float32).T
      halfnorm = tfd.HalfNormal(scale=scale)

      log_pdf = halfnorm.log_prob(x)
      self._testBatchShapes(halfnorm, log_pdf)

      pdf = halfnorm.prob(x)
      self._testBatchShapes(halfnorm, pdf)

      if not stats:
        return
      expected_log_pdf = stats.halfnorm(scale=self.evaluate(scale)).logpdf(x)
      self.assertAllClose(expected_log_pdf, self.evaluate(log_pdf))
      self.assertAllClose(np.exp(expected_log_pdf), self.evaluate(pdf))

  def testHalfNormalCDF(self):
    with self.test_session():
      batch_size = 50
      scale = self._rng.rand(batch_size) + 1.0
      x = np.linspace(-8.0, 8.0, batch_size).astype(np.float64)
      halfnorm = tfd.HalfNormal(scale=scale)

      cdf = halfnorm.cdf(x)
      self._testBatchShapes(halfnorm, cdf)

      log_cdf = halfnorm.log_cdf(x)
      self._testBatchShapes(halfnorm, log_cdf)

      if not stats:
        return
      expected_logcdf = stats.halfnorm(scale=scale).logcdf(x)
      self.assertAllClose(expected_logcdf, self.evaluate(log_cdf), atol=0)
      self.assertAllClose(np.exp(expected_logcdf), self.evaluate(cdf), atol=0)

  def testHalfNormalSurvivalFunction(self):
    with self.test_session():
      batch_size = 50
      scale = self._rng.rand(batch_size) + 1.0
      x = np.linspace(-8.0, 8.0, batch_size).astype(np.float64)
      halfnorm = tfd.HalfNormal(scale=scale)

      sf = halfnorm.survival_function(x)
      self._testBatchShapes(halfnorm, sf)

      log_sf = halfnorm.log_survival_function(x)
      self._testBatchShapes(halfnorm, log_sf)

      if not stats:
        return
      expected_logsf = stats.halfnorm(scale=scale).logsf(x)
      self.assertAllClose(expected_logsf, self.evaluate(log_sf), atol=0)
      self.assertAllClose(np.exp(expected_logsf), self.evaluate(sf), atol=0)

  def testHalfNormalQuantile(self):
    with self.test_session():
      batch_size = 50
      scale = self._rng.rand(batch_size) + 1.0
      p = np.linspace(0., 1.0, batch_size).astype(np.float64)

      halfnorm = tfd.HalfNormal(scale=scale)
      x = halfnorm.quantile(p)
      self._testBatchShapes(halfnorm, x)

      if not stats:
        return
      expected_x = stats.halfnorm(scale=scale).ppf(p)
      self.assertAllClose(expected_x, self.evaluate(x), atol=0)

  def testFiniteGradients(self):
    for dtype in [np.float32, np.float64]:
      g = tf.Graph()
      with g.as_default():
        scale = tf.Variable(dtype(3.0))
        dist = tfd.HalfNormal(scale=scale)
        x = np.array([0.01, 0.1, 1., 5., 10.]).astype(dtype)
        for func in [
            dist.cdf, dist.log_cdf, dist.survival_function,
            dist.log_prob, dist.prob, dist.log_survival_function,
        ]:
          print(func.__name__)
          value = func(x)
          grads = tf.gradients(value, [scale])
          with self.test_session(graph=g):
            tf.global_variables_initializer().run()
            self.assertAllFinite(value)
            self.assertAllFinite(grads[0])

  def testHalfNormalEntropy(self):
    with self.test_session():
      scale = np.array([[1.0, 2.0, 3.0]])
      halfnorm = tfd.HalfNormal(scale=scale)

      # See https://en.wikipedia.org/wiki/Half-normal_distribution for the
      # entropy formula used here.
      expected_entropy = 0.5 * np.log(np.pi * scale ** 2.0 / 2.0) + 0.5

      entropy = halfnorm.entropy()
      self._testBatchShapes(halfnorm, entropy)
      self.assertAllClose(expected_entropy, self.evaluate(entropy))

  def testHalfNormalMeanAndMode(self):
    with self.test_session():
      scale = np.array([11., 12., 13.])

      halfnorm = tfd.HalfNormal(scale=scale)
      expected_mean = scale * np.sqrt(2.0) / np.sqrt(np.pi)

      self.assertAllEqual((3,), self.evaluate(halfnorm.mean()).shape)
      self.assertAllEqual(expected_mean, self.evaluate(halfnorm.mean()))

      self.assertAllEqual((3,), self.evaluate(halfnorm.mode()).shape)
      self.assertAllEqual([0., 0., 0.], self.evaluate(halfnorm.mode()))

  def testHalfNormalVariance(self):
    with self.test_session():
      scale = np.array([7., 7., 7.])
      halfnorm = tfd.HalfNormal(scale=scale)
      expected_variance = scale ** 2.0 * (1.0 - 2.0 / np.pi)

      self.assertAllEqual((3,), self.evaluate(halfnorm.variance()).shape)
      self.assertAllEqual(expected_variance, self.evaluate(halfnorm.variance()))

  def testHalfNormalStandardDeviation(self):
    with self.test_session():
      scale = np.array([7., 7., 7.])
      halfnorm = tfd.HalfNormal(scale=scale)
      expected_variance = scale ** 2.0 * (1.0 - 2.0 / np.pi)

      self.assertAllEqual((3,), halfnorm.stddev().shape)
      self.assertAllEqual(
          np.sqrt(expected_variance), self.evaluate(halfnorm.stddev()))

  def testHalfNormalSample(self):
    with self.test_session():
      scale = tf.constant(3.0)
      n = tf.constant(100000)
      halfnorm = tfd.HalfNormal(scale=scale)

      sample = halfnorm.sample(n)

      self.assertEqual(self.evaluate(sample).shape, (100000,))
      self.assertAllClose(self.evaluate(sample).mean(),
                          3.0 * np.sqrt(2.0) / np.sqrt(np.pi), atol=1e-1)

      expected_shape = tf.TensorShape([self.evaluate(n)]).concatenate(
          tf.TensorShape(self.evaluate(halfnorm.batch_shape_tensor())))
      self.assertAllEqual(expected_shape, sample.shape)
      self.assertAllEqual(expected_shape, self.evaluate(sample).shape)

      expected_shape_static = (
          tf.TensorShape([self.evaluate(n)]).concatenate(halfnorm.batch_shape))
      self.assertAllEqual(expected_shape_static, sample.shape)
      self.assertAllEqual(expected_shape_static, self.evaluate(sample).shape)

  def testHalfNormalSampleMultiDimensional(self):
    with self.test_session():
      batch_size = 2
      scale = tf.constant([[2.0, 3.0]] * batch_size)
      n = tf.constant(100000)
      halfnorm = tfd.HalfNormal(scale=scale)

      sample = halfnorm.sample(n)
      self.assertEqual(sample.shape, (100000, batch_size, 2))
      self.assertAllClose(self.evaluate(sample)[:, 0, 0].mean(),
                          2.0 * np.sqrt(2.0) / np.sqrt(np.pi), atol=1e-1)
      self.assertAllClose(self.evaluate(sample)[:, 0, 1].mean(),
                          3.0 * np.sqrt(2.0) / np.sqrt(np.pi), atol=1e-1)

      expected_shape = tf.TensorShape([self.evaluate(n)]).concatenate(
          tf.TensorShape(self.evaluate(halfnorm.batch_shape_tensor())))
      self.assertAllEqual(expected_shape, sample.shape)
      self.assertAllEqual(expected_shape, self.evaluate(sample).shape)

      expected_shape_static = (
          tf.TensorShape([self.evaluate(n)]).concatenate(halfnorm.batch_shape))
      self.assertAllEqual(expected_shape_static, sample.shape)
      self.assertAllEqual(expected_shape_static, self.evaluate(sample).shape)

  def testNegativeSigmaFails(self):
    with self.test_session():
      halfnorm = tfd.HalfNormal(scale=[-5.], validate_args=True, name="G")
      with self.assertRaisesOpError("Condition x > 0 did not hold"):
        self.evaluate(halfnorm.mean())

  def testHalfNormalShape(self):
    with self.test_session():
      scale = tf.constant([6.0] * 5)
      halfnorm = tfd.HalfNormal(scale=scale)

      self.assertEqual(self.evaluate(halfnorm.batch_shape_tensor()), [5])
      self.assertEqual(halfnorm.batch_shape, tf.TensorShape([5]))
      self.assertAllEqual(self.evaluate(halfnorm.event_shape_tensor()), [])
      self.assertEqual(halfnorm.event_shape, tf.TensorShape([]))

  def testHalfNormalShapeWithPlaceholders(self):
    scale = tf.placeholder(dtype=tf.float32)
    halfnorm = tfd.HalfNormal(scale=scale)

    with self.test_session() as sess:
      # get_batch_shape should return an "<unknown>" tensor.
      self.assertEqual(halfnorm.batch_shape, tf.TensorShape(None))
      self.assertEqual(halfnorm.event_shape, ())
      self.assertAllEqual(self.evaluate(halfnorm.event_shape_tensor()), [])
      self.assertAllEqual(
          sess.run(halfnorm.batch_shape_tensor(),
                   feed_dict={scale: [1.0, 2.0]}), [2])


if __name__ == "__main__":
  tf.test.main()
