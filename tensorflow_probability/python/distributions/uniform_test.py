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
"""Tests for Uniform distribution."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import importlib

# Dependency imports
import numpy as np
import tensorflow as tf

from tensorflow_probability.python.distributions import uniform as uniform_lib
from tensorflow.python.eager import backprop
from tensorflow.python.framework import errors
from tensorflow.python.framework import test_util


def try_import(name):  # pylint: disable=invalid-name
  module = None
  try:
    module = importlib.import_module(name)
  except ImportError as e:
    tf.logging.warning("Could not import %s: %s" % (name, str(e)))
  return module


stats = try_import("scipy.stats")


class UniformTest(tf.test.TestCase):

  @test_util.run_in_graph_and_eager_modes
  def testUniformRange(self):
    a = 3.0
    b = 10.0
    uniform = uniform_lib.Uniform(low=a, high=b)
    self.assertAllClose(a, self.evaluate(uniform.low))
    self.assertAllClose(b, self.evaluate(uniform.high))
    self.assertAllClose(b - a, self.evaluate(uniform.range()))

  @test_util.run_in_graph_and_eager_modes
  def testUniformPDF(self):
    a = tf.constant([-3.0] * 5 + [15.0])
    b = tf.constant([11.0] * 5 + [20.0])
    uniform = uniform_lib.Uniform(low=a, high=b)

    a_v = -3.0
    b_v = 11.0
    x = np.array([-10.5, 4.0, 0.0, 10.99, 11.3, 17.0], dtype=np.float32)

    def _expected_pdf():
      pdf = np.zeros_like(x) + 1.0 / (b_v - a_v)
      pdf[x > b_v] = 0.0
      pdf[x < a_v] = 0.0
      pdf[5] = 1.0 / (20.0 - 15.0)
      return pdf

    expected_pdf = _expected_pdf()

    pdf = uniform.prob(x)
    self.assertAllClose(expected_pdf, self.evaluate(pdf))

    log_pdf = uniform.log_prob(x)
    self.assertAllClose(np.log(expected_pdf), self.evaluate(log_pdf))

  @test_util.run_in_graph_and_eager_modes
  def testUniformShape(self):
    a = tf.constant([-3.0] * 5)
    b = tf.constant(11.0)
    uniform = uniform_lib.Uniform(low=a, high=b)

    self.assertEqual(self.evaluate(uniform.batch_shape_tensor()), (5,))
    self.assertEqual(uniform.batch_shape, tf.TensorShape([5]))
    self.assertAllEqual(self.evaluate(uniform.event_shape_tensor()), [])
    self.assertEqual(uniform.event_shape, tf.TensorShape([]))

  @test_util.run_in_graph_and_eager_modes
  def testUniformPDFWithScalarEndpoint(self):
    a = tf.constant([0.0, 5.0])
    b = tf.constant(10.0)
    uniform = uniform_lib.Uniform(low=a, high=b)

    x = np.array([0.0, 8.0], dtype=np.float32)
    expected_pdf = np.array([1.0 / (10.0 - 0.0), 1.0 / (10.0 - 5.0)])

    pdf = uniform.prob(x)
    self.assertAllClose(expected_pdf, self.evaluate(pdf))

  @test_util.run_in_graph_and_eager_modes
  def testUniformCDF(self):
    batch_size = 6
    a = tf.constant([1.0] * batch_size)
    b = tf.constant([11.0] * batch_size)
    a_v = 1.0
    b_v = 11.0
    x = np.array([-2.5, 2.5, 4.0, 0.0, 10.99, 12.0], dtype=np.float32)

    uniform = uniform_lib.Uniform(low=a, high=b)

    def _expected_cdf():
      cdf = (x - a_v) / (b_v - a_v)
      cdf[x >= b_v] = 1
      cdf[x < a_v] = 0
      return cdf

    cdf = uniform.cdf(x)
    self.assertAllClose(_expected_cdf(), self.evaluate(cdf))

    log_cdf = uniform.log_cdf(x)
    self.assertAllClose(np.log(_expected_cdf()), self.evaluate(log_cdf))

  @test_util.run_in_graph_and_eager_modes
  def testUniformEntropy(self):
    a_v = np.array([1.0, 1.0, 1.0])
    b_v = np.array([[1.5, 2.0, 3.0]])
    uniform = uniform_lib.Uniform(low=a_v, high=b_v)

    expected_entropy = np.log(b_v - a_v)
    self.assertAllClose(expected_entropy, self.evaluate(uniform.entropy()))

  @test_util.run_in_graph_and_eager_modes
  def testUniformAssertMaxGtMin(self):
    a_v = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    b_v = np.array([1.0, 2.0, 3.0], dtype=np.float32)

    with self.assertRaisesWithPredicateMatch(errors.InvalidArgumentError,
                                             "x < y"):
      uniform = uniform_lib.Uniform(low=a_v, high=b_v, validate_args=True)
      self.evaluate(uniform.low)

  @test_util.run_in_graph_and_eager_modes
  def testUniformSample(self):
    a = tf.constant([3.0, 4.0])
    b = tf.constant(13.0)
    a1_v = 3.0
    a2_v = 4.0
    b_v = 13.0
    n = tf.constant(100000)
    uniform = uniform_lib.Uniform(low=a, high=b)

    samples = uniform.sample(n, seed=137)
    sample_values = self.evaluate(samples)
    self.assertEqual(sample_values.shape, (100000, 2))
    self.assertAllClose(
        sample_values[::, 0].mean(), (b_v + a1_v) / 2, atol=1e-1, rtol=0.)
    self.assertAllClose(
        sample_values[::, 1].mean(), (b_v + a2_v) / 2, atol=1e-1, rtol=0.)
    self.assertFalse(
        np.any(sample_values[::, 0] < a1_v) or np.any(sample_values >= b_v))
    self.assertFalse(
        np.any(sample_values[::, 1] < a2_v) or np.any(sample_values >= b_v))

  @test_util.run_in_graph_and_eager_modes
  def _testUniformSampleMultiDimensional(self):
    # DISABLED: Please enable this test once b/issues/30149644 is resolved.
    batch_size = 2
    a_v = [3.0, 22.0]
    b_v = [13.0, 35.0]
    a = tf.constant([a_v] * batch_size)
    b = tf.constant([b_v] * batch_size)

    uniform = uniform_lib.Uniform(low=a, high=b)

    n_v = 100000
    n = tf.constant(n_v)
    samples = uniform.sample(n)
    self.assertEqual(samples.shape, (n_v, batch_size, 2))

    sample_values = self.evaluate(samples)

    self.assertFalse(
        np.any(sample_values[:, 0, 0] < a_v[0]) or
        np.any(sample_values[:, 0, 0] >= b_v[0]))
    self.assertFalse(
        np.any(sample_values[:, 0, 1] < a_v[1]) or
        np.any(sample_values[:, 0, 1] >= b_v[1]))

    self.assertAllClose(
        sample_values[:, 0, 0].mean(), (a_v[0] + b_v[0]) / 2, atol=1e-2)
    self.assertAllClose(
        sample_values[:, 0, 1].mean(), (a_v[1] + b_v[1]) / 2, atol=1e-2)

  @test_util.run_in_graph_and_eager_modes
  def testUniformMean(self):
    a = 10.0
    b = 100.0
    uniform = uniform_lib.Uniform(low=a, high=b)
    if not stats:
      return
    s_uniform = stats.uniform(loc=a, scale=b - a)
    self.assertAllClose(self.evaluate(uniform.mean()), s_uniform.mean())

  @test_util.run_in_graph_and_eager_modes
  def testUniformVariance(self):
    a = 10.0
    b = 100.0
    uniform = uniform_lib.Uniform(low=a, high=b)
    if not stats:
      return
    s_uniform = stats.uniform(loc=a, scale=b - a)
    self.assertAllClose(self.evaluate(uniform.variance()), s_uniform.var())

  @test_util.run_in_graph_and_eager_modes
  def testUniformStd(self):
    a = 10.0
    b = 100.0
    uniform = uniform_lib.Uniform(low=a, high=b)
    if not stats:
      return
    s_uniform = stats.uniform(loc=a, scale=b - a)
    self.assertAllClose(self.evaluate(uniform.stddev()), s_uniform.std())

  @test_util.run_in_graph_and_eager_modes
  def testUniformNans(self):
    a = 10.0
    b = [11.0, 100.0]
    uniform = uniform_lib.Uniform(low=a, high=b)

    no_nans = tf.constant(1.0)
    nans = tf.constant(0.0) / tf.constant(0.0)
    self.assertTrue(self.evaluate(tf.is_nan(nans)))
    with_nans = tf.stack([no_nans, nans])

    pdf = uniform.prob(with_nans)

    is_nan = self.evaluate(tf.is_nan(pdf))
    self.assertFalse(is_nan[0])
    self.assertTrue(is_nan[1])

  @test_util.run_in_graph_and_eager_modes
  def testUniformSamplePdf(self):
    a = 10.0
    b = [11.0, 100.0]
    uniform = uniform_lib.Uniform(a, b)
    self.assertTrue(
        self.evaluate(
            tf.reduce_all(uniform.prob(uniform.sample(10)) > 0)))

  @test_util.run_in_graph_and_eager_modes
  def testUniformBroadcasting(self):
    a = 10.0
    b = [11.0, 20.0]
    uniform = uniform_lib.Uniform(a, b)

    pdf = uniform.prob([[10.5, 11.5], [9.0, 19.0], [10.5, 21.0]])
    expected_pdf = np.array([[1.0, 0.1], [0.0, 0.1], [1.0, 0.0]])
    self.assertAllClose(expected_pdf, self.evaluate(pdf))

  @test_util.run_in_graph_and_eager_modes
  def testUniformSampleWithShape(self):
    a = 10.0
    b = [11.0, 20.0]
    uniform = uniform_lib.Uniform(a, b)

    pdf = uniform.prob(uniform.sample((2, 3)))
    # pylint: disable=bad-continuation
    expected_pdf = [
        [[1.0, 0.1], [1.0, 0.1], [1.0, 0.1]],
        [[1.0, 0.1], [1.0, 0.1], [1.0, 0.1]],
    ]
    # pylint: enable=bad-continuation
    self.assertAllClose(expected_pdf, self.evaluate(pdf))

    pdf = uniform.prob(uniform.sample())
    expected_pdf = [1.0, 0.1]
    self.assertAllClose(expected_pdf, self.evaluate(pdf))

  def testFullyReparameterized(self):
    a = tf.constant(0.1)
    b = tf.constant(0.8)
    with backprop.GradientTape() as tape:
      tape.watch(a)
      tape.watch(b)
      uniform = uniform_lib.Uniform(a, b)
      samples = uniform.sample(100)
    grad_a, grad_b = tape.gradient(samples, [a, b])
    self.assertIsNotNone(grad_a)
    self.assertIsNotNone(grad_b)

  # Eager doesn't pass due to a type mismatch in one of the ops.
  def testUniformFloat64(self):
    uniform = uniform_lib.Uniform(
        low=np.float64(0.), high=np.float64(1.))

    self.assertAllClose(
        [1., 1.],
        self.evaluate(uniform.prob(np.array([0.5, 0.6], dtype=np.float64))))

    self.assertAllClose(
        [0.5, 0.6],
        self.evaluate(uniform.cdf(np.array([0.5, 0.6], dtype=np.float64))))

    self.assertAllClose(0.5, self.evaluate(uniform.mean()))
    self.assertAllClose(1 / 12., self.evaluate(uniform.variance()))
    self.assertAllClose(0., self.evaluate(uniform.entropy()))


if __name__ == "__main__":
  tf.test.main()
