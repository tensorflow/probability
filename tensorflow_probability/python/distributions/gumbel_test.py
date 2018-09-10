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
"""Tests for Gumbel."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
import numpy as np
from scipy import stats
import tensorflow as tf
import tensorflow_probability as tfp

from tensorflow_probability.python.internal import test_case
from tensorflow.python.framework import test_util

tfd = tfp.distributions


class _GumbelTest(object):

  def make_tensor(self, x):
    x = tf.cast(x, self._dtype)
    return tf.placeholder_with_default(
        input=x, shape=x.shape if self._use_static_shape else None)

  def testGumbelShape(self):
    loc = np.array([3.0] * 5, dtype=self._dtype)
    scale = np.array([3.0] * 5, dtype=self._dtype)
    gumbel = tfd.Gumbel(loc=loc, scale=scale)

    self.assertEqual((5,), self.evaluate(gumbel.batch_shape_tensor()))
    self.assertEqual(tf.TensorShape([5]), gumbel.batch_shape)
    self.assertAllEqual([], self.evaluate(gumbel.event_shape_tensor()))
    self.assertEqual(tf.TensorShape([]), gumbel.event_shape)

  def testInvalidScale(self):
    scale = [-.01, 0., 2.]
    with self.assertRaisesOpError("Condition x > 0"):
      gumbel = tfd.Gumbel(loc=0., scale=scale, validate_args=True)
      self.evaluate(gumbel.scale)

  def testGumbelLogPdf(self):
    batch_size = 6
    loc = np.array([0.] * batch_size, dtype=self._dtype)
    scale = np.array([3.] * batch_size, dtype=self._dtype)
    x = np.array([2., 3., 4., 5., 6., 7.], dtype=self._dtype)
    gumbel = tfd.Gumbel(
        loc=self.make_tensor(loc),
        scale=self.make_tensor(scale))
    log_pdf = gumbel.log_prob(self.make_tensor(x))
    self.assertAllClose(
        stats.gumbel_r.logpdf(x, loc=loc, scale=scale),
        self.evaluate(log_pdf))

    pdf = gumbel.prob(x)
    self.assertAllClose(
        stats.gumbel_r.pdf(x, loc=loc, scale=scale), self.evaluate(pdf))

  def testGumbelLogPdfMultidimensional(self):
    batch_size = 6
    loc = np.array([[2.0, 4.0, 5.0]] * batch_size, dtype=self._dtype)
    scale = np.array([1.0], dtype=self._dtype)
    x = np.array([[2., 3., 4., 5., 6., 7.]], dtype=self._dtype).T

    gumbel = tfd.Gumbel(
        loc=self.make_tensor(loc),
        scale=self.make_tensor(scale))
    log_pdf = gumbel.log_prob(self.make_tensor(x))
    self.assertAllClose(
        self.evaluate(log_pdf), stats.gumbel_r.logpdf(x, loc=loc, scale=scale))

    pdf = gumbel.prob(self.make_tensor(x))
    self.assertAllClose(
        self.evaluate(pdf), stats.gumbel_r.pdf(x, loc=loc, scale=scale))

  def testGumbelCDF(self):
    batch_size = 6
    loc = np.array([0.] * batch_size, dtype=self._dtype)
    scale = np.array([3.] * batch_size, dtype=self._dtype)
    x = np.array([2., 3., 4., 5., 6., 7.], dtype=self._dtype)

    gumbel = tfd.Gumbel(
        loc=self.make_tensor(loc),
        scale=self.make_tensor(scale))

    log_cdf = gumbel.log_cdf(self.make_tensor(x))
    self.assertAllClose(
        self.evaluate(log_cdf), stats.gumbel_r.logcdf(x, loc=loc, scale=scale))

    cdf = gumbel.cdf(self.make_tensor(x))
    self.assertAllClose(
        self.evaluate(cdf), stats.gumbel_r.cdf(x, loc=loc, scale=scale))

  def testGumbelCdfMultidimensional(self):
    batch_size = 6
    loc = np.array([[2.0, 4.0, 5.0]] * batch_size, dtype=self._dtype)
    scale = np.array([1.0], dtype=self._dtype)
    x = np.array([[2., 3., 4., 5., 6., 7.]], dtype=self._dtype).T

    gumbel = tfd.Gumbel(
        loc=self.make_tensor(loc),
        scale=self.make_tensor(scale))

    log_cdf = gumbel.log_cdf(self.make_tensor(x))
    self.assertAllClose(
        self.evaluate(log_cdf),
        stats.gumbel_r.logcdf(x, loc=loc, scale=scale))

    cdf = gumbel.cdf(self.make_tensor(x))
    self.assertAllClose(
        self.evaluate(cdf),
        stats.gumbel_r.cdf(x, loc=loc, scale=scale))

  def testGumbelMean(self):
    batch_size = 6
    loc = np.array([[2.0, 4.0, 5.0]] * batch_size, dtype=self._dtype)
    scale = np.array([1.0], dtype=self._dtype)

    gumbel = tfd.Gumbel(
        loc=self.make_tensor(loc),
        scale=self.make_tensor(scale))
    self.assertAllClose(self.evaluate(gumbel.mean()),
                        stats.gumbel_r.mean(loc=loc, scale=scale))

  def testGumbelVariance(self):
    batch_size = 6
    loc = np.array([[2.0, 4.0, 5.0]] * batch_size, dtype=self._dtype)
    scale = np.array([1.0], dtype=self._dtype)

    gumbel = tfd.Gumbel(
        loc=self.make_tensor(loc),
        scale=self.make_tensor(scale))

    self.assertAllClose(self.evaluate(gumbel.variance()),
                        stats.gumbel_r.var(loc=loc, scale=scale))

  def testGumbelStd(self):
    batch_size = 6
    loc = np.array([[2.0, 4.0, 5.0]] * batch_size, dtype=self._dtype)
    scale = np.array([1.0], dtype=self._dtype)

    gumbel = tfd.Gumbel(
        loc=self.make_tensor(loc),
        scale=self.make_tensor(scale))

    self.assertAllClose(self.evaluate(gumbel.stddev()),
                        stats.gumbel_r.std(loc=loc, scale=scale))

  def testGumbelMode(self):
    batch_size = 6
    loc = np.array([[2.0, 4.0, 5.0]] * batch_size, dtype=self._dtype)
    scale = np.array([1.0], dtype=self._dtype)

    gumbel = tfd.Gumbel(
        loc=self.make_tensor(loc),
        scale=self.make_tensor(scale))

    self.assertAllClose(self.evaluate(gumbel.mode()), self.evaluate(gumbel.loc))

  def testGumbelSample(self):
    loc = self._dtype(4.0)
    scale = self._dtype(1.0)
    n = int(100e3)

    gumbel = tfd.Gumbel(
        loc=self.make_tensor(loc),
        scale=self.make_tensor(scale))

    samples = gumbel.sample(n, seed=123456)
    sample_values = self.evaluate(samples)
    self.assertEqual((n,), sample_values.shape)
    self.assertAllClose(
        stats.gumbel_r.mean(loc=loc, scale=scale),
        sample_values.mean(), rtol=.01)
    self.assertAllClose(
        stats.gumbel_r.var(loc=loc, scale=scale),
        sample_values.var(), rtol=.01)

  def testGumbelSampleMultidimensionalMean(self):
    batch_size = 6
    loc = np.array([[2.0, 4.0, 5.0]] * batch_size, dtype=self._dtype)
    scale = np.array([1.0, 0.8, 0.5], dtype=self._dtype)
    n = int(300e3)

    gumbel = tfd.Gumbel(
        loc=self.make_tensor(loc),
        scale=self.make_tensor(scale))

    samples = gumbel.sample(n, seed=123456)
    sample_values = self.evaluate(samples)
    self.assertAllClose(
        stats.gumbel_r.mean(loc=loc, scale=scale),
        sample_values.mean(axis=0),
        rtol=.01,
        atol=0)

  def testGumbelSampleMultidimensionalVar(self):
    batch_size = 6
    loc = np.array([[2.0, 4.0, 5.0]] * batch_size, dtype=self._dtype)
    scale = np.array([1.0, 0.8, 0.5], dtype=self._dtype)
    n = int(500e3)

    gumbel = tfd.Gumbel(
        loc=self.make_tensor(loc),
        scale=self.make_tensor(scale))

    samples = gumbel.sample(n, seed=123456)
    sample_values = self.evaluate(samples)
    self.assertAllClose(
        stats.gumbel_r.var(loc=loc, scale=scale),
        sample_values.var(axis=0),
        rtol=.03,
        atol=0)


@test_util.run_all_in_graph_and_eager_modes
class GumbelTestStaticShape(test_case.TestCase, _GumbelTest):
  _dtype = np.float32
  _use_static_shape = True


@test_util.run_all_in_graph_and_eager_modes
class GumbelTestFloat64StaticShape(test_case.TestCase, _GumbelTest):
  _dtype = np.float64
  _use_static_shape = True


@test_util.run_all_in_graph_and_eager_modes
class GumbelTestDynamicShape(test_case.TestCase, _GumbelTest):
  _dtype = np.float32
  _use_static_shape = False


if __name__ == "__main__":
  tf.test.main()
