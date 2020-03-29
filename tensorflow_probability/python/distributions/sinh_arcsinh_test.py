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
"""Tests for SinhArcsinh."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
import numpy as np

import tensorflow.compat.v1 as tf1
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp

from tensorflow_probability.python.internal import test_util

tfd = tfp.distributions
rng = np.random.RandomState(123)


@test_util.test_all_tf_execution_regimes
class SinhArcsinhTest(test_util.TestCase):

  def testDefaultIsSameAsNormal(self):
    b = 10
    scale = rng.rand(b) + 0.5
    loc = rng.randn(b)
    norm = tfd.Normal(loc=loc, scale=scale, validate_args=True)
    sasnorm = tfd.SinhArcsinh(loc=loc, scale=scale, validate_args=True)

    x = rng.randn(5, b)
    norm_pdf, sasnorm_pdf = self.evaluate([norm.prob(x), sasnorm.prob(x)])
    self.assertAllClose(norm_pdf, sasnorm_pdf)

    norm_samps, sasnorm_samps = self.evaluate([
        norm.sample(10000, seed=test_util.test_seed()),
        sasnorm.sample(10000, seed=test_util.test_seed())
    ])
    self.assertAllClose(loc, sasnorm_samps.mean(axis=0), atol=0.1)
    self.assertAllClose(
        norm_samps.mean(axis=0), sasnorm_samps.mean(axis=0), atol=0.1)
    self.assertAllClose(
        norm_samps.std(axis=0), sasnorm_samps.std(axis=0), atol=0.1)

  def testBroadcastParamsDynamic(self):
    loc = tf1.placeholder_with_default(rng.rand(5), shape=None)
    scale = tf1.placeholder_with_default(
        np.float64(rng.rand()), shape=None)
    skewness = tf1.placeholder_with_default(rng.rand(5), shape=None)
    sasnorm = tfd.SinhArcsinh(
        loc=loc, scale=scale, skewness=skewness, validate_args=True)

    samp = self.evaluate(sasnorm.sample(seed=test_util.test_seed()))
    self.assertAllEqual((5,), samp.shape)

  def testPassingInLaplacePlusDefaultsIsSameAsLaplace(self):
    b = 10
    scale = rng.rand(b) + 0.5
    loc = rng.randn(b)
    lap = tfd.Laplace(loc=loc, scale=scale, validate_args=True)
    # TODO(b/151180729): When shape overrides of `TransformedDistribution` are
    # deprecated, change `distribution` below to
    # `tfd.Laplace(np.float64(np.zeros(b)), np.float64(1))`.
    saslap = tfd.SinhArcsinh(
        loc=loc,
        scale=scale,
        distribution=tfd.Laplace(np.float64(0), np.float64(1)),
        validate_args=True)

    x = rng.randn(5, b)
    lap_pdf, saslap_pdf = self.evaluate([lap.prob(x), saslap.prob(x)])
    self.assertAllClose(lap_pdf, saslap_pdf)

    lap_samps, saslap_samps = self.evaluate([
        lap.sample(10000, seed=test_util.test_seed()),
        saslap.sample(10000, seed=test_util.test_seed())
    ])
    self.assertAllClose(loc, saslap_samps.mean(axis=0), atol=0.1)
    self.assertAllClose(
        lap_samps.mean(axis=0), saslap_samps.mean(axis=0), atol=0.1)
    self.assertAllClose(
        lap_samps.std(axis=0), saslap_samps.std(axis=0), atol=0.1)

  def testTailweightSmallGivesFewerOutliersThanNormal(self):
    batch_size = 10
    scale = rng.rand(batch_size) + 0.5
    loc = 0.1 * rng.randn(batch_size)
    norm = tfd.Normal(loc=loc, scale=scale, validate_args=True)
    sasnorm = tfd.SinhArcsinh(
        loc=loc, scale=scale, tailweight=0.1, validate_args=True)

    # sasnorm.pdf(x) is smaller on outliers (+-10 are outliers)
    x = np.float64([[-10] * batch_size, [10] * batch_size])  # Shape [2, 10]
    norm_lp, sasnorm_lp = self.evaluate([norm.log_prob(x), sasnorm.log_prob(x)])
    np.testing.assert_array_less(sasnorm_lp, norm_lp)

    # 0.1% quantile and 99.9% quantile are outliers, and should be more
    # extreme in the normal.  The 97.772% quantiles should be the same.
    norm_samps, sasnorm_samps = self.evaluate([
        norm.sample(int(5e5), seed=test_util.test_seed()),
        sasnorm.sample(int(5e5), seed=test_util.test_seed())
    ])
    np.testing.assert_array_less(
        np.percentile(norm_samps, 0.1, axis=0),
        np.percentile(sasnorm_samps, 0.1, axis=0))
    np.testing.assert_array_less(
        np.percentile(sasnorm_samps, 99.9, axis=0),
        np.percentile(norm_samps, 99.9, axis=0))
    # 100. * sp.stats.norm.cdf(2.)
    q = 100 * 0.97724986805182079
    self.assertAllClose(
        np.percentile(sasnorm_samps, q, axis=0),
        np.percentile(norm_samps, q, axis=0),
        rtol=0.03)
    self.assertAllClose(
        np.percentile(sasnorm_samps, 100 - q, axis=0),
        np.percentile(norm_samps, 100 - q, axis=0),
        rtol=0.03)

  def testTailweightLargeGivesMoreOutliersThanNormal(self):
    batch_size = 10
    scale = rng.rand(batch_size) + 0.5
    loc = np.float64(0.)
    norm = tfd.Normal(loc=loc, scale=scale, validate_args=True)
    sasnorm = tfd.SinhArcsinh(
        loc=loc, scale=scale, tailweight=3., validate_args=True)

    # norm.pdf(x) is smaller on outliers (+-10 are outliers)
    x = np.float64([[-10] * batch_size, [10] * batch_size])  # Shape [2, 10]
    norm_lp, sasnorm_lp = self.evaluate([norm.log_prob(x), sasnorm.log_prob(x)])
    np.testing.assert_array_less(norm_lp, sasnorm_lp)

    # 0.1% quantile and 99.9% quantile are outliers, and should be more
    # extreme in the sasnormal.  The 97.772% quantiles should be the same.
    norm_samps, sasnorm_samps = self.evaluate([
        norm.sample(int(5e5), seed=test_util.test_seed()),
        sasnorm.sample(int(5e5), seed=test_util.test_seed())
    ])
    np.testing.assert_array_less(
        np.percentile(sasnorm_samps, 0.1, axis=0),
        np.percentile(norm_samps, 0.1, axis=0))
    np.testing.assert_array_less(
        np.percentile(norm_samps, 99.9, axis=0),
        np.percentile(sasnorm_samps, 99.9, axis=0))
    # 100. * sp.stats.norm.cdf(2.)
    q = 100 * 0.97724986805182079
    self.assertAllClose(
        np.percentile(sasnorm_samps, q, axis=0),
        np.percentile(norm_samps, q, axis=0),
        rtol=0.03)
    self.assertAllClose(
        np.percentile(sasnorm_samps, 100 - q, axis=0),
        np.percentile(norm_samps, 100 - q, axis=0),
        rtol=0.03)

  def testPositiveSkewnessMovesMeanToTheRight(self):
    batch_size = 10
    scale = rng.rand(batch_size) + 0.5
    loc = rng.randn(batch_size)
    sasnorm = tfd.SinhArcsinh(
        loc=loc, scale=scale, skewness=3.0, validate_args=True)

    sasnorm_samps = self.evaluate(
        sasnorm.sample(10000, seed=test_util.test_seed()))
    np.testing.assert_array_less(loc, sasnorm_samps.mean(axis=0))

  def testPdfReflectedForNegativeSkewness(self):
    sas_pos_skew = tfd.SinhArcsinh(
        loc=0., scale=1., skewness=2., validate_args=True)
    sas_neg_skew = tfd.SinhArcsinh(
        loc=0., scale=1., skewness=-2., validate_args=True)
    x = np.linspace(-2, 2, num=5).astype(np.float32)
    self.assertAllClose(*self.evaluate(
        [sas_pos_skew.prob(x), sas_neg_skew.prob(x[::-1])]))

  @test_util.tf_tape_safety_test
  def testVariableGradients(self):
    b = 10
    scale = tf.Variable(rng.rand(b) + 0.5)
    loc = tf.Variable(rng.randn(b))
    sasnorm = tfd.SinhArcsinh(loc=loc, scale=scale, validate_args=True)

    x = rng.randn(5, b)
    with tf.GradientTape() as tape:
      y = sasnorm.log_prob(x)
    grads = tape.gradient(y, sasnorm.trainable_variables)
    self.assertLen(grads, 2)
    self.assertAllNotNone(grads)

  @test_util.tf_tape_safety_test
  def testNonVariableGradients(self):
    b = 10
    scale = tf.convert_to_tensor(rng.rand(b) + 0.5)
    loc = tf.convert_to_tensor(rng.randn(b))
    sasnorm = tfd.SinhArcsinh(loc=loc, scale=scale, validate_args=True)

    x = rng.randn(5, b)
    with tf.GradientTape() as tape:
      tape.watch(loc)
      tape.watch(scale)
      y = sasnorm.log_prob(x)
    grads = tape.gradient(y, [loc, scale])
    self.assertLen(grads, 2)
    self.assertAllNotNone(grads)


if __name__ == '__main__':
  tf.test.main()
