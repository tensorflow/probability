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

import tensorflow as tf
import tensorflow_probability as tfp

from tensorflow_probability.python.internal import test_util as tfp_test_util
from tensorflow.python.framework import test_util  # pylint: disable=g-direct-tensorflow-import,g-import-not-at-top

tfd = tfp.distributions
rng = np.random.RandomState(123)


@test_util.run_all_in_graph_and_eager_modes
class SinhArcsinhTest(tf.test.TestCase):

  def test_default_is_same_as_normal(self):
    b = 10
    scale = rng.rand(b) + 0.5
    loc = rng.randn(b)
    norm = tfd.Normal(loc=loc, scale=scale, validate_args=True)
    sasnorm = tfd.SinhArcsinh(loc=loc, scale=scale, validate_args=True)

    x = rng.randn(5, b)
    norm_pdf, sasnorm_pdf = self.evaluate([norm.prob(x), sasnorm.prob(x)])
    self.assertAllClose(norm_pdf, sasnorm_pdf)

    norm_samps, sasnorm_samps = self.evaluate([
        norm.sample(10000, seed=tfp_test_util.test_seed()),
        sasnorm.sample(10000, seed=tfp_test_util.test_seed())
    ])
    self.assertAllClose(loc, sasnorm_samps.mean(axis=0), atol=0.1)
    self.assertAllClose(
        norm_samps.mean(axis=0), sasnorm_samps.mean(axis=0), atol=0.1)
    self.assertAllClose(
        norm_samps.std(axis=0), sasnorm_samps.std(axis=0), atol=0.1)

  def test_broadcast_params_dynamic(self):
    loc = tf.compat.v1.placeholder_with_default(input=rng.rand(5), shape=None)
    scale = tf.compat.v1.placeholder_with_default(
        input=np.float64(rng.rand()), shape=None)
    skewness = tf.compat.v1.placeholder_with_default(
        input=rng.rand(5), shape=None)
    sasnorm = tfd.SinhArcsinh(
        loc=loc, scale=scale, skewness=skewness, validate_args=True)

    samp = self.evaluate(sasnorm.sample())
    self.assertAllEqual((5,), samp.shape)

  def test_passing_in_laplace_plus_defaults_is_same_as_laplace(self):
    b = 10
    scale = rng.rand(b) + 0.5
    loc = rng.randn(b)
    lap = tfd.Laplace(loc=loc, scale=scale, validate_args=True)
    saslap = tfd.SinhArcsinh(
        loc=loc,
        scale=scale,
        distribution=tfd.Laplace(np.float64(0), np.float64(1)),
        validate_args=True)

    x = rng.randn(5, b)
    lap_pdf, saslap_pdf = self.evaluate([lap.prob(x), saslap.prob(x)])
    self.assertAllClose(lap_pdf, saslap_pdf)

    lap_samps, saslap_samps = self.evaluate([
        lap.sample(10000, seed=tfp_test_util.test_seed()),
        saslap.sample(10000, seed=tfp_test_util.test_seed())
    ])
    self.assertAllClose(loc, saslap_samps.mean(axis=0), atol=0.1)
    self.assertAllClose(
        lap_samps.mean(axis=0), saslap_samps.mean(axis=0), atol=0.1)
    self.assertAllClose(
        lap_samps.std(axis=0), saslap_samps.std(axis=0), atol=0.1)

  def test_tailweight_small_gives_fewer_outliers_than_normal(self):
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
        norm.sample(int(5e5), seed=tfp_test_util.test_seed()),
        sasnorm.sample(int(5e5), seed=tfp_test_util.test_seed())
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

  def test_tailweight_large_gives_more_outliers_than_normal(self):
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
        norm.sample(int(5e5), seed=tfp_test_util.test_seed()),
        sasnorm.sample(int(5e5), seed=tfp_test_util.test_seed())
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

  def test_positive_skewness_moves_mean_to_the_right(self):
    batch_size = 10
    scale = rng.rand(batch_size) + 0.5
    loc = rng.randn(batch_size)
    sasnorm = tfd.SinhArcsinh(
        loc=loc, scale=scale, skewness=3.0, validate_args=True)

    sasnorm_samps = self.evaluate(
        sasnorm.sample(10000, seed=tfp_test_util.test_seed()))
    np.testing.assert_array_less(loc, sasnorm_samps.mean(axis=0))

  def test_pdf_reflected_for_negative_skewness(self):
    sas_pos_skew = tfd.SinhArcsinh(
        loc=0., scale=1., skewness=2., validate_args=True)
    sas_neg_skew = tfd.SinhArcsinh(
        loc=0., scale=1., skewness=-2., validate_args=True)
    x = np.linspace(-2, 2, num=5).astype(np.float32)
    self.assertAllClose(*self.evaluate(
        [sas_pos_skew.prob(x), sas_neg_skew.prob(x[::-1])]))


if __name__ == "__main__":
  tf.test.main()
