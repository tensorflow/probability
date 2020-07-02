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
"""Tests for GLM families."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports

import numpy as np
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp
from tensorflow_probability.python import distributions as tfd
from tensorflow_probability.python.internal import test_util


class _GLMTestHarness(object):

  def testCallActualVsCallDefault(self):
    predicted_linear_response = np.stack([
        np.linspace(-7., -1e-3, 13),
        np.linspace(1e-3, 7, 13)]).reshape(2, 13, 1).astype(self.dtype)
    (
        expected_mean,
        expected_variance,
        expected_grad_mean,
    ) = tfp.glm.ExponentialFamily.__call__(
        self.model,
        predicted_linear_response)
    actual_mean, actual_variance, actual_grad_mean = self.model(
        predicted_linear_response)
    [
        expected_mean_,
        expected_variance_,
        expected_grad_mean_,
        actual_mean_,
        actual_variance_,
        actual_grad_mean_,
    ] = self.evaluate([
        expected_mean,
        expected_variance,
        expected_grad_mean,
        actual_mean,
        actual_variance,
        actual_grad_mean,
    ])
    self.assertAllClose(expected_mean_, actual_mean_,
                        atol=1e-6, rtol=1e-4)
    self.assertAllClose(expected_variance_, actual_variance_,
                        atol=1e-6, rtol=1e-3)
    self.assertAllClose(expected_grad_mean_, actual_grad_mean_,
                        atol=1e-6, rtol=1e-4)

  def testCallWorksCorrectly(self):
    predicted_linear_response = np.stack([
        np.linspace(-5., -1e-3, 11),
        np.linspace(1e-3, 5, 11)]).reshape(2, 11, 1).astype(self.dtype)
    expected_mean, expected_variance, expected_grad_mean = self.expected(
        predicted_linear_response)
    actual_mean, actual_variance, actual_grad_mean = self.model(
        predicted_linear_response)
    [
        expected_mean_,
        expected_variance_,
        expected_grad_mean_,
        actual_mean_,
        actual_variance_,
        actual_grad_mean_,
    ] = self.evaluate([
        expected_mean,
        expected_variance,
        expected_grad_mean,
        actual_mean,
        actual_variance,
        actual_grad_mean,
    ])
    self.assertAllClose(expected_mean_, actual_mean_,
                        atol=1e-6, rtol=1e-4)
    self.assertAllClose(expected_variance_, actual_variance_,
                        atol=1e-6, rtol=1e-3)
    self.assertAllClose(expected_grad_mean_, actual_grad_mean_,
                        atol=1e-6, rtol=1e-4)

  def testLogProbWorksCorrectly(self):
    predicted_linear_response = np.stack([
        np.linspace(-5., -1e-3, 11),
        np.linspace(1e-3, 5, 11)]).reshape(2, 11, 1).astype(self.dtype)
    actual_mean = self.expected.linear_model_to_mean_fn(
        predicted_linear_response)
    distribution = self.expected.distribution_fn(actual_mean)
    response = tf.cast(distribution.sample(seed=42), self.dtype)
    response = tf.identity(response, name='response')  # Disable bijector cache.
    expected_log_prob = distribution.log_prob(
        response, name='expected_log_prob')
    actual_log_prob = self.model.log_prob(
        response, predicted_linear_response)
    [
        expected_log_prob_,
        actual_log_prob_,
    ] = self.evaluate([
        expected_log_prob,
        actual_log_prob,
    ])
    self.assertAllClose(expected_log_prob_, actual_log_prob_,
                        atol=1e-6, rtol=1e-4)


@test_util.test_all_tf_execution_regimes
class BernoulliTest(test_util.TestCase, _GLMTestHarness):

  def setUp(self):
    super(BernoulliTest, self).setUp()
    self.dtype = np.float32
    self.model = tfp.glm.Bernoulli()
    self.expected = tfp.glm.CustomExponentialFamily(
        lambda mean: tfd.Bernoulli(probs=mean), tf.nn.sigmoid)


@test_util.test_all_tf_execution_regimes
class BernoulliNormalCDFTest(test_util.TestCase, _GLMTestHarness):

  def setUp(self):
    super(BernoulliNormalCDFTest, self).setUp()
    self.dtype = np.float32
    self.model = tfp.glm.BernoulliNormalCDF()
    def normal_cdf(r):
      r = tf.convert_to_tensor(value=r, name='r')
      n = tfd.Normal(loc=tf.zeros([], r.dtype.base_dtype),
                     scale=tf.ones([], r.dtype.base_dtype))
      return n.cdf(r)
    self.expected = tfp.glm.CustomExponentialFamily(
        lambda mean: tfd.Bernoulli(probs=mean), normal_cdf)


@test_util.test_all_tf_execution_regimes
class BinomialTest(test_util.TestCase, _GLMTestHarness):

  def setUp(self):
    super(BinomialTest, self).setUp()
    self.dtype = np.float32
    n = 2.
    self.model = tfp.glm.Binomial(n)
    self.expected = tfp.glm.CustomExponentialFamily(
        lambda mu: tfd.Binomial(n, probs=mu / n),
        lambda r: n * tf.nn.sigmoid(r))


@test_util.test_all_tf_execution_regimes
class GammaExpTest(test_util.TestCase, _GLMTestHarness):

  def setUp(self):
    super(GammaExpTest, self).setUp()
    self.dtype = np.float32
    c = 2.
    self.model = tfp.glm.GammaExp(c)
    self.expected = tfp.glm.CustomExponentialFamily(
        lambda mean: tfd.Gamma(concentration=c, rate=1. / mean),
        tf.math.exp)


@test_util.test_all_tf_execution_regimes
class GammaSoftplusTest(test_util.TestCase, _GLMTestHarness):

  def setUp(self):
    super(GammaSoftplusTest, self).setUp()
    self.dtype = np.float32
    c = 2.1
    self.model = tfp.glm.GammaSoftplus(c)
    self.expected = tfp.glm.CustomExponentialFamily(
        lambda mean: tfd.Gamma(concentration=c, rate=1. / mean),
        tf.nn.softplus)


@test_util.test_all_tf_execution_regimes
class LogNormalTest(test_util.TestCase, _GLMTestHarness):

  def setUp(self):
    super(LogNormalTest, self).setUp()
    self.dtype = np.float32
    s = 0.6
    self.model = tfp.glm.LogNormal(s)
    self.expected = tfp.glm.CustomExponentialFamily(
        lambda mean: tfd.LogNormal(tf.math.log(mean) - 0.5 * s**2, s),
        tf.math.exp)


@test_util.test_all_tf_execution_regimes
class LogNormalSoftplusTest(test_util.TestCase, _GLMTestHarness):

  def setUp(self):
    super(LogNormalSoftplusTest, self).setUp()
    self.dtype = np.float32
    s = 0.75
    self.model = tfp.glm.LogNormalSoftplus(s)
    self.expected = tfp.glm.CustomExponentialFamily(
        lambda mean: tfd.LogNormal(tf.math.log(mean) - 0.5 * s**2, s),
        tf.nn.softplus)


@test_util.test_all_tf_execution_regimes
class NormalTest(test_util.TestCase, _GLMTestHarness):

  def setUp(self):
    super(NormalTest, self).setUp()
    self.dtype = np.float32
    s = 0.9
    self.model = tfp.glm.Normal(s)
    self.expected = tfp.glm.CustomExponentialFamily(
        lambda mean: tfd.Normal(mean, s), tf.identity)


@test_util.test_all_tf_execution_regimes
class NormalReciprocalTest(test_util.TestCase, _GLMTestHarness):

  def setUp(self):
    super(NormalReciprocalTest, self).setUp()
    self.dtype = np.float32
    s = 0.85
    self.model = tfp.glm.NormalReciprocal(s)
    self.expected = tfp.glm.CustomExponentialFamily(
        lambda mean: tfd.Normal(mean, s), tf.math.reciprocal)


@test_util.test_all_tf_execution_regimes
class PoissonTest(test_util.TestCase, _GLMTestHarness):

  def setUp(self):
    super(PoissonTest, self).setUp()
    self.dtype = np.float32
    self.model = tfp.glm.Poisson()
    self.expected = tfp.glm.CustomExponentialFamily(
        lambda mean: tfd.Poisson(rate=mean), tf.math.exp)


@test_util.test_all_tf_execution_regimes
class PoissonSoftplusTest(test_util.TestCase, _GLMTestHarness):

  def setUp(self):
    super(PoissonSoftplusTest, self).setUp()
    self.dtype = np.float32
    self.model = tfp.glm.PoissonSoftplus()
    self.expected = tfp.glm.CustomExponentialFamily(
        lambda mean: tfd.Poisson(rate=mean), tf.nn.softplus)


@test_util.test_all_tf_execution_regimes
class NegativeBinomialTest(test_util.TestCase, _GLMTestHarness):

  def setUp(self):
    super(NegativeBinomialTest, self).setUp()
    self.dtype = np.float32
    n = 2.
    self.model = tfp.glm.NegativeBinomial(n)
    self.expected = tfp.glm.CustomExponentialFamily(
        lambda mean: tfd.NegativeBinomial(  # pylint: disable=g-long-lambda
            total_count=n,
            logits=tf.math.log(mean) - tf.math.log(n)),
        tf.math.exp)


@test_util.test_all_tf_execution_regimes
class NegativeBinomialSoftplusTest(test_util.TestCase, _GLMTestHarness):

  def setUp(self):
    super(NegativeBinomialSoftplusTest, self).setUp()
    self.dtype = np.float32
    n = 2.1
    self.model = tfp.glm.NegativeBinomialSoftplus(n)
    self.expected = tfp.glm.CustomExponentialFamily(
        lambda mean: tfd.NegativeBinomial(  # pylint: disable=g-long-lambda
            total_count=n,
            logits=tf.math.log(mean) - tf.math.log(n)),
        tf.math.softplus)


if __name__ == '__main__':
  tf.test.main()
