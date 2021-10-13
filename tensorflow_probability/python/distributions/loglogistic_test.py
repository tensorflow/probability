# Copyright 2020 The TensorFlow Probability Authors.
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
"""Tests for LogLogistic."""

# Dependency imports

import numpy as np
from scipy import stats

from tensorflow_probability.python import distributions as tfd
from tensorflow_probability.python.internal import test_util


@test_util.test_all_tf_execution_regimes
class LogLogisticTest(test_util.TestCase):

  def testLogLogisticMean(self):
    log_logistic_scale = np.float32([3., 1.5, 0.75])
    loc = np.log(log_logistic_scale)
    scale = np.float32([0.8, 0.9, 0.5])
    dist = tfd.LogLogistic(loc=loc, scale=scale, validate_args=True)

    self.assertAllClose(
        self.evaluate(dist.mean()),
        stats.fisk.mean(loc=0., scale=log_logistic_scale, c=1. / scale))

  def testLogLogisticMeanNoNanAllowed(self):
    log_logistic_scale = np.float32([3., 1.5, 0.75])
    loc = np.log(log_logistic_scale)
    scale = np.float32([0.4, 0.6, 1.5])
    dist = tfd.LogLogistic(
        loc=loc, scale=scale, validate_args=True, allow_nan_stats=False)

    with self.assertRaisesOpError('Condition x < y.*'):
      self.evaluate(dist.mean())

  def testLogLogisticVariance(self):
    log_logistic_scale = np.float32([3., 1.5, 0.75])
    loc = np.log(log_logistic_scale)
    scale = np.float32([0.4, 0.3, 0.2])
    dist = tfd.LogLogistic(loc=loc, scale=scale, validate_args=True)

    # scipy.stats.fisk.var only works on scalars, so we calculate this in a
    # loop:
    scipy_var = [stats.fisk.var(
        loc=0., scale=s, c=1. / c) for (s, c) in zip(log_logistic_scale, scale)]

    self.assertAllClose(self.evaluate(dist.variance()), scipy_var)
    self.assertAllClose(self.evaluate(dist.stddev()), np.sqrt(scipy_var))

  def testLogLogisticVarianceNoNanAllowed(self):
    log_logistic_scale = np.float32([3., 1.5, 0.75])
    loc = np.log(log_logistic_scale)
    scale = np.float32([0.4, 0.6, 1.5])
    dist = tfd.LogLogistic(
        loc=loc, scale=scale, validate_args=True, allow_nan_stats=False)

    with self.assertRaisesOpError('Condition x < y.*'):
      self.evaluate(dist.variance())

    with self.assertRaisesOpError('Condition x < y.*'):
      self.evaluate(dist.stddev())

  def testLogLogisticMode(self):
    log_logistic_scale = np.float32([3., 1.5, 0.75])
    loc = np.log(log_logistic_scale)
    scale = np.float32([0.4, 0.6, 1.5])
    dist = tfd.LogLogistic(loc=loc, scale=scale, validate_args=True)

    mode = log_logistic_scale * ((1. - scale) / (1. + scale))**scale
    mode[2] = 0.
    self.assertAllClose(self.evaluate(dist.mode()), mode)

  def testLogLogisticEntropy(self):
    log_logistic_scale = np.float32([3., 1.5, 0.75])
    loc = np.log(log_logistic_scale)
    scale = np.float32([0.4, 0.6, 1.5])
    dist = tfd.LogLogistic(loc=loc, scale=scale, validate_args=True)

    self.assertAllClose(
        self.evaluate(dist.entropy()),
        stats.fisk.entropy(loc=0., scale=log_logistic_scale, c=1. / scale))

  def testLogLogisticSample(self):
    log_logistic_scale = 1.5
    loc = np.log(log_logistic_scale).astype(np.float32)
    scale = 0.33
    dist = tfd.LogLogistic(loc=loc, scale=scale, validate_args=True)
    samples = self.evaluate(dist.sample(6000, seed=test_util.test_seed()))
    self.assertAllClose(np.mean(samples), self.evaluate(dist.mean()), atol=0.1)
    self.assertAllClose(np.std(samples), self.evaluate(dist.stddev()), atol=0.5)

  def testLogLogisticPDFLocBatch(self):
    log_logistic_scale = [1.5, 2.]
    loc = np.log(log_logistic_scale)
    scale = 2.5
    dist = tfd.LogLogistic(loc=loc, scale=scale, validate_args=True)

    x = np.array([1.], dtype=np.float32)

    pdf = dist.prob(x)

    self.assertAllClose(
        self.evaluate(pdf),
        stats.fisk.pdf(x, loc=0., scale=log_logistic_scale, c=1. / scale))

  def testLogLogisticPDFScaleBatch(self):
    log_logistic_scale = 1.5
    loc = np.log(log_logistic_scale)
    scale = np.array([2.5, 5.])
    dist = tfd.LogLogistic(loc=loc, scale=scale, validate_args=True)

    x = np.array([[1e-4, 1.0], [1.5, 2.0]], dtype=np.float32)

    pdf = dist.prob(x)

    self.assertAllClose(
        self.evaluate(pdf),
        stats.fisk.pdf(x, loc=0., scale=log_logistic_scale, c=1. / scale))

  def testLogLogisticLogPDFLocBatch(self):
    log_logistic_scale = [1.5, 2.]
    loc = np.log(log_logistic_scale)
    scale = 2.5
    dist = tfd.LogLogistic(loc=loc, scale=scale, validate_args=True)

    x = np.array([[1e-4, 1.0], [3.0, 2.0]], dtype=np.float32)

    log_pdf = dist.log_prob(x)

    self.assertAllClose(
        self.evaluate(log_pdf),
        stats.fisk.logpdf(x, loc=0., scale=log_logistic_scale, c=1. / scale))

  def testLogLogisticLogPDFScaleBatch(self):
    log_logistic_scale = [1.5, 2.]
    loc = np.log(log_logistic_scale)
    scale = np.array([2.5, 5.])
    dist = tfd.LogLogistic(loc=loc, scale=scale, validate_args=True)

    x = np.array([2.0], dtype=np.float32)

    log_pdf = dist.log_prob(x)

    self.assertAllClose(
        self.evaluate(log_pdf),
        stats.fisk.logpdf(x, loc=0., scale=log_logistic_scale, c=1. / scale))

  def testLogLogisticCDFLocBatch(self):
    log_logistic_scale = [0.5, 1.5]
    loc = np.log(log_logistic_scale)
    scale = 2.5
    dist = tfd.LogLogistic(loc=loc, scale=scale, validate_args=True)

    x = np.array([1e-4], dtype=np.float32)

    cdf = dist.cdf(x)
    self.assertAllClose(
        self.evaluate(cdf),
        stats.fisk.cdf(x, loc=0., scale=log_logistic_scale, c=1. / scale))

  def testLogLogisticCDFScaleBatch(self):
    log_logistic_scale = [0.5, 1.5]
    loc = np.log(log_logistic_scale)
    scale = np.array([0.5, 2.])
    dist = tfd.LogLogistic(loc=loc, scale=scale, validate_args=True)

    x = np.array([[1e-4, 2.0], [5.0, 2.0]], dtype=np.float32)

    cdf = dist.cdf(x)
    self.assertAllClose(
        self.evaluate(cdf),
        stats.fisk.cdf(x, loc=0., scale=log_logistic_scale, c=1. / scale))

  def testLogLogisticLogCDFLocBatch(self):
    log_logistic_scale = [0.75, 2.5]
    loc = np.log(log_logistic_scale)
    scale = 2.5
    dist = tfd.LogLogistic(loc=loc, scale=scale, validate_args=True)

    x = np.array([1e-4], dtype=np.float32)

    log_cdf = dist.log_cdf(x)
    self.assertAllClose(
        self.evaluate(log_cdf),
        stats.fisk.logcdf(x, loc=0., scale=log_logistic_scale, c=1. / scale))

  def testLogLogisticLogCDFScaleBatch(self):
    log_logistic_scale = [0.75, 2.5]
    loc = np.log(log_logistic_scale)
    scale = np.array([0.3, 2.1])
    dist = tfd.LogLogistic(loc=loc, scale=scale, validate_args=True)

    x = np.array([[1e-4, 1.0], [5.0, 2.0]], dtype=np.float32)

    log_cdf = dist.log_cdf(x)
    self.assertAllClose(
        self.evaluate(log_cdf),
        stats.fisk.logcdf(x, loc=0., scale=log_logistic_scale, c=1. / scale))

  def testLogLogisticLogSurvivalLocBatch(self):
    log_logistic_scale = [0.42, 1.3]
    loc = np.log(log_logistic_scale)
    scale = 2.5
    dist = tfd.LogLogistic(loc=loc, scale=scale, validate_args=True)

    x = np.array([[1e-4, 1.0], [3., 2.0]], dtype=np.float32)

    logsf = dist.log_survival_function(x)
    self.assertAllClose(
        self.evaluate(logsf),
        stats.fisk.logsf(x, loc=0., scale=log_logistic_scale, c=1. / scale))

  def testLogLogisticLogSurvivalScaleBatch(self):
    log_logistic_scale = 1.5
    loc = np.log(log_logistic_scale)
    scale = np.array([1.2, 5.1])
    dist = tfd.LogLogistic(loc=loc, scale=scale, validate_args=True)

    x = np.array([1.0], dtype=np.float32)

    logsf = dist.log_survival_function(x)
    self.assertAllClose(
        self.evaluate(logsf),
        stats.fisk.logsf(x, loc=0., scale=log_logistic_scale, c=1. / scale))

  def testAssertValidSample(self):
    dist = tfd.LogLogistic(
        loc=np.log([1., 1., 4.]), scale=2., validate_args=True)
    with self.assertRaisesOpError('Sample must be non-negative.'):
      self.evaluate(dist.cdf([3., -0.2, 1.]))

  def testSupportBijectorOutsideRange(self):
    dist = tfd.LogLogistic(loc=0., scale=0.5, validate_args=True)
    with self.assertRaisesOpError('must be greater than or equal to 0'):
      dist.experimental_default_event_space_bijector().inverse(
          [-4.2, -1e-6, -1.3])


if __name__ == '__main__':
  test_util.main()
