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
"""Tests for LogLogistic."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports

import numpy as np
import tensorflow.compat.v2 as tf
from tensorflow_probability.python import distributions as tfd
from tensorflow_probability.python.internal import test_util


@test_util.test_all_tf_execution_regimes
class LogLogiticTest(test_util.TestCase):

  def setUp(self):
    self._rng = np.random.RandomState(123)

  def testLogLogisticStats(self):
    scale = np.float32([3., 1.5, 0.75])
    concentration = np.float32([0.4, 1.1, 2.1])
    dist = tfd.LogLogistic(scale=scale, concentration=concentration,
                           validate_args=True)

    b = 1. / concentration
    mean = scale / np.sinc(b)
    mean[0] = np.nan
    self.assertAllClose(self.evaluate(dist.mean()), mean)

    variance = scale ** 2 * (1. / np.sinc(2 * b) - 1. / np.sinc(b) ** 2)
    variance[:2] = np.nan
    self.assertAllClose(self.evaluate(dist.variance()), variance)
    self.assertAllClose(self.evaluate(dist.stddev()),
                        np.sqrt(self.evaluate(dist.variance())))

    mode = scale * ((concentration - 1.) / (concentration + 1.)
                    ) ** (1. / concentration)
    mode[0] = np.nan
    self.assertAllClose(self.evaluate(dist.mode()), mode)

    entropy = np.log2(np.e ** 2 * scale / concentration)
    self.assertAllClose(self.evaluate(dist.entropy()), entropy)

  def testLogLogisticSample(self):
    scale, concentration = 1.5, 3.
    dist = tfd.LogLogistic(scale=scale, concentration=concentration,
                           validate_args=True)
    samples = self.evaluate(dist.sample(6000, seed=test_util.test_seed()))
    self.assertAllClose(np.mean(samples),
                        self.evaluate(dist.mean()),
                        atol=0.1)
    self.assertAllClose(np.std(samples),
                        self.evaluate(dist.stddev()),
                        atol=0.5)

  def testLogLogisticPDF(self):
    scale, concentration = 1.5, 0.4
    dist = tfd.LogLogistic(scale=scale, concentration=concentration,
                           validate_args=True)

    x = np.array([1e-4, 1.0, 2.0], dtype=np.float32)

    log_pdf = dist.log_prob(x)
    analytical_log_pdf = np.log(
        ((concentration/scale) * (x/scale) ** (concentration - 1)
         ) / (1 + (x/scale) ** concentration) ** 2)

    self.assertAllClose(self.evaluate(log_pdf), analytical_log_pdf)

  def testLogLogisticCDF(self):
    scale, concentration = 1.5, 0.4
    dist = tfd.LogLogistic(scale=scale, concentration=concentration,
                           validate_args=True)

    x = np.array([1e-4, 1.0, 2.0], dtype=np.float32)

    cdf = dist.cdf(x)
    analytical_cdf = 1. / (1 + (x / scale) ** (- concentration))
    self.assertAllClose(self.evaluate(cdf), analytical_cdf)

  def testAssertValidSample(self):
    dist = tfd.LogLogistic(scale=[1., 1., 4.], concentration=2.,
                           validate_args=True)
    with self.assertRaisesOpError('Sample must be non-negative.'):
      self.evaluate(dist.cdf([3., -0.2, 1.]))

  def testSupportBijectorOutsideRange(self):
    dist = tfd.LogLogistic(scale=1., concentration=2., validate_args=True)
    with self.assertRaisesOpError('must be greater than or equal to 0'):
      dist._experimental_default_event_space_bijector().inverse(
          [-4.2, -1e-6, -1.3])

if __name__ == '__main__':
  tf.test.main()
