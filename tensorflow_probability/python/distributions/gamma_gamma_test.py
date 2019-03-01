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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from tensorflow_probability.python.internal import test_util as tfp_test_util
tfd = tfp.distributions
from tensorflow.python.framework import test_util  # pylint: disable=g-direct-tensorflow-import,g-import-not-at-top


@test_util.run_all_in_graph_and_eager_modes
class GammaGammaTest(tf.test.TestCase):

  def testGammaGammaShape(self):
    gg = tfd.GammaGamma(
        concentration=[[2.], [4.]],
        mixing_concentration=[1., 2., 3.],
        mixing_rate=0.5)

    self.assertAllEqual(self.evaluate(gg.batch_shape_tensor()), [2, 3])
    self.assertEqual(gg.batch_shape, tf.TensorShape([2, 3]))
    self.assertAllEqual(self.evaluate(gg.event_shape_tensor()), [])
    self.assertEqual(gg.event_shape, tf.TensorShape([]))

  def testGammaGammaLogPDF(self):
    batch_size = 5
    alpha = tf.constant([2.] * batch_size)
    alpha0 = tf.constant([3.] * batch_size)
    beta0 = tf.constant([4.] * batch_size)
    x = np.array([6.] * batch_size, dtype=np.float32)

    # Let
    #   alpha = concentration = 2.
    #   alpha0 = mixing_concentration = 3.,
    #   beta0 = mixing_rate = 4.
    #
    # See the PDF derivation in formula (1) of
    # http://www.brucehardie.com/notes/025/gamma_gamma.pdf.
    #
    #               x**(alpha - 1) * beta0**alpha0
    # prob(x=6) = ------------------------------------------------
    #             B(alpha, alpha0) * (x + beta0)**(alpha + alpha0)
    #
    #                6 * 4**3
    #           = --------------- = 0.04608
    #             B(2, 3) * 10**5
    #
    # log_prob(x=6) = -3.077376
    expected_log_pdf = [-3.077376] * batch_size

    gg = tfd.GammaGamma(
        concentration=alpha, mixing_concentration=alpha0, mixing_rate=beta0)
    log_pdf = gg.log_prob(x)
    self.assertEqual(log_pdf.shape, (5,))
    self.assertAllClose(self.evaluate(log_pdf), expected_log_pdf)

  def testGammaGammaLogPDFAtZero(self):
    # When concentration = 1., the log pdf should be finite.
    gg = tfd.GammaGamma(
        concentration=1.,
        mixing_concentration=[0.01, 0.1, 2, 3],
        mixing_rate=[0.01, 0.1, 2, 3])
    log_pdf = self.evaluate(gg.log_prob(0.))
    self.assertAllEqual(
        np.ones_like(log_pdf, dtype=np.bool), np.isfinite(log_pdf))

  def testGammaGammaLogPDFMultidimensional(self):
    batch_size = 6
    alpha = tf.constant([[2., 4.]] * batch_size)
    alpha0 = tf.constant([[3., 6.]] * batch_size)
    beta0 = tf.constant([[4., 8.]] * batch_size)
    x = np.array([[2.5, 2.5, 4.0, 0.1, 1.0, 2.0]], dtype=np.float32).T

    gg = tfd.GammaGamma(
        concentration=alpha, mixing_concentration=alpha0, mixing_rate=beta0)
    log_pdf = gg.log_prob(x)
    self.assertEqual(log_pdf.shape, (6, 2))

  def testGammaGammaLogPDFMultidimensionalBroadcasting(self):
    batch_size = 6
    alpha = tf.constant([[2., 4.]] * batch_size)
    alpha0 = tf.constant(3.0)
    beta0 = tf.constant([4., 8.])
    x = np.array([[2.5, 2.5, 4.0, 0.1, 1.0, 2.0]], dtype=np.float32).T

    gg = tfd.GammaGamma(
        concentration=alpha, mixing_concentration=alpha0, mixing_rate=beta0)
    log_pdf = gg.log_prob(x)
    self.assertEqual(log_pdf.shape, (6, 2))

  def testGammaGammaMeanAllDefined(self):
    alpha_v = np.array([2., 4.])
    alpha0_v = np.array([3., 6.])
    beta0_v = np.array([4., 8.])
    expected_mean = alpha_v * beta0_v / (alpha0_v - 1.)

    gg = tfd.GammaGamma(
        concentration=alpha_v,
        mixing_concentration=alpha0_v,
        mixing_rate=beta0_v)
    self.assertEqual(gg.mean().shape, (2,))
    self.assertAllClose(self.evaluate(gg.mean()), expected_mean)

  def testGammaGammaMeanAllowNanStats(self):
    # Mean will not be defined for the first entry.
    alpha_v = np.array([2., 4.])
    alpha0_v = np.array([1., 6.])
    beta0_v = np.array([4., 8.])

    gg = tfd.GammaGamma(
        concentration=alpha_v,
        mixing_concentration=alpha0_v,
        mixing_rate=beta0_v,
        allow_nan_stats=False)
    with self.assertRaisesOpError('x < y'):
      self.evaluate(gg.mean())

  def testGammaGammaMeanNanStats(self):
    # Mean will not be defined for the first entry.
    alpha_v = np.array([2., 4.])
    alpha0_v = np.array([1., 6.])
    beta0_v = np.array([4., 8.])
    expected_mean = np.array([np.nan, 6.4])

    gg = tfd.GammaGamma(
        concentration=alpha_v,
        mixing_concentration=alpha0_v,
        mixing_rate=beta0_v)
    self.assertEqual(gg.mean().shape, (2,))
    self.assertAllClose(self.evaluate(gg.mean()), expected_mean)

  def testGammaGammaSample(self):
    with tf.compat.v1.Session():
      alpha_v = 2.0
      alpha0_v = 3.0
      beta0_v = 5.0
      n = 100000

      gg = tfd.GammaGamma(
          concentration=alpha_v,
          mixing_concentration=alpha0_v,
          mixing_rate=beta0_v)
      samples = gg.sample(n, seed=tfp_test_util.test_seed())
      sample_values = self.evaluate(samples)
      self.assertEqual(samples.shape, (n,))
      self.assertEqual(sample_values.shape, (n,))
      self.assertAllClose(
          sample_values.mean(), self.evaluate(gg.mean()), rtol=.01)

  def testGammaGammaSampleMultidimensionalMean(self):
    alpha_v = np.array([np.arange(3, 103, dtype=np.float32)])  # 1 x 100
    alpha0_v = 2.
    beta0_v = np.array([np.arange(1, 11, dtype=np.float32)]).T  # 10 x 1
    n = 10000

    gg = tfd.GammaGamma(
        concentration=alpha_v,
        mixing_concentration=alpha0_v,
        mixing_rate=beta0_v)
    samples = gg.sample(n, seed=tfp_test_util.test_seed())
    sample_values = self.evaluate(samples)
    self.assertEqual(samples.shape, (n, 10, 100))
    self.assertEqual(sample_values.shape, (n, 10, 100))
    self.assertAllClose(
        sample_values.mean(axis=0), self.evaluate(gg.mean()), rtol=.08)


if __name__ == '__main__':
  tf.test.main()
