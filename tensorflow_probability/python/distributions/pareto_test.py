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
from scipy import stats

import tensorflow as tf
import tensorflow_probability as tfp

from tensorflow_probability.python.internal import test_case
from tensorflow_probability.python.internal import test_util as tfp_test_util
from tensorflow.python.framework import test_util  # pylint: disable=g-direct-tensorflow-import
tfd = tfp.distributions


@test_util.run_all_in_graph_and_eager_modes
class ParetoTest(test_case.TestCase):

  def _scipy_pareto(self, concentration, scale):
    # In scipy pareto is defined with scale = 1, so we need to scale.
    return stats.pareto(concentration, scale=scale)

  def testParetoShape(self):
    scale = tf.constant([2.] * 5)
    concentration = tf.constant([2.] * 5)
    pareto = tfd.Pareto(concentration, scale)

    self.assertEqual(self.evaluate(pareto.batch_shape_tensor()), (5,))
    self.assertEqual(pareto.batch_shape, tf.TensorShape([5]))
    self.assertAllEqual(self.evaluate(pareto.event_shape_tensor()), [])
    self.assertEqual(pareto.event_shape, tf.TensorShape([]))

  def testParetoShapeBroadcast(self):
    scale = tf.constant([[3., 2.]])
    concentration = tf.constant([[4.], [5.], [6.]])
    pareto = tfd.Pareto(concentration, scale)

    self.assertAllEqual(self.evaluate(pareto.batch_shape_tensor()), (3, 2))
    self.assertAllEqual(pareto.batch_shape, tf.TensorShape([3, 2]))
    self.assertAllEqual(self.evaluate(pareto.event_shape_tensor()), [])
    self.assertEqual(pareto.event_shape, tf.TensorShape([]))

  def testInvalidScale(self):
    invalid_scales = [-.01, 0., -2.]
    concentration = 3.
    for scale in invalid_scales:
      with self.assertRaisesOpError("Condition x > 0"):
        pareto = tfd.Pareto(concentration, scale, validate_args=True)
        self.evaluate(pareto.scale)

  def testInvalidConcentration(self):
    scale = 1.
    invalid_concentrations = [-.01, 0., -2.]
    for concentration in invalid_concentrations:
      with self.assertRaisesOpError("Condition x > 0"):
        pareto = tfd.Pareto(concentration, scale, validate_args=True)
        self.evaluate(pareto.concentration)

  def testParetoLogPdf(self):
    batch_size = 6
    scale = tf.constant([3.] * batch_size)
    scale_v = 3.
    concentration = tf.constant([2.])
    concentration_v = 2.
    x = [3., 3.1, 4., 5., 6., 7.]
    pareto = tfd.Pareto(concentration, scale)
    log_prob = pareto.log_prob(x)
    self.assertEqual(log_prob.shape, (6,))
    self.assertAllClose(
        self.evaluate(log_prob),
        self._scipy_pareto(concentration_v, scale_v).logpdf(x))

    pdf = pareto.prob(x)
    self.assertEqual(pdf.shape, (6,))
    self.assertAllClose(
        self.evaluate(pdf),
        self._scipy_pareto(concentration_v, scale_v).pdf(x))

  def testParetoLogPdfValidateArgs(self):
    batch_size = 3
    scale = tf.constant([2., 3., 4.])
    concentration = tf.constant([2.] * batch_size)
    pareto = tfd.Pareto(concentration, scale, validate_args=True)

    with self.assertRaisesOpError("not in the support"):
      x = tf.compat.v1.placeholder_with_default(input=[2., 3., 3.], shape=[3])
      log_prob = pareto.log_prob(x)
      self.evaluate(log_prob)

    with self.assertRaisesOpError("not in the support"):
      x = tf.compat.v1.placeholder_with_default(input=[2., 2., 5.], shape=[3])
      log_prob = pareto.log_prob(x)
      self.evaluate(log_prob)

    with self.assertRaisesOpError("not in the support"):
      x = tf.compat.v1.placeholder_with_default(input=[1., 3., 5.], shape=[3])
      log_prob = pareto.log_prob(x)
      self.evaluate(log_prob)

  def testParetoLogPdfMultidimensional(self):
    batch_size = 6
    scale = tf.constant([[2., 4., 5.]] * batch_size)
    scale_v = [2., 4., 5.]
    concentration = tf.constant([[1.]] * batch_size)
    concentration_v = 1.

    x = np.array([[6., 7., 9.2, 5., 6., 7.]], dtype=np.float32).T

    pareto = tfd.Pareto(concentration, scale)
    log_prob = pareto.log_prob(x)
    self.assertEqual(log_prob.shape, (6, 3))
    self.assertAllClose(
        self.evaluate(log_prob),
        self._scipy_pareto(concentration_v, scale_v).logpdf(x))

    prob = pareto.prob(x)
    self.assertEqual(prob.shape, (6, 3))
    self.assertAllClose(
        self.evaluate(prob),
        self._scipy_pareto(concentration_v, scale_v).pdf(x))

  def testParetoLogCdf(self):
    batch_size = 6
    scale = tf.constant([3.] * batch_size)
    scale_v = 3.
    concentration = tf.constant([2.])
    concentration_v = 2.
    x = [3., 3.1, 4., 5., 6., 7.]
    pareto = tfd.Pareto(concentration, scale)
    log_cdf = pareto.log_cdf(x)
    self.assertEqual(log_cdf.shape, (6,))
    self.assertAllClose(
        self.evaluate(log_cdf),
        self._scipy_pareto(concentration_v, scale_v).logcdf(x))

    cdf = pareto.cdf(x)
    self.assertEqual(cdf.shape, (6,))
    self.assertAllClose(
        self.evaluate(cdf),
        self._scipy_pareto(concentration_v, scale_v).cdf(x))

  def testParetoLogCdfMultidimensional(self):
    batch_size = 6
    scale = tf.constant([[2., 4., 5.]] * batch_size)
    scale_v = [2., 4., 5.]
    concentration = tf.constant([[1.]] * batch_size)
    concentration_v = 1.

    x = np.array([[6., 7., 9.2, 5., 6., 7.]], dtype=np.float32).T

    pareto = tfd.Pareto(concentration, scale)
    log_cdf = pareto.log_cdf(x)
    self.assertEqual(log_cdf.shape, (6, 3))
    self.assertAllClose(
        self.evaluate(log_cdf),
        self._scipy_pareto(concentration_v, scale_v).logcdf(x))

    cdf = pareto.cdf(x)
    self.assertEqual(cdf.shape, (6, 3))
    self.assertAllClose(
        self.evaluate(cdf),
        self._scipy_pareto(concentration_v, scale_v).cdf(x))

  def testParetoPDFGradientZeroOutsideSupport(self):
    scale = tf.constant(1.)
    concentration = tf.constant(3.)
    # Check the gradient on the undefined portion.
    x = scale - 1
    self.assertAlmostEqual(self.evaluate(tfp.math.value_and_gradient(
        tfd.Pareto(concentration, scale).prob, x)[1]), 0.)

  def testParetoCDFGradientZeroOutsideSupport(self):
    scale = tf.constant(1.)
    concentration = tf.constant(3.)
    # Check the gradient on the undefined portion.
    x = scale - 1
    self.assertAlmostEqual(self.evaluate(tfp.math.value_and_gradient(
        tfd.Pareto(concentration, scale).cdf, x)[1]), 0.)

  def testParetoMean(self):
    scale = [1.4, 2., 2.5]
    concentration = [2., 3., 2.5]
    pareto = tfd.Pareto(concentration, scale)
    self.assertEqual(pareto.mean().shape, (3,))
    self.assertAllClose(
        self.evaluate(pareto.mean()),
        self._scipy_pareto(concentration, scale).mean())

  def testParetoMeanInf(self):
    scale = [1.4, 2., 2.5]
    concentration = [0.4, 0.9, 0.99]
    pareto = tfd.Pareto(concentration, scale)
    self.assertEqual(pareto.mean().shape, (3,))

    self.assertTrue(
        np.all(np.isinf(self.evaluate(pareto.mean()))))

  def testParetoVariance(self):
    scale = [1.4, 2., 2.5]
    concentration = [2., 3., 2.5]
    pareto = tfd.Pareto(concentration, scale)
    self.assertEqual(pareto.variance().shape, (3,))
    self.assertAllClose(
        self.evaluate(pareto.variance()),
        self._scipy_pareto(concentration, scale).var())

  def testParetoVarianceInf(self):
    scale = [1.4, 2., 2.5]
    concentration = [0.4, 0.9, 0.99]
    pareto = tfd.Pareto(concentration, scale)
    self.assertEqual(pareto.variance().shape, (3,))
    self.assertTrue(
        np.all(np.isinf(self.evaluate(pareto.variance()))))

  def testParetoStd(self):
    scale = [1.4, 2., 2.5]
    concentration = [2., 3., 2.5]
    pareto = tfd.Pareto(concentration, scale)
    self.assertEqual(pareto.stddev().shape, (3,))
    self.assertAllClose(
        self.evaluate(pareto.stddev()),
        self._scipy_pareto(concentration, scale).std())

  def testParetoMode(self):
    scale = [0.4, 1.4, 2., 2.5]
    concentration = [1., 2., 3., 2.5]
    pareto = tfd.Pareto(concentration, scale)
    self.assertEqual(pareto.mode().shape, (4,))
    self.assertAllClose(self.evaluate(pareto.mode()), scale)

  def testParetoSampleMean(self):
    scale = 4.
    concentration = 3.
    n = int(100e3)
    pareto = tfd.Pareto(concentration, scale)
    samples = pareto.sample(n, seed=tfp_test_util.test_seed())
    sample_values = self.evaluate(samples)
    self.assertEqual(samples.shape, (n,))
    self.assertEqual(sample_values.shape, (n,))
    self.assertAllClose(
        sample_values.mean(),
        self._scipy_pareto(concentration, scale).mean(),
        rtol=.01,
        atol=0)

  def testParetoSampleVariance(self):
    scale = 1.
    concentration = 3.
    n = int(400e3)
    pareto = tfd.Pareto(concentration, scale)
    samples = pareto.sample(
        n, seed=tfp_test_util.test_seed(hardcoded_seed=123456))
    sample_values = self.evaluate(samples)
    self.assertEqual(samples.shape, (n,))
    self.assertEqual(sample_values.shape, (n,))
    self.assertAllClose(
        sample_values.var(),
        self._scipy_pareto(concentration, scale).var(),
        rtol=.03,
        atol=0)

  def testParetoSampleMultidimensionalMean(self):
    scale = np.array([np.arange(1, 21, dtype=np.float32)])
    concentration = 3.
    pareto = tfd.Pareto(concentration, scale)
    n = int(100e3)
    samples = pareto.sample(n, seed=tfp_test_util.test_seed())
    sample_values = self.evaluate(samples)
    self.assertEqual(samples.shape, (n, 1, 20))
    self.assertEqual(sample_values.shape, (n, 1, 20))
    self.assertAllClose(
        sample_values.mean(axis=0),
        self._scipy_pareto(concentration, scale).mean(),
        rtol=.01,
        atol=0)

  def testParetoSampleMultidimensionalVariance(self):
    scale = np.array([np.arange(1, 11, dtype=np.float32)])
    concentration = 4.
    pareto = tfd.Pareto(concentration, scale)
    n = int(800e3)
    samples = pareto.sample(
        n, seed=tfp_test_util.test_seed(hardcoded_seed=123456))
    sample_values = self.evaluate(samples)
    self.assertEqual(samples.shape, (n, 1, 10))
    self.assertEqual(sample_values.shape, (n, 1, 10))

    self.assertAllClose(
        sample_values.var(axis=0),
        self._scipy_pareto(concentration, scale).var(),
        rtol=.05,
        atol=0)

  def testParetoParetoKLFinite(self):
    a_scale = np.arange(1.0, 5.0)
    a_concentration = 1.0
    b_scale = 1.0
    b_concentration = np.arange(2.0, 10.0, 2)

    a = tfd.Pareto(concentration=a_concentration, scale=a_scale)
    b = tfd.Pareto(concentration=b_concentration, scale=b_scale)

    true_kl = (b_concentration * (np.log(a_scale) - np.log(b_scale)) +
               np.log(a_concentration) - np.log(b_concentration) +
               b_concentration / a_concentration - 1.0)
    kl = tfd.kl_divergence(a, b)

    x = a.sample(
        int(1e5),
        seed=tfp_test_util.test_seed(hardcoded_seed=0, set_eager_seed=False))
    kl_sample = tf.reduce_mean(
        input_tensor=a.log_prob(x) - b.log_prob(x), axis=0)

    kl_, kl_sample_ = self.evaluate([kl, kl_sample])
    self.assertAllEqual(true_kl, kl_)
    self.assertAllClose(true_kl, kl_sample_, atol=0., rtol=1e-2)

    zero_kl = tfd.kl_divergence(a, a)
    true_zero_kl_, zero_kl_ = self.evaluate([tf.zeros_like(true_kl), zero_kl])
    self.assertAllEqual(true_zero_kl_, zero_kl_)

  def testParetoParetoKLInfinite(self):
    a = tfd.Pareto(concentration=1.0, scale=1.0)
    b = tfd.Pareto(concentration=1.0, scale=2.0)

    kl = tfd.kl_divergence(a, b)
    kl_ = self.evaluate(kl)
    self.assertAllEqual(np.inf, kl_)

if __name__ == "__main__":
  tf.test.main()
