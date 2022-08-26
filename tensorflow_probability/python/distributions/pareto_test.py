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
# Dependency imports
import numpy as np
from scipy import stats

import tensorflow.compat.v1 as tf1
import tensorflow.compat.v2 as tf

from tensorflow_probability.python.distributions import kullback_leibler
from tensorflow_probability.python.distributions import pareto
from tensorflow_probability.python.internal import test_util
from tensorflow_probability.python.math import gradient


@test_util.test_all_tf_execution_regimes
class ParetoTest(test_util.TestCase):

  def _scipy_pareto(self, concentration, scale):
    # In scipy pareto is defined with scale = 1, so we need to scale.
    return stats.pareto(concentration, scale=scale)

  def testParetoShape(self):
    scale = tf.constant([2.] * 5)
    concentration = tf.constant([2.] * 5)
    dist = pareto.Pareto(concentration, scale, validate_args=True)

    self.assertEqual(self.evaluate(dist.batch_shape_tensor()), (5,))
    self.assertEqual(dist.batch_shape, tf.TensorShape([5]))
    self.assertAllEqual(self.evaluate(dist.event_shape_tensor()), [])
    self.assertEqual(dist.event_shape, tf.TensorShape([]))

  def testParetoShapeBroadcast(self):
    scale = tf.constant([[3., 2.]])
    concentration = tf.constant([[4.], [5.], [6.]])
    dist = pareto.Pareto(concentration, scale, validate_args=True)

    self.assertAllEqual(self.evaluate(dist.batch_shape_tensor()), (3, 2))
    self.assertAllEqual(dist.batch_shape, tf.TensorShape([3, 2]))
    self.assertAllEqual(self.evaluate(dist.event_shape_tensor()), [])
    self.assertEqual(dist.event_shape, tf.TensorShape([]))

  def testInvalidScale(self):
    invalid_scales = [-.01, 0., -2.]
    concentration = 3.
    for scale in invalid_scales:
      with self.assertRaisesOpError('`scale` must be positive'):
        dist = pareto.Pareto(concentration, scale, validate_args=True)
        self.evaluate(dist.mean())

  def testInvalidConcentration(self):
    scale = 1.
    invalid_concentrations = [-.01, 0., -2.]
    for concentration in invalid_concentrations:
      with self.assertRaisesOpError('`concentration` must be positive'):
        dist = pareto.Pareto(concentration, scale, validate_args=True)
        self.evaluate(dist.mean())

  def testParetoLogPdf(self):
    batch_size = 6
    scale = tf.constant([3.] * batch_size)
    scale_v = 3.
    concentration = tf.constant([2.])
    concentration_v = 2.
    x = [3., 3.1, 4., 5., 6., 7.]
    dist = pareto.Pareto(concentration, scale, validate_args=True)
    log_prob = dist.log_prob(x)
    self.assertEqual(log_prob.shape, (6,))
    self.assertAllClose(
        self.evaluate(log_prob),
        self._scipy_pareto(concentration_v, scale_v).logpdf(x))

    pdf = dist.prob(x)
    self.assertEqual(pdf.shape, (6,))
    self.assertAllClose(
        self.evaluate(pdf),
        self._scipy_pareto(concentration_v, scale_v).pdf(x))

  def testParetoLogPdfValidateArgs(self):
    batch_size = 3
    scale = tf.constant([2., 3., 4.])
    concentration = tf.constant([2.] * batch_size)
    dist = pareto.Pareto(concentration, scale, validate_args=True)

    with self.assertRaisesOpError('not in the support'):
      x = tf1.placeholder_with_default([2., 3., 3.], shape=[3])
      log_prob = dist.log_prob(x)
      self.evaluate(log_prob)

    with self.assertRaisesOpError('not in the support'):
      x = tf1.placeholder_with_default([2., 2., 5.], shape=[3])
      log_prob = dist.log_prob(x)
      self.evaluate(log_prob)

    with self.assertRaisesOpError('not in the support'):
      x = tf1.placeholder_with_default([1., 3., 5.], shape=[3])
      log_prob = dist.log_prob(x)
      self.evaluate(log_prob)

  def testParetoLogPdfMultidimensional(self):
    batch_size = 6
    scale = tf.constant([[2., 4., 5.]] * batch_size)
    scale_v = [2., 4., 5.]
    concentration = tf.constant([[1.]] * batch_size)
    concentration_v = 1.

    x = np.array([[6., 7., 9.2, 5., 6., 7.]], dtype=np.float32).T

    dist = pareto.Pareto(concentration, scale, validate_args=True)
    log_prob = dist.log_prob(x)
    self.assertEqual(log_prob.shape, (6, 3))
    self.assertAllClose(
        self.evaluate(log_prob),
        self._scipy_pareto(concentration_v, scale_v).logpdf(x))

    prob = dist.prob(x)
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
    dist = pareto.Pareto(concentration, scale, validate_args=True)
    log_cdf = dist.log_cdf(x)
    self.assertEqual(log_cdf.shape, (6,))
    self.assertAllClose(
        self.evaluate(log_cdf),
        self._scipy_pareto(concentration_v, scale_v).logcdf(x))

    cdf = dist.cdf(x)
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

    dist = pareto.Pareto(concentration, scale, validate_args=True)
    log_cdf = dist.log_cdf(x)
    self.assertEqual(log_cdf.shape, (6, 3))
    self.assertAllClose(
        self.evaluate(log_cdf),
        self._scipy_pareto(concentration_v, scale_v).logcdf(x))

    cdf = dist.cdf(x)
    self.assertEqual(cdf.shape, (6, 3))
    self.assertAllClose(
        self.evaluate(cdf),
        self._scipy_pareto(concentration_v, scale_v).cdf(x))

  @test_util.numpy_disable_gradient_test
  def testParetoPDFGradientZeroOutsideSupport(self):
    scale = tf.constant(1.)
    concentration = tf.constant(3.)
    # Check the gradient on the undefined portion.
    x = scale - 1
    self.assertAlmostEqual(
        self.evaluate(
            gradient.value_and_gradient(
                pareto.Pareto(concentration, scale, validate_args=False).prob,
                x)[1]), 0.)

  @test_util.numpy_disable_gradient_test
  def testParetoCDFGradientZeroOutsideSupport(self):
    scale = tf.constant(1.)
    concentration = tf.constant(3.)
    # Check the gradient on the undefined portion.
    x = scale - 1
    self.assertAlmostEqual(
        self.evaluate(
            gradient.value_and_gradient(
                pareto.Pareto(concentration, scale, validate_args=False).cdf,
                x)[1]), 0.)

  def testParetoMean(self):
    scale = [1.4, 2., 2.5]
    concentration = [2., 3., 2.5]
    dist = pareto.Pareto(concentration, scale, validate_args=True)
    self.assertEqual(dist.mean().shape, (3,))
    self.assertAllClose(
        self.evaluate(dist.mean()),
        self._scipy_pareto(concentration, scale).mean())

  def testParetoMeanInf(self):
    scale = [1.4, 2., 2.5]
    concentration = [0.4, 0.9, 0.99]
    dist = pareto.Pareto(concentration, scale, validate_args=True)
    self.assertEqual(dist.mean().shape, (3,))

    self.assertTrue(np.all(np.isinf(self.evaluate(dist.mean()))))

  def testParetoVariance(self):
    scale = [1.4, 2., 2.5]
    concentration = [2., 3., 2.5]
    dist = pareto.Pareto(concentration, scale, validate_args=True)
    self.assertEqual(dist.variance().shape, (3,))
    self.assertAllClose(
        self.evaluate(dist.variance()),
        self._scipy_pareto(concentration, scale).var())

  def testParetoVarianceInf(self):
    scale = [1.4, 2., 2.5]
    concentration = [0.4, 0.9, 0.99]
    dist = pareto.Pareto(concentration, scale, validate_args=True)
    self.assertEqual(dist.variance().shape, (3,))
    self.assertTrue(np.all(np.isinf(self.evaluate(dist.variance()))))

  def testParetoStd(self):
    scale = [1.4, 2., 2.5]
    concentration = [2., 3., 2.5]
    dist = pareto.Pareto(concentration, scale, validate_args=True)
    self.assertEqual(dist.stddev().shape, (3,))
    self.assertAllClose(
        self.evaluate(dist.stddev()),
        self._scipy_pareto(concentration, scale).std())

  def testParetoMode(self):
    scale = [0.4, 1.4, 2., 2.5]
    concentration = [1., 2., 3., 2.5]
    dist = pareto.Pareto(concentration, scale, validate_args=True)
    self.assertEqual(dist.mode().shape, (4,))
    self.assertAllClose(self.evaluate(dist.mode()), scale)

  def testParetoSampleMean(self):
    scale = 4.
    concentration = 3.
    n = int(100e3)
    dist = pareto.Pareto(concentration, scale, validate_args=True)
    samples = dist.sample(n, seed=test_util.test_seed())
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
    n = int(6e5)
    dist = pareto.Pareto(concentration, scale, validate_args=True)
    samples = dist.sample(n, seed=test_util.test_seed(hardcoded_seed=123456))
    sample_values = self.evaluate(samples)
    self.assertEqual(samples.shape, (n,))
    self.assertEqual(sample_values.shape, (n,))
    self.assertAllClose(
        sample_values.var(),
        self._scipy_pareto(concentration, scale).var(),
        rtol=.05,
        atol=0)

  def testParetoSampleMultidimensionalMean(self):
    scale = np.array([np.arange(1, 21, dtype=np.float32)])
    concentration = 3.
    dist = pareto.Pareto(concentration, scale, validate_args=True)
    n = int(2e5)
    samples = dist.sample(n, seed=test_util.test_seed())
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
    dist = pareto.Pareto(concentration, scale, validate_args=True)
    n = int(800e3)
    samples = dist.sample(n, seed=test_util.test_seed(hardcoded_seed=123456))
    sample_values = self.evaluate(samples)
    self.assertEqual(samples.shape, (n, 1, 10))
    self.assertEqual(sample_values.shape, (n, 1, 10))

    self.assertAllClose(
        sample_values.var(axis=0),
        self._scipy_pareto(concentration, scale).var(),
        rtol=.05,
        atol=0)

  def testParetoParetoKLFinite(self):
    a_scale = np.arange(1., 5.)
    a_concentration = 1.
    b_scale = 1.
    b_concentration = np.arange(2., 10., 2)

    a = pareto.Pareto(
        concentration=a_concentration, scale=a_scale, validate_args=True)
    b = pareto.Pareto(
        concentration=b_concentration, scale=b_scale, validate_args=True)

    true_kl = (b_concentration * (np.log(a_scale) - np.log(b_scale)) +
               np.log(a_concentration) - np.log(b_concentration) +
               b_concentration / a_concentration - 1.)
    kl = kullback_leibler.kl_divergence(a, b)

    x = a.sample(
        int(3e5),
        seed=test_util.test_seed(hardcoded_seed=0, set_eager_seed=False))
    kl_samples = a.log_prob(x) - b.log_prob(x)

    kl_, kl_samples_ = self.evaluate([kl, kl_samples])
    self.assertAllClose(kl_, true_kl, atol=2e-15)
    self.assertAllMeansClose(kl_samples_, true_kl, axis=0, atol=0., rtol=1e-2)

    zero_kl = kullback_leibler.kl_divergence(a, a)
    true_zero_kl_, zero_kl_ = self.evaluate([tf.zeros_like(true_kl), zero_kl])
    self.assertAllEqual(true_zero_kl_, zero_kl_)

  def testParetoParetoKLInfinite(self):
    a = pareto.Pareto(concentration=1.0, scale=1., validate_args=True)
    b = pareto.Pareto(concentration=1.0, scale=2., validate_args=True)

    kl = kullback_leibler.kl_divergence(a, b)
    kl_ = self.evaluate(kl)
    self.assertAllEqual(np.inf, kl_)

  def testConcentrationVariable(self):
    c = tf.Variable([1., 2.])
    self.evaluate(c.initializer)
    d = pareto.Pareto(concentration=c, scale=1., validate_args=True)
    self.assertIs(d.concentration, c)
    self.evaluate(d.mean())
    with self.assertRaisesOpError('`concentration` must be positive'):
      with tf.control_dependencies([c.assign([-1, 2.])]):
        self.evaluate(d.mean())

  def testScaleVariable(self):
    s = tf.Variable([1., 2.])
    self.evaluate(s.initializer)
    d = pareto.Pareto(concentration=1., scale=s, validate_args=True)
    self.assertIs(d.scale, s)
    self.evaluate(d.mean())
    with self.assertRaisesOpError('`scale` must be positive'):
      with tf.control_dependencies([s.assign([-1, 2.])]):
        self.evaluate(d.mean())

  def testSupportBijectorOutsideRange(self):
    dist = pareto.Pareto(
        concentration=1., scale=[2., 5., 12.], validate_args=True)
    eps = 1e-6
    x = np.array([[2. - eps, 5. - eps, 12. - eps], [-0.5, 2.3, 10.]])
    bijector_inverse_x = dist.experimental_default_event_space_bijector(
        ).inverse(x)
    self.assertAllNan(self.evaluate(bijector_inverse_x))


if __name__ == '__main__':
  test_util.main()
