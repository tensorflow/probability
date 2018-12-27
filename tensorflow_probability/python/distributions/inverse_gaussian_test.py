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

import numpy as np
from scipy import stats
import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions
tfe = tf.contrib.eager


def _scipy_invgauss(loc, concentration):
  # Wrapper of scipy's invgauss function, which is used to generate expected
  # output.
  # scipy uses a different parameterization.
  # See https://github.com/scipy/scipy/issues/4654.
  return stats.invgauss(mu=loc/concentration, scale=concentration)


@tfe.run_all_tests_in_graph_and_eager_modes
class _InverseGaussianTest(object):

  def make_tensor(self, x):
    x = tf.cast(x, self.dtype)
    return tf.placeholder_with_default(
        input=x, shape=x.shape if self.use_static_shape else None)

  def testInverseGaussianShape(self):
    loc = self.make_tensor([2.] * 5)
    concentration = self.make_tensor([2.] * 5)
    inverse_gaussian = tfd.InverseGaussian(loc, concentration)

    self.assertEqual(self.evaluate(inverse_gaussian.batch_shape_tensor()), (5,))
    if self.use_static_shape:
      self.assertEqual(inverse_gaussian.batch_shape, tf.TensorShape([5]))
    self.assertAllEqual(self.evaluate(inverse_gaussian.event_shape_tensor()),
                        [])
    self.assertEqual(inverse_gaussian.event_shape, tf.TensorShape([]))

  def testInverseGaussianShapeBroadcast(self):
    loc = self.make_tensor([[4.], [5.], [6.]])
    concentration = self.make_tensor([[3., 2.]])
    inverse_gaussian = tfd.InverseGaussian(loc, concentration)

    self.assertAllEqual(self.evaluate(inverse_gaussian.batch_shape_tensor()),
                        (3, 2))
    if self.use_static_shape:
      self.assertAllEqual(inverse_gaussian.batch_shape, tf.TensorShape([3, 2]))
    self.assertAllEqual(self.evaluate(inverse_gaussian.event_shape_tensor()),
                        [])
    self.assertEqual(inverse_gaussian.event_shape, tf.TensorShape([]))

  def testInvalidLoc(self):
    invalid_locs = [-.01, 0., -2.]
    concentration_v = 1.

    for loc_v in invalid_locs:
      with self.assertRaisesOpError("Condition x > 0"):
        inverse_gaussian = tfd.InverseGaussian(
            self.make_tensor(loc_v),
            self.make_tensor(concentration_v),
            validate_args=True)
        self.evaluate(inverse_gaussian.loc)

  def testInvalidConcentration(self):
    loc_v = 3.
    invalid_concentrations = [-.01, 0., -2.]

    for concentration_v in invalid_concentrations:
      with self.assertRaisesOpError("Condition x > 0"):
        inverse_gaussian = tfd.InverseGaussian(
            self.make_tensor(loc_v),
            self.make_tensor(concentration_v),
            validate_args=True)
        self.evaluate(inverse_gaussian.concentration)

  def testInverseGaussianLogPdf(self):
    batch_size = 6
    loc_v = 2.
    concentration_v = 3.
    x_v = [3., 3.1, 4., 5., 6., 7.]
    inverse_gaussian = tfd.InverseGaussian(
        self.make_tensor([loc_v] * batch_size),
        self.make_tensor([concentration_v] * batch_size))

    log_prob = inverse_gaussian.log_prob(self.make_tensor(x_v))
    if self.use_static_shape:
      self.assertEqual(log_prob.shape, (6,))
    self.assertAllClose(
        self.evaluate(log_prob),
        _scipy_invgauss(loc_v, concentration_v).logpdf(x_v))

    pdf = inverse_gaussian.prob(self.make_tensor(x_v))
    if self.use_static_shape:
      self.assertEqual(pdf.shape, (6,))
    self.assertAllClose(
        self.evaluate(pdf),
        _scipy_invgauss(loc_v, concentration_v).pdf(x_v))

  def testInverseGaussianLogPdfValidateArgs(self):
    batch_size = 2
    loc = self.make_tensor([2.] * batch_size)
    concentration = self.make_tensor([2., 3.])
    x = self.make_tensor([-1., 2.])
    inverse_gaussian = tfd.InverseGaussian(loc, concentration,
                                           validate_args=True)

    with self.assertRaisesOpError("x must be positive."):
      self.evaluate(inverse_gaussian.log_prob(x))

  def testInverseGaussianPdfValidateArgs(self):
    batch_size = 2
    loc = self.make_tensor([2.] * batch_size)
    concentration = self.make_tensor([2., 3.])
    x = self.make_tensor([-1., 2.])
    inverse_gaussian = tfd.InverseGaussian(loc, concentration,
                                           validate_args=True)

    with self.assertRaisesOpError("x must be positive."):
      self.evaluate(inverse_gaussian.prob(x))

  def testInverseGaussianLogPdfMultidimensional(self):
    batch_size = 6
    loc_v = 1.
    concentration_v = [2., 4., 5.]
    x_v = np.array([[6., 7., 9.2, 5., 6., 7.]]).T
    inverse_gaussian = tfd.InverseGaussian(
        self.make_tensor([[loc_v]] * batch_size),
        self.make_tensor([concentration_v] * batch_size))

    log_prob = inverse_gaussian.log_prob(self.make_tensor(x_v))
    if self.use_static_shape:
      self.assertEqual(log_prob.shape, (6, 3))
    self.assertAllClose(
        self.evaluate(log_prob),
        _scipy_invgauss(loc_v, np.array(concentration_v)).logpdf(x_v))

    prob = inverse_gaussian.prob(self.make_tensor(x_v))
    if self.use_static_shape:
      self.assertEqual(prob.shape, (6, 3))
    self.assertAllClose(
        self.evaluate(prob),
        _scipy_invgauss(loc_v, np.array(concentration_v)).pdf(x_v))

  def testInverseGaussianLogCdf(self):
    batch_size = 6
    loc_v = 2.
    concentration_v = 3.
    x_v = [3., 3.1, 4., 5., 6., 7.]
    inverse_gaussian = tfd.InverseGaussian(
        self.make_tensor([loc_v] * batch_size),
        self.make_tensor([concentration_v] * batch_size))

    log_cdf = inverse_gaussian.log_cdf(self.make_tensor(x_v))
    if self.use_static_shape:
      self.assertEqual(log_cdf.shape, (6,))
    self.assertAllClose(
        self.evaluate(log_cdf),
        _scipy_invgauss(loc_v, concentration_v).logcdf(x_v))

    cdf = inverse_gaussian.cdf(self.make_tensor(x_v))
    if self.use_static_shape:
      self.assertEqual(cdf.shape, (6,))
    self.assertAllClose(
        self.evaluate(cdf),
        _scipy_invgauss(loc_v, concentration_v).cdf(x_v))

  def testInverseGaussianLogCdfValidateArgs(self):
    batch_size = 2
    loc = self.make_tensor([2.] * batch_size)
    concentration = self.make_tensor([2., 3.])
    x = self.make_tensor([-1., 2.])
    inverse_gaussian = tfd.InverseGaussian(loc, concentration,
                                           validate_args=True)

    with self.assertRaisesOpError("x must be positive."):
      self.evaluate(inverse_gaussian.log_cdf(x))

  def testInverseGaussianCdfValidateArgs(self):
    batch_size = 2
    loc = self.make_tensor([2.] * batch_size)
    concentration = self.make_tensor([2., 3.])
    x = self.make_tensor([-1., 2.])
    inverse_gaussian = tfd.InverseGaussian(loc, concentration,
                                           validate_args=True)

    with self.assertRaisesOpError("x must be positive."):
      self.evaluate(inverse_gaussian.cdf(x))

  def testInverseGaussianLogCdfMultidimensional(self):
    batch_size = 6
    loc_v = 1.
    concentration_v = [2., 4., 5.]
    x_v = np.array([[6., 7., 9.2, 5., 6., 7.]]).T
    inverse_gaussian = tfd.InverseGaussian(
        self.make_tensor([[loc_v]] * batch_size),
        self.make_tensor([concentration_v] * batch_size))

    log_cdf = inverse_gaussian.log_cdf(self.make_tensor(x_v))
    if self.use_static_shape:
      self.assertEqual(log_cdf.shape, (6, 3))
    self.assertAllClose(
        self.evaluate(log_cdf),
        _scipy_invgauss(loc_v, np.array(concentration_v)).logcdf(x_v))

    cdf = inverse_gaussian.cdf(self.make_tensor(x_v))
    if self.use_static_shape:
      self.assertEqual(cdf.shape, (6, 3))
    self.assertAllClose(
        self.evaluate(cdf),
        _scipy_invgauss(loc_v, np.array(concentration_v)).cdf(x_v))

  def testInverseGaussianMean(self):
    loc_v = [2., 3., 2.5]
    concentration_v = [1.4, 2., 2.5]
    inverse_gaussian = tfd.InverseGaussian(
        self.make_tensor(loc_v),
        self.make_tensor(concentration_v))
    if self.use_static_shape:
      self.assertEqual(inverse_gaussian.mean().shape, (3,))
    self.assertAllClose(
        self.evaluate(inverse_gaussian.mean()),
        _scipy_invgauss(np.array(loc_v), np.array(concentration_v)).mean())

  def testInverseGaussianMeanBroadCast(self):
    loc_v = 2.
    concentration_v = [1.4, 2., 2.5]
    inverse_gaussian = tfd.InverseGaussian(
        self.make_tensor(loc_v),
        self.make_tensor(concentration_v))
    if self.use_static_shape:
      self.assertEqual(inverse_gaussian.mean().shape, (3,))
    self.assertAllClose(
        self.evaluate(inverse_gaussian.mean()),
        _scipy_invgauss(np.array(loc_v), np.array(concentration_v)).mean())

  def testInverseGaussianVariance(self):
    loc_v = [2., 3., 2.5]
    concentration_v = [1.4, 2., 2.5]
    inverse_gaussian = tfd.InverseGaussian(
        self.make_tensor(loc_v),
        self.make_tensor(concentration_v))

    if self.use_static_shape:
      self.assertEqual(inverse_gaussian.variance().shape, (3,))
    self.assertAllClose(
        self.evaluate(inverse_gaussian.variance()),
        _scipy_invgauss(np.array(loc_v), np.array(concentration_v)).var())

  def testInverseGaussianVarianceBroadcast(self):
    loc_v = 2.
    concentration_v = [1.4, 2., 2.5]
    inverse_gaussian = tfd.InverseGaussian(
        self.make_tensor(loc_v),
        self.make_tensor(concentration_v))

    if self.use_static_shape:
      self.assertEqual(inverse_gaussian.variance().shape, (3,))
    self.assertAllClose(
        self.evaluate(inverse_gaussian.variance()),
        _scipy_invgauss(np.array(loc_v), np.array(concentration_v)).var())

  def testInverseGaussianSampleMean(self):
    loc_v = 3.
    concentration_v = 4.
    n = int(1e6)
    inverse_gaussian = tfd.InverseGaussian(
        self.make_tensor(loc_v),
        self.make_tensor(concentration_v))
    samples = inverse_gaussian.sample(n, seed=12345678)
    sample_values = self.evaluate(samples)

    if self.use_static_shape:
      self.assertEqual(samples.shape, (n,))
    self.assertEqual(sample_values.shape, (n,))
    self.assertAllClose(
        sample_values.mean(),
        _scipy_invgauss(loc_v, concentration_v).mean(),
        rtol=.02,
        atol=0)

  def testInverseGaussianSampleVariance(self):
    loc_v = 3.
    concentration_v = 4.
    n = int(1e6)
    inverse_gaussian = tfd.InverseGaussian(
        self.make_tensor(loc_v),
        self.make_tensor(concentration_v))
    samples = inverse_gaussian.sample(n, seed=12345678)
    sample_values = self.evaluate(samples)

    if self.use_static_shape:
      self.assertEqual(samples.shape, (n,))
    self.assertEqual(sample_values.shape, (n,))
    self.assertAllClose(
        sample_values.var(),
        _scipy_invgauss(loc_v, concentration_v).var(),
        rtol=.02,
        atol=0)

  def testInverseGaussianSampleMultidimensionalMean(self):
    loc_v = 3.
    concentration_v = np.array([np.arange(1, 11)])
    n = int(1e6)
    inverse_gaussian = tfd.InverseGaussian(
        self.make_tensor(loc_v),
        self.make_tensor(concentration_v))
    samples = inverse_gaussian.sample(n, seed=12345678)
    sample_values = self.evaluate(samples)

    if self.use_static_shape:
      self.assertEqual(samples.shape, (n, 1, 10))
    self.assertEqual(sample_values.shape, (n, 1, 10))
    self.assertAllClose(
        sample_values.mean(axis=0),
        _scipy_invgauss(loc_v, concentration_v).mean(),
        rtol=.02,
        atol=0)

  def testInverseGaussianSampleMultidimensionalVariance(self):
    loc_v = 3.
    concentration_v = np.array([np.arange(1, 11)])
    n = int(1e6)
    inverse_gaussian = tfd.InverseGaussian(
        self.make_tensor(loc_v),
        self.make_tensor(concentration_v))
    samples = inverse_gaussian.sample(n, seed=12345678)
    sample_values = self.evaluate(samples)

    if self.use_static_shape:
      self.assertEqual(samples.shape, (n, 1, 10))
    self.assertEqual(sample_values.shape, (n, 1, 10))
    self.assertAllClose(
        sample_values.var(axis=0),
        _scipy_invgauss(loc_v, concentration_v).var(),
        rtol=.02,
        atol=0)


class InverseGaussianTestStaticShapeFloat32(tf.test.TestCase,
                                            _InverseGaussianTest):
  dtype = tf.float32
  use_static_shape = True


class InverseGaussianTestDynamicShapeFloat32(tf.test.TestCase,
                                             _InverseGaussianTest):
  dtype = tf.float32
  use_static_shape = False


class InverseGaussianTestStaticShapeFloat64(tf.test.TestCase,
                                            _InverseGaussianTest):
  dtype = tf.float64
  use_static_shape = True


class InverseGaussianTestDynamicShapeFloat64(tf.test.TestCase,
                                             _InverseGaussianTest):
  dtype = tf.float64
  use_static_shape = False


if __name__ == "__main__":
  tf.test.main()
