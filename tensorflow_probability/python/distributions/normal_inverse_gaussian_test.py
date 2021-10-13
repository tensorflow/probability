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
import numpy as np
from scipy import stats
import tensorflow.compat.v2 as tf
from tensorflow_probability.python import distributions as tfd
from tensorflow_probability.python.internal import test_util


@test_util.test_all_tf_execution_regimes
class _NormalInverseGaussianTest(object):

  def testNormalInverseGaussianShape(self):
    loc = tf.ones([2, 1, 1, 1], dtype=self.dtype)
    scale = tf.ones([1, 3, 1, 1], dtype=self.dtype)
    tailweight = 2 * tf.ones([1, 1, 5, 1], dtype=self.dtype)
    skewness = tf.ones([1, 1, 1, 7], dtype=self.dtype)
    normal_inverse_gaussian = tfd.NormalInverseGaussian(
        loc, scale, tailweight, skewness, validate_args=True)

    self.assertAllEqual(self.evaluate(
        normal_inverse_gaussian.batch_shape_tensor()), (2, 3, 5, 7))
    self.assertEqual(
        normal_inverse_gaussian.batch_shape, tf.TensorShape([2, 3, 5, 7]))
    self.assertAllEqual(
        self.evaluate(normal_inverse_gaussian.event_shape_tensor()), [])
    self.assertEqual(normal_inverse_gaussian.event_shape, tf.TensorShape([]))

  def testInvalidScale(self):
    with self.assertRaisesOpError('`scale` must be positive'):
      normal_inverse_gaussian = tfd.NormalInverseGaussian(
          loc=0.,
          scale=-1.,
          tailweight=2.,
          skewness=1.,
          validate_args=True)
      self.evaluate(normal_inverse_gaussian.mean())

  def testInvalidSkewnessTailweight(self):
    with self.assertRaisesOpError('`tailweight > |skewness|`'):
      normal_inverse_gaussian = tfd.NormalInverseGaussian(
          loc=0.,
          scale=1.,
          tailweight=2.,
          skewness=3.,
          validate_args=True)
      self.evaluate(normal_inverse_gaussian.mean())

  def testNormalInverseGaussianScipyLogPdf(self):
    loc_v = self.dtype(0.)
    scale_v = self.dtype(1.)
    tailweight_v = self.dtype(1.)
    skewness_v = self.dtype(0.2)
    x_v = np.array([3., 3.1, 4., 5., 6., 7.]).astype(self.dtype)
    normal_inverse_gaussian = tfd.NormalInverseGaussian(
        loc_v, scale_v, tailweight_v, skewness_v, validate_args=True)

    log_prob = normal_inverse_gaussian.log_prob(x_v)
    self.assertAllClose(
        self.evaluate(log_prob),
        stats.norminvgauss(
            a=tailweight_v, b=skewness_v,
            loc=loc_v, scale=scale_v).logpdf(x_v))

    pdf = normal_inverse_gaussian.prob(x_v)
    self.assertAllClose(
        self.evaluate(pdf),
        stats.norminvgauss(
            a=tailweight_v, b=skewness_v,
            loc=loc_v, scale=scale_v).pdf(x_v))

  def testNormalInverseGaussianLogPdfScipyMultidimensional(self):
    loc_v = np.linspace(-10., 10., 5).astype(self.dtype).reshape((5, 1, 1, 1))
    scale_v = self.dtype(1.)
    tailweight_v = np.array([2., 0.5], dtype=self.dtype).reshape((2, 1))
    skewness_v = 0.5 * tailweight_v
    x_v = np.linspace(-9., 9., 20).astype(self.dtype)
    normal_inverse_gaussian = tfd.NormalInverseGaussian(
        loc_v, scale_v, tailweight_v, skewness_v, validate_args=True)
    log_prob = normal_inverse_gaussian.log_prob(x_v)
    self.assertAllClose(
        self.evaluate(log_prob),
        stats.norminvgauss(
            a=tailweight_v, b=skewness_v, loc=loc_v, scale=scale_v).logpdf(x_v))

    prob = normal_inverse_gaussian.prob(x_v)
    self.assertAllClose(
        self.evaluate(prob),
        stats.norminvgauss(
            a=tailweight_v, b=skewness_v, loc=loc_v, scale=scale_v).pdf(x_v))

  def testNormalInverseGaussianMeanScale1(self):
    loc_v = np.linspace(-10., 10., 5).astype(self.dtype).reshape((5, 1, 1, 1))
    scale_v = self.dtype(1.)
    tailweight_v = np.array([2., 0.5], dtype=self.dtype).reshape((2, 1))
    skewness_v = 0.5 * tailweight_v
    normal_inverse_gaussian = tfd.NormalInverseGaussian(
        loc_v, scale_v, tailweight_v, skewness_v, validate_args=True)
    self.assertAllClose(
        self.evaluate(normal_inverse_gaussian.mean()),
        stats.norminvgauss(
            a=tailweight_v, b=skewness_v, loc=loc_v, scale=scale_v).mean())

  def testNormalInverseGaussianVarianceScale1(self):
    loc_v = np.linspace(-10., 10., 5).astype(self.dtype).reshape((5, 1, 1, 1))
    scale_v = self.dtype(1.)
    tailweight_v = np.array([2., 0.5], dtype=self.dtype).reshape((2, 1))
    skewness_v = 0.5 * tailweight_v
    normal_inverse_gaussian = tfd.NormalInverseGaussian(
        loc_v, scale_v, tailweight_v, skewness_v, validate_args=True)
    self.assertAllClose(
        self.evaluate(normal_inverse_gaussian.variance()),
        stats.norminvgauss(
            a=tailweight_v, b=skewness_v, loc=loc_v, scale=scale_v).var())

  def testNormalInverseGaussianSampleMean(self):
    loc_v = self.dtype(-2.)
    scale_v = self.dtype(np.pi)
    tailweight_v = self.dtype(1.87)
    skewness_v = self.dtype(-1.)
    n = int(3e5)
    normal_inverse_gaussian = tfd.NormalInverseGaussian(
        loc_v, scale_v, tailweight_v, skewness_v, validate_args=True)
    samples = normal_inverse_gaussian.sample(n, seed=test_util.test_seed())
    sample_values = self.evaluate(samples)

    self.assertEqual(sample_values.shape, (n,))
    self.assertAllClose(
        sample_values.mean(),
        self.evaluate(normal_inverse_gaussian.mean()),
        rtol=.02,
        atol=0)

  def testNormalInverseGaussianSampleVariance(self):
    loc_v = self.dtype(-2.)
    scale_v = self.dtype(np.pi)
    tailweight_v = self.dtype(1.87)
    skewness_v = self.dtype(-1.)
    n = int(3e5)
    normal_inverse_gaussian = tfd.NormalInverseGaussian(
        loc_v, scale_v, tailweight_v, skewness_v, validate_args=True)
    samples = normal_inverse_gaussian.sample(n, seed=test_util.test_seed())
    sample_values = self.evaluate(samples)

    self.assertEqual(sample_values.shape, (n,))
    self.assertAllClose(
        sample_values.var(),
        self.evaluate(normal_inverse_gaussian.variance()),
        rtol=.02,
        atol=0)

  def testNormalInverseGaussianSampleMultidimensionalMean(self):
    loc_v = np.linspace(-10., 10., 5).astype(self.dtype).reshape((5, 1, 1))
    scale_v = np.linspace(1., 3., 3).astype(self.dtype).reshape((3, 1))
    tailweight_v = np.array([2., 0.5], dtype=self.dtype)
    skewness_v = 0.5 * tailweight_v
    n = int(3e5)
    normal_inverse_gaussian = tfd.NormalInverseGaussian(
        loc_v, scale_v, tailweight_v, skewness_v, validate_args=True)
    samples = normal_inverse_gaussian.sample(n, seed=test_util.test_seed())
    sample_values = self.evaluate(samples)

    self.assertEqual(sample_values.shape, (n, 5, 3, 2))
    self.assertAllClose(
        sample_values.mean(axis=0),
        self.evaluate(normal_inverse_gaussian.mean()),
        rtol=.02,
        atol=0)

  def testNormalInverseGaussianSampleMultidimensionalVariance(self):
    loc_v = np.linspace(-10., 10., 5).astype(self.dtype).reshape((5, 1, 1))
    scale_v = np.linspace(1., 3., 3).astype(self.dtype).reshape((3, 1))
    tailweight_v = np.array([2., 0.5], dtype=self.dtype)
    skewness_v = 0.5 * tailweight_v
    n = int(3e5)
    normal_inverse_gaussian = tfd.NormalInverseGaussian(
        loc_v, scale_v, tailweight_v, skewness_v, validate_args=True)
    samples = normal_inverse_gaussian.sample(n, seed=test_util.test_seed())
    sample_values = self.evaluate(samples)

    self.assertEqual(sample_values.shape, (n, 5, 3, 2))
    self.assertAllClose(
        sample_values.var(axis=0),
        self.evaluate(normal_inverse_gaussian.variance()),
        rtol=.05,
        atol=0)

  def testModifiedVariableAssertion(self):
    loc = 0.3
    scale = 2.
    tailweight = tf.Variable(1.9, dtype=self.dtype)
    skewness = tf.Variable(1.2, dtype=self.dtype)
    self.evaluate([tailweight.initializer, skewness.initializer])
    normal_inverse_gaussian = tfd.NormalInverseGaussian(
        loc=loc, scale=scale, tailweight=tailweight, skewness=skewness,
        validate_args=True)
    with self.assertRaisesOpError('`tailweight > |skewness|`'):
      with tf.control_dependencies([tailweight.assign(1.)]):
        self.evaluate(normal_inverse_gaussian.mean())

    tailweight = tf.Variable(1.9, dtype=self.dtype)
    skewness = tf.Variable(1.2, dtype=self.dtype)
    self.evaluate([tailweight.initializer, skewness.initializer])
    normal_inverse_gaussian = tfd.NormalInverseGaussian(
        loc=loc, scale=scale, tailweight=tailweight, skewness=skewness,
        validate_args=True)
    with self.assertRaisesOpError('`tailweight > |skewness|`'):
      with tf.control_dependencies([skewness.assign(-2.)]):
        self.evaluate(normal_inverse_gaussian.mean())


class NormalInverseGaussianTestFloat32(
    test_util.TestCase, _NormalInverseGaussianTest):
  dtype = np.float32


class NormalInverseGaussianTestFloat64(
    test_util.TestCase, _NormalInverseGaussianTest):
  dtype = np.float64


if __name__ == '__main__':
  test_util.main()
