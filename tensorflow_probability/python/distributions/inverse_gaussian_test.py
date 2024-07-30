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
import numpy as np
from scipy import misc as sp_misc
from scipy import stats
import tensorflow.compat.v1 as tf1
import tensorflow.compat.v2 as tf
from tensorflow_probability.python.distributions import inverse_gaussian
from tensorflow_probability.python.internal import test_util
from tensorflow_probability.python.math import gradient


def _scipy_invgauss(loc, concentration):
  # Wrapper of scipy's invgauss function, which is used to generate expected
  # output.
  # scipy uses a different parameterization.
  # See https://github.com/scipy/scipy/issues/4654.
  return stats.invgauss(mu=loc/concentration, scale=concentration)


@test_util.test_all_tf_execution_regimes
class _InverseGaussianTest(object):

  def make_tensor(self, x):
    x = tf.cast(x, self.dtype)
    return tf1.placeholder_with_default(
        x, shape=x.shape if self.use_static_shape else None)

  def testInverseGaussianShape(self):
    loc = self.make_tensor([2.] * 5)
    concentration = self.make_tensor([2.] * 5)
    dist = inverse_gaussian.InverseGaussian(
        loc, concentration, validate_args=True)

    self.assertEqual(self.evaluate(dist.batch_shape_tensor()), (5,))
    if self.use_static_shape:
      self.assertEqual(dist.batch_shape, tf.TensorShape([5]))
    self.assertAllEqual(self.evaluate(dist.event_shape_tensor()), [])
    self.assertEqual(dist.event_shape, tf.TensorShape([]))

  def testInverseGaussianShapeBroadcast(self):
    loc = self.make_tensor([[4.], [5.], [6.]])
    concentration = self.make_tensor([[3., 2.]])
    dist = inverse_gaussian.InverseGaussian(
        loc, concentration, validate_args=True)

    self.assertAllEqual(self.evaluate(dist.batch_shape_tensor()), (3, 2))
    if self.use_static_shape:
      self.assertAllEqual(dist.batch_shape, tf.TensorShape([3, 2]))
    self.assertAllEqual(self.evaluate(dist.event_shape_tensor()), [])
    self.assertEqual(dist.event_shape, tf.TensorShape([]))

  def testInvalidLoc(self):
    invalid_locs = [-.01, 0., -2.]
    concentration_v = 1.

    for loc_v in invalid_locs:
      with self.assertRaisesOpError('`loc` must be positive'):
        dist = inverse_gaussian.InverseGaussian(
            self.make_tensor(loc_v),
            self.make_tensor(concentration_v),
            validate_args=True)
        self.evaluate(dist.mean())

  def testInvalidConcentration(self):
    loc_v = 3.
    invalid_concentrations = [-.01, 0., -2.]

    for concentration_v in invalid_concentrations:
      with self.assertRaisesOpError('`concentration` must be positive'):
        dist = inverse_gaussian.InverseGaussian(
            self.make_tensor(loc_v),
            self.make_tensor(concentration_v),
            validate_args=True)
        self.evaluate(dist.mean())

  def testInverseGaussianLogPdf(self):
    batch_size = 6
    loc_v = 2.
    concentration_v = 3.
    x_v = [3., 3.1, 4., 5., 6., 7.]
    dist = inverse_gaussian.InverseGaussian(
        self.make_tensor([loc_v] * batch_size),
        self.make_tensor([concentration_v] * batch_size),
        validate_args=True)

    log_prob = dist.log_prob(self.make_tensor(x_v))
    if self.use_static_shape:
      self.assertEqual(log_prob.shape, (6,))
    self.assertAllClose(
        self.evaluate(log_prob),
        _scipy_invgauss(loc_v, concentration_v).logpdf(x_v))

    pdf = dist.prob(self.make_tensor(x_v))
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
    dist = inverse_gaussian.InverseGaussian(
        loc, concentration, validate_args=True)

    with self.assertRaisesOpError('must be non-negative.'):
      self.evaluate(dist.log_prob(x))

  def testInverseGaussianPdfValidateArgs(self):
    batch_size = 2
    loc = self.make_tensor([2.] * batch_size)
    concentration = self.make_tensor([2., 3.])
    x = self.make_tensor([-1., 2.])
    dist = inverse_gaussian.InverseGaussian(
        loc, concentration, validate_args=True)

    with self.assertRaisesOpError('must be non-negative.'):
      self.evaluate(dist.prob(x))

  def testInverseGaussianLogPdfMultidimensional(self):
    batch_size = 6
    loc_v = 1.
    concentration_v = [2., 4., 5.]
    x_v = np.array([[6., 7., 9.2, 5., 6., 7.]]).T
    dist = inverse_gaussian.InverseGaussian(
        self.make_tensor([[loc_v]] * batch_size),
        self.make_tensor([concentration_v] * batch_size),
        validate_args=True)

    log_prob = dist.log_prob(self.make_tensor(x_v))
    if self.use_static_shape:
      self.assertEqual(log_prob.shape, (6, 3))
    self.assertAllClose(
        self.evaluate(log_prob),
        _scipy_invgauss(loc_v, np.array(concentration_v)).logpdf(x_v))

    prob = dist.prob(self.make_tensor(x_v))
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
    dist = inverse_gaussian.InverseGaussian(
        self.make_tensor([loc_v] * batch_size),
        self.make_tensor([concentration_v] * batch_size),
        validate_args=True)

    log_cdf = dist.log_cdf(self.make_tensor(x_v))
    if self.use_static_shape:
      self.assertEqual(log_cdf.shape, (6,))
    self.assertAllClose(
        self.evaluate(log_cdf),
        _scipy_invgauss(loc_v, concentration_v).logcdf(x_v))

    cdf = dist.cdf(self.make_tensor(x_v))
    if self.use_static_shape:
      self.assertEqual(cdf.shape, (6,))
    self.assertAllClose(
        self.evaluate(cdf),
        _scipy_invgauss(loc_v, concentration_v).cdf(x_v))

  # TODO(b/144948687) Avoid `nan` at boundary. Ideally we'd do this test:
  # def testInverseGaussianPdfAtBoundary(self):
  #   dist = inverse_gaussian.InverseGaussian(
  #       loc=1., concentration=[2., 4., 5.], validate_args=True)
  #   pdf = self.evaluate(dist.prob(0.))
  #   log_pdf = self.evaluate(dist.log_prob(0.))
  #   self.assertAllEqual(pdf, np.zeros_like(pdf))
  #   self.assertTrue(np.isinf(log_pdf).all())

  def testInverseGaussianLogCdfValidateArgs(self):
    batch_size = 2
    loc = self.make_tensor([2.] * batch_size)
    concentration = self.make_tensor([2., 3.])
    x = self.make_tensor([-1., 2.])
    dist = inverse_gaussian.InverseGaussian(
        loc, concentration, validate_args=True)

    with self.assertRaisesOpError('must be non-negative.'):
      self.evaluate(dist.log_cdf(x))

  def testInverseGaussianCdfValidateArgs(self):
    batch_size = 2
    loc = self.make_tensor([2.] * batch_size)
    concentration = self.make_tensor([2., 3.])
    x = self.make_tensor([-1., 2.])
    dist = inverse_gaussian.InverseGaussian(
        loc, concentration, validate_args=True)

    with self.assertRaisesOpError('must be non-negative.'):
      self.evaluate(dist.cdf(x))

  def testInverseGaussianLogCdfMultidimensional(self):
    batch_size = 6
    loc_v = 1.
    concentration_v = [2., 4., 5.]
    x_v = np.array([[6., 7., 9.2, 5., 6., 7.]]).T
    dist = inverse_gaussian.InverseGaussian(
        self.make_tensor([[loc_v]] * batch_size),
        self.make_tensor([concentration_v] * batch_size),
        validate_args=True)

    log_cdf = dist.log_cdf(self.make_tensor(x_v))
    if self.use_static_shape:
      self.assertEqual(log_cdf.shape, (6, 3))
    self.assertAllClose(
        self.evaluate(log_cdf),
        _scipy_invgauss(loc_v, np.array(concentration_v)).logcdf(x_v))

    cdf = dist.cdf(self.make_tensor(x_v))
    if self.use_static_shape:
      self.assertEqual(cdf.shape, (6, 3))
    self.assertAllClose(
        self.evaluate(cdf),
        _scipy_invgauss(loc_v, np.array(concentration_v)).cdf(x_v))

  def testInverseGaussianMean(self):
    loc_v = [2., 3., 2.5]
    concentration_v = [1.4, 2., 2.5]
    dist = inverse_gaussian.InverseGaussian(
        self.make_tensor(loc_v),
        self.make_tensor(concentration_v),
        validate_args=True)
    if self.use_static_shape:
      self.assertEqual(dist.mean().shape, (3,))
    self.assertAllClose(
        self.evaluate(dist.mean()),
        _scipy_invgauss(np.array(loc_v), np.array(concentration_v)).mean())

  def testInverseGaussianMeanBroadCast(self):
    loc_v = 2.
    concentration_v = [1.4, 2., 2.5]
    dist = inverse_gaussian.InverseGaussian(
        self.make_tensor(loc_v),
        self.make_tensor(concentration_v),
        validate_args=True)
    if self.use_static_shape:
      self.assertEqual(dist.mean().shape, (3,))
    self.assertAllClose(
        self.evaluate(dist.mean()),
        _scipy_invgauss(np.array(loc_v), np.array(concentration_v)).mean())

  def testInverseGaussianVariance(self):
    loc_v = [2., 3., 2.5]
    concentration_v = [1.4, 2., 2.5]
    dist = inverse_gaussian.InverseGaussian(
        self.make_tensor(loc_v),
        self.make_tensor(concentration_v),
        validate_args=True)

    if self.use_static_shape:
      self.assertEqual(dist.variance().shape, (3,))
    self.assertAllClose(
        self.evaluate(dist.variance()),
        _scipy_invgauss(np.array(loc_v), np.array(concentration_v)).var())

  def testInverseGaussianVarianceBroadcast(self):
    loc_v = 2.
    concentration_v = [1.4, 2., 2.5]
    dist = inverse_gaussian.InverseGaussian(
        self.make_tensor(loc_v),
        self.make_tensor(concentration_v),
        validate_args=True)

    if self.use_static_shape:
      self.assertEqual(dist.variance().shape, (3,))
    self.assertAllClose(
        self.evaluate(dist.variance()),
        _scipy_invgauss(np.array(loc_v), np.array(concentration_v)).var())

  def testInverseGaussianSampleMean(self):
    loc_v = 3.
    concentration_v = 4.
    n = int(1e6)
    dist = inverse_gaussian.InverseGaussian(
        self.make_tensor(loc_v),
        self.make_tensor(concentration_v),
        validate_args=True)
    samples = dist.sample(n, seed=test_util.test_seed())
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
    dist = inverse_gaussian.InverseGaussian(
        self.make_tensor(loc_v),
        self.make_tensor(concentration_v),
        validate_args=True)
    samples = dist.sample(n, seed=test_util.test_seed())
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
    dist = inverse_gaussian.InverseGaussian(
        self.make_tensor(loc_v),
        self.make_tensor(concentration_v),
        validate_args=True)
    samples = dist.sample(n, seed=test_util.test_seed())
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
    dist = inverse_gaussian.InverseGaussian(
        self.make_tensor(loc_v),
        self.make_tensor(concentration_v),
        validate_args=True)
    samples = dist.sample(n, seed=test_util.test_seed())
    sample_values = self.evaluate(samples)

    if self.use_static_shape:
      self.assertEqual(samples.shape, (n, 1, 10))
    self.assertEqual(sample_values.shape, (n, 1, 10))
    self.assertAllClose(
        sample_values.var(axis=0),
        _scipy_invgauss(loc_v, concentration_v).var(),
        rtol=.02,
        atol=0)

  @test_util.numpy_disable_gradient_test
  def testInverseGaussianFullyReparameterized(self):
    concentration = tf.constant(4.0)
    loc = tf.constant(3.0)
    _, [grad_concentration, grad_loc] = gradient.value_and_gradient(
        lambda a, b: inverse_gaussian.InverseGaussian(a, b, validate_args=True).  # pylint: disable=g-long-lambda
        sample(100, seed=test_util.test_seed()),
        [concentration, loc])
    self.assertIsNotNone(grad_concentration)
    self.assertIsNotNone(grad_loc)

  @test_util.numpy_disable_gradient_test
  def testCompareToExplicitGradient(self):
    """Compare to the explicit reparameterization derivative."""
    self.skipTest('b/331471078')
    concentration_np = np.arange(4)[..., np.newaxis] + 1.
    concentration = tf.constant(concentration_np, self.dtype)
    loc_np = np.arange(3) + 1.
    loc = tf.constant(loc_np, self.dtype)

    def gen_samples(l, c):
      return inverse_gaussian.InverseGaussian(l, c).sample(
          2, seed=test_util.test_seed())

    samples, [loc_grad, concentration_grad] = self.evaluate(
        gradient.value_and_gradient(gen_samples, [loc, concentration]))
    self.assertEqual(samples.shape, (2, 4, 3))
    self.assertEqual(concentration_grad.shape, concentration.shape)
    self.assertEqual(loc_grad.shape, loc.shape)
    # Compute the gradient by computing the derivative of gammaincinv
    # over each entry and summing.
    def expected_grad(s, l, c):
      u = _scipy_invgauss(l, c).cdf(s)
      delta = 1e-4
      return (
          sp_misc.derivative(
              lambda x: _scipy_invgauss(x, c).ppf(u), l, dx=delta * l),
          sp_misc.derivative(
              lambda x: _scipy_invgauss(l, x).ppf(u), c, dx=delta * c))
    expected_loc_grad, expected_concentration_grad = expected_grad(
        samples, loc_np, concentration_np)

    self.assertAllClose(
        concentration_grad,
        np.sum(expected_concentration_grad, axis=(0, 2))[..., np.newaxis],
        rtol=1e-3)

    self.assertAllClose(
        loc_grad,
        np.sum(expected_loc_grad, axis=(0, 1)), rtol=1e-3)

  def testModifiedVariableAssertion(self):
    concentration = tf.Variable(0.9, dtype=self.dtype)
    loc = tf.Variable(1.2, dtype=self.dtype)
    self.evaluate([concentration.initializer, loc.initializer])
    dist = inverse_gaussian.InverseGaussian(
        loc=loc, concentration=concentration, validate_args=True)
    with self.assertRaisesOpError('`concentration` must be positive'):
      with tf.control_dependencies([concentration.assign(-2.)]):
        self.evaluate(dist.mean())
    with self.assertRaisesOpError('`loc` must be positive'):
      with tf.control_dependencies([loc.assign(-2.), concentration.assign(2.)]):
        self.evaluate(dist.mean())

  def testSupportBijectorOutsideRange(self):
    dist = inverse_gaussian.InverseGaussian(
        loc=self.make_tensor([7., 2., 5.]),
        concentration=self.make_tensor(2.),
        validate_args=True)
    eps = 1e-6
    x = np.array([[-7.2, -eps, -1.3], [-5., -12., -eps]])
    bijector_inverse_x = dist.experimental_default_event_space_bijector(
        ).inverse(x)
    self.assertAllNan(self.evaluate(bijector_inverse_x))


class InverseGaussianTestStaticShapeFloat32(test_util.TestCase,
                                            _InverseGaussianTest):
  dtype = tf.float32
  use_static_shape = True


class InverseGaussianTestDynamicShapeFloat32(test_util.TestCase,
                                             _InverseGaussianTest):
  dtype = tf.float32
  use_static_shape = False


class InverseGaussianTestStaticShapeFloat64(test_util.TestCase,
                                            _InverseGaussianTest):
  dtype = tf.float64
  use_static_shape = True


class InverseGaussianTestDynamicShapeFloat64(test_util.TestCase,
                                             _InverseGaussianTest):
  dtype = tf.float64
  use_static_shape = False


if __name__ == '__main__':
  test_util.main()
