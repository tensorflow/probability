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
"""Tests for SinhArcsinh Bijector."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports

from absl.testing import parameterized
import numpy as np
import tensorflow.compat.v2 as tf
from tensorflow_probability.python import bijectors as tfb
from tensorflow_probability.python.bijectors import bijector_test_util
from tensorflow_probability.python.internal import test_util


@test_util.test_all_tf_execution_regimes
class SinhArcsinhTest(test_util.TestCase):
  """Tests correctness of the power transformation."""

  def testBijectorVersusNumpyRewriteOfBasicFunctions(self):
    skewness = 0.2
    tailweight = 2.0
    multiplier = 2.0 / np.sinh(np.arcsinh(2.0) * tailweight)
    bijector = tfb.SinhArcsinh(
        skewness=skewness, tailweight=tailweight, validate_args=True)
    self.assertStartsWith(bijector.name, 'sinh_arcsinh')
    x = np.array([[[-2.01], [2.], [1e-4]]]).astype(np.float32)
    y = np.sinh((np.arcsinh(x) + skewness) * tailweight) * multiplier
    self.assertAllClose(y, self.evaluate(bijector.forward(x)))
    self.assertAllClose(x, self.evaluate(bijector.inverse(y)))
    self.assertAllClose(
        np.sum(
            np.log(np.cosh(
                np.arcsinh(y / multiplier) / tailweight - skewness)) -
            np.log(tailweight) - np.log(np.sqrt((y / multiplier)**2 + 1))
            - np.log(multiplier),
            axis=-1),
        self.evaluate(bijector.inverse_log_det_jacobian(y, event_ndims=1)),
        rtol=2e-6)
    self.assertAllClose(
        self.evaluate(-bijector.inverse_log_det_jacobian(y, event_ndims=1)),
        self.evaluate(bijector.forward_log_det_jacobian(x, event_ndims=1)),
        rtol=1e-4,
        atol=0.)

  def testSkew(self):
    # Will broadcast together to shape [3, 2].
    x = [-1., 1.]
    skewness = [[-1.], [0.], [1.]]
    bijector = tfb.SinhArcsinh(skewness=skewness, validate_args=True)
    y = self.evaluate(bijector.forward(x))

    # For skew < 0, |forward(-1)| > |forward(1)|
    self.assertGreater(np.abs(y[0, 0]), np.abs(y[0, 1]))

    # For skew = 0, |forward(-1)| = |forward(1)|
    self.assertAllClose(np.abs(y[1, 0]), np.abs(y[1, 1]))

    # For skew > 0, |forward(-1)| < |forward(1)|
    self.assertLess(np.abs(y[2, 0]), np.abs(y[2, 1]))

  def testKurtosis(self):
    x = np.logspace(-2, 2, 1000).astype(np.float32)
    tailweight = [[0.5], [1.0], [2.0]]
    bijector = tfb.SinhArcsinh(tailweight=tailweight, validate_args=True)
    y = self.evaluate(bijector.forward(x))
    mean = np.mean(x, axis=-1)
    stddev = np.std(x, axis=-1, ddof=0)
    kurtosis = np.mean((y - mean) ** 4, axis=-1) / (stddev ** 4)
    self.assertAllClose(kurtosis, np.sort(kurtosis))

  def testScalarCongruencySkewness1Tailweight0p5(self):
    bijector = tfb.SinhArcsinh(
        skewness=1.0, tailweight=0.5, validate_args=True)
    bijector_test_util.assert_scalar_congruency(
        bijector, lower_x=-2., upper_x=2.0, eval_func=self.evaluate, rtol=0.05)

  def testScalarCongruencySkewnessNeg1Tailweight1p5(self):
    bijector = tfb.SinhArcsinh(
        skewness=-1.0, tailweight=1.5, validate_args=True)
    bijector_test_util.assert_scalar_congruency(
        bijector, lower_x=-2., upper_x=2.0, eval_func=self.evaluate, rtol=0.05)

  def testBijectiveAndFiniteSkewnessNeg1Tailweight0p5(self):
    bijector = tfb.SinhArcsinh(
        skewness=-1., tailweight=0.5, validate_args=True)
    x = np.concatenate((-np.logspace(-2, 10, 1000), [0], np.logspace(
        -2, 10, 1000))).astype(np.float32)
    bijector_test_util.assert_bijective_and_finite(
        bijector, x, x, eval_func=self.evaluate, event_ndims=0, rtol=1e-3)

  def testBijectiveAndFiniteSkewness1Tailweight3(self):
    bijector = tfb.SinhArcsinh(skewness=1., tailweight=3., validate_args=True)
    x = np.concatenate((-np.logspace(-2, 5, 1000), [0], np.logspace(
        -2, 5, 1000))).astype(np.float32)
    bijector_test_util.assert_bijective_and_finite(
        bijector, x, x, eval_func=self.evaluate, event_ndims=0, rtol=1e-3)

  @parameterized.parameters(np.float32, np.float64)
  def testBijectorEndpoints(self, dtype):
    bijector = tfb.SinhArcsinh(
        skewness=dtype(0.), tailweight=dtype(1.), validate_args=True)
    # Use bounds that are very large to check that the transformation remains
    # bijective. We stray away from the largest/smallest value to avoid issues
    # at the boundary since XLA sinh will return `inf` for the largest value.
    bounds = np.array(
        [np.nextafter(np.finfo(dtype).min, 0.) / 10.,
         np.nextafter(np.finfo(dtype).max, 0.) / 10.], dtype=dtype)
    # Note that the above bijector is the identity bijector. Hence, the
    # log_det_jacobian will be 0. Because of this we use atol.
    bijector_test_util.assert_bijective_and_finite(
        bijector, bounds, bounds, eval_func=self.evaluate, event_ndims=0,
        atol=2e-6)

  @parameterized.parameters(np.float32, np.float64)
  def testBijectorOverRange(self, dtype):
    skewness = np.array([1.2, 5.], dtype=dtype)
    tailweight = np.array([2., 10.], dtype=dtype)
    # The inverse will be defined up to where sinh is valid, which is
    # for values close to arcsinh(np.finfo(dtype).max).
    # We stray away from the largest value to avoid issues
    # at the boundary since XLA sinh will return `inf` for the largest value.
    max_val = np.nextafter(np.finfo(dtype).max, 0.) / 10.

    log_boundary = np.log(
        np.sinh(np.arcsinh(max_val) / tailweight - skewness))
    x = np.array([
        np.logspace(-2, log_boundary[0], base=np.e, num=1000),
        np.logspace(-2, log_boundary[1], base=np.e, num=1000)
    ], dtype=dtype)
    # Ensure broadcasting works.
    x = np.swapaxes(x, 0, 1)
    multiplier = 2. / np.sinh(np.arcsinh(2.) * tailweight)
    y = np.sinh((np.arcsinh(x) + skewness) * tailweight) * multiplier
    bijector = tfb.SinhArcsinh(
        skewness=skewness, tailweight=tailweight, validate_args=True)

    self.assertAllClose(
        y, self.evaluate(bijector.forward(x)), rtol=1e-4, atol=0.)
    self.assertAllClose(
        x, self.evaluate(bijector.inverse(y)), rtol=1e-4, atol=0.)

    # On IBM PPC systems, longdouble (np.float128) is same as double except
    # that it can have more precision. Type double being of 8 bytes, can't
    # hold square of max of float64 (which is also 8 bytes).
    # Below test fails due to overflow error giving inf. This check avoids
    # that error by skipping square calculation and corresponding assert.

    if (np.amax(y) <= np.sqrt(np.finfo(np.float128).max) and
        np.fabs(np.amin(y)) <= np.sqrt(np.fabs(np.finfo(np.float128).min))):

      # Do the numpy calculation in float128 to avoid inf/nan.
      y_float128 = np.float128(y)
      self.assertAllClose(
          np.log(np.cosh(
              np.arcsinh(y_float128 / multiplier)
              / tailweight - skewness) / np.sqrt(
                  (y_float128 / multiplier)**2 + 1))
          - np.log(tailweight) - np.log(multiplier),
          self.evaluate(
              bijector.inverse_log_det_jacobian(y, event_ndims=0)),
          rtol=1e-4,
          atol=0.)
    self.assertAllClose(
        self.evaluate(-bijector.inverse_log_det_jacobian(y, event_ndims=0)),
        self.evaluate(bijector.forward_log_det_jacobian(x, event_ndims=0)),
        rtol=1e-4,
        atol=0.)

  def testZeroTailweightRaises(self):
    with self.assertRaisesOpError('Argument `tailweight` must be positive'):
      self.evaluate(
          tfb.SinhArcsinh(tailweight=0., validate_args=True).forward(1.0))

  def testDefaultDtypeIsFloat32(self):
    bijector = tfb.SinhArcsinh()
    self.assertEqual(bijector.tailweight.dtype, np.float32)
    self.assertEqual(bijector.skewness.dtype, np.float32)

  def testVariableTailweight(self):
    x = tf.Variable(1.)
    b = tfb.SinhArcsinh(tailweight=x, validate_args=True)
    self.evaluate(x.initializer)
    self.assertIs(x, b.tailweight)
    self.assertEqual((), self.evaluate(b.forward(0.5)).shape)
    with self.assertRaisesOpError('Argument `tailweight` must be positive.'):  # pylint:disable=g-error-prone-assert-raises
      with tf.control_dependencies([x.assign(-1.)]):
        self.assertEqual((), self.evaluate(b.forward(0.5)).shape)


if __name__ == '__main__':
  tf.test.main()
