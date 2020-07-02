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
"""Tests for Bijector."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports

from absl.testing import parameterized
import numpy as np
import tensorflow.compat.v2 as tf
from tensorflow_probability.python import bijectors as tfb
from tensorflow_probability.python import math as tfp_math
from tensorflow_probability.python.bijectors import bijector_test_util
from tensorflow_probability.python.internal import test_util


rng = np.random.RandomState(42)


@test_util.test_all_tf_execution_regimes
class SoftplusBijectorTest(test_util.TestCase):
  """Tests the correctness of the Y = g(X) = Log[1 + exp(X)] transformation."""

  def _softplus(self, x):
    return np.log(1 + np.exp(x))

  def _softplus_inverse(self, y):
    return np.log(np.exp(y) - 1)

  def _softplus_ildj_before_reduction(self, y):
    """Inverse log det jacobian, before being reduced."""
    return -np.log(1 - np.exp(-y))

  def testHingeSoftnessZeroRaises(self):
    with self.assertRaisesOpError(
        'Argument `hinge_softness` must be non-zero.'):
      self.evaluate(
          tfb.Softplus(hinge_softness=0., validate_args=True).forward(1.))

  def testBijectorForwardInverseEventDimsZero(self):
    bijector = tfb.Softplus()
    self.assertStartsWith(bijector.name, 'softplus')
    x = 2 * rng.randn(2, 10)
    y = self._softplus(x)

    self.assertAllClose(y, self.evaluate(bijector.forward(x)))
    self.assertAllClose(x, self.evaluate(bijector.inverse(y)))

  def testBijectorForwardInverseWithHingeSoftnessEventDimsZero(self):
    bijector = tfb.Softplus(hinge_softness=np.float64(1.5))
    x = 2 * rng.randn(2, 10)
    y = 1.5 * self._softplus(x / 1.5)

    self.assertAllClose(y, self.evaluate(bijector.forward(x)))
    self.assertAllClose(x, self.evaluate(bijector.inverse(y)))

  def testBijectorLogDetJacobianEventDimsZero(self):
    bijector = tfb.Softplus()
    y = 2 * rng.rand(2, 10)
    # No reduction needed if event_dims = 0.
    ildj = self._softplus_ildj_before_reduction(y)

    self.assertAllClose(
        ildj,
        self.evaluate(bijector.inverse_log_det_jacobian(
            y, event_ndims=0)))

  def testBijectorForwardInverseEventDimsOne(self):
    bijector = tfb.Softplus()
    self.assertStartsWith(bijector.name, 'softplus')
    x = 2 * rng.randn(2, 10)
    y = self._softplus(x)

    self.assertAllClose(y, self.evaluate(bijector.forward(x)))
    self.assertAllClose(x, self.evaluate(bijector.inverse(y)))

  def testBijectorLogDetJacobianEventDimsOne(self):
    bijector = tfb.Softplus()
    y = 2 * rng.rand(2, 10)
    ildj_before = self._softplus_ildj_before_reduction(y)
    ildj = np.sum(ildj_before, axis=1)

    self.assertAllClose(
        ildj,
        self.evaluate(
            bijector.inverse_log_det_jacobian(
                y, event_ndims=1)))

  def testScalarCongruency(self):
    bijector = tfb.Softplus()
    bijector_test_util.assert_scalar_congruency(
        bijector, lower_x=-2., upper_x=2., eval_func=self.evaluate, rtol=.02)

  def testScalarCongruencyWithPositiveHingeSoftness(self):
    bijector = tfb.Softplus(hinge_softness=1.3)
    bijector_test_util.assert_scalar_congruency(
        bijector, lower_x=-2., upper_x=2., eval_func=self.evaluate, rtol=.02)

  def testScalarCongruencyWithNegativeHingeSoftness(self):
    bijector = tfb.Softplus(hinge_softness=-1.3)
    bijector_test_util.assert_scalar_congruency(
        bijector, lower_x=-2., upper_x=2., eval_func=self.evaluate, rtol=.02)

  def testBijectiveAndFinite32bit(self):
    bijector = tfb.Softplus()
    x = np.linspace(-20., 20., 100).astype(np.float32)
    y = np.logspace(-10, 10, 100).astype(np.float32)
    bijector_test_util.assert_bijective_and_finite(
        bijector, x, y, eval_func=self.evaluate, event_ndims=0, rtol=1e-2,
        atol=1e-2)

  def testBijectiveAndFiniteWithPositiveHingeSoftness32Bit(self):
    bijector = tfb.Softplus(hinge_softness=1.23)
    x = np.linspace(-20., 20., 100).astype(np.float32)
    y = np.logspace(-10, 10, 100).astype(np.float32)
    bijector_test_util.assert_bijective_and_finite(
        bijector, x, y, eval_func=self.evaluate, event_ndims=0, rtol=1e-2,
        atol=1e-2)

  def testBijectiveAndFiniteWithNegativeHingeSoftness32Bit(self):
    bijector = tfb.Softplus(hinge_softness=-0.7)
    x = np.linspace(-20., 20., 100).astype(np.float32)
    y = -np.logspace(-10, 10, 100).astype(np.float32)
    bijector_test_util.assert_bijective_and_finite(
        bijector, x, y, eval_func=self.evaluate, event_ndims=0, rtol=1e-2,
        atol=1e-2)

  def testVariableHingeSoftness(self):
    hinge_softness = tf.Variable(1.)
    b = tfb.Softplus(hinge_softness=hinge_softness, validate_args=True)
    self.evaluate(hinge_softness.initializer)
    self.assertIs(hinge_softness, b.hinge_softness)
    self.assertEqual((), self.evaluate(b.forward(0.5)).shape)
    with self.assertRaisesOpError(
        'Argument `hinge_softness` must be non-zero.'):
      with tf.control_dependencies([hinge_softness.assign(0.)]):
        self.evaluate(b.forward(0.5))

  @parameterized.named_parameters(
      ('32bitGraph', np.float32, False),
      ('64bitGraph', np.float64, False),
      ('32bitXLA', np.float32, True),
      ('64bitXLA', np.float64, True),
  )
  @test_util.numpy_disable_gradient_test
  def testLeftTailGrad(self, dtype, do_compile):
    x = np.linspace(-50., -8., 1000).astype(dtype)

    @tf.function(autograph=False, experimental_compile=do_compile)
    def fn(x):
      return tf.math.log(tfb.Softplus().forward(x))

    _, grad = tfp_math.value_and_gradient(fn, x)

    true_grad = 1 / (1 + np.exp(-x)) / np.log1p(np.exp(x))
    self.assertAllClose(true_grad, self.evaluate(grad), atol=1e-3)


if __name__ == '__main__':
  tf.test.main()
