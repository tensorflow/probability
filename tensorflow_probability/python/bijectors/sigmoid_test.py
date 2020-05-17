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
"""Sigmoid Tests."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports

from absl.testing import parameterized
import numpy as np
from scipy import special
import tensorflow.compat.v2 as tf
from tensorflow_probability.python import bijectors as tfb
from tensorflow_probability.python.bijectors import bijector_test_util
from tensorflow_probability.python.internal import test_util


@test_util.test_all_tf_execution_regimes
class SigmoidBijectorTest(test_util.TestCase):
  """Tests correctness of the Y = g(X) = (1 + exp(-X))^-1 transformation."""

  def testBijector(self):
    self.assertStartsWith(tfb.Sigmoid().name, 'sigmoid')
    x = np.linspace(-10., 10., 100).reshape([2, 5, 10]).astype(np.float32)
    y = special.expit(x)
    ildj = -np.log(y) - np.log1p(-y)
    bijector = tfb.Sigmoid()
    self.assertAllClose(
        y, self.evaluate(bijector.forward(x)), atol=0., rtol=1e-2)
    self.assertAllClose(
        x, self.evaluate(bijector.inverse(y)), atol=0., rtol=1e-4)
    self.assertAllClose(
        ildj,
        self.evaluate(bijector.inverse_log_det_jacobian(
            y, event_ndims=0)), atol=0., rtol=1e-6)
    self.assertAllClose(
        -ildj,
        self.evaluate(bijector.forward_log_det_jacobian(
            x, event_ndims=0)), atol=0., rtol=1e-4)

  def testScalarCongruency(self):
    bijector_test_util.assert_scalar_congruency(
        tfb.Sigmoid(), lower_x=-7., upper_x=7., eval_func=self.evaluate,
        rtol=.1)

  def testBijectiveAndFinite(self):
    x = np.linspace(-100., 100., 100).astype(np.float32)
    eps = 1e-3
    y = np.linspace(eps, 1. - eps, 100).astype(np.float32)
    bijector_test_util.assert_bijective_and_finite(
        tfb.Sigmoid(), x, y, eval_func=self.evaluate, event_ndims=0, atol=0.,
        rtol=1e-4)


@test_util.test_all_tf_execution_regimes
class ShiftedScaledSigmoidBijectorTest(test_util.TestCase):
  """Tests correctness of Sigmoid with `low` and `high` parameters set."""

  def testBijector(self):
    low = np.array([-3., 0., 5.]).astype(np.float32)
    high = 12.

    bijector = tfb.Sigmoid(low=low, high=high, validate_args=True)

    equivalent_bijector = tfb.Chain([
        tfb.Shift(shift=low), tfb.Scale(scale=high-low), tfb.Sigmoid()])

    x = [[[1.], [2.], [-5.], [-0.3]]]
    y = self.evaluate(equivalent_bijector.forward(x))
    self.assertAllClose(y, self.evaluate(bijector.forward(x)))
    self.assertAllClose(x, self.evaluate(bijector.inverse(y)[..., -1:]))
    self.assertAllClose(
        self.evaluate(equivalent_bijector.inverse_log_det_jacobian(
            y, event_ndims=1)),
        self.evaluate(bijector.inverse_log_det_jacobian(
            y, event_ndims=1)),
        rtol=1e-5)
    self.assertAllClose(
        self.evaluate(equivalent_bijector.forward_log_det_jacobian(
            x, event_ndims=1)),
        self.evaluate(bijector.forward_log_det_jacobian(
            x, event_ndims=1)))

  def testNumericallySuperiorToEquivalentChain(self):
    x = np.array([-5., 3., 17., 23.]).astype(np.float32)
    low = -0.08587775
    high = 0.12498104

    bijector = tfb.Sigmoid(low=low, high=high, validate_args=True)

    equivalent_bijector = tfb.Chain([
        tfb.Shift(shift=low), tfb.Scale(scale=high-low), tfb.Sigmoid()])

    self.assertAllLessEqual(self.evaluate(bijector.forward(x)), high)

    # The mathematically equivalent `Chain` bijector can return values greater
    # than the intended upper bound of `high`.
    self.assertTrue(
        (self.evaluate(equivalent_bijector.forward(x)) > high).any())

  def testScalarCongruency(self):
    low = -2.
    high = 5.
    bijector = tfb.Sigmoid(low=low, high=high, validate_args=True)
    bijector_test_util.assert_scalar_congruency(
        bijector, lower_x=-5., upper_x=3.5, eval_func=self.evaluate,
        rtol=0.05)

  def testBijectiveAndFinite(self):
    low = -5.
    high = 8.
    bijector = tfb.Sigmoid(low=low, high=high, validate_args=True)
    x = np.linspace(-10, 10, num=100).astype(np.float32)
    eps = 1e-6
    y = np.linspace(low + eps, high - eps, num=100).astype(np.float32)
    bijector_test_util.assert_bijective_and_finite(
        bijector, x, y, eval_func=self.evaluate, event_ndims=0)

  def testAssertHighGtLow(self):
    low = np.array([1., 1., 1.], dtype=np.float32)
    high = np.array([1., 2., 3.], dtype=np.float32)

    with self.assertRaisesOpError('not defined when `low` >= `high`'):
      bijector = tfb.Sigmoid(low=low, high=high, validate_args=True)
      self.evaluate(bijector.forward(3.))

  def testEdgeCaseRequiringClipping(self):
    np.set_printoptions(floatmode='unique', precision=None)
    lo = np.float32(0.010489981)
    hi = test_util.floats_near(
        0.010499111, 100, dtype=np.float32)[:, np.newaxis]
    self.assertAllEqual([100, 1], hi.shape)
    xs = test_util.floats_near(9.814646, 100, dtype=np.float32)
    bijector = tfb.Sigmoid(low=lo, high=hi, validate_args=True)
    answers = bijector.forward(xs)
    self.assertAllEqual([100, 100], answers.shape)
    for ans1, hi1 in zip(self.evaluate(answers), hi):
      self.assertAllLessEqual(ans1, hi1)

  @parameterized.named_parameters(
      ('32bitGraph', np.float32, False),
      ('64bitGraph', np.float64, False),
      ('32bitXLA', np.float32, True),
      ('64bitXLA', np.float64, True),
  )
  @test_util.numpy_disable_gradient_test
  def testLeftTail(self, dtype, do_compile):
    x = np.linspace(-50., -8., 1000).astype(dtype)

    @tf.function(autograph=False, experimental_compile=do_compile)
    def fn(x):
      return tf.math.log(tfb.Sigmoid().forward(x))

    vals = fn(x)
    true_vals = -np.log1p(np.exp(-x))
    self.assertAllClose(true_vals, self.evaluate(vals), atol=1e-3)


if __name__ == '__main__':
  tf.test.main()
