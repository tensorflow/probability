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
import numpy as np
import tensorflow as tf
from tensorflow_probability.python import bijectors as tfb

from tensorflow_probability.python.bijectors import bijector_test_util
from tensorflow.python.framework import test_util  # pylint: disable=g-direct-tensorflow-import,g-import-not-at-top

rng = np.random.RandomState(42)


@test_util.run_all_in_graph_and_eager_modes
class SoftplusBijectorTest(tf.test.TestCase):
  """Tests the correctness of the Y = g(X) = Log[1 + exp(X)] transformation."""

  def _softplus(self, x):
    return np.log(1 + np.exp(x))

  def _softplus_inverse(self, y):
    return np.log(np.exp(y) - 1)

  def _softplus_ildj_before_reduction(self, y):
    """Inverse log det jacobian, before being reduced."""
    return -np.log(1 - np.exp(-y))

  def testHingeSoftnessZeroRaises(self):
    with self.assertRaisesOpError("must be non-zero"):
      bijector = tfb.Softplus(hinge_softness=0., validate_args=True)
      self.evaluate(bijector.forward([1., 1.]))

  def testBijectorForwardInverseEventDimsZero(self):
    bijector = tfb.Softplus()
    self.assertEqual("softplus", bijector.name)
    x = 2 * rng.randn(2, 10)
    y = self._softplus(x)

    self.assertAllClose(y, self.evaluate(bijector.forward(x)))
    self.assertAllClose(x, self.evaluate(bijector.inverse(y)))

  def testBijectorForwardInverseWithHingeSoftnessEventDimsZero(self):
    bijector = tfb.Softplus(hinge_softness=1.5)
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
    self.assertEqual("softplus", bijector.name)
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
        bijector, lower_x=-2., upper_x=2., eval_func=self.evaluate)

  def testScalarCongruencyWithPositiveHingeSoftness(self):
    bijector = tfb.Softplus(hinge_softness=1.3)
    bijector_test_util.assert_scalar_congruency(
        bijector, lower_x=-2., upper_x=2., eval_func=self.evaluate)

  def testScalarCongruencyWithNegativeHingeSoftness(self):
    bijector = tfb.Softplus(hinge_softness=-1.3)
    bijector_test_util.assert_scalar_congruency(
        bijector, lower_x=-2., upper_x=2., eval_func=self.evaluate)

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

  def testBijectiveAndFinite16bit(self):
    bijector = tfb.Softplus()
    # softplus(-20) is zero, so we can't use such a large range as in 32bit.
    x = np.linspace(-10., 20., 100).astype(np.float16)
    # Note that float16 is only in the open set (0, inf) for a smaller
    # logspace range.  The actual range was (-7, 4), so use something smaller
    # for the test.
    y = np.logspace(-6, 3, 100).astype(np.float16)
    bijector_test_util.assert_bijective_and_finite(
        bijector, x, y, eval_func=self.evaluate, event_ndims=0, rtol=1e-1,
        atol=1e-3)


if __name__ == "__main__":
  tf.test.main()
