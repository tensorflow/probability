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
import tensorflow.compat.v2 as tf
from tensorflow_probability.python import bijectors as tfb
from tensorflow_probability.python.bijectors import bijector_test_util
from tensorflow_probability.python.internal import test_util


@test_util.test_all_tf_execution_regimes
class SoftsignBijectorTest(test_util.TestCase):
  """Tests the correctness of the Y = g(X) = X / (1 + |X|) transformation."""

  def _softsign(self, x):
    return x / (1. + np.abs(x))

  def _softsign_ildj_before_reduction(self, y):
    """Inverse log det jacobian, before being reduced."""
    return -2. * np.log1p(-np.abs(y))

  def testBijectorBounds(self):
    bijector = tfb.Softsign(validate_args=True)
    with self.assertRaisesOpError(">= -1"):
      self.evaluate(bijector.inverse(-3.))
    with self.assertRaisesOpError(">= -1"):
      self.evaluate(bijector.inverse_log_det_jacobian(-3., event_ndims=0))

    with self.assertRaisesOpError("<= 1"):
      self.evaluate(bijector.inverse(3.))
    with self.assertRaisesOpError("<= 1"):
      self.evaluate(bijector.inverse_log_det_jacobian(3., event_ndims=0))

  def testBijectorForwardInverse(self):
    bijector = tfb.Softsign(validate_args=True)
    self.assertStartsWith(bijector.name, "softsign")
    x = 2. * np.random.randn(2, 10)
    y = self._softsign(x)

    self.assertAllClose(y, self.evaluate(bijector.forward(x)))
    self.assertAllClose(x, self.evaluate(bijector.inverse(y)))

  def testBijectorLogDetJacobianEventDimsZero(self):
    bijector = tfb.Softsign(validate_args=True)
    y = np.random.rand(2, 10)
    # No reduction needed if event_dims = 0.
    ildj = self._softsign_ildj_before_reduction(y)

    self.assertAllClose(ildj, self.evaluate(
        bijector.inverse_log_det_jacobian(y, event_ndims=0)))

  def testBijectorForwardInverseEventDimsOne(self):
    bijector = tfb.Softsign(validate_args=True)
    self.assertStartsWith(bijector.name, "softsign")
    x = 2. * np.random.randn(2, 10)
    y = self._softsign(x)
    self.assertAllClose(y, self.evaluate(bijector.forward(x)))
    self.assertAllClose(x, self.evaluate(bijector.inverse(y)))

  def testBijectorLogDetJacobianEventDimsOne(self):
    bijector = tfb.Softsign(validate_args=True)
    y = np.random.rand(2, 10)
    ildj_before = self._softsign_ildj_before_reduction(y)
    ildj = np.sum(ildj_before, axis=1)
    self.assertAllClose(
        ildj, self.evaluate(
            bijector.inverse_log_det_jacobian(y, event_ndims=1)))

  def testScalarCongruency(self):
    bijector = tfb.Softsign(validate_args=True)
    bijector_test_util.assert_scalar_congruency(
        bijector, lower_x=-20., upper_x=20., eval_func=self.evaluate, rtol=.05)

  def testBijectiveAndFinite(self):
    bijector = tfb.Softsign(validate_args=True)
    x = np.linspace(-20., 20., 100).astype(np.float32)
    y = np.linspace(-0.99, 0.99, 100).astype(np.float32)
    bijector_test_util.assert_bijective_and_finite(
        bijector, x, y, eval_func=self.evaluate, event_ndims=0, rtol=1e-3,
        atol=1e-3)


if __name__ == "__main__":
  tf.test.main()
