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
"""Tests for Power Bijector."""

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
class RaiseBijectorTest(test_util.TestCase):
  """Tests the correctness of the Y = X ** K transformation."""

  def testBijectorScalar(self):
    power = np.array([2.6, 0.3, -1.1])
    bijector = tfb.Power(power=power, validate_args=True)
    self.assertStartsWith(bijector.name, 'power')
    x = np.array([[[1., 5., 3.],
                   [2., 1., 7.]],
                  [[np.sqrt(2.), 3., 1.],
                   [np.sqrt(8.), 1., 0.4]]])
    y = np.power(x, power)
    ildj = -np.log(np.abs(power)) - (power - 1.) * np.log(x)
    self.assertAllClose(y, self.evaluate(bijector.forward(x)))
    self.assertAllClose(x, self.evaluate(bijector.inverse(y)))
    self.assertAllClose(
        ildj,
        self.evaluate(bijector.inverse_log_det_jacobian(
            y, event_ndims=0)), atol=0., rtol=1e-6)
    self.assertAllClose(
        self.evaluate(-bijector.inverse_log_det_jacobian(y, event_ndims=0)),
        self.evaluate(bijector.forward_log_det_jacobian(x, event_ndims=0)),
        atol=0.,
        rtol=1e-7)

  def testScalarCongruency(self):
    bijector = tfb.Power(power=2.6, validate_args=True)
    bijector_test_util.assert_scalar_congruency(
        bijector, lower_x=1e-3, upper_x=1.5, eval_func=self.evaluate,
        n=1e5,
        rtol=0.08)

  @parameterized.named_parameters(
      {'testcase_name': 'Square', 'power': 2., 'cls': tfb.Square},
      {'testcase_name': 'Reciprocal', 'power': -1., 'cls': tfb.Reciprocal})
  def testSpecialCases(self, power, cls):
    b = tfb.Power(power=power)
    b_other = cls()
    x = [[[1., 5],
          [2, 1]],
         [[np.sqrt(2.), 3],
          [np.sqrt(8.), 1]]]
    y, y_other = self.evaluate((b.forward(x), b_other.forward(x)))
    self.assertAllClose(y, y_other)

    x, x_other = self.evaluate((b.inverse(y),
                                b_other.inverse(y)))
    self.assertAllClose(x, x_other)

    ildj, ildj_other = self.evaluate((
        b.inverse_log_det_jacobian(y, event_ndims=0),
        b_other.inverse_log_det_jacobian(y, event_ndims=0)))
    self.assertAllClose(ildj, ildj_other)

  def testPowerOddInteger(self):
    power = np.array([3., -5., 5., -7.]).reshape((4, 1))
    bijector = tfb.Power(power=power, validate_args=True)
    self.assertStartsWith(bijector.name, 'power')
    x = np.linspace(-10., 10., 20)
    y = np.power(x, power)
    ildj = -np.log(np.abs(power)) - (power - 1.) * np.log(np.abs(x))
    self.assertAllClose(y, self.evaluate(bijector.forward(x)))
    self.assertAllClose(x * np.ones((4, 1)), self.evaluate(bijector.inverse(y)))
    self.assertAllClose(
        ildj,
        self.evaluate(bijector.inverse_log_det_jacobian(
            y, event_ndims=0)), atol=0., rtol=1e-6)
    self.assertAllClose(
        self.evaluate(-bijector.inverse_log_det_jacobian(y, event_ndims=0)),
        self.evaluate(bijector.forward_log_det_jacobian(x, event_ndims=0)),
        atol=0.,
        rtol=1e-7)

  def testZeroPowerRaisesError(self):
    with self.assertRaisesRegexp(Exception, 'must be non-zero'):
      b = tfb.Power(power=0., validate_args=True)
      b.forward(1.)

  def testPowerNegativeInputRaisesError(self):
    with self.assertRaisesOpError('must be non-negative'):
      b = tfb.Power(power=2.5, validate_args=True)
      self.evaluate(b.inverse(-1.))

    b = tfb.Power(power=3., validate_args=True)
    self.evaluate(b.inverse(-1.))

if __name__ == '__main__':
  tf.test.main()
