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
"""Tests for Bijector."""

# Dependency imports

import numpy as np
import tensorflow.compat.v2 as tf
from tensorflow_probability.python.bijectors import bijector_test_util
from tensorflow_probability.python.bijectors import shifted_gompertz_cdf
from tensorflow_probability.python.internal import test_util


@test_util.test_all_tf_execution_regimes
class ShiftedGompertzCDF(test_util.TestCase):
  """Tests correctness of the Shifted Gompertz bijector."""

  def testScalarCongruency(self):
    bijector_test_util.assert_scalar_congruency(
        shifted_gompertz_cdf.ShiftedGompertzCDF(concentration=0.1, rate=0.4),
        lower_x=1.,
        upper_x=10.,
        eval_func=self.evaluate,
        rtol=0.05)

  def testBijectiveAndFinite(self):
    bijector = shifted_gompertz_cdf.ShiftedGompertzCDF(
        concentration=0.2, rate=0.01, validate_args=True)
    x = np.logspace(-10, 2, num=10).astype(np.float32)
    y = np.linspace(0.01, 0.99, num=10).astype(np.float32)
    bijector_test_util.assert_bijective_and_finite(
        bijector, x, y, eval_func=self.evaluate, event_ndims=0, rtol=1e-3)

  @test_util.jax_disable_variable_test
  def testVariableConcentration(self):
    x = tf.Variable(1.)
    b = shifted_gompertz_cdf.ShiftedGompertzCDF(
        concentration=x, rate=1., validate_args=True)
    self.evaluate(x.initializer)
    self.assertIs(x, b.concentration)
    self.assertEqual((), self.evaluate(b.forward(1.)).shape)
    with self.assertRaisesOpError("Argument `concentration` must be positive."):
      with tf.control_dependencies([x.assign(-1.)]):
        self.evaluate(b.forward(1.))

  @test_util.jax_disable_variable_test
  def testVariableRate(self):
    x = tf.Variable(1.)
    b = shifted_gompertz_cdf.ShiftedGompertzCDF(
        concentration=1., rate=x, validate_args=True)
    self.evaluate(x.initializer)
    self.assertIs(x, b.rate)
    self.assertEqual((), self.evaluate(b.forward(1.)).shape)
    with self.assertRaisesOpError("Argument `rate` must be positive."):
      with tf.control_dependencies([x.assign(-1.)]):
        self.evaluate(b.forward(1.))


if __name__ == "__main__":
  test_util.main()
