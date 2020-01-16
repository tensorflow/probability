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
class PowerTransformBijectorTest(test_util.TestCase):
  """Tests correctness of the power transformation."""

  def testBijector(self):
    c = 0.2
    bijector = tfb.PowerTransform(power=c, validate_args=True)
    self.assertStartsWith(bijector.name, 'power_transform')
    x = np.array([[[-1.], [2.], [-5. + 1e-4]]])
    y = (1. + x * c)**(1. / c)
    self.assertAllClose(y, self.evaluate(bijector.forward(x)))
    self.assertAllClose(x, self.evaluate(bijector.inverse(y)))
    self.assertAllClose(
        (c - 1.) * np.sum(np.log(y), axis=-1),
        self.evaluate(bijector.inverse_log_det_jacobian(y, event_ndims=1)))
    self.assertAllClose(
        self.evaluate(-bijector.inverse_log_det_jacobian(y, event_ndims=1)),
        self.evaluate(bijector.forward_log_det_jacobian(x, event_ndims=1)),
        rtol=1e-4,
        atol=0.)

  def testScalarCongruency(self):
    bijector = tfb.PowerTransform(power=0.2, validate_args=True)
    bijector_test_util.assert_scalar_congruency(
        bijector, lower_x=-2., upper_x=1.5, eval_func=self.evaluate, rtol=0.05)

  def testBijectiveAndFinite(self):
    bijector = tfb.PowerTransform(power=0.2, validate_args=True)
    x = np.linspace(-4.999, 10, num=10).astype(np.float32)
    y = np.logspace(0.001, 10, num=10).astype(np.float32)
    bijector_test_util.assert_bijective_and_finite(
        bijector, x, y, eval_func=self.evaluate, event_ndims=0, rtol=1e-3)


if __name__ == '__main__':
  tf.test.main()
