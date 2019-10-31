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
class SquareBijectorTest(test_util.TestCase):
  """Tests the correctness of the Y = X ** 2 transformation."""

  def testBijectorScalar(self):
    bijector = tfb.Square(validate_args=True)
    self.assertStartsWith(bijector.name, "square")
    x = [[[1., 5],
          [2, 1]],
         [[np.sqrt(2.), 3],
          [np.sqrt(8.), 1]]]
    y = np.square(x)
    ildj = -np.log(2.) - np.log(x)
    self.assertAllClose(y, self.evaluate(bijector.forward(x)))
    self.assertAllClose(x, self.evaluate(bijector.inverse(y)))
    self.assertAllClose(
        ildj,
        self.evaluate(bijector.inverse_log_det_jacobian(
            y, event_ndims=0)), atol=0., rtol=1e-7)
    self.assertAllClose(
        self.evaluate(-bijector.inverse_log_det_jacobian(y, event_ndims=0)),
        self.evaluate(bijector.forward_log_det_jacobian(x, event_ndims=0)),
        atol=0.,
        rtol=1e-7)

  def testScalarCongruency(self):
    bijector = tfb.Square(validate_args=True)
    bijector_test_util.assert_scalar_congruency(
        bijector, lower_x=1e-3, upper_x=1.5, eval_func=self.evaluate,
        rtol=0.08)


if __name__ == "__main__":
  tf.test.main()
