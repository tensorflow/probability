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
"""Sinh Tests."""

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
class SinhBijectorTest(test_util.TestCase):
  """Tests correctness of the Y = g(X) = sinh(X) transformation."""

  def testBijector(self):
    self.assertStartsWith(tfb.Sinh().name, "sinh")
    x = np.linspace(-50., 50., 100).reshape([2, 5, 10]).astype(np.float64)
    y = np.sinh(x)
    ildj = -0.5 * np.log1p(np.square(y))
    bijector = tfb.Sinh()
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
        tfb.Sinh(), lower_x=-7., upper_x=7., eval_func=self.evaluate,
        n=int(1e4), rtol=.5)

  @parameterized.parameters(np.float32, np.float64)
  def testBijectiveAndFinite(self, dtype):
    bijector = tfb.Sinh(validate_args=True)
    # Use bounds that are very large to check that the transformation remains
    # bijective. We stray away from the largest/smallest value to avoid issues
    # at the boundary since XLA sinh will return `inf` for the largest value.
    y = np.array([np.nextafter(np.finfo(dtype).min, 0.) / 10.,
                  np.nextafter(np.finfo(dtype).max, 0.) / 10.], dtype=dtype)
    # Calculate `x` based on `y` which in turn is in widest range possible.
    x = np.arcsinh(y)
    bijector_test_util.assert_bijective_and_finite(
        bijector, x, y, eval_func=self.evaluate, event_ndims=0, atol=0.,
        rtol=1e-4)


if __name__ == "__main__":
  tf.test.main()
