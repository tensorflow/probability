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

# Dependency imports
from absl.testing import parameterized
from tensorflow_probability.python import bijectors as tfb
from tensorflow_probability.python.internal import test_util


@test_util.test_all_tf_execution_regimes
class HouseholderBijectorTest(test_util.TestCase, parameterized.TestCase):
  """Tests the correctness of the Householder bijector."""

  def testComputesCorrectValues(self):
    reflection_axis1 = [-0.8, 0.6]
    bijector1 = tfb.Householder(reflection_axis=reflection_axis1,
                                validate_args=True)
    x1 = [1., 0.]
    y1 = [-0.28, 0.96]
    self.assertAllClose(y1, bijector1.forward(x1))
    self.assertAllClose(x1, bijector1.inverse(y1))

    reflection_axis2 = [2/7, 3/7, 6/7]
    bijector2 = tfb.Householder(reflection_axis=reflection_axis2,
                                validate_args=True)
    x2 = [1., 0., 0.]
    y2 = [41/49, -12/49, -24/49]
    self.assertAllClose(y2, bijector2.forward(x2))
    self.assertAllClose(x2, bijector2.inverse(y2))
    self.assertAllClose(0.,
                        bijector2.inverse_log_det_jacobian(x2, event_ndims=1))

  @parameterized.named_parameters(
      dict(testcase_name='static', is_static=True),
      dict(testcase_name='dynamic', is_static=False),
  )

  def testBatch(self, is_static):
    reflection_axis1 = [[-0.8, 0.6], [0.8, 0.6]]
    bijector1 = tfb.Householder(reflection_axis=reflection_axis1,
                                validate_args=True)
    x = self.maybe_static([1., 1.], is_static)

    self.assertAllClose([[0.68, 1.24], [-1.24, -0.68]], bijector1.forward(x))
    self.assertAllClose(
        0.,
        bijector1.inverse_log_det_jacobian(x, event_ndims=1))

    reflection_axis2 = [2/7, 3/7, 6/7]
    bijector2 = tfb.Householder(reflection_axis=reflection_axis2,
                                validate_args=True)
    x = self.maybe_static([[1., 0., 0.], [0., 0., 1.]], is_static)

    self.assertAllClose([[41/49, -12/49, -24/49], [-24/49, -36/49, -23/49]],
                        bijector2.forward(x))
    self.assertAllClose(
        0.,
        bijector2.inverse_log_det_jacobian(x, event_ndims=1))


if __name__ == '__main__':
  test_util.main()
