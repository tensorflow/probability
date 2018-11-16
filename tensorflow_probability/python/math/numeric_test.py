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
"""Tests for tensorflow_probability.python.math.numeric."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import tensorflow as tf

from tensorflow_probability.python.internal import test_case
from tensorflow_probability.python.math import numeric
tfe = tf.contrib.eager


@tfe.run_all_tests_in_graph_and_eager_modes
class NumericTest(test_case.TestCase, parameterized.TestCase):

  # Expected values computed using arbitrary precision.
  # pyformat: disable
  # pylint: disable=bad-whitespace
  @parameterized.parameters(
      # x      expected_y   expected_dy_dx
      (-1e30,  138.155106, -2e-30),
      (-3.,    2.302585,   -0.6),
      (-1.,    0.693147,   -1.),
      ( 1e-30, 0.,          0.),
      ( 0.,    0.,          0.),
      ( 1e-30, 0.,          0.),
      ( 1.,    0.693147,    1.),
      ( 3.,    2.302585,    0.6),
      ( 1e30,  138.155106,  2e-30)
  )
  # pylint: enable=bad-whitespace
  # pyformat: enable
  def test_log1psquare(self, x, expected_y, expected_dy_dx):
    x = tf.convert_to_tensor(x)
    with tf.GradientTape() as tape:
      tape.watch(x)
      y = numeric.log1psquare(x)
    dy_dx = tape.gradient(y, x)

    y, dy_dx = self.evaluate([y, dy_dx])

    self.assertAllClose(expected_y, y)
    self.assertAllClose(expected_dy_dx, dy_dx)


if __name__ == '__main__':
  tf.test.main()
