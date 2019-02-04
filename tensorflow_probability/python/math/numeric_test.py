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

# Dependency imports
from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from tensorflow_probability.python.internal import test_case
from tensorflow_probability.python.math import numeric
tfe = tf.contrib.eager


@tfe.run_all_tests_in_graph_and_eager_modes
class Log1pSquareTest32(test_case.TestCase, parameterized.TestCase):

  dtype = tf.float32

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
    x = tf.convert_to_tensor(x, dtype=self.dtype)
    with tf.GradientTape() as tape:
      tape.watch(x)
      y = numeric.log1psquare(x)
    dy_dx = tape.gradient(y, x)

    y, dy_dx = self.evaluate([y, dy_dx])

    self.assertAllClose(expected_y, y)
    self.assertAllClose(expected_dy_dx, dy_dx)


@tfe.run_all_tests_in_graph_and_eager_modes
class Log1pSquareTest64(Log1pSquareTest32):
  dtype = tf.float64


@tfe.run_all_tests_in_graph_and_eager_modes
class SoftThresholdTest(test_case.TestCase, parameterized.TestCase):

  # Expected values computed using arbitrary precision.
  # pyformat: disable
  # pylint: disable=bad-whitespace
  @parameterized.parameters(
      # x   threshold  expected_y  expected_dy_dx
      (5., 5., 0., 1.),
      (2., 5., 0., 0.),
      (-2., 5., 0., 0.),
      (3., 2.5, 0.5, 1.),
      (-3., 2.5, -0.5, 1.),
      (-1., 1., 0., 1.),
      (-6., 5., -1., 1.),
      (0., 0., 0., 0.),
  )
  # pylint: enable=bad-whitespace
  # pyformat: enable
  def test_soft_threshold(self, x, threshold, expected_y, expected_dy_dx):
    x = tf.convert_to_tensor(x)
    with tf.GradientTape() as tape:
      tape.watch(x)
      y = numeric.soft_threshold(x, threshold)
    dy_dx = tape.gradient(y, x)

    y, dy_dx = self.evaluate([y, dy_dx])

    self.assertAllClose(expected_y, y)
    self.assertAllClose(expected_dy_dx, dy_dx)


@tfe.run_all_tests_in_graph_and_eager_modes
class ClipByValuePreserveGrad32(test_case.TestCase, parameterized.TestCase):

  dtype = tf.float32

  # pyformat: disable
  # pylint: disable=bad-whitespace
  @parameterized.parameters(
      # x    lo,   hi,    expected_y
      ( 0.,  -3.,   4.,   0.),
      (-5.,  -3.,   4.,  -3.),
      ( 5.,  -3.,   4.,   4.),
      ([[-4., -2], [2, 5]], -3., 4., [[-3., -2], [2, 4]]),
  )
  # pylint: enable=bad-whitespace
  # pyformat: enable
  def test_clip_by_value_preserve_grad(self, x, lo, hi, expected_y):
    expected_dy_dx = np.ones_like(x)
    x = tf.convert_to_tensor(x, dtype=self.dtype)
    with tf.GradientTape() as tape:
      tape.watch(x)
      y = numeric.clip_by_value_preserve_gradient(x, lo, hi)
    dy_dx = tape.gradient(y, x)

    y, dy_dx = self.evaluate([y, dy_dx])

    self.assertAllClose(expected_y, y)
    self.assertAllClose(expected_dy_dx, dy_dx)


@tfe.run_all_tests_in_graph_and_eager_modes
class ClipByValuePreserveGrad64(ClipByValuePreserveGrad32):

  dtype = tf.float64


if __name__ == '__main__':
  tf.test.main()
