# Copyright 2019 The TensorFlow Probability Authors.
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
"""Tests for the Pad Bijector."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v1 as tf1
import tensorflow.compat.v2 as tf

from tensorflow_probability.python import bijectors as tfb
from tensorflow_probability.python.internal import test_util


@test_util.test_all_tf_execution_regimes
class PadBijectorTest(test_util.TestCase):

  def test_defaults(self):
    b = tfb.Pad(validate_args=True)
    y1 = b.forward([3., 4.])
    y2 = b.forward([[1., 2.], [3., 4.]])
    x1 = b.inverse([3., 4., 0.])
    x2 = b.inverse([[1., 2., 0.], [3., 4., 0]])
    fldj = b.forward_log_det_jacobian([43.], event_ndims=1)
    ildj = b.inverse_log_det_jacobian([45., 0.], event_ndims=1)
    [y1_, y2_, x1_, x2_, fldj_, ildj_] = self.evaluate([
        y1, y2, x1, x2, fldj, ildj])
    self.assertAllEqual([3., 4., 0.], y1_)
    self.assertAllEqual([[1., 2., 0.], [3., 4., 0.]], y2_)
    self.assertAllEqual([3., 4.], x1_)
    self.assertAllEqual([[1., 2.], [3., 4.]], x2_)
    self.assertAllEqual(0., fldj_)
    self.assertAllEqual(0., ildj_)

  def test_left_right_3d(self):
    x1_actual = [[[3., 4.]]]
    y1_expected = [[
        [0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 3., 4., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0.],
    ]]
    x2_actual = [[[1., 2.],
                  [3., 4.]]]
    y2_expected = [[
        [0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 1., 2., 0., 0., 0., 0.],
        [0., 0., 0., 3., 4., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0.],
    ]]

    b = tfb.Pad(paddings=[[1, 2], [3, 4]], validate_args=True)
    y1 = b.forward(x1_actual)
    y2 = b.forward(x2_actual)
    x1 = b.inverse(y1_expected)
    x2 = b.inverse(y2_expected)
    fldj = b.forward_log_det_jacobian([[43.]], event_ndims=2)
    ildj = b.inverse_log_det_jacobian([[45., 0.]], event_ndims=2)
    [y1_, y2_, x1_, x2_, fldj_, ildj_] = self.evaluate([
        y1, y2, x1, x2, fldj, ildj])

    self.assertAllEqual(y1_expected, y1_)
    self.assertAllEqual(y2_expected, y2_)
    self.assertAllEqual(x1_actual, x1_)
    self.assertAllEqual(x2_actual, x2_)
    self.assertEqual(0., fldj_)
    self.assertEqual(0., ildj_)

  @test_util.jax_disable_variable_test
  def test_variable_paddings(self):
    x = tf.Variable([[1, 2]])
    b = tfb.Pad(paddings=x, validate_args=True)
    y = b.forward([[1, 2]])
    self.evaluate(b.paddings.initializer)
    self.assertAllEqual([[0, 1, 2, 0, 0]], self.evaluate(y))
    with tf.control_dependencies([b.paddings.assign([[1, 0]])]):
      y = b.forward([[1, 2]])
    self.assertAllEqual([[0, 1, 2]], self.evaluate(y))

  def test_axis_exceptions(self):
    if not tf.executing_eagerly():
      with self.assertRaisesWithPredicateMatch(
          NotImplementedError, 'Argument `axis` must be known statically.'):
        tfb.Pad(axis=tf1.placeholder_with_default([-1], shape=None),
                validate_args=True)
    with self.assertRaisesWithPredicateMatch(
        ValueError, 'Argument `axis` must be scalar or vector.'):
      tfb.Pad(axis=[[-1]], validate_args=True)
    with self.assertRaisesWithPredicateMatch(
        ValueError, 'Argument `axis` must be negative.'):
      tfb.Pad(axis=0, validate_args=True)
    with self.assertRaisesWithPredicateMatch(
        ValueError, 'Argument `axis` elements must be unique.'):
      tfb.Pad(axis=[-1, -1], validate_args=True)

  def test_paddings_exceptions(self):
    with self.assertRaisesWithPredicateMatch(
        ValueError, 'Argument `paddings` must be a vector of pairs.'):
      tfb.Pad(paddings=-1, validate_args=True)
    with self.assertRaisesWithPredicateMatch(
        ValueError, 'Argument `paddings` must be non-negative.'):
      tfb.Pad(paddings=[[-1, 0]], validate_args=True)
    with self.assertRaisesWithPredicateMatch(
        ValueError,
        ('Arguments `axis` and `paddings` must have the same number '
         'of elements.')):
      tfb.Pad(paddings=[[1, 0]], axis=[-2, -1], validate_args=True)

    if tf.executing_eagerly():
      return

    with self.assertRaisesWithPredicateMatch(
        tf.errors.InvalidArgumentError,
        'Argument `paddings` must be a vector of pairs.'):
      b = tfb.Pad(paddings=tf1.placeholder_with_default([[1]], shape=None),
                  axis=-1, validate_args=True)
      self.evaluate(b.forward([0]))
    with self.assertRaisesWithPredicateMatch(
        tf.errors.InvalidArgumentError,
        'Argument `paddings` must be non-negative.'):
      b = tfb.Pad(paddings=tf1.placeholder_with_default([[-1, 0]], shape=None),
                  axis=-1, validate_args=True)
      self.evaluate(b.forward([0]))
    with self.assertRaisesWithPredicateMatch(
        tf.errors.InvalidArgumentError,
        ('Arguments `axis` and `paddings` must have the same number '
         'of elements.')):
      b = tfb.Pad(paddings=tf1.placeholder_with_default([[1, 0]], shape=None),
                  axis=[-2, -1], validate_args=True)
      self.evaluate(b.forward([0]))


if __name__ == '__main__':
  tf.test.main()
