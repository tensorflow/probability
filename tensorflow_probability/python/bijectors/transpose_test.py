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
"""Tests for `Transpose` Bijector."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
import numpy as np

import tensorflow as tf

from tensorflow_probability.python import bijectors as tfb
from tensorflow.python.framework import test_util  # pylint: disable=g-direct-tensorflow-import,g-import-not-at-top


class _TransposeBijectorTest(object):
  """Tests correctness of the `Transpose` bijector."""

  def testTransposeFromPerm(self):
    perm_ = [2, 0, 1]
    actual_x_ = np.array([
        [[1, 2],
         [3, 4]],
        [[5, 6],
         [7, 8]],
    ], dtype=np.float32)
    actual_y_ = np.array([
        [[1, 3],
         [5, 7]],
        [[2, 4],
         [6, 8]],
    ], dtype=np.float32)
    if self.is_static:
      actual_x = tf.constant(actual_x_)
      actual_y = tf.constant(actual_y_)
      perm = tf.constant(perm_)
    else:
      actual_x = tf.compat.v1.placeholder_with_default(actual_x_, shape=None)
      actual_y = tf.compat.v1.placeholder_with_default(actual_y_, shape=None)
      perm = tf.compat.v1.placeholder_with_default(perm_, shape=[3])

    bijector = tfb.Transpose(perm=perm, validate_args=True)
    y = bijector.forward(actual_x)
    x = bijector.inverse(actual_y)
    fldj = bijector.forward_log_det_jacobian(x, event_ndims=3)
    ildj = bijector.inverse_log_det_jacobian(y, event_ndims=3)

    [y_, x_, ildj_, fldj_] = self.evaluate([y, x, ildj, fldj])

    self.assertEqual('transpose', bijector.name)
    self.assertAllEqual(actual_y, y_)
    self.assertAllEqual(actual_x, x_)
    self.assertAllEqual(0., ildj_)
    self.assertAllEqual(0., fldj_)

  def testTransposeFromEventNdim(self):
    rightmost_transposed_ndims_ = np.array(2, dtype=np.int32)
    actual_x_ = np.array([
        [[1, 2],
         [3, 4]],
        [[5, 6],
         [7, 8]],
    ], dtype=np.float32)
    actual_y_ = np.array([
        [[1, 3],
         [2, 4]],
        [[5, 7],
         [6, 8]],
    ], dtype=np.float32)
    if self.is_static:
      actual_x = tf.constant(actual_x_)
      actual_y = tf.constant(actual_y_)
      rightmost_transposed_ndims = tf.constant(rightmost_transposed_ndims_)
    else:
      actual_x = tf.compat.v1.placeholder_with_default(actual_x_, shape=None)
      actual_y = tf.compat.v1.placeholder_with_default(actual_y_, shape=None)
      rightmost_transposed_ndims = tf.constant(rightmost_transposed_ndims_)

    bijector = tfb.Transpose(
        rightmost_transposed_ndims=rightmost_transposed_ndims,
        validate_args=True)
    y = bijector.forward(actual_x)
    x = bijector.inverse(actual_y)
    fldj = bijector.forward_log_det_jacobian(x, event_ndims=2)
    ildj = bijector.inverse_log_det_jacobian(y, event_ndims=2)

    [y_, x_, ildj_, fldj_] = self.evaluate([y, x, ildj, fldj])

    self.assertEqual('transpose', bijector.name)
    self.assertAllEqual(actual_y, y_)
    self.assertAllEqual(actual_x, x_)
    self.assertAllEqual(0., ildj_)
    self.assertAllEqual(0., fldj_)

  def testInvalidPermException(self):
    msg = '`perm` must be a valid permutation vector.'
    if self.is_static or tf.executing_eagerly():
      with self.assertRaisesRegexp(ValueError, msg):
        bijector = tfb.Transpose(perm=[1, 2], validate_args=True)
    else:
      with self.assertRaisesOpError(msg):
        bijector = tfb.Transpose(
            perm=tf.compat.v1.placeholder_with_default([1, 2], shape=[2]),
            validate_args=True)
        self.evaluate(bijector.forward([[0, 1]]))

  def testInvalidEventNdimsException(self):
    msg = '`rightmost_transposed_ndims` must be non-negative.'
    with self.assertRaisesRegexp(ValueError, msg):
      tfb.Transpose(rightmost_transposed_ndims=-1, validate_args=True)


@test_util.run_all_in_graph_and_eager_modes
class TransposeBijectorDynamicTest(_TransposeBijectorTest, tf.test.TestCase):
  is_static = False


@test_util.run_all_in_graph_and_eager_modes
class TransposeBijectorStaticTest(_TransposeBijectorTest, tf.test.TestCase):
  is_static = True


if __name__ == '__main__':
  tf.test.main()
