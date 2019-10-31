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

import tensorflow.compat.v1 as tf1
import tensorflow.compat.v2 as tf

from tensorflow_probability.python import bijectors as tfb
from tensorflow_probability.python import distributions as tfd
from tensorflow_probability.python.internal import tensorshape_util
from tensorflow_probability.python.internal import test_util


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
      actual_x = tf1.placeholder_with_default(actual_x_, shape=None)
      actual_y = tf1.placeholder_with_default(actual_y_, shape=None)
      perm = tf1.placeholder_with_default(perm_, shape=[3])

    bijector = tfb.Transpose(perm=perm, validate_args=True)
    y = bijector.forward(actual_x)
    x = bijector.inverse(actual_y)
    fldj = bijector.forward_log_det_jacobian(x, event_ndims=3)
    ildj = bijector.inverse_log_det_jacobian(y, event_ndims=3)

    [y_, x_, ildj_, fldj_] = self.evaluate([y, x, ildj, fldj])

    self.assertStartsWith(bijector.name, 'transpose')
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
      actual_x = tf1.placeholder_with_default(actual_x_, shape=None)
      actual_y = tf1.placeholder_with_default(actual_y_, shape=None)
      rightmost_transposed_ndims = tf.constant(rightmost_transposed_ndims_)

    bijector = tfb.Transpose(
        rightmost_transposed_ndims=rightmost_transposed_ndims,
        validate_args=True)
    y = bijector.forward(actual_x)
    x = bijector.inverse(actual_y)
    fldj = bijector.forward_log_det_jacobian(x, event_ndims=2)
    ildj = bijector.inverse_log_det_jacobian(y, event_ndims=2)

    [y_, x_, ildj_, fldj_] = self.evaluate([y, x, ildj, fldj])

    self.assertStartsWith(bijector.name, 'transpose')
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
            perm=tf1.placeholder_with_default([1, 2], shape=[2]),
            validate_args=True)
        self.evaluate(bijector.forward([[0, 1]]))

  def testInvalidEventNdimsException(self):
    msg = '`rightmost_transposed_ndims` must be non-negative.'
    with self.assertRaisesRegexp(ValueError, msg):
      tfb.Transpose(rightmost_transposed_ndims=-1, validate_args=True)

  def testTransformedDist(self):
    d = tfd.Independent(tfd.Normal(tf.zeros([4, 3, 2]), 1), 3)
    dt = tfb.Transpose([1, 0])(d)
    self.assertEqual((4, 3, 2), d.event_shape)
    self.assertEqual((4, 2, 3), dt.event_shape)
    dt = tfb.Invert(tfb.Transpose([1, 0, 2]))(d)
    self.assertEqual((4, 3, 2), d.event_shape)
    self.assertEqual((3, 4, 2), dt.event_shape)

  def testEventShapes(self):
    shape_static = [5, 4, 3, 2]
    shape_dynamic = tf1.placeholder_with_default(
        tf.constant(shape_static), shape=None)

    def make_bijector(perm=None, rightmost_transposed_ndims=None):
      if perm is not None:
        perm = tf.convert_to_tensor(value=perm)
        if not self.is_static:
          perm = tf1.placeholder_with_default(perm, shape=perm.shape)
      return tfb.Transpose(
          perm, rightmost_transposed_ndims=rightmost_transposed_ndims)

    for is_shape_static, shape, shape_t in [
        (True, tf.zeros(shape_static).shape, tf.constant(shape_static)),
        (False, tf.zeros(shape_dynamic).shape, shape_dynamic)]:

      # pylint: disable=cell-var-from-loop
      def event_shape(b, direction):
        shape_fn = getattr(b, '{}_event_shape'.format(direction))
        if (is_shape_static and self.is_static) or tf.executing_eagerly():
          result = shape_fn(shape)
          self.assertTrue(tensorshape_util.is_fully_defined(result))
          return result
        if is_shape_static:
          self.assertEqual(len(shape), shape_fn(shape).ndims)
        else:
          self.assertIsNone(shape_fn(shape).ndims)
        shape_tensor_fn = getattr(b, '{}_event_shape_tensor'.format(direction))
        return self.evaluate(shape_tensor_fn(shape_t))
      # pylint: enable=cell-var-from-loop

      self.assertAllEqual((5, 3, 4, 2),
                          event_shape(make_bijector([1, 0, 2]), 'forward'))
      self.assertAllEqual((5, 2, 4, 3),
                          event_shape(make_bijector([2, 0, 1]), 'forward'))
      self.assertAllEqual(
          (5, 4, 2, 3),
          event_shape(make_bijector(rightmost_transposed_ndims=2), 'forward'))
      self.assertAllEqual(
          (5, 2, 3, 4),
          event_shape(make_bijector(rightmost_transposed_ndims=3), 'forward'))
      self.assertAllEqual((5, 3, 4, 2),
                          event_shape(make_bijector([1, 0, 2]), 'inverse'))
      self.assertAllEqual((5, 3, 2, 4),
                          event_shape(make_bijector([2, 0, 1]), 'inverse'))
      self.assertAllEqual(
          (5, 4, 2, 3),
          event_shape(make_bijector(rightmost_transposed_ndims=2), 'inverse'))
      self.assertAllEqual(
          (5, 2, 3, 4),
          event_shape(make_bijector(rightmost_transposed_ndims=3), 'inverse'))

  def testPartialStaticPermEventShapes(self):
    if tf.executing_eagerly(): return  # this test is not interesting in eager.
    perm = tf.convert_to_tensor(value=[
        tf.constant(2),
        tf1.placeholder_with_default(0, []),
        tf1.placeholder_with_default(1, [])
    ])
    self.assertAllEqual([2, None, None], tf.get_static_value(
        perm, partial=True))
    b = tfb.Transpose(perm)
    self.assertAllEqual([8, 5, None, None],
                        b.forward_event_shape([8, 7, 6, 5]).as_list())
    self.assertAllEqual([8, None, None, 7],
                        b.inverse_event_shape([8, 7, 6, 5]).as_list())

    # Process of elimination should allow us to deduce one non-static perm idx.
    perm = tf.convert_to_tensor(value=[
        tf.constant(2),
        tf1.placeholder_with_default(0, []),
        tf.constant(1)
    ])
    self.assertAllEqual([2, None, 1], tf.get_static_value(perm, partial=True))
    b = tfb.Transpose(perm)
    self.assertAllEqual([8, 5, 7, 6], b.forward_event_shape([8, 7, 6, 5]))
    self.assertAllEqual([8, 6, 5, 7], b.inverse_event_shape([8, 7, 6, 5]))

  def testNonNegativeAssertion(self):
    message = '`rightmost_transposed_ndims` must be non-negative'
    with self.assertRaisesRegexp(Exception, message):
      ndims = np.int32(-3)
      bijector = tfb.Transpose(rightmost_transposed_ndims=ndims,
                               validate_args=True)
      x = np.random.randn(4, 2, 3)
      _ = self.evaluate(bijector.forward(x))

  def testNonPermutationAssertion(self):
    message = '`perm` must be a valid permutation vector'
    with self.assertRaisesRegexp(Exception, message):
      permutation = np.int32([1, 0, 1])
      bijector = tfb.Transpose(perm=permutation, validate_args=True)
      x = np.random.randn(4, 2, 3)
      _ = self.evaluate(bijector.forward(x))

  def testVariableNonPermutationAssertion(self):
    message = '`perm` must be a valid permutation vector'
    permutation = tf.Variable(np.int32([1, 0, 1]))
    self.evaluate(permutation.initializer)
    with self.assertRaisesRegexp(Exception, message):
      bijector = tfb.Transpose(perm=permutation, validate_args=True)
      x = np.random.randn(4, 2, 3)
      _ = self.evaluate(bijector.forward(x))

  def testModifiedVariableNonPermutationAssertion(self):
    message = '`perm` must be a valid permutation vector'
    permutation = tf.Variable(np.int32([1, 0, 2]))
    self.evaluate(permutation.initializer)
    bijector = tfb.Transpose(perm=permutation, validate_args=True)
    with self.assertRaisesRegexp(Exception, message):
      with tf.control_dependencies([permutation.assign([1, 0, 1])]):
        x = np.random.randn(4, 2, 3)
        _ = self.evaluate(bijector.forward(x))


@test_util.test_all_tf_execution_regimes
class TransposeBijectorDynamicTest(_TransposeBijectorTest, test_util.TestCase):
  is_static = False


@test_util.test_all_tf_execution_regimes
class TransposeBijectorStaticTest(_TransposeBijectorTest, test_util.TestCase):
  is_static = True


if __name__ == '__main__':
  tf.test.main()
