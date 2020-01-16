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
"""Tests for FillTriangular bijector."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
import numpy as np

import tensorflow.compat.v2 as tf

from tensorflow_probability.python import bijectors as tfb
from tensorflow_probability.python.internal import tensorshape_util
from tensorflow_probability.python.internal import test_util


@test_util.test_all_tf_execution_regimes
class FillTriangularBijectorTest(test_util.TestCase):
  """Tests the correctness of the FillTriangular bijector."""

  def testBijector(self):
    x = np.float32(np.array([1., 2., 3.]))
    y = np.float32(np.array([[3., 0.],
                             [2., 1.]]))

    b = tfb.FillTriangular()

    y_ = self.evaluate(b.forward(x))
    self.assertAllClose(y, y_)

    x_ = self.evaluate(b.inverse(y))
    self.assertAllClose(x, x_)

    fldj = self.evaluate(b.forward_log_det_jacobian(x, event_ndims=1))
    self.assertAllClose(fldj, 0.)

    ildj = self.evaluate(b.inverse_log_det_jacobian(y, event_ndims=2))
    self.assertAllClose(ildj, 0.)

  def testShape(self):
    x_shape = tf.TensorShape([5, 4, 6])
    y_shape = tf.TensorShape([5, 4, 3, 3])

    b = tfb.FillTriangular(validate_args=True)

    x = tf.ones(shape=x_shape, dtype=tf.float32)
    y_ = b.forward(x)
    self.assertAllEqual(
        tensorshape_util.as_list(y_.shape), tensorshape_util.as_list(y_shape))
    x_ = b.inverse(y_)
    self.assertAllEqual(
        tensorshape_util.as_list(x_.shape), tensorshape_util.as_list(x_shape))

    y_shape_ = b.forward_event_shape(x_shape)
    self.assertAllEqual(
        tensorshape_util.as_list(y_shape_), tensorshape_util.as_list(y_shape))
    x_shape_ = b.inverse_event_shape(y_shape)
    self.assertAllEqual(
        tensorshape_util.as_list(x_shape_), tensorshape_util.as_list(x_shape))

    y_shape_tensor = self.evaluate(
        b.forward_event_shape_tensor(tensorshape_util.as_list(x_shape)))
    self.assertAllEqual(y_shape_tensor, tensorshape_util.as_list(y_shape))
    x_shape_tensor = self.evaluate(
        b.inverse_event_shape_tensor(tensorshape_util.as_list(y_shape)))
    self.assertAllEqual(x_shape_tensor, tensorshape_util.as_list(x_shape))

  def testShapeError(self):

    b = tfb.FillTriangular(validate_args=True)

    x_shape_bad = tf.TensorShape([5, 4, 7])
    with self.assertRaisesRegexp(ValueError, 'is not a triangular number'):
      b.forward_event_shape(x_shape_bad)
    with self.assertRaisesOpError('is not a triangular number'):
      self.evaluate(
          b.forward_event_shape_tensor(tensorshape_util.as_list(x_shape_bad)))

    y_shape_bad = tf.TensorShape([5, 4, 3, 2])
    with self.assertRaisesRegexp(ValueError, 'Matrix must be square'):
      b.inverse_event_shape(y_shape_bad)
    with self.assertRaisesOpError('Matrix must be square'):
      self.evaluate(
          b.inverse_event_shape_tensor(tensorshape_util.as_list(y_shape_bad)))


if __name__ == '__main__':
  tf.test.main()
