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
import tensorflow as tf
from tensorflow_probability.python import bijectors as tfb

from tensorflow.python.framework import test_util  # pylint: disable=g-direct-tensorflow-import,g-import-not-at-top


@test_util.run_all_in_graph_and_eager_modes
class FillTriangularBijectorTest(tf.test.TestCase):
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
    self.assertAllEqual(y_.shape.as_list(), y_shape.as_list())
    x_ = b.inverse(y_)
    self.assertAllEqual(x_.shape.as_list(), x_shape.as_list())

    y_shape_ = b.forward_event_shape(x_shape)
    self.assertAllEqual(y_shape_.as_list(), y_shape.as_list())
    x_shape_ = b.inverse_event_shape(y_shape)
    self.assertAllEqual(x_shape_.as_list(), x_shape.as_list())

    y_shape_tensor = self.evaluate(
        b.forward_event_shape_tensor(x_shape.as_list()))
    self.assertAllEqual(y_shape_tensor, y_shape.as_list())
    x_shape_tensor = self.evaluate(
        b.inverse_event_shape_tensor(y_shape.as_list()))
    self.assertAllEqual(x_shape_tensor, x_shape.as_list())

  def testShapeError(self):

    b = tfb.FillTriangular(validate_args=True)

    x_shape_bad = tf.TensorShape([5, 4, 7])
    with self.assertRaisesRegexp(ValueError, "is not a triangular number"):
      b.forward_event_shape(x_shape_bad)
    with self.assertRaisesOpError("is not a triangular number"):
      self.evaluate(b.forward_event_shape_tensor(x_shape_bad.as_list()))

    y_shape_bad = tf.TensorShape([5, 4, 3, 2])
    with self.assertRaisesRegexp(ValueError, "Matrix must be square"):
      b.inverse_event_shape(y_shape_bad)
    with self.assertRaisesOpError("Matrix must be square"):
      self.evaluate(b.inverse_event_shape_tensor(y_shape_bad.as_list()))


if __name__ == "__main__":
  tf.test.main()
