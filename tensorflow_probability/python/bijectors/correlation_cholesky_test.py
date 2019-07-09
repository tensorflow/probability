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
"""Tests for CorrelationCholesky bijector."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools

# Dependency imports
from absl.testing import parameterized
import numpy as np
import tensorflow.compat.v1 as tf1
import tensorflow.compat.v2 as tf

from tensorflow_probability.python import bijectors as tfb
from tensorflow_probability.python.bijectors import bijector_test_util
from tensorflow_probability.python.distributions import lkj
from tensorflow_probability.python.internal import tensorshape_util
from tensorflow.python.framework import test_util  # pylint: disable=g-direct-tensorflow-import


@test_util.run_all_in_graph_and_eager_modes
class CorrelationCholeskyBijectorTest(parameterized.TestCase, tf.test.TestCase):
  """Tests the correctness of the CorrelationCholesky bijector."""

  def testBijector(self):
    x = np.float32(np.array([7., -5., 5., 1., 2., -2.]))
    y = np.float32(
        np.array([[1., 0., 0., 0.], [0.707107, 0.707107, 0., 0.],
                  [-0.666667, 0.666667, 0.333333, 0.], [0.5, -0.5, 0.7, 0.1]]))

    b = tfb.CorrelationCholesky()

    y_ = self.evaluate(b.forward(x))
    self.assertAllClose(y, y_, atol=1e-5, rtol=1e-5)

    x_ = self.evaluate(b.inverse(y))
    self.assertAllClose(x, x_, atol=1e-5, rtol=1e-5)

    expected_fldj = -0.5 * np.sum([2, 3, 4] * np.log([2, 9, 100]))

    fldj = self.evaluate(b.forward_log_det_jacobian(x, event_ndims=1))
    self.assertAllClose(expected_fldj, fldj)

    ildj = self.evaluate(b.inverse_log_det_jacobian(y, event_ndims=2))
    self.assertAllClose(-expected_fldj, ildj)

  def testBijectorBatch(self):
    x = np.float32([[7., -5., 5., 1., 2., -2.], [1., 3., -5., 1., -4., 8.]])
    y = np.float32([
        [[1., 0., 0., 0.], [0.707107, 0.707107, 0., 0.],
         [-0.666667, 0.666667, 0.333333, 0.], [0.5, -0.5, 0.7, 0.1]],
        [[1., 0., 0., 0.], [0.707107, 0.707107, 0., 0.],
         [0.888889, -0.444444, 0.111111, 0.],
         [-0.833333, 0.5, 0.166667, 0.166667]],
    ])

    b = tfb.CorrelationCholesky()

    y_ = self.evaluate(b.forward(x))
    self.assertAllClose(y, y_, atol=1e-5, rtol=1e-5)

    x_ = self.evaluate(b.inverse(y))
    self.assertAllClose(x, x_, atol=1e-5, rtol=1e-5)

    expected_fldj = -0.5 * np.sum(
        [2, 3, 4] * np.log([[2, 9, 100], [2, 81, 36]]), axis=-1)

    fldj = self.evaluate(b.forward_log_det_jacobian(x, event_ndims=1))
    self.assertAllClose(expected_fldj, fldj)

    ildj = self.evaluate(b.inverse_log_det_jacobian(y, event_ndims=2))
    self.assertAllClose(-expected_fldj, ildj)

  def testShape(self):
    x_shape = tf.TensorShape([5, 4, 6])
    y_shape = tf.TensorShape([5, 4, 4, 4])

    b = tfb.CorrelationCholesky(validate_args=True)

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
    with self.assertRaisesRegexp(ValueError, "is not a triangular number"):
      b.forward_event_shape(x_shape_bad)
    with self.assertRaisesOpError("is not a triangular number"):
      self.evaluate(
          b.forward_event_shape_tensor(tensorshape_util.as_list(x_shape_bad)))

    y_shape_bad = tf.TensorShape([5, 4, 4, 3])
    with self.assertRaisesRegexp(ValueError, "Matrix must be square"):
      b.inverse_event_shape(y_shape_bad)
    with self.assertRaisesOpError("Matrix must be square"):
      self.evaluate(
          b.inverse_event_shape_tensor(tensorshape_util.as_list(y_shape_bad)))

  def testTheoreticalFldj(self):
    bijector = tfb.CorrelationCholesky()
    x = np.linspace(-50, 50, num=30).reshape(5, 6).astype(np.float64)
    y = self.evaluate(bijector.forward(x))
    bijector_test_util.assert_bijective_and_finite(
        bijector,
        x,
        y,
        eval_func=self.evaluate,
        event_ndims=1,
        inverse_event_ndims=2,
        rtol=1e-5)
    fldj = bijector.forward_log_det_jacobian(x, event_ndims=1)
    fldj_theoretical = bijector_test_util.get_fldj_theoretical(
        bijector,
        x,
        event_ndims=1,
        inverse_event_ndims=2,
        output_to_unconstrained=tfb.Invert(tfb.FillTriangular()))
    self.assertAllClose(
        self.evaluate(fldj_theoretical),
        self.evaluate(fldj),
        atol=1e-5,
        rtol=1e-5)

  def testBijectorWithVariables(self):
    x_ = np.array([1.], dtype=np.float32)
    y_ = np.array([[1., 0.], [0.707107, 0.707107]], dtype=np.float32)

    x = tf.Variable(x_, dtype=tf.float32)
    y = tf.Variable(y_, dtype=tf.float32)
    forward_event_ndims = tf.Variable(1, dtype=tf.int32)
    inverse_event_ndims = tf.Variable(2, dtype=tf.int32)
    self.evaluate(tf1.global_variables_initializer())

    bijector = tfb.CorrelationCholesky()
    self.assertAllClose(
        y_, self.evaluate(bijector.forward(x)), atol=1e-5, rtol=1e-5)
    self.assertAllClose(
        x_, self.evaluate(bijector.inverse(y)), atol=1e-5, rtol=1e-5)

    fldj = bijector.forward_log_det_jacobian(x, event_ndims=forward_event_ndims)
    self.assertAllClose(-np.log(2), self.evaluate(fldj))

    ildj = bijector.inverse_log_det_jacobian(y, event_ndims=inverse_event_ndims)
    self.assertAllClose(np.log(2), ildj)

  @parameterized.parameters(itertools.product([2, 3, 4, 5, 6, 7], [1., 2., 3.]))
  def testWithLKJSamples(self, dimension, concentration):
    bijector = tfb.CorrelationCholesky()
    lkj_dist = lkj.LKJ(
        dimension=dimension,
        concentration=np.float64(concentration),
        input_output_cholesky=True)
    batch_size = 10
    y = self.evaluate(lkj_dist.sample([batch_size]))
    x = self.evaluate(bijector.inverse(y))

    bijector_test_util.assert_bijective_and_finite(
        bijector,
        x,
        y,
        eval_func=self.evaluate,
        event_ndims=1,
        inverse_event_ndims=2,
        rtol=1e-5)

    fldj = bijector.forward_log_det_jacobian(x, event_ndims=1)
    fldj_theoretical = bijector_test_util.get_fldj_theoretical(
        bijector,
        x,
        event_ndims=1,
        inverse_event_ndims=2,
        output_to_unconstrained=tfb.Invert(tfb.FillTriangular()))
    self.assertAllClose(
        self.evaluate(fldj_theoretical),
        self.evaluate(fldj),
        atol=1e-5,
        rtol=1e-5)


if __name__ == "__main__":
  tf.test.main()
