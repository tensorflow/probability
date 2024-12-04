# Copyright 2022 The TensorFlow Probability Authors.
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
"""Tests for UnitVector Bijector."""

import numpy as np
import tensorflow.compat.v1 as tf1
import tensorflow.compat.v2 as tf
from tensorflow_probability.python.bijectors import bijector_test_util
from tensorflow_probability.python.bijectors import unit_vector
from tensorflow_probability.python.internal import tensorshape_util
from tensorflow_probability.python.internal import test_util

rng = np.random.RandomState(42)


@test_util.test_all_tf_execution_regimes
class UnitVectorBijectorTest(test_util.TestCase):
  """Tests correctness of the Y = g(X) = X / norm(X) transformation."""

  def testBijectorVector(self):
    uv = unit_vector.UnitVector()
    self.assertStartsWith(uv.name, "unit_vector")
    x = np.array([[2., 1., 2.], [0., 3., 0.]])
    y = np.array([[0.4, 0.2, 0.4, 0.8], [0, 0.6, 0, 0.8]])
    self.assertAllClose(y, self.evaluate(uv.forward(x)))
    self.assertAllClose(x, self.evaluate(uv.inverse(y)))
    self.assertAllClose(
        3 * np.log(2) - 3 * np.log([10, 10]),
        self.evaluate(uv.forward_log_det_jacobian(x, event_ndims=1)),
        atol=0.0,
        rtol=1e-7,
    )
    self.assertAllClose(
        self.evaluate(-uv.inverse_log_det_jacobian(y, event_ndims=1)),
        self.evaluate(uv.forward_log_det_jacobian(x, event_ndims=1)),
        atol=0.0,
        rtol=1e-7,
    )

  def testBijectorUnknownShape(self):
    uv = unit_vector.UnitVector()
    x_ = np.array([[2., 1., 2.], [0., 3., 0.]], dtype=np.float32)
    y_ = np.array(
        [[0.4, 0.2, 0.4, 0.8], [0, 0.6, 0, 0.8]],
        dtype=np.float32,
    )
    x = tf1.placeholder_with_default(x_, shape=[2, None])
    y = tf1.placeholder_with_default(y_, shape=[2, None])
    self.assertAllClose(y, self.evaluate(uv.forward(x)))
    self.assertAllClose(x, self.evaluate(uv.inverse(y)))
    self.assertAllClose(
        3 * np.log(2) - 3 * np.log([10, 10]),
        self.evaluate(uv.forward_log_det_jacobian(x, event_ndims=1)),
        atol=0.0,
        rtol=1e-6,
    )
    self.assertAllClose(
        self.evaluate(-uv.inverse_log_det_jacobian(y, event_ndims=1)),
        self.evaluate(uv.forward_log_det_jacobian(x, event_ndims=1)),
        atol=0.0,
        rtol=1e-6,
    )

  def testShapeGetters(self):
    x = tf.TensorShape([4])
    y = tf.TensorShape([5])
    uv = unit_vector.UnitVector(validate_args=True)
    self.assertAllEqual(y, uv.forward_event_shape(x))
    self.assertAllEqual(
        tensorshape_util.as_list(y),
        self.evaluate(
            uv.forward_event_shape_tensor(tensorshape_util.as_list(x))),
    )
    self.assertAllEqual(x, uv.inverse_event_shape(y))
    self.assertAllEqual(
        tensorshape_util.as_list(x),
        self.evaluate(
            uv.inverse_event_shape_tensor(tensorshape_util.as_list(y))),
    )

  def testShapeGettersWithBatchShape(self):
    x = tf.TensorShape([2, 4])
    y = tf.TensorShape([2, 5])
    uv = unit_vector.UnitVector(validate_args=True)
    self.assertAllEqual(y, uv.forward_event_shape(x))
    self.assertAllEqual(
        tensorshape_util.as_list(y),
        self.evaluate(
            uv.forward_event_shape_tensor(tensorshape_util.as_list(x))),
    )
    self.assertAllEqual(x, uv.inverse_event_shape(y))
    self.assertAllEqual(
        tensorshape_util.as_list(x),
        self.evaluate(
            uv.inverse_event_shape_tensor(tensorshape_util.as_list(y))),
    )

  def testShapeGettersWithDynamicShape(self):
    x = tf1.placeholder_with_default([2, 4], shape=None)
    y = tf1.placeholder_with_default([2, 5], shape=None)
    uv = unit_vector.UnitVector(validate_args=True)
    self.assertAllEqual([2, 5], self.evaluate(uv.forward_event_shape_tensor(x)))
    self.assertAllEqual([2, 4], self.evaluate(uv.inverse_event_shape_tensor(y)))

  def testBijectiveAndFinite(self):
    uv = unit_vector.UnitVector()
    x = np.linspace(-50, 50, num=10).reshape(5, 2).astype(np.float32)
    # Make y values on the simplex with a wide range.
    y_0 = np.ones(5).astype(np.float32)
    y_1 = (1e-5 * rng.rand(5)).astype(np.float32)
    y_2 = (1e1 * rng.rand(5)).astype(np.float32)
    y = np.array([y_0, y_1, y_2])
    y /= y.sum(axis=0)
    y = y.T  # y.shape = [5, 3]
    bijector_test_util.assert_bijective_and_finite(
        uv, x, y, eval_func=self.evaluate, event_ndims=1)

  def testAssertsValidArgToInverse(self):
    uv = unit_vector.UnitVector(validate_args=True)
    with self.assertRaisesOpError("must be a unit vector"):
      self.evaluate(uv.inverse([0.5, 0.2, 0.1]))

    with self.assertRaisesOpError("last axis of `y` cannot be 1."):
      self.evaluate(uv.inverse([0., 0., 0., 1.0]))

    with self.assertRaisesOpError(
        "must be a unit vector|last axis of `y` cannot be 1."):
      self.evaluate(uv.inverse([0.4, 0.5, 0.3, 1.0]))

  @test_util.numpy_disable_gradient_test
  def testTheoreticalFldj(self):
    uv = unit_vector.UnitVector()
    x = np.linspace(-15, 15, num=10).reshape(5, 2).astype(np.float64)

    fldj = uv.forward_log_det_jacobian(x, event_ndims=1)
    fldj_theoretical = bijector_test_util.get_fldj_theoretical(
        uv, x, event_ndims=1)
    self.assertAllClose(
        self.evaluate(fldj_theoretical),
        self.evaluate(fldj),
        atol=1e-5,
        rtol=1e-5)


if __name__ == "__main__":
  test_util.main()
