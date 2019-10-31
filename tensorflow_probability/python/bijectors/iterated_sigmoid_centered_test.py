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
"""Tests for IteratedSigmoidCenteredBijector."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports

import numpy as np
import tensorflow.compat.v1 as tf1
import tensorflow.compat.v2 as tf
from tensorflow_probability.python import bijectors as tfb
from tensorflow_probability.python.bijectors import bijector_test_util
from tensorflow_probability.python.internal import tensorshape_util
from tensorflow_probability.python.internal import test_util
from tensorflow_probability.python.math.gradient import batch_jacobian


@test_util.test_all_tf_execution_regimes
class _IteratedSigmoidCenteredBijectorTest(object):
  """Tests correctness of Stick breaking transformation."""

  def testBijectorVector(self):
    iterated_sigmoid = tfb.IteratedSigmoidCentered()
    self.assertStartsWith(iterated_sigmoid.name, "iterated_sigmoid")
    x = self.dtype([[0., 0., 0.], -np.log([1 / 3., 1 / 2., 1.])])
    y = self.dtype([[0.25, 0.25, 0.25, 0.25], [0.5, 0.25, 0.125, 0.125]])
    self.assertAllClose(y, self.evaluate(iterated_sigmoid.forward(x)))
    self.assertAllClose(x, self.evaluate(iterated_sigmoid.inverse(y)))
    self.assertAllClose(
        -np.sum(np.log(y), axis=1),
        self.evaluate(
            iterated_sigmoid.inverse_log_det_jacobian(y, event_ndims=1)),
        atol=0.,
        rtol=1e-7)
    self.assertAllClose(
        self.evaluate(
            -iterated_sigmoid.inverse_log_det_jacobian(y, event_ndims=1)),
        self.evaluate(
            iterated_sigmoid.forward_log_det_jacobian(x, event_ndims=1)),
        atol=0.,
        rtol=1e-7)

  def testBijectorUnknownShape(self):
    iterated_sigmoid = tfb.IteratedSigmoidCentered()
    self.assertStartsWith(iterated_sigmoid.name, "iterated_sigmoid")
    x_ = self.dtype([[0., 0., 0.], -np.log([1 / 3., 1 / 2., 1.])])
    y_ = self.dtype([[0.25, 0.25, 0.25, 0.25], [0.5, 0.25, 0.125, 0.125]])
    x = tf1.placeholder_with_default(x_, shape=[2, None])
    y = tf1.placeholder_with_default(y_, shape=[2, None])
    self.assertAllClose(y_, self.evaluate(iterated_sigmoid.forward(x)))
    self.assertAllClose(x_, self.evaluate(iterated_sigmoid.inverse(y)))
    self.assertAllClose(
        -np.sum(np.log(y_), axis=1),
        self.evaluate(
            iterated_sigmoid.inverse_log_det_jacobian(y, event_ndims=1)),
        atol=0.,
        rtol=1e-7)
    self.assertAllClose(
        -self.evaluate(
            iterated_sigmoid.inverse_log_det_jacobian(y, event_ndims=1)),
        self.evaluate(
            iterated_sigmoid.forward_log_det_jacobian(x, event_ndims=1)),
        atol=0.,
        rtol=1e-7)

  def testShapeGetters(self):
    x = tf.TensorShape([4])
    y = tf.TensorShape([5])
    bijector = tfb.IteratedSigmoidCentered(validate_args=True)
    self.assertAllEqual(y, bijector.forward_event_shape(x))
    self.assertAllEqual(
        tensorshape_util.as_list(y),
        self.evaluate(
            bijector.forward_event_shape_tensor(tensorshape_util.as_list(x))))
    self.assertAllEqual(x, bijector.inverse_event_shape(y))
    self.assertAllEqual(
        tensorshape_util.as_list(x),
        self.evaluate(
            bijector.inverse_event_shape_tensor(tensorshape_util.as_list(y))))

  def testBijectiveAndFinite(self):
    iterated_sigmoid = tfb.IteratedSigmoidCentered()

    # Grid of points in [-30, 30] x [-30, 30].
    x = np.mgrid[-30:30:0.5, -30:30:0.5].reshape(2, -1).T  # pylint: disable=invalid-slice-index
    # Make y values on the simplex with a wide range.
    y_0 = np.ones(x.shape[0], dtype=self.dtype)
    y_1 = self.dtype(1e-3 * np.random.rand(x.shape[0]))
    y_2 = self.dtype(1e1 * np.random.rand(x.shape[0]))
    y = np.array([y_0, y_1, y_2])
    y /= y.sum(axis=0)
    y = y.T
    bijector_test_util.assert_bijective_and_finite(
        iterated_sigmoid, x, y, eval_func=self.evaluate, event_ndims=1)

  @test_util.numpy_disable_gradient_test
  @test_util.jax_disable_test_missing_functionality(
      "https://github.com/google/jax/issues/1212")
  def testJacobianConsistent(self):
    bijector = tfb.IteratedSigmoidCentered()
    x = tf.constant((60 * np.random.rand(10) - 30).reshape(5, 2))
    jacobian_matrix = batch_jacobian(bijector.forward, x)
    # In our case, y[-1] is determined by all the other y, so we can drop it
    # for the jacobian calculation.
    jacobian_matrix = jacobian_matrix[..., :-1, :]
    self.assertAllClose(
        tf.linalg.slogdet(jacobian_matrix).log_abs_determinant,
        bijector.forward_log_det_jacobian(x, event_ndims=1),
        atol=0.,
        rtol=1e-7)


class IteratedSigmoidCenteredBijectorTestFloat32(
    test_util.TestCase,
    _IteratedSigmoidCenteredBijectorTest):
  dtype = np.float32


class IteratedSigmoidCenteredBijectorTestFloat64(
    test_util.TestCase,
    _IteratedSigmoidCenteredBijectorTest):
  dtype = np.float64


if __name__ == "__main__":
  tf.test.main()
