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
"""Tests for Gradient."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
import numpy as np

import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp


from tensorflow_probability.python.internal import test_util


@test_util.test_all_tf_execution_regimes
class GradientTest(test_util.TestCase):

  @test_util.numpy_disable_gradient_test
  def test_non_list(self):
    f = lambda x: x**2 / 2
    g = lambda x: x
    x = np.concatenate([np.linspace(-100, 100, int(1e1)), [0]], axis=0)
    y, dydx = self.evaluate(tfp.math.value_and_gradient(f, x))
    self.assertAllClose(f(x), y, atol=1e-6, rtol=1e-6)
    self.assertAllClose(g(x), dydx, atol=1e-6, rtol=1e-6)

  @test_util.numpy_disable_gradient_test
  def test_list(self):
    f = lambda x, y: x * y
    g = lambda x, y: [y, x]
    args = [np.linspace(0, 100, int(1e1)),
            np.linspace(-100, 0, int(1e1))]
    y, dydx = self.evaluate(tfp.math.value_and_gradient(f, args))
    self.assertAllClose(f(*args), y, atol=1e-6, rtol=1e-6)
    self.assertAllClose(g(*args), dydx, atol=1e-6, rtol=1e-6)

  @test_util.numpy_disable_gradient_test
  def test_output_list(self):
    f = lambda x, y: [x, x * y]
    g = lambda x, y: [y + 1., x]
    args = [np.linspace(0, 100, int(1e1)),
            np.linspace(-100, 0, int(1e1))]
    y, dydx = self.evaluate(tfp.math.value_and_gradient(f, args))
    self.assertAllClose(f(*args), y, atol=1e-6, rtol=1e-6)
    self.assertAllClose(g(*args), dydx, atol=1e-6, rtol=1e-6)

  @test_util.numpy_disable_gradient_test
  def test_output_gradients(self):
    jacobian = np.float32([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]])
    f = lambda x: tf.squeeze(tf.matmul(jacobian, x[:, tf.newaxis]))
    x = np.ones([3], dtype=np.float32)
    output_gradients = np.float32([1., 2., 3.])
    y, dydx = self.evaluate(
        tfp.math.value_and_gradient(f, x, output_gradients=output_gradients))
    self.assertAllClose(f(x), y, atol=1e-6, rtol=1e-6)
    self.assertAllClose(
        np.dot(output_gradients, jacobian), dydx, atol=1e-6, rtol=1e-6)


if __name__ == '__main__':
  tf.test.main()
