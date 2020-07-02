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
"""Tests for Custom Gradient."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
import numpy as np

import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp


from tensorflow_probability.python.internal import test_util


@test_util.test_all_tf_execution_regimes
class CustomGradientTest(test_util.TestCase):

  @test_util.numpy_disable_gradient_test
  def test_works_correctly(self):
    f = lambda x: x**2 / 2
    g = lambda x: (x - 1)**3 / 3
    x = np.concatenate([np.linspace(-100, 100, int(1e3)), [0.]], axis=0)
    fx, gx = self.evaluate(tfp.math.value_and_gradient(
        lambda x_: tfp.math.custom_gradient(f(x_), g(x_), x_), x))
    self.assertAllClose(f(x), fx)
    self.assertAllClose(g(x), gx)

  @test_util.numpy_disable_gradient_test
  def test_works_correctly_both_f_g_zero(self):
    f = lambda x: x**2 / 2
    g = lambda x: x**3 / 3
    x = np.concatenate([np.linspace(-100, 100, int(1e3)), [0.]], axis=0)
    fx, gx = self.evaluate(tfp.math.value_and_gradient(
        lambda x_: tfp.math.custom_gradient(f(x_), g(x_), x_), x))
    self.assertAllClose(f(x), fx)
    self.assertAllClose(g(x), gx)

  @test_util.numpy_disable_gradient_test
  @test_util.jax_disable_variable_test
  def test_works_correctly_vector_of_vars(self):
    x = tf.Variable(2, name='x', dtype=tf.float32)
    y = tf.Variable(3, name='y', dtype=tf.float32)
    self.evaluate([x.initializer, y.initializer])

    f = lambda z: z[0] * z[1]
    g = lambda z: z[0]**2 * z[1]**2 / 2

    with tf.GradientTape() as tape:
      z = tf.stack([x, y])
      fz = tfp.math.custom_gradient(f(z), g(z), z)
    gz = tape.gradient(fz, [x, y])
    [z_, fz_, gx_, gy_] = self.evaluate([z, fz, gz[0], gz[1]])

    self.assertEqual(f(z_), fz_)
    self.assertEqual(g(z_), gx_)
    self.assertEqual(g(z_), gy_)

  @test_util.numpy_disable_gradient_test
  @test_util.jax_disable_variable_test
  def test_works_correctly_side_vars(self):
    x_ = np.float32(2.1)  # Adding extra tenth to force imprecision.
    y_ = np.float32(3.1)
    x = tf.Variable(x_, name='x')
    y = tf.Variable(y_, name='y')
    self.evaluate([x.initializer, y.initializer])

    f = lambda x: x * y
    g = lambda z: tf.square(x) * y

    with tf.GradientTape() as tape:
      fx = tfp.math.custom_gradient(f(x), g(x), x)
    gx = tape.gradient(fx, [x, y])
    [x_, fx_, gx_] = self.evaluate([x, fx, gx[0]])
    gy_ = gx[1]

    self.assertEqual(x_ * y_, fx_)
    self.assertEqual(np.square(x_) * y_, gx_)
    self.assertIsNone(gy_)

  @test_util.numpy_disable_gradient_test
  @test_util.jax_disable_variable_test
  def test_works_correctly_fx_gx_manually_stopped(self):
    x_ = np.float32(2.1)  # Adding extra tenth to force imprecision.
    y_ = np.float32(3.1)
    x = tf.Variable(x_, name='x')
    y = tf.Variable(y_, name='y')
    self.evaluate([x.initializer, y.initializer])

    stop = tf.stop_gradient  # For readability.

    # Basically we need to stop the `x` portion of `f`. And when we supply the
    # arg to `custom_gradient` we need to stop the complement, i.e., the `y`
    # part.
    f = lambda x: stop(x) * y
    g = lambda x: stop(tf.square(x)) * y
    with tf.GradientTape() as tape:
      fx = tfp.math.custom_gradient(f(x), g(x), x + stop(y),
                                    fx_gx_manually_stopped=True)

    gx = tape.gradient(fx, [x, y])
    [x_, fx_, gx_, gy_] = self.evaluate([x, fx, gx[0], gx[1]])

    self.assertEqual(x_ * y_, fx_)
    self.assertEqual(np.square(x_) * y_, gx_)
    self.assertEqual(x_, gy_)


if __name__ == '__main__':
  tf.test.main()
