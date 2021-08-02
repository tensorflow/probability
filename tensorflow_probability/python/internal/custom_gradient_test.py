# Copyright 2021 The TensorFlow Probability Authors.
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
"""Tests for custom_gradient."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v2 as tf

from tensorflow_probability.python.internal import custom_gradient
from tensorflow_probability.python.internal import test_util
from tensorflow_probability.python.math import gradient as tfp_gradient

JAX_MODE = False


@test_util.numpy_disable_gradient_test
@test_util.test_all_tf_execution_regimes
class CustomGradientTest(test_util.TestCase):

  def testVJP(self):

    def f_vjp_fwd(x, y):
      return x**2 + y**2, (x, y)

    def f_vjp_bwd(x_y, dz):
      x, y = x_y
      return 7. * dz * x, 7. * dz * y

    @custom_gradient.custom_gradient(
        vjp_fwd=f_vjp_fwd,
        vjp_bwd=f_vjp_bwd,
    )
    def f(x, y):
      return f_vjp_fwd(x, y)[0]

    x = tf.constant(2.)
    y = tf.constant(3.)
    dz = tf.constant(5.)

    z1 = f(x, y)
    z2, (dx, dy) = tfp_gradient.value_and_gradient(
        f, (x, y), output_gradients=dz)

    self.assertAllClose(x**2 + y**2, z1)
    self.assertAllClose(x**2 + y**2, z2)
    self.assertAllClose(7. * dz * x, dx)
    self.assertAllClose(7. * dz * y, dy)

  @test_util.jax_disable_variable_test
  def testVJPWithVariables(self):

    def f_vjp_fwd(x):
      return x**2 + y**2, x

    def f_vjp_bwd(x, dz, variables):
      y = variables[0]
      return 7. * dz * x, [7. * dz * y]

    @custom_gradient.custom_gradient(
        vjp_fwd=f_vjp_fwd,
        vjp_bwd=f_vjp_bwd,
    )
    def f(x):
      return f_vjp_fwd(x)[0]

    x = tf.constant(2.)
    y = tf.Variable(3.)
    dz = tf.constant(5.)

    self.evaluate(y.initializer)

    z1 = f(x)

    # Use GradientTape to implicitly capture the variable.
    with tf.GradientTape() as tape:
      tape.watch(x)
      z2 = f(x)

    dx, dy = tape.gradient(z2, (x, y), output_gradients=dz)

    self.assertAllClose(x**2 + y**2, z1)
    self.assertAllClose(x**2 + y**2, z2)
    self.assertAllClose(7. * dz * x, dx)
    self.assertAllClose(7. * dz * y, dy)

  def testJVP(self):
    if not JAX_MODE:
      self.skipTest('Custom JVPs are JAX-only.')

    def f_vjp_fwd(x, y):
      # When a JVP is specified, this function is ignored.
      raise NotImplementedError()

    def f_vjp_bwd(x_y, dz):
      # When a JVP is specified, this function is ignored.
      raise NotImplementedError()

    def f_jvp(x_y, dx_dy):
      x, y = x_y
      dx, dy = dx_dy
      return f(x, y), 7. * (dx * x + dy * y)

    @custom_gradient.custom_gradient(
        vjp_fwd=f_vjp_fwd,
        vjp_bwd=f_vjp_bwd,
        jvp_fn=f_jvp,
    )
    def f(x, y):
      return x**2 + y**2

    x = tf.constant(2.)
    y = tf.constant(3.)
    dz = tf.constant(5.)

    z1 = f(x, y)
    z2, (dx, dy) = tfp_gradient.value_and_gradient(
        f, (x, y), output_gradients=dz)

    self.assertAllClose(x**2 + y**2, z1)
    self.assertAllClose(x**2 + y**2, z2)
    self.assertAllClose(7. * dz * x, dx)
    self.assertAllClose(7. * dz * y, dy)

    import jax  # pylint: disable=g-import-not-at-top

    z3, dz2 = jax.jvp(f, (x, y), (dx, dy))
    self.assertAllClose(x**2 + y**2, z3)
    self.assertAllClose(7. * (dx * x + dy * y), dz2)

  def testPreventGradient(self):

    def f(x):
      return custom_gradient.prevent_gradient(x, 'No gradient')

    _ = f(1.)

    with self.assertRaisesRegex(LookupError, 'No gradient'):
      tfp_gradient.value_and_gradient(f, (1.))


if __name__ == '__main__':
  tf.test.main()
