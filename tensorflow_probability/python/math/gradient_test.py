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

# Dependency imports
import numpy as np

import tensorflow.compat.v2 as tf

from tensorflow_probability.python.distributions import kullback_leibler
from tensorflow_probability.python.distributions import normal
from tensorflow_probability.python.internal import test_util
from tensorflow_probability.python.math import gradient
from tensorflow_probability.python.math.gradient import batch_jacobian
from tensorflow_probability.python.math.gradient import value_and_gradient_with_auto_expansion


@test_util.test_all_tf_execution_regimes
class GradientTest(test_util.TestCase):

  @test_util.numpy_disable_gradient_test
  def test_non_list(self):
    f = lambda x: x**2 / 2
    g = lambda x: x
    x = np.concatenate([np.linspace(-100, 100, int(1e1)), [0]], axis=0)
    y, dydx = self.evaluate(gradient.value_and_gradient(f, x))
    self.assertAllClose(f(x), y, atol=1e-6, rtol=1e-6)
    self.assertAllClose(g(x), dydx, atol=1e-6, rtol=1e-6)

  @test_util.numpy_disable_gradient_test
  def test_list(self):
    f = lambda x, y: x * y
    g = lambda x, y: [y, x]
    args = [np.linspace(0, 100, int(1e1)),
            np.linspace(-100, 0, int(1e1))]
    y, dydx = self.evaluate(gradient.value_and_gradient(f, args))
    self.assertAllClose(f(*args), y, atol=1e-6, rtol=1e-6)
    self.assertAllClose(g(*args), dydx, atol=1e-6, rtol=1e-6)

  @test_util.numpy_disable_gradient_test
  def test_nested(self):
    f = lambda value: value['x'] * value['y']
    g = lambda value: {'x': value['y'], 'y': value['x']}
    args = {'x': np.linspace(0, 100, int(1e1)),
            'y': np.linspace(-100, 0, int(1e1))}
    y, dydx = self.evaluate(gradient.value_and_gradient(f, args))
    self.assertAllClose(f(args), y, atol=1e-6, rtol=1e-6)
    self.assertAllCloseNested(g(args), dydx, atol=1e-6, rtol=1e-6)

  @test_util.numpy_disable_gradient_test
  def test_output_list(self):
    f = lambda x, y: [x, x * y]
    g = lambda x, y: [y + 1., x]
    args = [np.linspace(0, 100, int(1e1)),
            np.linspace(-100, 0, int(1e1))]
    y, dydx = self.evaluate(gradient.value_and_gradient(f, args))
    self.assertAllClose(f(*args), y, atol=1e-6, rtol=1e-6)
    self.assertAllClose(g(*args), dydx, atol=1e-6, rtol=1e-6)

  @test_util.numpy_disable_gradient_test
  @test_util.jax_disable_test_missing_functionality('value_and_gradient')
  def test_multi_input_old_style(self):
    arg0 = [2., 3., 4.]
    arg1 = [5., 6., 7.]
    f_actual = lambda x, y: x * np.log(y)
    g_actual = lambda x, y: (np.log(y), x / np.array(y))
    y, dydx = self.evaluate(
        gradient.value_and_gradient(lambda x, y: x * tf.math.log(y),
                                    [arg0, arg1]))
    self.assertAllClose(f_actual(arg0, arg1), y, atol=1e-6, rtol=1e-6)
    self.assertAllClose(g_actual(arg0, arg1), dydx, atol=1e-6, rtol=1e-6)

  @test_util.numpy_disable_gradient_test
  @test_util.jax_disable_test_missing_functionality('value_and_gradient')
  def test_multi_input_no_auto_unpack(self):
    arg0 = [2., 3., 4.]
    arg1 = [5., 6., 7.]
    f_actual = lambda x, y: x * np.log(y)
    g_actual = lambda x, y: (np.log(y), x / np.array(y))

    # This is how users would typically write things.
    y, dydx = self.evaluate(
        gradient.value_and_gradient(lambda x, y: x * tf.math.log(y), arg0,
                                    arg1))
    self.assertAllClose(f_actual(arg0, arg1), y, atol=1e-6, rtol=1e-6)
    self.assertAllClose(g_actual(arg0, arg1), dydx, atol=1e-6, rtol=1e-6)

    # This is uncommon but possible and unambigous under new style.
    y, dydx = self.evaluate(
        gradient.value_and_gradient(
            lambda x: x[0] * tf.math.log(x[1]), [arg0, arg1],
            auto_unpack_single_arg=False))
    self.assertAllClose(f_actual(arg0, arg1), y, atol=1e-6, rtol=1e-6)
    self.assertAllClose(g_actual(arg0, arg1), dydx, atol=1e-6, rtol=1e-6)

  @test_util.numpy_disable_gradient_test
  @test_util.jax_disable_test_missing_functionality('value_and_gradient')
  def test_simple_input_no_auto_unpack(self):
    x = [1., 2., 3.]
    y, dydx = self.evaluate(
        gradient.value_and_gradient(
            tf.math.log, x, auto_unpack_single_arg=False))
    self.assertAllClose(np.log(x), y, atol=1e-6, rtol=1e-6)
    self.assertAllClose(1. / np.array(x), dydx, atol=1e-6, rtol=1e-6)

  @test_util.numpy_disable_gradient_test
  @test_util.jax_disable_test_missing_functionality('value_and_gradient')
  def test_variable_and_constant_identical(self):
    expected = (2. * np.log(2.), [1. + np.log(2.), 1. + np.log(2.)])
    x = tf.constant(2.)
    self.assertAllClose(
        expected,
        self.evaluate(
            gradient.value_and_gradient(lambda a, b: a * tf.math.log(x), x, x)),
        atol=1e-6,
        rtol=1e-6)
    x = tf.Variable(2.)
    self.evaluate(x.initializer)
    self.assertAllClose(
        expected,
        self.evaluate(
            gradient.value_and_gradient(lambda a, b: a * tf.math.log(x), x, x)),
        atol=1e-6,
        rtol=1e-6)

  @test_util.numpy_disable_gradient_test
  @test_util.jax_disable_test_missing_functionality('value_and_gradient')
  def test_docstring_examples(self):
    # Case 1: argless `f`.
    x = tf.constant(2.)
    self.assertAllClose([np.log(2.), 0.5],
                        self.evaluate(
                            gradient.value_and_gradient(lambda: tf.math.log(x),
                                                        x)),
                        atol=1e-6,
                        rtol=1e-6)

    # Case 2: packed arguments.
    self.assertAllClose([2. * np.log(3.), (np.log(3.), 2. / 3)],
                        self.evaluate(
                            gradient.value_and_gradient(
                                lambda x, y: x * tf.math.log(y), [2., 3.])),
                        atol=1e-6,
                        rtol=1e-6)

    # Case 3: default.
    x = np.array([1., 2, 3])
    self.assertAllClose((np.log(x), 1. / x),
                        self.evaluate(
                            gradient.value_and_gradient(
                                tf.math.log, [1., 2., 3.],
                                auto_unpack_single_arg=False)),
                        atol=1e-6,
                        rtol=1e-6)

  @test_util.numpy_disable_gradient_test
  @test_util.jax_disable_test_missing_functionality('value_and_gradient')
  def test_variable_tracking(self):
    value_and_gradient = value_and_gradient_with_auto_expansion
    q = normal.Normal(tf.Variable(1.), tf.Variable(1., trainable=False))
    r = normal.Normal(0., tf.Variable(1.))
    self.evaluate([v.initializer for v in q.variables + r.variables])

    y, dydx = self.evaluate(
        value_and_gradient(kullback_leibler.kl_divergence, q, r))
    self.assertAllClose(0.5, y, atol=1e-6, rtol=1e-6)
    self.assertAllClose([[1.], [-1.]], dydx, atol=1e-6, rtol=1e-6)

    y, dydx = self.evaluate(
        value_and_gradient(lambda: kullback_leibler.kl_divergence(q, r), q, r))
    self.assertAllClose(0.5, y, atol=1e-6, rtol=1e-6)
    self.assertAllClose([[1.], [-1.]], dydx, atol=1e-6, rtol=1e-6)

    y, dydx = self.evaluate(
        value_and_gradient(lambda: kullback_leibler.kl_divergence(q, r),
                           *q.trainable_variables, r))
    self.assertAllClose(0.5, y, atol=1e-6, rtol=1e-6)
    self.assertAllClose([1., [-1.]], dydx, atol=1e-6, rtol=1e-6)

    y, dydx = self.evaluate(
        value_and_gradient(lambda *_: kullback_leibler.kl_divergence(q, r), q,
                           r))
    self.assertAllClose(0.5, y, atol=1e-6, rtol=1e-6)
    self.assertAllClose([[1.], [-1.]], dydx, atol=1e-6, rtol=1e-6)

    y, dydx = self.evaluate(
        value_and_gradient(
            lambda **kw: kullback_leibler.kl_divergence(  # pylint: disable=g-long-lambda
                normal.Normal(kw['loc_q'], 1), normal.Normal(0, kw['scale_r'])),
            loc_q=1.,
            scale_r=1.))
    self.assertAllClose(0.5, y, atol=1e-6, rtol=1e-6)
    self.assertAllClose({'loc_q': 1., 'scale_r': -1.}, dydx,
                        atol=1e-6, rtol=1e-6)

  @test_util.numpy_disable_gradient_test
  def test_output_gradients(self):
    jacobian = np.float32([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]])
    f = lambda x: tf.squeeze(tf.matmul(jacobian, x[:, tf.newaxis]))
    x = np.ones([3], dtype=np.float32)
    output_gradients = np.float32([1., 2., 3.])
    y, dydx = self.evaluate(
        gradient.value_and_gradient(f, x, output_gradients=output_gradients))
    self.assertAllClose(f(x), y, atol=1e-6, rtol=1e-6)
    self.assertAllClose(
        np.dot(output_gradients, jacobian), dydx, atol=1e-6, rtol=1e-6)

  @test_util.numpy_disable_gradient_test
  def test_batch_jacobian(self):
    w = tf.reshape(tf.range(12.) + 1., (4, 3))
    def f(x):
      return tf.math.cumsum(w * x, axis=-1)

    self.assertAllEqual(
        tf.convert_to_tensor([
            [[1., 0., 0.], [1., 2., 0.], [1., 2., 3.]],
            [[4., 0., 0.], [4., 5., 0.], [4., 5., 6.]],
            [[7., 0., 0.], [7., 8., 0.], [7., 8., 9.]],
            [[10., 0., 0.], [10., 11., 0.], [10., 11., 12.]]]),
        batch_jacobian(f, tf.ones((4, 3))))

  @test_util.numpy_disable_gradient_test
  def test_batch_jacobian_larger_rank_and_dtype(self):
    w1 = tf.reshape(tf.range(24., dtype=tf.float64) + 1., (4, 2, 3))
    w2 = tf.reshape(tf.range(24., dtype=tf.float32) * 0.5 - 6., (4, 2, 1, 3))
    def f(x, y):  # [4, 2, 3], [4, 2, 1, 3] -> [4, 3, 2]
      return tf.transpose(
          tf.cast(tf.math.cumsum(w1 * x, axis=-1), dtype=tf.float32)
          + tf.square(tf.reverse(w2 * y, axis=[-3]))[..., 0, :],
          perm=[0, 2, 1])

    x = tf.cast(np.random.uniform(size=(4, 2, 3)), dtype=tf.float64)
    y = tf.cast(np.random.uniform(size=(4, 2, 1, 3)), dtype=tf.float32)
    jac = batch_jacobian(f, [x, y])

    # Check shapes.
    self.assertLen(jac, 2)
    self.assertAllEqual([4, 3, 2, 2, 3], jac[0].shape)
    self.assertAllEqual([4, 3, 2, 2, 1, 3], jac[1].shape)
    self.assertEqual(tf.float64, jac[0].dtype)
    self.assertEqual(tf.float32, jac[1].dtype)

    # Check results against `value_and_gradient`.
    out_shape = f(x, y).shape[1:]
    for i in range(np.prod(out_shape)):
      idx = (slice(None),) + np.unravel_index(i, out_shape)
      # pylint: disable=cell-var-from-loop
      _, grad = gradient.value_and_gradient(lambda x, y: f(x, y)[idx], [x, y])
      print(grad[0].shape, jac[0].shape, jac[0][idx].shape)
      self.assertAllClose(grad[0], jac[0][idx])
      self.assertAllClose(grad[1], jac[1][idx])

  @test_util.numpy_disable_gradient_test
  def test_aux(self):
    x = tf.constant([[2.]])

    def f(x):
      return x**2, x

    (y, aux), dx = gradient.value_and_gradient(f, x, has_aux=True)

    self.assertAllClose(x**2, y)
    self.assertAllClose(2 * x, dx)
    self.assertAllClose(x, aux)

    dx, aux = batch_jacobian(f, x, has_aux=True)

    self.assertAllClose((2 * x)[..., tf.newaxis], dx)
    self.assertAllClose(x, aux)

  @test_util.numpy_disable_gradient_test
  def test_aux_multi_arg(self):
    x = tf.constant([[2.]])
    z = tf.constant([[3.]])

    def f(x, z):
      return x**2 + z**2, (x, z)

    (y, aux), (dx, dz) = gradient.value_and_gradient(f, (x, z), has_aux=True)

    self.assertAllClose(x**2 + z**2, y)
    self.assertAllClose(2 * x, dx)
    self.assertAllClose(2 * z, dz)
    self.assertAllClose(x, aux[0])
    self.assertAllClose(z, aux[1])

    (dx, dz), aux = batch_jacobian(f, (x, z), has_aux=True)

    self.assertAllClose((2 * x)[..., tf.newaxis], dx)
    self.assertAllClose((2 * z)[..., tf.newaxis], dz)
    self.assertAllClose(x, aux[0])
    self.assertAllClose(z, aux[1])

  @test_util.numpy_disable_gradient_test
  def test_aux_grads(self):
    """Tests that gradients flow through the auxiliary output."""
    x = tf.constant([[2.]])

    def f(x):
      return x**2, x**2

    def f2(x):
      (_, aux), _ = gradient.value_and_gradient(f, x, has_aux=True)
      return aux

    y, dx = gradient.value_and_gradient(f2, x)
    self.assertAllClose(x**2, y)
    self.assertAllClose(2 * x, dx)


if __name__ == '__main__':
  test_util.main()
