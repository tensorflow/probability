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
"""Tests for special."""

import hypothesis as hp
from hypothesis import strategies as hps
import numpy as np
import tensorflow.compat.v2 as tf
from tensorflow_probability.python.internal import hypothesis_testlib as tfp_hps
from tensorflow_probability.python.internal import test_util
from tensorflow_probability.python.math import integration


@test_util.test_all_tf_execution_regimes
class TrapzTest(test_util.TestCase):
  """Test for tfp.math.trapz."""

  def test_simple_use(self):
    y = tf.linspace(0., 10., 11)
    integral = self.evaluate(integration.trapz(y))
    self.assertAllClose(integral, 50.)

  def test_simple_use_with_x_arg(self):
    y = tf.linspace(0., 10., 11)
    x = tf.linspace(0., 10., 11)
    integral = self.evaluate(integration.trapz(y, x))
    self.assertAllClose(integral, 50.)

  def test_simple_use_with_dx_arg(self):
    y = tf.linspace(0., 10., 11)
    dx = 0.1
    integral = self.evaluate(integration.trapz(y, dx=dx))
    self.assertAllClose(integral, 5.0)

  def test_provide_multiple_axes_raises(self):
    with self.assertRaisesRegex(
        ValueError, 'Only permitted to specify one axis'):
      integration.trapz(y=tf.ones((2, 3)), axis=[0, 1])

  def test_non_scalar_dx_raises(self):
    with self.assertRaisesRegex(
        ValueError, 'Expected dx to be a scalar'):
      integration.trapz(y=tf.ones((2, 3)), dx=[0.1, 0.2])

  def test_provide_x_and_dx_args_raises(self):
    with self.assertRaisesWithLiteralMatch(
        ValueError, 'Not permitted to specify both x and dx input args.'):
      integration.trapz(y=[0, 0.1], x=[0, 0.1], dx=0.5)

  def test_incompatible_x_y_shape_raises(self):
    with self.assertRaisesWithLiteralMatch(
        ValueError, 'Shapes (3,) and (2, 3) are incompatible'):
      integration.trapz(y=tf.ones((2, 3)), x=tf.ones((3,)))

  def test_multidim(self):
    y = tf.ones((4, 5, 6), dtype=tf.float32)
    integral_0 = self.evaluate(integration.trapz(y, axis=0))
    integral_1 = self.evaluate(integration.trapz(y, axis=1))
    integral_2 = self.evaluate(integration.trapz(y))
    self.assertTupleEqual(integral_0.shape, (5, 6))
    self.assertTupleEqual(integral_1.shape, (4, 6))
    self.assertTupleEqual(integral_2.shape, (4, 5))
    self.assertAllClose(integral_0, np.ones((5, 6)) * 3.0)
    self.assertAllClose(integral_1, np.ones((4, 6)) * 4.0)
    self.assertAllClose(integral_2, np.ones((4, 5)) * 5.0)

  def test_multidim_with_x(self):
    y = tf.ones((4, 5), dtype=tf.float64)
    v0 = tf.cast(tf.linspace(0., 1., 4), tf.float64)
    v1 = tf.cast(tf.linspace(0., 4.2, 5), tf.float64)
    x0, x1 = tf.meshgrid(v0, v1, indexing='ij')
    integral_0 = self.evaluate(integration.trapz(y, x0, axis=0))
    integral_1 = self.evaluate(integration.trapz(y, x1, axis=1))
    self.assertTupleEqual(integral_0.shape, (5,))
    self.assertTupleEqual(integral_1.shape, (4,))
    self.assertAllClose(integral_0, np.ones((5,)) * 1.0)
    self.assertAllClose(integral_1, np.ones((4,)) * 4.2)

  def test_multidim_with_nonhomogeneous_x(self):
    # integration domain from x is different for each batch dim
    y = tf.ones((3, 4), dtype=tf.float32)
    x = tf.constant([[0, 1, 2, 3], [0, 2, 4, 6], [0, 3, 6, 9]],
                    dtype=tf.float32)
    integral = self.evaluate(integration.trapz(y, x, axis=-1))
    self.assertTupleEqual(integral.shape, (3,))
    np.testing.assert_almost_equal(integral, [3, 6, 9])

  def test_multidim_broadcast_1d_x(self):
    # To use trapz() with a 1d x array, first broadcast it with the shape of y
    y = tf.ones((4, 5), dtype=tf.float64)
    x1 = tf.cast(tf.linspace(0., 4.2, 5), tf.float64)
    x = x1 * tf.ones((4, 5), dtype=tf.float64)
    integral = self.evaluate(integration.trapz(y, x, axis=1))
    self.assertTupleEqual(integral.shape, (4,))
    self.assertAllClose(integral, np.ones((4,)) * 4.2)

  def test_convergence(self):
    # Each 10x decrease in dx gives a 100x increase in integral accuracy.
    # This demonstrates that trapz() has quadratic converence.
    def trapz_sin_fn(x_min, x_max, n, expected, rtol):
      pi = tf.constant(np.pi, dtype=tf.float32)
      x = tf.cast(tf.linspace(x_min, x_max, n), tf.float32)
      y = tf.sin(pi * x)
      np.testing.assert_allclose(
          self.evaluate(integration.trapz(y, x)), expected, rtol=rtol)

    with self.subTest('sin(2pi x) over [0, 1]'):
      # integral[sin(pi*x)] for x over [0, 1] is equal to 2 / pi.
      trapz_sin_fn(0, 1, 20, 2 / np.pi, rtol=2.3e-3)
      trapz_sin_fn(0, 1, 200, 2 / np.pi, rtol=2.1e-5)
      trapz_sin_fn(0, 1, 2000, 2 / np.pi, rtol=4.2e-7)

    with self.subTest('sin(2pi x) over [1, 2]'):
      # integral[sin(pi*x)] for x over [1, 2] is equal to -2 / pi.
      trapz_sin_fn(1, 2, 20, -2 / np.pi, rtol=2.3e-3)
      trapz_sin_fn(1, 2, 200, -2 / np.pi, rtol=2.1e-5)
      trapz_sin_fn(1, 2, 2000, -2 / np.pi, rtol=4.2e-7)

  def test_convergence_nonuniform_x(self):
    # Each 10x decrease in dx gives a 100x increase in integral accuracy.
    # This demonstrates that trapz() has quadratic converence.
    def trapz_sin_fn(x_min, x_max, n, expected, rtol):
      pi = tf.constant(np.pi, dtype=tf.float32)
      s = np.linspace(0, 1, n)**1.5
      x = tf.convert_to_tensor(x_min + s * (x_max - x_min), tf.float32)
      y = tf.sin(pi * x)
      np.testing.assert_allclose(
          self.evaluate(integration.trapz(y, x)), expected, rtol=rtol)

    with self.subTest('sin(2pi x) over [0, 1]'):
      # integral[sin(pi*x)] for x over [0, 1] is equal to 2 / pi.
      trapz_sin_fn(0, 1, 20, 2 / np.pi, rtol=3.2e-3)
      trapz_sin_fn(0, 1, 200, 2 / np.pi, rtol=2.9e-5)
      trapz_sin_fn(0, 1, 2000, 2 / np.pi, rtol=4.2e-7)

    with self.subTest('sin(2pi x) over [1, 2]'):
      # integral[sin(pi*x)] for x over [1, 2] is equal to -2 / pi.
      trapz_sin_fn(1, 2, 20, -2 / np.pi, rtol=3.2e-3)
      trapz_sin_fn(1, 2, 200, -2 / np.pi, rtol=2.9e-5)
      trapz_sin_fn(1, 2, 2000, -2 / np.pi, rtol=4.2e-7)

  @hp.given(hps.data())
  @tfp_hps.tfp_hp_settings(default_max_examples=5)
  def test_against_numpy(self, data):
    dtype = data.draw(hps.sampled_from([np.float32, np.float64]))
    shp = (data.draw(hps.integers(5, 10)), data.draw(hps.integers(5, 10)))
    axis = data.draw(hps.integers(0, len(shp) - 1))
    y = data.draw(tfp_hps.constrained_tensors(tfp_hps.identity_fn, shp, dtype))
    x_dx = data.draw(hps.sampled_from(['x', 'dx', None]))
    if x_dx is None:
      x = None
      dx = None
    elif x_dx == 'dx':
      x = None
      dx = data.draw(hps.floats(0.1, 10))
    else:
      x = data.draw(
          tfp_hps.constrained_tensors(tfp_hps.identity_fn, shp, dtype))
      dx = None
    np_soln = np.trapezoid(
        self.evaluate(y),
        x=self.evaluate(x) if x is not None else None,  # cannot evaluate(None)
        dx=dx or 1.0,  # numpy default is 1.0
        axis=axis)
    tf_soln = integration.trapz(y, x, dx, axis)
    self.assertAllClose(np_soln, tf_soln)

  def test_jit(self):
    self.skip_if_no_xla()
    def f(y, x=None, dx=None, axis=-1, name=None):
      @tf.function(jit_compile=True)
      def g(y, x=None, dx=None):
        return integration.trapz(y, x=x, dx=dx, axis=axis, name=name)
      return g(y, x=x, dx=dx)
    y = tf.ones((3, 4, 5), dtype=tf.float32, name='x')
    x = tf.ones((3, 4, 5), dtype=tf.float32, name='y')
    dx = tf.constant(0.2, dtype=tf.float32, name='dx')
    axis = tf.constant(1, dtype=tf.int32)
    with self.subTest('f(y)'):
      self.evaluate(f(y))
    with self.subTest('f(y, x)'):
      self.evaluate(f(y, x=x, axis=axis))
    with self.subTest('f(y, dx)'):
      self.evaluate(f(y, dx=dx, axis=axis))


if __name__ == '__main__':
  test_util.main()
