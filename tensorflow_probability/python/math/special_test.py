# Copyright 2020 The TensorFlow Probability Authors.
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import numpy as np
from scipy import special as scipy_special
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp

from tensorflow_probability.python.internal import test_util


def _w0(z):
  """Computes the principal branch W_0(z) of the Lambert W function."""
  # Treat -1 / exp(1) separately as special.lambertw() suffers from numerical
  # precision erros exactly at the boundary of z == exp(1)^(-1).

  if isinstance(z, float) and np.abs(z - (-1. / np.exp(1.))) < 1e-9:
    return -1.

  # This is a complex valued return value.
  return scipy_special.lambertw(z, k=0)


class LambertWTest(test_util.TestCase):

  # See https://en.wikipedia.org/wiki/Lambert_W_function#Special_values
  # for a list of special values and known identities.
  @parameterized.named_parameters(
      ("0", 0., 0.),
      ("ln(2)2", np.log(2.) * 2., np.log(2.)),
      ("exp(1)1", np.exp(1.), 1.),
      ("-1/exp(1)", -1. / np.exp(1.), -1.),
      ("10", 10., _w0(10)))
  def testLambertWKnownIdentities(self, value, expected):
    """Tests the LambertW function on some known identities."""
    scipy_wz = _w0(value)
    value = tf.convert_to_tensor(value)
    self.assertAllClose(tfp.math.lambertw(value), expected)
    self.assertAllClose(tfp.math.lambertw(value), scipy_wz)

  @parameterized.named_parameters(
      ("1D_array", np.array([1.0, 0.0])),
      ("2D_array", np.array([[1.0, 1.0], [0.0, 2.0]]))
      )
  def testLambertWWorksElementWise(self, value):
    """Tests the LambertW function works with multidimensional arrays."""
    scipy_wz = _w0(value)
    wz = tfp.math.lambertw(value)
    self.assertAllClose(wz, scipy_wz)
    self.assertEqual(value.shape, wz.shape)

  @parameterized.named_parameters(
      ("0", 0.),
      ("exp(1)", np.exp(1.)),
      # Use very large values to make sure approximation is good for large z as
      # well.
      ("5", 5.),
      ("10", 10.),
      ("100", 100.))
  def testLambertWApproxmiation(self, value):
    """Tests the approximation of the LambertW function."""
    exact = _w0(value)
    value = tf.convert_to_tensor(value)
    approx = tfp.math.lambertw_winitzki_approx(value)
    self.assertAllClose(approx, exact, rtol=0.05)

  @parameterized.named_parameters(("0", 0., 1.),
                                  ("exp(1)", np.exp(1.), 1. / (np.exp(1.) * 2)))
  @test_util.numpy_disable_gradient_test
  @test_util.jax_disable_test_missing_functionality(
      "Relies on Tensorflow gradient_checker")
  def testLambertWGradient(self, value, expected):
    """Tests the gradient of the LambertW function on some known identities."""
    x = tf.constant(value, dtype=tf.float64)
    with tf.GradientTape() as g:
      g.watch(x)
      y = tfp.math.lambertw(x)

    dy_dx = g.gradient(y, x)
    self.assertAllClose(dy_dx, expected)


if __name__ == "__main__":
  tf.test.main()

