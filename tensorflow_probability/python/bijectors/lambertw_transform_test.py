# Lint as: python3
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
"""Tests for LambertW bijectors."""

from absl.testing import parameterized
import numpy as np
from scipy import special
from scipy import stats
import tensorflow.compat.v2 as tf
from tensorflow_probability.python import bijectors as tfb
from tensorflow_probability.python.internal import test_util


def _w0(z):
  """Computes the principal branch W_0(z) of the Lambert W function."""
  # Treat -1 / exp(1) separately as special.lambertw() suffers from numerical
  # precision erros exactly at the boundary of z == exp(1)^(-1).

  if isinstance(z, float) and np.abs(z - (-1. / np.exp(1.))) < 1e-9:
    return -1.

  return np.real(special.lambertw(z, k=0))


def _w_delta_squared_numpy(z, delta):
  """Computes W_delta(z) = sign(z) * (W(delta * z^2) / delta)^(1/2)."""
  if delta == 0:
    return z

  return np.where(z == 0.,
                  np.zeros_like(z),
                  np.sign(z) * (_w0(delta * z**2.) / delta)**(0.5))


def _xexp_delta_squared_numpy(u, delta):
  """Inverse of the W_delta() function: z = u * exp(0.5 * delta * u**2)."""
  return u * np.exp(0.5 * delta * np.square(u))


def _lw_jacobian_term(z, delta):
  """Computes the jacobian term for the Lambert W transformation."""
  # See Eq (31) of https://www.hindawi.com/journals/tswj/2015/909231/.
  if delta == 0:
    return np.ones_like(z)
  return np.where(z == 0.,
                  np.ones_like(z),
                  _w_delta_squared_numpy(z, delta) /
                  (z * (1. + _w0(delta * z**2))))


# Test the heavy-tail only transformation test by setting shift=0 and scale=1.
class HeavyTailOnlyBijectorTest(test_util.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(("0", 0., 0.1, 0.),
                                  ("1", 1., 0.1, 1.051271))
  def testTailBijectorIdentities(self, value, delta, expected):
    """Tests that the output of the delta transformation is correct."""
    ht = tfb.LambertWTail(shift=0., scale=1.,
                          tailweight=tf.constant(delta, tf.float64))
    self.assertAllClose(ht(np.float64(value)), np.float64(expected))

  @parameterized.named_parameters(("0", 0.0, 0.1, 0.),
                                  ("1", np.exp(0.2 / 2. * 1.), 0.2, 1.0))
  def testTailBijectorInverseIdentities(self, value, delta, expected):
    """Tests that the output of the inverse delta transformation is correct."""
    ht = tfb.LambertWTail(shift=0., scale=1.,
                          tailweight=tf.constant(delta, tf.float64))
    self.assertAllClose(ht.inverse(np.float64(value)), np.float64(expected))

  def testTailBijectorRandomInputZeroDelta(self):
    """Tests that the output of the inverse delta transformation is correct."""
    vals = np.linspace(-1, 1, num=11)
    ht = tfb.LambertWTail(shift=0., scale=1., tailweight=0.0)
    self.assertAllClose(ht(vals), vals)

  @parameterized.named_parameters(("0.01", 0.01),
                                  ("0.1", 0.1)
                                 )
  def testTailBijectorRandomInputNonZeroDelta(self, delta):
    """Tests that the output of the inverse delta transformation is correct."""
    vals = np.linspace(-1, 1, num=10)
    ht = tfb.LambertWTail(shift=0., scale=1.,
                          tailweight=tf.constant(delta, tf.float64))
    with self.session():
      # Gaussianizing makes the values be further away from zero, i.e., their
      # ratio > 1 (for vals != 0).
      self.assertTrue(np.all(self.evaluate(ht(vals)) / vals > 1.))
      self.assertAllClose(ht.inverse(ht(vals)), vals)
      self.assertAllClose(ht(vals), _xexp_delta_squared_numpy(vals, delta))

  @parameterized.named_parameters(("0", 0.0, 0.1, 0.),
                                  ("1", 1.0, 0.2,
                                   np.log(_lw_jacobian_term(1.0, 0.2))),
                                  ("0_and_1",
                                   np.array([0., 1.]), 0.1,
                                   np.log(_lw_jacobian_term(np.array([0., 1.]),
                                                            0.1)))
                                 )
  def testTailBijectorLogDetJacobian(self, value, delta, expected):
    """Tests that the output of the inverse delta transformation is correct."""
    ht = tfb.LambertWTail(shift=0., scale=1.,
                          tailweight=tf.constant(delta, tf.float64))
    if isinstance(value, np.ndarray):
      value = value.astype(np.float64)
      expected = expected.astype(np.float64)
    else:
      value = np.float64(value)
      expected = np.float64(expected)
    self.assertAllClose(ht._inverse_log_det_jacobian(
        tf.convert_to_tensor(value)),
                        expected)


class LambertWGaussianizationTest(test_util.TestCase, parameterized.TestCase):

  def setUp(self):
    """This method will be run before each of the test methods in the class."""
    super(LambertWGaussianizationTest, self).setUp()
    self.tailweight = 0.2
    self.loc = 2.0  # location parameter of Normal
    self.scale = 0.1  # scale parameter of Normal

  def testLambertWGaussianizationDeltaZero(self):
    """Tests that the output of ShiftScaleTail is correct when delta=0."""
    values = np.random.normal(loc=self.loc, scale=self.scale, size=10)
    lsht = tfb.LambertWTail(shift=self.loc, scale=self.scale, tailweight=0.0)
    self.assertAllClose(values, lsht(values))
    self.assertAllClose(values, lsht.inverse(values))

  @parameterized.named_parameters(
      ("0_01", 0.01),
      ("0_5", 0.5)
      )
  def testLambertWGaussianizationDeltaNonZeroSpecificValues(self, delta):
    """Tests that the output of ShiftScaleTail is correct when delta!=0."""
    vals = np.linspace(-1, 1, 10) + self.loc
    lsht = tfb.LambertWTail(shift=self.loc, scale=self.scale, tailweight=delta)
    with self.session():
      scaled_vals = (vals - self.loc) / self.scale
      ht_vals = _xexp_delta_squared_numpy(scaled_vals, delta=delta)
      ht_vals *= self.scale
      ht_vals += self.loc
      self.assertAllClose(ht_vals, self.evaluate(lsht(vals)), rtol=0.0001)
      # Inverse-Gaussianizing pushes the values be further away from the mean,
      # i.e., their centered ratio > 1 (for vals - loc != 0).
      self.assertTrue(np.all((self.evaluate(lsht(vals)) - self.loc) /
                             (vals - self.loc)
                             > 1.))
      # Inverse function is correct.
      self.assertAllClose(lsht.inverse(lsht(vals)), vals)

  def testLambertWGaussianizationDeltaNonZero(self):
    """Tests that the inverse of the heavy tail transform is Normal."""
    vals = np.random.normal(loc=self.loc, scale=self.scale,
                            size=100).astype(np.float64)
    lsht = tfb.LambertWTail(shift=self.loc, scale=self.scale,
                            tailweight=self.tailweight)
    with self.session():
      heavy_tailed_vals = lsht(vals)
      _, p = stats.normaltest(self.evaluate(heavy_tailed_vals))
      self.assertLess(p, 1e-2)
      gaussianized_vals = lsht.inverse(heavy_tailed_vals)
      _, p = stats.normaltest(self.evaluate(gaussianized_vals))
      self.assertGreater(p, 0.05)
      self.assertAllClose(vals, gaussianized_vals)


if __name__ == "__main__":
  np.random.seed(10)
  tf.test.main()
