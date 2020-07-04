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

import functools

from absl.testing import parameterized
import numpy as np
from scipy import constants as scipy_constants
from scipy import special as scipy_special
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp

from tensorflow_probability.python import math as tfp_math
from tensorflow_probability.python.internal import test_util
from tensorflow_probability.python.math.special import _compute_general_continued_fraction


def _w0(z):
  """Computes the principal branch W_0(z) of the Lambert W function."""
  # Treat -1 / exp(1) separately as special.lambertw() suffers from numerical
  # precision erros exactly at the boundary of z == exp(1)^(-1).

  if isinstance(z, float) and np.abs(z - (-1. / np.exp(1.))) < 1e-9:
    return -1.

  # This is a complex valued return value.
  return scipy_special.lambertw(z, k=0)


class RoundExponentialBumpFunctionTest(test_util.TestCase):

  @parameterized.named_parameters(
      ("float32", np.float32),
      ("float64", np.float64),
  )
  def testValueOnSupportInterior(self, dtype):
    # round_exponential_bump_function(x) = 0 for x right at the edge of the
    # support, e.g. x = -0.999.  This is expected, due to the exponential and
    # division.
    x = tf.convert_to_tensor([
        -0.9925,
        -0.5,
        0.,
        0.5,
        0.9925
    ], dtype=dtype)
    y = tfp_math.round_exponential_bump_function(x)

    self.assertDTypeEqual(y, dtype)

    y_ = self.evaluate(y)

    self.assertAllFinite(y_)

    # round_exponential_bump_function(0) = 1.
    self.assertAllClose(1., y_[2])

    # round_exponential_bump_function(x) > 0 for |x| < 1.
    self.assertAllGreater(y_, 0)

  @test_util.numpy_disable_gradient_test
  @parameterized.named_parameters(
      ("float32", np.float32),
      ("float64", np.float64),
  )
  def testGradientOnSupportInterior(self, dtype):
    # round_exponential_bump_function(x) = 0 for x right at the edge of the
    # support, e.g. x = -0.999.  This is expected, due to the exponential and
    # division.
    x = tf.convert_to_tensor([
        -0.9925,
        -0.5,
        0.,
        0.5,
        0.9925
    ], dtype=dtype)

    _, dy_dx = tfp_math.value_and_gradient(
        tfp_math.round_exponential_bump_function, x)

    self.assertDTypeEqual(dy_dx, dtype)

    dy_dx_ = self.evaluate(dy_dx)

    # grad[round_exponential_bump_function](0) = 0
    self.assertEqual(0., dy_dx_[2])
    self.assertAllFinite(dy_dx_)

    # Increasing on (-1, 0), decreasing on (0, 1).
    self.assertAllGreater(dy_dx_[:2], 0)
    self.assertAllLess(dy_dx_[-2:], 0)

  @parameterized.named_parameters(
      ("float32", np.float32),
      ("float64", np.float64),
  )
  def testValueOutsideAndOnEdgeOfSupport(self, dtype):
    finfo = np.finfo(dtype)
    x = tf.convert_to_tensor([
        # Sqrt(finfo.max)**2 = finfo.max < Inf, so
        # round_exponential_bump_function == 0 here.
        -np.sqrt(finfo.max),
        # -2 is just outside the support, so round_exponential_bump_function
        # should == 0.
        -2.,
        # -1 is on boundary of support, so round_exponential_bump_function
        # should == 0.
        # The gradient should also equal 0.
        -1.,
        1.,
        2.0,
        np.sqrt(finfo.max),
    ],
                             dtype=dtype)
    y = tfp_math.round_exponential_bump_function(x)

    self.assertDTypeEqual(y, dtype)

    y_ = self.evaluate(y)

    # We hard-set y to 0 outside support.
    self.assertAllFinite(y_)
    self.assertAllEqual(y_, np.zeros((6,)))

  @test_util.numpy_disable_gradient_test
  @parameterized.named_parameters(
      ("float32", np.float32),
      ("float64", np.float64),
  )
  def testGradientOutsideAndOnEdgeOfSupport(self, dtype):
    finfo = np.finfo(dtype)
    x = tf.convert_to_tensor([
        # Sqrt(finfo.max)**2 = finfo.max < Inf, so
        # round_exponential_bump_function == 0 here.
        -np.sqrt(finfo.max),
        # -2 is just outside the support, so round_exponential_bump_function
        # should == 0.
        -2.,
        # -1 is on boundary of support, so round_exponential_bump_function
        # should == 0.
        # The gradient should also equal 0.
        -1.,
        1.,
        2.0,
        np.sqrt(finfo.max),
    ],
                             dtype=dtype)
    _, dy_dx = tfp_math.value_and_gradient(
        tfp_math.round_exponential_bump_function, x)

    self.assertDTypeEqual(dy_dx, dtype)

    dy_dx_ = self.evaluate(dy_dx)

    # Since x is outside the support, the gradient is zero.
    self.assertAllEqual(dy_dx_, np.zeros((6,)))


class BesselTest(test_util.TestCase):

  def testContinuedFraction(self):
    # Check that the simplest continued fraction returns the golden ratio.
    self.assertAllClose(
        self.evaluate(
            _compute_general_continued_fraction(
                100, [], partial_numerator_fn=lambda _: 1.)),
        scipy_constants.golden - 1.)

    # Check the continued fraction constant is returned.
    cf_constant_denominators = scipy_special.i1(2.) / scipy_special.i0(2.)

    self.assertAllClose(
        self.evaluate(
            _compute_general_continued_fraction(
                100,
                [],
                partial_denominator_fn=lambda i: i,
                tolerance=1e-5)),
        cf_constant_denominators, rtol=1e-5)

    cf_constant_numerators = np.sqrt(2 / (np.e * np.pi)) / (
        scipy_special.erfc(np.sqrt(0.5))) - 1.

    # Check that we can specify dtype and tolerance.
    self.assertAllClose(
        self.evaluate(
            _compute_general_continued_fraction(
                100, [], partial_numerator_fn=lambda i: i,
                tolerance=1e-5,
                dtype=tf.float64)),
        cf_constant_numerators, rtol=1e-5)

  def VerifyBesselIvRatio(self, v, z, rtol):
    bessel_iv_ratio, v, z = self.evaluate([
        tfp.math.bessel_iv_ratio(v, z), v, z])
    # Use ive to avoid nans.
    scipy_ratio = scipy_special.ive(v, z) / scipy_special.ive(v - 1., z)
    self.assertAllClose(bessel_iv_ratio, scipy_ratio, rtol=rtol)

  def testBesselIvRatioVAndZSmall(self):
    seed_stream = test_util.test_seed_stream()
    z = tf.random.uniform([int(1e5)], seed=seed_stream())
    v = tf.random.uniform([int(1e5)], seed=seed_stream())
    # When both values are small, both the scipy ratio and
    # the computation become numerically unstable.
    # Anecdotally (when comparing to mpmath) the computation is more often
    # 'right' compared to the naive ratio.
    self.VerifyBesselIvRatio(v, z, rtol=2e-4)

  def testBesselIvRatioVAndZMedium(self):
    seed_stream = test_util.test_seed_stream()
    z = tf.random.uniform([int(1e5)], 1., 10., seed=seed_stream())
    v = tf.random.uniform([int(1e5)], 1., 10., seed=seed_stream())
    self.VerifyBesselIvRatio(v, z, rtol=1e-6)

  def testBesselIvRatioVAndZLarge(self):
    seed_stream = test_util.test_seed_stream()
    # Use 50 as a cap. It's been observed that for v > 50, that
    # the scipy ratio can be quite wrong compared to mpmath.
    z = tf.random.uniform([int(1e5)], 10., 50., seed=seed_stream())
    v = tf.random.uniform([int(1e5)], 10., 50., seed=seed_stream())

    # For large v, z, scipy can return NaN values. Filter those out.
    bessel_iv_ratio, v, z = self.evaluate([
        tfp.math.bessel_iv_ratio(v, z), v, z])
    # Use ive to avoid nans.
    scipy_ratio = scipy_special.ive(v, z) / scipy_special.ive(v - 1., z)
    # Exclude zeros and NaN's from scipy. This can happen because the
    # individual function computations may zero out, and thus cause issues
    # in the ratio.
    safe_scipy_values = np.where(
        ~np.isnan(scipy_ratio) & (scipy_ratio != 0.))

    self.assertAllClose(
        bessel_iv_ratio[safe_scipy_values],
        scipy_ratio[safe_scipy_values],
        # We need to set a high rtol as the scipy computation is numerically
        # unstable.
        rtol=1e-5)

  def testBesselIvRatioVLessThanZ(self):
    seed_stream = test_util.test_seed_stream()
    z = tf.random.uniform([int(1e5)], 1., 10., seed=seed_stream())
    # Make v randomly less than z
    v = z * tf.random.uniform([int(1e5)], 0.1, 0.5, seed=seed_stream())
    self.VerifyBesselIvRatio(v, z, rtol=1e-6)

  def testBesselIvRatioVGreaterThanZ(self):
    seed_stream = test_util.test_seed_stream()
    v = tf.random.uniform([int(1e5)], 1., 10., seed=seed_stream())
    # Make z randomly less than v
    z = v * tf.random.uniform([int(1e5)], 0.1, 0.5, seed=seed_stream())
    self.VerifyBesselIvRatio(v, z, rtol=1e-6)

  @test_util.numpy_disable_gradient_test
  @test_util.jax_disable_test_missing_functionality(
      "Relies on Tensorflow gradient_checker")
  def testBesselIvRatioGradient(self):
    v = tf.constant([0.5, 1., 10., 20.])[..., tf.newaxis]
    x = tf.constant([0.1, 0.5, 0.9, 1., 12., 14., 22.])

    err = self.compute_max_gradient_error(
        functools.partial(tfp_math.bessel_iv_ratio, v), [x])
    self.assertLess(err, 2e-4)


class SpecialTest(test_util.TestCase):

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

  def testLogGammaCorrection(self):
    x = tfp.distributions.HalfCauchy(loc=8., scale=10.).sample(
        10000, test_util.test_seed())
    pi = 3.14159265
    stirling = x * tf.math.log(x) - x + 0.5 * tf.math.log(2 * pi / x)
    tfp_gamma_ = stirling + tfp_math.log_gamma_correction(x)
    tf_gamma, tfp_gamma = self.evaluate([tf.math.lgamma(x), tfp_gamma_])
    self.assertAllClose(tf_gamma, tfp_gamma, atol=0, rtol=1e-6)

  def testLogGammaDifference(self):
    y = tfp.distributions.HalfCauchy(loc=8., scale=10.).sample(
        10000, test_util.test_seed())
    y_64 = tf.cast(y, tf.float64)
    # Not testing x near zero because the naive method is too inaccurate.
    # We will get implicit coverage in testLogBeta, where a good reference
    # implementation is available (scipy_special.betaln).
    x = tfp.distributions.Uniform(low=4., high=12.).sample(
        10000, test_util.test_seed())
    x_64 = tf.cast(x, tf.float64)
    naive_64_ = tf.math.lgamma(y_64) - tf.math.lgamma(x_64 + y_64)
    naive_64, sophisticated, sophisticated_64 = self.evaluate(
        [naive_64_, tfp_math.log_gamma_difference(x, y),
         tfp_math.log_gamma_difference(x_64, y_64)])
    # Check that we're in the ballpark of the definition (which has to be
    # computed in double precision because it's so inaccurate).
    self.assertAllClose(naive_64, sophisticated_64, atol=1e-6, rtol=4e-4)
    # Check that we don't lose accuracy in single precision, at least relative
    # to ourselves.
    atol = 1e-8
    rtol = 1e-6
    self.assertAllClose(sophisticated, sophisticated_64, atol=atol, rtol=rtol)

  @test_util.numpy_disable_gradient_test
  def testLogGammaDifferenceGradient(self):
    def simple_difference(x, y):
      return tf.math.lgamma(y) - tf.math.lgamma(x + y)
    y = tfp.distributions.HalfCauchy(loc=8., scale=10.).sample(
        10000, test_util.test_seed())
    x = tfp.distributions.Uniform(low=0., high=8.).sample(
        10000, test_util.test_seed())
    _, [simple_gx_, simple_gy_] = tfp.math.value_and_gradient(
        simple_difference, [x, y])
    _, [gx_, gy_] = tfp.math.value_and_gradient(
        tfp_math.log_gamma_difference, [x, y])
    simple_gx, simple_gy, gx, gy = self.evaluate(
        [simple_gx_, simple_gy_, gx_, gy_])
    self.assertAllClose(gx, simple_gx)
    self.assertAllClose(gy, simple_gy)

  @test_util.numpy_disable_gradient_test
  def testLogGammaDifferenceGradientBroadcasting(self):
    def simple_difference(x, y):
      return tf.math.lgamma(y) - tf.math.lgamma(x + y)
    x = tf.constant(1.)
    y = tf.constant([[1., 2., 3.], [4., 5., 6.]])
    _, [simple_gx_, simple_gy_] = tfp.math.value_and_gradient(
        simple_difference, [x, y])
    _, [gx_, gy_] = tfp.math.value_and_gradient(
        tfp_math.log_gamma_difference, [x, y])
    simple_gx, simple_gy, gx, gy = self.evaluate(
        [simple_gx_, simple_gy_, gx_, gy_])
    self.assertAllClose(gx, simple_gx)
    self.assertAllClose(gy, simple_gy)

  def testLogBeta(self):
    strm = test_util.test_seed_stream()
    x = tfp.distributions.HalfCauchy(loc=1., scale=15.).sample(10000, strm())
    x = self.evaluate(x)
    y = tfp.distributions.HalfCauchy(loc=1., scale=15.).sample(10000, strm())
    y = self.evaluate(y)
    # Why not 1e-8?
    # - Could be because scipy does the reduction loops recommended
    #   by DiDonato and Morris 1988
    # - Could be that tf.math.lgamma is less accurate than scipy
    # - Could be that scipy evaluates in 64 bits internally
    atol = 1e-7
    rtol = 1e-5
    self.assertAllClose(
        scipy_special.betaln(x, y), tfp_math.lbeta(x, y),
        atol=atol, rtol=rtol)

  @test_util.numpy_disable_gradient_test
  def testLogBetaGradient(self):
    def simple_lbeta(x, y):
      return tf.math.lgamma(x) + tf.math.lgamma(y) - tf.math.lgamma(x + y)
    strm = test_util.test_seed_stream()
    x = tfp.distributions.HalfCauchy(loc=1., scale=15.).sample(10000, strm())
    y = tfp.distributions.HalfCauchy(loc=1., scale=15.).sample(10000, strm())
    _, [simple_gx_, simple_gy_] = tfp.math.value_and_gradient(
        simple_lbeta, [x, y])
    _, [gx_, gy_] = tfp.math.value_and_gradient(tfp_math.lbeta, [x, y])
    simple_gx, simple_gy, gx, gy = self.evaluate(
        [simple_gx_, simple_gy_, gx_, gy_])
    self.assertAllClose(gx, simple_gx)
    self.assertAllClose(gy, simple_gy)

  @test_util.numpy_disable_gradient_test
  def testLogBetaGradientBroadcasting(self):
    def simple_lbeta(x, y):
      return tf.math.lgamma(x) + tf.math.lgamma(y) - tf.math.lgamma(x + y)
    x = tf.constant(1.)
    y = tf.constant([[1., 2., 3.], [4., 5., 6.]])
    _, [simple_gx_, simple_gy_] = tfp.math.value_and_gradient(
        simple_lbeta, [x, y])
    _, [gx_, gy_] = tfp.math.value_and_gradient(tfp_math.lbeta, [x, y])
    simple_gx, simple_gy, gx, gy = self.evaluate(
        [simple_gx_, simple_gy_, gx_, gy_])
    self.assertAllClose(gx, simple_gx)
    self.assertAllClose(gy, simple_gy)

  @parameterized.parameters(tf.float32, tf.float64)
  def testLogBetaDtype(self, dtype):
    x = tf.constant([1., 2.], dtype=dtype)
    y = tf.constant([3., 4.], dtype=dtype)
    result = tfp_math.lbeta(x, y)
    self.assertEqual(result.dtype, dtype)


if __name__ == "__main__":
  tf.test.main()
