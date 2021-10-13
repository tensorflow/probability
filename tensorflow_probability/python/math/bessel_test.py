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

import functools

from absl.testing import parameterized
import numpy as np
from scipy import constants as scipy_constants
from scipy import special as scipy_special
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp

from tensorflow_probability.python import math as tfp_math
from tensorflow_probability.python.internal import test_util
from tensorflow_probability.python.math.bessel import _compute_general_continued_fraction


@test_util.test_all_tf_execution_regimes
class BesselIvRatioTest(test_util.TestCase):

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

    bessel_iv_ratio, v, z = self.evaluate([
        tfp.math.bessel_iv_ratio(v, z), v, z])
    scipy_ratio = scipy_special.ive(v, z) / scipy_special.ive(v - 1., z)

    safe_scipy_values = np.where(
        ~np.isnan(scipy_ratio) & (scipy_ratio != 0.))

    self.assertAllClose(
        bessel_iv_ratio[safe_scipy_values],
        scipy_ratio[safe_scipy_values], rtol=3e-4, atol=1e-6)

  def testBesselIvRatioVAndZMedium(self):
    seed_stream = test_util.test_seed_stream()
    z = tf.random.uniform([int(1e5)], 1., 10., seed=seed_stream())
    v = tf.random.uniform([int(1e5)], 1., 10., seed=seed_stream())
    self.VerifyBesselIvRatio(v, z, rtol=7e-6)

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
        rtol=1e-6)

  def testBesselIvRatioVLessThanZ(self):
    seed_stream = test_util.test_seed_stream()
    z = tf.random.uniform([int(1e5)], 1., 10., seed=seed_stream())
    # Make v randomly less than z
    v = z * tf.random.uniform([int(1e5)], 0.1, 0.5, seed=seed_stream())
    self.VerifyBesselIvRatio(v, z, rtol=6e-6)

  def testBesselIvRatioVGreaterThanZ(self):
    seed_stream = test_util.test_seed_stream()
    v = tf.random.uniform([int(1e5)], 1., 10., seed=seed_stream())
    # Make z randomly less than v
    z = v * tf.random.uniform([int(1e5)], 0.1, 0.5, seed=seed_stream())
    self.VerifyBesselIvRatio(v, z, rtol=1e-6)

  @test_util.numpy_disable_gradient_test
  def testBesselIvRatioGradient(self):
    v = tf.constant([0.5, 1., 10., 20.])[..., tf.newaxis]
    x = tf.constant([0.1, 0.5, 0.9, 1., 12., 14., 22.])

    err = self.compute_max_gradient_error(
        functools.partial(tfp_math.bessel_iv_ratio, v), [x])
    self.assertLess(err, 2e-4)


@test_util.test_all_tf_execution_regimes
class BesselIveTest(test_util.TestCase):

  def VerifyBesselIve(self, v, z, rtol, atol=1e-7):
    bessel_ive, v, z = self.evaluate([
        tfp.math.bessel_ive(v, z), v, z])
    scipy_ive = scipy_special.ive(v, z)
    self.assertAllClose(bessel_ive, scipy_ive, rtol=rtol, atol=atol)

  @parameterized.named_parameters(
      ("float32", np.float32),
      ("float64", np.float64),
  )
  def testBesselIveAtZero(self, dtype):
    # Check that z = 0 returns 1 for v = 0 and 0 otherwise.
    seed_stream = test_util.test_seed_stream()
    v = tf.random.uniform([10], 1., 10., seed=seed_stream(), dtype=dtype)
    z = tf.constant(0., dtype=dtype)
    self.assertAllClose(
        self.evaluate(tfp.math.bessel_ive(v, z)),
        np.zeros([10], dtype=dtype))

    v = tf.constant([0.], dtype=dtype)
    self.assertAllClose(
        self.evaluate(tfp.math.bessel_ive(v, z)),
        np.ones([1], dtype=dtype))

  @parameterized.named_parameters(
      ("float32", np.float32),
      ("float64", np.float64),
  )
  def testBesselIveZNegativeNaN(self, dtype):
    # Check that z < 0 returns NaN for non-integer v.
    seed_stream = test_util.test_seed_stream()
    v = np.linspace(1.1, 10.2, num=11, dtype=dtype)
    z = tf.random.uniform([11], -10., -1., seed=seed_stream(), dtype=dtype)
    bessel_ive = self.evaluate(tfp.math.bessel_ive(v, z))
    self.assertTrue(np.all(np.isnan(bessel_ive)))

  @parameterized.named_parameters(
      ("float32", np.float32, 1e-6),
      ("float64", np.float64, 1e-6),
  )
  def testBesselIveZNegativeVInteger(self, dtype, rtol):
    seed_stream = test_util.test_seed_stream()
    v = np.linspace(1., 10., num=10, dtype=dtype)
    z = tf.random.uniform([10], -10., -1., seed=seed_stream(), dtype=dtype)
    z, bessel_ive = self.evaluate([z, tfp.math.bessel_ive(v, z)])
    scipy_ive = scipy_special.ive(v, z)
    self.assertAllClose(bessel_ive, scipy_ive, rtol=rtol)

  @parameterized.named_parameters(
      ("float32", np.float32, 1e-6),
      ("float64", np.float64, 1e-6),
  )
  def testBesselIveZNegativeVLarge(self, dtype, rtol):
    seed_stream = test_util.test_seed_stream()
    v = np.linspace(100., 200., num=10, dtype=dtype)
    z = tf.random.uniform([10], -10., -1., seed=seed_stream(), dtype=dtype)
    z, bessel_ive = self.evaluate([z, tfp.math.bessel_ive(v, z)])
    scipy_ive = scipy_special.ive(v, z)
    self.assertAllClose(bessel_ive, scipy_ive, rtol=rtol)

  @parameterized.named_parameters(
      ("float32", np.float32, 1.5e-6),
      ("float64", np.float64, 1e-6),
  )
  def testBesselIveVAndZSmall(self, dtype, rtol):
    seed_stream = test_util.test_seed_stream()
    z = tf.random.uniform([int(1e5)], seed=seed_stream(), dtype=dtype)
    v = tf.random.uniform([int(1e5)], seed=seed_stream(), dtype=dtype)
    self.VerifyBesselIve(v, z, rtol=rtol)

  @parameterized.named_parameters(
      ("float32", np.float32, 3e-6),
      ("float64", np.float64, 1e-6),
  )
  def testBesselIveZTiny(self, dtype, rtol):
    seed_stream = test_util.test_seed_stream()
    z = tf.random.uniform(
        [int(1e5)], 1e-13, 1e-6, seed=seed_stream(), dtype=dtype)
    v = tf.random.uniform([int(1e5)], 0., 10., seed=seed_stream(), dtype=dtype)
    self.VerifyBesselIve(v, z, rtol=rtol, atol=1e-7)

  @parameterized.named_parameters(
      ("float32", np.float32, 7e-6),
      ("float64", np.float64, 6e-6),
  )
  def testBesselIveVAndZMedium(self, dtype, rtol):
    seed_stream = test_util.test_seed_stream()
    z = tf.random.uniform([int(1e5)], 1., 10., seed=seed_stream(), dtype=dtype)
    v = tf.random.uniform([int(1e5)], 1., 10., seed=seed_stream(), dtype=dtype)
    self.VerifyBesselIve(v, z, rtol=rtol)

  @parameterized.named_parameters(
      ("float32", np.float32, 1e-6),
      ("float64", np.float64, 1e-6),
  )
  def testBesselIveVAndZLarge(self, dtype, rtol):
    seed_stream = test_util.test_seed_stream()
    z = tf.random.uniform([int(1e5)], 10., 50., seed=seed_stream(), dtype=dtype)
    v = tf.random.uniform([int(1e5)], 10., 50., seed=seed_stream(), dtype=dtype)
    self.VerifyBesselIve(v, z, rtol=rtol)

  @parameterized.named_parameters(
      ("float32", np.float32, 1e-6),
      ("float64", np.float64, 1e-6),
  )
  def testBesselIveVAndZVeryLarge(self, dtype, rtol):
    seed_stream = test_util.test_seed_stream()
    z = tf.random.uniform(
        [int(1e5)], 50., 100., seed=seed_stream(), dtype=dtype)
    v = tf.random.uniform(
        [int(1e5)], 50., 100., seed=seed_stream(), dtype=dtype)
    self.VerifyBesselIve(v, z, rtol=rtol)

  @parameterized.named_parameters(
      ("float32", np.float32, 7e-6),
      ("float64", np.float64, 7e-6),
  )
  def testBesselIveVLessThanZ(self, dtype, rtol):
    seed_stream = test_util.test_seed_stream()
    z = tf.random.uniform([int(1e5)], 1., 10., seed=seed_stream(), dtype=dtype)
    # Make v randomly less than z
    v = z * tf.random.uniform(
        [int(1e5)], 0.1, 0.5, seed=seed_stream(), dtype=dtype)
    self.VerifyBesselIve(v, z, rtol=rtol)

  @parameterized.named_parameters(
      ("float32", np.float32, 1e-6),
      ("float64", np.float64, 1e-6),
  )
  def testBesselIveVGreaterThanZ(self, dtype, rtol):
    seed_stream = test_util.test_seed_stream()
    v = tf.random.uniform([int(1e5)], 1., 10., seed=seed_stream(), dtype=dtype)
    # Make z randomly less than v
    z = v * tf.random.uniform(
        [int(1e5)], 0.1, 0.5, seed=seed_stream(), dtype=dtype)
    self.VerifyBesselIve(v, z, rtol=rtol)

  @parameterized.named_parameters(
      ("float32", np.float32, 1e-4),
      ("float64", np.float64, 7e-6),
  )
  def testBesselIveVNegative(self, dtype, rtol):
    seed_stream = test_util.test_seed_stream()
    v = tf.random.uniform(
        [int(1e5)], -10., -1., seed=seed_stream(), dtype=dtype)
    z = tf.random.uniform([int(1e5)], 1., 15., seed=seed_stream(), dtype=dtype)
    self.VerifyBesselIve(v, z, rtol=rtol)

  @parameterized.named_parameters(
      ("float32", np.float32, 1e-6),
      ("float64", np.float64, 1e-6),
  )
  def testBesselIveVZero(self, dtype, rtol):
    seed_stream = test_util.test_seed_stream()
    v = tf.constant(0., dtype=dtype)
    z = tf.random.uniform([int(1e5)], 1., 15., seed=seed_stream(), dtype=dtype)
    self.VerifyBesselIve(v, z, rtol=rtol)

  @parameterized.named_parameters(
      ("float32", np.float32, 1e-6),
      ("float64", np.float64, 1e-6),
  )
  def testBesselIveLargeZ(self, dtype, rtol):
    seed_stream = test_util.test_seed_stream()
    v = tf.random.uniform(
        [int(1e5)], minval=0., maxval=0.5, seed=seed_stream(), dtype=dtype)
    z = tf.random.uniform(
        [int(1e5)], minval=100., maxval=10000., seed=seed_stream(), dtype=dtype)
    self.VerifyBesselIve(v, z, rtol=rtol)

  @test_util.numpy_disable_gradient_test
  @parameterized.named_parameters(
      ("float32", np.float32),
      ("float64", np.float64))
  def testBesselIveGradient(self, dtype):
    v = tf.constant([-1., 0.5, 1., 10., 20.], dtype=dtype)[..., tf.newaxis]
    z = tf.constant([0.2, 0.5, 0.9, 1., 12., 14., 22.], dtype=dtype)

    err = self.compute_max_gradient_error(
        functools.partial(tfp_math.bessel_ive, v), [z])
    self.assertLess(err, 2e-4)

  @test_util.numpy_disable_gradient_test
  @parameterized.named_parameters(
      ("float32", np.float32),
      ("float64", np.float64))
  def testBesselIveNegativeGradient(self, dtype):
    v = tf.constant([1., 10., 20.], dtype=dtype)[..., tf.newaxis]
    z = tf.constant([-.2, -2.5, -3.5, -5.], dtype=dtype)

    err = self.compute_max_gradient_error(
        functools.partial(tfp_math.bessel_ive, v), [z])
    self.assertLess(err, 2e-4)

  @parameterized.named_parameters(
      ("float32", np.float32, 1e-6),
      ("float64", np.float64, 1e-6),
  )
  def testLogBesselIveCorrect(self, dtype, rtol):
    seed_stream = test_util.test_seed_stream()
    v = tf.random.uniform(
        [int(1e5)], minval=0.1, maxval=0.5, seed=seed_stream(), dtype=dtype)
    z = tf.random.uniform(
        [int(1e5)], minval=1., maxval=10., seed=seed_stream(), dtype=dtype)
    _, _, log_bessel_ive_expected_, log_bessel_ive_actual_ = self.evaluate([
        v,
        z,
        tf.math.log(tfp.math.bessel_ive(v, z)),
        tfp.math.log_bessel_ive(v, z)])
    self.assertAllClose(
        log_bessel_ive_expected_, log_bessel_ive_actual_, rtol=rtol)

  def testLogBesselIveTestNonInf(self):
    # Test that log_bessel_ive(v, z) has more resolution than simply computing
    # log(bessel_ive(v, z)). The inputs below will return -inf in naive float64
    # computation.
    v = np.array([10., 12., 30., 50.], np.float32)
    z = np.logspace(-10., -1., 20).reshape((20, 1)).astype(np.float32)
    self.assertAllFinite(self.evaluate(tfp.math.log_bessel_ive(v, z)))

  @test_util.numpy_disable_gradient_test
  @parameterized.named_parameters(
      ("float32", np.float32, 1e-3),
      ("float64", np.float64, 1e-4))
  def testLogBesselIveGradient(self, dtype, tol):
    v = tf.constant([-0.2, -1., 1., 0.5, 2.], dtype=dtype)[..., tf.newaxis]
    z = tf.constant([0.3, 0.5, 0.9, 1., 12., 22.], dtype=dtype)

    err = self.compute_max_gradient_error(
        functools.partial(tfp_math.log_bessel_ive, v), [z])
    self.assertLess(err, tol)

  @test_util.numpy_disable_gradient_test
  @parameterized.named_parameters(("float32", np.float32),
                                  ("float64", np.float64))
  def testJitGradBcastLogBesselIve(self, dtype):
    self.skip_if_no_xla()

    @tf.function(jit_compile=True)
    def f(v, z):
      dy = tf.random.normal(z.shape, seed=test_util.test_seed(), dtype=dtype)
      return tf.nest.map_structure(
          lambda t: () if t is None else t,  # session.run doesn't like `None`.
          tfp.math.value_and_gradient(
              lambda v, z: tfp.math.log_bessel_ive(v, z)**2,
              (v, z),
              output_gradients=dy))

    v = tf.constant(0.5, dtype=dtype)
    z = tf.constant([[0.3, 0.5, 0.9], [1., 12., 22.]], dtype=dtype)

    self.evaluate(f(v, z))


@test_util.test_all_tf_execution_regimes
class BesselKveTest(test_util.TestCase):

  def VerifyBesselKve(self, v, z, rtol):
    bessel_kve, v, z = self.evaluate([
        tfp.math.bessel_kve(v, z), v, z])
    scipy_kve = scipy_special.kve(v, z)
    self.assertAllClose(bessel_kve, scipy_kve, rtol=rtol)

  @parameterized.named_parameters(
      ("float32", np.float32),
      ("float64", np.float64),
  )
  def testBesselKveAtZero(self, dtype):
    # Check that z = 0 returns inf for v = 0.
    seed_stream = test_util.test_seed_stream()
    v = tf.random.uniform([10], 1., 10., seed=seed_stream(), dtype=dtype)
    z = tf.constant(0., dtype=dtype)
    self.assertAllClose(
        self.evaluate(tfp.math.bessel_kve(v, z)),
        np.full([10], np.inf, dtype=dtype))

    v = tf.constant([0.], dtype=dtype)
    self.assertAllClose(
        self.evaluate(tfp.math.bessel_kve(v, z)),
        np.full([1], np.inf, dtype=dtype))

  @parameterized.named_parameters(
      ("float32", np.float32),
      ("float64", np.float64),
  )
  def testBesselKveZNegativeNaN(self, dtype):
    # Check that z < 0 returns NaN for non-integer v.
    seed_stream = test_util.test_seed_stream()
    v = np.linspace(1.1, 10.2, num=11, dtype=dtype)
    z = tf.random.uniform([11], -10., -1., seed=seed_stream(), dtype=dtype)
    bessel_kve = self.evaluate(tfp.math.bessel_kve(v, z))
    self.assertTrue(np.all(np.isnan(bessel_kve)))

  @parameterized.named_parameters(
      ("float32", np.float32, 1e-6),
      ("float64", np.float64, 1e-6),
  )
  def testBesselKveZNegativeVInteger(self, dtype, rtol):
    seed_stream = test_util.test_seed_stream()
    v = np.linspace(1., 10., num=10, dtype=dtype)
    z = tf.random.uniform([10], -10., -1., seed=seed_stream(), dtype=dtype)
    z, bessel_kve = self.evaluate([z, tfp.math.bessel_kve(v, z)])
    scipy_kve = scipy_special.kve(v, z)
    self.assertAllClose(bessel_kve, scipy_kve, rtol=rtol)

  @parameterized.named_parameters(
      ("float32", np.float32, 1e-6),
      ("float64", np.float64, 1e-6),
  )
  def testBesselKveZNegativeVLarge(self, dtype, rtol):
    seed_stream = test_util.test_seed_stream()
    v = np.linspace(100., 200., num=10, dtype=dtype)
    z = tf.random.uniform([10], -10., -1., seed=seed_stream(), dtype=dtype)
    z, bessel_kve = self.evaluate([z, tfp.math.bessel_kve(v, z)])
    scipy_kve = scipy_special.kve(v, z)
    self.assertAllClose(bessel_kve, scipy_kve, rtol=rtol)

  @parameterized.named_parameters(
      ("float32", np.float32, 1.5e-6),
      ("float64", np.float64, 1e-6),
  )
  def testBesselKveVAndZSmall(self, dtype, rtol):
    seed_stream = test_util.test_seed_stream()
    z = tf.random.uniform([int(1e5)], seed=seed_stream(), dtype=dtype)
    v = tf.random.uniform([int(1e5)], seed=seed_stream(), dtype=dtype)
    self.VerifyBesselKve(v, z, rtol=rtol)

  @parameterized.named_parameters(
      ("float32", np.float32, 4e-6),
      ("float64", np.float64, 1e-6),
  )
  def testBesselKveVAndZMedium(self, dtype, rtol):
    seed_stream = test_util.test_seed_stream()
    z = tf.random.uniform([int(1e5)], 1., 10., seed=seed_stream(), dtype=dtype)
    v = tf.random.uniform([int(1e5)], 1., 10., seed=seed_stream(), dtype=dtype)
    self.VerifyBesselKve(v, z, rtol=rtol)

  @parameterized.named_parameters(
      ("float32", np.float32, 3e-6),
      ("float64", np.float64, 1e-6),
  )
  def testBesselKveVAndZLarge(self, dtype, rtol):
    seed_stream = test_util.test_seed_stream()
    z = tf.random.uniform([int(1e5)], 10., 50., seed=seed_stream(), dtype=dtype)
    v = tf.random.uniform([int(1e5)], 10., 50., seed=seed_stream(), dtype=dtype)
    self.VerifyBesselKve(v, z, rtol=rtol)

  @parameterized.named_parameters(
      ("float32", np.float32, 7e-6),
      ("float64", np.float64, 7e-6),
  )
  def testBesselKveVLessThanZ(self, dtype, rtol):
    seed_stream = test_util.test_seed_stream()
    z = tf.random.uniform([int(1e5)], 1., 10., seed=seed_stream(), dtype=dtype)
    # Make v randomly less than z
    v = z * tf.random.uniform(
        [int(1e5)], 0.1, 0.5, seed=seed_stream(), dtype=dtype)
    self.VerifyBesselKve(v, z, rtol=rtol)

  @parameterized.named_parameters(
      ("float32", np.float32, 2e-6),
      ("float64", np.float64, 1e-6),
  )
  def testBesselKveVGreaterThanZ(self, dtype, rtol):
    seed_stream = test_util.test_seed_stream()
    v = tf.random.uniform([int(1e5)], 1., 10., seed=seed_stream(), dtype=dtype)
    # Make z randomly less than v
    z = v * tf.random.uniform(
        [int(1e5)], 0.1, 0.5, seed=seed_stream(), dtype=dtype)
    self.VerifyBesselKve(v, z, rtol=rtol)

  @parameterized.named_parameters(
      ("float32", np.float32, 4e-6),
      ("float64", np.float64, 1e-6),
  )
  def testBesselKveVNegative(self, dtype, rtol):
    seed_stream = test_util.test_seed_stream()
    v = tf.random.uniform(
        [int(1e5)], -10., -1., seed=seed_stream(), dtype=dtype)
    z = tf.random.uniform([int(1e5)], 1., 15., seed=seed_stream(), dtype=dtype)
    self.VerifyBesselKve(v, z, rtol=rtol)

  @parameterized.named_parameters(
      ("float32", np.float32, 1e-6),
      ("float64", np.float64, 1e-6),
  )
  def testBesselKveLargeZ(self, dtype, rtol):
    seed_stream = test_util.test_seed_stream()
    v = tf.random.uniform(
        [int(1e5)], minval=0., maxval=0.5, seed=seed_stream(), dtype=dtype)
    z = tf.random.uniform(
        [int(1e5)], minval=100., maxval=10000., seed=seed_stream(), dtype=dtype)
    self.VerifyBesselKve(v, z, rtol=rtol)

  @test_util.numpy_disable_gradient_test
  @parameterized.named_parameters(
      ("float32", np.float32),
      ("float64", np.float64),
  )
  def testBesselKveGradient(self, dtype):
    v = tf.constant([0.5, 1., 2., 5.])[..., tf.newaxis]
    z = tf.constant([10., 20., 30., 50., 12.,])

    err = self.compute_max_gradient_error(
        functools.partial(tfp_math.bessel_kve, v), [z])
    self.assertLess(err, 3e-4)

  @parameterized.named_parameters(
      ("float32", np.float32, 1e-6, 1e-5),
      ("float64", np.float64, 1e-6),
  )
  def testLogBesselKveCorrect(self, dtype, rtol, atol=1e-6):
    seed_stream = test_util.test_seed_stream()
    v = tf.random.uniform(
        [int(1e5)], minval=0.1, maxval=0.5, seed=seed_stream(), dtype=dtype)
    z = tf.random.uniform(
        [int(1e5)], minval=1., maxval=10., seed=seed_stream(), dtype=dtype)
    _, _, log_bessel_kve_expected_, log_bessel_kve_actual_ = self.evaluate([
        v,
        z,
        tf.math.log(tfp.math.bessel_kve(v, z)),
        tfp.math.log_bessel_kve(v, z)])
    self.assertAllClose(
        log_bessel_kve_expected_, log_bessel_kve_actual_, rtol=rtol, atol=atol)

  def testLogBesselTestNonInf(self):
    # Test that log_bessel_kve(v, z) has more resolution than simply computing
    # log(bessel_ive(v, z)). The inputs below will return inf in naive float64
    # computation.
    v = np.array([10., 12., 30., 50.], np.float32)
    z = np.logspace(-10., -1., 20).reshape((20, 1)).astype(np.float32)
    self.assertAllFinite(self.evaluate(tfp.math.log_bessel_kve(v, z)))

  @test_util.numpy_disable_gradient_test
  @parameterized.named_parameters(
      ("float32", np.float32, 1e-3),
      ("float64", np.float64, 1e-4))
  def testLogBesselKveGradient(self, dtype, tol):
    v = tf.constant([-0.2, -1., 1., 0.5, 2.], dtype=dtype)[..., tf.newaxis]
    z = tf.constant([0.3, 0.5, 0.9, 1., 12., 22.], dtype=dtype)

    err = self.compute_max_gradient_error(
        functools.partial(tfp_math.log_bessel_kve, v), [z])
    self.assertLess(err, tol)


if __name__ == "__main__":
  test_util.main()
