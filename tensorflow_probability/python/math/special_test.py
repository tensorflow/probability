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
import itertools

from absl.testing import parameterized
from mpmath import mp
import numpy as np
from scipy import special as scipy_special
import tensorflow.compat.v1 as tf1
import tensorflow.compat.v2 as tf

from tensorflow_probability.python.distributions import half_cauchy
from tensorflow_probability.python.distributions import half_normal
from tensorflow_probability.python.distributions import uniform
from tensorflow_probability.python.internal import test_util
from tensorflow_probability.python.math import gradient
from tensorflow_probability.python.math import special


def _w0(z):
  """Computes the principal branch W_0(z) of the Lambert W function."""
  # Treat -1 / exp(1) separately as special.lambertw() suffers from numerical
  # precision erros exactly at the boundary of z == exp(1)^(-1).

  if isinstance(z, float) and np.abs(z - (-1. / np.exp(1.))) < 1e-9:
    return -1.

  # This is a complex valued return value.
  return scipy_special.lambertw(z, k=0)


@test_util.test_graph_and_eager_modes
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
    y = special.round_exponential_bump_function(x)

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

    _, dy_dx = gradient.value_and_gradient(
        special.round_exponential_bump_function, x)

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
    y = special.round_exponential_bump_function(x)

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
    _, dy_dx = gradient.value_and_gradient(
        special.round_exponential_bump_function, x)

    self.assertDTypeEqual(dy_dx, dtype)

    dy_dx_ = self.evaluate(dy_dx)

    # Since x is outside the support, the gradient is zero.
    self.assertAllEqual(dy_dx_, np.zeros((6,)))


@test_util.test_graph_and_eager_modes
class BetaincTest(test_util.TestCase):

  def testBetainc(self):
    strm = test_util.test_seed_stream()
    a = half_cauchy.HalfCauchy(
        loc=np.float64(1.), scale=15.).sample(2000, strm())
    b = half_cauchy.HalfCauchy(
        loc=np.float64(1.), scale=15.).sample(2000, strm())
    x = uniform.Uniform(high=np.float64(1.)).sample(2000, strm())
    a, b, x = self.evaluate([a, b, x])

    self.assertAllClose(
        scipy_special.betainc(a, b, x),
        self.evaluate(special.betainc(a, b, x)),
        rtol=1e-6)

  def testBetaincBroadcast(self):
    a = np.ones([3, 2], dtype=np.float32)
    b = np.ones([5, 1, 1], dtype=np.float32)
    x = np.ones([7, 1, 1, 2], dtype=np.float32)
    self.assertAllEqual([7, 5, 3, 2], special.betainc(a, b, x).shape)

  def testBetaincFloat16(self):
    a = tf.constant([0.4, 0.4, 0.4, 0.4, -1., 0.4, 0.4], dtype=tf.float16)
    b = tf.constant([0.6, 0.6, 0.6, 0.6, 0.6, -1., 0.6], dtype=tf.float16)
    x = tf.constant([0.0, 0.1, 0.9, 1.0, 0.5, 0.5, -1.], dtype=tf.float16)
    result = special.betainc(a, b, x)

    self.assertEqual(a.dtype, result.dtype)

    expected_result = special.betainc(
        *[tf.cast(z, tf.float32) for z in [a, b, x]])
    expected_result = tf.cast(expected_result, a.dtype)

    self.assertAllEqual(*self.evaluate([expected_result, result]))

  @test_util.disable_test_for_backend(
      disable_numpy=True, disable_jax=False,
      reason="Numpy does not support bfloat16.")
  def testBetaincBFloat16(self):
    a = tf.constant([0.4, 0.4, 0.4, 0.4, -1., 0.4, 0.4], dtype=tf.bfloat16)
    b = tf.constant([0.6, 0.6, 0.6, 0.6, 0.6, -1., 0.6], dtype=tf.bfloat16)
    x = tf.constant([0.0, 0.1, 0.9, 1.0, 0.5, 0.5, -1.], dtype=tf.bfloat16)
    result = special.betainc(a, b, x)

    self.assertEqual(a.dtype, result.dtype)

    expected_result = special.betainc(
        *[tf.cast(z, tf.float32) for z in [a, b, x]])
    expected_result = tf.cast(expected_result, a.dtype)

    self.assertAllEqual(*self.evaluate([expected_result, result]))

  @parameterized.parameters(np.float32, np.float64)
  @test_util.numpy_disable_gradient_test
  def testBetaincGradient(self, dtype):
    space = np.logspace(-2., 2., 10).tolist()
    space_x = np.linspace(0.01, 0.99, 10).tolist()
    a, b, x = zip(*list(itertools.product(space, space, space_x)))

    a = np.array(a, dtype=dtype)
    b = np.array(b, dtype=dtype)
    x = np.array(x, dtype=dtype)

    # Wrap in tf.function and compile for faster computations.
    betainc = tf.function(special.betainc, autograph=False, jit_compile=True)

    delta = 1e-4 if dtype == np.float64 else 1e-3
    tolerance = 7e-3 if dtype == np.float64 else 7e-2
    tolerance_x = 1e-3 if dtype == np.float64 else 1e-1

    err = self.compute_max_gradient_error(
        lambda z: betainc(z, b, x), [a], delta=delta)
    self.assertLess(err, tolerance)

    err = self.compute_max_gradient_error(
        lambda z: betainc(a, z, x), [b], delta=delta)
    self.assertLess(err, tolerance)

    err = self.compute_max_gradient_error(
        lambda z: betainc(a, b, z), [x], delta=delta)
    self.assertLess(err, tolerance_x)

  @parameterized.parameters(np.float32, np.float64)
  @test_util.numpy_disable_gradient_test
  def testBetaincDerivativeFinite(self, dtype):
    eps = np.finfo(dtype).eps

    space = np.logspace(np.log10(eps), 5.).tolist()
    space_x = np.linspace(eps, 1. - eps).tolist()
    a, b, x = zip(*list(itertools.product(space, space, space_x)))

    a = np.array(a, dtype=dtype)
    b = np.array(b, dtype=dtype)
    x = np.array(x, dtype=dtype)

    def betainc_partials(a, b, x):
      return gradient.value_and_gradient(special.betainc, [a, b, x])[1]

    # Wrap in tf.function and compile for faster computations.
    betainc_partials = tf.function(
        betainc_partials, autograph=False, jit_compile=True)

    self.assertAllFinite(self.evaluate(betainc_partials(a, b, x)))

  @parameterized.parameters(np.float32, np.float64)
  @test_util.numpy_disable_gradient_test
  def testBetaincDerivativeBounds(self, dtype):

    def betainc_partials(a, b, x):
      return gradient.value_and_gradient(special.betainc, [a, b, x])[1]

    # Wrap in tf.function and compile for faster computations.
    betainc_partials = tf.function(
        betainc_partials, autograph=False, jit_compile=True)

    # Test out-of-range values (should return NaN output).
    a = np.array([-1., 0., 0.4, 0.4, 0.4, 0.4], dtype=dtype)
    b = np.array([0.6, 0.6, -1., 0., 0.6, 0.6], dtype=dtype)
    x = np.array([0.5, 0.5, 0.5, 0.5, -1., 2.], dtype=dtype)

    for partial in self.evaluate(betainc_partials(a, b, x)):
      self.assertAllNan(partial)

    # Test partials when x is equal to 0 or 1.
    a = np.array([0.4, 0.4], dtype=dtype)
    b = np.array([0.6, 0.6], dtype=dtype)
    x = np.array([0., 1.], dtype=dtype)

    partial_a, partial_b, _ = self.evaluate(betainc_partials(a, b, x))
    for partial in [partial_a, partial_b]:
      self.assertAllEqual(np.zeros_like(a), partial)

  def _testBetaincDerivative(self,
                             a,
                             b,
                             x,
                             rtol_a=1e-6,
                             atol_a=1e-6,
                             rtol_b=1e-6,
                             atol_b=1e-6,
                             rtol_x=1e-6,
                             atol_x=1e-6):

    def _mp_betainc_partial_a(a, b, x):
      return mp.diff(lambda z: mp.betainc(z, b, 0., x, regularized=True), a)

    def _mp_betainc_partial_b(a, b, x):
      return mp.diff(lambda z: mp.betainc(a, z, 0., x, regularized=True), b)

    def _mp_betainc_partial_x(a, b, x):
      return mp.diff(lambda z: mp.betainc(a, b, 0., z, regularized=True), x)

    mp_betainc_partial_a = np.frompyfunc(_mp_betainc_partial_a, 3, 1)
    mp_betainc_partial_b = np.frompyfunc(_mp_betainc_partial_b, 3, 1)
    mp_betainc_partial_x = np.frompyfunc(_mp_betainc_partial_x, 3, 1)
    mp_betainc_partials = [
        mp_betainc_partial_a, mp_betainc_partial_b, mp_betainc_partial_x]

    with mp.workdps(25):  # Set the decimal precision of mpmath.
      mp_partials = [
          mp_partial_fn(a, b, x).astype(a.dtype)
          for mp_partial_fn in mp_betainc_partials]

    def tfp_betainc_partials(a, b, x):
      return gradient.value_and_gradient(special.betainc, [a, b, x])[1]

    # Wrap in tf.function and compile for faster computations.
    tfp_betainc_partials = tf.function(
        tfp_betainc_partials, autograph=False, jit_compile=True)

    tfp_partials = self.evaluate(tfp_betainc_partials(a, b, x))

    # Check that partials preserve dtype.
    for partial in tfp_partials:
      self.assertEqual(a.dtype, partial.dtype)

    # Check that partials are accurate.
    self.assertAllClose(
        mp_partials[0], tfp_partials[0], rtol=rtol_a, atol=atol_a)
    self.assertAllClose(
        mp_partials[1], tfp_partials[1], rtol=rtol_b, atol=atol_b)
    self.assertAllClose(
        mp_partials[2], tfp_partials[2], rtol=rtol_x, atol=atol_x)

  def _test_betainc_uniform(
      self, a_high, b_high, x_high,
      rtol_a=1e-6,
      atol_a=1e-6,
      rtol_b=1e-6,
      atol_b=1e-6,
      rtol_x=1e-6,
      atol_x=1e-6):
    strm = test_util.test_seed_stream()
    a = uniform.Uniform(high=a_high).sample(50, strm())
    b = uniform.Uniform(high=b_high).sample(50, strm())
    x = uniform.Uniform(high=x_high).sample(50, strm())
    a, b, x = self.evaluate([a, b, x])

    self._testBetaincDerivative(
        a, b, x,
        rtol_a=rtol_a,
        atol_a=atol_a,
        rtol_b=rtol_b,
        atol_b=atol_b,
        rtol_x=rtol_x,
        atol_x=atol_x)

  def _test_betainc_half_normal(
      self, a_scale, b_scale, x_high,
      rtol_a=1e-6,
      atol_a=1e-6,
      rtol_b=1e-6,
      atol_b=1e-6,
      rtol_x=1e-6,
      atol_x=1e-6):
    strm = test_util.test_seed_stream()
    a = half_normal.HalfNormal(scale=a_scale).sample(50, strm())
    b = half_normal.HalfNormal(scale=b_scale).sample(50, strm())
    x = uniform.Uniform(high=x_high).sample(50, strm())
    a, b, x = self.evaluate([a, b, x])

    self._testBetaincDerivative(
        a, b, x,
        rtol_a=rtol_a,
        atol_a=atol_a,
        rtol_b=rtol_b,
        atol_b=atol_b,
        rtol_x=rtol_x,
        atol_x=atol_x)

  @test_util.numpy_disable_gradient_test
  @parameterized.named_parameters(
      {"testcase_name": "float32",
       "dtype": np.float32,
       "rtol_a": 5e-2,
       "rtol_b": 5e-2,
       "rtol_x": 1e-5},
      {"testcase_name": "float64",
       "dtype": np.float64,
       "atol_a": 1e-11,
       "atol_b": 1e-11,
       "atol_x": 1e-13})
  def testBetaincDerivativeVerySmall(self, dtype, **kwargs):
    self._test_betainc_half_normal(
        a_scale=dtype(1e-8),
        b_scale=dtype(1e-8),
        x_high=dtype(1.),
        **kwargs)

  @test_util.numpy_disable_gradient_test
  @parameterized.named_parameters(
      {"testcase_name": "float32",
       "dtype": np.float32,
       "atol_a": 1e-3,
       "atol_b": 1e-3,
       "atol_x": 0.01},
      {"testcase_name": "float64",
       "dtype": np.float64,
       "atol_a": 1e-12,
       "atol_b": 1e-12,
       "atol_x": 1e-11})
  def testBetaincDerivativeSmall(self, dtype, **kwargs):
    self._test_betainc_uniform(
        a_high=dtype(5.),
        b_high=dtype(5.),
        x_high=dtype(1.),
        **kwargs)

  @test_util.numpy_disable_gradient_test
  @parameterized.named_parameters(
      {"testcase_name": "float32",
       "dtype": np.float32,
       "atol_a": 1e-3,
       "atol_b": 1e-3,
       "atol_x": 1e-3},
      {"testcase_name": "float64",
       "dtype": np.float64,
       "atol_a": 1e-12,
       "atol_b": 1e-12,
       "atol_x": 1e-10})
  def testBetaincDerivative(self, dtype, **kwargs):
    self._test_betainc_uniform(
        a_high=dtype(100.),
        b_high=dtype(100.),
        x_high=dtype(1.),
        **kwargs)

  @parameterized.parameters(np.float32, np.float64)
  @test_util.numpy_disable_gradient_test
  def testBetaincDerivativeChallengingPoints(self, dtype):
    a = np.array(
        [0.014, 5.90, 0.536, 0.836, 0.3, 9., 1.69, 880.],
        dtype=dtype)
    b = np.array(
        [3.467, 0.01, 2.315, 0.221, 9., 0.24, 0.117, 990.],
        dtype=dtype)
    x = np.array(
        [0.007, 0.99, 0.215, 0.782, 1e-16, 1. - 1e-6, 3.4e-4, 0.47],
        dtype=dtype)

    if dtype == np.float32:
      self._testBetaincDerivative(
          a, b, x,
          atol_a=5e-4, atol_b=5e-4, atol_x=5e-4,
          rtol_a=5e-4, rtol_b=5e-4, rtol_x=5e-4)
    else:
      self._testBetaincDerivative(
          a, b, x,
          atol_a=1e-10, atol_b=1e-10, atol_x=1e-10,
          rtol_a=1e-11, rtol_b=1e-11, rtol_x=1e-11)

  @test_util.numpy_disable_gradient_test
  def testBetaincDerivativeFloat16AndBFloat16(self):
    for dtype in [tf.float16, tf.bfloat16]:
      a = tf.constant([0.4, 0.4, 0.4, 0.4, -1., 0.4, 0.4], dtype=dtype)
      b = tf.constant([0.6, 0.6, 0.6, 0.6, 0.6, -1., 0.6], dtype=dtype)
      x = tf.constant([0.0, 0.1, 0.9, 1.0, 0.5, 0.5, -1.], dtype=dtype)
      grads = gradient.value_and_gradient(special.betainc, [a, b, x])[1]

      for grad in grads:
        self.assertEqual(a.dtype, grad.dtype)

      expected_grads = gradient.value_and_gradient(
          special.betainc, *[tf.cast(z, tf.float32) for z in [a, b, x]])[1]
      expected_grads = [tf.cast(grad, a.dtype) for grad in expected_grads]

      self.assertAllEqual(*self.evaluate([expected_grads, grads]))

  @parameterized.parameters(np.float32, np.float64)
  @test_util.numpy_disable_gradient_test
  def testBetaincSecondDerivativeFinite(self, dtype):
    space = np.logspace(-2., 2., 5).tolist()
    space_x = np.linspace(0.01, 0.99, 5).tolist()
    a, b, x = zip(*list(itertools.product(space, space, space_x)))

    a = np.array(a, dtype=dtype)
    b = np.array(b, dtype=dtype)
    x = np.array(x, dtype=dtype)

    def betainc_partials(a, b, x):
      return gradient.value_and_gradient(special.betainc, [a, b, x])[1]

    def betainc_partials_of_partial_a(a, b, x):
      return gradient.value_and_gradient(
          lambda a, b, x: betainc_partials(a, b, x)[0], [a, b, x])[1]

    def betainc_partials_of_partial_b(a, b, x):
      return gradient.value_and_gradient(
          lambda a, b, x: betainc_partials(a, b, x)[1], [a, b, x])[1]

    def betainc_partials_of_partial_x(a, b, x):
      return gradient.value_and_gradient(
          lambda a, b, x: betainc_partials(a, b, x)[2], [a, b, x])[1]

    betainc_partials_of_partials = [
        betainc_partials_of_partial_a,
        betainc_partials_of_partial_b,
        betainc_partials_of_partial_x]

    # Wrap in tf.function and compile for faster computations.
    betainc_partials_of_partials = [
        tf.function(partial_fn, autograph=False, jit_compile=True)
        for partial_fn in betainc_partials_of_partials]

    partials_of_partials = [
        partial_fn(a, b, x) for partial_fn in betainc_partials_of_partials]

    self.assertAllFinite(self.evaluate(partials_of_partials))


@test_util.test_graph_and_eager_modes
class BetaincinvTest(test_util.TestCase):

  def testBetaincinvBroadcast(self):
    a = np.ones([3, 2], dtype=np.float32)
    b = np.ones([5, 1, 1], dtype=np.float32)
    y = np.ones([7, 1, 1, 2], dtype=np.float32)
    self.assertAllEqual([7, 5, 3, 2], special.betaincinv(a, b, y).shape)

  def _test_betaincinv_value(self, a_high, b_high, dtype, atol, rtol):
    tiny = np.finfo(dtype).tiny
    n = [int(5e3)]
    strm = test_util.test_seed_stream()
    a = uniform.Uniform(low=tiny, high=dtype(a_high)).sample(n, strm())
    b = uniform.Uniform(low=tiny, high=dtype(b_high)).sample(n, strm())
    y = uniform.Uniform(low=tiny, high=dtype(1.)).sample(n, strm())

    # Wrap in tf.function and compile for faster computations.
    betaincinv = tf.function(special.betaincinv, autograph=False)

    result, a, b, y = self.evaluate(
        [betaincinv(a, b, y), a, b, y])

    # Check that special.betaincinv preserves dtype.
    self.assertEqual(dtype, result.dtype)

    # Check that special.betaincinv is accurate.
    self.assertAllClose(
        scipy_special.betaincinv(a, b, y), result, atol=atol, rtol=rtol)

  @parameterized.named_parameters(
      {"testcase_name": "float32",
       "dtype": np.float32,
       "atol": 1e-6,
       "rtol": 2e-3},
      {"testcase_name": "float64",
       "dtype": np.float64,
       "atol": 1e-12,
       "rtol": 1e-11})
  def testBetaincinvSmall(self, dtype, atol, rtol):
    self._test_betaincinv_value(
        a_high=1., b_high=1., dtype=dtype, atol=atol, rtol=rtol)

  @parameterized.named_parameters(
      {"testcase_name": "float32",
       "dtype": np.float32,
       "atol": 1e-6,
       "rtol": 6e-4},
      {"testcase_name": "float64",
       "dtype": np.float64,
       "atol": 1e-12,
       "rtol": 0.})
  def testBetaincinvMedium(self, dtype, atol, rtol):
    self._test_betaincinv_value(
        a_high=100., b_high=100., dtype=dtype, atol=atol, rtol=rtol)

  @parameterized.named_parameters(
      {"testcase_name": "float32",
       "dtype": np.float32,
       "atol": 1e-5,
       "rtol": 2e-3},
      {"testcase_name": "float64",
       "dtype": np.float64,
       "atol": 1e-12,
       "rtol": 0.})
  def testBetaincinvLarge(self, dtype, atol, rtol):
    self._test_betaincinv_value(
        a_high=1e4, b_high=1e4, dtype=dtype, atol=atol, rtol=rtol)

  @parameterized.parameters(np.float32, np.float64)
  def testBetaincinvBounds(self, dtype):
    # Test out-of-range values (should return NaN output).
    a = np.array([-1., 0., 0.4, 0.4, 0.4, 0.4], dtype=dtype)
    b = np.array([0.6, 0.6, -1., 0., 0.6, 0.6], dtype=dtype)
    y = np.array([0.5, 0.5, 0.5, 0.5, -1., 2.], dtype=dtype)

    # Wrap in tf.function and compile for faster computations.
    betaincinv = tf.function(special.betaincinv, autograph=False)

    result = self.evaluate(special.betaincinv(a, b, y))
    self.assertEqual(dtype, result.dtype)
    self.assertAllNan(result)

    # Test tfp_math.betaincinv when y is equal to 0 or 1.
    a = dtype(0.4)
    b = dtype(0.6)
    y = np.array([0., 1.], dtype=dtype)
    result = self.evaluate(betaincinv(a, b, y))
    self.assertEqual(dtype, result.dtype)
    self.assertAllEqual(y, result)

  def testBetaincinvFloat16(self):
    a = tf.constant([0.4, 0.4, 0.4, 0.4, -1., 0.4, 0.4], dtype=tf.float16)
    b = tf.constant([0.6, 0.6, 0.6, 0.6, 0.6, -1., 0.6], dtype=tf.float16)
    y = tf.constant([0.0, 0.1, 0.9, 1.0, 0.5, 0.5, -1.], dtype=tf.float16)
    result = special.betaincinv(a, b, y)

    self.assertEqual(a.dtype, result.dtype)

    expected_result = special.betaincinv(
        *[tf.cast(z, tf.float32) for z in [a, b, y]])
    expected_result = tf.cast(expected_result, a.dtype)

    self.assertAllEqual(*self.evaluate([expected_result, result]))

  @test_util.disable_test_for_backend(
      disable_numpy=True, disable_jax=False,
      reason="Numpy does not support bfloat16.")
  def testBetaincinvBFloat16(self):
    a = tf.constant([0.4, 0.4, 0.4, 0.4, -1., 0.4, 0.4], dtype=tf.bfloat16)
    b = tf.constant([0.6, 0.6, 0.6, 0.6, 0.6, -1., 0.6], dtype=tf.bfloat16)
    y = tf.constant([0.0, 0.1, 0.9, 1.0, 0.5, 0.5, -1.], dtype=tf.bfloat16)
    result = special.betaincinv(a, b, y)

    self.assertEqual(a.dtype, result.dtype)

    expected_result = special.betaincinv(
        *[tf.cast(z, tf.float32) for z in [a, b, y]])
    expected_result = tf.cast(expected_result, a.dtype)

    self.assertAllEqual(*self.evaluate([expected_result, result]))

  @test_util.numpy_disable_gradient_test
  def testBetaincinvGradient(self):
    # Avoid small values for parameters a and b where the gradient can veer off
    # to infinity.
    space = np.logspace(np.log10(0.5), 3., 5).tolist()
    space_y = np.linspace(0.01, 0.99, 10).tolist()
    a, b, y = [
        tf.constant(z, dtype="float64")
        for z in zip(*list(itertools.product(space, space, space_y)))]

    # Wrap in tf.function for faster computations.
    betaincinv = tf.function(special.betaincinv, autograph=False)

    err = self.compute_max_gradient_error(
        lambda z: betaincinv(z, b, y), [a], delta=1e-7)
    self.assertLess(err, 2e-7)

    err = self.compute_max_gradient_error(
        lambda z: betaincinv(a, z, y), [b], delta=1e-7)
    self.assertLess(err, 2e-7)

    err = self.compute_max_gradient_error(
        lambda z: betaincinv(a, b, z), [y], delta=1e-7)
    self.assertLess(err, 5e-8)

  @parameterized.parameters(np.float32, np.float64)
  @test_util.numpy_disable_gradient_test
  def testBetaincinvGradientFinite(self, dtype):
    eps = np.finfo(dtype).eps
    small = np.sqrt(eps)

    space = np.logspace(np.log10(small), 4.).tolist()
    space_y = np.linspace(eps, 1. - small).tolist()
    a, b, y = [
        tf.constant(z, dtype=dtype)
        for z in zip(*list(itertools.product(space, space, space_y)))]

    def betaincinv_partials(a, b, y):
      return gradient.value_and_gradient(special.betaincinv, [a, b, y])[1]

    # Wrap in tf.function for faster computations.
    betaincinv_partials = tf.function(betaincinv_partials, autograph=False)

    self.assertAllFinite(self.evaluate(betaincinv_partials(a, b, y)))

  @test_util.numpy_disable_gradient_test
  def testBetaincinvGradientBroadcast(self):
    a = 0.5 * np.ones([3, 2], dtype=np.float32)
    b = 0.5 * np.ones([5, 1, 1], dtype=np.float32)
    y = 0.5 * np.ones([7, 1, 1, 2], dtype=np.float32)

    def simple_ternary_operator(a, b, y):
      return a + b + y

    _, simple_partials = gradient.value_and_gradient(simple_ternary_operator,
                                                     [a, b, y])
    _, betaincinv_partials = gradient.value_and_gradient(
        special.betaincinv, [a, b, y])
    a_partials, b_partials, y_partials = zip(
        [*simple_partials], [*betaincinv_partials])
    self.assertAllEqual(a_partials[0].shape, a_partials[1].shape)
    self.assertAllEqual(b_partials[0].shape, b_partials[1].shape)
    self.assertAllEqual(y_partials[0].shape, y_partials[1].shape)

  @test_util.numpy_disable_gradient_test
  def testBetaincinvGradientBFloat16(self):
    dtype = tf.bfloat16
    a = tf.constant([0.4, 0.4, 0.4, 0.4, -1., 0.4, 0.4], dtype=dtype)
    b = tf.constant([0.6, 0.6, 0.6, 0.6, 0.6, -1., 0.6], dtype=dtype)
    y = tf.constant([0.0, 0.1, 0.9, 1.0, 0.5, 0.5, -1.], dtype=dtype)
    grads = gradient.value_and_gradient(special.betaincinv, [a, b, y])[1]

    self.assertEqual(a.dtype, grads[0].dtype)
    self.assertEqual(b.dtype, grads[0].dtype)
    self.assertEqual(y.dtype, grads[0].dtype)

    expected_grads = gradient.value_and_gradient(
        special.betaincinv, *[tf.cast(z, tf.float32) for z in [a, b, y]])[1]
    expected_grads = [tf.cast(grad, a.dtype) for grad in expected_grads]

    self.assertAllEqual(*self.evaluate([expected_grads, grads]))

  @test_util.numpy_disable_gradient_test
  def testBetaincinvGradientFloat16(self):
    dtype = tf.float16
    a = tf.constant([0.4, 0.4, 0.4, 0.4, -1., 0.4, 0.4], dtype=dtype)
    b = tf.constant([0.6, 0.6, 0.6, 0.6, 0.6, -1., 0.6], dtype=dtype)
    y = tf.constant([0.0, 0.1, 0.9, 1.0, 0.5, 0.5, -1.], dtype=dtype)
    grads = gradient.value_and_gradient(special.betaincinv, [a, b, y])[1]

    self.assertEqual(a.dtype, grads[0].dtype)
    self.assertEqual(b.dtype, grads[0].dtype)
    self.assertEqual(y.dtype, grads[0].dtype)

    expected_grads = gradient.value_and_gradient(
        special.betaincinv, *[tf.cast(z, tf.float32) for z in [a, b, y]])[1]
    expected_grads = [tf.cast(grad, a.dtype) for grad in expected_grads]

    self.assertAllEqual(*self.evaluate([expected_grads, grads]))

  @parameterized.parameters(np.float32, np.float64)
  @test_util.numpy_disable_gradient_test
  def testBetaincinvSecondDerivativeFinite(self, dtype):
    # Avoid small values for parameters a and b where the gradient can veer off
    # to infinity.
    space = np.logspace(np.log10(0.5), 3., 5).tolist()
    space_y = np.linspace(0.01, 0.99, 5).tolist()
    a, b, y = [
        tf.constant(z, dtype=dtype)
        for z in zip(*list(itertools.product(space, space, space_y)))]

    def betaincinv_partials(a, b, y):
      return gradient.value_and_gradient(special.betaincinv, [a, b, y])[1]

    def betaincinv_partials_of_partial_a(a, b, y):
      return gradient.value_and_gradient(
          lambda a, b, y: betaincinv_partials(a, b, y)[0], [a, b, y])[1]

    def betaincinv_partials_of_partial_b(a, b, y):
      return gradient.value_and_gradient(
          lambda a, b, y: betaincinv_partials(a, b, y)[1], [a, b, y])[1]

    def betaincinv_partials_of_partial_y(a, b, y):
      return gradient.value_and_gradient(
          lambda a, b, y: betaincinv_partials(a, b, y)[2], [a, b, y])[1]

    betaincinv_partials_of_partials = [
        betaincinv_partials_of_partial_a,
        betaincinv_partials_of_partial_b,
        betaincinv_partials_of_partial_y]

    # Wrap in tf.function for faster computations.
    betaincinv_partials_of_partials = [
        tf.function(partial_fn, autograph=False)
        for partial_fn in betaincinv_partials_of_partials]

    partials_of_partials = [
        partial_fn(a, b, y) for partial_fn in betaincinv_partials_of_partials]

    self.assertAllFinite(self.evaluate(partials_of_partials))


@test_util.test_graph_and_eager_modes
class DawsnTest(test_util.TestCase):

  def testDawsnBoundary(self):
    self.assertAllClose(0., special.dawsn(0.))
    self.assertTrue(np.isnan(self.evaluate(special.dawsn(np.nan))))

  @parameterized.parameters(np.float32, np.float64)
  def testDawsnOdd(self, dtype):
    seed_stream = test_util.test_seed_stream()
    x = self.evaluate(
        tf.random.uniform(
            [int(1e4)], 0., 100., dtype=dtype, seed=seed_stream()))
    self.assertAllClose(
        self.evaluate(special.dawsn(x)), self.evaluate(-special.dawsn(-x)))

  @parameterized.parameters(np.float32, np.float64)
  def testDawsnSmall(self, dtype):
    seed_stream = test_util.test_seed_stream()
    x = self.evaluate(
        tf.random.uniform(
            [int(1e4)], 0., 1., dtype=dtype, seed=seed_stream()))
    self.assertAllClose(scipy_special.dawsn(x), self.evaluate(special.dawsn(x)))

  @parameterized.parameters(np.float32, np.float64)
  def testDawsnMedium(self, dtype):
    seed_stream = test_util.test_seed_stream()
    x = self.evaluate(
        tf.random.uniform([int(1e4)], 1., 10., dtype=dtype, seed=seed_stream()))
    self.assertAllClose(scipy_special.dawsn(x), self.evaluate(special.dawsn(x)))

  @parameterized.parameters(np.float32, np.float64)
  def testDawsnLarge(self, dtype):
    seed_stream = test_util.test_seed_stream()
    x = self.evaluate(tf.random.uniform(
        [int(1e4)], 10., 100., dtype=dtype, seed=seed_stream()))
    self.assertAllClose(scipy_special.dawsn(x), self.evaluate(special.dawsn(x)))

  @test_util.numpy_disable_gradient_test
  def testDawsnGradient(self):
    x = np.linspace(0.1, 100., 50)
    err = self.compute_max_gradient_error(special.dawsn, [x])
    self.assertLess(err, 2e-5)

  @test_util.numpy_disable_gradient_test
  def testDawsnSecondDerivative(self):
    x = np.linspace(0.1, 100., 50)
    err = self.compute_max_gradient_error(
        lambda z: gradient.value_and_gradient(special.dawsn, z)[1], [x])
    self.assertLess(err, 2e-5)


@test_util.test_graph_and_eager_modes
class IgammainvTest(test_util.TestCase):

  def test_igammainv_bounds(self):
    a = [-1., -4., 0.1, 2.]
    p = [0.2, 0.3, -1., 10.]
    # Out of bounds.
    self.assertAllClose(np.full_like(a, np.nan), special.igammainv(a, p))
    self.assertAllClose(np.full_like(a, np.nan), special.igammacinv(a, p))

    a = np.random.uniform(1., 5., size=4)

    self.assertAllClose(np.zeros_like(a), special.igammainv(a, 0.))
    self.assertAllClose(np.zeros_like(a), special.igammacinv(a, 1.))

    self.assertAllClose(np.full_like(a, np.inf), special.igammainv(a, 1.))
    self.assertAllClose(np.full_like(a, np.inf), special.igammacinv(a, 0.))
    self.assertTrue(np.isnan(self.evaluate(special.igammainv(np.nan, np.nan))))
    self.assertTrue(np.isnan(self.evaluate(special.igammacinv(np.nan, np.nan))))

  @parameterized.parameters((np.float32, 1.5e-4), (np.float64, 1e-6))
  def test_igammainv_inverse_small_a(self, dtype, rtol):
    seed_stream = test_util.test_seed_stream()
    a = tf.random.uniform([int(1e4)], 0., 1., dtype=dtype, seed=seed_stream())
    p = tf.random.uniform([int(1e4)], 0., 1., dtype=dtype, seed=seed_stream())
    igammainv, a, p = self.evaluate([special.igammainv(a, p), a, p])
    self.assertAllClose(scipy_special.gammaincinv(a, p), igammainv, rtol=rtol)

  @parameterized.parameters((np.float32, 1.5e-4), (np.float64, 1e-6))
  def test_igammacinv_inverse_small_a(self, dtype, rtol):
    seed_stream = test_util.test_seed_stream()
    a = tf.random.uniform([int(1e4)], 0., 1., dtype=dtype, seed=seed_stream())
    p = tf.random.uniform([int(1e4)], 0., 1., dtype=dtype, seed=seed_stream())
    igammacinv, a, p = self.evaluate([special.igammacinv(a, p), a, p])
    self.assertAllClose(scipy_special.gammainccinv(a, p), igammacinv, rtol=rtol)

  @parameterized.parameters((np.float32, 1e-4), (np.float64, 1e-6))
  def test_igammainv_inverse_medium_a(self, dtype, rtol):
    seed_stream = test_util.test_seed_stream()
    a = tf.random.uniform([int(1e4)], 1., 100., dtype=dtype, seed=seed_stream())
    p = tf.random.uniform([int(1e4)], 0., 1., dtype=dtype, seed=seed_stream())
    igammainv, a, p = self.evaluate([special.igammainv(a, p), a, p])
    self.assertAllClose(scipy_special.gammaincinv(a, p), igammainv, rtol=rtol)

  @parameterized.parameters((np.float32, 1e-4), (np.float64, 1e-6))
  def test_igammacinv_inverse_medium_a(self, dtype, rtol):
    seed_stream = test_util.test_seed_stream()
    a = tf.random.uniform([int(1e4)], 1., 100., dtype=dtype, seed=seed_stream())
    p = tf.random.uniform([int(1e4)], 0., 1., dtype=dtype, seed=seed_stream())
    igammacinv, a, p = self.evaluate([special.igammacinv(a, p), a, p])
    self.assertAllClose(scipy_special.gammainccinv(a, p), igammacinv, rtol=rtol)

  @parameterized.parameters((np.float32, 3e-4), (np.float64, 1e-6))
  def test_igammainv_inverse_large_a(self, dtype, rtol):
    seed_stream = test_util.test_seed_stream()
    a = tf.random.uniform(
        [int(1e4)], 100., 10000., dtype=dtype, seed=seed_stream())
    p = tf.random.uniform([int(1e4)], 0., 1., dtype=dtype, seed=seed_stream())
    igammainv, a, p = self.evaluate([special.igammainv(a, p), a, p])
    self.assertAllClose(scipy_special.gammaincinv(a, p), igammainv, rtol=rtol)

  @parameterized.parameters((np.float32, 3e-4), (np.float64, 1e-6))
  def test_igammacinv_inverse_large_a(self, dtype, rtol):
    seed_stream = test_util.test_seed_stream()
    a = tf.random.uniform(
        [int(1e4)], 100., 10000., dtype=dtype, seed=seed_stream())
    p = tf.random.uniform([int(1e4)], 0., 1., dtype=dtype, seed=seed_stream())
    igammacinv, a, p = self.evaluate([special.igammacinv(a, p), a, p])
    self.assertAllClose(scipy_special.gammainccinv(a, p), igammacinv, rtol=rtol)

  @test_util.numpy_disable_gradient_test
  def testIgammainvGradient(self):
    a = np.logspace(-2., 2., 11)[..., np.newaxis]
    # Avoid the end points where the gradient can veer off to infinity.
    p = np.linspace(0.1, 0.7, 23)

    # Wrap in tf.function for faster computations.
    igammainv = tf.function(special.igammainv)

    err = self.compute_max_gradient_error(
        lambda x: igammainv(a, x), [p], delta=1e-4)
    self.assertLess(err, 2e-5)

    err = self.compute_max_gradient_error(
        lambda x: igammainv(x, p), [a], delta=1e-4)
    self.assertLess(err, 2e-5)

  @test_util.numpy_disable_gradient_test
  def testIgammacinvGradient(self):
    a = np.logspace(-2., 2., 11)[..., np.newaxis]
    # Avoid the end points where the gradient can veer off to infinity.
    p = np.linspace(0.1, 0.7, 23)

    # Wrap in tf.function for faster computations.
    igammacinv = tf.function(special.igammacinv)

    err = self.compute_max_gradient_error(
        lambda x: igammacinv(a, x), [p], delta=1e-4)
    self.assertLess(err, 2e-5)

    err = self.compute_max_gradient_error(
        lambda x: igammacinv(x, p), [a], delta=1e-4)
    self.assertLess(err, 2e-5)

  @test_util.numpy_disable_gradient_test
  def testIgammainvSecondDerivativeNotNaN(self):
    a = tf.constant(np.linspace(2., 5., 11)[..., np.newaxis])
    # Avoid the end points where the gradient can veer off to infinity.
    p = tf.constant(np.linspace(0.1, 0.7, 23))

    def igammainv_first_der(a, p):
      return gradient.value_and_gradient(lambda z: special.igammainv(a, z),
                                         p)[1]
    err = self.compute_max_gradient_error(
        lambda x: igammainv_first_der(a, x), [p], delta=1e-4)
    self.assertLess(err, 1e-3)

    err = self.compute_max_gradient_error(
        lambda y: igammainv_first_der(y, p), [a], delta=1e-4)
    self.assertLess(err, 1e-3)

    def igammacinv_first_der(a, p):
      return gradient.value_and_gradient(lambda z: special.igammacinv(a, z),
                                         p)[1]

    err = self.compute_max_gradient_error(
        lambda x: igammacinv_first_der(a, x), [p], delta=1e-4)
    self.assertLess(err, 2e-3)

    err = self.compute_max_gradient_error(
        lambda y: igammacinv_first_der(y, p), [a], delta=1e-4)
    self.assertLess(err, 2e-3)


@test_util.test_all_tf_execution_regimes
class OwensTTest(test_util.TestCase):

  @parameterized.parameters(np.float32, np.float64)
  def testOwensTOddEven(self, dtype):
    seed_stream = test_util.test_seed_stream()
    a = self.evaluate(
        tf.random.uniform(
            shape=[int(1e3)],
            minval=0.,
            maxval=100.,
            dtype=dtype,
            seed=seed_stream()))
    h = self.evaluate(
        tf.random.uniform(
            shape=[int(1e3)],
            minval=0.,
            maxval=100.,
            dtype=dtype,
            seed=seed_stream()))
    # OwensT(h, a) = OwensT(-h, a)
    self.assertAllClose(
        self.evaluate(special.owens_t(h, a)),
        self.evaluate(special.owens_t(-h, a)),
    )
    # OwensT(h, a) = -OwensT(h, -a)
    self.assertAllClose(
        self.evaluate(special.owens_t(h, a)),
        self.evaluate(-special.owens_t(h, -a)),
    )

  @parameterized.parameters(np.float32, np.float64)
  def testOwensTSmall(self, dtype):
    seed_stream = test_util.test_seed_stream()
    a = tf.random.uniform(
        shape=[int(1e4)],
        minval=0.,
        maxval=1.,
        dtype=dtype,
        seed=seed_stream())
    h = tf.random.uniform(
        shape=[int(1e4)],
        minval=0.,
        maxval=1.,
        dtype=dtype,
        seed=seed_stream())
    a_, h_, owens_t_ = self.evaluate([a, h, special.owens_t(h, a)])
    self.assertAllClose(scipy_special.owens_t(h_, a_), owens_t_)

  @parameterized.parameters(np.float32, np.float64)
  def testOwensTLarger(self, dtype):
    seed_stream = test_util.test_seed_stream()
    a = tf.random.uniform(
        shape=[int(1e4)],
        minval=1.,
        maxval=100.,
        dtype=dtype,
        seed=seed_stream())
    h = tf.random.uniform(
        shape=[int(1e4)],
        minval=1.,
        maxval=100.,
        dtype=dtype,
        seed=seed_stream())
    a_, h_, owens_t_ = self.evaluate([a, h, special.owens_t(h, a)])
    self.assertAllClose(scipy_special.owens_t(h_, a_), owens_t_)

  @parameterized.parameters(np.float32, np.float64)
  def testOwensTLarge(self, dtype):
    seed_stream = test_util.test_seed_stream()
    a = tf.random.uniform(
        shape=[int(1e4)],
        minval=100.,
        maxval=1000.,
        dtype=dtype,
        seed=seed_stream())
    h = tf.random.uniform(
        shape=[int(1e4)],
        minval=100.,
        maxval=1000.,
        dtype=dtype,
        seed=seed_stream())
    a_, h_, owens_t_ = self.evaluate([a, h, special.owens_t(h, a)])
    self.assertAllClose(scipy_special.owens_t(h_, a_), owens_t_)

  @test_util.numpy_disable_gradient_test
  def testOwensTGradient(self):
    h = tf.constant([0.01, 0.1, 0.5, 1., 10.])[..., tf.newaxis]
    a = tf.constant([0.01, 0.1, 0.5, 1., 10.])

    err = self.compute_max_gradient_error(
        functools.partial(special.owens_t, h), [a])
    self.assertLess(err, 2e-4)

    err = self.compute_max_gradient_error(lambda x: special.owens_t(x, a), [h])
    self.assertLess(err, 2e-4)

    err = self.compute_max_gradient_error(
        functools.partial(special.owens_t, -h), [a])
    self.assertLess(err, 2e-4)

    err = self.compute_max_gradient_error(lambda x: special.owens_t(x, -a), [h])
    self.assertLess(err, 2e-4)

  @test_util.disable_test_for_backend(
      disable_numpy=True, disable_jax=True,
      reason="Only relevant for dynamic shapes in TF.")
  def testOwensPartiallyKnownShape(self):
    h = tf1.placeholder_with_default(np.array([1.]).reshape([1, 1]),
                                     shape=(None, 1))
    a = tf1.placeholder_with_default(np.array([1.]).reshape([1, 1]),
                                     shape=(None, 1))
    # We simply verify that this runs without an Exception.
    _ = special.owens_t(h, a)


@test_util.test_graph_and_eager_modes
class SpecialTest(test_util.TestCase):

  @parameterized.parameters(np.float32, np.float64)
  def testAtanDifferenceSmall(self, dtype):
    seed_stream = test_util.test_seed_stream()
    x = tf.random.uniform(
        shape=[int(1e5)],
        minval=-10.,
        maxval=10.,
        dtype=dtype,
        seed=seed_stream())
    y = tf.random.uniform(
        shape=[int(1e5)],
        minval=-10.,
        maxval=10.,
        dtype=dtype,
        seed=seed_stream())

    x_, y_, atan_diff_ = self.evaluate([x, y, special.atan_difference(x, y)])
    self.assertAllClose(np.arctan(x_) - np.arctan(y_), atan_diff_)

  @parameterized.parameters(np.float32, np.float64)
  def testAtanDifferenceLarge(self, dtype):
    seed_stream = test_util.test_seed_stream()
    x = tf.random.uniform(
        shape=[int(1e5)],
        minval=-100.,
        maxval=100.,
        dtype=dtype,
        seed=seed_stream())
    y = tf.random.uniform(
        shape=[int(1e5)],
        minval=-100.,
        maxval=100.,
        dtype=dtype,
        seed=seed_stream())

    x_, y_, atan_diff_ = self.evaluate([x, y, special.atan_difference(x, y)])
    self.assertAllClose(np.arctan(x_) - np.arctan(y_), atan_diff_)

  @parameterized.parameters(np.float32, np.float64)
  def testAtanDifferenceCloseInputs(self, dtype):
    y = np.linspace(1e4, 1e5, 100).astype(dtype)
    x = y + 1.

    atan_diff_ = self.evaluate(special.atan_difference(x, y))
    # Ensure there isn't cancellation for large values.
    self.assertAllGreater(atan_diff_, 0.)

  @parameterized.parameters(np.float32, np.float64)
  def testAtanDifferenceProductIsNegativeOne(self, dtype):
    seed_stream = test_util.test_seed_stream()
    x = tf.random.uniform(
        shape=[int(1e5)],
        minval=-10.,
        maxval=10.,
        dtype=dtype,
        seed=seed_stream())
    y = -tf.math.reciprocal(x)

    x_, y_, atan_diff_ = self.evaluate([x, y, special.atan_difference(x, y)])
    self.assertAllClose(np.arctan(x_) - np.arctan(y_), atan_diff_)

  @parameterized.parameters(np.float32, np.float64)
  def testErfcinvPreservesDtype(self, dtype):
    x = self.evaluate(
        tf.random.uniform(
            shape=[int(1e5)],
            minval=0.,
            maxval=1.,
            dtype=dtype,
            seed=test_util.test_seed()))
    self.assertEqual(x.dtype, special.erfcinv(x).dtype)

  def testErfcinv(self):
    x = self.evaluate(
        tf.random.uniform(
            shape=[int(1e5)],
            minval=0.,
            maxval=1.,
            seed=test_util.test_seed()))
    erfcinv = special.erfcinv(x)
    x_prime = tf.math.erfc(erfcinv)
    x_prime, erfcinv = self.evaluate([x_prime, erfcinv])

    self.assertFalse(np.all(np.isnan(erfcinv)))
    self.assertGreaterEqual(np.min(erfcinv), 0.)
    # Check that erfc(erfcinv(x)) = x.
    self.assertAllClose(x_prime, x, rtol=1e-6)

  @parameterized.parameters(tf.float32, tf.float64)
  def testErfcxSmall(self, dtype):
    x = tf.random.uniform(
        shape=[int(1e5)],
        minval=0.,
        maxval=1.,
        dtype=dtype,
        seed=test_util.test_seed())

    x_, erfcx_ = self.evaluate([x, special.erfcx(x)])
    self.assertAllClose(scipy_special.erfcx(x_), erfcx_)

  @parameterized.parameters(tf.float32, tf.float64)
  def testErfcxMedium(self, dtype):
    x = tf.random.uniform(
        shape=[int(1e5)],
        minval=1.,
        maxval=20.,
        dtype=dtype,
        seed=test_util.test_seed())

    x_, erfcx_ = self.evaluate([x, special.erfcx(x)])
    self.assertAllClose(scipy_special.erfcx(x_), erfcx_)

  @parameterized.parameters(tf.float32, tf.float64)
  def testErfcxLarge(self, dtype):
    x = tf.random.uniform(
        shape=[int(1e5)],
        minval=20.,
        maxval=100.,
        dtype=dtype,
        seed=test_util.test_seed())

    x_, erfcx_ = self.evaluate([x, special.erfcx(x)])
    self.assertAllClose(scipy_special.erfcx(x_), erfcx_)

  @parameterized.parameters(tf.float32, tf.float64)
  def testErfcxSmallNegative(self, dtype):
    x = tf.random.uniform(
        shape=[int(1e5)],
        minval=-1.,
        maxval=0.,
        dtype=dtype,
        seed=test_util.test_seed())

    x_, erfcx_ = self.evaluate([x, special.erfcx(x)])
    self.assertAllClose(scipy_special.erfcx(x_), erfcx_)

  @parameterized.parameters(tf.float32, tf.float64)
  def testErfcxMediumNegative(self, dtype):
    x = tf.random.uniform(
        shape=[int(1e5)],
        minval=-20.,
        maxval=-1.,
        dtype=dtype,
        seed=test_util.test_seed())

    x_, erfcx_ = self.evaluate([x, special.erfcx(x)])
    self.assertAllClose(scipy_special.erfcx(x_), erfcx_, rtol=4.5e-6)

  @parameterized.parameters(tf.float32, tf.float64)
  def testErfcxLargeNegative(self, dtype):
    x = tf.random.uniform(
        shape=[int(1e5)],
        minval=-100.,
        maxval=-20.,
        dtype=dtype,
        seed=test_util.test_seed())

    x_, erfcx_ = self.evaluate([x, special.erfcx(x)])
    self.assertAllClose(scipy_special.erfcx(x_), erfcx_)

  @test_util.numpy_disable_gradient_test
  def testErfcxGradient(self):
    x = np.linspace(-1., 3., 20).astype(np.float32)
    err = self.compute_max_gradient_error(special.erfcx, [x])
    self.assertLess(err, 2.1e-4)

  @test_util.numpy_disable_gradient_test
  def testErfcxSecondDerivative(self):
    x = np.linspace(-1., 3., 20).astype(np.float32)
    err = self.compute_max_gradient_error(
        lambda z: gradient.value_and_gradient(special.erfcx, z)[1], [x])
    self.assertLess(err, 1e-3)

  @parameterized.parameters(tf.float32, tf.float64)
  def testLogErfc(self, dtype):
    x = tf.random.uniform(
        shape=[int(1e5)],
        minval=-3.,
        maxval=3.,
        dtype=dtype,
        seed=test_util.test_seed())

    x_, logerfc_ = self.evaluate([x, special.logerfc(x)])
    self.assertAllClose(np.log(scipy_special.erfc(x_)), logerfc_)

  @parameterized.parameters(tf.float32, tf.float64)
  @test_util.numpy_disable_gradient_test
  def testLogErfcValueAndGradientNoNaN(self, dtype):
    x = tf.constant(np.logspace(1., 10., 40), dtype=dtype)
    logerfc_, grad_logerfc_ = self.evaluate(
        gradient.value_and_gradient(special.logerfc, x))
    self.assertAllNotNan(logerfc_)
    self.assertAllNotNan(grad_logerfc_)

    logerfc_, grad_logerfc_ = self.evaluate(
        gradient.value_and_gradient(special.logerfc, -x))
    self.assertAllNotNan(logerfc_)
    self.assertAllNotNan(grad_logerfc_)

  @parameterized.parameters(tf.float32, tf.float64)
  def testLogErfcx(self, dtype):
    x = tf.random.uniform(
        shape=[int(1e5)],
        minval=-3.,
        maxval=3.,
        dtype=dtype,
        seed=test_util.test_seed())

    x_, logerfcx_ = self.evaluate([x, special.logerfcx(x)])
    self.assertAllClose(np.log(scipy_special.erfcx(x_)), logerfcx_)

  @parameterized.parameters(tf.float32, tf.float64)
  @test_util.numpy_disable_gradient_test
  def testLogErfcxValueAndGradientNoNaN(self, dtype):
    x = tf.constant(np.logspace(1., 10., 40), dtype=dtype)
    logerfcx_, grad_logerfcx_ = self.evaluate(
        gradient.value_and_gradient(special.logerfcx, x))
    self.assertAllNotNan(logerfcx_)
    self.assertAllNotNan(grad_logerfcx_)

    logerfcx_, grad_logerfcx_ = self.evaluate(
        gradient.value_and_gradient(special.logerfcx, -x))
    self.assertAllNotNan(logerfcx_)
    self.assertAllNotNan(grad_logerfcx_)

  @parameterized.parameters(tf.float32, tf.float64)
  def testLogErfcxAtZero(self, dtype):
    x = tf.constant(0., dtype=dtype)
    logerfcx_, logerfc_ = self.evaluate(
        [special.logerfcx(x), special.logerfc(x)])
    self.assertAllClose(np.log(scipy_special.erfc(0.)), logerfc_)
    self.assertAllClose(np.log(scipy_special.erfcx(0.)), logerfcx_)

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
    self.assertAllClose(special.lambertw(value), expected)
    self.assertAllClose(special.lambertw(value), scipy_wz)

  @parameterized.named_parameters(
      ("1D_array", np.array([1.0, 0.0])),
      ("2D_array", np.array([[1.0, 1.0], [0.0, 2.0]]))
      )
  def testLambertWWorksElementWise(self, value):
    """Tests the LambertW function works with multidimensional arrays."""
    scipy_wz = _w0(value)
    wz = special.lambertw(value)
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
  def testLambertWApproximation(self, value):
    """Tests the approximation of the LambertW function."""
    exact = _w0(value)
    value = tf.convert_to_tensor(value)
    approx = special.lambertw_winitzki_approx(value)
    self.assertAllClose(approx, exact, rtol=0.05)

  @parameterized.named_parameters(("0", 0., 1.),
                                  ("exp(1)", np.exp(1.), 1. / (np.exp(1.) * 2)))
  @test_util.numpy_disable_gradient_test
  def testLambertWGradient(self, value, expected):
    """Tests the gradient of the LambertW function on some known identities."""
    x = tf.constant(value, dtype=tf.float64)
    _, dy_dx = gradient.value_and_gradient(special.lambertw, x)
    self.assertAllClose(dy_dx, expected)

  def testLogGammaCorrection(self):
    x = half_cauchy.HalfCauchy(
        loc=8., scale=10.).sample(10000, test_util.test_seed())
    pi = 3.14159265
    stirling = x * tf.math.log(x) - x + 0.5 * tf.math.log(2 * pi / x)
    tfp_gamma_ = stirling + special.log_gamma_correction(x)
    tf_gamma, tfp_gamma = self.evaluate([tf.math.lgamma(x), tfp_gamma_])
    self.assertAllClose(tf_gamma, tfp_gamma, atol=0, rtol=1e-6)

  def testLogGammaDifference(self):
    y = half_cauchy.HalfCauchy(
        loc=8., scale=10.).sample(10000, test_util.test_seed())
    y_64 = tf.cast(y, tf.float64)
    # Not testing x near zero because the naive method is too inaccurate.
    # We will get implicit coverage in testLogBeta, where a good reference
    # implementation is available (scipy_special.betaln).
    x = uniform.Uniform(low=4., high=12.).sample(10000, test_util.test_seed())
    x_64 = tf.cast(x, tf.float64)
    naive_64_ = tf.math.lgamma(y_64) - tf.math.lgamma(x_64 + y_64)
    naive_64, sophisticated, sophisticated_64 = self.evaluate([
        naive_64_,
        special.log_gamma_difference(x, y),
        special.log_gamma_difference(x_64, y_64)
    ])
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

    y = half_cauchy.HalfCauchy(
        loc=8., scale=10.).sample(10000, test_util.test_seed())
    x = uniform.Uniform(low=0., high=8.).sample(10000, test_util.test_seed())
    _, [simple_gx_,
        simple_gy_] = gradient.value_and_gradient(simple_difference, [x, y])
    _, [gx_, gy_] = gradient.value_and_gradient(special.log_gamma_difference,
                                                [x, y])
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
    _, [simple_gx_,
        simple_gy_] = gradient.value_and_gradient(simple_difference, [x, y])
    _, [gx_, gy_] = gradient.value_and_gradient(special.log_gamma_difference,
                                                [x, y])
    simple_gx, simple_gy, gx, gy = self.evaluate(
        [simple_gx_, simple_gy_, gx_, gy_])
    self.assertAllClose(gx, simple_gx)
    self.assertAllClose(gy, simple_gy)

  def testLogBeta(self):
    strm = test_util.test_seed_stream()
    x = half_cauchy.HalfCauchy(loc=1., scale=15.).sample(10000, strm())
    x = self.evaluate(x)
    y = half_cauchy.HalfCauchy(loc=1., scale=15.).sample(10000, strm())
    y = self.evaluate(y)
    # Why not 1e-8?
    # - Could be because scipy does the reduction loops recommended
    #   by DiDonato and Morris 1988
    # - Could be that tf.math.lgamma is less accurate than scipy
    # - Could be that scipy evaluates in 64 bits internally
    atol = 1e-7
    rtol = 1e-5
    self.assertAllClose(
        scipy_special.betaln(x, y), special.lbeta(x, y), atol=atol, rtol=rtol)

  @test_util.numpy_disable_gradient_test
  def testLogBetaGradient(self):
    def simple_lbeta(x, y):
      return tf.math.lgamma(x) + tf.math.lgamma(y) - tf.math.lgamma(x + y)
    strm = test_util.test_seed_stream()
    x = half_cauchy.HalfCauchy(loc=1., scale=15.).sample(10000, strm())
    y = half_cauchy.HalfCauchy(loc=1., scale=15.).sample(10000, strm())
    _, [simple_gx_,
        simple_gy_] = gradient.value_and_gradient(simple_lbeta, [x, y])
    _, [gx_, gy_] = gradient.value_and_gradient(special.lbeta, [x, y])
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
    _, [simple_gx_,
        simple_gy_] = gradient.value_and_gradient(simple_lbeta, [x, y])
    _, [gx_, gy_] = gradient.value_and_gradient(special.lbeta, [x, y])
    simple_gx, simple_gy, gx, gy = self.evaluate(
        [simple_gx_, simple_gy_, gx_, gy_])
    self.assertAllClose(gx, simple_gx)
    self.assertAllClose(gy, simple_gy)

  @parameterized.parameters(tf.float32, tf.float64)
  def testLogBetaDtype(self, dtype):
    x = tf.constant([1., 2.], dtype=dtype)
    y = tf.constant([3., 4.], dtype=dtype)
    result = special.lbeta(x, y)
    self.assertEqual(result.dtype, dtype)


if __name__ == "__main__":
  test_util.main()
