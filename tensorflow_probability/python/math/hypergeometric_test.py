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
import mpmath
import numpy as np
from scipy import special as scipy_special
import tensorflow.compat.v2 as tf

from tensorflow_probability.python.internal import test_util
from tensorflow_probability.python.math import hypergeometric as tfp_math


class Hyp2F1Test(test_util.TestCase):

  def _mpmath_hyp2f1(self, a, b, c, z):
    result = []
    for a_, b_, c_, z_ in zip(a, b, c, z):
      result.append(np.float64(mpmath.hyp2f1(a_, b_, c_, z_)))
    return np.array(result)

  def GenParam(self, low, high, dtype, seed):
    return tf.random.uniform(
        [int(1e4)], seed=seed,
        minval=low, maxval=high, dtype=dtype)

  def VerifyHyp2F1(
      self,
      dtype,
      rtol,
      a,
      b,
      c,
      z_lower=-0.9,
      z_upper=0.9,
      use_mpmath=False):
    comparison_hyp2f1 = self._mpmath_hyp2f1 if use_mpmath else scipy_special.hyp2f1
    seed_stream = test_util.test_seed_stream()
    z = tf.random.uniform(
        [int(1e4)], seed=seed_stream(),
        minval=z_lower, maxval=z_upper, dtype=dtype)

    hyp2f1, a, b, c, z = self.evaluate([
        tfp_math.hyp2f1_small_argument(a, b, c, z), a, b, c, z])
    expected = comparison_hyp2f1(a, b, c, z)
    self.assertAllClose(hyp2f1, expected, rtol=rtol)

  @parameterized.parameters(
      ([1], [1], [1], [1]),
      ([2], [3, 1], [5, 1, 1], [7, 1, 1, 1]),
      ([2, 1], [3], [5, 1, 1, 1], [7, 1, 1]),
      ([2, 1, 1, 1], [3, 1, 1], [5], [7, 1]),
      ([2, 1, 1], [3, 1, 1, 1], [5, 1], [7])
  )
  def testHyp2F1ShapeBroadcast(self, a_shape, b_shape, c_shape, z_shape):
    a = tf.zeros(a_shape, dtype=tf.float32)
    b = tf.zeros(b_shape, dtype=tf.float32)
    c = 10.5 * tf.ones(c_shape, dtype=tf.float32)
    z = tf.zeros(z_shape, dtype=tf.float32)
    broadcast_shape = functools.reduce(
        tf.broadcast_dynamic_shape, [a_shape, b_shape, c_shape, z_shape])
    hyp2f1 = tfp_math.hyp2f1_small_argument(a, b, c, z)
    broadcast_shape = self.evaluate(broadcast_shape)
    self.assertAllEqual(hyp2f1.shape, broadcast_shape)

  @parameterized.named_parameters(
      ("float32", np.float32, 6e-3),
      ("float64", np.float64, 1e-6))
  def testHyp2F1AtOne(self, dtype, rtol):
    seed_stream = test_util.test_seed_stream()
    a = self.GenParam(-10., 10., dtype, seed_stream())
    b = self.GenParam(-10., 10., dtype, seed_stream())
    # Ensure c > a + b so the evaluation is defined.
    c = a + b + 1.
    hyp2f1, a, b, c = self.evaluate([
        tfp_math.hyp2f1_small_argument(a, b, c, dtype(1.)), a, b, c])
    scipy_hyp2f1 = scipy_special.hyp2f1(a, b, c, dtype(1.))
    self.assertAllClose(hyp2f1, scipy_hyp2f1, rtol=rtol)

  @parameterized.named_parameters(
      ("float32", np.float32, 6e-3),
      ("float64", np.float64, 1e-6))
  def testHyp2F1EqualParameters(self, dtype, rtol):
    seed_stream = test_util.test_seed_stream()
    a = self.GenParam(-11.5, 11.5, dtype, seed_stream())
    b = self.GenParam(-11.5, 11.5, dtype, seed_stream())
    self.VerifyHyp2F1(dtype, rtol, a, b, a)
    self.VerifyHyp2F1(dtype, rtol, a, b, b)

  @parameterized.named_parameters(
      ("float32", np.float32, 6e-3),
      ("float64", np.float64, 1e-6))
  def testHyp2F1ParamsSmallZSmallCLargerPositive(self, dtype, rtol):
    # Ensure that |c| > |b|.
    seed_stream = test_util.test_seed_stream()
    a = self.GenParam(-0.5, 0.5, dtype, seed_stream())
    b = self.GenParam(-0.5, 0.5, dtype, seed_stream())
    c = self.GenParam(0.5, 1., dtype, seed_stream())
    self.VerifyHyp2F1(dtype, rtol, a, b, c)

  @parameterized.named_parameters(
      ("float64", np.float64, 1e-6))
  def testHyp2F1ParamsSmallZSmallCLargerNegative(self, dtype, rtol):
    # Ensure that |c| > |b|.
    seed_stream = test_util.test_seed_stream()
    a = self.GenParam(-0.5, 0.5, dtype, seed_stream())
    b = self.GenParam(-0.5, 0.5, dtype, seed_stream())
    c = self.GenParam(-1., -0.5, dtype, seed_stream())
    self.VerifyHyp2F1(dtype, rtol, a, b, c)

  @parameterized.named_parameters(
      ("float64", np.float64, 4e-5))
  def testHyp2F1ParamsSmallZSmallCSmaller(self, dtype, rtol):
    # Ensure that |c| < |b|.
    seed_stream = test_util.test_seed_stream()
    a = self.GenParam(0.5, 1., dtype, seed_stream())
    b = self.GenParam(0.5, 1., dtype, seed_stream())
    c = self.GenParam(0., 0.5, dtype, seed_stream())
    self.VerifyHyp2F1(dtype, rtol, a, b, c)

  @parameterized.named_parameters(
      ("float64", np.float64, 2e-4))
  def testHyp2F1ParamsSmallZPositiveLargeCLarger(self, dtype, rtol):
    seed_stream = test_util.test_seed_stream()
    a = self.GenParam(-0.5, 0.5, dtype, seed_stream())
    b = self.GenParam(-0.5, 0.5, dtype, seed_stream())
    c = self.GenParam(0.5, 1., dtype, seed_stream())
    self.VerifyHyp2F1(dtype, rtol, a, b, c, z_lower=0.9, z_upper=1.)

  @parameterized.named_parameters(
      ("float64", np.float64, 4e-6))
  def testHyp2F1ParamsSmallZPositiveLargeCSmaller(self, dtype, rtol):
    seed_stream = test_util.test_seed_stream()
    a = self.GenParam(0.5, 1., dtype, seed_stream())
    b = self.GenParam(0.5, 1., dtype, seed_stream())
    c = self.GenParam(-0.5, 0.5, dtype, seed_stream())
    self.VerifyHyp2F1(dtype, rtol, a, b, c, z_lower=0.9, z_upper=1.)

  @parameterized.named_parameters(
      ("float32", np.float32, 6e-3),
      ("float64", np.float64, 1e-6))
  def testHyp2F1ParamsSmallZNegativeLargeCLarger(self, dtype, rtol):
    seed_stream = test_util.test_seed_stream()
    a = self.GenParam(-0.5, 0.5, dtype, seed_stream())
    b = self.GenParam(-0.5, 0.5, dtype, seed_stream())
    c = self.GenParam(0.5, 1., dtype, seed_stream())
    self.VerifyHyp2F1(dtype, rtol, a, b, c, z_lower=-1., z_upper=-0.9)

  @parameterized.named_parameters(
      ("float64", np.float64, 1e-6))
  def testHyp2F1ParamsSmallZNegativeLargeCSmaller(self, dtype, rtol):
    seed_stream = test_util.test_seed_stream()
    a = self.GenParam(0.5, 1., dtype, seed_stream())
    b = self.GenParam(0.5, 1., dtype, seed_stream())
    c = self.GenParam(-0.5, 0.5, dtype, seed_stream())
    self.VerifyHyp2F1(dtype, rtol, a, b, c, z_lower=-1., z_upper=-0.9)

  @parameterized.named_parameters(
      ("float64", np.float64, 2e-5))
  def testHyp2F1ParamsMediumCLarger(self, dtype, rtol):
    seed_stream = test_util.test_seed_stream()
    a = self.GenParam(-10., 10., dtype, seed_stream())
    b = self.GenParam(-10., 10., dtype, seed_stream())
    c = self.GenParam(10., 20., dtype, seed_stream())
    self.VerifyHyp2F1(dtype, rtol, a, b, c, z_lower=-1., z_upper=1.)

  @parameterized.named_parameters(
      ("float64", np.float64, 1e-6))
  def testHyp2F1ParamsLargerCLarger(self, dtype, rtol):
    seed_stream = test_util.test_seed_stream()
    a = self.GenParam(10., 50., dtype, seed_stream())
    b = self.GenParam(10., 50., dtype, seed_stream())
    c = self.GenParam(50., 100., dtype, seed_stream())
    self.VerifyHyp2F1(dtype, rtol, a, b, c, z_lower=-1., z_upper=0.7)

  @parameterized.named_parameters(
      ("float64", np.float64, 6e-6))
  def testHyp2F1ParamsLargerCSmaller(self, dtype, rtol):
    seed_stream = test_util.test_seed_stream()
    a = self.GenParam(50., 80., dtype, seed_stream())
    b = self.GenParam(50., 80., dtype, seed_stream())
    c = self.GenParam(20., 50., dtype, seed_stream())
    self.VerifyHyp2F1(dtype, rtol, a, b, c, z_lower=-1., z_upper=1.)

  @test_util.numpy_disable_gradient_test
  @test_util.jax_disable_test_missing_functionality(
      "Gradients not supported in JAX.")
  def test2F1HypergeometricGradient(self):
    a = tf.constant([-0.1,], dtype=np.float64)[..., tf.newaxis]
    b = tf.constant([0.8,], dtype=np.float64)[..., tf.newaxis]
    c = tf.constant([9.9,], dtype=np.float64)[..., tf.newaxis]
    z = tf.constant([0.1], dtype=np.float64)
    err = self.compute_max_gradient_error(
        functools.partial(tfp_math.hyp2f1_small_argument, a, b, c), [z])
    self.assertLess(err, 2e-4)


if __name__ == "__main__":
  tf.test.main()
