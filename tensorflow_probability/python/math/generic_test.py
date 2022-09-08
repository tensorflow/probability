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
"""Tests for tensorflow_probability.python.math.generic."""

import functools
import itertools
import os

# Dependency imports
from absl.testing import parameterized
import numpy as np
from scipy import special as sp_special

import tensorflow.compat.v1 as tf1
import tensorflow.compat.v2 as tf
from tensorflow_probability.python.distributions import categorical
from tensorflow_probability.python.distributions import mixture_same_family
from tensorflow_probability.python.distributions import normal
from tensorflow_probability.python.internal import distribute_test_lib
from tensorflow_probability.python.internal import test_util
from tensorflow_probability.python.math import generic
from tensorflow_probability.python.math import gradient
from tensorflow_probability.python.math import linalg
from tensorflow_probability.python.math.gradient import batch_jacobian


JAX_MODE = False


def _allow_all_gather(fn):
  return functools.partial(fn, experimental_allow_all_gather=True)


@test_util.test_all_tf_execution_regimes
class LogHarmonicMeanExpTest(test_util.TestCase):

  @parameterized.named_parameters(
      ('Defaults', None, False),
      ('Axis', (0,), False),
      ('KeepDims', (0,), True),
  )
  def testBasic(self, axis, keepdims):
    x = np.array([[0.1, 0.2], [0.3, 0.4]]).astype(np.float32)
    true_val = np.log(1 / (1 / x).mean(axis=axis, keepdims=keepdims))
    val = generic.reduce_log_harmonic_mean_exp(
        np.log(x), axis=axis, keepdims=keepdims)
    self.assertAllClose(true_val, val)


@test_util.test_all_tf_execution_regimes
class LogCombinationsTest(test_util.TestCase):

  def testLogCombinationsBinomial(self):
    n = [2, 5, 12, 15]
    k = [1, 2, 4, 11]

    log_combs = np.log(sp_special.binom(n, k))

    n = np.array(n, dtype=np.float32)
    counts = [[1., 1], [2., 3], [4., 8], [11, 4]]
    log_binom = generic.log_combinations(n, counts)
    self.assertEqual([4], log_binom.shape)
    self.assertAllClose(log_combs, self.evaluate(log_binom))

  def testLogCombinationsShape(self):
    # Shape [2, 2]
    n = [[2, 5], [12, 15]]

    n = np.array(n, dtype=np.float32)
    # Shape [2, 2, 4]
    counts = [[[1., 1, 0, 0], [2., 2, 1, 0]], [[4., 4, 1, 3], [10, 1, 1, 4]]]
    log_binom = generic.log_combinations(n, counts)
    self.assertEqual([2, 2], log_binom.shape)


@test_util.test_all_tf_execution_regimes
class ReduceWeightedLogSumExp(test_util.TestCase):

  def _reduce_weighted_logsumexp(self, logx, w, axis, keep_dims=False):
    m = np.max(logx, axis=axis, keepdims=True)
    sum_ = np.sum(w * np.exp(logx - m), axis=axis, keepdims=keep_dims)
    sgn = np.sign(sum_)
    if not keep_dims:
      m = np.squeeze(m, axis=axis)
    return m + np.log(sgn * sum_), sgn

  @test_util.numpy_disable_gradient_test
  def testNoWeights(self):
    logx_ = np.array([[0., -1, 1000.],
                      [0, 1, -1000.],
                      [-5, 0, 5]])
    logx = tf.constant(logx_)
    expected = tf.reduce_logsumexp(logx, axis=-1)
    grad_expected, _ = gradient.value_and_gradient(
        lambda logx: tf.reduce_logsumexp(logx, axis=-1), logx)
    actual, actual_sgn = generic.reduce_weighted_logsumexp(
        logx, axis=-1, return_sign=True)
    grad_actual, _ = gradient.value_and_gradient(
        lambda logx: generic.reduce_weighted_logsumexp(logx, axis=-1), logx)
    [
        actual_,
        actual_sgn_,
        grad_actual_,
        expected_,
        grad_expected_,
    ] = self.evaluate([
        actual,
        actual_sgn,
        grad_actual,
        expected,
        grad_expected,
    ])
    self.assertAllEqual(expected_, actual_)
    self.assertAllEqual(grad_expected_, grad_actual_)
    self.assertAllEqual([1., 1, 1], actual_sgn_)

  def testNegativeWeights(self):
    logx_ = np.array([[0., -1, 1000.],
                      [0, 1, -1000.],
                      [-5, 0, 5]])
    w_ = np.array([[1., 1, -1],
                   [1, -2, 1],
                   [1, 0, 1]])
    expected, _ = self._reduce_weighted_logsumexp(logx_, w_, axis=-1)
    logx = tf.constant(logx_)
    w = tf.constant(w_)
    actual, actual_sgn = generic.reduce_weighted_logsumexp(
        logx, w, axis=-1, return_sign=True)
    actual_, actual_sgn_ = self.evaluate([actual, actual_sgn])
    self.assertAllEqual(expected, actual_)
    self.assertAllEqual([-1., -1, 1], actual_sgn_)

  def testKeepDims(self):
    logx_ = np.array([[0., -1, 1000.],
                      [0, 1, -1000.],
                      [-5, 0, 5]])
    w_ = np.array([[1., 1, -1],
                   [1, -2, 1],
                   [1, 0, 1]])
    expected, _ = self._reduce_weighted_logsumexp(
        logx_, w_, axis=-1, keep_dims=True)
    logx = tf.constant(logx_)
    w = tf.constant(w_)
    actual, actual_sgn = generic.reduce_weighted_logsumexp(
        logx, w, axis=-1, return_sign=True, keep_dims=True)
    actual_, actual_sgn_ = self.evaluate([actual, actual_sgn])
    self.assertAllEqual(expected, actual_)
    self.assertAllEqual([[-1.], [-1], [1]], actual_sgn_)

  def testDocString(self):
    """This test verifies the correctness of the docstring examples."""

    x = tf.constant([[0., 0, 0],
                     [0, 0, 0]])

    w = tf.constant([[-1., 1, 1],
                     [1, 1, 1]])

    self.assertAllClose(
        np.log(4), self.evaluate(generic.reduce_weighted_logsumexp(x, w)))

    with np.errstate(divide='ignore'):
      self.assertAllClose(
          np.log([0, 2, 2]),
          self.evaluate(generic.reduce_weighted_logsumexp(x, w, axis=0)))

    self.assertAllClose(
        np.log([1, 3]),
        self.evaluate(generic.reduce_weighted_logsumexp(x, w, axis=1)))

    self.assertAllClose(
        np.log([[1], [3]]),
        self.evaluate(
            generic.reduce_weighted_logsumexp(x, w, axis=1, keep_dims=True)))

    self.assertAllClose(
        np.log(4),
        self.evaluate(generic.reduce_weighted_logsumexp(x, w, axis=[0, 1])))


@test_util.test_all_tf_execution_regimes
class SoftThresholdTest(test_util.TestCase):

  dtype = tf.float32

  # Expected values computed using arbitrary precision.
  # pyformat: disable
  # pylint: disable=bad-whitespace
  @parameterized.parameters(
      # x   threshold  expected_y  expected_dydx
      (5., 5., 0., 1.),
      (2., 5., 0., 0.),
      (-2., 5., 0., 0.),
      (3., 2.5, 0.5, 1.),
      (-3., 2.5, -0.5, 1.),
      (-1., 1., 0., 1.),
      (-6., 5., -1., 1.),
      (0., 0., 0., 0.),
  )
  @test_util.numpy_disable_gradient_test
  # pylint: enable=bad-whitespace
  # pyformat: enable
  def test_soft_threshold(self, x, threshold, expected_y, expected_dydx):
    x = tf.convert_to_tensor(x, dtype=self.dtype)
    y, dydx = gradient.value_and_gradient(
        lambda x_: generic.soft_threshold(x_, threshold), x)
    y_, dydx_ = self.evaluate([y, dydx])
    self.assertAllClose(expected_y, y_)
    self.assertAllClose(expected_dydx, dydx_)


# TODO(jvdillon): Merge this test back into:
# tensorflow/python/kernel_tests/softplus_op_test.py
# once TF core is accepting new ops.
@test_util.test_all_tf_execution_regimes
class SoftplusInverseTest(test_util.TestCase):

  def _npSoftplus(self, np_features):
    np_features = np.asarray(np_features)
    zero = np.asarray(0).astype(np_features.dtype)
    return np.logaddexp(zero, np_features)

  def _testSoftplus(self, np_features, use_gpu=False):
    np_features = np.asarray(np_features)
    np_softplus = self._npSoftplus(np_features)
    softplus = tf.math.softplus(np_features)
    softplus_inverse = generic.softplus_inverse(softplus)
    [tf_softplus, tf_softplus_inverse] = self.evaluate([
        softplus, softplus_inverse])
    self.assertAllCloseAccordingToType(np_softplus, tf_softplus)
    rtol = {'float16': 0.07, 'float32': 0.003, 'float64': 0.002}.get(
        str(np_features.dtype), 1e-6)
    # This will test that we correctly computed the inverse by verifying we
    # recovered the original input.
    self.assertAllCloseAccordingToType(
        np_features, tf_softplus_inverse,
        atol=0., rtol=rtol)
    self.assertAllEqual(np.ones_like(tf_softplus).astype(np.bool_),
                        tf_softplus > 0)

    self.assertShapeEqual(np_softplus, softplus)
    self.assertShapeEqual(np_softplus, softplus_inverse)

    self.assertAllEqual(np.ones_like(tf_softplus).astype(np.bool_),
                        np.isfinite(tf_softplus))
    self.assertAllEqual(np.ones_like(tf_softplus_inverse).astype(np.bool_),
                        np.isfinite(tf_softplus_inverse))

  @test_util.numpy_disable_gradient_test  # TODO(sharadmv): fix Numpy test
  def testNumbers(self):
    for t in [np.float32, np.float64]:
      lower = {np.float32: -50, np.float64: -50}.get(t, -100)
      upper = {np.float32: 50, np.float64: 50}.get(t, 100)
      self._testSoftplus(
          np.array(np.linspace(lower, upper, int(1e3)).astype(t)).reshape(
              [2, -1]),
          use_gpu=False)
      self._testSoftplus(
          np.array(np.linspace(lower, upper, int(1e3)).astype(t)).reshape(
              [2, -1]),
          use_gpu=True)
      log_eps = np.log(np.finfo(t).eps)
      one = t(1)
      ten = t(10)
      self._testSoftplus(
          [
              log_eps,
              log_eps - one,
              log_eps + one,
              log_eps - ten,
              log_eps + ten,
              -log_eps,
              -log_eps - one,
              -log_eps + one,
              -log_eps - ten,
              -log_eps + ten,
          ],
          use_gpu=False)
      self._testSoftplus(
          [
              log_eps,
              log_eps - one,
              log_eps + one,
              log_eps - ten,
              log_eps + ten - log_eps,
              -log_eps - one,
              -log_eps + one,
              -log_eps - ten,
              -log_eps + ten,
          ],
          use_gpu=True)

  @test_util.numpy_disable_gradient_test
  def testGradient(self):
    x = tf.constant(
        [-0.9, -0.7, -0.5, -0.3, -0.1, 0.1, 0.3, 0.5, 0.7, 0.9],
        shape=[2, 5],
        name='x')
    err = self.compute_max_gradient_error(tf.math.softplus, [x])
    tf1.logging.vlog(2, 'softplus (float) gradient err = ', err)
    self.assertLess(err, 1e-4)

  @test_util.numpy_disable_gradient_test
  def testInverseSoftplusGradientNeverNan(self):
    # Note that this range contains both zero and inf.
    x = tf.constant(np.logspace(-8, 6).astype(np.float16))
    _, grads = self.evaluate(
        gradient.value_and_gradient(generic.softplus_inverse, x))
    # Equivalent to `assertAllFalse` (if it existed).
    self.assertAllEqual(np.zeros_like(grads).astype(np.bool_), np.isnan(grads))

  @test_util.numpy_disable_gradient_test
  def testInverseSoftplusGradientFinite(self):
    # This range of x is all finite, and so is 1 / x.  So the
    # gradient and its approximations should be finite as well.
    x = tf.constant(np.logspace(-4.8, 4.5).astype(np.float16))
    _, grads = self.evaluate(
        gradient.value_and_gradient(generic.softplus_inverse, x))
    # Equivalent to `assertAllTrue` (if it existed).
    self.assertAllEqual(
        np.ones_like(grads).astype(np.bool_), np.isfinite(grads))


@test_util.test_all_tf_execution_regimes
class LogCumsumExpTests(test_util.TestCase):

  def _testCumulativeLogSumExp(self, x, axis=0):
    result_naive = tf.cumsum(tf.exp(x), axis=axis)
    result_fused = tf.exp(generic.log_cumsum_exp(x, axis=axis))
    self.assertAllClose(result_naive, result_fused)

  def testMinusInfinity(self):
    x = np.log([0., 0., 1., 1., 1., 1., 0., 0.])
    self._testCumulativeLogSumExp(x)

  def test1D(self):
    x = np.arange(10) / 10.0 - 0.5
    self._testCumulativeLogSumExp(x)

  def test2D(self):
    x = np.reshape(np.arange(20) / 20.0 - 0.5, (2, 10))
    for axis in (-2, -1, 0, 1):
      self._testCumulativeLogSumExp(x, axis=axis)

  @test_util.numpy_disable_gradient_test
  def testGradient(self):
    x = tf.convert_to_tensor(
        np.arange(10)[np.newaxis, ...] / 10.0 - 0.5, dtype=tf.float64)
    jac_naive = batch_jacobian(lambda t: tf.cumsum(tf.exp(t), axis=-1), x)
    jac_fused = batch_jacobian(
        lambda t: tf.exp(generic.log_cumsum_exp(t, axis=-1)), x)
    self.assertAllClose(jac_naive, jac_fused)

  def test1DLarge(self):
    # This test ensures that the operation is correct even when the naive
    # implementation would overflow.
    x = tf.convert_to_tensor(np.arange(20) * 20.0, dtype=tf.float32)
    result_fused = self.evaluate(generic.log_cumsum_exp(x))
    result_map = self.evaluate(tf.map_fn(
        lambda i: tf.reduce_logsumexp(x[:i + 1]),
        tf.range(tf.shape(x)[0]),
        dtype=x.dtype))
    self.assertAllClose(result_fused, result_map)

  @parameterized.named_parameters(
      ('not_compiled', False),
      ('xla_compiled', True))
  @test_util.numpy_disable_gradient_test
  @test_util.jax_disable_test_missing_functionality(
      '`GradientTape` does not have `jacobian` method')
  def testGradientAtMinusInf(self, xla_compile):
    # This ensures that cumulative sums involving `-np.inf` behave
    # correctly even when compiled with XLA.
    x = tf.constant([1., -np.inf, -np.inf, 4., 5., 6., 7., 8.])
    @tf.function(jit_compile=xla_compile)
    def compute_jacobian(x):
      with tf.GradientTape() as g:
        g.watch(x)
        y = generic.log_cumsum_exp(x)
      return g.jacobian(y, x)
    # The rows of the Jacobian of `log_cumsum_exp` are given by
    # `tf.math.softmax`.
    rows = [tf.concat([tf.math.softmax(x[:i + 1]),
                       tf.zeros([7 - i])], axis=0)
            for i in range(8)]
    expected_jacobian = tf.stack(rows, axis=0)
    jacobian = compute_jacobian(x)
    self.assertAllClose(jacobian, expected_jacobian, atol=1e-7)

  @test_util.numpy_disable_gradient_test
  @test_util.jax_disable_test_missing_functionality(
      '`GradientTape` does not have `jacobian` method')
  def testGradientCumsumViaLogCumsumExp(self):
    # Regression test for b/156297366.
    x = tf.constant([1., 2., 3., 4.])
    with tf.GradientTape(persistent=True) as g:
      g.watch(x)
      z = tf.exp(generic.log_cumsum_exp(tf.math.log(x)))
    expected_gradients = linalg.fill_triangular(tf.ones(4 * (4 + 1) // 2))
    gradients = g.jacobian(z, x)
    self.assertAllClose(gradients, expected_gradients, atol=1e-7)


@test_util.test_all_tf_execution_regimes
class LogAddExpTest(test_util.TestCase):

  @test_util.numpy_disable_gradient_test
  def test_small(self):
    x = [-2, -1000]
    y = [-1000, -3]
    z, g = self.evaluate(
        gradient.value_and_gradient(generic.log_add_exp, [x, y]))
    self.assertAllClose([-2., -3.], z, atol=0., rtol=1e-5)
    self.assertAllEqual(np.eye(2), g)

  @test_util.numpy_disable_gradient_test
  def test_medium(self):
    x = [-2, -3]
    y = [-3, 2]
    z, g = self.evaluate(
        gradient.value_and_gradient(generic.log_add_exp, [x, y]))
    self.assertAllClose(np.log(np.exp(x) + np.exp(y)), z, atol=0., rtol=1e-5)
    self.assertAllNotNone(g)

  @test_util.numpy_disable_gradient_test
  def test_big(self):
    x = [2, 1000]
    y = [1000, 3]
    z, g = self.evaluate(
        gradient.value_and_gradient(generic.log_add_exp, [x, y]))
    self.assertAllClose([1000., 1000.], z, atol=0., rtol=1e-5)
    self.assertAllEqual(1. - np.eye(2), g)

  @test_util.numpy_disable_gradient_test
  def test_equal_arguments(self):
    # The standard way to compute `log_add_exp` makes use of
    # the subexpression `abs(x - y)` which has a discontinuous
    # gradient at `x == y`.
    x = np.log(np.arange(1, 21, dtype=np.float32))
    z, g = self.evaluate(
        gradient.value_and_gradient(generic.log_add_exp, [x, x]))
    self.assertAllClose(np.log(2.0) + x, z, atol=0., rtol=1e-5)
    self.assertAllClose(g, np.full([2, 20], 0.5))


@test_util.test_all_tf_execution_regimes
class LogSubExpTest(test_util.TestCase):

  def testLogSubExp(self):
    self.assertAllClose(-np.inf, self.evaluate(generic.log_sub_exp(1., 1.)))

    # Try log(exp(-1000) - (exp(-1000) + 2)
    # log(e^-k / 2) = log(e^-k) - log(2), or
    # = log(e^-k - .5*e^-k)
    # = log(e^-k - e^(-k + log(.5)))
    self.assertAllClose(
        -1000. - np.log(2.),
        self.evaluate(generic.log_sub_exp(-1000., -1000. + np.log(.5))))

  @test_util.numpy_disable_gradient_test
  def test_small(self):
    x = [-2]
    y = [-1000]
    z, g = self.evaluate(
        gradient.value_and_gradient(generic.log_sub_exp, [x, y]))
    self.assertAllClose([-2.], z, atol=0., rtol=1e-5)
    self.assertAllClose([[1.], [0.]], g)

  @test_util.numpy_disable_gradient_test
  def test_medium(self):
    x = [-2, -3, -5, -3]
    y = [-3, -5, -3, -2]
    z, g = self.evaluate(
        gradient.value_and_gradient(generic.log_sub_exp, [x, y]))
    self.assertAllClose(np.log(np.abs(np.exp(x) - np.exp(y))), z,
                        atol=0., rtol=1e-5)
    self.assertAllEqual([1., 1, -1, -1],
                        generic.log_sub_exp(x, y, return_sign=True)[1])
    self.assertAllNotNone(g)

  @test_util.numpy_disable_gradient_test
  def test_big(self):
    x = [1000, -3]
    y = [2, 1000]
    z, g = self.evaluate(
        gradient.value_and_gradient(generic.log_sub_exp, [x, y]))
    self.assertAllClose([1000., 1000.], z, atol=0., rtol=1e-5)
    self.assertAllEqual([[1., 0.], [0., 1.]], g)
    self.assertAllEqual([1., -1.],
                        generic.log_sub_exp(x, y, return_sign=True)[1])


@test_util.test_all_tf_execution_regimes
class Log1mexpTest(test_util.TestCase):

  def testLog1mexp(self):
    self.assertAllClose(-np.inf, self.evaluate(generic.log1mexp(0.)))
    self.assertAllClose(0., self.evaluate(generic.log1mexp(np.inf)))

    x = np.linspace(0.1, 20, 100)
    self.assertAllClose(
        np.log(-np.expm1(-x)), self.evaluate(generic.log1mexp(x)))

    x = np.linspace(-20., -0.1, 100)
    self.assertAllClose(
        np.log(-np.expm1(x)), self.evaluate(generic.log1mexp(x)))


@test_util.test_all_tf_execution_regimes
class Sqrt1pm1Test(test_util.TestCase):

  def testSqrt1pm1(self):
    self.assertAllClose(-1., self.evaluate(generic.sqrt1pm1(-1.)))
    self.assertAllClose(np.inf, self.evaluate(generic.sqrt1pm1(np.inf)))

    x = np.linspace(0.1, 20, 100)
    self.assertAllClose(
        np.sqrt(x + 1.) - 1., self.evaluate(generic.sqrt1pm1(x)))


@test_util.test_all_tf_execution_regimes
class LogCoshTest(test_util.TestCase):

  def testLogCoshNonNegative(self):
    x = np.logspace(-10., 6., 100)
    self.assertAllGreaterEqual(generic.log_cosh(x), 0.)

  def testLogCoshAtZero(self):
    self.assertAllClose(0., self.evaluate(generic.log_cosh(0.)))

  def testLogCoshSymmetric(self):
    x = np.linspace(0., 20., 100)
    self.assertAllClose(
        self.evaluate(generic.log_cosh(x)), self.evaluate(generic.log_cosh(-x)))

  def testLogCoshNoInf(self):
    # Check that the computation succeeds over a large range of values.
    x = np.logspace(10., 20., 100)
    self.assertAllEqual(
        np.zeros(x.shape, dtype=np.bool_),
        self.evaluate(tf.math.is_inf(generic.log_cosh(x))))

  def testLogCosh(self):
    x = np.linspace(10., 40., 100)
    self.assertAllClose(np.log(np.cosh(x)), self.evaluate(generic.log_cosh(x)))

    # Test for small values
    x = np.logspace(-10, -2, 100)
    self.assertAllClose(np.log(np.cosh(x)), self.evaluate(generic.log_cosh(x)))

    # Test for larger values.
    x = np.logspace(1., 2., 100)
    self.assertAllClose(np.log(np.cosh(x)), self.evaluate(generic.log_cosh(x)))

  @test_util.numpy_disable_gradient_test
  def testLogCoshGrad(self):
    x = np.linspace(-30., 30., 100)
    err = self.compute_max_gradient_error(generic.log_cosh, [x])
    self.assertLess(err, 1e-6)


@test_util.test_all_tf_execution_regimes
class Smootherstep(test_util.TestCase):

  @test_util.numpy_disable_gradient_test
  def test_value_vector(self):
    x = tf.constant([-np.inf, -20., 0., 0.5, 1., 20., np.inf])
    y, _ = gradient.value_and_gradient(generic.smootherstep, x)
    self.assertAllEqual([7], y.shape)
    y_ = self.evaluate(y)
    self.assertAllClose([0., 0., 0., 0.5, 1., 1., 1.], y_, atol=1e-5, rtol=1e-5)

  @test_util.numpy_disable_gradient_test
  def test_gradient_matrix(self):
    x = tf.constant([[-np.inf, -20., 0., 0.5],
                     [np.inf, 20., 1., 0.5]])
    _, g = gradient.value_and_gradient(generic.smootherstep, x)
    self.assertAllEqual([2, 4], g.shape)
    g_ = self.evaluate(g)
    self.assertAllClose([[0., 0., 0., 1.875]] * 2, g_, atol=1e-5, rtol=1e-5)


@test_util.test_all_tf_execution_regimes
class SoftSortingMatrixTest(test_util.TestCase):

  # By applying an argmax on each column of the generated matrix,
  # we should recover an argsort. This is an invariant with respect
  # to temperature.
  @parameterized.parameters(
      {'shape': (4,), 'temperature': 1e2},
      {'shape': (4,), 'temperature': 1e1},
      {'shape': (4,), 'temperature': 1e0},
      {'shape': (4,), 'temperature': 1e-1},
      {'shape': (5, 5, 4), 'temperature': 1e2},
      {'shape': (5, 5, 4), 'temperature': 1e1},
      {'shape': (5, 5, 4), 'temperature': 1e0},
      {'shape': (5, 5, 4), 'temperature': 1e-1},
  )
  def testMatchesArgsort(self, shape, temperature):
    x = np.random.randn(*shape)
    # We sort in decreasing order.
    expected_sort = np.flip(np.argsort(x, axis=-1), axis=-1)
    soft_sort_permutation_ = self.evaluate(
        generic.soft_sorting_matrix(x=x, temperature=temperature))
    # Check that the rows sum to 1.
    self.assertAllClose(np.ones(shape), np.sum(soft_sort_permutation_, axis=-1))
    # Check non-negativity.
    self.assertTrue(np.all(soft_sort_permutation_ >= 0.))

    # Check that by applying an argmax on the columns we actually get
    # the indices that correspond to the argsort.
    actual_sort_ = np.argmax(soft_sort_permutation_, axis=-1)
    self.assertAllClose(expected_sort, actual_sort_)


@test_util.test_all_tf_execution_regimes
class _KahanSumTest(test_util.TestCase):

  @parameterized.named_parameters(
      dict(testcase_name='_all',
           sample_shape=[3, int(1e6)], axis=None),
      dict(testcase_name='_ax1',
           sample_shape=[13, int(1e6)], axis=1),
      dict(testcase_name='_ax1_list_keepdims',
           sample_shape=[13, int(1e6)], axis=[-1], keepdims=True),
      dict(testcase_name='_ax_both_tuple',
           sample_shape=[3, int(1e6)], axis=(-2, 1)),
      dict(testcase_name='_ax_01_keepdims',
           sample_shape=[2, int(1e6), 13], axis=[0, 1], keepdims=True),
      dict(testcase_name='_ax_21',
           sample_shape=[13, int(1e6), 3], axis=[2, -2]))
  def testKahanSum(self, sample_shape, axis, keepdims=False):
    fn = functools.partial(
        generic.reduce_kahan_sum, axis=axis, keepdims=keepdims)
    if self.jit:
      self.skip_if_no_xla()
      fn = tf.function(fn, jit_compile=True)
    dist = mixture_same_family.MixtureSameFamily(
        categorical.Categorical(logits=[0., 0]),
        normal.Normal(loc=[0., 1e6], scale=[1., 1e3]))
    vals = self.evaluate(dist.sample(sample_shape, seed=test_util.test_seed()))
    oracle = tf.reduce_sum(tf.cast(vals, tf.float64), axis=axis,
                           keepdims=keepdims)
    result = fn(vals)
    self.assertEqual(oracle.shape, result.total.shape)
    self.assertEqual(oracle.shape, result.correction.shape)
    kahan64 = (tf.cast(result.total, tf.float64) -
               self.evaluate(tf.cast(result.correction, tf.float64)))
    if np.prod(result.correction.shape) > 1:
      self.assertNotAllEqual(
          result.correction, tf.zeros_like(result.correction))
    self.assertAllClose(oracle, kahan64)  # passes even with --vary_seed
    # The counterpoint naive sum below would not typically pass (but does not
    # reliably fail, either). It can fail w/ rtol as high as 0.006.
    # naive_sum = tf.cast(tf.reduce_sum(vals, axis=axis, keepdims=keepdims),
    #                     tf.float64)
    # self.assertAllClose(oracle, naive_sum)

    if test_util.is_numpy_not_jax_mode():
      # Skip checking gradients for Numpy.
      return

    dy = self.evaluate(
        normal.Normal(0,
                      1).sample(result.total.shape, seed=test_util.test_seed()))
    def grad_fn(x):
      return gradient.value_and_gradient(
          lambda x: fn(x).total**2., x, output_gradients=dy)[1]
    if self.jit:
      grad_fn = tf.function(grad_fn, jit_compile=True)
    grad = grad_fn(vals)
    self.assertIsNotNone(grad)
    keepdims_shape = tf.reduce_sum(vals, axis=axis, keepdims=True).shape
    self.assertAllClose(
        tf.broadcast_to(
            tf.reshape(dy * 2. * result.total, keepdims_shape), vals.shape),
        grad)


class KahanSumJitTest(_KahanSumTest):
  jit = True


class KahanSumTest(_KahanSumTest):
  jit = False

del _KahanSumTest


@test_util.test_all_tf_execution_regimes
class CollectiveTest(distribute_test_lib.DistributedTest):

  @test_util.numpy_disable_test_missing_functionality(
      'NumPy backend does not support distributed computation.')
  @parameterized.named_parameters(*(
      (f'{name} {ax}', op, ax)  # pylint: disable=g-complex-comprehension
      for (name, op), ax in itertools.product((
          ('logmeanexp', _allow_all_gather(generic.reduce_logmeanexp)),
          ('log_harmonicmeanexp',
           _allow_all_gather(generic.reduce_log_harmonic_mean_exp)),
          ('reduce_weighted_logsumexp',
           _allow_all_gather(generic.reduce_weighted_logsumexp)),
      ), (None, 0, 1, 2, [0, 1], [1, 2], [0, 2], [0, 1, 2]))))
  def test_reduce_with_collectives_matches_reduce_without_collectives(
      self, reduce_op, axes):

    if axes is None:
      pos_axes = list(range(2))
      named_axes = [self.axis_name]
    else:
      axes = [axes] if not isinstance(axes, list) else axes
      pos_axes = [x - 1 for x in axes if x != 0]
      named_axes = [self.axis_name] if 0 in axes else []

    x = tf.reshape(
        tf.range(distribute_test_lib.NUM_DEVICES * 6.) / 5.,
        [distribute_test_lib.NUM_DEVICES, 3, 2])

    def run(x):
      return reduce_op(x, axis=pos_axes, experimental_named_axis=named_axes)

    def distributed_run(x):
      return self.per_replica_to_tensor(
          self.strategy_run(run, (self.shard_values(x),)))
    reduce_out = reduce_op(x, axis=axes)
    dist_out = distributed_run(x)
    # If we reduce over the 0 dimension, it will still be present in the
    # distributed op
    if axes is None or 0 in axes:
      for i in range(distribute_test_lib.NUM_DEVICES):
        self.assertAllClose(reduce_out, dist_out[i])
    else:
      self.assertAllClose(reduce_out, dist_out)


if __name__ == '__main__':
  # TODO(b/173158845): XLA:CPU reassociates away the Kahan correction term.
  os.environ['XLA_FLAGS'] = '--xla_cpu_enable_fast_math=false'
  test_util.main()
