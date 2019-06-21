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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
from absl.testing import parameterized
import numpy as np
from scipy import special as sp_special

import tensorflow as tf
import tensorflow_probability as tfp

from tensorflow_probability.python.internal import test_case
from tensorflow.python.framework import test_util  # pylint: disable=g-direct-tensorflow-import
from tensorflow.python.ops import gradient_checker_v2  # pylint: disable=g-direct-tensorflow-import


@test_util.run_all_in_graph_and_eager_modes
class LogCombinationsTest(tf.test.TestCase):

  def testLogCombinationsBinomial(self):
    n = [2, 5, 12, 15]
    k = [1, 2, 4, 11]

    log_combs = np.log(sp_special.binom(n, k))

    n = np.array(n, dtype=np.float32)
    counts = [[1., 1], [2., 3], [4., 8], [11, 4]]
    log_binom = tfp.math.log_combinations(n, counts)
    self.assertEqual([4], log_binom.shape)
    self.assertAllClose(log_combs, self.evaluate(log_binom))

  def testLogCombinationsShape(self):
    # Shape [2, 2]
    n = [[2, 5], [12, 15]]

    n = np.array(n, dtype=np.float32)
    # Shape [2, 2, 4]
    counts = [[[1., 1, 0, 0], [2., 2, 1, 0]], [[4., 4, 1, 3], [10, 1, 1, 4]]]
    log_binom = tfp.math.log_combinations(n, counts)
    self.assertEqual([2, 2], log_binom.shape)


@test_util.run_all_in_graph_and_eager_modes
class ReduceWeightedLogSumExp(tf.test.TestCase):

  def _reduce_weighted_logsumexp(self, logx, w, axis, keep_dims=False):
    m = np.max(logx, axis=axis, keepdims=True)
    sum_ = np.sum(w * np.exp(logx - m), axis=axis, keepdims=keep_dims)
    sgn = np.sign(sum_)
    if not keep_dims:
      m = np.squeeze(m, axis=axis)
    return m + np.log(sgn * sum_), sgn

  def testNoWeights(self):
    logx_ = np.array([[0., -1, 1000.],
                      [0, 1, -1000.],
                      [-5, 0, 5]])
    logx = tf.constant(logx_)
    with tf.GradientTape() as tape:
      tape.watch(logx)
      expected = tf.reduce_logsumexp(input_tensor=logx, axis=-1)
    grad_expected = tape.gradient(expected, logx)
    with tf.GradientTape() as tape:
      tape.watch(logx)
      actual, actual_sgn = tfp.math.reduce_weighted_logsumexp(
          logx, axis=-1, return_sign=True)
    grad_actual = tape.gradient(actual, logx)
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
    actual, actual_sgn = tfp.math.reduce_weighted_logsumexp(
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
    actual, actual_sgn = tfp.math.reduce_weighted_logsumexp(
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
        np.log(4),
        self.evaluate(tfp.math.reduce_weighted_logsumexp(x, w)))

    with np.errstate(divide='ignore'):
      self.assertAllClose(
          np.log([0, 2, 2]),
          self.evaluate(
              tfp.math.reduce_weighted_logsumexp(x, w, axis=0)))

    self.assertAllClose(
        np.log([1, 3]),
        self.evaluate(
            tfp.math.reduce_weighted_logsumexp(x, w, axis=1)))

    self.assertAllClose(
        np.log([[1], [3]]),
        self.evaluate(
            tfp.math.reduce_weighted_logsumexp(
                x, w, axis=1, keep_dims=True)))

    self.assertAllClose(
        np.log(4),
        self.evaluate(
            tfp.math.reduce_weighted_logsumexp(x, w, axis=[0, 1])))


@test_util.run_all_in_graph_and_eager_modes
class SoftThresholdTest(test_case.TestCase, parameterized.TestCase):

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
  # pylint: enable=bad-whitespace
  # pyformat: enable
  def test_soft_threshold(self, x, threshold, expected_y, expected_dydx):
    x = tf.convert_to_tensor(value=x, dtype=self.dtype)
    y, dydx = tfp.math.value_and_gradient(
        lambda x_: tfp.math.soft_threshold(x_, threshold), x)
    y_, dydx_ = self.evaluate([y, dydx])
    self.assertAllClose(expected_y, y_)
    self.assertAllClose(expected_dydx, dydx_)


# TODO(jvdillon): Merge this test back into:
# tensorflow/python/kernel_tests/softplus_op_test.py
# once TF core is accepting new ops.
@test_util.run_all_in_graph_and_eager_modes
class SoftplusInverseTest(tf.test.TestCase):

  def _npSoftplus(self, np_features):
    np_features = np.asarray(np_features)
    zero = np.asarray(0).astype(np_features.dtype)
    return np.logaddexp(zero, np_features)

  def _testSoftplus(self, np_features, use_gpu=False):
    np_features = np.asarray(np_features)
    np_softplus = self._npSoftplus(np_features)
    softplus = tf.nn.softplus(np_features)
    softplus_inverse = tfp.math.softplus_inverse(softplus)
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
    self.assertAllEqual(np.ones_like(tf_softplus).astype(np.bool),
                        tf_softplus > 0)

    self.assertShapeEqual(np_softplus, softplus)
    self.assertShapeEqual(np_softplus, softplus_inverse)

    self.assertAllEqual(np.ones_like(tf_softplus).astype(np.bool),
                        np.isfinite(tf_softplus))
    self.assertAllEqual(np.ones_like(tf_softplus_inverse).astype(np.bool),
                        np.isfinite(tf_softplus_inverse))

  def testNumbers(self):
    for t in [np.float16, np.float32, np.float64]:
      lower = {np.float16: -15, np.float32: -50, np.float64: -50}.get(t, -100)
      upper = {np.float16: 50, np.float32: 50, np.float64: 50}.get(t, 100)
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

  def testGradient(self):
    x = tf.constant(
        [-0.9, -0.7, -0.5, -0.3, -0.1, 0.1, 0.3, 0.5, 0.7, 0.9],
        shape=[2, 5],
        name='x')
    err = gradient_checker_v2.max_error(
        *gradient_checker_v2.compute_gradient(tf.nn.softplus, [x]))
    tf.compat.v1.logging.vlog(2, 'softplus (float) gradient err = ', err)
    self.assertLess(err, 1e-4)

  def testInverseSoftplusGradientNeverNan(self):
    # Note that this range contains both zero and inf.
    x = tf.constant(np.logspace(-8, 6).astype(np.float16))
    _, grads = self.evaluate(tfp.math.value_and_gradient(
        tfp.math.softplus_inverse, x))
    # Equivalent to `assertAllFalse` (if it existed).
    self.assertAllEqual(np.zeros_like(grads).astype(np.bool), np.isnan(grads))

  def testInverseSoftplusGradientFinite(self):
    # This range of x is all finite, and so is 1 / x.  So the
    # gradient and its approximations should be finite as well.
    x = tf.constant(np.logspace(-4.8, 4.5).astype(np.float16))
    _, grads = self.evaluate(tfp.math.value_and_gradient(
        tfp.math.softplus_inverse, x))
    # Equivalent to `assertAllTrue` (if it existed).
    self.assertAllEqual(
        np.ones_like(grads).astype(np.bool), np.isfinite(grads))


if __name__ == '__main__':
  tf.test.main()
