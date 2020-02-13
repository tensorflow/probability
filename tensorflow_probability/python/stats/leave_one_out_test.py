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
"""Tests for tensorflow_probability.python.stats.leave_one_out."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
import numpy as np

import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp

from tensorflow_probability.python.internal import test_util


tfd = tfp.distributions


@test_util.test_all_tf_execution_regimes
class LogSooMeanTest(test_util.TestCase):

  def _make_flatteners(self, old_shape, axis):
    axis = np.array(axis).reshape(-1)
    old_shape = np.array(old_shape).reshape(-1)
    rank = len(old_shape)
    axis = np.where(axis < 0, axis + rank, axis)
    other_axis = np.setdiff1d(range(rank), axis)
    new_shape = np.concatenate([[-1], old_shape[other_axis]], axis=0)
    new_perm = np.concatenate([axis, other_axis], axis=0)
    def _flatten(x):
      return np.reshape(np.transpose(x, new_perm), new_shape)
    undo_permutation = np.argsort(new_perm)
    permuted_shape = np.concatenate(
        [old_shape[axis], old_shape[other_axis]], axis=0)
    def _unflatten(x):
      return np.transpose(np.reshape(x, permuted_shape), undo_permutation)
    return _flatten, _unflatten

  def _numpy_log_soomean_exp_impl(self, logx, axis):
    """Numpy implementation of `log_soo_sum`; readable but not efficient."""
    # Since this is a naive/intuitive implementation, we compensate by using the
    # highest precision we can.
    logx = np.float128(logx)
    n = logx.shape[0]
    u = np.exp(logx)
    loogeomean_u = []  # Leave-one-out geometric-average of exp(logx).
    for j in range(n):
      loogeomean_u.append(np.exp(np.mean(
          [logx[i, ...] for i in range(n) if i != j],
          axis=0)))
    loogeomean_u = np.array(loogeomean_u)
    loosum_x = []  # Leave-one-out sum of exp(logx).
    for j in range(n):
      loosum_x.append(np.sum(
          [u[i, ...] for i in range(n) if i != j],
          axis=0))
    loosum_x = np.array(loosum_x)
    # Natural log of the average u except each is swapped-out for its
    # leave-`i`-th-out Geometric average.
    log_soomean_u = np.log(loosum_x + loogeomean_u) - np.log(n)
    log_mean_u = np.log(np.mean(u, axis=0))
    return log_soomean_u, log_mean_u

  def _numpy_log_soomean_exp(self, logx, axis):
    flatten, unflatten = self._make_flatteners(logx.shape, axis)
    logx = flatten(logx)
    log_soomean_u, log_mean_u = self._numpy_log_soomean_exp_impl(logx, axis)
    log_soomean_u = unflatten(log_soomean_u)
    return log_soomean_u, log_mean_u

  def _numpy_grad_log_soomean_exp(self, logx, axis, delta):
    """Finite difference approximation of `grad(csiszar_vimco_helper, logx)`."""
    if axis != 0:
      raise NotImplementedError('Currently only `axis=0` is supported.')
    flatten, unflatten = self._make_flatteners(logx.shape, axis=axis)
    logx = flatten(logx)
    # This code actually estimates the sum of the Jacobian because that's what
    # TF's gradient calculation does.
    np_log_soomean_u1, np_log_mean_u1 = self._numpy_log_soomean_exp_impl(
        logx[..., np.newaxis] + np.diag([delta]*len(logx)),
        axis=0)
    np_log_soomean_u, np_log_mean_u = self._numpy_log_soomean_exp_impl(
        logx[..., np.newaxis], axis=0)
    grad_log_soomean_u = np.sum(
        np_log_soomean_u1 - np_log_soomean_u, axis=0) / delta
    grad_log_mean_u = (np_log_mean_u1 - np_log_mean_u) / delta
    # TODO(jvdillon): We're not actually unflattening correctly so if we want to
    # check the grad when axis!=0, we need to fix this logic.
    grad_log_soomean_u = unflatten(grad_log_soomean_u)
    return grad_log_soomean_u, grad_log_mean_u

  def test_log_soomean_exp_1(self):
    """Tests that function calculation correctly handles batches."""

    logx = np.linspace(-100., 100., 100).reshape([10, 2, 5])
    np_log_soomean_u, np_log_mean_u = self._numpy_log_soomean_exp(logx, axis=0)
    [log_soomean_u, log_mean_u] = self.evaluate(
        tfp.stats.log_soomean_exp(logx, axis=0))
    self.assertAllClose(np_log_mean_u, log_mean_u, rtol=1e-8, atol=0.)
    self.assertAllClose(np_log_soomean_u, log_soomean_u, rtol=1e-7, atol=0.)

  def test_log_soomean_exp_2(self):
    """Tests that function calculation correctly handles overflow."""

    # Using 700 (rather than 1e3) since naive numpy version can't handle higher.
    logx = np.float32([0., 700, -1, 1])
    np_log_soomean_u, np_log_mean_u = self._numpy_log_soomean_exp(logx, axis=0)
    [log_soomean_u, log_mean_u] = self.evaluate(
        tfp.stats.log_soomean_exp(logx, axis=0))
    self.assertAllClose(np_log_mean_u, log_mean_u, rtol=1e-6, atol=0.)
    self.assertAllClose(np_log_soomean_u, log_soomean_u, rtol=1e-5, atol=0.)

  def test_log_soomean_exp_3(self):
    """Tests that function calculation correctly handles underlow."""

    logx = np.float32([0., -1000, -1, 1])
    np_log_soomean_u, np_log_mean_u = self._numpy_log_soomean_exp(logx, axis=0)
    [log_soomean_u, log_mean_u] = self.evaluate(
        tfp.stats.log_soomean_exp(logx, axis=0))
    self.assertAllClose(np_log_mean_u, log_mean_u, rtol=1e-5, atol=0.)
    self.assertAllClose(np_log_soomean_u, log_soomean_u, rtol=1e-4, atol=1e-15)

  def test_log_soomean_exp_gradient_using_finite_difference_1(self):
    """Tests that gradient calculation correctly handles batches."""

    logx_ = np.linspace(-100., 100., 100).reshape([10, 2, 5])
    logx = tf.constant(logx_)

    _, grad_log_soomean_u = self.evaluate(
        tfp.math.value_and_gradient(
            lambda logx: tfp.stats.log_soomean_exp(logx, axis=0)[0],
            logx))

    _, grad_log_mean_u = self.evaluate(
        tfp.math.value_and_gradient(
            lambda logx: tfp.stats.log_soomean_exp(logx, axis=0)[1],
            logx))

    # We skip checking against finite-difference approximation since it
    # doesn't support batches.

    # Verify claim in docstring.
    self.assertAllClose(
        np.ones_like(grad_log_mean_u.sum(axis=0)),
        grad_log_mean_u.sum(axis=0))
    self.assertAllClose(
        np.ones_like(grad_log_soomean_u.mean(axis=0)),
        grad_log_soomean_u.mean(axis=0))

  def test_log_soomean_exp_gradient_using_finite_difference_2(self):
    """Tests that gradient calculation correctly handles overflow."""

    delta = 1e-3
    logx_ = np.float32([0., 1000, -1, 1])
    logx = tf.constant(logx_)

    [
        np_grad_log_soomean_u,
        np_grad_log_mean_u,
    ] = self._numpy_grad_log_soomean_exp(logx_, axis=0, delta=delta)

    _, grad_log_soomean_u = self.evaluate(
        tfp.math.value_and_gradient(
            lambda logx: tfp.stats.log_soomean_exp(logx, axis=0)[0],
            logx))

    _, grad_log_mean_u = self.evaluate(
        tfp.math.value_and_gradient(
            lambda logx: tfp.stats.log_soomean_exp(logx, axis=0)[1],
            logx))

    self.assertAllClose(np_grad_log_mean_u, grad_log_mean_u,
                        rtol=delta, atol=0.)
    self.assertAllClose(np_grad_log_soomean_u, grad_log_soomean_u,
                        rtol=delta, atol=0.)
    # Verify claim in docstring.
    self.assertAllClose(
        np.ones_like(grad_log_mean_u.sum(axis=0)),
        grad_log_mean_u.sum(axis=0))
    self.assertAllClose(
        np.ones_like(grad_log_soomean_u.mean(axis=0)),
        grad_log_soomean_u.mean(axis=0))

  def test_log_soomean_exp_gradient_using_finite_difference_3(self):
    """Tests that gradient calculation correctly handles underlow."""

    delta = 1e-3
    logx_ = np.float32([0., -1000, -1, 1])
    logx = tf.constant(logx_)

    [
        np_grad_log_soomean_u,
        np_grad_log_mean_u,
    ] = self._numpy_grad_log_soomean_exp(logx_, axis=0, delta=delta)

    _, grad_log_soomean_u = self.evaluate(
        tfp.math.value_and_gradient(
            lambda logx: tfp.stats.log_soomean_exp(logx, axis=0)[0],
            logx))

    _, grad_log_mean_u = self.evaluate(
        tfp.math.value_and_gradient(
            lambda logx: tfp.stats.log_soomean_exp(logx, axis=0)[1],
            logx))

    self.assertAllClose(np_grad_log_mean_u, grad_log_mean_u,
                        rtol=delta, atol=delta)
    self.assertAllClose(np_grad_log_soomean_u, grad_log_soomean_u,
                        rtol=delta, atol=delta)
    # Verify claim in docstring.
    self.assertAllClose(
        np.ones_like(grad_log_mean_u.sum(axis=0)),
        grad_log_mean_u.sum(axis=0))
    self.assertAllClose(
        np.ones_like(grad_log_soomean_u.mean(axis=0)),
        grad_log_soomean_u.mean(axis=0))

  def test_sum_vs_mean(self):
    logx_ = np.float32([[0., -1000, -1, 1]])
    logx = tf.constant(logx_)
    [means, sums] = self.evaluate([
        tfp.stats.log_soomean_exp(logx, axis=1),
        tfp.stats.log_soosum_exp(logx, axis=1),
    ])
    self.assertEqual((1, 4), means[0].shape)
    self.assertEqual((1,), means[1].shape)
    self.assertEqual((1, 4), sums[0].shape)
    self.assertEqual((1,), sums[1].shape)
    self.assertAllClose(means,
                        [sums[0] - np.log(4.),
                         sums[1] - np.log(4.)],
                        atol=0., rtol=1e-5)


@test_util.test_all_tf_execution_regimes
class LogLooMeanTest(test_util.TestCase):

  # We skip correctness test since SOO covers LOO.

  def test_sum_vs_mean(self):
    logx_ = np.float32([[0., -1000, -1, 1]])
    logx = tf.constant(logx_)
    [means, sums] = self.evaluate([
        tfp.stats.log_loomean_exp(logx, axis=1),
        tfp.stats.log_loosum_exp(logx, axis=1),
    ])
    self.assertEqual((1, 4), means[0].shape)
    self.assertEqual((1,), means[1].shape)
    self.assertEqual((1, 4), sums[0].shape)
    self.assertEqual((1,), sums[1].shape)
    self.assertAllClose(means,
                        [sums[0] - np.log(3.),
                         sums[1] - np.log(4.)],
                        atol=0., rtol=1e-5)


if __name__ == '__main__':
  tf.test.main()
