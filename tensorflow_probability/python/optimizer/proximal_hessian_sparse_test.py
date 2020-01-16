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
"""Tests for Proximal Hessian Optimizer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
import numpy as np

import tensorflow.compat.v1 as tf1
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp

from tensorflow_probability.python.internal import test_util


class _ProximalHessianTest(object):

  def _make_placeholder(self, x):
    return tf1.placeholder_with_default(
        x, shape=(x.shape if self.use_static_shape else None))

  def _adjust_dtype_and_shape_hints(self, x):
    x_ = tf.cast(x, self.dtype)

    # Since there is no sparse_placeholder_with_default, we manually feed in the
    # constituent dense Tensors to create a defacto placeholder SparseTensor.
    if isinstance(x_, tf.SparseTensor):
      indices_placeholder = self._make_placeholder(x_.indices)
      values_placeholder = self._make_placeholder(x_.values)

      if self.use_static_shape:
        dense_shape_placeholder = x_.dense_shape
      else:
        dense_shape_placeholder = self._make_placeholder(x_.dense_shape)

      x_ = tf.SparseTensor(
          indices=indices_placeholder,
          values=values_placeholder,
          dense_shape=dense_shape_placeholder)
    else:
      x_ = self._make_placeholder(x_)
    return x_

  def _test_finding_sparse_solution(self, batch_shape=None):
    # Test that Proximal Hessian descent prefers sparse solutions when
    # l1_regularizer is large enough.
    #
    # Define
    #
    #     Loss(x) := (x[0] - a[0])**2 + epsilon * sum(
    #                    (x[i] - a[i])**2 for i in range(1, n))
    #
    # where `a` is a constant and epsilon is small.  Set
    # l2_regularizer = 0 and set l1_regularizer such that
    #
    #     epsilon << l1_regularizer << 1.
    #
    # L1 regularization should cause the computed optimum to have zeros in all
    # but the 0th coordinate: optimal_x ~= [a[0], 0, ..., 0].
    n = 10
    epsilon = 1e-6
    if batch_shape is None:
      batch_shape = []
    # Set a[0] explicitly to make sure it's not very close to zero
    a0 = 6.
    a_ = np.concatenate([
        np.full(batch_shape + [1], a0),
        np.random.random(size=batch_shape + [n - 1])
    ],
                        axis=-1)
    a = self._adjust_dtype_and_shape_hints(a_)

    def _grad_and_hessian_unregularized_loss_fn(x):
      diff = x - a
      grad = 2. * tf.concat([diff[..., :1], epsilon * diff[..., 1:]], axis=-1)

      hessian_outer = tf.SparseTensor(
          indices=[
              b + (i, i) for i in range(n) for b in np.ndindex(*batch_shape)
          ],
          values=tf.ones(
              shape=[np.prod(batch_shape, dtype=int) * n], dtype=self.dtype),
          dense_shape=batch_shape + [n, n])

      hessian_middle_per_batch = 2 * tf.concat(
          [[1.], epsilon * tf.ones([n - 1], dtype=self.dtype)], axis=0)
      hessian_middle = tf.zeros(
          batch_shape + [n], dtype=self.dtype) + hessian_middle_per_batch
      return grad, hessian_outer, hessian_middle

    w, is_converged, num_iter = tfp.optimizer.proximal_hessian_sparse_minimize(
        _grad_and_hessian_unregularized_loss_fn,
        x_start=tf.zeros(batch_shape + [n], dtype=self.dtype),
        l1_regularizer=1e-2,
        l2_regularizer=None,
        maximum_iterations=10,
        maximum_full_sweeps_per_iteration=10,
        tolerance=1e-5,
        learning_rate=1.)

    w_, is_converged_, _ = self.evaluate([w, is_converged, num_iter])

    expected_w = tf.concat(
        [a[..., :1], tf.zeros(batch_shape + [n - 1], self.dtype)], axis=-1)

    # Using atol=0 ensures that w must be exactly zero in all coordinates
    # where expected_w is exactly zero.
    self.assertAllEqual(is_converged_, True)
    self.assertAllClose(w_, expected_w, atol=0., rtol=1e-3)

  def testFindingSparseSolution_SingleInstance(self):
    self._test_finding_sparse_solution()

  def testFindingSparseSolution_SingleBatch(self):
    self._test_finding_sparse_solution(batch_shape=[1])

  def testFindingSparseSolution_BatchOfRank2(self):
    self._test_finding_sparse_solution(batch_shape=[2, 3])

  def testL2Regularization(self):
    # Define Loss(x) := ||x - a||_2**2, where a is a constant.
    # Set l1_regularizer = 0 and l2_regularizer = 1.
    # Then the regularized loss is
    #
    #     ||x - a||_2**2 + ||x||_2**2
    #
    # And the true optimum is x = 0.5 * a.
    n = 100
    np.random.seed(42)
    a_ = np.random.random(size=(n,))
    a = self._adjust_dtype_and_shape_hints(a_)

    def _grad_and_hessian_unregularized_loss_fn(x):
      grad = 2 * (x - a)
      hessian_outer = tf.eye(n, dtype=a.dtype)
      hessian_middle = 2. * tf.ones_like(a)
      return grad, hessian_outer, hessian_middle

    w, is_converged, num_iter = tfp.optimizer.proximal_hessian_sparse_minimize(
        _grad_and_hessian_unregularized_loss_fn,
        x_start=tf.zeros_like(a_, dtype=self.dtype),
        l1_regularizer=0.,
        l2_regularizer=1.,
        maximum_iterations=4,
        maximum_full_sweeps_per_iteration=4,
        tolerance=1e-5,
        learning_rate=1.)

    w_, is_converged_, _ = self.evaluate([w, is_converged, num_iter])

    expected_w = 0.5 * a
    self.assertAllEqual(is_converged_, True)
    self.assertAllClose(w_, expected_w, atol=0., rtol=0.03)

  def testNumIter(self):
    # Same as testL2Regularization, except we set
    # maximum_full_sweeps_per_iteration = 1 and check that the number of sweeps
    # is equals what we expect it to (usually we don't know the exact number,
    # but in this simple case we do -- explanation below).
    #
    # Since l1_regularizer = 0, the soft threshold operator is actually the
    # identity operator, hence the `proximal_hessian_sparse_minimize` algorithm
    # becomes literally coordinatewise Newton's method being used to find the
    # zeros of grad Loss(x), which in this case is a linear function of x. Hence
    # Newton's method should find the exact correct answer in 1 sweep.  At the
    # end of the first sweep the algorithm does not yet know it has converged;
    # it takes a second sweep, when the algorithm notices that its answer hasn't
    # changed at all, to become aware that convergence has happened.  Hence we
    # expect two sweeps.  So with maximum_full_sweeps_per_iteration = 1, that
    # means we expect 2 iterations of the outer loop.
    n = 100
    np.random.seed(42)
    a_ = np.random.random(size=(n,))
    a = self._adjust_dtype_and_shape_hints(a_)

    def _grad_and_hessian_unregularized_loss_fn(x):
      grad = 2 * (x - a)
      hessian_outer = tf.linalg.tensor_diag(tf.ones_like(a))
      hessian_middle = 2. * tf.ones_like(a)
      return grad, hessian_outer, hessian_middle

    w, is_converged, num_iter = tfp.optimizer.proximal_hessian_sparse_minimize(
        _grad_and_hessian_unregularized_loss_fn,
        x_start=tf.zeros_like(a_, dtype=self.dtype),
        l1_regularizer=0.,
        l2_regularizer=1.,
        maximum_iterations=4,
        maximum_full_sweeps_per_iteration=1,
        tolerance=1e-5,
        learning_rate=1.)

    w_, is_converged_, num_iter_ = self.evaluate([w, is_converged, num_iter])

    expected_w = 0.5 * a
    self.assertAllEqual(is_converged_, True)
    self.assertAllEqual(num_iter_, 2)
    self.assertAllClose(w_, expected_w, atol=0., rtol=0.03)


@test_util.test_all_tf_execution_regimes
class ProximalHessianTestStaticShapeFloat32(test_util.TestCase,
                                            _ProximalHessianTest):
  dtype = tf.float32
  use_static_shape = True


@test_util.test_all_tf_execution_regimes
class ProximalHessianTestDynamicShapeFloat32(test_util.TestCase,
                                             _ProximalHessianTest):
  dtype = tf.float32
  use_static_shape = False


@test_util.test_all_tf_execution_regimes
class ProximalHessianTestStaticShapeFloat64(test_util.TestCase,
                                            _ProximalHessianTest):
  dtype = tf.float64
  use_static_shape = True


@test_util.test_all_tf_execution_regimes
class ProximalHessianTestDynamicShapeFloat64(test_util.TestCase,
                                             _ProximalHessianTest):
  dtype = tf.float64
  use_static_shape = False


if __name__ == '__main__':
  tf.test.main()
