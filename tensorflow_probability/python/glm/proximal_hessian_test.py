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
"""Tests for GLM fitting with Proximal Hessian method."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
import numpy as np

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability.python.glm.proximal_hessian import minimize_sparse
from tensorflow.contrib.layers.python.ops import sparse_ops
from tensorflow.python.framework import test_util

tfd = tfp.distributions


@test_util.run_all_in_graph_and_eager_modes
class _ProximalHessianTest(object):

  # https://tminka.github.io/papers/logreg/minka-logreg.pdf
  #
  # For a given dimensionality d, feature vectors are drawn from a standard
  # normal: x ~ N (0, Id). A true parameter vector is chosen randomly on the
  # surface of the d-dimensional sphere with radius sqrt(2). Finally, the
  # feature vectors are classified randomly according to the logistic model.
  # Using this scaling of w, about 16% of the data will be mislabeled.
  #
  # Collins, M., Schapire, R. E., & Singer, Y. (2002). Logistic regression,
  # AdaBoost and Bregman distances. Machine Learning, 48, 253--285.
  # http://www.cs.princeton.edu/~schapire/papers/breg-dist.ps.gz.
  def _make_dataset(self,
                    n,
                    d,
                    link,
                    scale=1.,
                    batch_shape=None,
                    dtype=np.float32,
                    seed=42):
    seed = tfd.SeedStream(seed=seed, salt='tfp.glm.proximal_hessian_test')

    if batch_shape is None:
      batch_shape = []

    model_coefficients = tfd.Uniform(
        low=np.array(-1, dtype), high=np.array(1, dtype)).sample(
            batch_shape + [d], seed=seed())

    radius = np.sqrt(2.)
    model_coefficients *= (
        radius / tf.linalg.norm(model_coefficients, axis=-1)[..., tf.newaxis])

    mask = tfd.Bernoulli(probs=0.5, dtype=tf.bool).sample(batch_shape + [d])
    model_coefficients = tf.where(mask, model_coefficients,
                                  tf.zeros_like(model_coefficients))
    model_matrix = tfd.Normal(
        loc=np.array(0, dtype), scale=np.array(1, dtype)).sample(
            batch_shape + [n, d], seed=seed())
    scale = tf.convert_to_tensor(scale, dtype)
    linear_response = tf.matmul(model_matrix,
                                model_coefficients[..., tf.newaxis])[..., 0]

    if link == 'linear':
      response = tfd.Normal(
          loc=linear_response, scale=scale).sample(seed=seed())
    elif link == 'probit':
      response = tf.cast(
          tfd.Normal(loc=linear_response, scale=scale).sample(seed=seed()) > 0,
          dtype)
    elif link == 'logit':
      response = tfd.Bernoulli(logits=linear_response).sample(seed=seed())
    else:
      raise ValueError('unrecognized true link: {}'.format(link))
    return self.evaluate([model_matrix, response, model_coefficients, mask])

  def _make_placeholder(self, x):
    return tf.placeholder_with_default(
        input=x, shape=(x.shape if self.use_static_shape else None))

  def _adjust_dtype_and_shape_hints(self, x):
    x_ = tf.cast(x, self.dtype)

    # Since there is no sparse_placeholder_with_default, we manually feed in the
    # constituent dense Tensors to create a defacto placeholder SparseTensor.
    if isinstance(x_, tf.SparseTensor):
      indices_placeholder = self._make_placeholder(x_.indices)
      values_placeholder = self._make_placeholder(x_.values)
      dense_shape_placeholder = (
          x_.dense_shape if self.use_static_shape else
          self._make_placeholder(x_.dense_shape))
      x_ = tf.SparseTensor(
          indices=indices_placeholder,
          values=values_placeholder,
          dense_shape=dense_shape_placeholder)
    else:
      x_ = self._make_placeholder(x_)
    return x_

  def _prepare_inputs_for_fit_sparse(self,
                                     model_matrix,
                                     response,
                                     model_coefficients_start=None,
                                     convert_to_sparse_tensor=False):
    if model_coefficients_start is None:
      model_coefficients_start = np.zeros(model_matrix.shape[:-2] +
                                          model_matrix.shape[-1:])
    if convert_to_sparse_tensor:
      model_matrix = sparse_ops.dense_to_sparse_tensor(model_matrix)

    model_matrix = self._adjust_dtype_and_shape_hints(model_matrix)
    response = self._adjust_dtype_and_shape_hints(response)
    model_coefficients_start = self._adjust_dtype_and_shape_hints(
        model_coefficients_start)

    return model_matrix, response, model_coefficients_start

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
          values=tf.ones(shape=[np.product(batch_shape) * n], dtype=self.dtype),
          dense_shape=batch_shape + [n, n])

      hessian_middle_per_batch = 2 * tf.concat(
          [[1.], epsilon * tf.ones([n - 1], dtype=self.dtype)], axis=0)
      hessian_middle = tf.zeros(
          batch_shape + [n], dtype=self.dtype) + hessian_middle_per_batch
      return grad, hessian_outer, hessian_middle

    w, is_converged, num_iter = minimize_sparse(
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

    w, is_converged, num_iter = minimize_sparse(
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
    # identity operator, hence the `minimize_sparse` algorithm becomes literally
    # coordinatewise Newton's method being used to find the zeros of grad
    # Loss(x), which in this case is a linear function of x.  Hence Newton's
    # method should find the exact correct answer in 1 sweep.  At the end of the
    # first sweep the algorithm does not yet know it has converged; it takes a
    # second sweep, when the algorithm notices that its answer hasn't changed at
    # all, to become aware that convergence has happened.  Hence we expect two
    # sweeps.  So with maximum_full_sweeps_per_iteration = 1, that means we
    # expect 2 iterations of the outer loop.
    n = 100
    np.random.seed(42)
    a_ = np.random.random(size=(n,))
    a = self._adjust_dtype_and_shape_hints(a_)

    def _grad_and_hessian_unregularized_loss_fn(x):
      grad = 2 * (x - a)
      hessian_outer = tf.diag(tf.ones_like(a))
      hessian_middle = 2. * tf.ones_like(a)
      return grad, hessian_outer, hessian_middle

    w, is_converged, num_iter = minimize_sparse(
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

  def testTwoSweepsAreBetterThanOne(self):
    # Compare the log-likelihood after one sweep of fit_sparse to the
    # log-likelihood after two sweeps.  Expect greater log-likelihood after two
    # sweeps.  (This should be true typically but is not guaranteed to be true
    # in every case.)
    x_, y_, _, _ = self._make_dataset(n=int(1e5), d=100, link='logit')

    model = tfp.glm.BernoulliNormalCDF()
    model_coefficients_0 = tf.zeros(x_.shape[-1], self.dtype)

    x_ = self._adjust_dtype_and_shape_hints(x_)
    y_ = self._adjust_dtype_and_shape_hints(y_)

    model_coefficients_1, is_converged, _ = tfp.glm.fit_sparse_one_step(
        model_matrix=x_,
        response=y_,
        model=model,
        model_coefficients_start=model_coefficients_0,
        l1_regularizer=800.,
        l2_regularizer=None,
        maximum_full_sweeps=1,
        tolerance=1e-6,
        learning_rate=None)
    model_coefficients_1_ = self.evaluate(model_coefficients_1)

    self.assertAllEqual(False, is_converged)

    model_coefficients_2, _, _ = tfp.glm.fit_sparse_one_step(
        model_matrix=x_,
        response=y_,
        model=model,
        model_coefficients_start=tf.convert_to_tensor(model_coefficients_1_),
        l1_regularizer=800.,
        l2_regularizer=None,
        maximum_full_sweeps=1,
        tolerance=1e-6,
        learning_rate=None)
    model_coefficients_2_ = self.evaluate(model_coefficients_2)

    def _joint_log_prob(model_coefficients_):
      predicted_linear_response_ = tf.linalg.matvec(x_, model_coefficients_)
      return tf.reduce_sum(model.log_prob(y_, predicted_linear_response_))

    self.assertAllGreater(
        _joint_log_prob(model_coefficients_2_) -
        _joint_log_prob(model_coefficients_1_), 0)

  def _test_fit_glm_from_data(self,
                              n,
                              d,
                              link,
                              model,
                              batch_shape=None,
                              use_sparse_tensor=False):
    if batch_shape is None:
      batch_shape = []
    # Create synthetic data according to the given `link` function.
    model_matrix_, response_, model_coefficients_true_, _ = self._make_dataset(
        n=n, d=d, link=link, batch_shape=batch_shape)

    # Run tfp.glm.fit_sparse on the synthetic data for the given model.
    # Also adjust dtype and shape hints depending on the test mode.
    model_matrix, response, model_coefficients_start = (
        self._prepare_inputs_for_fit_sparse(
            model_matrix_,
            response_,
            convert_to_sparse_tensor=use_sparse_tensor))

    model_coefficients_, is_converged_, _ = self.evaluate(
        tfp.glm.fit_sparse(
            model_matrix,
            response,
            model,
            model_coefficients_start,
            l1_regularizer=800.,
            l2_regularizer=None,
            maximum_iterations=10,
            maximum_full_sweeps_per_iteration=10,
            tolerance=1e-6,
            learning_rate=None))

    # Ensure that we have converged and learned coefficients are close to the
    # true coefficients.
    self.assertAllEqual(is_converged_, True)
    self.assertAllClose(
        model_coefficients_, model_coefficients_true_, atol=0.1, rtol=0.1)

  def testFitGLMFromData_SimilarModel(self):
    # Run fit_sparse where the loss function is negative log likelihood of a
    # synthetic data set generated from a similar model (probit vs. logit).
    # Expect the returned value of model_coefficients to be close to the true
    # parameters.
    self._test_fit_glm_from_data(
        n=int(1e5),
        d=100,
        link='probit',
        model=tfp.glm.Bernoulli(),
        batch_shape=None)

  def testFitGLMFromData_SingleBatch(self):
    batch_shape = [1]
    self._test_fit_glm_from_data(
        n=int(1e4),
        d=100,
        link='linear',
        model=tfp.glm.Normal(),
        batch_shape=batch_shape)

  def testFitGLMFromData_BatchOfRank1(self):
    batch_shape = [3]
    self._test_fit_glm_from_data(
        n=int(1e4),
        d=25,
        link='linear',
        model=tfp.glm.Normal(),
        batch_shape=batch_shape)

  def testFitGLMFromData_BatchOfRank2(self):
    batch_shape = [3, 2]
    self._test_fit_glm_from_data(
        n=int(1e4),
        d=25,
        link='linear',
        model=tfp.glm.Normal(),
        batch_shape=batch_shape)

  def testFitGLMFromData_SparseTensorSingleBatch(self):
    batch_shape = [1]
    self._test_fit_glm_from_data(
        n=int(1e4),
        d=25,
        link='linear',
        model=tfp.glm.Normal(),
        batch_shape=batch_shape,
        use_sparse_tensor=True)

  def testFitGLMFromData_SparseTensorBatchOfRank1(self):
    batch_shape = [3]
    self._test_fit_glm_from_data(
        n=int(1e4),
        d=25,
        link='linear',
        model=tfp.glm.Normal(),
        batch_shape=batch_shape,
        use_sparse_tensor=True)

  def testFitGLMFromData_SparseTensorBatchOfRank2(self):
    batch_shape = [2, 3]
    self._test_fit_glm_from_data(
        n=int(1e4),
        d=25,
        link='linear',
        model=tfp.glm.Normal(),
        batch_shape=batch_shape,
        use_sparse_tensor=True)

  def testFitGLMFromData_SparseTensorBatchOfRank3(self):
    batch_shape = [2, 1, 2]
    self._test_fit_glm_from_data(
        n=int(1e4),
        d=25,
        link='linear',
        model=tfp.glm.Normal(),
        batch_shape=batch_shape,
        use_sparse_tensor=True)

  def _test_compare_batch_to_single_instance(self, use_sparse_tensor=False):
    n = int(1e4)
    d = 25
    link = 'linear'
    model = tfp.glm.Normal()

    # Create two sets of synthetic data according to the given `link` function.
    model_matrix_1_, response_1_, _, _ = self._make_dataset(
        n=n, d=d, link=link, seed=41)
    model_matrix_2_, response_2_, _, _ = self._make_dataset(
        n=n, d=d, link=link, seed=42)

    # Fit both the batches of data individually.
    model_matrix_1, response_1, model_coefficients_start_1 = (
        self._prepare_inputs_for_fit_sparse(
            model_matrix_1_,
            response_1_,
            convert_to_sparse_tensor=use_sparse_tensor))

    model_coefficients_1_, _, _ = self.evaluate(
        tfp.glm.fit_sparse(
            model_matrix_1,
            response_1,
            model,
            model_coefficients_start_1,
            l1_regularizer=800.,
            l2_regularizer=None,
            maximum_iterations=10,
            maximum_full_sweeps_per_iteration=10,
            tolerance=1e-6,
            learning_rate=None))

    model_matrix_2, response_2, model_coefficients_start_2 = (
        self._prepare_inputs_for_fit_sparse(
            model_matrix_2_,
            response_2_,
            convert_to_sparse_tensor=use_sparse_tensor))

    model_coefficients_2_, _, _ = self.evaluate(
        tfp.glm.fit_sparse(
            model_matrix_2,
            response_2,
            model,
            model_coefficients_start_2,
            l1_regularizer=800.,
            l2_regularizer=None,
            maximum_iterations=10,
            maximum_full_sweeps_per_iteration=10,
            tolerance=1e-6,
            learning_rate=None))

    # Combine the data into a single batch of 2 and fit the batched data.
    model_matrix_ = np.stack([model_matrix_1_, model_matrix_2_])
    response_ = np.stack([response_1_, response_2_])

    model_matrix, response, model_coefficients_start = (
        self._prepare_inputs_for_fit_sparse(
            model_matrix_,
            response_,
            convert_to_sparse_tensor=use_sparse_tensor))

    model_coefficients_, _, _ = self.evaluate(
        tfp.glm.fit_sparse(
            model_matrix,
            response,
            model,
            model_coefficients_start,
            l1_regularizer=800.,
            l2_regularizer=None,
            maximum_iterations=10,
            maximum_full_sweeps_per_iteration=10,
            tolerance=1e-6,
            learning_rate=None))

    # Ensure that the learned coefficients from the individual samples are close
    # to those learned from the batched samples.
    self.assertAllClose(
        model_coefficients_1_, model_coefficients_[0], atol=0., rtol=1e-3)
    self.assertAllClose(
        model_coefficients_2_, model_coefficients_[1], atol=0., rtol=1e-3)

  def testCompareBatchResultsToSingleInstance_Dense(self):
    self._test_compare_batch_to_single_instance(use_sparse_tensor=False)

  def testCompareBatchResultsToSingleInstance_Sparse(self):
    self._test_compare_batch_to_single_instance(use_sparse_tensor=True)


class ProximalHessianTestStaticShapeFloat32(tf.test.TestCase,
                                            _ProximalHessianTest):
  dtype = tf.float32
  use_static_shape = True


class ProximalHessianTestDynamicShapeFloat32(tf.test.TestCase,
                                             _ProximalHessianTest):
  dtype = tf.float32
  use_static_shape = False


class ProximalHessianTestStaticShapeFloat64(tf.test.TestCase,
                                            _ProximalHessianTest):
  dtype = tf.float64
  use_static_shape = True


class ProximalHessianTestDynamicShapeFloat64(tf.test.TestCase,
                                             _ProximalHessianTest):
  dtype = tf.float64
  use_static_shape = False


if __name__ == '__main__':
  tf.test.main()
