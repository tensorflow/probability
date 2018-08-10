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
from tensorflow_probability.python.math import matvecmul
from tensorflow.python.framework import test_util

tfd = tfp.distributions


# https://tminka.github.io/papers/logreg/minka-logreg.pdf
#
# For a given dimensionality d, feature vectors are drawn from a standard
# normal: x ~ N (0, Id). A true parameter vector is chosen randomly on the
# surface of the d-dimensional sphere with radius sqrt(2). Finally, the feature
# vectors are classified randomly according to the logistic model. Using this
# scaling of w, about 16% of the data will be mislabeled.
#
# Collins, M., Schapire, R. E., & Singer, Y. (2002). Logistic regression,
# AdaBoost and Bregman distances. Machine Learning, 48, 253--285.
# http://www.cs.princeton.edu/~schapire/papers/breg-dist.ps.gz.
def _make_dataset(n, d, link, scale=1., dtype=np.float32):
  model_coefficients = tfd.Uniform(
      low=np.array(-1, dtype), high=np.array(1, dtype)).sample(
          d, seed=42)
  radius = np.sqrt(2.)
  model_coefficients *= radius / tf.linalg.norm(model_coefficients)
  mask = tf.random_shuffle(tf.range(d)) < tf.to_int32(0.5 * tf.to_float(d))
  model_coefficients = tf.where(mask, model_coefficients,
                                tf.zeros_like(model_coefficients))
  model_matrix = tfd.Normal(
      loc=np.array(0, dtype), scale=np.array(1, dtype)).sample(
          [n, d], seed=43)
  scale = tf.convert_to_tensor(scale, dtype)
  linear_response = tf.matmul(model_matrix,
                              model_coefficients[..., tf.newaxis])[..., 0]
  if link == 'linear':
    response = tfd.Normal(loc=linear_response, scale=scale).sample(seed=44)
  elif link == 'probit':
    response = tf.cast(
        tfd.Normal(loc=linear_response, scale=scale).sample(seed=44) > 0, dtype)
  elif link == 'logit':
    response = tfd.Bernoulli(logits=linear_response).sample(seed=44)
  else:
    raise ValueError('unrecognized true link: {}'.format(link))
  return model_matrix, response, model_coefficients, mask


class _ProximalHessianTest(object):

  def adjust_dtype_and_shape_hints(self, x):
    x_ = tf.cast(x, self.dtype)
    x_ = tf.placeholder_with_default(
        input=x_, shape=(x_.shape if self.use_static_shape else None))
    return x_

  # TODO(b/111924846): The SparseTensor manipulation used here doesn't work in
  # eager mode.
  # @test_util.run_in_graph_and_eager_modes
  def testFindingSparseSolution(self):
    # Test that Proximal Hessian descent prefers sparse solutions when
    # l1_regularization_weight is large enough.
    #
    # Define
    #
    #     Loss(x) := (x[0] - a[0])**2 + epsilon * sum(
    #                    (x[i] - a[i])**2 for i in range(1, n))
    #
    # where `a` is a constant and epsilon is small.  Set
    # l2_regularization_weight = 0 and set l1_regularization_weight such that
    #
    #     epsilon << l1_regularization_weight << 1.
    #
    # L1 regularization should cause the computed optimum to have zeros in all
    # but the 0th coordinate: optimal_x ~= [a[0], 0, ..., 0].
    n = 100
    epsilon = 1e-6
    # Set a[0] explicitly to make sure it's not very close to zero
    a0 = 6.
    np.random.seed(10)
    a_ = np.concatenate([[a0], np.random.random(size=(n - 1,))], axis=0)
    a = self.adjust_dtype_and_shape_hints(a_)

    def _grad_and_hessian_unregularized_loss_fn(x):
      diff = x - a
      grad = 2. * tf.concat([[diff[0]], epsilon * diff[1:]], axis=0)
      hessian_outer = tf.SparseTensor(
          indices=[(i, i) for i in range(n)],
          values=tf.ones_like(a),
          dense_shape=[n, n])
      hessian_middle = 2. * tf.concat(
          [[1.], epsilon * tf.ones([n - 1], dtype=self.dtype)], axis=0)
      return grad, hessian_outer, hessian_middle

    w, is_converged, num_iter = minimize_sparse(
        _grad_and_hessian_unregularized_loss_fn,
        x_start=tf.zeros([n], dtype=self.dtype),
        l1_regularization_weight=1e-2,
        l2_regularization_weight=None,
        maximum_iterations=10,
        maximum_full_sweeps_per_iteration=10,
        tolerance=1e-5,
        learning_rate=1.)

    init_op = tf.global_variables_initializer()
    self.evaluate(init_op)
    w_, is_converged_, _ = self.evaluate([w, is_converged, num_iter])

    expected_w = tf.concat([[a[0]], tf.zeros([n - 1], self.dtype)], axis=0)
    # Using atol=0 ensures that w must be exactly zero in all coordinates
    # where expected_w is exactly zero.
    self.assertAllEqual(is_converged_, True)
    self.assertAllClose(w_, expected_w, atol=0., rtol=1e-3)

  @test_util.run_in_graph_and_eager_modes
  def testL2Regularization(self):
    # Define Loss(x) := ||x - a||_2**2, where a is a constant.
    # Set l1_regularization_weight = 0 and l2_regularization_weight = 1.
    # Then the regularized loss is
    #
    #     ||x - a||_2**2 + ||x||_2**2
    #
    # And the true optimum is x = 0.5 * a.
    n = 100
    np.random.seed(42)
    a_ = np.random.random(size=(n,))
    a = self.adjust_dtype_and_shape_hints(a_)

    def _grad_and_hessian_unregularized_loss_fn(x):
      grad = 2 * (x - a)
      hessian_outer = tf.eye(n, dtype=a.dtype)
      hessian_middle = 2. * tf.ones_like(a)
      return grad, hessian_outer, hessian_middle

    w, is_converged, num_iter = minimize_sparse(
        _grad_and_hessian_unregularized_loss_fn,
        x_start=tf.zeros_like(a_, dtype=self.dtype),
        l1_regularization_weight=0.,
        l2_regularization_weight=1.,
        maximum_iterations=4,
        maximum_full_sweeps_per_iteration=4,
        tolerance=1e-5,
        learning_rate=1.)

    init_op = tf.global_variables_initializer()
    self.evaluate(init_op)
    w_, is_converged_, _ = self.evaluate([w, is_converged, num_iter])

    expected_w = 0.5 * a
    self.assertAllEqual(is_converged_, True)
    self.assertAllClose(w_, expected_w, atol=0., rtol=0.03)

  @test_util.run_in_graph_and_eager_modes
  def testNumIter(self):
    # Same as testL2Regularization, except we set
    # maximum_full_sweeps_per_iteration = 1 and check that the number of sweeps
    # is equals what we expect it to (usually we don't know the exact number,
    # but in this simple case we do -- explanation below).
    #
    # Since l1_regularization_weight = 0, the soft threshold operator is
    # actually the identity operator, hence the `minimize_sparse` algorithm
    # becomes literally coordinatewise Newton's method being used to find the
    # zeros of grad Loss(x), which in this case is a linear function of x.
    # Hence Newton's method should find the exact correct answer in 1 sweep.
    # At the end of the first sweep the algorithm does not yet know it has
    # converged; it takes a second sweep, when the algorithm notices that its
    # answer hasn't changed at all, to become aware that convergence has
    # happened.  Hence we expect two sweeps.  So with
    # max_full_sweeps_per_iteration = 1, that means we expect 2 iterations of
    # the outer loop.
    n = 100
    np.random.seed(42)
    a_ = np.random.random(size=(n,))
    a = self.adjust_dtype_and_shape_hints(a_)

    def _grad_and_hessian_unregularized_loss_fn(x):
      grad = 2 * (x - a)
      hessian_outer = tf.diag(tf.ones_like(a))
      hessian_middle = 2. * tf.ones_like(a)
      return grad, hessian_outer, hessian_middle

    w, is_converged, num_iter = minimize_sparse(
        _grad_and_hessian_unregularized_loss_fn,
        x_start=tf.zeros_like(a_, dtype=self.dtype),
        l1_regularization_weight=0.,
        l2_regularization_weight=1.,
        maximum_iterations=4,
        maximum_full_sweeps_per_iteration=1,
        tolerance=1e-5,
        learning_rate=1.)

    init_op = tf.global_variables_initializer()

    self.evaluate(init_op)
    w_, is_converged_, num_iter_ = self.evaluate([w, is_converged, num_iter])

    expected_w = 0.5 * a
    self.assertAllEqual(is_converged_, True)
    self.assertAllEqual(num_iter_, 2)
    self.assertAllClose(w_, expected_w, atol=0., rtol=0.03)

  @test_util.run_in_graph_and_eager_modes
  def testTwoSweepsAreBetterThanOne(self):
    # Compare the log-likelihood after one sweep of fit_sparse to the
    # log-likelihood after two sweeps.  Expect greater log-likelihood after two
    # sweeps.  (This should be true typically but is not guaranteed to be true
    # in every case.)
    x_, y_, _, _ = self.evaluate(
        _make_dataset(n=int(1e5), d=100, link='logit'))

    model = tfp.glm.BernoulliNormalCDF()
    model_coefficients_0 = tf.zeros(x_.shape[-1], self.dtype)

    x_ = self.adjust_dtype_and_shape_hints(x_)
    y_ = self.adjust_dtype_and_shape_hints(y_)

    init_op = tf.global_variables_initializer()
    self.evaluate(init_op)

    x_update_var = tf.get_variable(
        name='x_update_var',
        initializer=tf.zeros_like(model_coefficients_0),
        trainable=False,
        use_resource=True)

    model_coefficients_1, is_converged, _ = tfp.glm.fit_sparse_one_step(
        model_matrix=x_,
        response=y_,
        model=model,
        model_coefficients_start=model_coefficients_0,
        l1_regularizer=800.,
        l2_regularizer=None,
        maximum_full_sweeps=1,
        tolerance=1e-6,
        learning_rate=None,
        model_coefficients_update_var=x_update_var)
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
        learning_rate=None,
        model_coefficients_update_var=x_update_var)
    model_coefficients_2_ = self.evaluate(model_coefficients_2)

    def _joint_log_prob(model_coefficients_):
      predicted_linear_response_ = matvecmul(x_, model_coefficients_)
      return tf.reduce_sum(model.log_prob(y_, predicted_linear_response_))

    self.assertAllGreater(
        _joint_log_prob(model_coefficients_2_) -
        _joint_log_prob(model_coefficients_1_), 0)

  @test_util.run_in_graph_and_eager_modes
  def testFitGLMFromData(self):
    # Run fit_sparse where the loss function is negative log likelihood of a
    # synthetic data set generated from a similar model (probit vs. logit).
    # Expect the returned value of model_coefficients to be close to the true
    # parameters.
    x_, y_, model_coefficients_true_, _ = self.evaluate(
        _make_dataset(n=int(1e5), d=100, link='probit'))

    model = tfp.glm.Bernoulli()
    model_coefficients_start = tf.zeros(x_.shape[-1], self.dtype)

    with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
      model_coefficients, is_converged, num_iter = tfp.glm.fit_sparse(
          self.adjust_dtype_and_shape_hints(x_),
          self.adjust_dtype_and_shape_hints(y_),
          model,
          model_coefficients_start,
          l1_regularizer=800.,
          l2_regularizer=None,
          maximum_iterations=10,
          maximum_full_sweeps_per_iteration=10,
          tolerance=1e-6,
          learning_rate=None)
    model_coefficients_, is_converged_, _ = self.evaluate(
        [model_coefficients, is_converged, num_iter])

    self.assertAllEqual(is_converged_, True)
    self.assertAllClose(
        model_coefficients_, model_coefficients_true_, atol=0.1, rtol=0.1)


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
