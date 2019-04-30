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
"""GLM fitter with Hessian/proximal gradient descent based optimization.

This optimizer uses proximal gradient descent and a step size dependent on the
Hessian to efficiently minimize a convex loss function with L1 and L2
regularization.  For GLMs, we approximate the Hessian with the Fisher
information matrix.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import tensorflow as tf
import tensorflow_probability as tfp

from tensorflow_probability.python.math.linalg import sparse_or_dense_matvecmul


__all__ = [
    'fit_sparse',
    'fit_sparse_one_step',
]


def _grad_neg_log_likelihood_and_fim(model_matrix, linear_response, response,
                                     model):
  """Computes the neg-log-likelihood gradient and Fisher information for a GLM.

  Note that Fisher information is related to the Hessian of the log-likelihood
  by the equation

  ```none
  FisherInfo = E[Hessian with respect to model_coefficients of -LogLikelihood(
      Y | model_matrix, model_coefficients)]
  ```

  where `LogLikelihood` is the log-likelihood of a generalized linear model
  parameterized by `model_matrix` and `model_coefficients`, and the expectation
  is taken over Y, distributed according to the same GLM with the same parameter
  values.

  Args:
    model_matrix: (Batch of) matrix-shaped, `float` `Tensor` or `SparseTensor`
      where each row represents a sample's features.  Has shape `[N, n]` where
      `N` is the number of data samples and `n` is the number of features per
      sample.
    linear_response: (Batch of) vector-shaped `Tensor` with the same dtype as
      `model_matrix`, equal to `model_matix @ model_coefficients` where
      `model_coefficients` are the coefficients of the linear component of the
      GLM.
    response: (Batch of) vector-shaped `Tensor` with the same dtype as
      `model_matrix` where each element represents a sample's observed response
      (to the corresponding row of features).
    model: `tfp.glm.ExponentialFamily`-like instance, which specifies the link
      function and distribution of the GLM, and thus characterizes the negative
      log-likelihood. Must have sufficient statistic equal to the response, that
      is, `T(y) = y`.

  Returns:
    grad_neg_log_likelihood: (Batch of) vector-shaped `Tensor` with the same
      shape and dtype as a single row of `model_matrix`, representing the
      gradient of the negative log likelihood of `response` given linear
      response `linear_response`.
    fim_middle: (Batch of) vector-shaped `Tensor` with the same shape and dtype
      as a single column of `model_matrix`, satisfying the equation
      `Fisher information =
      Transpose(model_matrix)
      @ diag(fim_middle)
      @ model_matrix`.
  """
  # TODO(b/111926503): Determine whether there are some practical cases where it
  # is computationally favorable to compute the full FIM.
  mean, variance, grad_mean = model(linear_response)

  is_valid = (
      tf.math.is_finite(grad_mean) & tf.not_equal(grad_mean, 0.)
      & tf.math.is_finite(variance) & (variance > 0.))

  def _mask_if_invalid(x, mask):
    mask = tf.fill(
        tf.shape(input=x), value=np.array(mask, x.dtype.as_numpy_dtype))
    return tf.where(is_valid, x, mask)

  # TODO(b/111923449): Link to derivation once it's available.
  v = (response - mean) * _mask_if_invalid(grad_mean, 1) / _mask_if_invalid(
      variance, np.inf)
  grad_log_likelihood = sparse_or_dense_matvecmul(
      model_matrix, v, adjoint_a=True)
  fim_middle = _mask_if_invalid(grad_mean, 0.)**2 / _mask_if_invalid(
      variance, np.inf)
  return -grad_log_likelihood, fim_middle


def fit_sparse_one_step(model_matrix,
                        response,
                        model,
                        model_coefficients_start,
                        tolerance,
                        l1_regularizer,
                        l2_regularizer=None,
                        maximum_full_sweeps=None,
                        learning_rate=None,
                        name=None):
  """One step of (the outer loop of) the GLM fitting algorithm.

  This function returns a new value of `model_coefficients`, equal to
  `model_coefficients_start + model_coefficients_update`.  The increment
  `model_coefficients_update in R^n` is computed by a coordinate descent method,
  that is, by a loop in which each iteration updates exactly one coordinate of
  `model_coefficients_update`.  (Some updates may leave the value of the
  coordinate unchanged.)

  The particular update method used is to apply an L1-based proximity operator,
  "soft threshold", whose fixed point `model_coefficients_update^*` is the
  desired minimum

  ```none
  model_coefficients_update^* = argmin{
      -LogLikelihood(model_coefficients_start + model_coefficients_update')
        + l1_regularizer *
            ||model_coefficients_start + model_coefficients_update'||_1
        + l2_regularizer *
            ||model_coefficients_start + model_coefficients_update'||_2**2
      : model_coefficients_update' }
  ```

  where in each iteration `model_coefficients_update'` has at most one nonzero
  coordinate.

  This update method preserves sparsity, i.e., tends to find sparse solutions if
  `model_coefficients_start` is sparse.  Additionally, the choice of step size
  is based on curvature (Fisher information matrix), which significantly speeds
  up convergence.

  Args:
    model_matrix: (Batch of) matrix-shaped, `float` `Tensor` or `SparseTensor`
      where each row represents a sample's features.  Has shape `[N, n]` where
      `N` is the number of data samples and `n` is the number of features per
      sample.
    response: (Batch of) vector-shaped `Tensor` with the same dtype as
      `model_matrix` where each element represents a sample's observed response
      (to the corresponding row of features).
    model: `tfp.glm.ExponentialFamily`-like instance, which specifies the link
      function and distribution of the GLM, and thus characterizes the negative
      log-likelihood which will be minimized. Must have sufficient statistic
      equal to the response, that is, `T(y) = y`.
    model_coefficients_start: (Batch of) vector-shaped, `float` `Tensor` with
      the same dtype as `model_matrix`, representing the initial values of the
      coefficients for the GLM regression.  Has shape `[n]` where `model_matrix`
      has shape `[N, n]`.
    tolerance: scalar, `float` `Tensor` representing the convergence threshold.
      The optimization step will terminate early, returning its current value of
      `model_coefficients_start + model_coefficients_update`, once the following
      condition is met:
      `||model_coefficients_update_end - model_coefficients_update_start||_2
         / (1 + ||model_coefficients_start||_2)
       < sqrt(tolerance)`,
      where `model_coefficients_update_end` is the value of
      `model_coefficients_update` at the end of a sweep and
      `model_coefficients_update_start` is the value of
      `model_coefficients_update` at the beginning of that sweep.
    l1_regularizer: scalar, `float` `Tensor` representing the weight of the L1
      regularization term (see equation above).
    l2_regularizer: scalar, `float` `Tensor` representing the weight of the L2
      regularization term (see equation above).
      Default value: `None` (i.e., no L2 regularization).
    maximum_full_sweeps: Python integer specifying maximum number of sweeps to
      run.  A "sweep" consists of an iteration of coordinate descent on each
      coordinate. After this many sweeps, the algorithm will terminate even if
      convergence has not been reached.
      Default value: `1`.
    learning_rate: scalar, `float` `Tensor` representing a multiplicative factor
      used to dampen the proximal gradient descent steps.
      Default value: `None` (i.e., factor is conceptually `1`).
    name: Python string representing the name of the TensorFlow operation. The
      default name is `"fit_sparse_one_step"`.

  Returns:
    model_coefficients: (Batch of) `Tensor` having the same shape and dtype as
      `model_coefficients_start`, representing the updated value of
      `model_coefficients`, that is, `model_coefficients_start +
      model_coefficients_update`.
    is_converged: scalar, `bool` `Tensor` indicating whether convergence
      occurred across all batches within the specified number of sweeps.
    iter: scalar, `int` `Tensor` representing the actual number of coordinate
      updates made (before achieving convergence).  Since each sweep consists of
      `tf.size(model_coefficients_start)` iterations, the maximum number of
      updates is `maximum_full_sweeps * tf.size(model_coefficients_start)`.
  """
  graph_deps = [
      model_matrix,
      response,
      model_coefficients_start,
      l1_regularizer,
      l2_regularizer,
      maximum_full_sweeps,
      tolerance,
      learning_rate,
  ]
  with tf.compat.v1.name_scope(name, 'fit_sparse_one_step', graph_deps):
    predicted_linear_response = sparse_or_dense_matvecmul(
        model_matrix, model_coefficients_start)
    g, h_middle = _grad_neg_log_likelihood_and_fim(
        model_matrix, predicted_linear_response, response, model)

    return tfp.optimizer.proximal_hessian_sparse_one_step(
        gradient_unregularized_loss=g,
        hessian_unregularized_loss_outer=model_matrix,
        hessian_unregularized_loss_middle=h_middle,
        x_start=model_coefficients_start,
        l1_regularizer=l1_regularizer,
        l2_regularizer=l2_regularizer,
        maximum_full_sweeps=maximum_full_sweeps,
        tolerance=tolerance,
        learning_rate=learning_rate,
        name=name)


def fit_sparse(model_matrix,
               response,
               model,
               model_coefficients_start,
               tolerance,
               l1_regularizer,
               l2_regularizer=None,
               maximum_iterations=None,
               maximum_full_sweeps_per_iteration=1,
               learning_rate=None,
               name=None):
  r"""Fits a GLM using coordinate-wise FIM-informed proximal gradient descent.

  This function uses a L1- and L2-regularized, second-order quasi-Newton method
  to find maximum-likelihood parameters for the given model and observed data.
  The second-order approximations use negative Fisher information in place of
  the Hessian, that is,

  ```none
  FisherInfo = E_Y[Hessian with respect to model_coefficients of -LogLikelihood(
      Y | model_matrix, current value of model_coefficients)]
  ```

  For large, sparse data sets, `model_matrix` should be supplied as a
  `SparseTensor`.

  Args:
    model_matrix: (Batch of) matrix-shaped, `float` `Tensor` or `SparseTensor`
      where each row represents a sample's features.  Has shape `[N, n]` where
      `N` is the number of data samples and `n` is the number of features per
      sample.
    response: (Batch of) vector-shaped `Tensor` with the same dtype as
      `model_matrix` where each element represents a sample's observed response
      (to the corresponding row of features).
    model: `tfp.glm.ExponentialFamily`-like instance, which specifies the link
      function and distribution of the GLM, and thus characterizes the negative
      log-likelihood which will be minimized. Must have sufficient statistic
      equal to the response, that is, `T(y) = y`.
    model_coefficients_start: (Batch of) vector-shaped, `float` `Tensor` with
      the same dtype as `model_matrix`, representing the initial values of the
      coefficients for the GLM regression.  Has shape `[n]` where `model_matrix`
      has shape `[N, n]`.
    tolerance: scalar, `float` `Tensor` representing the tolerance for each
      optiization step; see the `tolerance` argument of `fit_sparse_one_step`.
    l1_regularizer: scalar, `float` `Tensor` representing the weight of the L1
      regularization term.
    l2_regularizer: scalar, `float` `Tensor` representing the weight of the L2
      regularization term.
      Default value: `None` (i.e., no L2 regularization).
    maximum_iterations: Python integer specifying maximum number of iterations
      of the outer loop of the optimizer (i.e., maximum number of calls to
      `fit_sparse_one_step`).  After this many iterations of the outer loop, the
      algorithm will terminate even if the return value `model_coefficients` has
      not converged.
      Default value: `1`.
    maximum_full_sweeps_per_iteration: Python integer specifying the maximum
      number of coordinate descent sweeps allowed in each iteration.
      Default value: `1`.
    learning_rate: scalar, `float` `Tensor` representing a multiplicative factor
      used to dampen the proximal gradient descent steps.
      Default value: `None` (i.e., factor is conceptually `1`).
    name: Python string representing the name of the TensorFlow operation.
      The default name is `"fit_sparse"`.

  Returns:
    model_coefficients: (Batch of) `Tensor` of the same shape and dtype as
      `model_coefficients_start`, representing the computed model coefficients
      which minimize the regularized negative log-likelihood.
    is_converged: scalar, `bool` `Tensor` indicating whether the minimization
      procedure converged across all batches within the specified number of
      iterations.  Here convergence means that an iteration of the inner loop
      (`fit_sparse_one_step`) returns `True` for its `is_converged` output
      value.
    iter: scalar, `int` `Tensor` indicating the actual number of iterations of
      the outer loop of the optimizer completed (i.e., number of calls to
      `fit_sparse_one_step` before achieving convergence).

  #### Example

  ```python
  from __future__ import print_function
  import numpy as np
  import tensorflow as tf
  import tensorflow_probability as tfp
  tfd = tfp.distributions

  def make_dataset(n, d, link, scale=1., dtype=np.float32):
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
          tfd.Normal(loc=linear_response, scale=scale).sample(seed=44) > 0,
                     dtype)
    elif link == 'logit':
      response = tfd.Bernoulli(logits=linear_response).sample(seed=44)
    else:
      raise ValueError('unrecognized true link: {}'.format(link))
    return model_matrix, response, model_coefficients, mask

  with tf.Session() as sess:
    x_, y_, model_coefficients_true_, _ = sess.run(make_dataset(
        n=int(1e5), d=100, link='probit'))

    model = tfp.glm.Bernoulli()
    model_coefficients_start = tf.zeros(x_.shape[-1], np.float32)

    model_coefficients, is_converged, num_iter = tfp.glm.fit_sparse(
        model_matrix=tf.convert_to_tensor(x_),
        response=tf.convert_to_tensor(y_),
        model=model,
        model_coefficients_start=model_coefficients_start,
        l1_regularizer=800.,
        l2_regularizer=None,
        maximum_iterations=10,
        maximum_full_sweeps_per_iteration=10,
        tolerance=1e-6,
        learning_rate=None)

    model_coefficients_, is_converged_, num_iter_ = sess.run([
        model_coefficients, is_converged, num_iter])

    print("is_converged:", is_converged_)
    print("    num_iter:", num_iter_)
    print("\nLearned / True")
    print(np.concatenate(
        [[model_coefficients_], [model_coefficients_true_]], axis=0).T)

  # ==>
  # is_converged: True
  #     num_iter: 1
  #
  # Learned / True
  # [[ 0.          0.        ]
  #  [ 0.          0.        ]
  #  [ 0.          0.        ]
  #  [ 0.11195257  0.12484948]
  #  [ 0.          0.        ]
  #  [ 0.05191106  0.06394956]
  #  [-0.15090358 -0.15325639]
  #  [-0.18187316 -0.18825999]
  #  [-0.06140942 -0.07994166]
  #  [ 0.          0.        ]
  #  [ 0.          0.        ]
  #  [ 0.          0.        ]
  #  [ 0.14474444  0.15810856]
  #  [ 0.          0.        ]
  #  [-0.25249591 -0.24260855]
  #  [ 0.          0.        ]
  #  [ 0.          0.        ]
  #  [-0.03888761 -0.06755984]
  #  [ 0.          0.        ]
  #  [ 0.          0.        ]
  #  [ 0.          0.        ]
  #  [-0.0192222  -0.04169233]
  #  [ 0.          0.        ]
  #  [ 0.          0.        ]
  #  [ 0.01434913  0.03568212]
  #  [-0.11336883 -0.12873614]
  #  [ 0.          0.        ]
  #  [-0.24496339 -0.24048163]
  #  [ 0.          0.        ]
  #  [ 0.          0.        ]
  #  [ 0.04088281  0.06565224]
  #  [-0.12784363 -0.13359821]
  #  [ 0.05618424  0.07396613]
  #  [ 0.          0.        ]
  #  [ 0.          0.        ]
  #  [ 0.          0.        ]
  #  [ 0.         -0.01719233]
  #  [ 0.          0.        ]
  #  [ 0.          0.        ]
  #  [-0.00076072 -0.03607186]
  #  [ 0.21801499  0.21146794]
  #  [-0.02161094 -0.04031265]
  #  [ 0.0918689   0.10487888]
  #  [ 0.0106154   0.03233612]
  #  [-0.07817317 -0.09725142]
  #  [ 0.          0.        ]
  #  [ 0.          0.        ]
  #  [-0.23725343 -0.24194022]
  #  [ 0.          0.        ]
  #  [-0.08725718 -0.1048776 ]
  #  [ 0.          0.        ]
  #  [ 0.          0.        ]
  #  [-0.02114314 -0.04145789]
  #  [ 0.          0.        ]
  #  [ 0.          0.        ]
  #  [-0.02710908 -0.04590397]
  #  [ 0.15293184  0.15415154]
  #  [ 0.2114463   0.2088728 ]
  #  [-0.10969634 -0.12368613]
  #  [ 0.         -0.01505797]
  #  [-0.01140458 -0.03234904]
  #  [ 0.16051085  0.1680062 ]
  #  [ 0.09816848  0.11094204]
  ```

  #### References

  [1]: Jerome Friedman, Trevor Hastie and Rob Tibshirani. Regularization Paths
       for Generalized Linear Models via Coordinate Descent. _Journal of
       Statistical Software_, 33(1), 2010.
       https://www.jstatsoft.org/article/view/v033i01/v33i01.pdf

  [2]: Guo-Xun Yuan, Chia-Hua Ho and Chih-Jen Lin. An Improved GLMNET for
       L1-regularized Logistic Regression. _Journal of Machine Learning
       Research_, 13, 2012.
       http://www.jmlr.org/papers/volume13/yuan12a/yuan12a.pdf
  """
  graph_deps = [
      model_matrix,
      response,
      model_coefficients_start,
      l1_regularizer,
      l2_regularizer,
      maximum_iterations,
      maximum_full_sweeps_per_iteration,
      # TODO(b/111925792): Replace `tolerance` arg with something like
      # `convergence_criteria_fn`.
      tolerance,
      learning_rate,
  ]
  with tf.compat.v1.name_scope(name, 'fit_sparse', graph_deps):
    # TODO(b/111922388): Include dispersion and offset parameters.
    def _grad_neg_log_likelihood_and_fim_fn(x):
      predicted_linear_response = sparse_or_dense_matvecmul(model_matrix, x)
      g, h_middle = _grad_neg_log_likelihood_and_fim(
          model_matrix, predicted_linear_response, response, model)
      return g, model_matrix, h_middle

    return tfp.optimizer.proximal_hessian_sparse_minimize(
        _grad_neg_log_likelihood_and_fim_fn,
        x_start=model_coefficients_start,
        l1_regularizer=l1_regularizer,
        l2_regularizer=l2_regularizer,
        maximum_iterations=maximum_iterations,
        maximum_full_sweeps_per_iteration=maximum_full_sweeps_per_iteration,
        learning_rate=learning_rate,
        tolerance=tolerance,
        name=name)


def _fit_sparse_exact_hessian(  # pylint: disable = missing-docstring
    model_matrix,
    response,
    model,
    model_coefficients_start,
    tolerance,
    l1_regularizer,
    l2_regularizer=None,
    maximum_iterations=None,
    maximum_full_sweeps_per_iteration=1,
    learning_rate=None,
    name=None):
  graph_deps = [
      model_matrix,
      response,
      model_coefficients_start,
      l1_regularizer,
      l2_regularizer,
      maximum_iterations,
      maximum_full_sweeps_per_iteration,
      # TODO(b/111925792): Replace `tolerance` arg with something like
      # `convergence_criteria_fn`.
      tolerance,
      learning_rate,
  ]
  with tf.compat.v1.name_scope(name, 'fit_sparse_exact_hessian', graph_deps):
    # TODO(b/111922388): Include dispersion and offset parameters.
    def _neg_log_likelihood(x):
      predicted_linear_response = sparse_or_dense_matvecmul(model_matrix, x)
      log_probs = model.log_prob(response, predicted_linear_response)
      return -log_probs

    def _grad_and_hessian_loss_fn(x):
      loss = _neg_log_likelihood(x)
      grad_loss = tf.gradients(ys=loss, xs=[x])[0]
      hessian_loss = tf.hessians(ys=loss, xs=[x])[0]
      hessian_chol = tf.linalg.cholesky(hessian_loss)
      return grad_loss, hessian_chol, tf.ones_like(grad_loss)

    return tfp.optimizer.proximal_hessian_sparse_minimize(
        _grad_and_hessian_loss_fn,
        x_start=model_coefficients_start,
        l1_regularizer=l1_regularizer,
        l2_regularizer=l2_regularizer,
        maximum_iterations=maximum_iterations,
        maximum_full_sweeps_per_iteration=maximum_full_sweeps_per_iteration,
        learning_rate=learning_rate,
        tolerance=tolerance,
        name=name)
