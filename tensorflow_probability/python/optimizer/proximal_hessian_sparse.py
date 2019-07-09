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
"""Hessian-type proximal descent optimizer.

This optimizer uses proximal gradient descent and a step size dependent on the
Hessian to efficiently minimize a convex loss function with L1 and L2
regularization.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from tensorflow_probability.python.internal import prefer_static
from tensorflow_probability.python.math.generic import soft_threshold
from tensorflow_probability.python.math.linalg import sparse_or_dense_matvecmul

__all__ = [
    'minimize',
    'minimize_one_step',
]


def _reduce_ignoring_nones(fn, args):
  result = None
  for x in args:
    if x is not None:
      result = (x if result is None else fn(result, x))
  return result


def _reduce_exiting_on_none(fn, args):
  args = list(args)
  if any(x is None for x in args):
    return None
  return _reduce_ignoring_nones(fn, args)


def _add_ignoring_nones(*args):
  return _reduce_ignoring_nones(lambda a, b: a + b, args)


def _mul_ignoring_nones(*args):
  return _reduce_ignoring_nones(lambda a, b: a * b, args)


def _mul_or_none(*args):
  return _reduce_exiting_on_none(lambda a, b: a * b, args)


def _get_shape(x, out_type=tf.int32):
  # Return the shape of a Tensor or a SparseTensor as an np.array if its shape
  # is known statically. Otherwise return a Tensor representing the shape.
  if x.shape.is_fully_defined():
    return np.array(x.shape.as_list(), dtype=out_type.as_numpy_dtype)
  return tf.shape(input=x, out_type=out_type)


def _sparse_or_dense_matmul_onehot(sparse_or_dense_matrix, col_index):
  """Returns a (dense) column of a Tensor or SparseTensor.

  Args:
    sparse_or_dense_matrix: matrix-shaped, `float` `Tensor` or `SparseTensor`.
    col_index: scalar, `int` `Tensor` representing the index of the desired
      column.

  Returns:
    column: vector-shaped, `float` `Tensor` with the same dtype as
      `sparse_or_dense_matrix`, representing the `col_index`th column of
      `sparse_or_dense_matrix`.
  """
  if isinstance(sparse_or_dense_matrix,
                (tf.SparseTensor, tf.compat.v1.SparseTensorValue)):
    # TODO(b/111924846): Implement better (ideally in a way that allows us to
    # eliminate the `num_rows` arg, if possible).
    num_rows = _get_shape(sparse_or_dense_matrix)[-2]
    batch_shape = _get_shape(sparse_or_dense_matrix)[:-2]
    slice_start = tf.concat([tf.zeros_like(batch_shape), [0, col_index]],
                            axis=0)
    slice_size = tf.concat([batch_shape, [num_rows, 1]], axis=0)
    # We momentarily lose static shape information in tf.sparse_slice. However
    # we regain it in the following tf.reshape.
    sparse_slice = tf.sparse.slice(sparse_or_dense_matrix,
                                   tf.cast(slice_start, tf.int64),
                                   tf.cast(slice_size, tf.int64))

    output_shape = tf.concat([batch_shape, [num_rows]], axis=0)
    return tf.reshape(tf.sparse.to_dense(sparse_slice), output_shape)
  else:
    return tf.gather(sparse_or_dense_matrix, col_index, axis=-1)


def _one_hot_like(x, indices, on_value=None):
  output_dtype = x.dtype.base_dtype
  if tf.compat.dimension_value(x.shape[-1]) is None:
    depth = tf.shape(input=x)[-1]
  else:
    depth = tf.compat.dimension_value(x.shape[-1])
  if on_value is not None:
    on_value = tf.cast(on_value, output_dtype)
  return tf.one_hot(indices, depth=depth, on_value=on_value, dtype=output_dtype)


def minimize_one_step(gradient_unregularized_loss,
                      hessian_unregularized_loss_outer,
                      hessian_unregularized_loss_middle,
                      x_start,
                      tolerance,
                      l1_regularizer,
                      l2_regularizer=None,
                      maximum_full_sweeps=1,
                      learning_rate=None,
                      name=None):
  """One step of (the outer loop of) the minimization algorithm.

  This function returns a new value of `x`, equal to `x_start + x_update`.  The
  increment `x_update in R^n` is computed by a coordinate descent method, that
  is, by a loop in which each iteration updates exactly one coordinate of
  `x_update`.  (Some updates may leave the value of the coordinate unchanged.)

  The particular update method used is to apply an L1-based proximity operator,
  "soft threshold", whose fixed point `x_update_fix` is the desired minimum

  ```none
  x_update_fix = argmin{
      Loss(x_start + x_update')
        + l1_regularizer * ||x_start + x_update'||_1
        + l2_regularizer * ||x_start + x_update'||_2**2
      : x_update' }
  ```

  where in each iteration `x_update'` is constrained to have at most one nonzero
  coordinate.

  This update method preserves sparsity, i.e., tends to find sparse solutions if
  `x_start` is sparse.  Additionally, the choice of step size is based on
  curvature (Hessian), which significantly speeds up convergence.

  This algorithm assumes that `Loss` is convex, at least in a region surrounding
  the optimum.  (If `l2_regularizer > 0`, then only weak convexity is needed.)

  Args:
    gradient_unregularized_loss: (Batch of) `Tensor` with the same shape and
      dtype as `x_start` representing the gradient, evaluated at `x_start`, of
      the unregularized loss function (denoted `Loss` above).  (In all current
      use cases, `Loss` is the negative log likelihood.)
    hessian_unregularized_loss_outer: (Batch of) `Tensor` or `SparseTensor`
      having the same dtype as `x_start`, and shape `[N, n]` where `x_start` has
      shape `[n]`, satisfying the property
      `Transpose(hessian_unregularized_loss_outer)
      @ diag(hessian_unregularized_loss_middle)
      @ hessian_unregularized_loss_inner
      = (approximation of) Hessian matrix of Loss, evaluated at x_start`.
    hessian_unregularized_loss_middle: (Batch of) vector-shaped `Tensor` having
      the same dtype as `x_start`, and shape `[N]` where
      `hessian_unregularized_loss_outer` has shape `[N, n]`, satisfying the
      property
      `Transpose(hessian_unregularized_loss_outer)
      @ diag(hessian_unregularized_loss_middle)
      @ hessian_unregularized_loss_inner
      = (approximation of) Hessian matrix of Loss, evaluated at x_start`.
    x_start: (Batch of) vector-shaped, `float` `Tensor` representing the current
      value of the argument to the Loss function.
    tolerance: scalar, `float` `Tensor` representing the convergence threshold.
      The optimization step will terminate early, returning its current value of
      `x_start + x_update`, once the following condition is met:
      `||x_update_end - x_update_start||_2 / (1 + ||x_start||_2)
      < sqrt(tolerance)`,
      where `x_update_end` is the value of `x_update` at the end of a sweep and
      `x_update_start` is the value of `x_update` at the beginning of that
      sweep.
    l1_regularizer: scalar, `float` `Tensor` representing the weight of the L1
      regularization term (see equation above).  If L1 regularization is not
      required, then `tfp.glm.fit_one_step` is preferable.
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
    name: Python string representing the name of the TensorFlow operation.
      The default name is `"minimize_one_step"`.

  Returns:
    x: (Batch of) `Tensor` having the same shape and dtype as `x_start`,
      representing the updated value of `x`, that is, `x_start + x_update`.
    is_converged: scalar, `bool` `Tensor` indicating whether convergence
      occurred across all batches within the specified number of sweeps.
    iter: scalar, `int` `Tensor` representing the actual number of coordinate
      updates made (before achieving convergence).  Since each sweep consists of
      `tf.size(x_start)` iterations, the maximum number of updates is
      `maximum_full_sweeps * tf.size(x_start)`.

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
      gradient_unregularized_loss,
      hessian_unregularized_loss_outer,
      hessian_unregularized_loss_middle,
      x_start,
      l1_regularizer,
      l2_regularizer,
      maximum_full_sweeps,
      tolerance,
      learning_rate,
  ]
  with tf.compat.v1.name_scope(name, 'minimize_one_step', graph_deps):
    x_shape = _get_shape(x_start)
    batch_shape = x_shape[:-1]
    dims = x_shape[-1]

    def _hessian_diag_elt_with_l2(coord):  # pylint: disable=missing-docstring
      # Returns the (coord, coord) entry of
      #
      #   Hessian(UnregularizedLoss(x) + l2_regularizer * ||x||_2**2)
      #
      # evaluated at x = x_start.
      inner_square = tf.reduce_sum(
          input_tensor=_sparse_or_dense_matmul_onehot(
              hessian_unregularized_loss_outer, coord)**2,
          axis=-1)
      unregularized_component = (
          hessian_unregularized_loss_middle[..., coord] * inner_square)
      l2_component = _mul_or_none(2., l2_regularizer)
      return _add_ignoring_nones(unregularized_component, l2_component)

    grad_loss_with_l2 = _add_ignoring_nones(
        gradient_unregularized_loss, _mul_or_none(2., l2_regularizer, x_start))

    # We define `x_update_diff_norm_sq_convergence_threshold` such that the
    # convergence condition
    #     ||x_update_end - x_update_start||_2 / (1 + ||x_start||_2)
    #     < sqrt(tolerance)
    # is equivalent to
    #     ||x_update_end - x_update_start||_2**2
    #     < x_update_diff_norm_sq_convergence_threshold.
    x_update_diff_norm_sq_convergence_threshold = (
        tolerance * (1. + tf.norm(tensor=x_start, ord=2, axis=-1))**2)

    # Reshape update vectors so that the coordinate sweeps happen along the
    # first dimension. This is so that we can use tensor_scatter_update to make
    # sparse updates along the first axis without copying the Tensor.
    # TODO(b/118789120): Switch to something like tf.tensor_scatter_nd_add if
    # or when it exists.
    update_shape = tf.concat([[dims], batch_shape], axis=-1)

    def _loop_cond(iter_, x_update_diff_norm_sq, x_update,
                   hess_matmul_x_update):
      del x_update
      del hess_matmul_x_update
      sweep_complete = (iter_ > 0) & tf.equal(iter_ % dims, 0)
      small_delta = (
          x_update_diff_norm_sq < x_update_diff_norm_sq_convergence_threshold)
      converged = sweep_complete & small_delta
      allowed_more_iterations = iter_ < maximum_full_sweeps * dims
      return allowed_more_iterations & tf.reduce_any(input_tensor=~converged)

    def _loop_body(  # pylint: disable=missing-docstring
        iter_, x_update_diff_norm_sq, x_update, hess_matmul_x_update):
      # Inner loop of the minimizer.
      #
      # This loop updates a single coordinate of x_update.  Ideally, an
      # iteration of this loop would set
      #
      #   x_update[j] += argmin{ LocalLoss(x_update + z*e_j) : z in R }
      #
      # where
      #
      #   LocalLoss(x_update')
      #     = LocalLossSmoothComponent(x_update')
      #         + l1_regularizer * (||x_start + x_update'||_1 -
      #                             ||x_start + x_update||_1)
      #    := (UnregularizedLoss(x_start + x_update') -
      #        UnregularizedLoss(x_start + x_update)
      #         + l2_regularizer * (||x_start + x_update'||_2**2 -
      #                             ||x_start + x_update||_2**2)
      #         + l1_regularizer * (||x_start + x_update'||_1 -
      #                             ||x_start + x_update||_1)
      #
      # In this algorithm approximate the above argmin using (univariate)
      # proximal gradient descent:
      #
      # (*)  x_update[j] = prox_{t * l1_regularizer * L1}(
      #                 x_update[j] -
      #                 t * d/dz|z=0 UnivariateLocalLossSmoothComponent(z))
      #
      # where
      #
      #   UnivariateLocalLossSmoothComponent(z)
      #       := LocalLossSmoothComponent(x_update + z*e_j)
      #
      # and we approximate
      #
      #       d/dz UnivariateLocalLossSmoothComponent(z)
      #     = grad LocalLossSmoothComponent(x_update))[j]
      #    ~= (grad LossSmoothComponent(x_start)
      #         + x_update matmul HessianOfLossSmoothComponent(x_start))[j].
      #
      # To choose the parameter t, we squint and pretend that the inner term of
      # (*) is a Newton update as if we were using Newton's method to minimize
      # UnivariateLocalLossSmoothComponent.  That is, we choose t such that
      #
      #   -t * d/dz ULLSC = -learning_rate * (d/dz ULLSC) / (d^2/dz^2 ULLSC)
      #
      # at z=0.  Hence
      #
      #   t = learning_rate / (d^2/dz^2|z=0 ULLSC)
      #     = learning_rate / HessianOfLossSmoothComponent(
      #                           x_start + x_update)[j,j]
      #    ~= learning_rate / HessianOfLossSmoothComponent(
      #                           x_start)[j,j]
      #
      # The above approximation is equivalent to assuming that
      # HessianOfUnregularizedLoss is constant, i.e., ignoring third-order
      # effects.
      #
      # Note that because LossSmoothComponent is (assumed to be) convex, t is
      # positive.

      # In above notation, coord = j.
      coord = iter_ % dims
      # x_update_diff_norm_sq := ||x_update_end - x_update_start||_2**2,
      # computed incrementally, where x_update_end and x_update_start are as
      # defined in the convergence criteria.  Accordingly, we reset
      # x_update_diff_norm_sq to zero at the beginning of each sweep.
      x_update_diff_norm_sq = tf.compat.v1.where(
          tf.equal(coord, 0), tf.zeros_like(x_update_diff_norm_sq),
          x_update_diff_norm_sq)

      # Recall that x_update and hess_matmul_x_update has the rightmost
      # dimension transposed to the leftmost dimension.
      w_old = x_start[..., coord] + x_update[coord, ...]
      # This is the coordinatewise Newton update if no L1 regularization.
      # In above notation, newton_step = -t * (approximation of d/dz|z=0 ULLSC).
      second_deriv = _hessian_diag_elt_with_l2(coord)
      newton_step = -_mul_ignoring_nones(  # pylint: disable=invalid-unary-operand-type
          learning_rate, grad_loss_with_l2[..., coord] +
          hess_matmul_x_update[coord, ...]) / second_deriv

      # Applying the soft-threshold operator accounts for L1 regularization.
      # In above notation, delta =
      #     prox_{t*l1_regularizer*L1}(w_old + newton_step) - w_old.
      delta = (
          soft_threshold(
              w_old + newton_step,
              _mul_ignoring_nones(learning_rate, l1_regularizer) / second_deriv)
          - w_old)

      def _do_update(x_update_diff_norm_sq, x_update, hess_matmul_x_update):  # pylint: disable=missing-docstring
        hessian_column_with_l2 = sparse_or_dense_matvecmul(
            hessian_unregularized_loss_outer,
            hessian_unregularized_loss_middle * _sparse_or_dense_matmul_onehot(
                hessian_unregularized_loss_outer, coord),
            adjoint_a=True)

        if l2_regularizer is not None:
          hessian_column_with_l2 += _one_hot_like(
              hessian_column_with_l2, coord, on_value=2. * l2_regularizer)

        # Move the batch dimensions of `hessian_column_with_l2` to rightmost in
        # order to conform to `hess_matmul_x_update`.
        n = tf.rank(hessian_column_with_l2)
        perm = tf.roll(tf.range(n), shift=1, axis=0)
        hessian_column_with_l2 = tf.transpose(
            a=hessian_column_with_l2, perm=perm)

        # Update the entire batch at `coord` even if `delta` may be 0 at some
        # batch coordinates. In those cases, adding `delta` is a no-op.
        x_update = tf.tensor_scatter_nd_add(x_update, [[coord]], [delta])

        with tf.control_dependencies([x_update]):
          x_update_diff_norm_sq_ = x_update_diff_norm_sq + delta**2
          hess_matmul_x_update_ = (
              hess_matmul_x_update + delta * hessian_column_with_l2)

          # Hint that loop vars retain the same shape.
          x_update_diff_norm_sq_.set_shape(
              x_update_diff_norm_sq_.shape.merge_with(
                  x_update_diff_norm_sq.shape))
          hess_matmul_x_update_.set_shape(
              hess_matmul_x_update_.shape.merge_with(
                  hess_matmul_x_update.shape))

          return [x_update_diff_norm_sq_, x_update, hess_matmul_x_update_]

      inputs_to_update = [x_update_diff_norm_sq, x_update, hess_matmul_x_update]
      return [iter_ + 1] + prefer_static.cond(
          # Note on why checking delta (a difference of floats) for equality to
          # zero is ok:
          #
          # First of all, x - x == 0 in floating point -- see
          # https://stackoverflow.com/a/2686671
          #
          # Delta will conceptually equal zero when one of the following holds:
          # (i)   |w_old + newton_step| <= threshold and w_old == 0
          # (ii)  |w_old + newton_step| > threshold and
          #       w_old + newton_step - sign(w_old + newton_step) * threshold
          #          == w_old
          #
          # In case (i) comparing delta to zero is fine.
          #
          # In case (ii), newton_step conceptually equals
          #     sign(w_old + newton_step) * threshold.
          # Also remember
          #     threshold = -newton_step / (approximation of d/dz|z=0 ULLSC).
          # So (i) happens when
          #     (approximation of d/dz|z=0 ULLSC) == -sign(w_old + newton_step).
          # If we did not require LossSmoothComponent to be strictly convex,
          # then this could actually happen a non-negligible amount of the time,
          # e.g. if the loss function is piecewise linear and one of the pieces
          # has slope 1.  But since LossSmoothComponent is strictly convex, (i)
          # should not systematically happen.
          tf.reduce_all(input_tensor=tf.equal(delta, 0.)),
          lambda: inputs_to_update,
          lambda: _do_update(*inputs_to_update))

    base_dtype = x_start.dtype.base_dtype
    iter_, x_update_diff_norm_sq, x_update, _ = tf.while_loop(
        cond=_loop_cond,
        body=_loop_body,
        loop_vars=[
            tf.zeros([], dtype=np.int32, name='iter'),
            tf.zeros(
                batch_shape, dtype=base_dtype, name='x_update_diff_norm_sq'),
            tf.zeros(update_shape, dtype=base_dtype, name='x_update'),
            tf.zeros(
                update_shape, dtype=base_dtype, name='hess_matmul_x_update'),
        ])

    # Convert back x_update to the shape of x_start by transposing the leftmost
    # dimension to the rightmost.
    n = tf.rank(x_update)
    perm = tf.roll(tf.range(n), shift=-1, axis=0)
    x_update = tf.transpose(a=x_update, perm=perm)

    converged = tf.reduce_all(input_tensor=x_update_diff_norm_sq <
                              x_update_diff_norm_sq_convergence_threshold)
    return x_start + x_update, converged, iter_ / dims


def minimize(grad_and_hessian_loss_fn,
             x_start,
             tolerance,
             l1_regularizer,
             l2_regularizer=None,
             maximum_iterations=1,
             maximum_full_sweeps_per_iteration=1,
             learning_rate=None,
             name=None):
  """Minimize using Hessian-informed proximal gradient descent.

  This function solves the regularized minimization problem

  ```none
  argmin{ Loss(x)
            + l1_regularizer * ||x||_1
            + l2_regularizer * ||x||_2**2
          : x in R^n }
  ```

  where `Loss` is a convex C^2 function (typically, `Loss` is the negative log
  likelihood of a model and `x` is a vector of model coefficients).  The `Loss`
  function does not need to be supplied directly, but this optimizer does need a
  way to compute the gradient and Hessian of the Loss function at a given value
  of `x`.  The gradient and Hessian are often computationally expensive, and
  this optimizer calls them relatively few times compared with other algorithms.

  Args:
    grad_and_hessian_loss_fn: callable that takes as input a (batch of) `Tensor`
      of the same shape and dtype as `x_start` and returns the triple
      `(gradient_unregularized_loss, hessian_unregularized_loss_outer,
      hessian_unregularized_loss_middle)` as defined in the argument spec of
      `minimize_one_step`.
    x_start: (Batch of) vector-shaped, `float` `Tensor` representing the initial
      value of the argument to the `Loss` function.
    tolerance: scalar, `float` `Tensor` representing the tolerance for each
      optimization step; see the `tolerance` argument of
      `minimize_one_step`.
    l1_regularizer: scalar, `float` `Tensor` representing the weight of the L1
      regularization term (see equation above).
    l2_regularizer: scalar, `float` `Tensor` representing the weight of the L2
      regularization term (see equation above).
      Default value: `None` (i.e., no L2 regularization).
    maximum_iterations: Python integer specifying the maximum number of
      iterations of the outer loop of the optimizer.  After this many iterations
      of the outer loop, the algorithm will terminate even if the return value
      `optimal_x` has not converged.
      Default value: `1`.
    maximum_full_sweeps_per_iteration: Python integer specifying the maximum
      number of sweeps allowed in each iteration of the outer loop of the
      optimizer.  Passed as the `maximum_full_sweeps` argument to
      `minimize_one_step`.
      Default value: `1`.
    learning_rate: scalar, `float` `Tensor` representing a multiplicative factor
      used to dampen the proximal gradient descent steps.
      Default value: `None` (i.e., factor is conceptually `1`).
    name: Python string representing the name of the TensorFlow operation.
      The default name is `"minimize"`.

  Returns:
    x: `Tensor` of the same shape and dtype as `x_start`, representing the
      (batches of) computed values of `x` which minimizes `Loss(x)`.
    is_converged: scalar, `bool` `Tensor` indicating whether the minimization
      procedure converged within the specified number of iterations across all
      batches.  Here convergence means that an iteration of the inner loop
      (`minimize_one_step`) returns `True` for its `is_converged` output value.
    iter: scalar, `int` `Tensor` indicating the actual number of iterations of
      the outer loop of the optimizer completed (i.e., number of calls to
      `minimize_one_step` before achieving convergence).

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
      x_start,
      l1_regularizer,
      l2_regularizer,
      maximum_iterations,
      maximum_full_sweeps_per_iteration,
      tolerance,
      learning_rate,
  ],
  with tf.compat.v1.name_scope(name, 'minimize', graph_deps):

    def _loop_cond(x_start, converged, iter_):
      del x_start
      return tf.logical_and(iter_ < maximum_iterations,
                            tf.logical_not(converged))

    def _loop_body(x_start, converged, iter_):  # pylint: disable=missing-docstring
      g, h_outer, h_middle = grad_and_hessian_loss_fn(x_start)
      x_start, converged, _ = minimize_one_step(
          gradient_unregularized_loss=g,
          hessian_unregularized_loss_outer=h_outer,
          hessian_unregularized_loss_middle=h_middle,
          x_start=x_start,
          l1_regularizer=l1_regularizer,
          l2_regularizer=l2_regularizer,
          maximum_full_sweeps=maximum_full_sweeps_per_iteration,
          tolerance=tolerance,
          learning_rate=learning_rate)
      return x_start, converged, iter_ + 1

    return tf.while_loop(
        cond=_loop_cond,
        body=_loop_body,
        loop_vars=[
            x_start,
            tf.zeros([], np.bool, name='converged'),
            tf.zeros([], np.int32, name='iter'),
        ])
