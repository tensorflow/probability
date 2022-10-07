# Copyright 2021 The TensorFlow Probability Authors.
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
"""Experimental functions to use as marginals for GaussianProcess(es)."""

import tensorflow.compat.v2 as tf
from tensorflow_probability.python.distributions import mvn_linear_operator


def make_backoff_cholesky(alternate_cholesky, name='BackoffCholesky'):
  """Make a function that tries Cholesky then the user-specified function.

  Warning: This function uses an XLA-compiled `tf.linalg.cholesky` to capture
  factorization failures.

  Args:
    alternate_cholesky: A callable with the same signature as
      `tf.linalg.cholesky`.
    name: Python `str` name prefixed to Ops created by this function.
      Default value: 'BackoffCholesky'.

  Returns:
    run_backoff: An function that attempts a standard Cholesky, and then tries
      `alternate_cholesky` on failure.
  """
  def run_backoff(covariance):
    with tf.name_scope(name):
      chol = tf.linalg.cholesky(covariance)
      ok = tf.reduce_all(tf.math.is_finite(chol))
      return tf.cond(
          ok,
          lambda: chol,
          lambda: alternate_cholesky(covariance))

  return run_backoff


def make_cholesky_like_marginal_fn(cholesky_like,
                                   name='CholeskyLikeMarginalFn'):
  """Use a Cholesky-like function for `GaussianProcess` `marginal_fn`.

  For use with "Cholesky-like" lower-triangular factorizations (LL^T).  See
  `make_backoff_cholesky` for one way to create such functions.

  Args:
    cholesky_like: A callable with the same signature as `tf.linalg.cholesky.`
    name: Python `str` name prefixed to Ops created by this function.
      Default value: 'CholeskyLikeMarginalFn'.

  Returns:
    marginal_function: A function that can be used with the `marginal_fn`
      argument to `GaussianProcess`.
  """
  def marginal_fn(
      loc,
      covariance,
      validate_args=False,
      allow_nan_stats=False,
      name=name):
    with tf.name_scope(name) as name:
      scale = tf.linalg.LinearOperatorLowerTriangular(
          cholesky_like(covariance),
          is_non_singular=True)
      return mvn_linear_operator.MultivariateNormalLinearOperator(
          loc=loc,
          scale=scale,
          validate_args=validate_args,
          allow_nan_stats=allow_nan_stats)

  return marginal_fn


def make_eigh_marginal_fn(tol=1e-6,
                          name='EigHMarginalFn'):
  """Make an eigenvalue decomposition-based `marginal_fn`.

  For use with `GaussianProcess` classes.

  A matrix square root is produced using an eigendecomposition. Eigenvalues are
  forced to be above a tolerance, to ensure positive-definiteness.

  Args:
    tol: Scalar float `Tensor`. Eigenvalues below `tol` are raised to `tol`.
    name: Python `str` name prefixed to Ops created by this function.
      Default value: 'EigHMarginalFn'.

  Returns:
    marginal_function: A function that can be used with the `marginal_fn`
      argument to `GaussianProcess`.
  """
  def eigh_marginal_fn(
      loc,
      covariance,
      validate_args=False,
      allow_nan_stats=False,
      name=name):
    """Compute EigH-based square root and return a MVN."""
    with tf.name_scope(name) as name:
      values, vectors = tf.linalg.eigh(covariance)
      safe_root = tf.math.sqrt(tf.where(values < tol, tol, values))
      scale = tf.linalg.LinearOperatorFullMatrix(
          tf.einsum('...ij,...j->...ij', vectors, safe_root),
          is_square=True,
          is_positive_definite=True,
          is_non_singular=True,
          name='GaussianProcessEigHScaleLinearOperator')
      return mvn_linear_operator.MultivariateNormalLinearOperator(
          loc=loc,
          scale=scale,
          validate_args=validate_args,
          allow_nan_stats=allow_nan_stats,
          name=name)

  return eigh_marginal_fn


def retrying_cholesky(
    matrix, jitter=None, max_iters=5, name='retrying_cholesky'):
  """Computes a modified Cholesky decomposition for a batch of square matrices.

  Given a symmetric matrix `A`, this function attempts to give a factorization
  `A + E = LL^T` where `L` is lower triangular, `LL^T` is positive definite, and
  `E` is small in some suitable sense. This is useful for nearly positive
  definite symmetric matrices that are otherwise numerically difficult to
  Cholesky factor.

  In particular, this function first attempts a Cholesky decomposition of
  the input matrix.  If that decomposition fails, exponentially-increasing
  diagonal jitter is added to the matrix until either a Cholesky decomposition
  succeeds or until the maximum specified number of iterations is reached.

  This function is similar in spirit to a true modified Cholesky factorization
  ([1], [2]). However, it does not use pivoting or other strategies to ensure
  stability, so may not work well for e.g. ill-conditioned matrices.  Further,
  this function may perform multiple Cholesky factorizations, while a true
  modified Cholesky can be done with only slightly more work than a single
  decomposition.

  #### References

  [1]: Nicholas Higham. What is a modified Cholesky factorization?
    https://nhigham.com/2020/12/22/what-is-a-modified-cholesky-factorization/

  [2]: Sheung Hun Cheng and Nicholas Higham, A Modified Cholesky Algorithm Based
    on a Symmetric Indefinite Factorization, SIAM J. Matrix Anal. Appl. 19(4),
    1097–1110, 1998.

  Args:
    matrix: A batch of symmetric square matrices, with shape `[..., n, n]`.
    jitter: Initial jitter to add to the diagonal.  Default: 1e-6, unless
      `matrix.dtype` is float64, in which case the default is 1e-10.
    max_iters: Maximum number of times to retry the Cholesky decomposition
      with larger diagonal jitter.  Default: 5.
    name: Python `str` name prefixed to Ops created by this function.
      Default value: 'retrying_cholesky'.

  Returns:
    triangular_factor: A Tensor with shape `[..., n, n]`.  The lower triangular
      Cholesky factor, modified as above.  If the Cholesky decomposition failed
      for a batch member, then all lower triangular entries returned for that
      batch member will be NaN.
    diagonal_shift: A tensor of shape `[...]`. `diag_shift[i]` is the value
      added to the diagonal of `matrix[i]` in computing `triangular_factor[i]`.
  """
  with tf.name_scope(name) as name:
    matrix = tf.convert_to_tensor(matrix)
    if jitter is None:
      jitter = 1e-10 if matrix.dtype == tf.float64 else 1e-6
    jitter = tf.convert_to_tensor(jitter, dtype=matrix.dtype)

    n = tf.compat.dimension_value(matrix.shape[-1])
    if n is None:
      n = tf.shape(matrix)[-1]

    one = tf.convert_to_tensor(1., dtype=matrix.dtype)
    ten = tf.convert_to_tensor(10., dtype=matrix.dtype)

    def cond(i, _, triangular_factor):
      return ((i < max_iters)
              & tf.reduce_any(tf.math.is_nan(triangular_factor[..., 0, 0])))

    def body(i, shift, triangular_factor):
      triangular_factor = tf.linalg.cholesky(
          tf.linalg.set_diag(matrix, tf.linalg.diag_part(matrix) + shift))
      shift = shift * tf.where(
          tf.math.is_nan(triangular_factor[..., :1, 0]), ten, one)
      return [i + 1, shift, triangular_factor]

    triangular_factor = tf.linalg.cholesky(matrix)
    shift = tf.where(tf.math.is_nan(triangular_factor[..., :1, 0]), jitter, 0.)
    _, shift, triangular_factor = tf.while_loop(
        cond, body,
        loop_vars=[tf.convert_to_tensor(0), shift, triangular_factor],
        maximum_iterations=max_iters)

    # To avoid NaN gradients, run the Cholesky decomposition again.
    #
    # TODO(jburnim): Implement a version of `retrying_cholesky` that uses
    # `tf.custom_gradient` to avoid having this redundant `tf.linalg.cholesky`.
    shift = tf.stop_gradient(shift)
    triangular_factor = tf.linalg.cholesky(
        tf.linalg.set_diag(matrix, tf.linalg.diag_part(matrix) + shift))
    return triangular_factor, shift[..., 0]
