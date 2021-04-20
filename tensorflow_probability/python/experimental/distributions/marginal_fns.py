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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v2 as tf
from tensorflow_probability.python.distributions import mvn_linear_operator


def make_backoff_choleksy(alternate_cholesky, name='BackoffCholesky'):
  """Make a function that tries Cholesky then the user-specified function.

  Note this will NOT work under a gradient tape until b/177365178 is resolved.
  Also this uses XLA compilation, which is necessary until b/144845034 is
  resolved.

  Args:
    alternate_cholesky: A callable with the same signature as
      `tf.linalg.cholesky`.
    name: Python `str` name prefixed to Ops created by this function.
      Default value: 'BackoffCholesky'.

  Returns:
    run_backoff: An function that attempts a standard Cholesky, and then tries
      `alternate_cholesky` on failure.
  """
  # TODO(leben) Remove this XLA bit once b/144845034 is resolved
  def cholesky_ok(covariance):
    try:
      chol = tf.linalg.cholesky(covariance)
      return chol, tf.reduce_all(tf.math.is_finite(chol))
    except tf.errors.InvalidArgumentError:
      return covariance, False

  cholesky_ok = tf.function(cholesky_ok, autograph=False, jit_compile=True)

  def run_backoff(covariance):
    with tf.name_scope(name):
      chol, ok = cholesky_ok(covariance)
      return tf.cond(
          ok,
          lambda: chol,
          lambda: alternate_cholesky(covariance))

  return run_backoff


def make_cholesky_like_marginal_fn(cholesky_like,
                                   name='CholeskyLikeMarginalFn'):
  """Use a Cholesky-like function for `GaussianProcess` `marginal_fn`.

  For use with "Cholesky-like" lower-triangular factorizations (LL^T).  See
  `make_backoff_choleksy` for one way to create such functions.

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
