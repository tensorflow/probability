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

"""Utilities for computing Cholesky factorizations."""

import tensorflow.compat.v2 as tf


def make_cholesky_with_jitter_fn(jitter=1e-6):
  """Make a function that adds diagonal jitter before Cholesky factoring.

  Suitable for use in the `cholesky_fn` parameter for
  `GaussianProcessRegressionModelWithCholesky`.

  Args:
    jitter: Float diagonal jitter to add. Default value: 1e-6.

  Returns:
    cholesky_with_jitter: Function that computes jittered Cholesky.
  """
  def cholesky_with_jitter(matrix):
    jittered = tf.linalg.set_diag(
        matrix,
        tf.linalg.diag_part(matrix) + jitter)
    return tf.linalg.cholesky(jittered)
  return cholesky_with_jitter


def cholesky_from_fn(linop, cholesky_fn):
  """Compute Cholesky factor with respect to `linop`.

  Computing a Cholesky decomposition via `tf.linalg.cholesky(linop.to_dense())`
  can be both numerically unstable and slow. This method allows using alternate
  Cholesky decomposition algorithms via `cholesky_fn` to enable numerical
  stability, while trying to be as efficient as possible for some structured
  operators.

  Args:
    linop: Positive-definite `LinearOperator`.
    cholesky_fn: Callable which takes a single (batch) matrix argument and
        returns a Cholesky-like lower triangular factor.

  Returns:
    cholesky_factor: `LinearOperator` representing the Cholesky factor for
      `linop`.
  """
  if isinstance(linop, tf.linalg.LinearOperatorIdentity):
    return linop
  elif isinstance(linop, tf.linalg.LinearOperatorDiag):
    return tf.linalg.LinearOperatorDiag(
        tf.math.sqrt(linop.diag),
        is_non_singular=True,
        is_positive_definite=True)
  elif isinstance(linop, tf.linalg.LinearOperatorBlockDiag):
    return tf.linalg.LinearOperatorBlockDiag(
        [cholesky_from_fn(
            o, cholesky_fn) for o in linop.operators],
        is_non_singular=True)
  elif isinstance(linop, tf.linalg.LinearOperatorKronecker):
    return tf.linalg.LinearOperatorKronecker(
        [cholesky_from_fn(
            o, cholesky_fn) for o in linop.operators],
        is_non_singular=True)
  else:
    # This handles the `LinearOperatorFullMatrix` case among others.
    return tf.linalg.LinearOperatorLowerTriangular(
        cholesky_fn(linop.to_dense()),
        is_non_singular=True)
