# Copyright 2019 The TensorFlow Probability Authors.
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
"""Deprecated linalg utilities for probability distributions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v2 as tf

from tensorflow.python.util import deprecation  # pylint: disable=g-direct-tensorflow-import


@deprecation.deprecated('2019-10-01', 'This function is deprecated.')
def tridiag(below=None, diag=None, above=None, name=None):
  """Creates a matrix with values set above, below, and on the diagonal.

  Example:

  ```python
  tridiag(below=[1., 2., 3.],
          diag=[4., 5., 6., 7.],
          above=[8., 9., 10.])
  # ==> array([[  4.,   8.,   0.,   0.],
  #            [  1.,   5.,   9.,   0.],
  #            [  0.,   2.,   6.,  10.],
  #            [  0.,   0.,   3.,   7.]], dtype=float32)
  ```

  Warning: This Op is intended for convenience, not efficiency.

  Args:
    below: `Tensor` of shape `[B1, ..., Bb, d-1]` corresponding to the below
      diagonal part. `None` is logically equivalent to `below = 0`.
    diag: `Tensor` of shape `[B1, ..., Bb, d]` corresponding to the diagonal
      part.  `None` is logically equivalent to `diag = 0`.
    above: `Tensor` of shape `[B1, ..., Bb, d-1]` corresponding to the above
      diagonal part.  `None` is logically equivalent to `above = 0`.
    name: Python `str`. The name to give this op.

  Returns:
    tridiag: `Tensor` with values set above, below and on the diagonal.

  Raises:
    ValueError: if all inputs are `None`.
  """

  def _pad(x):
    """Prepends and appends a zero to every vector in a batch of vectors."""
    shape = tf.concat([tf.shape(x)[:-1], [1]], axis=0)
    z = tf.zeros(shape, dtype=x.dtype)
    return tf.concat([z, x, z], axis=-1)

  def _add(*x):
    """Adds list of Tensors, ignoring `None`."""
    s = None
    for y in x:
      if y is None:
        continue
      elif s is None:
        s = y
      else:
        s = s + y
    if s is None:
      raise ValueError('Must specify at least one of `below`, `diag`, `above`.')
    return s

  with tf.name_scope(name or 'tridiag'):
    if below is not None:
      below = tf.convert_to_tensor(below, name='below')
      below = tf.linalg.diag(_pad(below))[..., :-1, 1:]
    if diag is not None:
      diag = tf.convert_to_tensor(diag, name='diag')
      diag = tf.linalg.diag(diag)
    if above is not None:
      above = tf.convert_to_tensor(above, name='above')
      above = tf.linalg.diag(_pad(above))[..., 1:, :-1]
    # TODO(jvdillon): Consider using scatter_nd instead of creating three full
    # matrices.
    return _add(below, diag, above)


@deprecation.deprecated(
    '2019-10-01', 'This function has been deprecated; use tf.linalg.set_diag.')
def matrix_diag_transform(matrix, transform=None, name=None):
  """Transform diagonal of [batch-]matrix, leave rest of matrix unchanged.

  Create a trainable covariance defined by a Cholesky factor:

  ```python
  # Transform network layer into 2 x 2 array.
  matrix_values = tf.contrib.layers.fully_connected(activations, 4)
  matrix = tf.reshape(matrix_values, (batch_size, 2, 2))

  # Make the diagonal positive. If the upper triangle was zero, this would be a
  # valid Cholesky factor.
  chol = matrix_diag_transform(matrix, transform=tf.nn.softplus)

  # LinearOperatorLowerTriangular ignores the upper triangle.
  operator = LinearOperatorLowerTriangular(chol)
  ```

  Example of heteroskedastic 2-D linear regression.

  ```python
  tfd = tfp.distributions

  # Get a trainable Cholesky factor.
  matrix_values = tf.contrib.layers.fully_connected(activations, 4)
  matrix = tf.reshape(matrix_values, (batch_size, 2, 2))
  chol = matrix_diag_transform(matrix, transform=tf.nn.softplus)

  # Get a trainable mean.
  mu = tf.contrib.layers.fully_connected(activations, 2)

  # This is a fully trainable multivariate normal!
  dist = tfd.MultivariateNormalTriL(mu, chol)

  # Standard log loss. Minimizing this will 'train' mu and chol, and then dist
  # will be a distribution predicting labels as multivariate Gaussians.
  loss = -1 * tf.reduce_mean(dist.log_prob(labels))
  ```

  Args:
    matrix:  Rank `R` `Tensor`, `R >= 2`, where the last two dimensions are
      equal.
    transform:  Element-wise function mapping `Tensors` to `Tensors`. To be
      applied to the diagonal of `matrix`. If `None`, `matrix` is returned
      unchanged. Defaults to `None`.
    name:  A name to give created ops. Defaults to 'matrix_diag_transform'.

  Returns:
    A `Tensor` with same shape and `dtype` as `matrix`.
  """
  with tf.name_scope(name or 'matrix_diag_transform'):
    matrix = tf.convert_to_tensor(matrix, name='matrix')
    if transform is None:
      return matrix
    # Replace the diag with transformed diag.
    diag = tf.linalg.diag_part(matrix)
    transformed_diag = transform(diag)
    transformed_mat = tf.linalg.set_diag(matrix, transformed_diag)
  return transformed_mat
