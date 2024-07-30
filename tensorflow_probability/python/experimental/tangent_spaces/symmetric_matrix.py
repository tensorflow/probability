# Copyright 2023 The TensorFlow Probability Authors.
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

"""Tangent Spaces related to symmetric matrices."""

import numpy as np
import tensorflow.compat.v2 as tf
from tensorflow_probability.python.experimental.tangent_spaces import spaces
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import prefer_static as ps
from tensorflow_probability.python.math import linalg


class SymmetricMatrixSpace(spaces.TangentSpace):
  """Tangent space of M the space of `n x n` symmetric matrices."""

  def compute_basis(self, x):
    """Compute basis of symmetric matrices."""
    # For each diagonal element, we have a basis element with one in that
    # position, while for off-diagonal elements we have 1 / sqrt(2) and the
    # corresponding symmetric element.
    # For example, in the case of 2x2 matrices:
    # [[1., 0.]     [[0, 1/sqrt(2)]     [[0., 0.]
    #  [0., 0.]]     [1/sqrt(2), 0.]]    [0., 1.]]
    dim = ps.shape(x)[-1]
    n = tf.cast(dim * (dim + 1) / 2, dtype=np.int32)
    basis_tensor = tf.eye(n, dtype=x.dtype)
    basis_tensor = linalg.fill_triangular(basis_tensor)
    sqrt_2 = dtype_util.as_numpy_dtype(x.dtype)(np.sqrt(2.))
    basis_tensor = (
        basis_tensor + tf.linalg.matrix_transpose(basis_tensor)) / sqrt_2
    basis_tensor = tf.linalg.set_diag(
        basis_tensor, tf.linalg.diag_part(basis_tensor) / sqrt_2)
    return spaces.DenseBasis(basis_tensor)

  def _transform_general(self, x, f, event_ndims=None, **kwargs):
    basis = self.compute_basis(x)
    if event_ndims is None:
      event_ndims = 2
    return spaces.GeneralSpace(basis, computed_log_volume=0.).transform_general(
        x, f, event_ndims=event_ndims, **kwargs)

  def _transform_coordinatewise(self, x, f, **kwargs):
    diag_jacobian = spaces.coordinatewise_jvp(f, x)
    basis_tensor = self.compute_basis(x).to_dense()
    batch_shape = ps.shape(diag_jacobian)[:-2]
    basis_shape = ps.shape(basis_tensor)
    basis_tensor = tf.reshape(
        basis_tensor,
        ps.concat(
            [[basis_shape[0]],
             ps.ones_like(batch_shape),
             basis_shape[-2:]], axis=0))
    new_basis = spaces.DenseBasis(basis_tensor * diag_jacobian)

    correction = tf.math.square(diag_jacobian)
    correction = (correction + tf.linalg.matrix_transpose(correction)) / 2.
    correction = tf.linalg.band_part(tf.math.log(correction), -1, 0)
    log_volume = 0.5 * tf.math.reduce_sum(correction, axis=[-1, -2])
    return log_volume, spaces.GeneralSpace(
        new_basis, computed_log_volume=log_volume)


class ConstantDiagonalSymmetricMatrixSpace(spaces.TangentSpace):
  """Tangent space of M the space of `n x n` symmetric matrices, with fixed constant diagonal."""

  def compute_basis(self, x):
    """Compute basis of symmetric matrices."""
    # We only need to consider off-diagonal elements, where we have 1/sqrt(2)
    # in the corresponding positions
    # For example, in the case of 2x2 matrices:
    # [[0, 1/sqrt(2)]
    #  [1/sqrt(2), 0.]]
    dim = ps.shape(x)[-1]
    # Given that the diagonals are constant, we will fill the lower triangular
    # part of (dim - 1) x (dim - 1) matrices and pad back up.
    n = tf.cast(dim * (dim - 1) / 2, dtype=np.int32)
    basis_tensor = tf.eye(n, dtype=x.dtype)
    # These are (dim - 1) x (dim - 1) matrices. Add row and columns of zeros.
    basis_tensor = linalg.fill_triangular(basis_tensor)
    basis_tensor = tf.concat([
        tf.zeros([n, 1, dim - 1], dtype=x.dtype),
        basis_tensor], axis=-2)
    basis_tensor = tf.concat([
        basis_tensor,
        tf.zeros([n, dim, 1], dtype=x.dtype)], axis=-1)
    sqrt_2 = dtype_util.as_numpy_dtype(x.dtype)(np.sqrt(2.))
    basis_tensor = (
        basis_tensor + tf.linalg.matrix_transpose(basis_tensor)) / sqrt_2
    return spaces.DenseBasis(basis_tensor)

  def _transform_general(self, x, f, event_ndims=None, **kwargs):
    basis = self.compute_basis(x)
    if event_ndims is None:
      event_ndims = 2
    return spaces.GeneralSpace(basis, computed_log_volume=0.).transform_general(
        x, f, event_ndims=event_ndims, **kwargs)

  def _transform_coordinatewise(self, x, f, **kwargs):
    diag_jacobian = spaces.coordinatewise_jvp(f, x)
    basis_tensor = self.compute_basis(x).to_dense()
    batch_shape = ps.shape(diag_jacobian)[:-2]
    basis_shape = ps.shape(basis_tensor)
    basis_tensor = tf.reshape(
        basis_tensor,
        ps.concat(
            [[basis_shape[0]],
             ps.ones_like(batch_shape),
             basis_shape[-2:]], axis=0))
    new_basis = spaces.DenseBasis(basis_tensor * diag_jacobian)

    correction = tf.math.square(diag_jacobian)
    correction = (correction + tf.linalg.matrix_transpose(correction)) / 2.
    # Ignore the diagonal terms since these don't affect the correction.
    correction = tf.math.log(correction)
    correction = tf.linalg.set_diag(
        correction, tf.zeros_like(tf.linalg.diag_part(correction)))
    correction = tf.linalg.band_part(correction, -1, 0)
    log_volume = 0.5 * tf.math.reduce_sum(correction, axis=[-1, -2])
    return log_volume, spaces.GeneralSpace(
        new_basis, computed_log_volume=log_volume)
