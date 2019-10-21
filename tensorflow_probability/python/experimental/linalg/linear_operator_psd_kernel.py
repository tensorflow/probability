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
"""A kernel covariance matrix LinearOperator."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

import tensorflow.compat.v2 as tf

from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import tensor_util


class LinearOperatorPSDKernel(tf.linalg.LinearOperator):
  """A `tf.linalg.LinearOperator` representing a kernel covariance matrix."""

  def __init__(self,
               kernel,
               x1,
               x2=None,
               is_non_singular=None,
               is_self_adjoint=None,
               is_positive_definite=None,
               is_square=None,
               name=None):
    """Initializes the LinearOperator.

    This object implicitly represents the covariance matrix of `x1` and `x2`
    (`x1` if `x2` not provided) under the given `kernel`. This operator assumes
    one example dimension on each set of index points, which indexes the
    corresponding axis of the matrix. All outer dimensions are considered
    batch dimensions.

    Use this to avoid materializing the full matrix for such operations as:
    - accessing diagonal (`diag_part` method)
    - accessing a [batch of] row[s] (`row` method)
    - accessing a [batch of] column[s] (`col` method)

    [not yet implemented...]
    Use this to perform matrix-vector products on very large covariance matrices
    by chunking the covariance matrix into parts, computing just that part of
    the output, computing a part of the covariance matrix, and the matrix-vector
    product, then forgetting the partial covariance matrix. Internally, uses
    `@tf.recompute_grad` to avoid retaining infeasibly-large intermediate
    activations.

    Args:
      kernel: A `tfp.math.psd_kernels.PositiveSemidefiniteKernel` instance.
      x1: A floating point `Tensor`, the row index points.
      x2: Optional, a floating point `Tensor`, the column index points. If not
        provided, uses `x1`.
      is_non_singular:  Expect that this operator is non-singular.
      is_self_adjoint:  Expect that this operator is equal to its hermitian
        transpose.  If `dtype` is real, this is equivalent to being symmetric.
      is_positive_definite:  Expect that this operator is positive definite,
        meaning the quadratic form `x^H A x` has positive real part for all
        nonzero `x`.  Note that we do not require the operator to be
        self-adjoint to be positive-definite.  See:
        https://en.wikipedia.org/wiki/Positive-definite_matrix#Extension_for_non-symmetric_matrices
      is_square:  Expect that this operator acts like square [batch] matrices.
      name: Optional name for related ops.
    """
    dtype = dtype_util.common_dtype([kernel, x1, x2], dtype_hint=tf.float64)
    self._kernel = kernel
    self._x1 = tensor_util.convert_nonref_to_tensor(x1, dtype=dtype)
    self._x2 = tensor_util.convert_nonref_to_tensor(x2, dtype=dtype)
    if self._x2 is None:
      if is_non_singular is not None and not is_non_singular:
        raise ValueError(
            'Operator is non-singular with a single set of index points.')
      is_non_singular = True
      if is_self_adjoint is not None and not is_self_adjoint:
        raise ValueError(
            'Operator is self-adjoint with a single set of index points.')
      is_self_adjoint = True
      if is_positive_definite is not None and not is_positive_definite:
        raise ValueError(
            'Operator is positive-definite with a single set of index points.')
      is_positive_definite = True
      if is_square is not None and not is_square:
        raise ValueError(
            'Operator is square with a single set of index points.')
      is_square = True
    super(LinearOperatorPSDKernel, self).__init__(
        dtype,
        is_non_singular=is_non_singular,
        is_self_adjoint=is_self_adjoint,
        is_positive_definite=is_positive_definite,
        is_square=is_square,
        name=name or 'LinearOperatorPSDKernel')

  @property
  def kernel(self):
    return self._kernel

  @property
  def x1(self):
    return self._x1

  @property
  def x2(self):
    return self._x2

  def _x1_x2_axis(self):
    x1 = tf.convert_to_tensor(self.x1)
    x2 = x1 if self.x2 is None else tf.convert_to_tensor(self.x2)
    return x1, x2, -self.kernel.feature_ndims - 1

  def _matmul(self, x, adjoint=False, adjoint_arg=False):
    x1, x2, _ = self._x1_x2_axis()
    return tf.matmul(
        self.kernel.matrix(x1, x2), x, adjoint_a=adjoint, adjoint_b=adjoint_arg)

  def _shape(self):
    x1, x2, axis = self._x1_x2_axis()
    return functools.reduce(tf.broadcast_static_shape,
                            (self.kernel.batch_shape, x1.shape[:axis],
                             x2.shape[:axis])).concatenate(
                                 [x1.shape[axis], x2.shape[axis]])

  def _shape_tensor(self):
    x1, x2, axis = self._x1_x2_axis()
    batch_shape = functools.reduce(tf.broadcast_dynamic_shape,
                                   (self.kernel.batch_shape_tensor(),
                                    tf.shape(x1)[:axis], tf.shape(x2)[:axis]))
    return tf.concat([batch_shape, [tf.shape(x1)[axis],
                                    tf.shape(x2)[axis]]],
                     axis=0)

  def _diag_part(self):
    x1, x2, axis = self._x1_x2_axis()
    ax_minsize = tf.minimum(tf.shape(x1)[axis], tf.shape(x2)[axis])

    def slice_of(xn):
      slice_size = tf.where(
          tf.equal(tf.range(tf.rank(xn)),
                   tf.rank(xn) + axis), ax_minsize, tf.shape(xn))
      return tf.slice(xn, begin=tf.zeros_like(tf.shape(xn)), size=slice_size)

    return self.kernel.apply(slice_of(x1), slice_of(x2))

  def row(self, i):
    i = tf.convert_to_tensor(i)
    x1, x2, axis = self._x1_x2_axis()
    batch_shp = tf.broadcast_dynamic_shape(tf.shape(x1)[:axis], tf.shape(i))
    x1 = tf.broadcast_to(x1,
                         tf.concat([batch_shp, tf.shape(x1)[axis:]], axis=0))
    x1_row = tf.broadcast_to(i, tf.shape(x1)[:axis])
    x1 = tf.gather(x1, x1_row[..., tf.newaxis], batch_dims=len(x1.shape[:axis]))
    return self.kernel.matrix(x1, x2)[..., 0, :]

  def col(self, j):
    j = tf.convert_to_tensor(j)
    x1, x2, axis = self._x1_x2_axis()
    batch_shp = tf.broadcast_dynamic_shape(tf.shape(x2)[:axis], tf.shape(j))
    x2 = tf.broadcast_to(x2,
                         tf.concat([batch_shp, tf.shape(x2)[axis:]], axis=0))
    x2_col = tf.broadcast_to(j, tf.shape(x2)[:-self.kernel.feature_ndims - 1])
    x2 = tf.gather(x2, x2_col[..., tf.newaxis], batch_dims=len(x2.shape[:axis]))
    return self.kernel.matrix(x1, x2)[..., 0]
