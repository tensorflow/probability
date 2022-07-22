# Copyright 2022 The TensorFlow Probability Authors.
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
"""A kernel covariance matrix LinearOperator with structured interpolation."""

import functools

import numpy as np
import tensorflow.compat.v2 as tf

from tensorflow_probability.python.internal import distribution_util
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import prefer_static as ps
from tensorflow_probability.python.internal import tensor_util
from tensorflow_probability.python.math import gradient
from tensorflow_probability.python.math import interpolation

__all__ = [
    'LinearOperatorInterpolatedPSDKernel',
]

USE_EXACT_DIAG = True


class LinearOperatorInterpolatedPSDKernel(tf.linalg.LinearOperator):
  """Structured interpolation to approximate a large matrix.

  This implements a component of the Structured Kernel Interpolation [1]
  algorithm. We approximate the pairwise `kernel` values for two inputs `x1`,
  `x2` with the following product:

  ```none
  k(x1, x2) = r(x1) @ k(u, u) @ r(x2)^T
  ```

  where `u` is a set of points regularly spaced on grid and `r` is an
  interpolation matrix. In short, instead of evaluating `kernel` on `x1` and
  `x2`, we evaluate it on `u`, which is chosen to contain fewer points than `x1`
  and `x2` and the interpolate using the interpolation matrix.

  This construction lets us compute matrix products efficiently. If `x1`/`x2`
  are of shape `O(n)` and `u` is of shape `O(m)`, this reduces the compute cost
  and memory to `O(n + m^2)` from `O(n^2)`.

  In practice, the interpolation matrix is implicitly defined using `interp_fn`,
  of which `tfp.math.batch_interp_regular_nd_grid` linear interpolation is a
  prototypical example. When `x1 == x2` we can relatively cheaply compute the
  diagonal of that matrix exactly, to preserve positive-semi-definiteness.

  Since we rely on a dense grid `u`, this method works best when the the kernel
  inputs are low dimensional (2 or 3).

  #### References

  [1]: Wilson, A. G., & Nickisch, H. Kernel Interpolation for Scalable
       Structured Gaussian Processes (KISS-GP). 2005. arXiv.
       http://arxiv.org/abs/1503.01057

  """

  def __init__(self,
               kernel,
               bounds_min,
               bounds_max,
               num_interpolation_points,
               x1,
               x2=None,
               interp_fn=interpolation.batch_interp_regular_nd_grid,
               diag_shift=None,
               is_non_singular=None,
               is_self_adjoint=None,
               is_positive_definite=None,
               is_square=None,
               name=None):
    """Initializes the linear operator.

    `x1`, `kernel`, `x2` and `diag_shift` must broadcast to a shared batch
    dimension
    `[B]`. The overall shape of the operator will be `[B] + [R, C]`, given `x1`
    and `x2` with shapes `[B, R, D]` and `[B, C, D]` respectively.

    Args:
      kernel: Instance of `tfp.math.psd_kernels.PositiveSemidefiniteKernel'.
        Must have `feature_ndims == 1`.
      bounds_min: Floating point `Tensor`. Minimum bounds of the interpolation
        grid. Shape: [D]
      bounds_max: Floating point `Tensor`. Maximum bounds of the interpolation
        grid. Shape: [D]
      num_interpolation_points: Python integer. Number of inducing points per
        dimension.
      x1: Floating point `Tensor`. First input to the kernel. Shape: [B, R, D]
      x2: Optional floating point `Tensor`. Second input to the kernel. Shape:
        [B, C, D] Omit this argument to statically indicate that this operator
        is square, self-adjoint, positive definite and non-singular.
      interp_fn: Interpolation function with an API same as
        `tfp.math.batch_interp_regular_nd_grid`. This must implicitly define a
        matrix as in the class docstring, i.e. it must be linear in the value of
        the `y_ref` argument.
      diag_shift: Optional floating point `Tensor`. A diagonal offset to add to
        the resultant matrix. Must be `None` if the operator is not square and
        self-adjoint. Shape: [B]
      is_non_singular:  Expect that this operator is non-singular.
      is_self_adjoint:  Expect that this operator is equal to its hermitian
        transpose.  If `dtype` is real, this is equivalent to being symmetric.
      is_positive_definite:  Expect that this operator is positive definite,
        meaning the quadratic form `x^H A x` has positive real part for all
        nonzero `x`.
      is_square:  Expect that this operator acts like square [B] matrices.
      name: Name of the operator. Default: LinearOperatorInterpolatedPSDKernel
    """
    dtype = dtype_util.common_dtype(
        [kernel, x1, x2, bounds_min, bounds_max, diag_shift],
        dtype_hint=tf.float32)

    if kernel.feature_ndims != 1:
      raise NotImplementedError('`kernel.feature_ndims` must be 1, saw: '
                                f'{kernel.feature_ndims}')

    if x2 is None:
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
    if not is_self_adjoint:
      if diag_shift is not None:
        raise NotImplementedError(
            'Adding diag_shift is not implemented when operator is not '
            'self-adjoint.')

    self._x1 = tensor_util.convert_nonref_to_tensor(x1, dtype=dtype)
    self._x2 = tensor_util.convert_nonref_to_tensor(x2, dtype=dtype)
    self._num_interpolation_points = num_interpolation_points
    self._bounds_min = tensor_util.convert_nonref_to_tensor(
        bounds_min, dtype=dtype)
    self._bounds_max = tensor_util.convert_nonref_to_tensor(
        bounds_max, dtype=dtype)
    self._diag_shift = tensor_util.convert_nonref_to_tensor(
        diag_shift, dtype=dtype)
    self._kernel = kernel
    self._interp_fn = interp_fn

    super().__init__(
        dtype,
        is_non_singular=is_non_singular,
        is_self_adjoint=is_self_adjoint,
        is_positive_definite=is_positive_definite,
        is_square=is_square,
        name=name or 'LinearOperatorInterpolatedPSDKernel')

  @property
  def kernel(self):
    return self._kernel

  @property
  def bounds_min(self):
    return self._bounds_min

  @property
  def bounds_max(self):
    return self._bounds_max

  @property
  def num_interpolation_points(self):
    return self._num_interpolation_points

  @property
  def x1(self):
    return self._x1

  @property
  def x2(self):
    return self._x2

  @property
  def diag_shift(self):
    return self._diag_shift

  def _interpolate(self, x, y_ref, axis):
    return self._interp_fn(
        x, self._bounds_min, self._bounds_max, y_ref, fill_value=0., axis=axis)

  def _base_mat(self):
    bounds_min = _assert_bounds_is_vector(self.bounds_min, 'bounds_min')
    bounds_max = _assert_bounds_is_vector(self.bounds_max, 'bounds_max')

    axis_points = tf.linspace(
        bounds_min, bounds_max, self.num_interpolation_points, axis=-2)
    dim = tf.get_static_value(ps.dimension_size(bounds_min, -1))
    if dim is None:
      raise ValueError('The last dimension of bounds_min must be '
                       'statically known.')

    inducing_points = tf.stack(
        tf.meshgrid(*[axis_points[..., i] for i in range(dim)]),
        axis=-1,
    )
    inducing_points_flat = tf.reshape(
        inducing_points,
        ps.stack([-1, ps.dimension_size(inducing_points, -1)], axis=-1))
    base_mat = self.kernel.matrix(inducing_points_flat, inducing_points_flat)

    if tf.get_static_value(ps.shape_slice(base_mat, np.s_[-2:])) is None:
      raise ValueError('The last two dimensions of the matrix induced by '
                       '`kernel` must be statically known. Check your '
                       'kernel parameters\' shapes. Saw shape: '
                       f'{base_mat.shape}')

    return base_mat

  def _x1_x2(self):
    x1 = tf.convert_to_tensor(self.x1, self.dtype)
    # The static requirements come primarily from the default interp_fn we use.
    if tf.get_static_value(ps.shape_slice(x1, np.s_[-2:])) is None:
      raise ValueError('The last two dimensions of `x1` must be statically '
                       f'known. Saw shape: {x1.shape}')
    if self.x2 is None:
      x2 = x1
    else:
      x2 = tf.convert_to_tensor(self.x2, self.dtype)
      if tf.get_static_value(ps.shape_slice(x2, np.s_[-2:])) is None:
        raise ValueError('The last two dimensions of `x2` must be statically '
                         f'known. Saw shape: {x2.shape}')
    return x1, x2

  def _matmul(self, x, adjoint=False, adjoint_arg=False):
    # The product is: r @ base_mat @ r^T @ x
    #
    # Where:
    #
    # x:     [B] + [R, C]
    # r:     [B] + [R, M^n]
    # r^T:   [B] + [M^n, R]
    # base_mat:   [B] + [M^n, M^n]
    #
    # We do the products right to left.
    if adjoint or adjoint_arg:
      raise NotImplementedError(
          '`LinearOperatorInterpolatedPSDKernel.adjoint` not implemented.')
    x = tf.convert_to_tensor(x, self.dtype)
    if tf.get_static_value(ps.shape_slice(x, np.s_[-2:])) is None:
      raise ValueError('The last two dimensions of `x` must be statically '
                       f'known. Saw shape: {x.shape}')
    x1, x2 = self._x1_x2()

    batch_shape = ps.broadcast_shape(
        ps.shape_slice(x, np.s_[:-2]), ps.shape_slice(x2, np.s_[:-2]))
    base_mat = self._base_mat()

    # y = r^T @ x
    # [B] + [*M^n, C]
    # We implement the transpose using autodiff, to make it easier to use custom
    # interpolation functions. TODO(siege): Maybe adjust the interpolation_fn
    # API to somehow allow the user to write a custom transpose.
    _, y = gradient.value_and_gradient(
        lambda inp: self._interpolate(  # pylint: disable=g-long-lambda
            x2,
            inp,
            axis=-1 - ps.dimension_size(x1, -1)),
        tf.ones(
            ps.concat([
                batch_shape,
                ps.stack([self._num_interpolation_points] *
                         self._bounds_min.shape[-1] +
                         [ps.dimension_size(x, -1)])
            ], -1), x.dtype),
        output_gradients=tf.broadcast_to(
            x,
            ps.concat([batch_shape, ps.shape_slice(x, np.s_[-2:])], -1)))
    # [B] + [M^n, C]
    y = tf.reshape(
        y,
        ps.concat([
            ps.shape_slice(y, np.s_[:-1 - ps.dimension_size(x1, -1)]),
            ps.stack(
                [ps.dimension_size(base_mat, -1),
                 ps.dimension_size(y, -1)])
        ], -1))

    # [B] + [M^n, C]
    # y = base_mat @ r^T @ x
    y = base_mat @ y

    # [B] + [M..., C]
    y = tf.reshape(
        y,
        ps.concat([
            ps.shape_slice(y, np.s_[:-2]),
            ps.stack([self._num_interpolation_points] *
                     self._bounds_min.shape[-1] + [-1])
        ], -1))
    # [B] + [R, C]
    # y = r @ mat @ r^T @ x
    res = self._interpolate(x1, y, axis=-1 - ps.dimension_size(x1, -1))

    if self.is_self_adjoint and USE_EXACT_DIAG:
      res += (self._exact_diag_part() -
              self._approx_diag_part())[..., tf.newaxis] * x

    if self._diag_shift is not None:
      res += self._diag_shift[..., tf.newaxis, tf.newaxis] * x
    return res

  def _shape(self):
    x1, x2 = self._x1_x2()

    batch_parts = [
        self.kernel.batch_shape,
        x1.shape[:-2],
        x2.shape[:-2],
    ]
    if self.diag_shift is not None:
      batch_parts.append(self.diag_shift.shape)

    return functools.reduce(tf.broadcast_static_shape, batch_parts).concatenate(
        x1.shape[-2:-1]).concatenate(x2.shape[-2:-1])

  def _shape_tensor(self):
    x1, x2 = self._x1_x2()

    batch_parts = [
        self.kernel.batch_shape_tensor(),
        tf.shape(x1)[:-2],
        tf.shape(x2)[:-2],
    ]
    if self.diag_shift is not None:
      batch_parts.append(tf.shape(self.diag_shift))

    return tf.concat([
        functools.reduce(tf.broadcast_dynamic_shape, batch_parts),
        tf.shape(x1)[-2:-1],
        tf.shape(x2)[-2:-1]
    ], -1)

  def _exact_diag_part(self, x1=None):
    if self._x2 is not None:
      assert False

    if x1 is None:
      x1 = self.x1

    res = self.kernel.apply(x1, x1, example_ndims=1)
    return res

  def _approx_diag_part(self, x1=None):
    # The matrix is: r @ base_mat @ r^T
    #
    # Where:
    #
    # r:        [B] + [R, M^n]
    # r^T:      [B] + [M^n, R]
    # base_mat: [B] + [M^n, M^n]
    #
    # We first compute the first term: k = r @ base_mat
    # Then we do a batched matrix multiply to compute the dot products of rows
    # of k and columns of r^T.
    if self._x2 is not None:
      assert False

    if x1 is None:
      x1 = self.x1

    base_mat = self._base_mat()
    # [B] + [*M^n, *M^n]
    base_mat = tf.reshape(
        base_mat,
        ps.concat([
            ps.shape(base_mat)[:-2],
            ps.repeat((self._num_interpolation_points,),
                      2 * ps.shape(self._bounds_min)[-1])
        ], 0))
    # r @ base_mat
    # [B] + [R, *M^n]
    diag_part = self._interpolate(x1, base_mat, axis=-2 * int(x1.shape[-1]))
    res = self._interpolate(
        x1[..., tf.newaxis, :], diag_part, axis=-int(x1.shape[-1]))
    res = tf.squeeze(res, -1)
    return res

  def _diag_part(self):
    if USE_EXACT_DIAG:
      res = self._exact_diag_part()
    else:
      res = self._approx_diag_part()
    if self._diag_shift is not None:
      res += self._diag_shift[..., tf.newaxis]
    return res

  def row(self, index):
    """Gets a row from the dense operator.

    Args:
      index: The index (indices) of the row[s] to get, may be scalar or up to
        batch shape.

    Returns:
      rows: Row[s] of the matrix, with shape `(...batch_shape..., num_cols)`.
        Effectively the same as `operator.to_dense()[..., index, :]` for a
        scalar `index`, analogous to gather for non-scalar.
    """
    index = tf.convert_to_tensor(index)
    x1, x2 = self._x1_x2()
    index_rank = tf.get_static_value(ps.rank(index))
    if index_rank is None:
      raise ValueError('Rank of `index` must be known statically. Saw shape: '
                       f'{index.shape}')
    if index_rank == 0:
      scalar_index = True
      index = index[tf.newaxis]
    else:
      scalar_index = False
    x1_rows = tf.gather(x1, index, axis=-2)

    base_mat = self._base_mat()
    # [B] + [*M^n, *M^n]
    base_mat = tf.reshape(
        base_mat,
        ps.concat([
            ps.shape(base_mat)[:-2],
            ps.repeat((self._num_interpolation_points,),
                      2 * ps.shape(self._bounds_min)[-1])
        ], 0))

    # [B] + [R', *M^n]
    row = self._interpolate(x1_rows, base_mat, axis=-2 * int(x1.shape[-1]))
    # [B] + [R, R']
    row = self._interpolate(
        x2,
        distribution_util.move_dimension(row, -1 - int(x1.shape[-1]), -1),
        axis=-1 - int(x1.shape[-1]))
    # [B] + [R', R]
    row = distribution_util.move_dimension(row, -1, -2)

    if self.is_self_adjoint and USE_EXACT_DIAG:
      exact_val = self._exact_diag_part(x1_rows)
      approx_val = self._approx_diag_part(x1_rows)
      # [B] + [R']
      corr = exact_val - approx_val
      corr_shape = ps.shape(corr)
      batch_shape = corr_shape[:-1]
      corr_ndims = ps.size(corr_shape)

      # [R', R] + [B]
      zeros = tf.zeros(
          ps.concat(
              [ps.shape(row)[-2:], ps.shape(corr)[:-1]], axis=-1), self.dtype)
      diag_corrections = tf.tensor_scatter_nd_update(
          zeros,
          ps.stack([index, ps.range(ps.size(index))], axis=-1),
          tf.transpose(
              corr,
              ps.concat([[corr_ndims - 1],
                         ps.range(ps.size(batch_shape))],
                        axis=-1)),
      )
      # [B] + [R', R]
      diag_corrections = tf.transpose(
          diag_corrections,
          ps.concat([2 + ps.range(ps.size(batch_shape)), [0, 1]], axis=-1))
      row += diag_corrections

    if self._diag_shift is not None:
      # Normally we'd use one_hot, but it doesn't understand batching.
      # tf.eye, on the other hand, doesn't understand numpy arrays... but we can
      # fix this.
      num_rows = ps.size(index)
      num_cols = ps.shape(row)[-1]
      num_rows_ = tf.get_static_value(num_rows)
      num_cols_ = tf.get_static_value(num_cols)
      if num_rows_ is not None:
        num_rows = int(num_rows_)
      if num_cols_ is not None:
        num_cols = int(num_cols_)
      row += tf.eye(
          num_rows, num_cols,
          dtype=row.dtype) * self._diag_shift[..., tf.newaxis, tf.newaxis]

    if scalar_index:
      row = row[..., 0, :]
    return row


def _assert_bounds_is_vector(v, name):
  rank_ = tf.get_static_value(ps.rank(v))
  if rank_ is None:
    with tf.control_dependencies(
        [tf.assert_rank(v, 1, f'`{name}` must be a vector.')]):
      v = tf.identity(v)
  else:
    if rank_ != 1:
      raise ValueError(f'`{name}` must be a vector, saw: {ps.shape(v)}')
  return v
