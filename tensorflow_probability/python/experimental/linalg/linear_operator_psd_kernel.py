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

import functools

import tensorflow.compat.v1 as tf1
import tensorflow.compat.v2 as tf

from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import prefer_static
from tensorflow_probability.python.internal import tensor_util
from tensorflow.python.ops import control_flow_util  # pylint: disable=g-direct-tensorflow-import


def _slice(x, begin, size, axis):
  """Slice wrapper which has XLA-compatible gradients."""

  # Not in an XLA context => don't need a custom gradient, can use pad.
  if (tf.executing_eagerly() or
      not control_flow_util.GraphOrParentsInXlaContext(
          tf1.get_default_graph())):
    return tf.slice(x, begin, size)

  # TODO(b/143313126): Upstream something like this to TF proper.
  @tf.custom_gradient
  def _custom_slice_helper(x):
    """tf.slice uses tf.pad for backprop, unknown shape breaks XLA."""

    def grad(dy):
      dshp = tf.shape(x) - tf.shape(dy)
      z = tf.zeros(
          tf.where(tf.equal(0, dshp), tf.shape(x), dshp), dtype=x.dtype)
      dx = tf.roll(tf.concat([dy, z], axis=axis), shift=begin[axis], axis=axis)
      return dx

    return tf.slice(x, begin, size), grad

  return _custom_slice_helper(x)


def _forward_matmul_one_part(kernel,
                             x1,
                             x2,
                             x,
                             part_size,
                             part_index,
                             remainder_part_size=None):
  """Applies a single chunk of matmul split along the axis defined by `x2`."""
  x2_ax = tf.rank(x2) - kernel.feature_ndims - 1
  x2_ax_selector = tf.equal(tf.range(tf.rank(x2)), x2_ax)
  begin_x2 = tf.where(x2_ax_selector, part_size * part_index, 0)
  size_x2 = tf.where(
      x2_ax_selector,
      part_size if remainder_part_size is None else remainder_part_size,
      tf.shape(x2))

  x_ax = tf.rank(x) - 2
  x_ax_selector = tf.equal(tf.range(tf.rank(x)), x_ax)
  begin_x = tf.where(x_ax_selector, part_size * part_index, 0)
  size_x = tf.where(
      x_ax_selector,
      part_size if remainder_part_size is None else remainder_part_size,
      tf.shape(x))

  return tf.matmul(
      kernel.matrix(x1, _slice(x2, begin_x2, size_x2, x2_ax)),
      _slice(x, begin_x, size_x, x_ax))


def _backward_matmul_one_part(dcovx,
                              kernel_fn,
                              kernel_args,
                              x1,
                              x2,
                              x,
                              part_size,
                              part_index,
                              remainder_part_size=None):
  """Applies a single chunk of backprop split along the axis defined by `x1`."""
  # Assume `cov = kernel.matrix(x1, x2)` has shape (A,B), and `x` has shape
  # (B,C). Then `cov @ x` had shape (A,C), as will `dcovx`. Below, we refer to
  # the "part" size as P.
  dcovx_ax = tf.rank(dcovx) - 2
  dcovx_ax_selector = tf.equal(tf.range(tf.rank(dcovx)), dcovx_ax)
  begin_dcovx = tf.where(dcovx_ax_selector, part_size * part_index, 0)
  size_dcovx = tf.where(
      dcovx_ax_selector,
      part_size if remainder_part_size is None else remainder_part_size,
      tf.shape(dcovx))
  dcovxpart = _slice(dcovx, begin_dcovx, size_dcovx, dcovx_ax)  # PxC
  dcovpart = tf.matmul(dcovxpart, x, transpose_b=True)  # PxC @ (BxC).T = PxB
  with tf.GradientTape(watch_accessed_variables=False) as tape:
    # Here begins the "recomputed" part of the gradient.
    tape.watch((x1, x2, kernel_args))
    kernel = kernel_fn(*kernel_args)
    x1_ax = tf.rank(x1) - kernel.feature_ndims - 1
    x1_ax_selector = tf.equal(tf.range(tf.rank(x1)), x1_ax)
    begin_x1 = tf.where(x1_ax_selector, part_size * part_index, 0)
    size_x1 = tf.where(
        x1_ax_selector,
        part_size if remainder_part_size is None else remainder_part_size,
        tf.shape(x1))
    covpart = kernel.matrix(_slice(x1, begin_x1, size_x1, x1_ax), x2)  # PxB
  dx1part, dx2part, dkernel_args = tape.gradient(
      covpart, (x1, x2, kernel_args), output_gradients=dcovpart)
  dxpart = tf.matmul(covpart, dcovxpart, transpose_a=True)  # (PxB).T @ PxC
  dxpart = tf.reduce_sum(dxpart, axis=tf.range(tf.rank(dxpart) - tf.rank(x)))
  return dx1part, dx2part, dxpart, dkernel_args


def _chunked_matmul(kernel_fn, kernel_args, x1, x2, x, num_matmul_parts,
                    operator_shape):
  """Implements while-loop matmul with recomputed gradients."""
  # We can't use kwargs with tf.custom_gradient in graph mode, so we close over.
  kernel_args_structure = tf.nest.pack_sequence_as(
      kernel_args, [0] * len(tf.nest.flatten(kernel_args)))

  @tf.custom_gradient
  def _chunked_matmul_cgrad(x1, x2, x, *kernel_args):
    """Chunk-at-a-time matrix multiplication and backprop."""
    kernel_args = tf.nest.pack_sequence_as(kernel_args_structure, kernel_args)
    kernel = kernel_fn(*kernel_args)
    fwd_ax_size = tf.shape(x2)[-kernel.feature_ndims - 1]
    fwd_part_size = fwd_ax_size // num_matmul_parts

    dist_ctx = tf.distribute.get_replica_context()
    replica_id = dist_ctx.replica_id_in_sync_group
    num_replicas = dist_ctx.num_replicas_in_sync
    replica_num_parts = num_matmul_parts // num_replicas + tf.cast(
        num_matmul_parts % num_replicas > replica_id, num_matmul_parts.dtype)
    replica_begin = ((num_matmul_parts // num_replicas) * replica_id +
                     tf.minimum(replica_id, num_matmul_parts % num_replicas))

    def cond(i, _):
      return i < replica_begin + replica_num_parts

    def body(i, covx):
      return i + 1, covx + _forward_matmul_one_part(
          kernel, x1, x2, x, fwd_part_size, i)

    result_batch_shape = tf.broadcast_dynamic_shape(
        operator_shape[:-2], tf.shape(x)[:-2])
    result_shape = tf.concat(
        [result_batch_shape, [operator_shape[-2], tf.shape(x)[-1]]],
        axis=0)
    _, covx = tf.while_loop(
        cond,
        body, (replica_begin, tf.zeros(result_shape, dtype=x.dtype)),
        back_prop=False,
        parallel_iterations=1)

    remainder = _forward_matmul_one_part(
        kernel,
        x1,
        x2,
        x,
        fwd_part_size,
        num_matmul_parts,
        remainder_part_size=fwd_ax_size - (num_matmul_parts * fwd_part_size))

    covx = dist_ctx.all_reduce(tf.distribute.ReduceOp.SUM, covx) + remainder

    del result_batch_shape, result_shape

    def grad_fn(dcovx):
      """Chunk-at-a-time backprop."""
      # Backward, we partition along the `x1`-defined axis.
      bwd_ax_size = tf.shape(x1)[-kernel.feature_ndims - 1]
      bwd_part_size = bwd_ax_size // num_matmul_parts

      dist_ctx = tf.distribute.get_replica_context()
      replica_id = dist_ctx.replica_id_in_sync_group
      num_replicas = dist_ctx.num_replicas_in_sync
      replica_num_parts = num_matmul_parts // num_replicas + tf.cast(
          num_matmul_parts % num_replicas > replica_id, num_matmul_parts.dtype)
      replica_begin = ((num_matmul_parts // num_replicas) * replica_id +
                       tf.minimum(replica_id, num_matmul_parts % num_replicas))

      def bw_cond(i, *_):
        return i < replica_begin + replica_num_parts

      def bw_body(i, dx1, dx2, dx, dkernel_args):
        """tf.while_loop body for backprop."""
        dx1part, dx2part, dxpart, dkernel_argspart = _backward_matmul_one_part(
            dcovx, kernel_fn, kernel_args, x1, x2, x, bwd_part_size, i)
        dx1, dx2, dx, dkernel_args = tf.nest.pack_sequence_as(
            (dx1, dx2, dx, dkernel_args),
            [a + b for a, b in zip(  # pylint: disable=g-complex-comprehension
                tf.nest.flatten((dx1, dx2, dx, dkernel_args)),
                tf.nest.flatten((dx1part, dx2part, dxpart, dkernel_argspart)))])
        return i + 1, dx1, dx2, dx, dkernel_args

      _, dx1, dx2, dx, dkernel_args = tf.while_loop(
          bw_cond,
          bw_body,
          (replica_begin,) + tf.nest.map_structure(tf.zeros_like,
                                                   (x1, x2, x, kernel_args)),
          back_prop=False,
          parallel_iterations=1)
      dx1rem, dx2rem, dxrem, dkernel_argsrem = _backward_matmul_one_part(
          dcovx,
          kernel_fn,
          kernel_args,
          x1,
          x2,
          x,
          bwd_part_size,
          num_matmul_parts,
          remainder_part_size=bwd_ax_size - (num_matmul_parts * bwd_part_size))
      flat_xdevice = tf.nest.flatten((dx1, dx2, dx, dkernel_args))
      flat_remainder = tf.nest.flatten((dx1rem, dx2rem, dxrem, dkernel_argsrem))
      return tuple(
          dist_ctx.all_reduce(tf.distribute.ReduceOp.SUM, a) + b
          for a, b in zip(flat_xdevice, flat_remainder))

    return covx, grad_fn

  return _chunked_matmul_cgrad(x1, x2, x, *tf.nest.flatten(kernel_args))


class LinearOperatorPSDKernel(tf.linalg.LinearOperator):
  """A `tf.linalg.LinearOperator` representing a kernel covariance matrix."""

  def __init__(self,
               kernel_fn,
               x1,
               x2=None,
               kernel_args=None,
               num_matmul_parts=None,
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

    Use this to perform matrix-vector products on very large covariance matrices
    by chunking the covariance matrix into parts, computing just that part of
    the output, computing a part of the covariance matrix, and the matrix-vector
    product, then forgetting the partial covariance matrix. Internally, uses
    recomputed gradients to avoid retaining infeasibly-large intermediate
    activations.

    Args:
      kernel_fn: A Python callable which takes `*kernel_args` and returns an
        instance of `tfp.math.psd_kernels.PositiveSemidefiniteKernel`. As a
        convenience, an instance may be passed instead of a function, but in
        this case gradients to kernel hyperparameters will not be available when
        using `num_matmul_parts`, and `kernel_args` must be `None`.
      x1: A floating point `Tensor`, the row index points.
      x2: Optional, a floating point `Tensor`, the column index points. If not
        provided, uses `x1`.
      kernel_args: A tuple of arguments (which may themselves be `tf.nest`
        compatible structures) to be passed to `kernel_fn`. This argument
        identifies the set of tensors which will have gradients in a
        `num_matmul_parts`-chunked matmul/backprop.
      num_matmul_parts: An optional Python `int` specifying the number of
        partitions into which the matrix should be broken when applying this
        linear operator. (An extra remainder partition is implied for uneven
        partitioning.) Because the use-case is avoiding a memory blowup, the
        partitioned matmul uses `parallel_iterations=1` and `back_prop=False`.
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
    if not callable(kernel_fn):
      kernel = kernel_fn
      kernel_fn = lambda: kernel
      if kernel_args is not None:
        raise ValueError('Cannot pass a kernel instance for `kernel_fn` while '
                         'also specifying `kernel_args`.')
    if kernel_args is None:
      kernel_args = ()
    dtype = dtype_util.common_dtype([kernel_fn(*kernel_args), x1, x2],
                                    dtype_hint=tf.float64)
    self._kernel_fn = kernel_fn
    self._kernel_args = tf.nest.map_structure(
        tensor_util.convert_nonref_to_tensor, kernel_args)
    self._x1 = tensor_util.convert_nonref_to_tensor(x1, dtype=dtype)
    self._x2 = tensor_util.convert_nonref_to_tensor(x2, dtype=dtype)
    self._num_matmul_parts = tensor_util.convert_nonref_to_tensor(
        num_matmul_parts, dtype=tf.int32)
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
  def kernel_fn(self):
    return self._kernel_fn

  @property
  def kernel_args(self):
    return self._kernel_args

  def _kernel(self):
    return self.kernel_fn(*self.kernel_args)  # pylint: disable=not-callable

  @property
  def x1(self):
    return self._x1

  @property
  def x2(self):
    return self._x2

  def _x1_x2(self):
    x1 = tf.convert_to_tensor(self.x1)
    x2 = x1 if self.x2 is None else tf.convert_to_tensor(self.x2)
    return x1, x2

  def _x1_x2_axis_kernel(self):
    x1, x2 = self._x1_x2()
    kernel = self._kernel()
    return x1, x2, -kernel.feature_ndims - 1, kernel

  def _matmul(self, x, adjoint=False, adjoint_arg=False):
    x1, x2 = self._x1_x2()
    if (self._num_matmul_parts is None or
        prefer_static.equal(self._num_matmul_parts, 1)):
      return tf.matmul(
          self._kernel().matrix(x1, x2),
          x,
          adjoint_a=adjoint,
          adjoint_b=adjoint_arg)

    if adjoint or adjoint_arg:
      raise NotImplementedError(
          '`adjoint`, `adjoint_arg` NYI when `num_matmul_parts` specified.')

    return _chunked_matmul(
        kernel_fn=self.kernel_fn,
        kernel_args=self.kernel_args,
        x1=x1,
        x2=x2,
        x=x,
        num_matmul_parts=self._num_matmul_parts,
        operator_shape=self.shape_tensor())

  def _shape(self):
    x1, x2, axis, kernel = self._x1_x2_axis_kernel()
    return functools.reduce(
        tf.broadcast_static_shape,
        (kernel.batch_shape, x1.shape[:axis], x2.shape[:axis])).concatenate(
            [x1.shape[axis], x2.shape[axis]])

  def _shape_tensor(self):
    x1, x2, axis, kernel = self._x1_x2_axis_kernel()
    batch_shape = functools.reduce(
        tf.broadcast_dynamic_shape,
        (kernel.batch_shape_tensor(), tf.shape(x1)[:axis], tf.shape(x2)[:axis]))
    return tf.concat([batch_shape, [tf.shape(x1)[axis], tf.shape(x2)[axis]]],
                     axis=0)

  def _diag_part(self):
    x1, x2, axis, kernel = self._x1_x2_axis_kernel()
    ax_minsize = tf.minimum(tf.shape(x1)[axis], tf.shape(x2)[axis])

    def slice_of(xn):
      slice_size = tf.where(
          tf.equal(tf.range(tf.rank(xn)), tf.rank(xn) + axis),
          ax_minsize,
          tf.shape(xn))
      return tf.slice(xn, begin=tf.zeros_like(tf.shape(xn)), size=slice_size)

    return kernel.apply(slice_of(x1), slice_of(x2))

  def row(self, index):
    """Gets a row from the dense operator.

    Args:
      index: The index (indices) of the row[s] to get, may be scalar or up to
        batch shape. Suggest int64 dtype if using with GPU.

    Returns:
      rows: Row[s] of the matrix, with shape `(...batch_shape..., num_cols)`.
        Effectively the same as `operator.to_dense()[..., index, :]` for a
        scalar `index`, analogous to gather for non-scalar.
    """
    index = tf.convert_to_tensor(index, dtype_hint=tf.int64)
    x1, x2, axis, kernel = self._x1_x2_axis_kernel()
    batch_shp = tf.broadcast_dynamic_shape(tf.shape(x1)[:axis], tf.shape(index))
    x1 = tf.broadcast_to(x1,
                         tf.concat([batch_shp, tf.shape(x1)[axis:]], axis=0))
    x1_row = tf.broadcast_to(index, tf.shape(x1)[:axis])
    x1 = tf.gather(x1, x1_row[..., tf.newaxis], batch_dims=len(x1.shape[:axis]))
    return kernel.matrix(x1, x2)[..., 0, :]

  def col(self, index):
    """Gets a column from the dense operator.

    Args:
      index: The index (indices) of the column[s] to get, may be scalar or up to
        batch shape. Suggest int64 dtype if using with GPU.

    Returns:
      cols: Column[s] of the matrix, with shape `(...batch_shape..., num_rows)`.
        Effectively the same as `operator.to_dense()[..., index]` for a scalar
        `index`, analogous to gather for non-scalar.
    """
    index = tf.convert_to_tensor(index, dtype_hint=tf.int64)
    x1, x2, axis, kernel = self._x1_x2_axis_kernel()
    batch_shp = tf.broadcast_dynamic_shape(tf.shape(x2)[:axis], tf.shape(index))
    x2 = tf.broadcast_to(x2,
                         tf.concat([batch_shp, tf.shape(x2)[axis:]], axis=0))
    x2_col = tf.broadcast_to(index, tf.shape(x2)[:axis])
    x2 = tf.gather(x2, x2_col[..., tf.newaxis], batch_dims=len(x2.shape[:axis]))
    return kernel.matrix(x1, x2)[..., 0]
