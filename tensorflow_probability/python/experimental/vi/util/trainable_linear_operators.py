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
"""Utilities for constructing trainable linear operators."""

import numpy as np
import tensorflow.compat.v2 as tf

from tensorflow_probability.python import util as tfp_util
from tensorflow_probability.python.bijectors import bijector as bijector_lib
from tensorflow_probability.python.bijectors import fill_scale_tril
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import prefer_static as ps
from tensorflow_probability.python.internal import samplers

from tensorflow.python.util import nest  # pylint: disable=g-direct-tensorflow-import


def build_trainable_linear_operator_block(
    operators,
    block_dims=None,
    batch_shape=(),
    dtype=None,
    seed=None,
    name=None):
  """Builds a trainable blockwise `tf.linalg.LinearOperator`.

  This function returns a trainable blockwise `LinearOperator`. If `operators`
  is a flat list, it is interpreted as blocks along the diagonal of the
  structure and an instance of `tf.linalg.LinearOperatorBlockDiag` is returned.
  If `operators` is a doubly nested list, then a
  `tf.linalg.LinearOperatorBlockLowerTriangular` instance is returned, with
  the block in row `i` column `j` (`i >= j`) given by `operators[i][j]`.
  The `operators` list may contain `LinearOperator` instances, `LinearOperator`
  subclasses, or callables that return `LinearOperator` instances. The
  dimensions of the blocks are given by `block_dims`; this argument may be
  omitted if `operators` contains only `LinearOperator` instances.

  ### Examples

  ```python
  # Build a 5x5 trainable `LinearOperatorBlockDiag` given `LinearOperator`
  # subclasses and `block_dims`.
  op = build_trainable_linear_operator_block(
    operators=(tf.linalg.LinearOperatorDiag,
               tf.linalg.LinearOperatorLowerTriangular),
    block_dims=[3, 2],
    dtype=tf.float32)

  # Build an 8x8 `LinearOperatorBlockLowerTriangular`, with a callable that
  # returns a `LinearOperator` in the upper left block, and `LinearOperator`
  # subclasses in the lower two blocks.
  op = build_trainable_linear_operator_block(
    operators=(
      (lambda shape, dtype: tf.linalg.LinearOperatorScaledIdentity(
         num_rows=shape[-1], multiplier=tf.Variable(1., dtype=dtype))),
      (tf.linalg.LinearOperatorFullMatrix,
      tf.linalg.LinearOperatorLowerTriangular))
    block_dims=[4, 4],
    dtype=tf.float64)

  # Build a 6x6 `LinearOperatorBlockDiag` with batch shape `(4,)`. Since
  # `operators` contains only `LinearOperator` instances, the `block_dims`
  # argument is not necessary.
  op = build_trainable_linear_operator_block(
    operators=(tf.linalg.LinearOperatorDiag(tf.Variable(tf.ones((4, 3)))),
               tf.linalg.LinearOperatorFullMatrix([4.]),
               tf.linalg.LinearOperatorIdentity(2)))
  ```

  Args:
    operators: A list or tuple containing `LinearOperator` subclasses,
      `LinearOperator` instances, or callables returning `LinearOperator`
      instances. If the list is flat, a `tf.linalg.LinearOperatorBlockDiag`
      instance is returned. Otherwise, the list must be singly nested, with the
      first element of length 1, second element of length 2, etc.; the
      elements of the outer list are interpreted as rows of a lower-triangular
      block structure, and a `tf.linalg.LinearOperatorBlockLowerTriangular`
      instance is returned. Callables contained in the lists must take three
      arguments -- `shape`, the shape of the `tf.Variable` instantiating the
      `LinearOperator`, `dtype`, the `tf.dtype` of the `LinearOperator`, and
      `seed`, a seed for generating random values.
    block_dims: List or tuple of integers, representing the sizes of the blocks
      along one dimension of the (square) blockwise `LinearOperator`. If
      `operators` contains only `LinearOperator` instances, `block_dims` may be
      `None` and the dimensions are inferred.
    batch_shape: Batch shape of the `LinearOperator`.
    dtype: `tf.dtype` of the `LinearOperator`.
    seed: PRNG seed; see `tfp.random.sanitize_seed` for details.
    name: str, name for `tf.name_scope`.

  Returns:
    Trainable instance of `tf.linalg.LinearOperatorBlockDiag` or
      `tf.linalg.LinearOperatorBlockLowerTriangular`.
  """
  with tf.name_scope(name or 'build_trainable_blockwise_tril_operator'):
    operator_instances = [op for op in nest.flatten(operators)
                          if isinstance(op, tf.linalg.LinearOperator)]
    if (block_dims is None
        and len(operator_instances) < len(nest.flatten(operators))):
      # If `operator_instances` contains fewer elements than `operators`,
      # then some elements of `operators` are not instances of `LinearOperator`.
      raise ValueError('Argument `block_dims` must be defined unless '
                       '`operators` contains only `tf.linalg.LinearOperator` '
                       'instances.')

    batch_shape = ps.cast(batch_shape, tf.int32)
    if dtype is None:
      dtype = dtype_util.common_dtype(operator_instances)

    def convert_operator(path, op, seed):
      if isinstance(op, tf.linalg.LinearOperator):
        return op
      builder = _OPERATOR_BUILDERS.get(op, op)
      if len(set(path)) == 1:  # for operators on the diagonal
        return builder(
            ps.concat([batch_shape, [block_dims[path[0]]]], axis=0),
            dtype=dtype,
            seed=seed)
      return builder(
          ps.concat([batch_shape, [block_dims[path[0]], block_dims[path[1]]]],
                    axis=0),
          dtype=dtype,
          seed=seed)

    operator_blocks = nest.map_structure_with_tuple_paths(
        convert_operator,
        operators,
        tf.nest.pack_sequence_as(
            operators,
            samplers.split_seed(
                seed, n=len(tf.nest.flatten(operators)))))
    paths = nest.yield_flat_paths(operators)
    if all(len(p) == 1 for p in paths):
      return tf.linalg.LinearOperatorBlockDiag(
          operator_blocks, is_non_singular=True)
    elif all(len(p) == 2 for p in paths):
      return tf.linalg.LinearOperatorBlockLowerTriangular(
          operator_blocks, is_non_singular=True)
    else:
      raise ValueError(
          'Argument `operators` must be a flat or singly-nested sequence.')


def build_trainable_linear_operator_tril(
    shape,
    scale_initializer=1e-2,
    diag_bijector=None,
    dtype=None,
    seed=None,
    name=None):
  """Build a trainable `LinearOperatorLowerTriangular` instance.

  Args:
    shape: Shape of the `LinearOperator`, equal to `[b0, ..., bn, d]`, where
      `b0...bn` are batch dimensions and `d` is the length of the diagonal.
    scale_initializer: Variables are initialized with samples from
      `Normal(0, scale_initializer)`.
    diag_bijector: Bijector to apply to the diagonal of the operator.
    dtype: `tf.dtype` of the `LinearOperator`.
    seed: PRNG seed; see `tfp.random.sanitize_seed` for details.
    name: str, name for `tf.name_scope`.

  Returns:
    operator: Trainable instance of `tf.linalg.LinearOperatorLowerTriangular`.
  """
  with tf.name_scope(name or 'build_trainable_linear_operator_tril'):
    if dtype is None:
      dtype = dtype_util.common_dtype(
          [scale_initializer], dtype_hint=tf.float32)

    scale_initializer = tf.convert_to_tensor(scale_initializer, dtype=dtype)
    diag_bijector = diag_bijector or _DefaultScaleDiagonal()
    batch_shape, dim = ps.split(shape, num_or_size_splits=[-1, 1])

    scale_tril_bijector = fill_scale_tril.FillScaleTriL(
        diag_bijector, diag_shift=tf.zeros([], dtype=dtype))
    flat_initial_scale = samplers.normal(
        mean=0.,
        stddev=scale_initializer,
        shape=ps.concat([batch_shape, dim * (dim + 1) // 2], axis=0),
        seed=seed,
        dtype=dtype)
    return tf.linalg.LinearOperatorLowerTriangular(
        tril=tfp_util.TransformedVariable(
            scale_tril_bijector.forward(flat_initial_scale),
            bijector=scale_tril_bijector,
            name='tril'),
        is_non_singular=True)


def build_trainable_linear_operator_diag(
    shape,
    scale_initializer=1e-2,
    diag_bijector=None,
    dtype=None,
    seed=None,
    name=None):
  """Build a trainable `LinearOperatorDiag` instance.

  Args:
    shape: Shape of the `LinearOperator`, equal to `[b0, ..., bn, d]`, where
      `b0...bn` are batch dimensions and `d` is the length of the diagonal.
    scale_initializer: Variables are initialized with samples from
      `Normal(0, scale_initializer)`.
    diag_bijector: Bijector to apply to the diagonal of the operator.
    dtype: `tf.dtype` of the `LinearOperator`.
    seed: PRNG seed; see `tfp.random.sanitize_seed` for details.
    name: str, name for `tf.name_scope`.

  Returns:
    operator: Trainable instance of `tf.linalg.LinearOperatorDiag`.
  """
  with tf.name_scope(name or 'build_trainable_linear_operator_diag'):
    if dtype is None:
      dtype = dtype_util.common_dtype(
          [scale_initializer], dtype_hint=tf.float32)
    scale_initializer = tf.convert_to_tensor(scale_initializer, dtype=dtype)

    diag_bijector = diag_bijector or _DefaultScaleDiagonal()
    initial_scale_diag = samplers.normal(
        mean=0.,
        stddev=scale_initializer,
        shape=shape,
        dtype=dtype,
        seed=seed)
    return tf.linalg.LinearOperatorDiag(
        tfp_util.TransformedVariable(
            diag_bijector.forward(initial_scale_diag),
            bijector=diag_bijector,
            name='diag'),
        is_non_singular=True)


def build_trainable_linear_operator_full_matrix(
    shape,
    scale_initializer=1e-2,
    dtype=None,
    seed=None,
    name=None):
  """Build a trainable `LinearOperatorFullMatrix` instance.

  Args:
    shape: Shape of the `LinearOperator`, equal to `[b0, ..., bn, h, w]`, where
      `b0...bn` are batch dimensions `h` and `w` are the height and width of the
      matrix represented by the `LinearOperator`.
    scale_initializer: Variables are initialized with samples from
      `Normal(0, scale_initializer)`.
    dtype: `tf.dtype` of the `LinearOperator`.
    seed: PRNG seed; see `tfp.random.sanitize_seed` for details.
    name: str, name for `tf.name_scope`.

  Returns:
    operator: Trainable instance of `tf.linalg.LinearOperatorFullMatrix`.
  """
  with tf.name_scope(name or 'build_trainable_linear_operator_full_matrix'):
    if dtype is None:
      dtype = dtype_util.common_dtype([scale_initializer],
                                      dtype_hint=tf.float32)
    scale_initializer = tf.convert_to_tensor(scale_initializer, dtype)

    initial_scale_matrix = samplers.normal(
        mean=0., stddev=scale_initializer, shape=shape, dtype=dtype, seed=seed)
    return tf.linalg.LinearOperatorFullMatrix(
        matrix=tf.Variable(initial_scale_matrix, name='full_matrix'))


def build_linear_operator_zeros(
    shape,
    dtype=None,
    seed=None,
    name=None):
  """Build an instance of `LinearOperatorZeros`.

  Args:
    shape: Shape of the `LinearOperator`, equal to `[b0, ..., bn, h, w]`, where
      `b0...bn` are batch dimensions `h` and `w` are the height and width of the
      matrix represented by the `LinearOperator`.
    dtype: `tf.dtype` of the `LinearOperator`.
    seed: PRNG seed; see `tfp.random.sanitize_seed` for details.
    name: str, name for `tf.name_scope`.

  Returns:
    operator: Instance of `tf.linalg.LinearOperatorZeros`.
  """
  del seed  # Unused.
  with tf.name_scope(name or 'build_linear_operator_zeros'):
    batch_shape, rows, cols = ps.split(
        shape, num_or_size_splits=[-1, 1, 1])
    num_rows, num_cols = rows[0], cols[0]
    is_square = num_rows == num_cols
    return tf.linalg.LinearOperatorZeros(
        num_rows, num_cols, batch_shape=batch_shape, is_square=is_square,
        is_self_adjoint=is_square, dtype=dtype)


class _DefaultScaleDiagonal(bijector_lib.AutoCompositeTensorBijector):
  """Default bijector for constraining the diagonal of scale matrices."""

  def __init__(self):
    parameters = dict(locals())
    name = 'default_scale_diagonal'
    with tf.name_scope(name) as name:
      super(_DefaultScaleDiagonal, self).__init__(
          forward_min_event_ndims=0,
          validate_args=False,
          parameters=parameters,
          name=name)

  def _forward(self, x):
    dtype = dtype_util.base_dtype(x.dtype)
    return tf.math.abs(x) + np.finfo(dtype.as_numpy_dtype).eps

  def _inverse(self, y):
    dtype = dtype_util.base_dtype(y.dtype)
    return tf.math.abs(y) - np.finfo(dtype.as_numpy_dtype).eps

  def _inverse_log_det_jacobian(self, y):
    return tf.zeros([], dtype_util.base_dtype(y.dtype))


_OPERATOR_BUILDERS = {
    tf.linalg.LinearOperatorLowerTriangular:
        build_trainable_linear_operator_tril,
    tf.linalg.LinearOperatorDiag: build_trainable_linear_operator_diag,
    tf.linalg.LinearOperatorFullMatrix:
        build_trainable_linear_operator_full_matrix,
    tf.linalg.LinearOperatorZeros: build_linear_operator_zeros,
    None: build_linear_operator_zeros,
}
