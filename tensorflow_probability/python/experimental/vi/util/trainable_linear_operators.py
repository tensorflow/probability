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

import functools
import numpy as np
import tensorflow.compat.v2 as tf

from tensorflow_probability.python.bijectors import bijector as bijector_lib
from tensorflow_probability.python.bijectors import fill_scale_tril
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import nest_util
from tensorflow_probability.python.internal import prefer_static as ps
from tensorflow_probability.python.internal import samplers
from tensorflow_probability.python.internal import trainable_state_util

from tensorflow.python.util import nest  # pylint: disable=g-direct-tensorflow-import


def _trainable_linear_operator_block(
    operators,
    block_dims=None,
    batch_shape=(),
    dtype=None,
    name=None):
  """Builds a trainable blockwise `tf.linalg.LinearOperator`.

  This function returns a trainable blockwise `LinearOperator`. If `operators`
  is a flat list, it is interpreted as blocks along the diagonal of the
  structure and an instance of `tf.linalg.LinearOperatorBlockDiag` is returned.
  If `operators` is a doubly nested list, then a
  `tf.linalg.LinearOperatorBlockLowerTriangular` instance is returned, with
  the block in row `i` column `j` (`i >= j`) given by `operators[i][j]`.
  The `operators` list may contain `LinearOperator` instances, `LinearOperator`
  subclasses, or callables defining custom constructors (see example below).
  The dimensions of the blocks are given by `block_dims`; this argument may be
  omitted if `operators` contains only `LinearOperator` instances.

  Args:
    operators: A list or tuple containing `LinearOperator` subclasses,
      `LinearOperator` instances, and/or callables returning
      `(init_fn, apply_fn)` pairs. If the list is flat, a
      `tf.linalg.LinearOperatorBlockDiag` instance is returned. Otherwise, the
      list must be singly nested, with the
      first element of length 1, second element of length 2, etc.; the
      elements of the outer list are interpreted as rows of a lower-triangular
      block structure, and a `tf.linalg.LinearOperatorBlockLowerTriangular`
      instance is returned. Callables contained in the lists must take two
      arguments -- `shape`, the shape of the parameter instantiating the
      `LinearOperator`, and `dtype`, the `tf.dtype` of the `LinearOperator` --
      and return a further pair of callables representing a stateless trainable
      operator (see example below).
    block_dims: List or tuple of integers, representing the sizes of the blocks
      along one dimension of the (square) blockwise `LinearOperator`. If
      `operators` contains only `LinearOperator` instances, `block_dims` may be
      `None` and the dimensions are inferred.
    batch_shape: Batch shape of the `LinearOperator`.
    dtype: `tf.dtype` of the `LinearOperator`.
    name: str, name for `tf.name_scope`.
  Yields:
    *parameters: sequence of `trainable_state_util.Parameter` namedtuples.
      These are intended to be consumed by
      `trainable_state_util.as_stateful_builder` and
      `trainable_state_util.as_stateless_builder` to define stateful and
      stateless variants respectively.

  ### Examples

  To build a 5x5 trainable `LinearOperatorBlockDiag` given `LinearOperator`
  subclasses and `block_dims`:

  ```python
  op = build_trainable_linear_operator_block(
    operators=(tf.linalg.LinearOperatorDiag,
               tf.linalg.LinearOperatorLowerTriangular),
    block_dims=[3, 2],
    dtype=tf.float32)
  ```

  If `operators` contains only `LinearOperator` instances, the `block_dims`
  argument is not necessary:

  ```python
  # Builds a 6x6 `LinearOperatorBlockDiag` with batch shape `(4,).
  op = build_trainable_linear_operator_block(
    operators=(tf.linalg.LinearOperatorDiag(tf.Variable(tf.ones((4, 3)))),
               tf.linalg.LinearOperatorFullMatrix([4.]),
               tf.linalg.LinearOperatorIdentity(2)))

  ```

  A custom operator constructor may be specified as a callable taking
  arguments `shape` and `dtype`, and returning a pair of callables
  `(init_fn, apply_fn)` describing a parameterized operator, with the following
  signatures:

  ```python
  raw_parameters = init_fn(seed)
  linear_operator = apply_fn(raw_parameters)
  ```

  For example, to define a custom initialization for a diagonal operator:

  ```python
  import functools

  def diag_operator_with_uniform_initialization(shape, dtype):
    init_fn = functools.partial(
        samplers.uniform, shape, maxval=2., dtype=dtype)
    apply_fn = lambda scale_diag: tf.linalg.LinearOperatorDiag(
        scale_diag, is_non_singular=True)
    return init_fn, apply_fn

  # Build an 8x8 `LinearOperatorBlockLowerTriangular`, with our custom diagonal
  # operator in the upper left block, and `LinearOperator` subclasses in the
  # lower two blocks.
  op = build_trainable_linear_operator_block(
    operators=(diag_operator_with_uniform_initialization,
               (tf.linalg.LinearOperatorFullMatrix,
                tf.linalg.LinearOperatorLowerTriangular)),
    block_dims=[4, 4],
    dtype=tf.float64)
  ```

  """
  with tf.name_scope(name or 'trainable_linear_operator_block'):
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

    def convert_operator(path, op):
      if isinstance(op, tf.linalg.LinearOperator):
        return op
      if len(set(path)) == 1:  # for operators on the diagonal
        shape = ps.concat([batch_shape, [block_dims[path[0]]]], axis=0)
      else:
        shape = ps.concat([batch_shape,
                           [block_dims[path[0]], block_dims[path[1]]]], axis=0)
      if op in _OPERATOR_COROUTINES:
        operator = yield from _OPERATOR_COROUTINES[op](shape=shape, dtype=dtype)
      else:  # Custom stateless constructor.
        init_fn, apply_fn = op(shape=shape, dtype=dtype)
        raw_params = yield trainable_state_util.Parameter(init_fn)
        operator = apply_fn(raw_params)
      return operator

    # Build a structure of component trainable LinearOperators.
    operator_blocks = yield from nest_util.map_structure_coroutine(
        convert_operator,
        operators,
        _with_tuple_paths=True)
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


def _trainable_linear_operator_tril(
    shape,
    scale_initializer=1e-2,
    diag_bijector=None,
    dtype=None,
    name=None):
  """Build a trainable `LinearOperatorLowerTriangular` instance.

  Args:
    shape: Shape of the `LinearOperator`, equal to `[b0, ..., bn, d]`, where
      `b0...bn` are batch dimensions and `d` is the length of the diagonal.
    scale_initializer: Variables are initialized with samples from
      `Normal(0, scale_initializer)`.
    diag_bijector: Bijector to apply to the diagonal of the operator.
    dtype: `tf.dtype` of the `LinearOperator`.
    name: str, name for `tf.name_scope`.
  Yields:
    *parameters: sequence of `trainable_state_util.Parameter` namedtuples.
      These are intended to be consumed by
      `trainable_state_util.as_stateful_builder` and
      `trainable_state_util.as_stateless_builder` to define stateful and
      stateless variants respectively.
  """
  with tf.name_scope(name or 'trainable_linear_operator_tril'):
    if dtype is None:
      dtype = dtype_util.common_dtype(
          [scale_initializer], dtype_hint=tf.float32)

    scale_initializer = tf.convert_to_tensor(scale_initializer, dtype=dtype)
    diag_bijector = diag_bijector or _DefaultScaleDiagonal()
    batch_shape, dim = ps.split(shape, num_or_size_splits=[-1, 1])

    scale_tril_bijector = fill_scale_tril.FillScaleTriL(
        diag_bijector, diag_shift=tf.zeros([], dtype=dtype))
    scale_tril = yield trainable_state_util.Parameter(
        init_fn=lambda seed: scale_tril_bijector(  # pylint: disable=g-long-lambda
            samplers.normal(
                mean=0.,
                stddev=scale_initializer,
                shape=ps.concat([batch_shape, dim * (dim + 1) // 2], axis=0),
                seed=seed,
                dtype=dtype)),
        name='scale_tril',
        constraining_bijector=scale_tril_bijector)
    return tf.linalg.LinearOperatorLowerTriangular(
        tril=scale_tril, is_non_singular=True)


def _trainable_linear_operator_diag(
    shape,
    scale_initializer=1e-2,
    diag_bijector=None,
    dtype=None,
    name=None):
  """Build a trainable `LinearOperatorDiag` instance.

  Args:
    shape: Shape of the `LinearOperator`, equal to `[b0, ..., bn, d]`, where
      `b0...bn` are batch dimensions and `d` is the length of the diagonal.
    scale_initializer: Variables are initialized with samples from
      `Normal(0, scale_initializer)`.
    diag_bijector: Bijector to apply to the diagonal of the operator.
    dtype: `tf.dtype` of the `LinearOperator`.
    name: str, name for `tf.name_scope`.
  Yields:
    *parameters: sequence of `trainable_state_util.Parameter` namedtuples.
      These are intended to be consumed by
      `trainable_state_util.as_stateful_builder` and
      `trainable_state_util.as_stateless_builder` to define stateful and
      stateless variants respectively.
  """
  with tf.name_scope(name or 'trainable_linear_operator_diag'):
    if dtype is None:
      dtype = dtype_util.common_dtype(
          [scale_initializer], dtype_hint=tf.float32)
    scale_initializer = tf.convert_to_tensor(scale_initializer, dtype=dtype)

    diag_bijector = diag_bijector or _DefaultScaleDiagonal()
    scale_diag = yield trainable_state_util.Parameter(
        init_fn=lambda seed: diag_bijector(  # pylint: disable=g-long-lambda
            samplers.normal(
                mean=0.,
                stddev=scale_initializer,
                shape=shape,
                dtype=dtype,
                seed=seed)),
        name='scale_diag',
        constraining_bijector=diag_bijector)
    return tf.linalg.LinearOperatorDiag(scale_diag, is_non_singular=True)


def _trainable_linear_operator_full_matrix(
    shape,
    scale_initializer=1e-2,
    dtype=None,
    name=None):
  """Build a trainable `LinearOperatorFullMatrix` instance.

  Args:
    shape: Shape of the `LinearOperator`, equal to `[b0, ..., bn, h, w]`, where
      `b0...bn` are batch dimensions `h` and `w` are the height and width of the
      matrix represented by the `LinearOperator`.
    scale_initializer: Variables are initialized with samples from
      `Normal(0, scale_initializer)`.
    dtype: `tf.dtype` of the `LinearOperator`.
    name: str, name for `tf.name_scope`.
  Yields:
    *parameters: sequence of `trainable_state_util.Parameter` namedtuples.
      These are intended to be consumed by
      `trainable_state_util.as_stateful_builder` and
      `trainable_state_util.as_stateless_builder` to define stateful and
      stateless variants respectively.
  """
  with tf.name_scope(
      name or 'trainable_linear_operator_full_matrix'):
    if dtype is None:
      dtype = dtype_util.common_dtype([scale_initializer],
                                      dtype_hint=tf.float32)
    scale_initializer = tf.convert_to_tensor(scale_initializer, dtype)
    scale_matrix = yield trainable_state_util.Parameter(
        init_fn=functools.partial(
            samplers.normal,
            mean=0.,
            stddev=scale_initializer,
            shape=shape,
            dtype=dtype),
        name='scale_matrix')
    return tf.linalg.LinearOperatorFullMatrix(matrix=scale_matrix)


def _linear_operator_zeros(shape, dtype=None, name=None):
  """Build an instance of `LinearOperatorZeros`.

  Args:
    shape: Shape of the `LinearOperator`, equal to `[b0, ..., bn, h, w]`, where
      `b0...bn` are batch dimensions `h` and `w` are the height and width of the
      matrix represented by the `LinearOperator`.
    dtype: `tf.dtype` of the `LinearOperator`.
    name: str, name for `tf.name_scope`.
  Yields:
    *parameters: sequence of `trainable_state_util.Parameter` namedtuples.
      These are intended to be consumed by
      `trainable_state_util.as_stateful_builder` and
      `trainable_state_util.as_stateless_builder` to define stateful and
      stateless variants respectively.
  """
  with tf.name_scope(name or 'linear_operator_zeros'):
    batch_shape, rows, cols = ps.split(
        shape, num_or_size_splits=[-1, 1, 1])
    num_rows, num_cols = rows[0], cols[0]
    is_square = num_rows == num_cols
    return tf.linalg.LinearOperatorZeros(
        num_rows, num_cols, batch_shape=batch_shape, is_square=is_square,
        is_self_adjoint=is_square, dtype=dtype)
    # Tell Python that this fn is really a (trivial) generator.
    yield  # pylint: disable=unreachable


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
    return tf.math.abs(x) + dtype_util.eps(dtype)

  def _inverse(self, y):
    dtype = dtype_util.base_dtype(y.dtype)
    return tf.math.abs(y) - np.finfo(dtype.as_numpy_dtype).eps

  def _inverse_log_det_jacobian(self, y):
    return tf.zeros([], dtype_util.base_dtype(y.dtype))


_OPERATOR_COROUTINES = {
    tf.linalg.LinearOperatorLowerTriangular:
        _trainable_linear_operator_tril,
    tf.linalg.LinearOperatorDiag: _trainable_linear_operator_diag,
    tf.linalg.LinearOperatorFullMatrix:
        _trainable_linear_operator_full_matrix,
    tf.linalg.LinearOperatorZeros: _linear_operator_zeros,
    None: _linear_operator_zeros,
}


# TODO(davmre): also expose stateless builders.
build_trainable_linear_operator_block = (
    trainable_state_util.as_stateful_builder(
        _trainable_linear_operator_block))
build_trainable_linear_operator_tril = (
    trainable_state_util.as_stateful_builder(
        _trainable_linear_operator_tril))
build_trainable_linear_operator_diag = (
    trainable_state_util.as_stateful_builder(
        _trainable_linear_operator_diag))
build_trainable_linear_operator_full_matrix = (
    trainable_state_util.as_stateful_builder(
        _trainable_linear_operator_full_matrix))
build_linear_operator_zeros = (
    trainable_state_util.as_stateful_builder(
        _linear_operator_zeros))
