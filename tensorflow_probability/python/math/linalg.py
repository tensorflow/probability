# Copyright 2018 The TensorFlow Probability Authors.
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
"""Functions for common linear algebra operations.

Note: Many of these functions will eventually be migrated to core TensorFlow.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
import numpy as np

import tensorflow as tf


__all__ = [
    'pinv',
]


def pinv(a, rcond=None, validate_args=False, name=None):
  """Compute the Moore-Penrose pseudo-inverse of a matrix.

  Calculate the [generalized inverse of a matrix](
  https://en.wikipedia.org/wiki/Moore%E2%80%93Penrose_inverse) using its
  singular-value decomposition (SVD) and including all large singular values.

  The pseudo-inverse of a matrix `A`, is defined as: "the matrix that 'solves'
  [the least-squares problem] `A @ x = b`," i.e., if `x_hat` is a solution, then
  `A_pinv` is the matrix such that `x_hat = A_pinv @ b`. It can be shown that if
  `U @ Sigma @ V.T = A` is the singular value decomposition of `A`, then
  `A_pinv = V @ inv(Sigma) U^T`. [(Strang, 1980)][1]

  This function is analogous to [`numpy.linalg.pinv`](
  https://docs.scipy.org/doc/numpy/reference/generated/numpy.linalg.pinv.html).
  It differs only in default value of `rcond`. In `numpy.linalg.pinv`, the
  default `rcond` is `1e-15`. Here the default is
  `10. * max(num_rows, num_cols) * np.finfo(dtype).eps`.

  Args:
    a: (Batch of) `float`-like matrix-shaped `Tensor`(s) which are to be
      pseudo-inverted.
    rcond: `Tensor` of small singular value cutoffs.  Singular values smaller
      (in modulus) than `rcond` * largest_singular_value (again, in modulus) are
      set to zero. Must broadcast against `tf.shape(a)[:-2]`.
      Default value: `10. * max(num_rows, num_cols) * np.finfo(a.dtype).eps`.
    validate_args: When `True`, additional assertions might be embedded in the
      graph.
      Default value: `False` (i.e., no graph assertions are added).
    name: Python `str` prefixed to ops created by this function.
      Default value: "pinv".

  Returns:
    a_pinv: The pseudo-inverse of input `a`. Has same shape as `a` except
      rightmost two dimensions are transposed.

  Raises:
    TypeError: if input `a` does not have `float`-like `dtype`.
    ValueError: if input `a` has fewer than 2 dimensions.

  #### Examples

  ```python
  import tensorflow as tf
  import tensorflow_probability as tfp

  a = tf.constant([[1.,  0.4,  0.5],
                   [0.4, 0.2,  0.25],
                   [0.5, 0.25, 0.35]])
  tf.matmul(tfp.math.pinv(a), a)
  # ==> array([[1., 0., 0.],
               [0., 1., 0.],
               [0., 0., 1.]], dtype=float32)

  a = tf.constant([[1.,  0.4,  0.5,  1.],
                   [0.4, 0.2,  0.25, 2.],
                   [0.5, 0.25, 0.35, 3.]])
  tf.matmul(tfp.math.pinv(a), a)
  # ==> array([[ 0.76,  0.37,  0.21, -0.02],
               [ 0.37,  0.43, -0.33,  0.02],
               [ 0.21, -0.33,  0.81,  0.01],
               [-0.02,  0.02,  0.01,  1.  ]], dtype=float32)
  ```

  #### References

  [1]: G. Strang. "Linear Algebra and Its Applications, 2nd Ed." Academic Press,
       Inc., 1980, pp. 139-142.
  """
  with tf.name_scope(name, 'pinv', [a, rcond]):
    a = tf.convert_to_tensor(a, name='a')

    if not a.dtype.is_floating:
      raise TypeError('Input `a` must have `float`-like `dtype` '
                      '(saw {}).'.format(a.dtype.name))
    if a.shape.ndims is not None:
      if a.shape.ndims < 2:
        raise ValueError('Input `a` must have at least 2 dimensions '
                         '(saw: {}).'.format(a.shape.ndims))
    elif validate_args:
      assert_rank_at_least_2 = tf.assert_rank_at_least(
          a, rank=2,
          message='Input `a` must have at least 2 dimensions.')
      with tf.control_dependencies([assert_rank_at_least_2]):
        a = tf.identity(a)

    dtype = a.dtype.as_numpy_dtype

    if rcond is None:
      def get_dim_size(dim):
        if a.shape.ndims is not None and a.shape[dim].value is not None:
          return a.shape[dim].value
        return tf.shape(a)[dim]
      num_rows = get_dim_size(-2)
      num_cols = get_dim_size(-1)
      if isinstance(num_rows, int) and isinstance(num_cols, int):
        max_rows_cols = float(max(num_rows, num_cols))
      else:
        max_rows_cols = tf.cast(tf.maximum(num_rows, num_cols), dtype)
      rcond = 10. * max_rows_cols * np.finfo(dtype).eps

    rcond = tf.convert_to_tensor(rcond, dtype=dtype, name='rcond')

    # Calculate pseudo inverse via SVD.
    # Note: if a is symmetric then u == v. (We might observe additional
    # performance by explicitly setting `v = u` in such cases.)
    [
        singular_values,         # Sigma
        left_singular_vectors,   # U
        right_singular_vectors,  # V
    ] = tf.linalg.svd(a, full_matrices=False, compute_uv=True)

    # Saturate small singular values to inf. This has the effect of make
    # `1. / s = 0.` while not resulting in `NaN` gradients.
    cutoff = rcond * tf.reduce_max(singular_values, axis=-1)
    singular_values = tf.where(
        singular_values > cutoff[..., tf.newaxis],
        singular_values,
        tf.fill(tf.shape(singular_values), np.array(np.inf, dtype)))

    # Although `a == tf.matmul(u, s * v, transpose_b=True)` we swap
    # `u` and `v` here so that `tf.matmul(pinv(A), A) = tf.eye()`, i.e.,
    # a matrix inverse has "transposed" semantics.
    a_pinv = tf.matmul(
        right_singular_vectors / singular_values[..., tf.newaxis, :],
        left_singular_vectors,
        adjoint_b=True)

    if a.shape.ndims is not None:
      a_pinv.set_shape(a.shape[:-2].concatenate([a.shape[-1], a.shape[-2]]))

    return a_pinv
