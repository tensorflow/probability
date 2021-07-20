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
"""No-pivot LDL and friends."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v2 as tf
from tensorflow_probability.python.internal import tensorshape_util


class _Slice2Idx:
  """Utility to convert numpy basic slices into TF scatter_nd indices."""

  def __init__(self, tensor):
    self.rank = tensorshape_util.rank(tensor.shape)
    if self.rank is None:
      raise ValueError('_Slice2Idx: dynamic tensor ranks not supported yet.')
    shape = tf.shape(tensor)
    self.ranges = [tf.range(shape[d], dtype=tf.int32) for d in range(self.rank)]

  def __getitem__(self, slices):
    if not isinstance(slices, tuple):
      slices = [slices]
    else:
      slices = list(slices)
    if Ellipsis in slices:
      idx = slices.index(Ellipsis)
      slices[idx:idx+1] = [slice(None)] * (self.rank - len(slices) + 1)
    # Remove trailing full slices for performance.
    while (slices
           and isinstance(slices[-1], slice)
           and slices[-1] == slice(None)):
      slices.pop()
    grid = tf.meshgrid(*(rng[sl] for rng, sl in zip(self.ranges, slices)),
                       indexing='ij')
    stack = tf.stack(grid, axis=-1)
    to_squeeze = [i for i, sl in enumerate(slices) if not isinstance(sl, slice)]
    if to_squeeze:
      stack = tf.squeeze(stack, axis=to_squeeze)
    return stack


def no_pivot_ldl(matrix, name='no_pivot_ldl'):
  """Non-pivoted batched LDL factorization.

  Performs the LDL factorization, using the outer product algorithm from [1]. No
  pivoting (or block pivoting) is done, so this should be less stable than
  e.g. Bunch-Kaufman sytrf. This is implemented as a tf.foldl, so should have
  gradients and be accelerator-friendly, but is not particularly performant.

  If compiling with XLA, make sure any surrounding GradientTape is also
  XLA-compiled (b/193584244).

  #### References
  [1]: Gene H. Golub, Charles F. Van Loan. Matrix Computations, 4th ed., 2013.

  Args:
    matrix: A batch of symmetric square matrices, with shape `[..., n, n]`.
    name: Python `str` name prefixed to Ops created by this function.
      Default value: 'no_pivot_ldl'.

  Returns:
    triangular_factor: The unit lower triangular L factor of the LDL
      factorization of `matrix`, with the same shape `[..., n, n]`. Callers
      should check for `nans` and other indicators of instability.
    diag: The diagonal from the LDL factorization, with shape `[..., n]`.
  """
  with tf.name_scope(name) as name:
    matrix = tf.convert_to_tensor(matrix)
    triangular_factor = tf.linalg.band_part(matrix, num_lower=-1, num_upper=0)
    # TODO(b/182276317) Deal with dynamic ranks better.
    slix = _Slice2Idx(triangular_factor)

    def fn(triangular_factor, i):
      column_head = triangular_factor[..., i, i, tf.newaxis]
      column_tail = triangular_factor[..., i+1:, i]
      rescaled_tail = column_tail / column_head
      triangular_factor = tf.tensor_scatter_nd_update(
          triangular_factor,
          slix[..., i+1:, i],
          rescaled_tail)
      triangular_factor = tf.tensor_scatter_nd_sub(
          triangular_factor,
          slix[..., i+1:, i+1:],
          tf.linalg.band_part(
              tf.einsum('...i,...j->...ij', column_tail, rescaled_tail),
              num_lower=-1, num_upper=0))
      return triangular_factor

    triangular_factor = tf.foldl(
        fn=fn,
        elems=tf.range(tf.shape(triangular_factor)[-1]),
        initializer=triangular_factor)

    diag = tf.linalg.diag_part(triangular_factor)
    triangular_factor = tf.linalg.set_diag(
        triangular_factor, tf.ones_like(diag))

    return triangular_factor, diag


def simple_robustified_cholesky(
    matrix, tol=1e-6, name='simple_robustified_cholesky'):
  """Use `no_pivot_ldl` to robustify a Cholesky factorization.

  Given a symmetric matrix `A`, this function attempts to give a factorization
  `A + E = LL^T` where `L` is lower triangular, `LL^T` is positive definite, and
  `E` is small in some suitable sense. This is useful for nearly positive
  definite symmetric matrices that are otherwise numerically difficult to
  Cholesky factor.

  The algorithm proceeds as follows. The input is factored `A = LDL^T`, and the
  too-small diagonal entries of `D` are increased to the tolerance.  Then `L @
  sqrt(D)` is returned.

  This algorithm is similar in spirit to a true modified Cholesky factorization
  ([1], [2]). However, it does not use pivoting or other strategies to ensure
  stability, so may not work well for e.g. ill-conditioned matrices. Generally
  speaking, a modified Cholesky factorization of a symmetric matrix `A` is a
  factorization `P(A+E)P^T = LDL^T`, where `P` is a permutation matrix, `L` is
  unit lower triangular, and `D` is (block) diagonal and positive
  definite. Ideally such an algorithm would ensure the following:

  1. If `A` is sufficiently positive definite, `E` is zero.

  2. If `F` is the smallest matrix (in Frobenius norm) such that `A + F` is
    positive definite, then `E` is not much larger than `F`.

  3. `A + E` is reasonably well-conditioned.

  4. It is not too much more expensive than the usual Cholesky factorization.

  The references give more sophisticated algorithms to ensure all of the
  above. In the simple case where `A = LDL^T` does not require pivoting,
  `simple_robustified_cholesky` will agree and satisfy the above
  criteria. However, in general it may fail to be stable or satisfy 2 and 3.

  #### References

  [1]: Nicholas Higham. What is a modified Cholesky factorization?
    https://nhigham.com/2020/12/22/what-is-a-modified-cholesky-factorization/

  [2]: Sheung Hun Cheng and Nicholas Higham, A Modified Cholesky Algorithm Based
    on a Symmetric Indefinite Factorization, SIAM J. Matrix Anal. Appl. 19(4),
    1097–1110, 1998.

  Args:
    matrix: A batch of symmetric square matrices, with shape `[..., n, n]`.
    tol: Minimum for the diagonal.  Default: 1e-6.
    name: Python `str` name prefixed to Ops created by this function.
      Default value: 'simple_robustified_cholesky'.

  Returns:
    triangular_factor: The lower triangular Cholesky factor, modified as above.
      This will have shape `[..., n, n]`. Callers should check for `nans` or
      other evidence of instability.
  """
  with tf.name_scope(name) as name:
    matrix = tf.convert_to_tensor(matrix)
    tol = tf.convert_to_tensor(tol, dtype=matrix.dtype)
    triangular_factor, diag = no_pivot_ldl(matrix)
    new_diag = tf.where(diag < tol, tol, diag)
    return tf.einsum(
        '...ij,...j->...ij', triangular_factor, tf.math.sqrt(new_diag))
