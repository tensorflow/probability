# Copyright 2020 The TensorFlow Probability Authors.
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
"""Gram-Schmidt orthonormalization algorithm."""

import tensorflow.compat.v2 as tf

from tensorflow_probability.python.internal import prefer_static as ps
from tensorflow_probability.python.internal import tensorshape_util


__all__ = [
    'gram_schmidt',
]


def gram_schmidt(vectors, num_vectors=None):
  """Implementation of the modified Gram-Schmidt orthonormalization algorithm.

  We assume here that the vectors are linearly independent. Zero vectors will be
  left unchanged, but will also consume an iteration against `num_vectors`.

  From [1]: "MGS is numerically equivalent to Householder QR factorization
  applied to the matrix A augmented with a square matrix of zero elements on
  top."

  Historical note, see [1]: "modified" Gram-Schmidt was derived by Laplace [2],
  for elimination and not as an orthogonalization algorithm. "Classical"
  Gram-Schmidt actually came later [2]. Classical Gram-Schmidt has a sometimes
  catastrophic loss of orthogonality for badly conditioned matrices, which is
  discussed further in [1].

  #### References

  [1] Bjorck, A. (1994). Numerics of gram-schmidt orthogonalization. Linear
      Algebra and Its Applications, 197, 297-316.

  [2] P. S. Laplace, Thiorie Analytique des Probabilites. Premier Supple'ment,
      Mme. Courtier, Paris, 1816.

  [3] E. Schmidt, über die Auflosung linearer Gleichungen mit unendlich vielen
      Unbekannten, Rend. Circ. Mat. Pulermo (1) 25:53-77 (1908).

  Args:
    vectors: A Tensor of shape `[..., d, n]` of `d`-dim column vectors to
      orthonormalize.
    num_vectors: Optional, number of leading vectors of the result to make
      orthogonal. If unspecified, then `num_vectors = n`, implying that each
      vector except for the last will be used in sequence.

  Returns:
    A Tensor of shape `[..., d, n]` corresponding to the orthonormalization.
  """
  with tf.name_scope('gram_schmidt'):
    n = ps.shape(vectors)[-1]
    if num_vectors is None:
      num_vectors = n
    cond = lambda vecs, i: i < num_vectors - 1

    def body_fn(vecs, i):
      # Slice out the vector w.r.t. which we're orthogonalizing the rest.
      u = tf.math.l2_normalize(vecs[..., i, tf.newaxis], axis=-2)
      # Find weights by dotting the d x 1 against the d x n.
      weights = tf.einsum('...dm,...dn->...n', u, vecs)
      # Project out vector `u` from the trailing vectors.
      masked_weights = tf.where(
          tf.range(n) > i, weights, 0.)[..., tf.newaxis, :]
      vecs = vecs - tf.math.multiply_no_nan(u, masked_weights)
      tensorshape_util.set_shape(vecs, vectors.shape)
      return vecs, i + 1

    vectors, _ = tf.while_loop(cond, body_fn, (vectors, tf.zeros([], tf.int32)))
    vec_norm = tf.linalg.norm(vectors, ord=2, axis=-2, keepdims=True)
    return tf.math.divide_no_nan(vectors, vec_norm)
