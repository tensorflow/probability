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
"""Invertible 1x1 Convolution used in GLOW."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow_probability.python.bijectors import bijector
from tensorflow_probability.python.math.linalg import lu_reconstruct
from tensorflow_probability.python.math.linalg import lu_solve


from tensorflow.python.ops.linalg import linear_operator_util


__all__ = [
    'MatvecLU'
]


class MatvecLU(bijector.Bijector):
  """Matrix-vector multiply using LU decomposition.

  This bijector is identical to the "Convolution1x1" used in Glow
  [(Kingma and Dhariwal, 2018)[1].

  Warning: this bijector never verifies the scale matrix (as parameterized by LU
  decomposition) is invertible. Ensuring this is the case is the caller's
  responsibility.

  #### Examples

  Here's an example of initialization via random weights matrix:

  ```python
  def trainable_lu_factorization(
      event_size, batch_shape=(), seed=None, dtype=tf.float32, name=None):
    with tf.compat.v1.name_scope(name, 'trainable_lu_factorization',
                       [event_size, batch_shape]):
      event_size = tf.convert_to_tensor(
          event_size, preferred_dtype=tf.int32, name='event_size')
      batch_shape = tf.convert_to_tensor(
          batch_shape, preferred_dtype=event_size.dtype, name='batch_shape')
      random_matrix = tf.random_uniform(
          shape=tf.concat([batch_shape, [event_size, event_size]], axis=0),
          dtype=dtype,
          seed=seed)
      random_orthonormal = tf.linalg.qr(random_matrix)[0]
      lower_upper, permutation = tf.linalg.lu(random_orthonormal)
      lower_upper = tf.Variable(
          initial_value=lower_upper,
          trainable=True,
          use_resource=True,
          name='lower_upper')
      return lower_upper, permutation

  channels = 3
  conv1x1 = tfb.MatvecLU(*trainable_lu_factorization(channels),
                         validate_args=True)
  x = tf.random_uniform(shape=[2, 28, 28, channels])
  fwd = conv1x1.forward(x)
  rev_fwd = conv1x1.inverse(fwd)
  # ==> x
  ```

  To initialize this variable outside of TensorFlow, one can also use SciPy,
  e.g.,

  ```python
  def lu_factorized_random_orthonormal_matrix(channels, dtype=np.float32):
    random_matrix = np.random.rand(channels, channels).astype(dtype)
    lower_upper = scipy.linalg.qr(random_matrix)[0]
    permutation = scipy.linalg.lu(lower_upper, overwrite_a=True)[0]
    permutation = np.argmax(permutation, axis=-2)
    return lower_upper, permutation
  ```

  #### References

  [1]: Diederik P. Kingma, Prafulla Dhariwal. Glow: Generative Flow with
       Invertible 1x1 Convolutions. _arXiv preprint arXiv:1807.03039_, 2018.
       https://arxiv.org/abs/1807.03039
  """

  def __init__(self,
               lower_upper,
               permutation,
               validate_args=False,
               name=None):
    """Creates the MatvecLU bijector.

    Args:
      lower_upper: The LU factorization as returned by `tf.linalg.lu`.
      permutation: The LU factorization permutation as returned by
        `tf.linalg.lu`.
      validate_args: Python `bool` indicating whether arguments should be
        checked for correctness.
        Default value: `False`.
      name: Python `str` name given to ops managed by this object.
        Default value: `None` (i.e., "MatvecLU").

    Raises:
      ValueError: If both/neither `channels` and `lower_upper`/`permutation` are
        specified.
    """
    with tf.compat.v1.name_scope(name, 'MatvecLU',
                                 [lower_upper, permutation]) as name:
      self._lower_upper = tf.convert_to_tensor(
          value=lower_upper, dtype_hint=tf.float32, name='lower_upper')
      self._permutation = tf.convert_to_tensor(
          value=permutation, dtype_hint=tf.int32, name='permutation')
    super(MatvecLU, self).__init__(
        is_constant_jacobian=True,
        forward_min_event_ndims=1,
        validate_args=validate_args,
        name=name)

  @property
  def lower_upper(self):
    return self._lower_upper

  @property
  def permutation(self):
    return self._permutation

  def _forward(self, x):
    w = lu_reconstruct(lower_upper=self.lower_upper,
                       perm=self.permutation,
                       validate_args=self.validate_args)
    return linear_operator_util.matmul_with_broadcast(
        w, x[..., tf.newaxis])[..., 0]

  def _inverse(self, y):
    return lu_solve(
        lower_upper=self.lower_upper,
        perm=self.permutation,
        rhs=y[..., tf.newaxis],
        validate_args=self.validate_args)[..., 0]

  def _forward_log_det_jacobian(self, unused_x):
    return tf.reduce_sum(
        input_tensor=tf.math.log(
            tf.abs(tf.linalg.tensor_diag_part(self.lower_upper))),
        axis=-1)
