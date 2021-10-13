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

import tensorflow.compat.v2 as tf

from tensorflow_probability.python.bijectors import bijector
from tensorflow_probability.python.internal import assert_util
from tensorflow_probability.python.internal import parameter_properties
from tensorflow_probability.python.internal import prefer_static as ps
from tensorflow_probability.python.internal import tensor_util
from tensorflow_probability.python.math.linalg import lu_reconstruct
from tensorflow_probability.python.math.linalg import lu_reconstruct_assertions
from tensorflow_probability.python.math.linalg import lu_solve
from tensorflow.python.util import deprecation  # pylint: disable=g-direct-tensorflow-import


__all__ = [
    'MatvecLU',  # Deprecated
    'ScaleMatvecLU',
]


class ScaleMatvecLU(bijector.AutoCompositeTensorBijector):
  """Matrix-vector multiply using LU decomposition.

  This bijector is identical to the 'Convolution1x1' used in Glow
  [(Kingma and Dhariwal, 2018)[1].

  #### Examples

  Here's an example of initialization via random weights matrix:

  ```python
  def trainable_lu_factorization(
      event_size, batch_shape=(), seed=None, dtype=tf.float32, name=None):
    with tf.name_scope(name or 'trainable_lu_factorization'):
      event_size = tf.convert_to_tensor(
          event_size, dtype_hint=tf.int32, name='event_size')
      batch_shape = tf.convert_to_tensor(
          batch_shape, dtype_hint=event_size.dtype, name='batch_shape')
      random_matrix = tf.random.uniform(
          shape=tf.concat([batch_shape, [event_size, event_size]], axis=0),
          dtype=dtype,
          seed=seed)
      random_orthonormal = tf.linalg.qr(random_matrix)[0]
      lower_upper, permutation = tf.linalg.lu(random_orthonormal)
      lower_upper = tf.Variable(
          initial_value=lower_upper,
          trainable=True,
          name='lower_upper')
      # Initialize a non-trainable variable for the permutation indices so
      # that its value isn't re-sampled from run-to-run.
      permutation = tf.Variable(
          initial_value=permutation,
          trainable=False,
          name='permutation')
      return lower_upper, permutation

  channels = 3
  conv1x1 = tfb.ScaleMatvecLU(*trainable_lu_factorization(channels),
                              validate_args=True)
  x = tf.random.uniform(shape=[2, 28, 28, channels])
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
    """Creates the ScaleMatvecLU bijector.

    Args:
      lower_upper: The LU factorization as returned by `tf.linalg.lu`.
      permutation: The LU factorization permutation as returned by
        `tf.linalg.lu`.
      validate_args: Python `bool` indicating whether arguments should be
        checked for correctness.
        Default value: `False`.
      name: Python `str` name given to ops managed by this object.
        Default value: `None` (i.e., 'ScaleMatvecLU').

    Raises:
      ValueError: If both/neither `channels` and `lower_upper`/`permutation` are
        specified.
    """
    parameters = dict(locals())
    with tf.name_scope(name or 'ScaleMatvecLU') as name:
      self._lower_upper = tensor_util.convert_nonref_to_tensor(
          lower_upper, dtype_hint=tf.float32, name='lower_upper')
      self._permutation = tensor_util.convert_nonref_to_tensor(
          permutation, dtype_hint=tf.int32, name='permutation')
      super(ScaleMatvecLU, self).__init__(
          dtype=self._lower_upper.dtype,
          is_constant_jacobian=True,
          forward_min_event_ndims=1,
          validate_args=validate_args,
          parameters=parameters,
          name=name)

  @classmethod
  def _parameter_properties(cls, dtype):
    # pylint: disable=g-long-lambda
    return dict(
        lower_upper=parameter_properties.ParameterProperties(
            event_ndims=2,
            shape_fn=lambda sample_shape: ps.concat(
                [sample_shape, sample_shape[-1:]], axis=0),
        ),
        permutation=parameter_properties.ParameterProperties(
            event_ndims=1,
            default_constraining_bijector_fn=parameter_properties
            .BIJECTOR_NOT_IMPLEMENTED))
    # pylint: enable=g-long-lambda

  @property
  def lower_upper(self):
    return self._lower_upper

  @property
  def permutation(self):
    return self._permutation

  def _broadcast_params(self):
    lower_upper = tf.convert_to_tensor(self.lower_upper)
    perm = tf.convert_to_tensor(self.permutation)
    shape = ps.broadcast_shape(ps.shape(lower_upper)[:-1],
                               ps.shape(perm))
    lower_upper = tf.broadcast_to(
        lower_upper, ps.concat([shape, shape[-1:]], 0))
    perm = tf.broadcast_to(perm, shape)
    return lower_upper, perm

  def _forward(self, x):
    lu, perm = self._broadcast_params()
    w = lu_reconstruct(lower_upper=lu,
                       perm=perm,
                       validate_args=self.validate_args)
    return tf.linalg.matvec(w, x)

  def _inverse(self, y):
    lu, perm = self._broadcast_params()
    return lu_solve(
        lower_upper=lu,
        perm=perm,
        rhs=y[..., tf.newaxis],
        validate_args=self.validate_args)[..., 0]

  def _forward_log_det_jacobian(self, unused_x):
    return tf.reduce_sum(
        tf.math.log(tf.abs(tf.linalg.diag_part(self.lower_upper))),
        axis=-1)

  def _parameter_control_dependencies(self, is_init):
    if not self.validate_args:
      return []

    lu, perm = None, None
    assertions = []
    if (is_init != tensor_util.is_ref(self.lower_upper) or
        is_init != tensor_util.is_ref(self.permutation)):
      lu, perm = self._broadcast_params()
      assertions.extend(lu_reconstruct_assertions(
          lu, perm, self.validate_args))

    if is_init != tensor_util.is_ref(self.lower_upper):
      lu = tf.convert_to_tensor(self.lower_upper) if lu is None else lu
      assertions.append(assert_util.assert_none_equal(
          tf.linalg.diag_part(lu), tf.zeros([], dtype=lu.dtype),
          message='Invertible `lower_upper` must have nonzero diagonal.'))

    return assertions


class MatvecLU(ScaleMatvecLU):
  """Matrix-vector multiply using LU decomposition.

  This bijector is identical to the 'Convolution1x1' used in Glow
  [(Kingma and Dhariwal, 2018)[1].

  #### Examples

  Here's an example of initialization via random weights matrix:

  ```python
  def trainable_lu_factorization(
      event_size, batch_shape=(), seed=None, dtype=tf.float32, name=None):
    with tf.name_scope(name or 'trainable_lu_factorization'):
      event_size = tf.convert_to_tensor(
          event_size, dtype_hint=tf.int32, name='event_size')
      batch_shape = tf.convert_to_tensor(
          batch_shape, dtype_hint=event_size.dtype, name='batch_shape')
      random_matrix = tf.random.uniform(
          shape=tf.concat([batch_shape, [event_size, event_size]], axis=0),
          dtype=dtype,
          seed=seed)
      random_orthonormal = tf.linalg.qr(random_matrix)[0]
      lower_upper, permutation = tf.linalg.lu(random_orthonormal)
      lower_upper = tf.Variable(
          initial_value=lower_upper,
          trainable=True,
          name='lower_upper')
      # Initialize a non-trainable variable for the permutation indices so
      # that its value isn't re-sampled from run-to-run.
      permutation = tf.Variable(
          initial_value=permutation,
          trainable=False,
          name='permutation')
      return lower_upper, permutation

  channels = 3
  conv1x1 = tfb.MatvecLU(*trainable_lu_factorization(channels),
                         validate_args=True)
  x = tf.random.uniform(shape=[2, 28, 28, channels])
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

  @deprecation.deprecated(
      '2020-01-01',
      '`MatvecLU` has been deprecated and renamed `ScaleMatvecLU`; please use '
      'that symbol instead.')
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
        Default value: `None` (i.e., 'MatvecLU').

    Raises:
      ValueError: If both/neither `channels` and `lower_upper`/`permutation` are
        specified.
    """
    super(MatvecLU, self).__init__(
        lower_upper, permutation, validate_args=False, name=name or 'MatvecLU')
