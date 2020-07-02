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
"""ScaleMatvecDiag bijector."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v2 as tf

from tensorflow_probability.python.bijectors import scale_matvec_linear_operator
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import tensor_util


__all__ = [
    'ScaleMatvecDiag',
]


class ScaleMatvecDiag(scale_matvec_linear_operator.ScaleMatvecLinearOperator):
  """Compute `Y = g(X; scale) = scale @ X`.

  In TF parlance, the `scale` term is logically equivalent to:

  ```python
  scale = tf.diag(scale_diag)
  ```

  The `scale` term is applied without materializing a full dense matrix.

  #### Examples

  ```python
  # Y = tf.diag(d1) @ X
  b = ScaleMatvecDiag(scale_diag=[-1., 2, 1])  # Implicitly 3x3.
  ```

  """

  def __init__(self,
               scale_diag,
               adjoint=False,
               validate_args=False,
               name='scale_matvec_diag',
               dtype=None):
    """Instantiates the `ScaleMatvecDiag` bijector.

    This `Bijector`'s forward operation is:

    ```none
    Y = g(X) = scale @ X
    ```

    where the `scale` term is logically equivalent to:

    ```python
    scale = tf.diag(scale_diag)
    ```

    Args:
      scale_diag: Floating-point `Tensor` representing the diagonal matrix.
        `scale_diag` has shape `[N1, N2, ...  k]`, which represents a k x k
        diagonal matrix.
      adjoint: Python `bool` indicating whether to use the `scale` matrix as
        specified or its adjoint.
        Default value: `False`.
      validate_args: Python `bool` indicating whether arguments should be
        checked for correctness.
      name: Python `str` name given to ops managed by this object.
      dtype: `tf.DType` to prefer when converting args to `Tensor`s. Else, we
        fall back to a common dtype inferred from the args, finally falling back
        to float32.

    Raises:
      ValueError: if `scale_diag` is not specified.
    """
    parameters = dict(locals())
    with tf.name_scope(name) as name:
      if dtype is None:
        dtype = dtype_util.common_dtype([scale_diag, scale_diag])

      scale_diag = tensor_util.convert_nonref_to_tensor(
          scale_diag, name='scale_diag', dtype=dtype)

      super(ScaleMatvecDiag, self).__init__(
          scale=tf.linalg.LinearOperatorDiag(
              diag=scale_diag,
              is_non_singular=True),
          adjoint=adjoint,
          validate_args=validate_args,
          parameters=parameters,
          name=name)
