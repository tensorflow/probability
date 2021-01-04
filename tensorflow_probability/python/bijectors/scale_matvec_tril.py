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
"""ScaleMatvecTriL bijector."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v2 as tf
from tensorflow_probability.python.bijectors import fill_triangular as fill_triangular_bijector
from tensorflow_probability.python.bijectors import scale_matvec_linear_operator
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import parameter_properties
from tensorflow_probability.python.internal import prefer_static as ps
from tensorflow_probability.python.internal import tensor_util

__all__ = [
    'ScaleMatvecTriL',
]


class ScaleMatvecTriL(scale_matvec_linear_operator.ScaleMatvecLinearOperator):
  """Compute `Y = g(X; scale) = scale @ X`.

  The `scale` term is presumed lower-triangular and non-singular (ie, no zeros
  on the diagonal), which permits efficient determinant calculation (linear in
  matrix dimension, instead of cubic).

  #### Examples

  ```python
  # Y = scale_tril @ X
  b = ScaleMatvecTriL(scale_tril=[[1., 0.], [1., 1.])
  ```

  """

  def __init__(self,
               scale_tril,
               adjoint=False,
               validate_args=False,
               name='scale_matvec_tril',
               dtype=None):
    """Instantiates the `ScaleMatvecTriL` bijector.

    With a `Tensor` `scale_tril` argument, this bijector's forward operation is:

    ```none
    Y = g(X) = scale_tril @ X
    ```

    where the `scale_tril` term is a lower-triangular matrix compatible with
    `X`.

    Args:
      scale_tril: Floating-point `Tensor` representing the lower triangular
        matrix. `scale_tril` has shape `[N1, N2, ...  k, k]`, which represents a
        k x k lower triangular matrix.
        When `None` no `scale_tril` term is added to `scale`.
        The upper triangular elements above the diagonal are ignored.
      adjoint: Python `bool` indicating whether to use the `scale` matrix as
        specified or its adjoint. Note that lower-triangularity is taken into
        account first: the region above the diagonal of `scale_tril` is treated
        as zero (irrespective of the `adjoint` setting). A lower-triangular
        input with `adjoint=True` will behave like an upper triangular
        transform.
        Default value: `False`.
      validate_args: Python `bool` indicating whether arguments should be
        checked for correctness.
      name: Python `str` name given to ops managed by this object.
      dtype: `tf.DType` to prefer when converting args to `Tensor`s. Else, we
        fall back to a common dtype inferred from the args, finally falling back
        to float32.

    Raises:
      ValueError: if `scale_tril` is not specified.
    """
    parameters = dict(locals())
    with tf.name_scope(name) as name:
      if dtype is None:
        dtype = dtype_util.common_dtype([scale_tril])

      scale_tril = tensor_util.convert_nonref_to_tensor(
          scale_tril, name='scale_tril', dtype=dtype)

      super(ScaleMatvecTriL, self).__init__(
          scale=tf.linalg.LinearOperatorLowerTriangular(
              tril=scale_tril,
              is_non_singular=True),
          adjoint=adjoint,
          validate_args=validate_args,
          parameters=parameters,
          name=name)

  @classmethod
  def _parameter_properties(cls, dtype):
    # pylint: disable=g-long-lambda
    return dict(
        scale_tril=parameter_properties.ParameterProperties(
            event_ndims=2,
            shape_fn=lambda sample_shape: ps.concat(
                [sample_shape, sample_shape[-1:]], axis=0),
            default_constraining_bijector_fn=fill_triangular_bijector
            .FillTriangular))
    # pylint: enable=g-long-lambda

  @property
  def _composite_tensor_nonshape_params(self):
    """A tuple describing which parameters are non-shape-related tensors.

    Flattening in JAX involves many of the same considerations with regards to
    identifying tensor arguments for the purposes of CompositeTensor, except
    that shape-related items will be considered metadata.  This property
    identifies the keys of parameters that are expected to be tensors, except
    those that are shape-related.
    """
    return ('scale_tril',)
