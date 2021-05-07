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
"""FillScaleTriL bijector."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v2 as tf
from tensorflow_probability.python.bijectors import chain
from tensorflow_probability.python.bijectors import fill_triangular
from tensorflow_probability.python.bijectors import shift
from tensorflow_probability.python.bijectors import softplus
from tensorflow_probability.python.bijectors import transform_diagonal
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import tensor_util


__all__ = [
    'FillScaleTriL',
]


# TODO(b/182603117): Enable AutoCompositeTensor once Chain subclasses it.
class FillScaleTriL(chain.Chain):
  """Transforms unconstrained vectors to TriL matrices with positive diagonal.

  This is implemented as a simple `tfb.Chain` of `tfb.FillTriangular`
  followed by `tfb.TransformDiagonal`, and provided mostly as a
  convenience. The default setup is somewhat opinionated, using a
  Softplus transformation followed by a small shift (`1e-5`) which
  attempts to avoid numerical issues from zeros on the diagonal.

  #### Examples

  ```python
  tfb = tfp.distributions.bijectors
  b = tfb.FillScaleTriL(
       diag_bijector=tfb.Exp(),
       diag_shift=None)
  b.forward(x=[0., 0., 0.])
  # Result: [[1., 0.],
  #          [0., 1.]]
  b.inverse(y=[[1., 0],
               [.5, 2]])
  # Result: [log(2), .5, log(1)]

  # Define a distribution over PSD matrices of shape `[3, 3]`,
  # with `1 + 2 + 3 = 6` degrees of freedom.
  dist = tfd.TransformedDistribution(
          tfd.Normal(tf.zeros(6), tf.ones(6)),
          tfb.Chain([tfb.CholeskyOuterProduct(), tfb.FillScaleTriL()]))

  # Using an identity transformation, FillScaleTriL is equivalent to
  # tfb.FillTriangular.
  b = tfb.FillScaleTriL(
       diag_bijector=tfb.Identity(),
       diag_shift=None)

  # For greater control over initialization, one can manually encode
  # pre- and post- shifts inside of `diag_bijector`.
  b = tfb.FillScaleTriL(
       diag_bijector=tfb.Chain([
         tfb.Shift(1e-3),
         tfb.Softplus(),
         tfb.Shift(0.5413)]),  # tfp.math.softplus_inverse(1.)
                               #  = log(expm1(1.)) = 0.5413
       diag_shift=None)
  ```

  """

  def __init__(self,
               diag_bijector=None,
               diag_shift=1e-5,
               validate_args=False,
               name='fill_scale_tril'):
    """Instantiates the `FillScaleTriL` bijector.

    Args:
      diag_bijector: `Bijector` instance, used to transform the output diagonal
        to be positive.
        Default value: `None` (i.e., `tfb.Softplus()`).
      diag_shift: Float value broadcastable and added to all diagonal entries
        after applying the `diag_bijector`. Setting a positive
        value forces the output diagonal entries to be positive, but
        prevents inverting the transformation for matrices with
        diagonal entries less than this value.
        Default value: `1e-5`.
      validate_args: Python `bool` indicating whether arguments should be
        checked for correctness.
        Default value: `False` (i.e., arguments are not validated).
      name: Python `str` name given to ops managed by this object.
        Default value: `fill_scale_tril`.
    """
    parameters = dict(locals())
    with tf.name_scope(name) as name:
      if diag_bijector is None:
        diag_bijector = softplus.Softplus(validate_args=validate_args)

      if diag_shift is not None:
        dtype = dtype_util.common_dtype([diag_bijector, diag_shift], tf.float32)
        diag_shift = tensor_util.convert_nonref_to_tensor(diag_shift,
                                                          name='diag_shift',
                                                          dtype=dtype)
        diag_bijector = chain.Chain([
            shift.Shift(shift=diag_shift),
            diag_bijector
        ])

      super(FillScaleTriL, self).__init__(
          [transform_diagonal.TransformDiagonal(diag_bijector=diag_bijector),
           fill_triangular.FillTriangular()],
          validate_args=validate_args,
          validate_event_size=False,
          parameters=parameters,
          name=name)
