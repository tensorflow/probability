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
"""Discrete Cosine Transform bijector."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow_probability.python.bijectors import bijector


__all__ = [
    'DiscreteCosineTransform',
]


class DiscreteCosineTransform(bijector.Bijector):
  """Compute `Y = g(X) = DCT(X)`, where DCT type is indicated by the `type` arg.

  The [discrete cosine transform](
  https://en.wikipedia.org/wiki/Discrete_cosine_transform) efficiently applies
  a unitary DCT operator. This can be useful for mixing and decorrelating across
  the innermost event dimension.

  The inverse `X = g^{-1}(Y) = IDCT(Y)`, where IDCT is DCT-III for type==2.

  This bijector can be interleaved with Affine bijectors to build a cascade of
  structured efficient linear layers as in [1].

  Note that the operator applied is orthonormal (i.e. `norm='ortho'`).

  #### References

  [1]: Moczulski M, Denil M, Appleyard J, de Freitas N. ACDC: A structured
       efficient linear layer. In _International Conference on Learning
       Representations_, 2016. https://arxiv.org/abs/1511.05946
  """

  def __init__(self, dct_type=2, validate_args=False, name='dct'):
    """Instantiates the `DiscreteCosineTransform` bijector.

    Args:
      dct_type: Python `int`, the DCT type performed by the forward
        transformation. Currently, only 2 and 3 are supported.
      validate_args: Python `bool` indicating whether arguments should be
        checked for correctness.
      name: Python `str` name given to ops managed by this object.
    """
    # TODO(b/115910664): Support other DCT types.
    if dct_type not in (2, 3):
      raise NotImplementedError('`type` must be one of 2 or 3')
    self._dct_type = dct_type
    super(DiscreteCosineTransform, self).__init__(
        forward_min_event_ndims=1,
        inverse_min_event_ndims=1,
        is_constant_jacobian=True,
        validate_args=validate_args,
        name=name)

  def _forward(self, x):
    return tf.spectral.dct(x, type=self._dct_type, norm='ortho')

  def _inverse(self, y):
    return tf.spectral.idct(y, type=self._dct_type, norm='ortho')

  def _inverse_log_det_jacobian(self, y):
    return tf.constant(0., dtype=y.dtype.base_dtype)

  def _forward_log_det_jacobian(self, x):
    return tf.constant(0., dtype=x.dtype.base_dtype)
