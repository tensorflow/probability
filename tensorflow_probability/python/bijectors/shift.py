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
"""Shift bijector."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v2 as tf

from tensorflow_probability.python.bijectors import bijector
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import tensor_util


__all__ = [
    'Shift',
]


class Shift(bijector.Bijector):
  """Compute `Y = g(X; shift) = X + shift`.

  where `shift` is a numeric `Tensor`.

  Example Use:

  ```python
  linalg = tf.linalg

  shift = Shift([-1., 0., 1], scale)
  x = [1., 2, 3]
  # `forward` is equivalent to:
  # y = x + shift
  y = affine.forward(x)  # [0., 2., 4.]
  ```

  """

  def __init__(self,
               shift,
               name='affine_linear_operator'):
    """Instantiates the `Shift` bijector.

    Args:
      shift: Floating-point `Tensor`.
      name: Python `str` name given to ops managed by this object.
    """
    with tf.name_scope(name) as name:
      dtype = dtype_util.common_dtype([shift], dtype_hint=tf.float32)
      self._shift = tensor_util.convert_nonref_to_tensor(
          shift, dtype=dtype, name='shift')
      super(Shift, self).__init__(
          forward_min_event_ndims=1,
          is_constant_jacobian=True,
          dtype=dtype,
          validate_args=False,
          name=name)

  @property
  def shift(self):
    """The `shift` `Tensor` in `Y = X + shift`."""
    return self._shift

  def _forward(self, x):
    return x + self.shift

  def _inverse(self, y):
    return y - self.shift

  def _forward_log_det_jacobian(self, x):
    # is_constant_jacobian = True for this bijector, hence the
    # `log_det_jacobian` need only be specified for a single input, as this will
    # be tiled to match `event_ndims`.
    return tf.constant(0., dtype=dtype_util.base_dtype(x.dtype))
