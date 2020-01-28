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
"""Scale bijector."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v2 as tf

from tensorflow_probability.python.bijectors import bijector
from tensorflow_probability.python.internal import assert_util
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import tensor_util


__all__ = [
    'Scale',
]


class Scale(bijector.Bijector):
  """Compute `Y = g(X; scale) = scale * X`.

  Examples:

  ```python
  # Y = 2 * X
  b = Scale(scale=2.)
  ```

  """

  def __init__(self,
               scale,
               validate_args=False,
               name='scale'):
    """Instantiates the `Scale` bijector.

    This `Bijector`'s forward operation is:

    ```none
    Y = g(X) = scale * X
    ```

    Args:
      scale: Floating-point `Tensor`.
      validate_args: Python `bool` indicating whether arguments should be
        checked for correctness.
      name: Python `str` name given to ops managed by this object.
    """
    parameters = dict(locals())
    with tf.name_scope(name) as name:
      dtype = dtype_util.common_dtype([scale], dtype_hint=tf.float32)
      self._scale = tensor_util.convert_nonref_to_tensor(
          scale, dtype=dtype, name='scale')

      super(Scale, self).__init__(
          forward_min_event_ndims=0,
          is_constant_jacobian=True,
          validate_args=validate_args,
          dtype=dtype,
          parameters=parameters,
          name=name)

  @property
  def scale(self):
    """The `scale` term in `Y = scale * X`."""
    return self._scale

  def _is_increasing(self):
    return self.scale > 0

  def _forward(self, x):
    return x * self.scale

  def _inverse(self, y):
    return y / self.scale

  def _forward_log_det_jacobian(self, x):
    return tf.math.log(tf.abs(self.scale))

  def _parameter_control_dependencies(self, is_init):
    if not self.validate_args:
      return []
    assertions = []
    if (self.scale is not None and
        is_init != tensor_util.is_ref(self.scale)):
      assertions.append(
          assert_util.assert_none_equal(
              self.scale,
              tf.zeros([], dtype=self._scale.dtype),
              message='Argument `scale` must be non-zero.'))
    return assertions
