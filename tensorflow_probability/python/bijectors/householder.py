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
"""Householder bijector."""

import tensorflow.compat.v2 as tf
from tensorflow_probability.python.bijectors import bijector
from tensorflow_probability.python.bijectors import scale_matvec_linear_operator as smlo
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import parameter_properties
from tensorflow_probability.python.internal import tensor_util

__all__ = [
    'Householder',
]


class Householder(bijector.AutoCompositeTensorBijector):
  """Compute the reflection of a vector across a hyperplane.

  The reflection of x across the hyperplane with unit normal vector `v`
  (the `reflection_axis`) is `x - 2 * <x, v> * v`.


  #### Examples

  ```python
  householder = tfp.bijectors.Householder(reflection_axis = [-0.8, 0.6])
  x = [1.0, 0.0]
  householder.forward(x)  # [-0.28, 0.96]
  ```

  **Note:** The norm of `reflection_axis` should be 1. If the norm
  is less than 1e-6, results may not be accurate.
  """

  def __init__(self,
               reflection_axis,
               validate_args=False,
               name='householder'):
    """Instantiates the `Householder` bijector.

    Args:
      reflection_axis: Vector normal to the reflection hyperplane.
      validate_args: Python `bool` indicating whether arguments should be
        checked for correctness.
      name: Python `str` name given to ops managed by this object.
    """
    parameters = dict(locals())
    with tf.name_scope(name) as name:
      dtype = dtype_util.common_dtype([reflection_axis], dtype_hint=tf.float32)
      self._reflection_axis = tensor_util.convert_nonref_to_tensor(
          reflection_axis, dtype=dtype, name='reflection_axis')
      super(Householder, self).__init__(
          forward_min_event_ndims=1,
          is_constant_jacobian=True,
          dtype=dtype,
          validate_args=validate_args,
          parameters=parameters,
          name=name)

  @classmethod
  def _parameter_properties(cls, dtype):
    return dict(
        reflection_axis=parameter_properties.ParameterProperties(event_ndims=1))

  @property
  def reflection_axis(self):
    return self._reflection_axis

  def _forward(self, x):
    scale = tf.linalg.LinearOperatorHouseholder(self.reflection_axis)
    reflection = smlo.ScaleMatvecLinearOperator(scale)
    return reflection.forward(x)

  def _inverse(self, y):
    return self._forward(y)

  def _forward_log_det_jacobian(self, x):
    return tf.zeros([], dtype=dtype_util.base_dtype(x.dtype))
