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
"""CategoricalToDiscrete bijector.

This bijector is hidden from public API for now because it is only valid for
categorical distribution.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v2 as tf

from tensorflow_probability.python.bijectors import bijector
from tensorflow_probability.python.internal import assert_util
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import parameter_properties
from tensorflow_probability.python.internal import tensor_util
from tensorflow_probability.python.internal import tensorshape_util

__all__ = [
    'CategoricalToDiscrete',
]


class CategoricalToDiscrete(bijector.AutoCompositeTensorBijector):
  """Bijector which computes `Y = g(X) = values[X]`.

  Example Usage:

  ```python
  bijector = CategoricalToDiscrete(map_values=[0.01, 0.1, 1., 10.])
  bijector.forward([1, 3, 2, 1, 0]) = [1., 10., 1., 0.1, 0.01]
  bijector.inverse([1., 10., 1., 0.1, 0.01]) = [1, 3, 2, 1, 0]
  ```

  """

  def __init__(self,
               map_values,
               validate_args=False,
               name='categorical_to_discrete'):
    """Instantiates `CategoricalToDiscrete` bijector.

    Args:
      map_values: 1D numerical tensor of discrete values to map to, sorted in
        strictly increasing order.
      validate_args: Python `bool` indicating whether arguments should be
        checked for correctness.
      name: Python `str` name given to ops managed by this object.
    """
    parameters = dict(locals())
    with tf.name_scope(name):
      dtype = dtype_util.common_dtype([map_values], tf.float32)
      self._map_values = tensor_util.convert_nonref_to_tensor(
          map_values, name='map_values', dtype=dtype)
      super(CategoricalToDiscrete, self).__init__(
          forward_min_event_ndims=0,
          is_constant_jacobian=True,
          validate_args=validate_args,
          parameters=parameters,
          name=name)

  @classmethod
  def _parameter_properties(cls, dtype):
    return dict(
        map_values=parameter_properties.ParameterProperties(event_ndims=1,))

  def _forward(self, x):
    map_values = tf.convert_to_tensor(self.map_values)
    if self.validate_args:
      with tf.control_dependencies([
          assert_util.assert_equal(
              (0 <= x) & (x < tf.size(map_values)),
              True,
              message='indices out of bound')
      ]):
        x = tf.identity(x)
    # If we want batch dims in self.map_values, we can (after broadcasting),
    # use:
    # tf.gather(self.map_values, x, batch_dims=-1, axis=-1)
    return tf.gather(map_values, indices=x)

  def _inverse(self, y):
    map_values = tf.convert_to_tensor(self.map_values)
    flat_y = tf.reshape(y, shape=[-1])
    # Search for the indices of map_values that are closest to flat_y.
    # Since map_values is strictly increasing, the closest is either the
    # first one that is strictly greater than flat_y, or the one before it.
    upper_candidates = tf.minimum(
        tf.size(map_values) - 1,
        tf.searchsorted(map_values, values=flat_y, side='right'))
    lower_candidates = tf.maximum(0, upper_candidates - 1)
    candidates = tf.stack([lower_candidates, upper_candidates], axis=-1)
    lower_cand_diff = tf.abs(flat_y - self._forward(lower_candidates))
    upper_cand_diff = tf.abs(flat_y - self._forward(upper_candidates))
    if self.validate_args:
      with tf.control_dependencies([
          assert_util.assert_near(
              tf.minimum(lower_cand_diff, upper_cand_diff),
              0,
              message='inverse value not found')
      ]):
        candidates = tf.identity(candidates)
    candidate_selector = tf.stack([
        tf.range(tf.size(flat_y), dtype=tf.int32),
        tf.argmin([lower_cand_diff, upper_cand_diff],
                  output_type=tf.int32)], axis=-1)
    return tf.reshape(
        tf.gather_nd(candidates, candidate_selector), shape=y.shape)

  def _inverse_log_det_jacobian(self, y):
    return tf.constant(0., dtype=y.dtype)

  @property
  def map_values(self):
    return self._map_values

  def _parameter_control_dependencies(self, is_init):
    return _maybe_check_valid_map_values(self.map_values, self.validate_args)


def _maybe_check_valid_map_values(map_values, validate_args):
  """Validate `map_values` if `validate_args`==True."""
  assertions = []

  message = 'Rank of map_values must be 1.'
  if tensorshape_util.rank(map_values.shape) is not None:
    if tensorshape_util.rank(map_values.shape) != 1:
      raise ValueError(message)
  elif validate_args:
    assertions.append(assert_util.assert_rank(map_values, 1, message=message))

  message = 'Size of map_values must be greater than 0.'
  if tensorshape_util.num_elements(map_values.shape) is not None:
    if tensorshape_util.num_elements(map_values.shape) == 0:
      raise ValueError(message)
  elif validate_args:
    assertions.append(
        assert_util.assert_greater(tf.size(map_values), 0, message=message))

  if validate_args:
    assertions.append(
        assert_util.assert_equal(
            tf.math.is_strictly_increasing(map_values),
            True,
            message='map_values is not strictly increasing.'))

  return assertions
