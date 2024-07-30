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
"""Identity bijector."""

import tensorflow.compat.v2 as tf

from tensorflow_probability.python.bijectors import bijector

__all__ = [
    'Identity',
]


class _NoOpCache(dict):

  def __getitem__(self, _):
    return {}


class Identity(
    bijector.CoordinatewiseBijectorMixin,
    bijector.AutoCompositeTensorBijector):
  """Compute Y = g(X) = X.

    Example Use:

    ```python
    # Create the Y=g(X)=X transform which is intended for Tensors with 1 batch
    # ndim and 1 event ndim (i.e., vector of vectors).
    identity = Identity()
    x = [[1., 2],
         [3, 4]]
    x == identity.forward(x) == identity.inverse(x)
    ```

  """

  def __init__(self, validate_args=False, name='identity'):
    parameters = dict(locals())
    with tf.name_scope(name) as name:
      super(Identity, self).__init__(
          forward_min_event_ndims=0,
          is_constant_jacobian=True,
          validate_args=validate_args,
          parameters=parameters,
          name=name)
    # Override superclass private fields to eliminate caching, avoiding a memory
    # leak caused by the `y is x` characteristic of this bijector.
    self._from_x = self._from_y = _NoOpCache()

  @classmethod
  def _is_increasing(cls):
    return True

  @classmethod
  def _parameter_properties(cls, dtype):
    return dict()

  @property
  def _is_permutation(self):
    return True

  def _forward(self, x):
    return x

  def _inverse(self, y):
    return y

  def _inverse_log_det_jacobian(self, y):
    return tf.constant(0., dtype=y.dtype)

  def _forward_log_det_jacobian(self, x):
    return tf.constant(0., dtype=x.dtype)
