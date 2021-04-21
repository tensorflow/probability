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
"""Ascending bijector."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v2 as tf

from tensorflow_probability.python.bijectors import bijector
from tensorflow_probability.python.internal import assert_util
from tensorflow_probability.python.internal import auto_composite_tensor


__all__ = [
    'Ascending',
]


@auto_composite_tensor.auto_composite_tensor(omit_kwargs=('name',))
class Ascending(bijector.AutoCompositeTensorBijector):
  """Maps unconstrained R^n to R^n in ascending order.

  Both the domain and the codomain of the mapping is `[-inf, inf]^n`, however,
  the input of the inverse mapping must be strictly increasing.

  On the last dimension of the tensor, the Ascending bijector performs:
  `y = tf.cumsum([x[0], tf.exp(x[1]), tf.exp(x[2]), ..., tf.exp(x[-1])])`

  #### Example Use:

  ```python
  bijectors.Ascending().inverse([2, 3, 4])
  # Result: [2., 0., 0.]

  bijectors.Ascending().forward([0.06428002, -1.07774478, -0.71530371])
  # Result: [0.06428002, 0.40464228, 0.8936858]
  ```
  """

  _type_spec_id = 366918634

  def __init__(self, validate_args=False, name='ascending'):
    parameters = dict(locals())
    with tf.name_scope(name) as name:
      super(Ascending, self).__init__(
          forward_min_event_ndims=1,
          validate_args=validate_args,
          parameters=parameters,
          name=name)

  @classmethod
  def _parameter_properties(cls, dtype):
    return dict()

  def _forward(self, x):
    y0 = x[..., :1]
    yk = tf.exp(x[..., 1:])
    y = tf.concat([y0, yk], axis=-1)
    return tf.cumsum(y, axis=-1)

  def _inverse(self, y):
    with tf.control_dependencies(self._assertions(y)):
      x0 = y[..., :1]
      xk = tf.math.log(y[..., 1:] - y[..., :-1])
      x = tf.concat([x0, xk], axis=-1)
      return x

  def _forward_log_det_jacobian(self, x):
    # The Jacobian of the forward mapping is lower
    # triangular, with the diagonal elements being:
    # J[i,i] = 1 if i=1, and
    #          exp(x_i) if 1<i<=K
    # which gives the absolute Jacobian determinant:
    # |det(Jac)| = prod_{i=1}^{K} exp(x[i]).
    # (1) - Stan Modeling Language User's Guide and Reference Manual
    #       Version 2.17.0 session 35.2
    return tf.reduce_sum(x[..., 1:], axis=-1)

  def _inverse_log_det_jacobian(self, y):
    with tf.control_dependencies(self._assertions(y)):
      return -tf.reduce_sum(tf.math.log(y[..., 1:] - y[..., :-1]), axis=-1)

  def _assertions(self, t):
    if not self.validate_args:
      return []
    return [assert_util.assert_greater(
        t[..., 1:], t[..., :-1],
        message='Inverse transformation input must be strictly increasing.')]
