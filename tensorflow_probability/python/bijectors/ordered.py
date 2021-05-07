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
"""Ordered bijector."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v2 as tf

from tensorflow_probability.python.bijectors import bijector
from tensorflow_probability.python.internal import assert_util
from tensorflow_probability.python.internal import auto_composite_tensor
from tensorflow.python.util import deprecation  # pylint: disable=g-direct-tensorflow-import


__all__ = [
    'Ordered',
]


@auto_composite_tensor.auto_composite_tensor(
    omit_kwargs=('name',), module_name='tfp.bijectors')
class Ordered(bijector.AutoCompositeTensorBijector):
  """Maps a vector of increasing elements to an unconstrained vector.

  Both the domain and the codomain of the mapping is [-inf, inf], however,
  the input of the forward mapping must be strictly increasing.

  On the last dimension of the tensor, Ordered bijector performs:
  `y[0] = x[0]`
  `y[1:] = tf.log(x[1:] - x[:-1])`

  #### Example Use:

  ```python
  bijectors.Ordered().forward([2, 3, 4])
  # Result: [2., 0., 0.]

  bijectors.Ordered().inverse([0.06428002, -1.07774478, -0.71530371])
  # Result: [0.06428002, 0.40464228, 0.8936858]
  ```
  """

  @deprecation.deprecated(
      '2021-01-09', '`Ordered` bijector is deprecated; please use '
      '`tfb.Invert(tfb.Ascending())` instead.',
      warn_once=True)
  def __init__(self, validate_args=False, name='ordered'):
    parameters = dict(locals())
    with tf.name_scope(name) as name:
      super(Ordered, self).__init__(
          forward_min_event_ndims=1,
          validate_args=validate_args,
          parameters=parameters,
          name=name)

  @classmethod
  def _parameter_properties(cls, dtype):
    return dict()

  def _forward(self, x):
    with tf.control_dependencies(self._assertions(x)):
      y0 = x[..., :1]
      yk = tf.math.log(x[..., 1:] - x[..., :-1])
      y = tf.concat([y0, yk], axis=-1)
      return y

  def _inverse(self, y):
    x0 = y[..., :1]
    xk = tf.exp(y[..., 1:])
    x = tf.concat([x0, xk], axis=-1)
    return tf.cumsum(x, axis=-1)

  def _inverse_log_det_jacobian(self, y):
    # The Jacobian of the inverse mapping is lower
    # triangular, with the diagonal elements being:
    # J[i,i] = 1 if i=1, and
    #          exp(y_i) if 1<i<=K
    # which gives the absolute Jacobian determinant:
    # |det(Jac)| = prod_{i=1}^{K} exp(y[i]).
    # (1) - Stan Modeling Language User's Guide and Reference Manual
    #       Version 2.17.0 session 35.2
    return tf.reduce_sum(y[..., 1:], axis=-1)

  def _forward_log_det_jacobian(self, x):
    with tf.control_dependencies(self._assertions(x)):
      return -tf.reduce_sum(tf.math.log(x[..., 1:] - x[..., :-1]), axis=-1)

  def _assertions(self, t):
    if not self.validate_args:
      return []
    return [assert_util.assert_positive(
        t[..., 1:] - t[..., :-1],
        message='Forward transformation input must be strictly increasing.')]
