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
"""Expm1 bijector."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v2 as tf
from tensorflow_probability.python.bijectors import bijector
from tensorflow_probability.python.bijectors import invert


__all__ = [
    'Log1p',
    'Expm1',
]


class Expm1(bijector.Bijector):
  """Compute `Y = g(X) = exp(X) - 1`.

    This `Bijector` is no different from Chain([AffineScalar(shift=-1), Exp()]).

    However, this makes use of the more numerically stable routines
    `tf.math.expm1` and `tf.log1p`.

    Example Use:

    ```python
    # Create the Y=g(X)=expm1(X) transform.
    expm1 = Expm1()
    x = [[[1., 2],
           [3, 4]],
          [[5, 6],
           [7, 8]]]
    expm1(x) == expm1.forward(x)
    log1p(x) == expm1.inverse(x)
    ```

    Note: the expm1(.) is applied element-wise but the Jacobian is a reduction
    over the event space.
  """

  def __init__(self,
               validate_args=False,
               name='expm1'):
    parameters = dict(locals())
    with tf.name_scope(name) as name:
      super(Expm1, self).__init__(
          forward_min_event_ndims=0,
          validate_args=validate_args,
          parameters=parameters,
          name=name)

  @classmethod
  def _is_increasing(cls):
    return True

  def _forward(self, x):
    """Returns the forward `Bijector` evaluation, i.e., X = g(Y)."""
    return tf.math.expm1(x)

  def _inverse(self, y):
    """Returns the inverse `Bijector` evaluation, i.e., Y = g^-1(X)."""
    return tf.math.log1p(y)

  def _inverse_log_det_jacobian(self, y):
    """Returns the (log o det o Jacobian o g^-1)(y)."""
    return -tf.math.log1p(y)

  def _forward_log_det_jacobian(self, x):
    """Returns the (log o det o Jacobian o g)(x)."""
    return tf.identity(x)


class Log1p(invert.Invert):
  """Compute `Y = log1p(X)`. This is `Invert(Expm1())`."""

  def __init__(self, validate_args=False, name='log1p'):
    parameters = dict(locals())
    with tf.name_scope(name) as name:
      bij = Expm1(validate_args=validate_args)
      super(Log1p, self).__init__(
          bijector=bij,
          validate_args=validate_args,
          parameters=parameters,
          name=name)
