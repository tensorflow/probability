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
"""Exp bijector."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v2 as tf

from tensorflow_probability.python.bijectors import invert
from tensorflow_probability.python.bijectors import power_transform


__all__ = [
    'Exp',
    'Log',
]


class Exp(power_transform.PowerTransform):
  """Compute `Y = g(X) = exp(X)`.

    Example Use:

    ```python
    # Create the Y=g(X)=exp(X) transform which works only on Tensors with 1
    # batch ndim 2.
    exp = Exp()
    x = [[[1., 2],
           [3, 4]],
          [[5, 6],
           [7, 8]]]
    exp(x) == exp.forward(x)
    log(x) == exp.inverse(x)
    ```

    Note: the exp(.) is applied element-wise but the Jacobian is a reduction
    over the event space.
  """

  def __init__(self,
               validate_args=False,
               name='exp'):
    """Instantiates the `Exp` bijector.

    Args:
      validate_args: Python `bool` indicating whether arguments should be
        checked for correctness.
      name: Python `str` name given to ops managed by this object.
    """
    # forward_min_event_ndims = 0.
    # No forward_min_event_ndims specified as this is done in PowerTransform.
    with tf.name_scope(name) as name:
      super(Exp, self).__init__(
          validate_args=validate_args,
          name=name)


class Log(invert.Invert):
  """Compute `Y = log(X)`. This is `Invert(Exp())`."""

  def __init__(self, validate_args=False, name='log'):
    with tf.name_scope(name) as name:
      bijector = Exp(validate_args=validate_args)
      super(Log, self).__init__(
          bijector=bijector, validate_args=validate_args, name=name)
