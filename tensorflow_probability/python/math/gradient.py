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
"""Functions for computing gradients."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf


__all__ = [
    'value_and_gradient',
]


def value_and_gradient(f, xs, watch_accessed_variables=True, name=None):
  """Computes `f(*xs)` and its gradients wrt to `*xs`.

  Args:
    f: Python `callable` to be differentiated. If `f` returns a scalar, this
      scalar will be differentiated. If `f` returns a tensor or list of tensors,
      by default a scalar will be computed by adding all their values to produce
      a single scalar. If desired, the tensors can be elementwise multiplied by
      the tensors passed as the `dy` keyword argument to the returned gradient
      function.
    xs: Python list of parameters of f for which to differentiate. (Can also
      be single `Tensor`.)
    watch_accessed_variables: Boolean controlling whether the tape will
      automatically `watch` any (trainable) variables accessed while the tape is
      active. Defaults to True meaning gradients can be requested from any
      result computed in the tape derived from reading a trainable `Variable`.
      If False users must explicitly `watch` any `Variable`s they want to
      request gradients from.
      Default value: `True`.
    name: Python `str` name prefixed to ops created by this function.
      Default value: `None` (i.e., `'value_and_gradient'`).

  Returns:
    y: `y = f(*xs)`.
    dydx: Gradient of `y` wrt each of `xs`.
  """
  with tf.name_scope(name, 'value_and_gradient', [xs]):
    is_xs_list_like = isinstance(xs, (tuple, list))
    if not is_xs_list_like:
      xs = [xs]
    xs = [tf.convert_to_tensor(value=x, name='x{}'.format(i))
          for i, x in enumerate(xs)]
    with tf.GradientTape(
        persistent=len(xs) > 1,
        watch_accessed_variables=watch_accessed_variables) as tape:
      for x in xs:
        tape.watch(x)
      y = tf.convert_to_tensor(value=f(*xs), name='y')
    dydx = [tape.gradient(y, x) for x in xs]
    if not is_xs_list_like:
      dydx = dydx[0]
    return y, dydx
