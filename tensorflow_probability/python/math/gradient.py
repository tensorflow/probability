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


def value_and_gradient(f, xs, use_gradient_tape=False, name=None):
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
    use_gradient_tape: Python `bool` indicating that `tf.GradientTape`
      should be used regardless of `tf.executing_eagerly()` status.
      Default value: `False`.
    name: Python `str` name prefixed to ops created by this function.
      Default value: `None` (i.e., `'value_and_gradient'`).

  Returns:
    y: `y = f(*xs)`.
    dydx: Gradient of `y` wrt each of `xs`.
  """
  with tf.compat.v1.name_scope(name, 'value_and_gradient', [xs]):
    is_xs_list_like = isinstance(xs, (tuple, list))
    if not is_xs_list_like:
      xs = [xs]
    xs = [tf.convert_to_tensor(value=x, name='x{}'.format(i))
          for i, x in enumerate(xs)]
    if tf.executing_eagerly() or use_gradient_tape:
      with tf.GradientTape(watch_accessed_variables=False) as tape:
        for x in xs:
          tape.watch(x)
        y = f(*xs)
      dydx = tape.gradient(y, xs)
    else:
      y = f(*xs)
      dydx = tf.gradients(ys=y, xs=xs)
    if not is_xs_list_like:
      dydx = dydx[0]
    return y, dydx
