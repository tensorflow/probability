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


import tensorflow.compat.v2 as tf


__all__ = [
    'value_and_gradient',
]


def _prepare_args(xs):
  """Returns a `list` and a `bool` indicating whether args started list-like."""
  is_list_like = isinstance(xs, (tuple, list))
  if not is_list_like:
    xs = [xs]
  xs = [
      tf.convert_to_tensor(x, dtype_hint=tf.float32, name='x{}'.format(i))
      for i, x in enumerate(xs)
  ]
  return xs, is_list_like


def value_and_gradient(f,
                       xs,
                       output_gradients=None,
                       use_gradient_tape=False,
                       name=None):
  """Computes `f(*xs)` and its gradients wrt to `*xs`.

  Args:
    f: Python `callable` to be differentiated. If `f` returns a scalar, this
      scalar will be differentiated. If `f` returns a tensor or list of tensors,
      by default a scalar will be computed by adding all their values to produce
      a single scalar. If desired, the tensors can be elementwise multiplied by
      the tensors passed as the `dy` keyword argument to the returned gradient
      function.
    xs: Python list of parameters of `f` for which to differentiate. (Can also
      be single `Tensor`.)
    output_gradients: A `Tensor` or list of `Tensor`s the same size as the
      result `ys = f(*xs)` and holding the gradients computed for each `y` in
      `ys`. This argument is forwarded to the underlying gradient implementation
      (i.e., either the `grad_ys` argument of `tf.gradients` or the
      `output_gradients` argument of `tf.GradientTape.gradient`).
    use_gradient_tape: Python `bool` indicating that `tf.GradientTape` should be
      used regardless of `tf.executing_eagerly()` status.
      Default value: `False`.
    name: Python `str` name prefixed to ops created by this function.
      Default value: `None` (i.e., `'value_and_gradient'`).

  Returns:
    y: `y = f(*xs)`.
    dydx: Gradient of `y` wrt each of `xs`.
  """
  with tf.name_scope(name or 'value_and_gradient'):
    xs, is_xs_list_like = _prepare_args(xs)
    if tf.executing_eagerly() or use_gradient_tape:
      with tf.GradientTape(watch_accessed_variables=False) as tape:
        for x in xs:
          tape.watch(x)
        y = f(*xs)
      dydx = tape.gradient(y, xs, output_gradients=output_gradients)
    else:
      y = f(*xs)
      dydx = tf.gradients(ys=y, xs=xs, grad_ys=output_gradients)
    if not is_xs_list_like:
      dydx = dydx[0]
    return y, dydx


def value_and_batch_jacobian(f, xs):
  """Computes the value and batch jacobian of `f(arg)` w.r.t. `arg`.

  Args:
    f: Python callable, returning a 2D `(batch, n)` shaped `Tensor`.
    xs: 2D `(batch, n)`-shaped argument `Tensor`(s). If multiple are provided,
      a tuple of jacobians are returned.

  Returns:
    value: The result of `f(xs)`.
    jacobian: A `(batch, n, n)` shaped `Tensor`, `d f(xs) / d xs`, or a tuple
      thereof.
  """
  xs, is_xs_list_like = _prepare_args(xs)
  with tf.GradientTape(persistent=True) as tape:
    tape.watch(xs)
    result = f(*xs)
  try:
    jacobian = tuple(tape.batch_jacobian(result, x) for x in xs)
  except ValueError:  # Fallback to for-loop jacobian.
    jacobian = tuple(
        tape.batch_jacobian(result, x, experimental_use_pfor=False) for x in xs)
  if not is_xs_list_like:
    jacobian = jacobian[0]
  return result, jacobian


def batch_jacobian(f, xs):
  """Computes the batch jacobian of `f(xs)` w.r.t. `xs`.

  Args:
    f: Python callable, returning a 2D `(batch, n)` shaped `Tensor`.
    xs: 2D `(batch, n)`-shaped argument `Tensor`(s). If multiple are provided,
      a tuple of jacobians are returned.

  Returns:
    jacobian: A `(batch, n, n)` shaped `Tensor`, `d f(xs) / d xs`, or a tuple
      thereof.
  """
  return value_and_batch_jacobian(f, xs)[1]


JAX_MODE = False  # Rewritten by script.

if JAX_MODE:
  import jax  # pylint: disable=g-import-not-at-top
  import jax.numpy as np  # pylint: disable=g-import-not-at-top

  def value_and_gradient(f,  # pylint: disable=function-redefined
                         xs,
                         output_gradients=None,
                         use_gradient_tape=False,  # pylint: disable=unused-argument
                         name=None):  # pylint: disable=unused-argument
    """Computes `f(*xs)` and its gradients wrt to `*xs`."""
    xs, is_xs_list_like = _prepare_args(xs)
    y, f_vjp = jax.vjp(f, *xs)
    if output_gradients is None:
      output_gradients = tf.nest.map_structure(np.ones_like, y)
    dydx = list(f_vjp(output_gradients))
    if not is_xs_list_like:
      dydx = dydx[0]
    return y, dydx

  def value_and_batch_jacobian(f, xs):  # pylint: disable=function-redefined,unused-argument
    raise NotImplementedError('f(xs) non-vmap, jacobian(f) vmap = incompatible')

  def batch_jacobian(f, xs):  # pylint: disable=function-redefined
    """Computes the batch jacobian of `f(xs)` w.r.t. `xs`."""
    xs, is_xs_list_like = _prepare_args(xs)
    jacobian = jax.vmap(jax.jacrev(f, argnums=tuple(range(len(xs)))))(*xs)
    if not is_xs_list_like:
      jacobian = jacobian[0]
    return jacobian
