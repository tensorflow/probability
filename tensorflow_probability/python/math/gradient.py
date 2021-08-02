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

from tensorflow_probability.python.internal import tensor_util
from tensorflow_probability.python.internal.backend.numpy import tf_inspect


__all__ = [
    'value_and_gradient',
]


def value_and_gradient(f,
                       *args,
                       output_gradients=None,
                       use_gradient_tape=False,
                       auto_unpack_single_arg=True,
                       has_aux=False,
                       name=None,
                       **kwargs):
  """Computes `f(*args, **kwargs)` and its gradients wrt to `args`, `kwargs`.

  The function `f` is invoked according to one of the following rules:

  1. If `f` is a function of no arguments then it is called as `f()`.

  2. If `len(args) == 1`, `len(kwargs) == 0`, `auto_unpack_single_arg == True`
     and `isinstance(args[0], (list, tuple))` then `args` is presumed to be a
     packed sequence of args, i.e., the function is called as `f(*args[0])`.

  3. Otherwise, the function is called as `f(*args, **kwargs)`.

  Regardless of how `f` is called, gradients are computed with respect to `args`
  and `kwargs`.

  #### Examples

  ```python
  tfd = tfp.distributions
  tfm = tfp.math

  # Case 1: argless `f`.
  x = tf.constant(2.)
  tfm.value_and_gradient(lambda: tf.math.log(x), x)
  # ==> [log(2.), 0.5]

  # Case 2: packed arguments.
  tfm.value_and_gradient(lambda x, y: x * tf.math.log(y), [2., 3.])
  # ==> [2. * np.log(3.), (np.log(3.), 2. / 3)]

  # Case 3: default.
  tfm.value_and_gradient(tf.math.log, [1., 2., 3.],
                         auto_unpack_single_arg=False)
  # ==> [(log(1.), log(2.), log(3.)), (1., 0.5, 0.333)]
  ```

  Args:
    f: Python `callable` to be differentiated. If `f` returns a scalar, this
      scalar will be differentiated. If `f` returns a tensor or list of tensors,
      the gradient will be the sum of the gradients of each part. If desired the
      sum can be weighted by `output_gradients` (see below).
    *args: Arguments as in `f(*args, **kwargs)` and basis for gradient.
    output_gradients: A `Tensor` or structure of `Tensor`s the same size as the
      result `ys = f(*args, **kwargs)` and holding the gradients computed for
      each `y` in `ys`. This argument is forwarded to the underlying gradient
      implementation (i.e., either the `grad_ys` argument of `tf.gradients` or
      the `output_gradients` argument of `tf.GradientTape.gradient`).
      Default value: `None`.
    use_gradient_tape: Python `bool` indicating that `tf.GradientTape` should be
      used rather than `tf.gradient` and regardless of `tf.executing_eagerly()`.
      (It is only possible to use `tf.gradient` when `not use_gradient_tape and
      not tf.executing_eagerly()`.)
      Default value: `False`.
    auto_unpack_single_arg: Python `bool` which when `False` means the single
      arg case will not be interpreted as a list of arguments. (See case 2.)
      Default value: `True`.
    has_aux: Whether `f(*args, **kwargs)` actually returns two outputs, the
      first being `y` and the second being an auxiliary output that does not get
      gradients computed.
    name: Python `str` name prefixed to ops created by this function.
      Default value: `None` (i.e., `'value_and_gradient'`).
    **kwargs: Named arguments as in `f(*args, **kwargs)` and basis for gradient.

  Returns:
    If `has_aux` is `False`:
      y: `y = f(*args, **kwargs)`.
      dydx: Gradients of `y` with respect to each of `args` and `kwargs`.
    otherwise:
      A tuple `((y, aux), dydx)`, where `y, aux = f(*args, **kwargs)` and `dydx`
      are the gradients of `y` with respect to each of `args` and `kwargs`.
  """
  with tf.name_scope(name or 'value_and_gradient'):
    return _value_and_grad_impl(
        f,
        _gradient_new if tf.executing_eagerly() or use_gradient_tape else
        _gradient_old,
        *args,
        output_gradients=output_gradients,
        auto_unpack_single_arg=auto_unpack_single_arg,
        expand_tf_modules_as_trainable_vars=False,
        has_aux=has_aux,
        **kwargs)


def value_and_gradient_with_auto_expansion(f,
                                           *args,
                                           output_gradients=None,
                                           use_gradient_tape=False,
                                           auto_unpack_single_arg=True,
                                           has_aux=False,
                                           name=None,
                                           **kwargs):
  """Computes `f(*args, **kwargs)` and its gradients wrt to `args`, `kwargs`.

  The function `f` is invoked according to one of the following rules:

  1. If `f` is a function of no arguments then it is called as `f()`.

  2. If `len(args) == 1`, `len(kwargs) == 0`, `auto_unpack_single_arg == True`
     and `isinstance(args[0], (list, tuple))` then `args` is presumed to be a
     packed sequence of args, i.e., the function is called as `f(*args[0])`.

  3. Otherwise, the function is called as `f(*args, **kwargs)`.

  Regardless of how `f` is called, gradients are computed with respect to `args`
  and `kwargs`.

  #### Examples

  ```python
  tfd = tfp.distributions
  tfm = tfp.math

  # Case 1: argless `f`.
  x = tf.constant(2.)
  tfm.value_and_gradient(lambda: tf.math.log(x), x)
  # ==> [log(2.), 0.5]

  # Case 2: packed arguments.
  tfm.value_and_gradient(lambda x, y: x * tf.math.log(y), [2., 3.])
  # ==> [2. * np.log(3.), (np.log(3.), 2. / 3)]

  # Case 3: default.
  tfm.value_and_gradient(tf.math.log, [1., 2., 3.],
                         auto_unpack_single_arg=False)
  # ==> [(log(1.), log(2.), log(3.)), (1., 0.5, 0.333)]

  # The following examples demonstrate computing gradients wrt trainable
  # variables.
  q = tfd.Normal(tf.Variable(1.), tf.Variable(1., trainable=False))
  r = tfd.Normal(0., tf.Variable(1.))
  tfm.value_and_gradient(tfd.kl_divergence, q, r)
  # ==> 0.5, [[1.], [-1.]]

  # The following all produce the same numerical result as above.
  tfm.value_and_gradient(lambda: tfd.kl_divergence(q, r), q, r)
  tfm.value_and_gradient(lambda *_: tfd.kl_divergence(q, r), q, r)
  tfm.value_and_gradient(lambda **kw: tfd.kl_divergence(
      tfd.Normal(kw['loc_q'], 1), tfd.Normal(0, kw['scale_r'])),
      loc_q=1., scale_r=1.)

  ```

  Args:
    f: Python `callable` to be differentiated. If `f` returns a scalar, this
      scalar will be differentiated. If `f` returns a tensor or list of tensors,
      the gradient will be the sum of the gradients of each part. If desired the
      sum can be weighted by `output_gradients` (see below).
    *args: Arguments as in `f(*args, **kwargs)` and basis for gradient.
    output_gradients: A `Tensor` or structure of `Tensor`s the same size as the
      result `ys = f(*args, **kwargs)` and holding the gradients computed for
      each `y` in `ys`. This argument is forwarded to the underlying gradient
      implementation (i.e., either the `grad_ys` argument of `tf.gradients` or
      the `output_gradients` argument of `tf.GradientTape.gradient`).
      Default value: `None`.
    use_gradient_tape: Python `bool` indicating that `tf.GradientTape` should be
      used rather than `tf.gradient` and regardless of `tf.executing_eagerly()`.
      (It is only possible to use `tf.gradient` when `not use_gradient_tape and
      not tf.executing_eagerly()`.)
      Default value: `False`.
    auto_unpack_single_arg: Python `bool` which when `False` means the single
      arg case will not be interpreted as a list of arguments. (See case 2.)
      Default value: `True`.
    has_aux: Whether `f(*args, **kwargs)` actually returns two outputs, the
      first being `y` and the second being an auxiliary output that does not get
      gradients computed.
    name: Python `str` name prefixed to ops created by this function.
      Default value: `None` (i.e., `'value_and_gradient'`).
    **kwargs: Named arguments as in `f(*args, **kwargs)` and basis for gradient.

  Returns:
    If `has_aux` is `False`:
      y: `y = f(*args, **kwargs)`.
      dydx: Gradients of `y` with respect to each of `args` and `kwargs`.
    otherwise:
      A tuple `((y, aux), dydx)`, where `y, aux = f(*args, **kwargs)` and `dydx`
      are the gradients of `y` with respect to each of `args` and `kwargs`.
  """
  with tf.name_scope(name or 'value_and_gradient'):
    return _value_and_grad_impl(
        f,
        _gradient_new if tf.executing_eagerly() or use_gradient_tape else
        _gradient_old,
        *args,
        output_gradients=output_gradients,
        auto_unpack_single_arg=auto_unpack_single_arg,
        expand_tf_modules_as_trainable_vars=True,
        has_aux=has_aux,
        **kwargs)


def value_and_batch_jacobian(f,
                             *args,
                             auto_unpack_single_arg=True,
                             has_aux=False,
                             name=None,
                             **kwargs):
  """Computes `f(*args, **kwargs)` and batch Jacobian wrt to `args`, `kwargs`.

  Args:
    f: Python `callable`, returning a 2D `(batch, n)` shaped `Tensor`.
    *args: Arguments as in `f(*args, **kwargs)` and basis for Jacobian. Each
      element must be 2D `(batch, n)`-shaped argument `Tensor`(s). If multiple
      are provided, a tuple of jacobians are returned.
    auto_unpack_single_arg: Python `bool` which when `False` means the single
      arg case will not be interpreted as a list of arguments.
      Default value: `True`.
    has_aux: Whether `f(*args, **kwargs)` actually returns two outputs, the
      first being `y` and the second being an auxiliary output that does not get
      gradients computed.
    name: Python `str` name prefixed to ops created by this function.
      Default value: `None` (i.e., `'value_and_gradient'`).
    **kwargs: Named arguments as in `f(*args, **kwargs)` and basis for Jacobian.
      Each element must be 2D `(batch, n)`-shaped argument `Tensor`(s). If
      multiple are provided, a tuple of jacobians are returned.

  Returns:
    If `has_aux` is `False`:
      y: `y = f(*args, **kwargs)`.
      jacobian: A `(batch, n, n)` shaped `Tensor`, `dy/dx`, or a tuple thereof.
    otherwise:
      A tuple `((y, aux), jacobian)`, where `y, aux = f(*args, **kwargs)` and
      `jacobian` is a `(batch, n, n)` shaped `Tensor`, `dy/dx`, or a tuple
      thereof.
  """
  with tf.name_scope(name or 'value_and_batch_jacobian'):
    return _value_and_grad_impl(
        f,
        _jacobian,
        *args,
        output_gradients=None,
        auto_unpack_single_arg=auto_unpack_single_arg,
        expand_tf_modules_as_trainable_vars=False,
        has_aux=has_aux,
        **kwargs)


def batch_jacobian(f,
                   *args,
                   auto_unpack_single_arg=True,
                   has_aux=False,
                   name=None,
                   **kwargs):
  """Computes batch Jacobian of `f(*args, **kwargs)` wrt to `args`, `kwargs`.

  Args:
    f: Python `callable`, returning a 2D `(batch, n)` shaped `Tensor`.
    *args: Arguments as in `f(*args, **kwargs)` and basis for Jacobian. Each
      element must be 2D `(batch, n)`-shaped argument `Tensor`(s). If multiple
      are provided, a tuple of jacobians are returned.
    auto_unpack_single_arg: Python `bool` which when `False` means the single
      arg case will not be interpreted as a list of arguments.
      Default value: `True`.
    has_aux: Whether `f(*args, **kwargs)` actually returns two outputs, the
      first being `y` and the second being an auxiliary output that does not get
      gradients computed.
    name: Python `str` name prefixed to ops created by this function.
      Default value: `None` (i.e., `'value_and_gradient'`).
    **kwargs: Named arguments as in `f(*args, **kwargs)` and basis for Jacobian.
      Each element must be 2D `(batch, n)`-shaped argument `Tensor`(s). If
      multiple are provided, a tuple of jacobians are returned.

  Returns:
    If `has_aux` is `False`:
      jacobian: A `(batch, n, n)` shaped `Tensor`, `dy/dx`, or a tuple thereof.
    otherwise:
      jacobian: A `(batch, n, n)` shaped `Tensor`, `dy/dx`, or a tuple thereof.
      aux: The auxiliary output of the function `y, aux = f(*args, **kwargs)`.
  """
  res = value_and_batch_jacobian(
      f,
      *args,
      auto_unpack_single_arg=auto_unpack_single_arg,
      name=name,
      has_aux=has_aux,
      **kwargs)
  if has_aux:
    (_, aux), jacobian = res
    return jacobian, aux
  else:
    _, jacobian = res
    return jacobian


def _gradient_new(f, xs, grad_ys):
  with tf.GradientTape(watch_accessed_variables=False) as tape:
    for x in xs:
      tape.watch(x)
    y, aux = f()
  return y, tape.gradient(y, xs, output_gradients=grad_ys), aux


def _gradient_old(f, xs, grad_ys):
  assert not tf.executing_eagerly()
  y, aux = f()
  return y, tf.gradients(y, xs, grad_ys=grad_ys), aux


def _jacobian(f, xs, grad_ys):
  assert grad_ys is None
  with tf.GradientTape(persistent=True, watch_accessed_variables=False) as tape:
    for x in xs:
      tape.watch(x)
    y, aux = f()
  try:
    return y, tuple(tape.batch_jacobian(y, x) for x in xs), aux
  except ValueError:  # Fallback to for-loop jacobian.
    return y, tuple(tape.batch_jacobian(y, x, experimental_use_pfor=False)
                    for x in xs), aux


def _value_and_grad_impl(f, grad_fn, *args, output_gradients,
                         auto_unpack_single_arg,
                         expand_tf_modules_as_trainable_vars=False,
                         has_aux=False,
                         **kwargs):
  """Helper which generalizes gradient / Jacobian."""
  if not args and not kwargs:
    raise ValueError('Gradient is not defined unless at least one of `arg` or '
                     '`kwarg` is specified.')
  # The following is for backwards compatibility. In the one arg case with no
  # kwargs we can't tell which protocol to use if not for
  # `auto_unpack_single_arg`.  When `True` and when the sole arg is a tuple
  # or list then we unpack it as if it was the args, i.e., preserve the old
  # behavior.
  do_unpack = (auto_unpack_single_arg and len(args) == 1 and not(kwargs) and
               isinstance(args[0], (tuple, list)))
  if do_unpack:
    args = args[0]
  args, kwargs = _prepare_args(args, kwargs)
  if expand_tf_modules_as_trainable_vars:
    expand_args, expand_kwargs = tf.nest.map_structure(
        lambda x: x.trainable_variables if tensor_util.is_module(x) else x,
        [args, kwargs])
  else:
    expand_args, expand_kwargs = args, kwargs

  if not has_aux:
    real_f = f
    f = lambda *args, **kwargs: (real_f(*args, **kwargs)  # pylint: disable=g-long-lambda
                                 if _has_args(real_f) else real_f(), ())

  y, dydx, aux = grad_fn(lambda: f(*args, **kwargs) if _has_args(f) else f(),
                         tf.nest.flatten([expand_args, expand_kwargs]),
                         output_gradients)
  dydx_args, dydx_kwargs = tf.nest.pack_sequence_as(
      [expand_args, expand_kwargs], dydx)
  if len(args) == 1 and not do_unpack:
    dydx_args = dydx_args[0]

  if has_aux:
    res = ((y, aux),)
  else:
    res = (y,)

  if args:
    res += (dydx_args,)
  if kwargs:
    res += (dydx_kwargs,)

  return res


def _prepare_args(args, kwargs):
  """Returns structures like inputs with values as Tensors."""
  i = -1
  def c2t(x):
    nonlocal i
    # Don't use convert_nonref_to_tensor here. We want to have semantics like
    # tf.GradientTape which watches only trainable_variables. (Note: we also
    # don't want to cal c2t on non-trainable variables since these are already
    # watchable by GradientTape.)
    if tensor_util.is_module(x) or tensor_util.is_variable(x):
      return x
    i += 1
    return tf.convert_to_tensor(
        x, dtype_hint=tf.float32, name='x{}'.format(i))
  return tf.nest.map_structure(c2t, (args, kwargs))


def _has_args(fn):
  """Returns `True` if the function takes an argument."""
  argspec = tf_inspect.getfullargspec(fn)
  return (bool(argspec.args) or
          bool(argspec.kwonlyargs) or
          argspec.varargs is not None or
          argspec.varkw is not None)


JAX_MODE = False  # Rewritten by script.

if JAX_MODE:
  import jax  # pylint: disable=g-import-not-at-top
  import jax.numpy as np  # pylint: disable=g-import-not-at-top
  import numpy as onp  # pylint: disable=g-import-not-at-top

  def value_and_gradient(f,  # pylint: disable=function-redefined
                         *args,
                         output_gradients=None,
                         use_gradient_tape=False,  # pylint: disable=unused-argument
                         auto_unpack_single_arg=True,
                         has_aux=False,
                         name=None,  # pylint: disable=unused-argument
                         **kwargs):
    """Computes `f(*args)` and its gradients wrt to `*args`."""
    if kwargs:
      raise NotImplementedError('Jax version of `value_and_gradient` does '
                                'not support `kwargs`.')
    do_unpack = (auto_unpack_single_arg and len(args) == 1 and
                 isinstance(args[0], (tuple, list)))
    if do_unpack:
      args = args[0]
    args, _ = _prepare_args(args, {})
    if has_aux:
      y, f_vjp, aux = jax.vjp(f, *args, has_aux=True)
    else:
      y, f_vjp = jax.vjp(f, *args)
    if output_gradients is None:
      output_gradients = tf.nest.map_structure(np.ones_like, y)
    dydx = list(f_vjp(output_gradients))
    if len(args) == 1 and not do_unpack:
      dydx = dydx[0]
    if has_aux:
      return (y, aux), dydx
    else:
      return y, dydx

  def value_and_batch_jacobian(  # pylint: disable=function-redefined
      f,
      *args,
      auto_unpack_single_arg=True,
      has_aux=False,
      name=None,  # pylint: disable=unused-argument
      **kwargs):
    """JAX implementation of value_and_batch_jacobian."""
    if kwargs:
      raise NotImplementedError('Jax version of `value_and_batch_jacobian` '
                                'does not support `kwargs`.')
    do_unpack = (auto_unpack_single_arg and len(args) == 1 and
                 isinstance(args[0], (tuple, list)))
    if do_unpack:
      args = args[0]
    args, _ = _prepare_args(args, {})
    if has_aux:
      y, f_vjp, aux = jax.vjp(f, *args, has_aux=True)
    else:
      y, f_vjp = jax.vjp(f, *args)

    # Let `[B, E_1, ..., E_k]` be the shape of `y`, where the first dimension
    # is a batch dimension.  We construct a basis for the cotangent space
    # `[E_1, ..., E_k]`.
    size = onp.prod(y.shape[1:])
    basis = np.reshape(np.eye(size, dtype=y.dtype),
                       (1, size,) + y.shape[1:])  # `[1, size, E_1, ..., E_k]`
    basis = np.broadcast_to(
        basis, y.shape[:1] + basis.shape[1:])  # `[B, size, E_1, ..., E_k]`

    dydx = jax.vmap(f_vjp, in_axes=1, out_axes=1)(basis)
    dydx = [x.reshape(y.shape + x.shape[2:]) for x in dydx]
    if len(args) == 1 and not do_unpack:
      dydx = dydx[0]
    if has_aux:
      return (y, aux), dydx
    else:
      return y, dydx

  def batch_jacobian(  # pylint: disable=function-redefined
      f,
      *args,
      auto_unpack_single_arg=True,
      has_aux=False,
      name=None,
      **kwargs):  # pylint: disable=unused-argument
    """Computes the batch jacobian of `f(xs)` w.r.t. `xs`."""
    res = value_and_batch_jacobian(
        f,
        *args,
        auto_unpack_single_arg=auto_unpack_single_arg,
        has_aux=has_aux,
        **kwargs)
    if has_aux:
      (_, aux), jacobian = res
      return jacobian, aux
    else:
      _, jacobian = res
      return jacobian
