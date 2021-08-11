# Copyright 2021 The TensorFlow Probability Authors.
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
"""Manually implemented special functions.

Normally you'd just use functions coming from the array library you're using,
but on some platforms (like TPU) the default implementations are insufficiently
precise for certain tasks when running under 32 bits (64 bit implementations are
typically okay).

This file provides manual implementations of some special functions.

You can either use these functions directly, or monkey-patch them in via
`patch_manual_special_functions`.
"""

import contextlib

import numpy as np
import tensorflow.compat.v2 as tf

from tensorflow_probability.python.internal import custom_gradient
from tensorflow_probability.python.internal import dtype_util

JAX_MODE = False

__all__ = [
    'exp_pade_4_4',
    'expm1_pade_4_4',
    'log1p_pade_4_4',
    'log_pade_4_4_newton',
    'patch_manual_special_functions',
    'reduce_logsumexp',
    'softplus',
]

_real_log = tf.math.log


def reduce_logsumexp(a, axis=None, keepdims=False, name='reduce_logsumexp'):
  """Like tf.math.reduce_logsumexp.

  This has no functional difference from the regular version, except that it's
  implemented inline here, allowing monkey-patching of the special functions it
  uses (e.g. exp).

  Args:
    a: A tensor.
    axis: Dimensions to reduce. If `None`, reduces all dimensions.
    keepdims: If `True`, retains the reduced dimensions with length 1.
    name: Name for the op.

  Returns:
    y: The reduced tensor.
  """
  with tf.name_scope(name):
    amax_with_dims = tf.math.reduce_max(a, axis=axis, keepdims=True)
    amax_with_dims = tf.where(
        tf.math.is_finite(amax_with_dims), amax_with_dims, 0)
    if keepdims:
      amax = amax_with_dims
    else:
      amax = tf.squeeze(amax_with_dims, axis)
    return amax + tf.math.log(
        tf.math.reduce_sum(
            tf.math.exp(a - amax_with_dims), axis=axis, keepdims=keepdims))


def softplus(x, name='softplus'):
  """Like tf.math.reduce_logsumexp.

  This has no functional difference from the regular version, except that it's
  implemented inline here, allowing monkey-patching of the special functions it
  uses (e.g. exp).

  Args:
    x: A Tensor.
    name: Name for the op.

  Returns:
    y: softplus(x)
  """
  with tf.name_scope(name):
    x = tf.convert_to_tensor(x, dtype_hint=tf.float32)
    return tf.math.log1p(tf.math.exp(-tf.math.abs(x))) + tf.math.maximum(x, 0)


def _horner(x, coeffs):
  """Horner's method to evaluate polynomials."""
  res = coeffs[0]
  for c in coeffs[1:]:
    res = c + x * res
  return res


def _exp_pade_4_4_fwd(x):  # pylint: disable=missing-function-docstring
  x = tf.convert_to_tensor(x, dtype_hint=tf.float32)
  raw_x = x
  dtype = dtype_util.as_numpy_dtype(x.dtype)
  inf = np.float32('inf').astype(dtype)

  log2e = np.log(2).astype(dtype)

  n = tf.math.floor(x / log2e)
  x = x - n * log2e

  coeffs_p = np.array([1 / 1680, 1 / 84, 3 / 28, 1 / 2, 1], dtype)
  coeffs_q = np.array([1 / 1680, -1 / 84, 3 / 28, -1 / 2, 1], dtype)
  res = _horner(x, coeffs_p) / _horner(x, coeffs_q)

  if JAX_MODE:
    import jax.numpy as jnp  # pylint: disable=g-import-not-at-top
    res = res * jnp.exp2(n)
  else:
    res = res * (2**n)

  res = tf.where(tf.equal(raw_x, -inf), tf.zeros_like(x), res)
  res = tf.where(tf.equal(raw_x, inf), inf, res)
  return res, res


def _exp_pade_4_4_bwd(y, dy):
  return y * dy


def _exp_pade_4_4_jvp(x, dx):
  y = _exp_pade_4_4_fwd(x[0])[0]
  return y, _exp_pade_4_4_bwd(y, dx[0])


@custom_gradient.custom_gradient(
    vjp_fwd=_exp_pade_4_4_fwd,
    vjp_bwd=_exp_pade_4_4_bwd,
    jvp_fn=_exp_pade_4_4_jvp,
)
def _exp_pade_4_4_impl(x):
  return _exp_pade_4_4_fwd(x)[0]


def exp_pade_4_4(x, name='exp_pade_4_4'):
  """exp using the Pade(4,4) approximant."""
  with tf.name_scope(name):
    return _exp_pade_4_4_impl(x)


def _log_pade_4_4_newton_fwd(x):
  x = tf.convert_to_tensor(x, dtype_hint=tf.float32)
  dtype = dtype_util.as_numpy_dtype(x.dtype)
  r = _real_log(x)
  res = r - 1. + x / exp_pade_4_4(r)
  res = tf.where(tf.equal(x, 0), -np.float32('inf').astype(dtype), res)
  res = tf.where(tf.equal(x, 1), tf.zeros_like(res), res)
  return res, x


def _log_pade_4_4_newton_bwd(x, dy):
  return dy / x


def _log_pade_4_4_newton_jvp(x, dx):
  return (_log_pade_4_4_newton_fwd(x[0])[0],
          _log_pade_4_4_newton_bwd(x[0], dx[0]))


@custom_gradient.custom_gradient(
    vjp_fwd=_log_pade_4_4_newton_fwd,
    vjp_bwd=_log_pade_4_4_newton_bwd,
    jvp_fn=_log_pade_4_4_newton_jvp,
)
def _log_pade_4_4_newton_impl(x):
  return _log_pade_4_4_newton_fwd(x)[0]


def log_pade_4_4_newton(x, name='log_pade_4_4_newton'):
  """log using the Pade(4,4) approximant + a Newton's method step."""
  with tf.name_scope(name):
    return _log_pade_4_4_newton_impl(x)


def _expm1_pade_4_4_fwd(x):  # pylint: disable=missing-function-docstring
  x = tf.convert_to_tensor(x, dtype_hint=tf.float32)
  one = tf.ones([], x.dtype)
  dtype = dtype_util.as_numpy_dtype(x.dtype)

  for_large_x = exp_pade_4_4(x) - one

  # The leading coefficient is zero for the numerator.
  coeffs_p = np.array([1 / 42, 0, 1, 0], dtype)
  coeffs_q = np.array([1 / 1680, -1 / 84, 3 / 28, -1 / 2, 1], dtype)
  for_small_x = _horner(x, coeffs_p) / _horner(x, coeffs_q)

  abs_x = tf.math.abs(x)
  exponent_is_small_thresh = 1.
  x_is_small = abs_x < exponent_is_small_thresh
  res = tf.where(x_is_small, for_small_x, for_large_x)
  return res, res


def _expm1_pade_4_4_bwd(y, dy):
  return dy * (y + 1)


def _expm1_pade_4_4_jvp(x, dx):
  y = _expm1_pade_4_4_fwd(x[0])[0]
  return y, _expm1_pade_4_4_bwd(y, dx[0])


@custom_gradient.custom_gradient(
    vjp_fwd=_expm1_pade_4_4_fwd,
    vjp_bwd=_expm1_pade_4_4_bwd,
    jvp_fn=_expm1_pade_4_4_jvp,
)
def _expm1_pade_4_4_impl(x):
  return _expm1_pade_4_4_fwd(x)[0]


def expm1_pade_4_4(x, name='expm1_pade_4_4'):
  """expm1 using the Pade(4,4) approximant."""
  with tf.name_scope(name):
    return _expm1_pade_4_4_impl(x)


def _log1p_pade_4_4_fwd(x):
  x = tf.convert_to_tensor(x, dtype_hint=tf.float32)
  dtype = dtype_util.as_numpy_dtype(x.dtype)
  coeffs_p = np.array([5 / 84, 13 / 21, 3 / 2, 1, 0], dtype)
  coeffs_q = np.array([1 / 70, 2 / 7, 9 / 7, 2, 1], dtype)
  for_large_x = log_pade_4_4_newton(1 + x)
  for_small_x = _horner(x, coeffs_p) / _horner(x, coeffs_q)
  x_is_small = x < 0.7
  return tf.where(x_is_small, for_small_x, for_large_x), x


def _log1p_pade_4_4_bwd(x, g):
  return g / (1 + x)


def _log1p_pade_4_4_jvp(x, g):
  return _log1p_pade_4_4_fwd(x[0])[0], _log1p_pade_4_4_bwd(x[0], g[0])


@custom_gradient.custom_gradient(
    vjp_fwd=_log1p_pade_4_4_fwd,
    vjp_bwd=_log1p_pade_4_4_bwd,
    jvp_fn=_log1p_pade_4_4_jvp,
)
def _log1p_pade_4_4_impl(x):
  return _log1p_pade_4_4_fwd(x)[0]


def log1p_pade_4_4(x, name='log1p_pade_4_4'):
  """log1p using the Pade(4,4) approximant."""
  with tf.name_scope(name):
    return _log1p_pade_4_4_impl(x)


@contextlib.contextmanager
def patch_manual_special_functions():
  """Patches in the manually implemented special functions.

  Normally you'd just use functions coming from the array library you're using,
  but on some platforms (like TPU) the default implementations are
  insufficiently precise for certain tasks when running under 32 bits (64 bit
  implementations are typically okay).

  This patches in manual implementations of those functions which are provide
  higher precision at the cost of speed. The list of affected functions is:

  - `exp`
  - `log`
  - `expm1`
  - `log1p`
  - `logsumexp` (aka `reduce_logsumexp`)
  - `softplus`

  Yields:
    Nothing.
  """
  if JAX_MODE:
    import jax  # pylint: disable=g-import-not-at-top

    old_expm1 = jax.numpy.expm1
    old_log1p = jax.numpy.log1p
    old_exp = jax.numpy.exp
    old_log = jax.numpy.log
    old_logsumexp = jax.scipy.special.logsumexp
    old_softplus = jax.nn.softplus
  else:
    old_expm1 = tf.math.expm1
    old_log1p = tf.math.log1p
    old_exp = tf.math.exp
    old_log = tf.math.log
    old_logsumexp = tf.math.reduce_logsumexp
    old_softplus = tf.math.softplus

  try:
    if JAX_MODE:
      jax.numpy.expm1 = expm1_pade_4_4
      jax.numpy.log1p = log1p_pade_4_4
      jax.numpy.exp = exp_pade_4_4
      jax.numpy.log = log_pade_4_4_newton
      jax.scipy.special.logsumexp = reduce_logsumexp
      jax.nn.softplus = softplus
    else:
      tf.math.expm1 = expm1_pade_4_4
      tf.math.log1p = log1p_pade_4_4
      tf.math.exp = exp_pade_4_4
      tf.exp = exp_pade_4_4
      tf.math.log = log_pade_4_4_newton
      tf.math.reduce_logsumexp = reduce_logsumexp
      tf.reduce_logsumexp = reduce_logsumexp
      tf.math.softplus = softplus
      tf.nn.softplus = softplus
    yield
  finally:
    if JAX_MODE:
      jax.numpy.expm1 = old_expm1
      jax.numpy.log1p = old_log1p
      jax.numpy.exp = old_exp
      jax.numpy.log = old_log
      jax.scipy.special.logsumexp = old_logsumexp
      jax.nn.softplus = old_softplus
    else:
      tf.math.expm1 = old_expm1
      tf.math.log1p = old_log1p
      tf.math.exp = old_exp
      tf.exp = old_exp
      tf.math.log = old_log
      tf.math.reduce_logsumexp = old_logsumexp
      tf.reduce_logsumexp = old_logsumexp
      tf.math.softplus = old_softplus
      tf.nn.softplus = old_softplus
