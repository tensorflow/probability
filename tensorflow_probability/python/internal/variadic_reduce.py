# Copyright 2020 The TensorFlow Probability Authors.
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
"""Helper for generic variadic reductions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow.compat.v1 as tf1
import tensorflow.compat.v2 as tf

from tensorflow_probability.python.internal import custom_gradient as tfp_custom_gradient
from tensorflow_probability.python.internal import implementation_selection
from tensorflow_probability.python.internal import prefer_static as ps
from tensorflow_probability.python.internal import tensorshape_util
from tensorflow.python.ops import control_flow_util  # pylint: disable=g-direct-tensorflow-import


JAX_MODE = False


def _variadic_reduce(t, init, axis, reducer):
  """Implements a basic variadic reduce by repeated halving.

  The function executes recursively with each call trimming off one axis. Per
  axis, the computation uses `tf.while_loop` to repeatedly reduce the two halves
  of the tensor along the axis in question (padding with `init` as needed).

  Args:
    t: The tuple of tensors to reduce.
    init: A tuple of scalar initializations with dtypes aligned with `t`.
    axis: A sequence of python `int`.
    reducer: The function implementing the reduction. Each of the two arguments
      to `reducer` is a tuple like `t`.

  Returns:
    reduced: A tuple like `t` with the given reduction applied.
  """
  if not axis:
    return t

  if len(axis) > 1:
    if any(ax < 0 for ax in axis):
      raise ValueError('All axis args must be non-negative: {}'.format(axis))
    axis = sorted(axis)
    # Reduce innermost dims first, so that positive `axis` args remain aligned.
    reduced_innermost_axis = _variadic_reduce(t, init, axis[-1:], reducer)
    return _variadic_reduce(reduced_innermost_axis, init, axis[:-1], reducer)

  ax = axis[0]
  if any(part.shape[ax] != t[0].shape[ax] for part in t):
    raise ValueError(
        'Mismatched shapes along axis {}: {}'.format(
            ax, [part.shape for part in t]))

  def cond(*t):
    return tf.not_equal(tf.shape(t[0])[ax], 1)

  def body(*t):
    dim = tf.shape(t[0])[ax]
    lhs, rhs = [], []
    for part, part_init in zip(t, init):
      p0, p1 = tf.split(part, [dim // 2, dim // 2 + dim % 2], axis=ax)
      paddings = [(0, 0)] * tensorshape_util.rank(p0.shape)
      # Ensure we have two `f` operands of matching size:
      # 1. pad 0 sized dim up to 1.
      # 2. pad smaller-by-1 dim up to larger.
      paddings[ax] = (0, tf.maximum(tf.shape(p1)[ax], 1) - tf.shape(p0)[ax])
      p0 = tf.pad(p0, paddings, constant_values=part_init)
      lhs.append(p0)
      paddings[ax] = (0, tf.shape(p0)[ax] - tf.shape(p1)[ax])
      # p1 may need padding if both are size 0 in the dim.
      p1 = tf.pad(p1, paddings, constant_values=part_init)
      rhs.append(p1)
    return reducer(tuple(lhs), tuple(rhs))

  shape_invariants = tuple(
      part.shape[:ax] + (None,) + part.shape[ax + 1:] for part in t)
  t = tuple(t)
  result = tf.while_loop(cond, body, t, shape_invariants=shape_invariants)
  # Squeeze out the singleton dim in ax.
  return tuple(tf.squeeze(part, axis=ax) for part in result)


def make_variadic_reduce(reducer, vjp_bwd, tangents_fn):
  """Wraps a generic reducer function as a variadic reduction.

  The current use-case for this is TFP-internal. This function captures logic
  related to specific substrates and XLA, and enables some sharing of logic
  around `axis` and `keepdims` args.

  Args:
    reducer: The reducer callable. Takes two tuple args and returns a single
      reduced tuple.
    vjp_bwd: Custom VJP function. Takes `(aux, grads)` args with
      `aux = (operands, inits, axis, unsqueezed_shape)` and returns a tuple of
      grads w.r.t `operands`. Gradients w.r.t. `inits` are presumed `None`.
    tangents_fn: Custom JVP function. Takes `(inits, axis, primals, tangents)`
      args with `primals = (operands,)` and corresponding tangents and returns
      `tangents_out` (which must be linear w.r.t. `tangents`).

  Returns:
    reduce_fn: A callable with taking args
      `(operands, inits, axis=None, keepdims=False)`.
  """

  # Top-level `tf.function` for XLA (closed-over by the returned reduce_fn).
  @implementation_selection.never_runs_functions_eagerly
  @tf.function(jit_compile=True)
  def _xla_reduce(operands, inits, axis):
    """JIT-ed wrapper for TF `xla.variadic_reduce(..., reducer)`."""
    from tensorflow.compiler.tf2xla.python import xla  # pylint: disable=g-direct-tensorflow-import,g-import-not-at-top
    result = xla.variadic_reduce(
        operands,
        init_value=inits,
        dimensions_to_reduce=axis,
        reducer=tf.function(reducer).get_concrete_function(inits, inits))
    # Graph mode: variadic reduce doesn't specify output shapes. Patch that.
    shp = operands[0].shape
    for arg in operands:
      shp = tensorshape_util.merge_with(shp, arg.shape)
    for part in result:
      tensorshape_util.set_shape(
          part, tuple(dim for i, dim in enumerate(shp) if i not in axis))
    return result

  def _variadic_reduce_no_grad(operands, inits, axis, reducer):
    if JAX_MODE:
      from jax import lax  # pylint: disable=g-import-not-at-top
      return lax.reduce(
          operands, init_values=inits, dimensions=axis, computation=reducer)
    elif (tf.executing_eagerly() or
          not control_flow_util.GraphOrParentsInXlaContext(
              tf1.get_default_graph())):
      return _variadic_reduce(
          operands, init=inits, axis=axis, reducer=reducer)
    else:
      return _xla_reduce(operands, inits, axis)

  def _variadic_reduce_fwd(operands, inits, axis, reducer, unsqueezed_shape):
    del unsqueezed_shape
    return (_variadic_reduce_no_grad(operands, inits, axis, reducer),
            (operands, inits))

  def _variadic_reduce_jvp(axis, reducer, unsqueezed_shape, primals,
                           tangents):
    del unsqueezed_shape
    operands, inits = primals
    return (_variadic_reduce_no_grad(operands, inits, axis, reducer),
            tangents_fn(axis, primals, tangents))

  @tfp_custom_gradient.custom_gradient(vjp_fwd=_variadic_reduce_fwd,
                                       vjp_bwd=vjp_bwd,
                                       jvp_fn=_variadic_reduce_jvp,
                                       nondiff_argnums=(2, 3, 4))
  def _variadic_reduce_custom_grad(operands, inits, axis, reducer,
                                   unsqueezed_shape):
    del unsqueezed_shape  # provided for backprop convenience
    return _variadic_reduce_no_grad(operands, inits, axis, reducer)

  def reduce_fn(operands, inits, axis=None, keepdims=False):
    """Applies `reducer` to the given operands along the given axes.

    Args:
      operands: tuple of tensors, all having the same shape.
      inits: tuple of scalar tensors, with dtypes aligned to those of operands.
      axis: The axis or axes to reduce. One of `None`, an `int` or a sequence of
        `int`. `None` is taken to mean "reduce all axes".
      keepdims: When `True`, we do not squeeze away the reduced dims, instead
        returning values with singleton dims in those axes.

    Returns:
      reduced: A tuple of the reduced operands.
    """
    # Static shape consistency checks.
    args_shape = operands[0].shape
    for arg in operands[1:]:
      args_shape = tensorshape_util.merge_with(args_shape, arg.shape)
    ndims = tensorshape_util.rank(args_shape)
    if ndims is None:
      raise ValueError(
          'Rank of at least one of `operands` must be known statically.')
    # Ensure the 'axis' arg is a tuple of non-negative ints.
    axis = np.arange(ndims) if axis is None else np.array(axis)
    if axis.ndim > 1:
      raise ValueError('`axis` must be `None`, an `int`, or a sequence of '
                       '`int`, but got {}'.format(axis))
    axis = np.reshape(axis, [-1])
    axis = np.where(axis < 0, axis + ndims, axis)
    axis = tuple(int(ax) for ax in axis)

    axis_nhot = ps.reduce_sum(
        ps.one_hot(axis, depth=ndims,
                   on_value=True, off_value=False, dtype=tf.bool),
        axis=0)
    in_shape = args_shape
    if not tensorshape_util.is_fully_defined(in_shape):
      in_shape = tf.shape(operands[0])
    unsqueezed_shape = ps.where(axis_nhot, 1, in_shape)

    result = _variadic_reduce_custom_grad(
        operands, inits, axis, reducer, unsqueezed_shape)

    if keepdims:
      result = tf.nest.map_structure(
          lambda t: tf.reshape(t, unsqueezed_shape), result)
    return result

  return reduce_fn

