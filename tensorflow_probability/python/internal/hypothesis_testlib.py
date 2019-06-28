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
"""Hypothesis strategies for TFP."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import contextlib
import os
import traceback

# Dependency imports
from hypothesis.extra import numpy as hpnp
import hypothesis.strategies as hps
import numpy as np
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp
from tensorflow_probability.python.internal import assert_util
from tensorflow_probability.python.internal import tensorshape_util

tfd = tfp.distributions


def derandomize_hypothesis():
  # Use --test_env=TFP_DERANDOMIZE_HYPOTHESIS=0 to get random coverage.
  return bool(int(os.environ.get('TFP_DERANDOMIZE_HYPOTHESIS', 1)))


def hypothesis_max_examples():
  # Use --test_env=TFP_HYPOTHESIS_MAX_EXAMPLES=1000 to get fuller coverage.
  return int(os.environ.get('TFP_HYPOTHESIS_MAX_EXAMPLES', 20))


VAR_USAGES = {}


def usage_counting_identity(var):
  VAR_USAGES[var] = VAR_USAGES.get(var, []) + [traceback.format_stack(limit=15)]
  return tf.identity(var)


@contextlib.contextmanager
def assert_no_excessive_var_usage(cls_and_method, max_permissible=2):
  """Fails if too many convert_to_tensor calls happen to any DeferredTensor."""
  VAR_USAGES.clear()
  yield
  # TODO(jvdillon): Reduce max_permissible to 1?
  var_nusages = {var: len(usages) for var, usages in VAR_USAGES.items()}
  if any(len(usages) > max_permissible for usages in VAR_USAGES.values()):
    for var, usages in VAR_USAGES.items():
      if len(usages) > max_permissible:
        print('While executing {}, saw {} Tensor conversions of {}:'.format(
            cls_and_method, len(usages), var))
        for i, usage in enumerate(usages):
          print('Conversion {} of {}:\n{}'.format(i + 1, len(usages),
                                                  ''.join(usage)))
    raise AssertionError(
        'Excessive tensor conversions detected for {}: {}'.format(
            cls_and_method, var_nusages))


def constrained_tensors(constraint_fn, shape):
  """Draws the value of a single constrained parameter."""
  # TODO(bjp): Allow a wider range of floats.
  # float32s = hps.floats(
  #     np.finfo(np.float32).min / 2, np.finfo(np.float32).max / 2,
  #     allow_nan=False, allow_infinity=False)
  float32s = hps.floats(-200, 200, allow_nan=False, allow_infinity=False)

  def mapper(x):
    result = assert_util.assert_finite(
        constraint_fn(tf.convert_to_tensor(x)), message='param non-finite')
    if tf.executing_eagerly():
      return result.numpy()
    return result

  return hpnp.arrays(
      dtype=np.float32, shape=shape, elements=float32s).map(mapper)


# pylint: disable=no-value-for-parameter


@hps.composite
def batch_shapes(draw, min_ndims=0, max_ndims=3, min_lastdimsize=1):
  """Draws array shapes with some control over rank/dim sizes."""
  rank = draw(hps.integers(min_value=min_ndims, max_value=max_ndims))
  shape = tf.TensorShape(None).with_rank(rank)
  if rank > 0:

    def resize_lastdim(x):
      return x[:-1] + (max(x[-1], min_lastdimsize),)

    shape = draw(
        hpnp.array_shapes(min_dims=rank, max_dims=rank).map(resize_lastdim).map(
            tf.TensorShape))
  return shape


def identity_fn(x):
  return x


@hps.composite
def broadcasting_params(draw,
                        batch_shape,
                        event_dim=None,
                        enable_vars=False,
                        params_event_ndims=None,
                        constraint_fn_for=lambda param: identity_fn,
                        mutex_params=frozenset()):
  """Draws a dict of parameters which should yield the given batch shape."""
  if event_dim is None:
    event_dim = draw(hps.integers(min_value=2, max_value=6))

  params_event_ndims = params_event_ndims or {}
  remaining_params = set(params_event_ndims.keys())
  params_to_use = []
  while remaining_params:
    param = draw(hps.one_of(map(hps.just, remaining_params)))
    params_to_use.append(param)
    remaining_params.remove(param)
    for mutex_set in mutex_params:
      if param in mutex_set:
        remaining_params -= mutex_set

  param_batch_shapes = draw(
      broadcasting_named_shapes(batch_shape, params_to_use))
  params_kwargs = dict()
  for param in params_to_use:
    param_batch_shape = param_batch_shapes[param]
    param_event_rank = params_event_ndims[param]
    param_strategy = constrained_tensors(
        constraint_fn_for(param), (tensorshape_util.as_list(param_batch_shape) +
                                   [event_dim] * param_event_rank))
    params_kwargs[param] = tf.convert_to_tensor(
        draw(param_strategy), dtype=tf.float32, name=param)
    if enable_vars and draw(hps.booleans()):
      params_kwargs[param] = tf.Variable(params_kwargs[param], name=param)
      alt_value = tf.convert_to_tensor(
          draw(param_strategy),
          dtype=tf.float32,
          name='{}_alt_value'.format(param))
      setattr(params_kwargs[param], '_tfp_alt_value', alt_value)
      if draw(hps.booleans()):
        params_kwargs[param] = tfp.util.DeferredTensor(usage_counting_identity,
                                                       params_kwargs[param])
  return params_kwargs


@hps.composite
def broadcasting_named_shapes(draw, batch_shape, param_names):
  """Draws a set of parameter batch shapes that broadcast to `batch_shape`.

  For each parameter we need to choose its batch rank, and whether or not each
  axis i is 1 or batch_shape[i]. This function chooses a set of shapes that
  have possibly mismatched ranks, and possibly broadcasting axes, with the
  promise that the broadcast of the set of all shapes matches `batch_shape`.

  Args:
    draw: Hypothesis sampler.
    batch_shape: `tf.TensorShape`, the target (fully-defined) batch shape .
    param_names: Iterable of `str`, the parameters whose batch shapes need
      determination.

  Returns:
    param_batch_shapes: `dict` of `str->tf.TensorShape` where the set of
        shapes broadcast to `batch_shape`. The shapes are fully defined.
  """
  n = len(param_names)
  return dict(
      zip(
          draw(hps.permutations(param_names)),
          draw(broadcasting_shapes(batch_shape, n))))


def _compute_rank_and_fullsize_reqd(draw, target_shape, current_shape, is_last):
  """Returns a param rank and a list of bools for full-size-required by axis.

  Args:
    draw: Hypothesis data sampler.
    target_shape: `tf.TensorShape`, the target broadcasted shape.
    current_shape: `tf.TensorShape`, the broadcasted shape of the shapes
      selected thus far. This is ignored for non-last shapes.
    is_last: bool indicator of whether this is the last shape (in which case, we
      must achieve the target shape).

  Returns:
    next_rank: Sampled rank for the next shape.
    force_fullsize_dim: `next_rank`-sized list of bool indicating whether the
      corresponding axis of the shape must be full-sized (True) or is allowed to
      be 1 (i.e., broadcast) (False).
  """
  target_rank = target_shape.ndims
  if is_last:
    # We must force full size dim on any mismatched axes, and proper rank.
    full_rank_current = tf.broadcast_static_shape(
        current_shape, tf.TensorShape([1] * target_rank))
    # Identify axes in which the target shape is not yet matched.
    axis_is_mismatched = [
        full_rank_current[i] != target_shape[i] for i in range(target_rank)
    ]
    min_rank = target_rank
    if current_shape.ndims == target_rank:
      # Current rank might be already correct, but we could have a case like
      # batch_shape=[4,3,2] and current_batch_shape=[4,1,2], in which case
      # we must have at least 2 axes on this param's batch shape.
      min_rank -= (axis_is_mismatched + [True]).index(True)
    next_rank = draw(hps.integers(min_value=min_rank, max_value=target_rank))
    # Get the last param_batch_rank (possibly 0!) items.
    force_fullsize_dim = axis_is_mismatched[target_rank - next_rank:]
  else:
    # There are remaining params to be drawn, so we will be able to force full
    # size axes on subsequent params.
    next_rank = draw(hps.integers(min_value=0, max_value=target_rank))
    force_fullsize_dim = [False] * next_rank
  return next_rank, force_fullsize_dim


@hps.composite
def broadcast_compatible_shape(draw, batch_shape):
  """Draws a shape which is broadcast-compatible with `batch_shape`."""
  # broadcasting_shapes draws a sequence of shapes, so that the last "completes"
  # the broadcast to fill out batch_shape. Here we just draw two and take the
  # first (incomplete) one.
  return draw(broadcasting_shapes(batch_shape, 2))[0]


@hps.composite
def broadcasting_shapes(draw, target_shape, n):
  """Draws a set of `n` shapes that broadcast to `target_shape`.

  For each shape we need to choose its rank, and whether or not each axis i is 1
  or target_shape[i]. This function chooses a set of `n` shapes that have
  possibly mismatched ranks, and possibly broadcasting axes, with the promise
  that the broadcast of the set of all shapes matches `target_shape`.

  Args:
    draw: Hypothesis sampler.
    target_shape: The target (fully-defined) batch shape.
    n: `int`, the number of shapes to draw.

  Returns:
    shapes: Sequence of `tf.TensorShape` such that the set of shapes broadcast
      to `target_shape`. The shapes are fully defined.
  """
  target_shape = tf.TensorShape(target_shape)
  target_rank = target_shape.ndims
  result = []
  current_shape = tf.TensorShape([])
  for is_last in [False] * (n - 1) + [True]:
    next_rank, force_fullsize_dim = _compute_rank_and_fullsize_reqd(
        draw, target_shape, current_shape, is_last=is_last)

    # Get the last next_rank (possibly 0!) dimensions.
    next_shape = target_shape[target_rank - next_rank:].as_list()
    for i, force_fullsize in enumerate(force_fullsize_dim):
      if not force_fullsize and draw(hps.booleans()):
        # Choose to make this param broadcast against some other param.
        next_shape[i] = 1
    next_shape = tf.TensorShape(next_shape)
    current_shape = tf.broadcast_static_shape(current_shape, next_shape)
    result.append(next_shape)
  return result


# Utility functions for constraining parameters and/or domain/codomain members.


def softplus_plus_eps(eps=1e-6):
  return lambda x: tf.nn.softplus(x) + eps


def symmetric(x):
  return (x + tf.linalg.matrix_transpose(x)) / 2


def positive_definite(x):
  shp = tensorshape_util.as_list(x.shape)
  psd = (
      tf.matmul(x, x, transpose_b=True) +
      .1 * tf.linalg.eye(shp[-1], batch_shape=shp[:-2]))
  return symmetric(psd)


def lower_tril_positive_definite(x):
  return tf.linalg.band_part(
      tfd.matrix_diag_transform(x, softplus_plus_eps()), -1, 0)
