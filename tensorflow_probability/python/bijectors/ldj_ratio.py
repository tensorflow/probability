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
"""Computes log-ratios of Jacobian determinants numerically stably."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import inspect

import tensorflow.compat.v2 as tf

from tensorflow_probability.python.bijectors import bijector as bijector_lib
from tensorflow_probability.python.internal import prefer_static as ps

__all__ = [
    'forward_log_det_jacobian_ratio',
    'inverse_log_det_jacobian_ratio',
    'RegisterFLDJRatio',
    'RegisterILDJRatio',
]

_fldj_ratio_registry = {}
_ildj_ratio_registry = {}


def _reduce_ldj_ratio(unreduced_ldj_ratio, p, q, input_shape, min_event_ndims,
                      event_ndims):
  """Reduces an LDJ ratio computed with event_ndims=min_event_ndims."""
  # pylint: disable=protected-access
  have_parameter_batch_shape = (
      p._parameter_batch_shape is not None and
      q._parameter_batch_shape is not None)
  if have_parameter_batch_shape:
    parameter_batch_shape = ps.broadcast_shape(p._parameter_batch_shape,
                                               q._parameter_batch_shape)
  else:
    parameter_batch_shape = None

  reduce_shape, assertions = bijector_lib.ldj_reduction_shape(
      input_shape,
      event_ndims=event_ndims,
      min_event_ndims=min_event_ndims,
      parameter_batch_shape=parameter_batch_shape,
      allow_event_shape_broadcasting=not (p._parts_interact or
                                          q._parts_interact),
      validate_args=p.validate_args or q.validate_args)

  sum_fn = getattr(p, '_sum_fn', getattr(q, '_sum_fn', tf.reduce_sum))
  with tf.control_dependencies(assertions):
    return bijector_lib.reduce_jacobian_det_over_shape(
        unreduced_ldj_ratio, reduce_shape=reduce_shape, sum_fn=sum_fn)


def _default_fldj_ratio_fn(p, x, q, y, event_ndims, p_kwargs, q_kwargs):
  min_event_ndims = p.forward_min_event_ndims
  unreduced_fldj_ratio = (
      p.forward_log_det_jacobian(x, event_ndims=min_event_ndims, **p_kwargs) -
      q.forward_log_det_jacobian(y, event_ndims=min_event_ndims, **q_kwargs))
  return _reduce_ldj_ratio(unreduced_fldj_ratio, p, q, ps.shape(x),
                           min_event_ndims, event_ndims)


def _default_ildj_ratio_fn(p, x, q, y, event_ndims, p_kwargs, q_kwargs):
  min_event_ndims = p.inverse_min_event_ndims
  unreduced_fldj_ratio = (
      p.inverse_log_det_jacobian(x, event_ndims=min_event_ndims, **p_kwargs) -
      q.inverse_log_det_jacobian(y, event_ndims=min_event_ndims, **q_kwargs))
  return _reduce_ldj_ratio(unreduced_fldj_ratio, p, q, ps.shape(x),
                           min_event_ndims, event_ndims)


def _get_ldj_ratio_fn(cls, registry):
  ldj_ratio_fn = None
  for ref_cls in inspect.getmro(cls):
    if ref_cls in registry:
      ldj_ratio_fn = registry[ref_cls]
      break
  return ldj_ratio_fn


def forward_log_det_jacobian_ratio(
    p,
    x,
    q,
    y,
    event_ndims,
    p_kwargs=None,
    q_kwargs=None,
):
  """Computes `p.fldj(x, ndims) - q.fdlj(y, ndims)`, numerically stably.

  `p_kwargs` and `q_kwargs` are passed to the registered `fldj_ratio_fn`. The
  fallback implementation passes them to the `forward_log_det_jacobian` methods
  of `p` and `q`.

  Args:
    p: A bijector instance.
    x: A tensor from the preimage of `p.forward`.
    q: A bijector instance of the same type as `p`, with matching shape.
    y: A tensor from the preimage of `q.forward`.
    event_ndims: The number of right-hand dimensions comprising the event shapes
      of `x` and `y`.
    p_kwargs: Keyword args to pass to `p`.
    q_kwargs: Keyword args to pass to `q`.

  Returns:
    fldj_ratio: `log ((abs o det o jac p)(x) / (abs o det o jac q)(y))`,
      i.e. in TFP code, `p.forward_log_det_jacobian(x, event_ndims) -
      q.forward_log_det_jacobian(y, event_ndims)`. In some cases
      this will be computed with better than naive numerical precision, e.g. by
      moving differences inside of a sum reduction.
  """
  assert type(p) == type(q)  # pylint: disable=unidiomatic-typecheck
  if p_kwargs is None:
    p_kwargs = {}
  if q_kwargs is None:
    q_kwargs = {}

  fldj_ratio_fn = _get_ldj_ratio_fn(type(p), registry=_fldj_ratio_registry)
  ildj_ratio_fn = _get_ldj_ratio_fn(type(p), registry=_ildj_ratio_registry)

  def inverse_fldj_ratio_fn(p, x, q, y, event_ndims, p_kwargs, q_kwargs):
    # p.fldj(x) - q.fldj(y) = q.ildj(q(y)) - p.ildj(p(x))
    return ildj_ratio_fn(q, q.forward(y), p, p.forward(x),
                         q.forward_event_ndims(event_ndims), q_kwargs, p_kwargs)

  if fldj_ratio_fn is None:
    if ildj_ratio_fn is None:
      fldj_ratio_fn = _default_fldj_ratio_fn
    else:
      fldj_ratio_fn = inverse_fldj_ratio_fn

  return fldj_ratio_fn(p, x, q, y, event_ndims, p_kwargs, q_kwargs)


def inverse_log_det_jacobian_ratio(
    p,
    x,
    q,
    y,
    event_ndims,
    p_kwargs=None,
    q_kwargs=None,
):
  """Computes `p.ildj(x, ndims) - q.idlj(y, ndims)`, numerically stably.

  `p_kwargs` and `q_kwargs` are passed to the registered `ildj_ratio_fn`. The
  fallback implementation passes them to the `inverse_log_det_jacobian` methods
  of `p` and `q`.

  Args:
    p: A bijector instance.
    x: A tensor from the image of `p.forward`.
    q: A bijector instance of the same type as `p`, with matching shape.
    y: A tensor from the image of `q.forward`.
    event_ndims: The number of right-hand dimensions comprising the event shapes
      of `x` and `y`.
    p_kwargs: Keyword args to pass to `p`.
    q_kwargs: Keyword args to pass to `q`.

  Returns:
    ildj_ratio: `log ((abs o det o jac p^-1)(x) / (abs o det o jac q^-1)(y))`,
      i.e. in TFP code, `p.inverse_log_det_jacobian(x, event_ndims) -
      q.inverse_log_det_jacobian(y, event_ndims)`. In some cases
      this will be computed with better than naive numerical precision, e.g. by
      moving differences inside of a sum reduction.
  """
  assert type(p) == type(q)  # pylint: disable=unidiomatic-typecheck
  if p_kwargs is None:
    p_kwargs = {}
  if q_kwargs is None:
    q_kwargs = {}

  ildj_ratio_fn = _get_ldj_ratio_fn(type(p), registry=_ildj_ratio_registry)
  fldj_ratio_fn = _get_ldj_ratio_fn(type(p), registry=_fldj_ratio_registry)

  def inverse_ildj_ratio_fn(p, x, q, y, event_ndims, p_kwargs, q_kwargs):
    # p.ildj(x) - q.ildj(y) = q.fldj(q^-1(y)) - p.fldj(p^-1(x))
    return fldj_ratio_fn(q, q.inverse(y), p, p.inverse(x),
                         q.inverse_event_ndims(event_ndims), q_kwargs, p_kwargs)

  if ildj_ratio_fn is None:
    if fldj_ratio_fn is None:
      ildj_ratio_fn = _default_ildj_ratio_fn
    else:
      ildj_ratio_fn = inverse_ildj_ratio_fn

  return ildj_ratio_fn(p, x, q, y, event_ndims, p_kwargs, q_kwargs)


class RegisterFLDJRatio(object):

  def __init__(self, bijector_class):
    self.cls = bijector_class

  def __call__(self, fn):
    assert self.cls not in _fldj_ratio_registry
    _fldj_ratio_registry[self.cls] = fn
    return fn


class RegisterILDJRatio(object):

  def __init__(self, bijector_class):
    self.cls = bijector_class

  def __call__(self, fn):
    assert self.cls not in _ildj_ratio_registry
    _ildj_ratio_registry[self.cls] = fn
    return fn
