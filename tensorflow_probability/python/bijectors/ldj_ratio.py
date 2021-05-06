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

from tensorflow_probability.python import math as tfp_math
from tensorflow_probability.python.internal import prefer_static as ps

__all__ = [
    'forward_log_det_jacobian_ratio',
    'inverse_log_det_jacobian_ratio',
    'RegisterFLDJRatio',
    'RegisterILDJRatio',
]


_fldj_ratio_registry = {}
_ildj_ratio_registry = {}


def forward_log_det_jacobian_ratio(p, x, q, y, event_ndims, use_kahan_sum=True):
  """Computes `p.fldj(x, ndims) - q.fdlj(y, ndims)`, numerically stably.

  Args:
    p: A bijector instance.
    x: A tensor from the preimage of `p.forward`.
    q: A bijector instance of the same type as `p`, with matching shape.
    y: A tensor from the preimage of `q.forward`.
    event_ndims: The number of right-hand dimensions comprising the event shapes
      of `x` and `y`.
    use_kahan_sum: When `True`, the reduction of any remaining `event_ndims`
      beyond the minimum is done using Kahan summation. This requires statically
      known ranks.

  Returns:
    fldj_ratio: `log ((abs o det o jac p)(x) / (abs o det o jac q)(y))`,
      i.e. in TFP code, `p.forward_log_det_jacobian(x, event_ndims) -
      q.forward_log_det_jacobian(y, event_ndims)`. In some cases
      this will be computed with better than naive numerical precision, e.g. by
      moving differences inside of a sum reduction.
  """
  assert type(p) == type(q)  # pylint: disable=unidiomatic-typecheck

  min_event_ndims = p.forward_min_event_ndims
  def default_fldj_ratio_fn(p, x, q, y):
    return (p.forward_log_det_jacobian(x, event_ndims=min_event_ndims) -
            q.forward_log_det_jacobian(y, event_ndims=min_event_ndims))

  fldj_ratio_fn = None
  ildj_ratio_fn = None
  for cls in inspect.getmro(type(p)):
    if cls in _fldj_ratio_registry:
      fldj_ratio_fn = _fldj_ratio_registry[cls]
    if cls in _ildj_ratio_registry:
      ildj_ratio_fn = _ildj_ratio_registry[cls]

  if fldj_ratio_fn is None:
    if ildj_ratio_fn is None:
      fldj_ratio_fn = default_fldj_ratio_fn
    else:
      # p.fldj(x) - q.fldj(y) = q.ildj(q(y)) - p.ildj(p(x))
      fldj_ratio_fn = (
          lambda p, x, q, y: ildj_ratio_fn(q, q.forward(y), p, p.forward(x)))

  if use_kahan_sum:
    sum_fn = lambda x, axis: tfp_math.reduce_kahan_sum(x, axis=axis).total
  else:
    sum_fn = tf.reduce_sum
  return sum_fn(fldj_ratio_fn(p, x, q, y),
                axis=-1 - ps.range(event_ndims - min_event_ndims))


def inverse_log_det_jacobian_ratio(p, x, q, y, event_ndims, use_kahan_sum=True):
  """Computes `p.ildj(x, ndims) - q.idlj(y, ndims)`, numerically stably.

  Args:
    p: A bijector instance.
    x: A tensor from the image of `p.forward`.
    q: A bijector instance of the same type as `p`, with matching shape.
    y: A tensor from the image of `q.forward`.
    event_ndims: The number of right-hand dimensions comprising the event shapes
      of `x` and `y`.
    use_kahan_sum: When `True`, the reduction of any remaining `event_ndims`
      beyond the minimum is done using Kahan summation. This requires statically
      known ranks.

  Returns:
    ildj_ratio: `log ((abs o det o jac p^-1)(x) / (abs o det o jac q^-1)(y))`,
      i.e. in TFP code, `p.inverse_log_det_jacobian(x, event_ndims) -
      q.inverse_log_det_jacobian(y, event_ndims)`. In some cases
      this will be computed with better than naive numerical precision, e.g. by
      moving differences inside of a sum reduction.
  """
  assert type(p) == type(q)  # pylint: disable=unidiomatic-typecheck

  min_event_ndims = p.inverse_min_event_ndims
  def default_ildj_ratio_fn(p, x, q, y):
    return (p.inverse_log_det_jacobian(x, event_ndims=min_event_ndims) -
            q.inverse_log_det_jacobian(y, event_ndims=min_event_ndims))

  ildj_ratio_fn = None
  fldj_ratio_fn = None
  for cls in inspect.getmro(type(p)):
    if cls in _ildj_ratio_registry:
      ildj_ratio_fn = _ildj_ratio_registry[cls]
    if cls in _fldj_ratio_registry:
      fldj_ratio_fn = _fldj_ratio_registry[cls]

  if ildj_ratio_fn is None:
    if fldj_ratio_fn is None:
      ildj_ratio_fn = default_ildj_ratio_fn
    else:
      # p.ildj(x) - q.ildj(y) = q.fldj(q^-1(y)) - p.fldj(p^-1(x))
      ildj_ratio_fn = (
          lambda p, x, q, y: fldj_ratio_fn(q, q.inverse(y), p, p.inverse(x)))

  if use_kahan_sum:
    sum_fn = lambda x, axis: tfp_math.reduce_kahan_sum(x, axis=axis).total
  else:
    sum_fn = tf.reduce_sum
  return sum_fn(ildj_ratio_fn(p, x, q, y),
                axis=-1 - ps.range(event_ndims - min_event_ndims))


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

