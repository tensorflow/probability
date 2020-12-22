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
    'inverse_log_det_jacobian_ratio',
    'RegisterILDJRatio',
]


_ildj_ratio_registry = {}


def inverse_log_det_jacobian_ratio(p, x, q, y, event_ndims, use_kahan_sum=True):
  """Computes `p.ildj(x, ndims) - q.idlj(y, ndims)`, numerically stably.

  Args:
    p: A bijector instance.
    x: A tensor from the support of `p.forward`.
    q: A bijector instance of the same type as `p`, with matching shape.
    y: A tensor from the support of `q.forward`.
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
  def ildj_ratio_fn(p, x, q, y):
    return (p.inverse_log_det_jacobian(x, event_ndims=min_event_ndims) -
            q.inverse_log_det_jacobian(y, event_ndims=min_event_ndims))

  for cls in inspect.getmro(type(p)):
    if cls in _ildj_ratio_registry:
      ildj_ratio_fn = _ildj_ratio_registry[cls]

  if use_kahan_sum:
    sum_fn = lambda x, axis: tfp_math.reduce_kahan_sum(x, axis=axis).total
  else:
    sum_fn = tf.reduce_sum
  return sum_fn(ildj_ratio_fn(p, x, q, y),
                axis=-1 - ps.range(event_ndims - min_event_ndims))


class RegisterILDJRatio(object):

  def __init__(self, bijector_class):
    self.cls = bijector_class

  def __call__(self, fn):
    assert self.cls not in _ildj_ratio_registry
    _ildj_ratio_registry[self.cls] = fn
    return fn

