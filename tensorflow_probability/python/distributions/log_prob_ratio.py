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
"""Computes log-ratios of probs numerically stably."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import inspect


__all__ = [
    'log_prob_ratio',
    'RegisterLogProbRatio',
]


_log_prob_ratio_registry = {}


def log_prob_ratio(p, x, q, y):
  """Computes `p.log_prob(x) - q.log_prob(y)`, numerically stably.

  Args:
    p: A distribution instance.
    x: A tensor from the support of `p`.
    q: A distribution instance in the same family as `p`, with matching shape.
    y: A tensor from the support of `q`.

  Returns:
    lp_ratio: `log (p(x) / q(y)) = p.log_prob(x) - q.log_prob(y)`. In some cases
      this will be computed with better than naive numerical precision, e.g. by
      moving the difference inside of a sum reduction.
  """
  assert type(p) == type(q)  # pylint: disable=unidiomatic-typecheck
  for cls in inspect.getmro(type(p)):
    if cls in _log_prob_ratio_registry:
      return _log_prob_ratio_registry[cls](p, x, q, y)
  return p.log_prob(x) - q.log_prob(y)


class RegisterLogProbRatio(object):

  def __init__(self, dist_family):
    self.family = dist_family

  def __call__(self, fn):
    assert self.family not in _log_prob_ratio_registry
    _log_prob_ratio_registry[self.family] = fn
    return fn

