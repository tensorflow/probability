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
# Lint as: python3
"""Probability functions for tfp.experimental.lazybones."""

from __future__ import absolute_import
from __future__ import division
# [internal] enable type annotations
from __future__ import print_function

import functools
import operator


import tensorflow.compat.v2 as tf

from tensorflow_probability.python.experimental.lazybones import deferred
from tensorflow_probability.python.experimental.lazybones import deferred_scope


__all__ = [
    'distribution_measure',
    'log_prob',
    'prob',
]


def log_prob(vertexes, values):
  """Returns `log_prob` when `vertexes` take on `values`."""
  return distribution_measure(vertexes, values, 'log_prob', sum)


def prob(vertexes, values):
  """Returns `prob` when `vertexes` take on `values`."""
  return distribution_measure(vertexes, values, 'prob', _prod)


def distribution_measure(vertexes, values, attr, combine):
  """Returns `getattr(distribution)` when `vertexes` take on `values`."""
  vertexes = tf.nest.flatten(vertexes)
  values = tf.nest.flatten(values)
  distributions = []
  with deferred_scope.DeferredScope():
    for x, v in zip(vertexes, values):
      if not isinstance(x, deferred.DeferredBase):
        raise ValueError()
      if v is not None:
        # TODO(jvdillon): If eval recursively eval'ed we could assign this as a
        # deferred try-cast.
        x.value = v
      d = x.parents[0].parents[0]
      distributions.append(d)
    r = combine(getattr(d, attr)(x) for d, x in zip(distributions, vertexes))
    return r.eval()


def _prod(iterable):
  return functools.reduce(operator.mul, iterable, 1)
