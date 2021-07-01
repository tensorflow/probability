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
"""Registers custom rules for neural networks in the stateful function API.

The Oryx state API enables having a custom unzip rules when `init`-ing a
function. We use this for neural networks to thread kwargs through the Jaxpr
that is created when unzipping a function. This module implements this by first
replacing instances of `layer_cau` with a `FlatPrimitive`s, which avoids
using a call primitive, which we would be difficult to pass new keyword
arguments into. We can more easily override the behavior of a regular primitive.
"""

from jax import tree_util

from oryx.core import state
from oryx.experimental.nn import base

__all__ = [
]


def layer_cau_kwargs_rule(*flat_args, num_consts, in_tree, kwargs, **_):
  """Custom kwargs rule for layer_cau primitive."""
  flat_args = flat_args[num_consts:]
  layer, *args = tree_util.tree_unflatten(in_tree, flat_args)
  kwargs = dict(kwargs)
  has_rng = kwargs.pop('has_rng', False)
  if has_rng:
    rng, args = args[0], args[1:]
    kwargs = dict(kwargs, rng=rng)
  ans = layer.call_and_update(*args, **kwargs)
  return tree_util.tree_leaves(ans)


state.kwargs_rules[base.layer_cau_p] = layer_cau_kwargs_rule
