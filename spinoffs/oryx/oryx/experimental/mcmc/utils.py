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
"""Contains MCMC utilities."""
from typing import Callable

from jax import tree_util
import jax.numpy as np

from oryx.core import ppl
from oryx.core.interpreters.inverse import core as inverse

__all__ = [
    'constrain'
]

LogProbFunction = ppl.LogProbFunction


def constrain(mapping_fn) -> Callable[[LogProbFunction], LogProbFunction]:
  """Returns a log prob function that operates in an unconstrained space."""

  def wrap_log_prob(target_log_prob):

    def wrapped(*args):
      mapped_args = mapping_fn(*args)
      ildjs = inverse.ildj(mapping_fn, *args)(mapped_args)
      return target_log_prob(mapped_args) - np.sum(
          np.array(tree_util.tree_leaves(ildjs)))

    return wrapped

  return wrap_log_prob
