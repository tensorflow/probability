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
"""Wraps TFP distributions for use with Jax."""
from typing import Optional

from jax import tree_util
from jax import util as jax_util
from oryx.core import ppl
from oryx.core import primitive
from oryx.core.interpreters import harvest
from oryx.core.interpreters import inverse
from oryx.core.interpreters import log_prob
from oryx.core.interpreters import unzip
from tensorflow_probability.substrates import jax as tfp

tfd = tfp.distributions


InverseAndILDJ = inverse.core.InverseAndILDJ

random_variable_p = primitive.InitialStylePrimitive('random_variable')
unzip.block_registry.add(random_variable_p)


def random_variable_log_prob_rule(flat_incells, flat_outcells, **params):
  """Registers Oryx distributions with the log_prob transformation."""
  del params
  return flat_incells, flat_outcells, None
log_prob.log_prob_rules[random_variable_p] = random_variable_log_prob_rule


def random_variable_log_prob(flat_incells, val, *, num_consts, in_tree, **_):
  """Registers Oryx distributions with the log_prob transformation."""
  _, flat_incells = jax_util.split_list(flat_incells, [num_consts])
  _, dist = tree_util.tree_unflatten(in_tree, flat_incells)
  if any(not cell.top() for cell in flat_incells[1:]
         if isinstance(val, InverseAndILDJ)):
    return None
  return dist.log_prob(val)


log_prob.log_prob_registry[
    random_variable_p] = random_variable_log_prob


def _sample_distribution(key, dist):
  return dist.sample(seed=key)


@ppl.random_variable.register(tfd.Distribution)
def distribution_random_variable(dist: tfd.Distribution, *,
                                 name: Optional[str] = None):
  """Converts a distribution into a sampling function."""
  def wrapped(key):
    def sample(key):
      result = primitive.initial_style_bind(
          random_variable_p,
          distribution_name=dist.__class__.__name__)(_sample_distribution)(
              key, dist)
      return result
    if name is not None:
      return ppl.random_variable(
          harvest.nest(sample, scope=name)(key), name=name)
    return sample(key)
  return wrapped


@ppl.log_prob.register(tfd.Distribution)
def distribution_log_prob(dist: tfd.Distribution):
  def wrapped(value):
    return dist.log_prob(value)
  return wrapped
