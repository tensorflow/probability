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
tfed = tfp.experimental.distribute

InverseAndILDJ = inverse.core.InverseAndILDJ

random_variable_p = primitive.InitialStylePrimitive('random_variable')
unzip.block_registry.add(random_variable_p)


def random_variable_log_prob_rule(flat_incells, flat_outcells, *, num_consts,
                                  in_tree, out_tree, **_):
  """Registers Oryx distributions with the log_prob transformation."""
  _, incells = jax_util.split_list(flat_incells, [num_consts])
  val_incells = incells[1:]
  if not all(cell.top() for cell in val_incells):
    return flat_incells, flat_outcells, None
  if not all(cell.top() for cell in flat_outcells):
    return flat_incells, flat_outcells, None
  seed_flat_invals = [object()] + [cell.val for cell in val_incells]
  flat_outvals = [cell.val for cell in flat_outcells]
  _, dist = tree_util.tree_unflatten(in_tree, seed_flat_invals)
  outval = tree_util.tree_unflatten(out_tree, flat_outvals)
  return flat_incells, flat_outcells, dist.log_prob(outval)

log_prob.log_prob_rules[random_variable_p] = random_variable_log_prob_rule

log_prob.log_prob_registry.add(random_variable_p)


def _sample_distribution(key, dist):
  return dist.sample(seed=key)


@ppl.random_variable.register(tfd.Distribution)
def distribution_random_variable(dist: tfd.Distribution, *,
                                 name: Optional[str] = None,
                                 plate: Optional[str] = None):
  """Converts a distribution into a sampling function."""
  if plate is not None:
    dist = tfed.Sharded(dist, plate)
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
