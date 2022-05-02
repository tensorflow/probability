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
"""Wraps TFP distributions for use with Jax."""
import itertools as it

from typing import Optional

import jax
from jax import tree_util
from jax import util as jax_util
from jax.interpreters import batching
from oryx.core import ppl
from oryx.core import primitive
from oryx.core import trace_util
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
                                  in_tree, out_tree, batch_ndims, **_):
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
  return flat_incells, flat_outcells, dist.log_prob(outval).sum(
      axis=list(range(batch_ndims)))

log_prob.log_prob_rules[random_variable_p] = random_variable_log_prob_rule

log_prob.log_prob_registry.add(random_variable_p)


def random_variable_batching_rule(args, dims, *, num_consts, batch_ndims,
                                  jaxpr, **params):
  """Batching (vmap) rule for the `random_variable` primitive."""
  old_consts = args[:num_consts]
  args, dims = args[num_consts:], dims[num_consts:]
  def _run(*args):
    return random_variable_p.impl(*it.chain(old_consts, args),
                                  num_consts=len(old_consts),
                                  jaxpr=jaxpr,
                                  batch_ndims=batch_ndims,
                                  **params)
  run = jax.vmap(_run, in_axes=dims, out_axes=0)
  closed_jaxpr, _ = trace_util.stage(run, dynamic=True)(*args)
  new_jaxpr, new_consts = closed_jaxpr.jaxpr, closed_jaxpr.literals
  result = random_variable_p.bind(*it.chain(new_consts, args),
                                  num_consts=len(new_consts),
                                  jaxpr=new_jaxpr,
                                  batch_ndims=batch_ndims + 1,
                                  **params)
  return result, (0,) * len(result)
batching.primitive_batchers[random_variable_p] = random_variable_batching_rule


def _sample_distribution(key, dist):
  return dist.sample(seed=key)


@ppl.random_variable.register(tfd.Distribution)
def distribution_random_variable(dist: tfd.Distribution, *,
                                 name: Optional[str] = None,
                                 plate: Optional[str] = None):
  """Converts a distribution into a sampling function."""
  if plate is not None:
    dist = tfed.Sharded(dist, plate)
  if dist.batch_shape != []:  # pylint: disable=g-explicit-bool-comparison
    raise ValueError(
        f'Cannot use a distribution with `batch_shape`: {dist.batch_shape}. '
        'Instead, use `jax.vmap` or `ppl.plate` to draw independent samples.')
  def wrapped(key):
    def sample(key):
      result = primitive.initial_style_bind(
          random_variable_p,
          batch_ndims=0,
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
