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
"""Module for log_prob transformation."""
from jax import core as jax_core
from jax import random
from jax import tree_util

from oryx.core import trace_util
from oryx.core.interpreters import inverse
from oryx.core.interpreters import propagate

__all__ = [
    'LogProbRules',
    'log_prob'
]

safe_map = jax_core.safe_map

InverseAndILDJ = inverse.core.InverseAndILDJ
ildj_registry = inverse.core.ildj_registry


class LogProbRules(dict):
  """Default dictionary for log_prob propagation rules.

  By default, the rules for LogProb propagation are just the InverseAndILDJ
  rules, but instead of raising a NotImplementedError, LogProb will silently
  fail. This default dict-like class implements this behavior, but also allows
  primitives to register custom propagation rules.
  """

  def __missing__(self, prim):
    self[prim] = rule = make_default_rule(prim)
    return rule


log_prob_rules = LogProbRules()

# The log_prob_registry is used to compute log_prob values from samples after
# propagation is done.
log_prob_registry = set()


def log_prob(f):
  """LogProb function transformation."""

  def wrapped(sample, *args, **kwargs):
    """Function wrapper that takes in log_prob arguments."""
    # Trace the function using a random seed
    dummy_seed = random.PRNGKey(0)
    jaxpr, _ = trace_util.stage(f, dynamic=False)(dummy_seed, *args, **kwargs)
    flat_outargs, _ = tree_util.tree_flatten(sample)
    flat_inargs, _ = tree_util.tree_flatten(args)
    constcells = [InverseAndILDJ.new(val) for val in jaxpr.literals]
    flat_incells = [
        InverseAndILDJ.unknown(trace_util.get_shaped_aval(dummy_seed))
    ] + [InverseAndILDJ.new(val) for val in flat_inargs]
    flat_outcells = [InverseAndILDJ.new(a) for a in flat_outargs]
    return log_prob_jaxpr(jaxpr.jaxpr, constcells, flat_incells, flat_outcells)

  return wrapped


@tree_util.register_pytree_node_class
class FailedLogProb:

  def tree_flatten(self):
    return (), ()

  @classmethod
  def tree_unflatten(cls, data, xs):
    del data, xs
    return FailedLogProb()


# sentinel for being unable to compute a log_prob
failed_log_prob = FailedLogProb()


def log_prob_jaxpr(jaxpr, constcells, flat_incells, flat_outcells):
  """Runs log_prob propagation on a Jaxpr."""

  def reducer(env, eqn, curr_log_prob, new_log_prob):
    if (isinstance(curr_log_prob, FailedLogProb)
        or isinstance(new_log_prob, FailedLogProb)):
      # If `curr_log_prob` is `None` that means we were unable to compute
      # a log_prob elsewhere, so the propagate failed.
      return failed_log_prob
    if eqn.primitive in log_prob_registry and new_log_prob is None:
      # We are unable to compute a log_prob for this primitive.
      return failed_log_prob
    if new_log_prob is not None:
      cells = [env.read(var) for var in eqn.outvars]
      ildjs = sum([cell.ildj.sum() for cell in cells if cell.top()])
      return curr_log_prob + new_log_prob + ildjs
    return curr_log_prob

  # Re-use the InverseAndILDJ propagation but silently fail instead of
  # erroring when we hit a primitive we can't invert. We accumulate the log
  # probability values using the propagater state.
  _, final_log_prob = propagate.propagate(
      InverseAndILDJ,
      log_prob_rules,
      jaxpr,
      constcells,
      flat_incells,
      flat_outcells,
      reducer=reducer,
      initial_state=0.)
  if final_log_prob is failed_log_prob:
    raise ValueError('Cannot compute log_prob of function.')
  return final_log_prob


def make_default_rule(prim):
  """Creates rule for prim without a registered log_prob."""

  def rule(incells, outcells, **params):
    """Executes the inverse rule but fails if the inverse isn't implemented."""
    try:
      return ildj_registry[prim](incells, outcells, **params)
    except NotImplementedError:
      return incells, outcells, None

  return rule
