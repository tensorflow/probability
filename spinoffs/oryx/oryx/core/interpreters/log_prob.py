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
"""Module for log_prob transformation."""
from jax import core as jax_core
from jax import random
from jax import tree_util
import jax.numpy as np

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
log_prob_registry = {}


def log_prob(f):
  """LogProb function transformation."""
  def wrapped(sample, *args, **kwargs):
    """Function wrapper that takes in log_prob arguments."""
    # Trace the function using a random seed
    dummy_seed = random.PRNGKey(0)
    jaxpr, _ = trace_util.stage(f)(dummy_seed, *args, **kwargs)
    flat_outargs, _ = tree_util.tree_flatten(sample)
    flat_inargs, _ = tree_util.tree_flatten(args)
    constcells = [InverseAndILDJ.new(val) for val in jaxpr.literals]
    flat_incells = [
        InverseAndILDJ.unknown(trace_util.get_shaped_aval(dummy_seed))
    ] + [InverseAndILDJ.new(val) for val in flat_inargs]
    flat_outcells = [InverseAndILDJ.new(a) for a in flat_outargs]
    # Re-use the InverseAndILDJ propagation but silently fail instead of
    # erroring when we hit a primitive we can't invert.
    env = propagate.propagate(InverseAndILDJ, log_prob_rules, jaxpr.jaxpr,
                              constcells, flat_incells, flat_outcells)
    # Traverse the resulting environment, looking for primitives that have
    # registered log_probs.
    final_log_prob = _accumulate_log_probs(env)
    return final_log_prob
  return wrapped


def _accumulate_log_probs(env):
  """Recursively traverses Jaxprs to accumulate log_prob values."""
  final_log_prob = 0.0
  eqns = safe_map(propagate.Equation.from_jaxpr_eqn, env.jaxpr.eqns)
  for eqn in eqns:
    if eqn.primitive in log_prob_registry:
      var, = eqn.outvars
      if var not in env:
        raise ValueError('Cannot compute log_prob of function.')
      incells = [env.read(v) for v in eqn.invars]
      outcells = [env.read(v) for v in eqn.outvars]
      outcell, = outcells
      if not outcell.top():
        raise ValueError('Cannot compute log_prob of function.')
      lp = log_prob_registry[eqn.primitive](
          [cell if not cell.top() else cell.val for cell in incells],
          outcell.val, **eqn.params
      )
      assert np.ndim(lp) == 0, 'log_prob must return a scalar.'
      # Accumulate ILDJ term
      final_log_prob += lp + np.sum(outcell.ildj)
  for subenv in env.subenvs.values():
    sub_lp = _accumulate_log_probs(subenv)
    final_log_prob += sub_lp
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
