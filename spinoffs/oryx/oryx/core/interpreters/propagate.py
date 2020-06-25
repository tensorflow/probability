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
# lint as: python3
"""Module for the propagate custom Jaxpr interpreter.

The propagate Jaxpr interpreter converts a Jaxpr to a directed graph where
vars are nodes and primitives are edges. It initializes invars and outvars with
Cells (an interface defined below), where a Cell encapsulates a value (or a set
of values) that a node in the graph can take on, and the Cell is computed from
neighboring Cells, using a set of propagation rules for each primitive.Each rule
indicates whether the propagation has been completed for the given edge.
If so, the propagate interpreter continues on to that primitive's neighbors
in the graph. Propagation continues until there are Cells for every node, or
when no further progress can be made. Finally, Cell values for all nodes in the
graph are returned.
"""
import collections
import functools
import itertools as it
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

import dataclasses
from jax import core as jax_core
from jax import linear_util as lu
from jax import tree_util
from jax import util as jax_util
from jax.interpreters import partial_eval as pe

safe_map = jax_core.safe_map


class Cell:
  """Base interface for objects used during propagation.

  Transformations that use propagate need to pass in objects that are Cell-like.
  A Cell needs to specify how to create a new default cell from a literal value,
  using the `new` class method. A Cell also needs to indicate if it is a known
  value wih the `is_unknown` method, but by default, Cells are known.
  """

  def is_unknown(self):
    return False

  @classmethod
  def new(cls, obj):
    """Creates a new instance of a Cell from a value."""
    raise NotImplementedError


class Unknown(Cell):
  """Sentinel type for unknown quantities during propagation."""

  def is_unknown(self):
    return True

  def __repr__(self):
    return '?'
unknown = Unknown()  # canonical unknown instance

tree_util.register_pytree_node(
    Unknown,
    lambda cell: ((), ()),
    lambda data, xs: unknown
)


@dataclasses.dataclass(frozen=True)
class Equation:
  """Hashable wrapper for jax_core.Jaxprs."""
  invars: Tuple[jax_core.Var]
  outvars: Tuple[jax_core.Var]
  primitive: jax_core.Primitive
  params_flat: Tuple[Any]
  params_tree: Any

  @classmethod
  def from_jaxpr_eqn(cls, eqn):
    params_flat, params_tree = tree_util.tree_flatten(eqn.params)
    return Equation(tuple(eqn.invars), tuple(eqn.outvars), eqn.primitive,
                    tuple(params_flat), params_tree)

  @property
  def params(self):
    return tree_util.tree_unflatten(self.params_tree, self.params_flat)

  def __hash__(self):
    # Override __hash__ to use Literal object IDs because Literals are not
    # natively hashable
    hashable_invars = tuple(id(invar) if isinstance(invar, jax_core.Literal)
                            else invar for invar in self.invars)
    return hash((hashable_invars, self.outvars, self.primitive,
                 self.params_tree))

  def __str__(self):
    return '{outvars} = {primitive} {invars}'.format(
        invars=' '.join(map(str, self.invars)),
        outvars=' '.join(map(str, self.outvars)),
        primitive=self.primitive,
    )


class Environment:
  """Keeps track of variables and their values during propagation."""

  def __init__(self, cell_type, jaxpr):
    self.cell_type = cell_type
    self.env = {}
    self.subenvs = {}
    self.jaxpr = jaxpr

  def read(self, var):
    if isinstance(var, jax_core.Literal):
      return self.cell_type.new(var.val)
    else:
      return self.env.get(var, unknown)

  def write(self, var, cell):
    if isinstance(var, jax_core.Literal):
      return
    if not cell.is_unknown():
      self.env[var] = cell

  def __getitem__(self, key):
    return self.env.get(key, unknown)

  def __setitem__(self, key, val):
    raise NotImplementedError

  def __contains__(self, key):
    return key in self.env

  def write_subenv(self, eqn, subenv):
    self.subenvs[eqn] = subenv

  def to_tuple(self):
    env_keys, env_values = jax_util.unzip2(self.env.items())
    subenv_keys, subenv_values = jax_util.unzip2(self.subenvs.items())
    return (env_values, subenv_values), (env_keys, subenv_keys, self.cell_type,
                                         self.jaxpr)

  @classmethod
  def from_tuple(cls, data, xs):
    env_keys, subenv_keys, cell_type, jaxpr = data
    env_values, subenv_values = xs
    env = Environment(cell_type, jaxpr)
    env.env = dict(zip(env_keys, env_values))
    env.subenvs = dict(zip(subenv_keys, subenv_values))
    return env

tree_util.register_pytree_node(
    Environment,
    lambda env: env.to_tuple(),
    Environment.from_tuple
)


def construct_graph_representation(eqns):
  """Constructs a graph representation of a Jaxpr."""
  neighbors = collections.defaultdict(set)
  for eqn in eqns:
    for var in it.chain(eqn.invars, eqn.outvars):
      if isinstance(var, jax_core.Literal):
        continue
      neighbors[var].add(eqn)

  def get_neighbors(var):
    if isinstance(var, jax_core.Literal):
      return set()
    return neighbors[var]
  return get_neighbors


def update_queue_state(queue, cur_eqn, get_neighbor_eqns, done, done_eqns,
                       checked_eqns, incells, outcells, new_incells,
                       new_outcells):
  """Updates the queue, done_eqns, and checked_eqns from the result of a propagation."""
  all_vars = cur_eqn.invars + cur_eqn.outvars
  old_cells = incells + outcells
  new_cells = new_incells + new_outcells
  updated = False
  for var, old_cell, new_cell in zip(all_vars, old_cells, new_cells):
    if old_cell.is_unknown() and not new_cell.is_unknown():
      updated = True
  if updated:
    checked_eqns.clear()
  if done:
    # Reset checked_eqns and enqueue new equations for updated variables
    done_eqns.add(cur_eqn)
  else:
    # If equation is not done, it might have to be revisited after other
    # equations have been propagated
    assert cur_eqn not in done_eqns
    checked_eqns.add(cur_eqn)
    queue.append(cur_eqn)
  for var, old_cell, new_cell in zip(all_vars, old_cells, new_cells):
    if old_cell.is_unknown() and not new_cell.is_unknown():
      # Extend left as a heuristic because in graphs corresponding to
      # chains of unary functions, we immediately want to pop off these
      # neighbors in the next iteration
      queue.extendleft(get_neighbor_eqns(var) - set(queue) - done_eqns)


PropagationRule = Callable[
    [List[Any], List[Cell]],
    Tuple[List[Cell], List[Cell], bool, Optional[Environment]],
]


def propagate(cell_type: Type[Cell],
              rules: Dict[jax_core.Primitive, PropagationRule],
              jaxpr: pe.Jaxpr,
              constcells: List[Cell],
              incells: List[Cell],
              outcells: List[Cell]) -> Environment:
  """Propagates cells in a Jaxpr using a set of rules.

  Args:
    cell_type: used to instantiate literals into cells
    rules: maps JAX primitives to propagation rule functions
    jaxpr: used to construct the propagation graph
    constcells: used to populate the Jaxpr's constvars
    incells: used to populate the Jaxpr's invars
    outcells: used to populate the Jaxpr's outcells
  Returns:
    The Jaxpr environment after propagation has terminated
  """
  env = Environment(cell_type, jaxpr)

  safe_map(env.write, jaxpr.constvars, constcells)
  safe_map(env.write, jaxpr.invars, incells)
  safe_map(env.write, jaxpr.outvars, outcells)

  eqns = safe_map(Equation.from_jaxpr_eqn, jaxpr.eqns)

  get_neighbor_eqns = construct_graph_representation(eqns)
  # Initialize propagation queue with equations neighboring constvars, invars,
  # and outvars.
  out_eqns = set()
  for var in it.chain(jaxpr.outvars, jaxpr.invars, jaxpr.constvars):
    out_eqns.update(get_neighbor_eqns(var))
  queue = collections.deque(out_eqns)
  done_eqns = set()
  # checked_eqns is used to stop propagation if all equations in queue have
  # been checked without the propagation progressing
  checked_eqns = set()
  while queue:
    eqn = queue.popleft()
    assert eqn not in done_eqns

    incells = safe_map(env.read, eqn.invars)
    outcells = safe_map(env.read, eqn.outvars)

    rule = rules[eqn.primitive]
    call_jaxpr, params = jax_core.extract_call_jaxpr(eqn.primitive, eqn.params)
    if call_jaxpr:
      subfuns = [
          lu.wrap_init(functools.partial(propagate, cell_type, rules,
                                         call_jaxpr, ()))
      ]
    else:
      subfuns = []
    new_incells, new_outcells, done, subenv = rule(subfuns + incells,
                                                   outcells, **params)
    if subenv:
      env.write_subenv(eqn, subenv)

    safe_map(env.write, eqn.invars, new_incells)
    safe_map(env.write, eqn.outvars, new_outcells)

    update_queue_state(queue, eqn, get_neighbor_eqns, done, done_eqns,
                       checked_eqns, incells, outcells, new_incells,
                       new_outcells)
    if checked_eqns == set(queue):
      break
  return env
