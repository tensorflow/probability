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
from typing import Any, Callable, Dict, List, Tuple, Type, Union

import dataclasses
from jax import core as jax_core
from jax import linear_util as lu
from jax import tree_util
from jax.interpreters import partial_eval as pe
from jax.interpreters import xla

from oryx.core import pytree
from oryx.core.interpreters import harvest

__all__ = [
    'Cell',
    'Equation',
    'Environment',
    'propagate'
]

State = Any
VarOrLiteral = Union[jax_core.Var, jax_core.Literal]
safe_map = jax_core.safe_map


class Cell(pytree.Pytree):
  """Base interface for objects used during propagation.

  A Cell represents a member of a lattice, defined by the `top`, `bottom`
  and `join` methods. Conceptually, a "top" cell represents complete information
  about a value and a "bottom" cell represents no information about a value.
  Cells that are neither top nor bottom thus have partial information.
  The `join` method is used to combine two cells to create a cell no less than
  the two input cells. During the propagation, we hope to join cells until
  all cells are "top".

  Transformations that use propagate need to pass in objects that are Cell-like.
  A Cell needs to specify how to create a new default cell from a literal value,
  using the `new` class method. A Cell also needs to indicate if it is a known
  value with the `is_unknown` method, but by default, Cells are known.
  """

  def __init__(self, aval):
    self.aval = aval

  def __lt__(self, other: Any) -> bool:
    raise NotImplementedError

  def top(self) -> bool:
    raise NotImplementedError

  def bottom(self) -> bool:
    raise NotImplementedError

  def join(self, other: 'Cell') -> 'Cell':
    raise NotImplementedError

  @property
  def shape(self) -> Tuple[int]:
    return self.aval.shape

  @property
  def ndim(self) -> int:
    return len(self.shape)

  def is_unknown(self):
    # Convenient alias
    return self.bottom()

  @classmethod
  def new(cls, value):
    """Creates a new instance of a Cell from a value."""
    raise NotImplementedError

  @classmethod
  def unknown(cls, aval):
    """Creates an unknown Cell from an abstract value."""
    raise NotImplementedError


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
    return Equation(
        tuple(eqn.invars), tuple(eqn.outvars), eqn.primitive,
        tuple(params_flat), params_tree)

  @property
  def params(self):
    return tree_util.tree_unflatten(self.params_tree, self.params_flat)

  def __hash__(self):
    # Override __hash__ to use Literal object IDs because Literals are not
    # natively hashable
    hashable_invars = tuple(
        id(invar) if isinstance(invar, jax_core.Literal) else invar
        for invar in self.invars)
    return hash(
        (hashable_invars, self.outvars, self.primitive, self.params_tree))

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
    self.env: Dict[jax_core.Var, Cell] = {}
    self.states: Dict[Equation, Cell] = {}
    self.jaxpr: jax_core.Jaxpr = jaxpr

  def read(self, var: VarOrLiteral) -> Cell:
    if isinstance(var, jax_core.Literal):
      return self.cell_type.new(var.val)
    else:
      return self.env.get(var, self.cell_type.unknown(var.aval))

  def write(self, var: VarOrLiteral, cell: Cell) -> Cell:
    if isinstance(var, jax_core.Literal):
      return cell
    cur_cell = self.read(var)
    if var is jax_core.dropvar:
      return cur_cell
    self.env[var] = cur_cell.join(cell)
    return self.env[var]

  def __getitem__(self, var: VarOrLiteral) -> Cell:
    return self.read(var)

  def __setitem__(self, key, val):
    raise ValueError('Environments do not support __setitem__. Please use the '
                     '`write` method instead.')

  def __contains__(self, var: VarOrLiteral):
    if isinstance(var, jax_core.Literal):
      return True
    return var in self.env

  def read_state(self, eqn: Equation) -> State:
    return self.states.get(eqn, None)

  def write_state(self, eqn: Equation, state: State) -> None:
    self.states[eqn] = state


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


def update_queue_state(queue, cur_eqn, get_neighbor_eqns, incells, outcells,
                       new_incells, new_outcells):
  """Updates the queue from the result of a propagation."""
  all_vars = cur_eqn.invars + cur_eqn.outvars
  old_cells = incells + outcells
  new_cells = new_incells + new_outcells

  for var, old_cell, new_cell in zip(all_vars, old_cells, new_cells):
    # If old_cell is less than new_cell, we know the propagation has made
    # progress.
    if old_cell < new_cell:
      # Extend left as a heuristic because in graphs corresponding to
      # chains of unary functions, we immediately want to pop off these
      # neighbors in the next iteration
      neighbors = get_neighbor_eqns(var) - set(queue) - {cur_eqn}
      queue.extendleft(neighbors)


PropagationRule = Callable[[List[Any], List[Cell]], Tuple[List[Cell],
                                                          List[Cell], State]]


def identity_reducer(env, eqn, state, new_state):
  del env, eqn, new_state
  return state


def propagate(cell_type: Type[Cell],
              rules: Dict[jax_core.Primitive, PropagationRule],
              jaxpr: pe.Jaxpr,
              constcells: List[Cell],
              incells: List[Cell],
              outcells: List[Cell],
              reducer: Callable[[Environment, Equation, State, State],
                                State] = identity_reducer,
              initial_state: State = None) -> Tuple[Environment, State]:
  """Propagates cells in a Jaxpr using a set of rules.

  Args:
    cell_type: used to instantiate literals into cells
    rules: maps JAX primitives to propagation rule functions
    jaxpr: used to construct the propagation graph
    constcells: used to populate the Jaxpr's constvars
    incells: used to populate the Jaxpr's invars
    outcells: used to populate the Jaxpr's outcells
    reducer: An optional callable used to reduce over the state at each
      equation in the Jaxpr. `reducer` takes in `(env, eqn, state, new_state)`
      as arguments and should return an updated state. The `new_state` value
      is provided by each equation.
    initial_state: The initial `state` value used in the reducer
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
  for eqn in jaxpr.eqns:
    for var in it.chain(eqn.invars, eqn.outvars):
      env.write(var, cell_type.unknown(var.aval))

  for var in it.chain(jaxpr.outvars, jaxpr.invars, jaxpr.constvars):
    out_eqns.update(get_neighbor_eqns(var))
  queue = collections.deque(out_eqns)
  while queue:
    eqn = queue.popleft()

    incells = safe_map(env.read, eqn.invars)
    outcells = safe_map(env.read, eqn.outvars)

    call_jaxpr, params = jax_core.extract_call_jaxpr(eqn.primitive, eqn.params)
    if call_jaxpr:
      subfuns = [
          lu.wrap_init(
              functools.partial(propagate, cell_type, rules, call_jaxpr, (),
                                initial_state=initial_state,
                                reducer=reducer))
      ]
      if eqn.primitive not in rules:
        rule = default_call_rules.get(eqn.primitive)
      else:
        rule = rules[eqn.primitive]
    else:
      subfuns = []
      rule = rules[eqn.primitive]
    new_incells, new_outcells, eqn_state = rule(subfuns + incells, outcells,
                                                **params)
    env.write_state(eqn, eqn_state)

    new_incells = safe_map(env.write, eqn.invars, new_incells)
    new_outcells = safe_map(env.write, eqn.outvars, new_outcells)

    update_queue_state(queue, eqn, get_neighbor_eqns, incells, outcells,
                       new_incells, new_outcells)
  state = initial_state
  for eqn in eqns:
    state = reducer(env, eqn, state, env.read_state(eqn))
  return env, state


@lu.transformation_with_aux
def flat_propagate(tree, *flat_invals):
  invals, outvals = tree_util.tree_unflatten(tree, flat_invals)
  env, state = yield ((invals, outvals), {})
  new_incells = [env.read(var) for var in env.jaxpr.invars]
  new_outcells = [env.read(var) for var in env.jaxpr.outvars]
  flat_out, out_tree = tree_util.tree_flatten(
      (new_incells, new_outcells, state))
  yield flat_out, out_tree


def call_rule(prim, incells, outcells, **params):
  """Propagate rule for call primitives."""
  f, incells = incells[0], incells[1:]
  flat_vals, in_tree = tree_util.tree_flatten((incells, outcells))
  new_params = dict(params)
  if 'donated_invars' in params:
    new_params['donated_invars'] = (False,) * len(flat_vals)
  f, aux = flat_propagate(f, in_tree)
  flat_out = prim.bind(f, *flat_vals, **new_params)
  out_tree = aux()
  return tree_util.tree_unflatten(out_tree, flat_out)


default_call_rules = {}
default_call_rules[xla.xla_call_p] = functools.partial(call_rule,
                                                       xla.xla_call_p)
default_call_rules[jax_core.call_p] = functools.partial(call_rule,
                                                        jax_core.call_p)
default_call_rules[pe.remat_call_p] = functools.partial(call_rule,
                                                        pe.remat_call_p)
default_call_rules[harvest.nest_p] = functools.partial(call_rule,
                                                       harvest.nest_p)
