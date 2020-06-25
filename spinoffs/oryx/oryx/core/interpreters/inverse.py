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
"""Module for inverse function transformation.

Utilizes the propagate evaluator with
rules that compute inverses and inverse
log-det Jacobians (ILDJs).
"""
import jax
from jax import core as jax_core
from jax import lax
from jax import linear_util as lu
from jax import tree_util
from jax import util as jax_util
from jax.interpreters import partial_eval as pe
from jax.interpreters import pxla
from jax.interpreters import xla
import jax.numpy as np

from oryx.core import primitive
from oryx.core import trace_util
from oryx.core.interpreters import harvest
from oryx.core.interpreters import propagate

safe_map = jax_core.safe_map
safe_zip = jax_core.safe_zip
unknown = propagate.unknown


class InverseAndILDJ(propagate.Cell):
  """Propagates inverse values and their ILDJs."""

  def __init__(self, val, ildj):  # pylint: disable=redefined-outer-name
    self.val = val
    self.ildj = ildj

  def __repr__(self):
    return 'InverseAndILDJ({}, {})'.format(self.val, self.ildj)

  @classmethod
  def new(cls, val):
    return InverseAndILDJ(val, np.array(0.))


tree_util.register_pytree_node(
    InverseAndILDJ,
    lambda cell: ((cell.val, cell.ildj), ()),
    lambda data, xs: InverseAndILDJ(xs[0], xs[1])
)


def inverse_and_ildj(f, *trace_args):
  """Inverse and ILDJ function transformation."""
  def wrapped(*args, **kwargs):
    """Function wrapper that takes in inverse arguments."""
    forward_args = trace_args if len(trace_args) else args
    jaxpr, (in_tree, _) = trace_util.stage(f)(*forward_args, **kwargs)
    flat_forward_args, _ = tree_util.tree_flatten(forward_args)
    flat_args, _ = tree_util.tree_flatten(args)
    flat_constcells = safe_map(InverseAndILDJ.new, jaxpr.literals)
    flat_incells = [unknown] * len(flat_forward_args)
    flat_outcells = safe_map(InverseAndILDJ.new, flat_args)
    env = propagate.propagate(InverseAndILDJ, ildj_registry, jaxpr.jaxpr,
                              flat_constcells, flat_incells, flat_outcells)
    flat_incells = [env.read(invar) for invar in jaxpr.jaxpr.invars]
    if any(flat_incell.is_unknown() for flat_incell in flat_incells):
      raise ValueError('Cannot invert function.')
    flat_cells, flat_ildjs = jax_util.unzip2([
        (flat_incell.val, flat_incell.ildj) for flat_incell in flat_incells
    ])
    vals = tree_util.tree_unflatten(in_tree, flat_cells)
    ildjs = tree_util.tree_unflatten(in_tree, flat_ildjs)
    if len(trace_args) == 1:
      vals, ildjs = vals[0], ildjs[0]
    return vals, ildjs
  return wrapped


def inverse(f, *trace_args):
  def wrapped(*args, **kwargs):
    return inverse_and_ildj(f, *trace_args)(*args, **kwargs)[0]
  return wrapped


def ildj(f, *trace_args):
  def wrapped(*args, **kwargs):
    return inverse_and_ildj(f, *trace_args)(*args, **kwargs)[1]
  return wrapped


def default_rule(prim, invals, outvals, **params):
  """Default inversion rule that only does forward eval."""
  if all(outval.is_unknown() for outval in outvals):
    if all(not inval.is_unknown() for inval in invals):
      vals = [inval.val for inval in invals]
      ans = prim.bind(*vals, **params)
      if not prim.multiple_results:
        ans = [ans]
      # Propagate can only invert functions that are constructed
      # autoregressively, and therefore the Jacobians of propagate-invertible
      # functions are lower-triangular. We are therefore safe assign outvals an
      # ILDJ value of 0 as they are part of forward propagation that will fill
      # in an off-diagonal entry of the Jacobian and will not contribute to the
      # log-det Jacobian.
      outvals = safe_map(InverseAndILDJ.new, ans)
      return invals, outvals, True, None
    else:
      return invals, outvals, False, None
  if any(outval.is_unknown() for outval in outvals):
    return invals, outvals, False, None
  raise NotImplementedError('No registered inverse for `{}`.'.format(prim))


def check_all_known(f):
  """Ensures only one path to compute the same value."""
  def wrapped(invals, outvals, **params):
    """Wraps inverse rule."""
    if all(not val.is_unknown() for val in invals + outvals):
      # If all values are already known, then
      # there are multiple ways of computing
      # at least one of the variables in the graph
      # and could result in inconsistent values.
      # For example, take the function
      # f = lambda x: (x, x + 1.), where calling
      # inverse(f)(3., 5.) will result in inconsistent
      # values for x. These types of function are disallowed.
      raise ValueError('Conflicting inverse paths.')
    return f(invals, outvals, **params)
  return wrapped


class InverseDict(object):
  """Default rules dictionary that uses a default rule for inverse."""

  def __init__(self):
    self.rules = {}

  def __getitem__(self, prim):
    if prim in custom_rules:
      return custom_rules[prim]
    if prim not in self.rules:
      self[prim] = jax_util.partial(default_rule, prim)
    return self.rules[prim]

  def __setitem__(self, prim, val):
    self.rules[prim] = check_all_known(val)


def register_elementwise(prim):
  """Registers an elementwise primitive with ILDJ."""
  def make_rule(f):
    """Accepts an inverse function for a primitive."""
    def ildj_rule(invals, outvals):
      """General InverseAndILDJ rule for elementwise functions."""
      outval, = outvals
      inval, = invals
      done = False
      if inval.is_unknown() and not outval.is_unknown():
        val = outval.val
        f_sum = lambda x: f(x).sum()
        invals = [InverseAndILDJ(f(val), outval.ildj +
                                 np.log(jax.grad(f_sum)(val)).sum())]
        done = True
      elif outval.is_unknown() and not inval.is_unknown():
        val = inval.val
        outvals = [InverseAndILDJ.new(prim.bind(val))]
        done = True
      return invals, outvals, done, None
    ildj_registry[prim] = ildj_rule
  return make_rule

custom_rules = {}

ildj_registry = InverseDict()
register_elementwise(lax.exp_p)(np.log)
register_elementwise(lax.log_p)(np.exp)
register_elementwise(lax.sin_p)(np.arcsin)
register_elementwise(lax.cos_p)(np.arccos)
register_elementwise(lax.expm1_p)(np.log1p)
register_elementwise(lax.log1p_p)(np.expm1)


def add_ildj(invals, outvals):
  """InverseAndILDJ rule for the add primitive."""
  outval, = outvals
  left, right = invals
  done = False
  if not outval.is_unknown():
    val, ildj_ = outval.val, outval.ildj
    if not left.is_unknown():
      invals = [left, InverseAndILDJ(val - left.val, ildj_)]
      done = True
    elif not right.is_unknown():
      invals = [InverseAndILDJ(val - right.val, ildj_), right]
      done = True
  elif outval.is_unknown() and not left.is_unknown() and not right.is_unknown():
    outvals = [InverseAndILDJ.new(left.val + right.val)]
    done = True
  return invals, outvals, done, None
ildj_registry[lax.add_p] = add_ildj


def sub_ildj(invals, outvals):
  """InverseAndILDJ rule for the add primitive."""
  outval, = outvals
  left, right = invals
  done = False
  if not outval.is_unknown():
    val, ildj_ = outval.val, outval.ildj
    if not left.is_unknown():
      invals = [left, InverseAndILDJ(left.val - val, ildj_)]
      done = True
    elif not right.is_unknown():
      invals = [InverseAndILDJ(val + right.val, ildj_), right]
      done = True
  elif outval.is_unknown() and not left.is_unknown() and not right.is_unknown():
    outvals = [InverseAndILDJ.new(left.val - right.val)]
    done = True
  return invals, outvals, done, None
ildj_registry[lax.sub_p] = sub_ildj


def mul_ildj(invals, outvals):
  """InverseAndILDJ rule for the mul primitive."""
  outval, = outvals
  left, right = invals
  done = False
  if not outval.is_unknown():
    val, ildj_ = outval.val, outval.ildj
    if not left.is_unknown():
      invals = [left, InverseAndILDJ(val / left.val, -np.log(
          np.abs(left.val)) + ildj_)]
      done = True
    elif not right.is_unknown():
      invals = [InverseAndILDJ(
          val / right.val, -np.log(np.abs(right.val)) + ildj_), right]
      done = True
  elif outval.is_unknown() and not left.is_unknown() and not right.is_unknown():
    outvals = [InverseAndILDJ.new(left.val * right.val)]
    done = True
  return invals, outvals, done, None
ildj_registry[lax.mul_p] = mul_ildj


def div_ildj(invals, outvals):
  """InverseAndILDJ rule for the mul primitive."""
  outval, = outvals
  left, right = invals
  done = False
  if not outval.is_unknown():
    val, ildj_ = outval.val, outval.ildj
    if not left.is_unknown():
      invals = [left, InverseAndILDJ(
          left.val / val, np.log(left.val) - 2 * np.log(val) + ildj_)]
      done = True
    elif not right.is_unknown():
      invals = [InverseAndILDJ(
          val * right.val, np.log(np.abs(right.val)) + ildj_), right]
      done = True
  elif outval.is_unknown() and not left.is_unknown() and not right.is_unknown():
    outvals = [InverseAndILDJ.new(left.val / right.val)]
    done = True
  return invals, outvals, done, None
ildj_registry[lax.div_p] = div_ildj


@lu.transformation_with_aux
def flat_propagate(tree, *flat_invals):
  invals, outvals = tree_util.tree_unflatten(tree, flat_invals)
  subenv = yield ((invals, outvals), {})
  subenv_vals, subenv_tree = tree_util.tree_flatten(subenv)
  yield subenv_vals, subenv_tree


def call_ildj(prim, invals, outvals, **params):
  """InverseAndILDJ rule for call primitives."""
  f, invals = invals[0], invals[1:]
  flat_vals, in_tree = tree_util.tree_flatten((invals, outvals))
  new_params = dict(params)
  if 'donated_invars' in params:
    new_params['donated_invars'] = (False,) * len(flat_vals)
  f, aux = flat_propagate(f, in_tree)
  subenv_vals = prim.bind(f, *flat_vals, **new_params)
  subenv_tree = aux()
  subenv = tree_util.tree_unflatten(subenv_tree, subenv_vals)
  new_invals = [subenv.read(var) for var in subenv.jaxpr.invars]
  new_outvals = [subenv.read(var) for var in subenv.jaxpr.outvars]
  done = all(not val.is_unknown() for val in new_invals + new_outvals)
  return new_invals, new_outvals, done, subenv
custom_rules[xla.xla_call_p] = jax_util.partial(call_ildj, xla.xla_call_p)
custom_rules[jax_core.call_p] = jax_util.partial(call_ildj, jax_core.call_p)
custom_rules[pe.remat_call_p] = jax_util.partial(call_ildj, pe.remat_call_p)
custom_rules[harvest.nest_p] = jax_util.partial(call_ildj, harvest.nest_p)


def map_ildj(prim, invals, outvals, **params):
  """InverseAndILDJ rule for the map primitives."""
  f, invals = invals[0], invals[1:]

  invals = [v if v.is_unknown() else InverseAndILDJ(v.val, np.broadcast_to(
      v.ildj, v.val.shape[:1])) for v in invals]
  outvals = [v if v.is_unknown() else InverseAndILDJ(v.val, np.broadcast_to(
      v.ildj, v.val.shape[:1])) for v in outvals]
  flat_vals, in_tree = tree_util.tree_flatten((invals, outvals))
  f, aux = flat_propagate(f, in_tree)
  # Assume all invars as mapped
  new_mapped_invars = (True,) * len(flat_vals)
  new_params = dict(params, mapped_invars=new_mapped_invars)
  subenv_vals = prim.bind(f, *flat_vals, **new_params)
  subenv_tree = aux()
  subenv = tree_util.tree_unflatten(subenv_tree, subenv_vals)
  new_invals = [subenv.read(var) for var in subenv.jaxpr.invars]
  new_outvals = [subenv.read(var) for var in subenv.jaxpr.outvars]
  new_invals = [v if v.is_unknown() else InverseAndILDJ(v.val,
                                                        np.sum(v.ildj, 0))
                for v in new_invals]
  new_outvals = [v if v.is_unknown() else InverseAndILDJ(v.val, 0.)
                 for v in new_outvals]
  done = all(not val.is_unknown() for val in new_invals + new_outvals)
  return new_invals, new_outvals, done, subenv
custom_rules[pxla.xla_pmap_p] = jax_util.partial(map_ildj, pxla.xla_pmap_p)


def sow_ildj(incells, outcells, **params):
  del params
  if (all(outcell.is_unknown() for outcell in outcells) and
      not any(incell.is_unknown() for incell in incells)):
    return incells, incells, True, None
  elif (not any(outcell.is_unknown() for outcell in outcells) and
        all(incell.is_unknown() for incell in incells)):
    return outcells, outcells, True, None
  return incells, outcells, False, None
ildj_registry[harvest.sow_p] = sow_ildj


def tie_all_ildj(incells, outcells, **params):
  """InverseAndILDJ rule for the tie_all primitive."""
  del params
  new_cells = []
  for incell, outcell in safe_zip(incells, outcells):
    if incell.is_unknown() and not outcell.is_unknown():
      new_cells.append(outcell)
    else:
      new_cells.append(incell)
  done = (not any(outcell.is_unknown() for outcell in outcells) and
          not any(incell.is_unknown() for incell in incells))
  return new_cells, new_cells, done, None
ildj_registry[primitive.tie_all_p] = tie_all_ildj
