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
from jax import abstract_arrays
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

__all__ = [
    'InverseAndILDJ',
    'inverse_and_ildj',
    'inverse',
    'ildj',
    'register_elementwise',
    'ildj_registry',
]

safe_map = jax_core.safe_map
safe_zip = jax_core.safe_zip
Cell = propagate.Cell


class InverseAndILDJ(Cell):
  """Propagates inverse values and their ILDJs."""

  def __init__(self, aval, val, ildj):  # pylint: disable=redefined-outer-name
    super().__init__(aval)
    self.val = val
    self.ildj = ildj

  def top(self) -> bool:
    return self.val is not None

  def bottom(self) -> bool:
    return self.val is None

  def __lt__(self, other: Cell) -> bool:
    return other.top() and self.bottom()

  def join(self, other: Cell) -> Cell:
    if other.top():
      return other
    return self

  def __repr__(self):
    return 'InverseAndILDJ({}, {})'.format(self.val, self.ildj)

  @classmethod
  def new(cls, val):
    aval = jax_core.get_aval(val)
    if aval is jax_core.abstract_unit:
      return cls.unknown(aval)
    aval = abstract_arrays.raise_to_shaped(aval)
    return InverseAndILDJ(aval, val, np.array(0.))

  @classmethod
  def unknown(cls, aval):
    return InverseAndILDJ(aval, None, None)

  def flatten(self):
    return (self.val, self.ildj), (self.aval,)

  @classmethod
  def unflatten(cls, data, xs):
    return InverseAndILDJ(data[0], xs[0], xs[1])


def inverse_and_ildj(f, *trace_args):
  """Inverse and ILDJ function transformation."""
  def wrapped(*args, **kwargs):
    """Function wrapper that takes in inverse arguments."""
    forward_args = trace_args if len(trace_args) else args
    jaxpr, (in_tree, _) = trace_util.stage(f)(*forward_args, **kwargs)
    flat_forward_args, _ = tree_util.tree_flatten(forward_args)
    flat_args, _ = tree_util.tree_flatten(args)
    flat_constcells = safe_map(InverseAndILDJ.new, jaxpr.literals)
    flat_forward_avals = [
        trace_util.get_shaped_aval(arg)
        for arg in flat_forward_args]
    flat_incells = [InverseAndILDJ.unknown(aval) for aval in flat_forward_avals]
    flat_outcells = safe_map(InverseAndILDJ.new, flat_args)
    env = propagate.propagate(InverseAndILDJ, ildj_registry, jaxpr.jaxpr,
                              flat_constcells, flat_incells, flat_outcells)
    flat_incells = [env.read(invar) for invar in jaxpr.jaxpr.invars]
    if any(not flat_incell.top() for flat_incell in flat_incells):
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
  """Returns the inverse of a function."""
  def wrapped(*args, **kwargs):
    return inverse_and_ildj(f, *trace_args)(*args, **kwargs)[0]
  return wrapped


def ildj(f, *trace_args):
  """Computes the log determininant of a function's Jacobian."""
  def wrapped(*args, **kwargs):
    return inverse_and_ildj(f, *trace_args)(*args, **kwargs)[1]
  return wrapped


def default_rule(prim, invals, outvals, **params):
  """Default inversion rule that only does forward eval."""
  if all(outval.bottom() for outval in outvals):
    if all(inval.top() for inval in invals):
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
    return invals, outvals, None
  if any(outval.bottom() for outval in outvals):
    return invals, outvals, None
  raise NotImplementedError(f'No registered inverse for `{prim}`.')


class InverseDict(object):
  """Default rules dictionary that uses a default rule for inverse."""

  def __init__(self):
    self.rules = {}

  def __getitem__(self, prim):
    if prim not in self.rules:
      self[prim] = jax_util.partial(default_rule, prim)
    return self.rules[prim]

  def __setitem__(self, prim, val):
    self.rules[prim] = val


def register_elementwise(prim):
  """Registers an elementwise primitive with ILDJ."""
  def make_rule(f):
    """Accepts an inverse function for a primitive."""
    def ildj_rule(incells, outcells, **params):
      """General InverseAndILDJ rule for elementwise functions."""
      outcell, = outcells
      incell, = incells
      if not incell.top() and outcell.top():
        val = outcell.val
        f_sum = lambda x: f(x).sum()
        incells = [InverseAndILDJ(outcell.aval, f(val), outcell.ildj +
                                  np.log(jax.grad(f_sum)(val)).sum())]
      elif not outcell.top() and incell.top():
        outcells = [InverseAndILDJ.new(prim.bind(incell.val, **params))]
      return incells, outcells, None
    ildj_registry[prim] = ildj_rule
  return make_rule


def register_binary(prim):
  """Registers an binary primitive with ILDJ."""
  def make_rule(f_left, f_right):
    def ildj_rule(incells, outcells, **params):
      outcell, = outcells
      left, right = incells
      if not outcell.bottom():
        val, ildj_ = outcell.val, outcell.ildj
        if not left.bottom():
          right_val, right_ildj = f_left(left.val, val, ildj_)
          incells = [left, InverseAndILDJ(right.aval, right_val, right_ildj)]
        elif not right.bottom():
          left_val, left_ildj = f_right(right.val, val, ildj_)
          incells = [InverseAndILDJ(left.aval, left_val, left_ildj), right]
      elif (outcell.bottom() and not left.bottom() and
            not right.bottom()):
        out_val = prim.bind(left.val, right.val, **params)
        outcells = [InverseAndILDJ.new(out_val)]
      return incells, outcells, None
    ildj_registry[prim] = ildj_rule
  return make_rule


ildj_registry = InverseDict()
register_elementwise(lax.exp_p)(np.log)
register_elementwise(lax.log_p)(np.exp)
register_elementwise(lax.sin_p)(np.arcsin)
register_elementwise(lax.cos_p)(np.arccos)
register_elementwise(lax.expm1_p)(np.log1p)
register_elementwise(lax.log1p_p)(np.expm1)
register_elementwise(lax.neg_p)(lambda x: -x)


def add_left(left_val, out_val, ildj_):
  return out_val - left_val, ildj_


def add_right(right_val, out_val, ildj_):
  return out_val - right_val, ildj_
register_binary(lax.add_p)(add_left, add_right)


def sub_left(left_val, out_val, ildj_):
  return left_val - out_val, ildj_


def sub_right(right_val, out_val, ildj_):
  return out_val + right_val, ildj_
register_binary(lax.sub_p)(sub_left, sub_right)


def mul_left(left_val, out_val, ildj_):
  return out_val / left_val, -np.log(np.abs(left_val)).sum() + ildj_


def mul_right(right_val, out_val, ildj_):
  return out_val / right_val, -np.log(np.abs(right_val)).sum() + ildj_
register_binary(lax.mul_p)(mul_left, mul_right)


def div_left(left_val, out_val, ildj_):
  return left_val / out_val, (
      (np.log(left_val) - 2 * np.log(out_val)).sum() + ildj_)


def div_right(right_val, out_val, ildj_):
  return out_val * right_val, np.log(np.abs(right_val)).sum() + ildj_
register_binary(lax.div_p)(div_left, div_right)


@lu.transformation_with_aux
def flat_propagate(tree, *flat_invals):
  invals, outvals = tree_util.tree_unflatten(tree, flat_invals)
  subenv = yield ((invals, outvals), {})
  subenv_vals, subenv_tree = tree_util.tree_flatten(subenv)
  yield subenv_vals, subenv_tree


def call_ildj(prim, incells, outcells, **params):
  """InverseAndILDJ rule for call primitives."""
  f, incells = incells[0], incells[1:]
  flat_vals, in_tree = tree_util.tree_flatten((incells, outcells))
  new_params = dict(params)
  if 'donated_invars' in params:
    new_params['donated_invars'] = (False,) * len(flat_vals)
  f, aux = flat_propagate(f, in_tree)
  subenv_vals = prim.bind(f, *flat_vals, **new_params)
  subenv_tree = aux()
  subenv = tree_util.tree_unflatten(subenv_tree, subenv_vals)
  new_incells = [subenv.read(var) for var in subenv.jaxpr.invars]
  new_outcells = [subenv.read(var) for var in subenv.jaxpr.outvars]
  return new_incells, new_outcells, subenv
ildj_registry[xla.xla_call_p] = jax_util.partial(call_ildj, xla.xla_call_p)
ildj_registry[jax_core.call_p] = jax_util.partial(call_ildj, jax_core.call_p)
ildj_registry[pe.remat_call_p] = jax_util.partial(call_ildj, pe.remat_call_p)
ildj_registry[harvest.nest_p] = jax_util.partial(call_ildj, harvest.nest_p)


def map_ildj(prim, incells, outcells, **params):
  """InverseAndILDJ rule for the map primitives."""
  f, incells = incells[0], incells[1:]

  def slice_aval(aval):
    return abstract_arrays.ShapedArray(aval.shape[1:], aval.dtype,
                                       aval.weak_type)

  mapped_incells = [
      v if v.bottom() else InverseAndILDJ(
          slice_aval(v.aval), v.val, np.broadcast_to(v.ildj, v.aval.shape[:1]))
      for v in incells
  ]
  mapped_outcells = [
      v if v.bottom() else InverseAndILDJ(
          slice_aval(v.aval), v.val, np.broadcast_to(v.ildj, v.aval.shape[:1]))
      for v in outcells
  ]
  flat_vals, in_tree = tree_util.tree_flatten((mapped_incells, mapped_outcells))
  f, aux = flat_propagate(f, in_tree)
  # Assume all invars as mapped
  new_mapped_invars = (True,) * len(flat_vals)
  new_params = dict(params, mapped_invars=new_mapped_invars)
  subenv_vals = prim.bind(f, *flat_vals, **new_params)
  subenv_tree = aux()
  subenv = tree_util.tree_unflatten(subenv_tree, subenv_vals)
  new_incells = [subenv.read(var) for var in subenv.jaxpr.invars]
  new_outcells = [subenv.read(var) for var in subenv.jaxpr.outvars]
  new_incells = [v if v.bottom() else InverseAndILDJ(
      old_v.aval, v.val, np.sum(v.ildj, 0))
                 for old_v, v in safe_zip(incells, new_incells)]
  new_outcells = [v if v.bottom() else InverseAndILDJ(
      old_v.aval, v.val, 0.)
                  for old_v, v in safe_zip(outcells, new_outcells)]
  return new_incells, new_outcells, subenv
ildj_registry[pxla.xla_pmap_p] = jax_util.partial(map_ildj, pxla.xla_pmap_p)


def sow_ildj(incells, outcells, **params):
  del params
  new_cells = [incell.join(outcell) for incell, outcell
               in safe_zip(incells, outcells)]
  return new_cells, new_cells, None
ildj_registry[harvest.sow_p] = sow_ildj
ildj_registry[primitive.tie_all_p] = sow_ildj
