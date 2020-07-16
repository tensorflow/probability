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
"""Module for the harvest transformation.

This module contains a general-purpose set of tools for transforming
functions with a specific side-effect mechanism into pure functions. The names
of the transformations in this module are inspired by the Sow/Reap mechanism in
Mathematica.

The harvest module exposes two main functions: `sow` and `harvest`. `sow` is
used to tag values and `harvest` can inject values into functions or pull out
tagged values.

`harvest` is a very general purpose transformation purely focused on converting
functions that have special side-effects (defined using `sow`) and
"functionalizing" them. Specifically, a function
`f :: X -> Y` has a set of defined intermediates, or `Sows`. This set
can be divided into intermediates you are "collecting" and intermediates you are
"injecting", or `Reaps` and `Plants` respectively. Functionalizing
`f` now gives you `harvest(f) :: (Plants, X) -> Y -> Reaps`. Generally, most
users will not need to use `harvest` directly, but will use wrappers around it.

## `sow`

`sow` is the function used to tag values in a function. It takes in a single
positional argument, `value`, which is returned as an output, so `sow` outside
of a tracing context behaves like the identity function, i.e.
`sow(x, ...) == x`. It also takes in two mandatory keyword arguments,
`tag` and `name`. `tag` is a string used to namespace intermediate values in a
function. For example, some intermediates may be useful for probabilistic
programming (samples), and others may be useful to logging (summaries). The tag
enables `harvest` to interact with only one set of intermediates at a time.
The `name` is a string that describes the value you are `sow`-ing. Eventually,
when calling `harvest` on a function, the `name` is used as the identifier
for the intermediate value.

Finally, `sow` takes in an optional string keyword argument `mode`, which is by
default set to `'strict'`. The `mode` of a `sow` describes how it behaves when
the same name appears multiple times. In "strict" mode, `sow` will error if the
same `(tag, name)` appears more than once. Another option is `'append'`, in
which all sows of the same name will be appended into a growing array. Finally,
there is `'clobber'`, where only the final sown value for a given `(tag, name)`
will be returned. The final optional argument for `sow` is `key`, which will
automatically be tied-in to the output of `sow` to introduce a fake
data-dependence. By default, it is `None`.

## `harvest`

`harvest` is a function transformation that augments the behaviors of `sow`s
in the function body. Recall, that by default, `sow`s act as identity functions
and do not affect the semantics of a function. Harvesting `f` produces a
function that can take advantage of `sow`s present in its execution. `harvest`
is a function that takes in a function `f` and a string `tag`. `harvest` will
only interact with `sow`s whose tag matches the input `tag`. The returned
function can interact with the `sow`s in the function body in either of two
ways. The first is via "injection", where intermediate values in the function
values can be overridden. `harvest(f)` takes in an additional initial argument,
`plants`, a dictionary mapping names to values. Each name in `plants` should
correspond to a `sow` in `f`, and while running `harvest(f)` rather than using
the value at runtime for the `sow`, we substitute in the value from the `plants`
dictionary. The other way in which `harvest(f)` interacts with `sow`s is that
if it encounters a `sow` whose tag matches and whose name is *not* in
`plants`, it will add the output of the `sow` to a dictionary mapping the sow
name to its output, called `reaps`. The `reaps` dictionary, at the end of
`harvest(f)`'s execution, will contain the outputs of all `sow`s whose values
were not injected, or "planted."

The general convention is that, for any given execution of
`harvest(f, tag=tag)`, there will be *no more remaining sows* of the given tag
if the function were to be reharvested, i.e. if we were to nest harvests with
the same tag `harvest(harvest(f, tag='some_tag'), tag='some_tag')`, the outer
harvest would have nothing to plant or to reap.

## Examples:

#### Using `sow` and `harvest`
```python
def f(x):
  y = sow(x + 1., tag='intermediate', name='y')
  return y + 1.

# Injecting, or "planting" a value for `y`.
harvest(f, tag='intermediate')({'y': 0.}, 1.)  # ==> (1., {})
harvest(f, tag='intermediate')({'y': 0.}, 5.)  # ==> (1., {})

# Collecting , or "reaping" the value of `y`.
harvest(f, tag='intermediate')({}, 1.)  # ==> (3., {'y': 2.})
harvest(f, tag='intermediate')({}, 5.)  # ==> (7., {'y': 6.})
```

#### Using `reap` and `plant`.
`reap` and `plant` are simple wrappers around `harvest`. `reap` only pulls
intermediate values without injecting, and `plant` only injects values without
collecting intermediate values.

```python
def f(x):
  y = sow(x + 1., tag='intermediate', name='y')
  return y + 1.

# Injecting, or "planting" a value for `y`.
plant(f, tag='intermediate')({'y': 0.}, 1.)  # ==> 1.
plant(f, tag='intermediate')({'y': 0.}, 5.)  # ==> 1.

# Collecting , or "reaping" the value of `y`.
reap(f, tag='intermediate')(1.)  # ==> {'y': 2.}
reap(f, tag='intermediate')(5.)  # ==> {'y': 6.}
```
"""
from typing import Any, Callable, Dict, Iterable, List, FrozenSet, Tuple, Union

import dataclasses
import jax
from jax import abstract_arrays
from jax import api_util
from jax import core as jax_core
from jax import lax
from jax import linear_util as lu
from jax import tree_util
from jax import util as jax_util
from jax.interpreters import ad
from jax.interpreters import batching
from jax.interpreters import masking
from jax.interpreters import xla
from jax.lib.xla_bridge import xla_client as xc
import jax.numpy as np

from oryx.core import primitive as prim
from oryx.core import trace_util

__all__ = [
    'HarvestTrace',
    'HarvestTracer',
    'sow',
    'harvest',
    'reap',
    'plant',
    'nest',
]

safe_map = jax_util.safe_map
safe_zip = jax_util.safe_zip


@dataclasses.dataclass(frozen=True)
class HarvestSettings:
  """Contains the settings for a HarvestTrace."""
  tag: str
  blocklist: FrozenSet[str]
  allowlist: Union[FrozenSet[str], None]


@dataclasses.dataclass
class HarvestContext:
  """Contains the settings and storage for the current trace in the stack."""
  settings: HarvestSettings
  reaps: Dict[str, Any]
  plants: Dict[str, Any]

  def __post_init__(self):
    self._already_planted = set()

  def handle_sow(self, values, *, name, tag, mode, tree):
    """Determines if a value should be planted or reaped and calls the appropriate handler."""
    if tag != self.settings.tag:
      return sow_p.bind(*values, name=name, tag=tag, mode=mode, tree=tree)
    if (self.settings.allowlist is not None and
        name not in self.settings.allowlist):
      return values
    if name in self.settings.blocklist:
      return values
    if name in self.plants:
      return self.handle_plant(values, name=name, mode=mode, tree=tree)
    else:
      return self.handle_reap(values, name=name, mode=mode, tree=tree)

  def handle_reap(self, values, *, name, mode, tree):
    """Stores values in the context."""
    unflat_values = tree_util.tree_unflatten(tree, values)
    if mode == 'strict' and name in self.reaps:
      raise ValueError(f'Variable has already been reaped: {name}')
    if mode == 'append':
      if name not in self.reaps:
        self.reaps[name] = HarvestList([])
      self.reaps[name].append(unflat_values)
    elif mode == 'clobber' or mode == 'strict':
      self.reaps[name] = unflat_values
    else:
      raise ValueError(f'Invalid sow mode: {mode}')
    return values

  def handle_plant(self, values, *, name, mode, **_):
    """Pulls values from the context."""
    del values
    if mode == 'strict' and name in self._already_planted:
      raise ValueError(f'Variable has already been planted: {name}')
    self._already_planted.add(name)
    values = self.plants[name]
    if mode == 'append':
      if not isinstance(values, HarvestList):
        values = self.plants[name] = HarvestList(values)
      values = values.pop()
    return tree_util.tree_leaves(values)


harvest_custom_rules = {}


class HarvestList:
  """Class used to store sows with mode = 'append'."""

  def __init__(self, data, size=0, idx=0):
    self.data = data
    self.size = size
    self.idx = idx

  def pop(self):
    out = self.data[self.idx]
    self.idx += 1
    return out

  def append(self, tracers):
    self.data.append(tracers)
    self.size += 1

  def as_array(self):
    return tree_util.tree_multimap(lambda *args: np.array(list(args)),
                                   *self.data)

  def flatten(self):
    return (self.data,), (self.size, self.idx)

  @classmethod
  def unflatten(cls, data, xs):
    size, idx = data
    return HarvestList(xs[0], size=size, idx=idx)


tree_util.register_pytree_node(HarvestList, HarvestList.flatten,
                               HarvestList.unflatten)


class HarvestTrace(jax_core.Trace):
  """A HarvestTrace manages HarvestTracer objects.

  Since HarvestTracers are just wrappers around known values, HarvestTrace
  just passes these values through primitives, except in the case of
  `sow` and `nest`, which are specially handled by the active HarvestContext.

  Default primitive logic lives in `process_primitive`, with special logic for
  `sow` in `handle_sow`.
  """

  def pure(self, val):
    return HarvestTracer(self, val)

  def sublift(self, tracer):
    return self.pure(tracer.val)

  def lift(self, val):
    return self.pure(val)

  def instantiate_const(self, val):
    if isinstance(val, HarvestTracer):
      return val
    return self.pure(val)

  def process_primitive(self, primitive, tracers, params):
    tracers = safe_map(self.instantiate_const, tracers)
    if primitive in harvest_custom_rules:
      return harvest_custom_rules[primitive](self, *tracers, **params)
    if primitive is sow_p:
      return self.handle_sow(*tracers, **params)
    vals = [t.val for t in tracers]
    outvals = primitive.bind(*vals, **params)
    if not primitive.multiple_results:
      outvals = [outvals]
    out_tracers = safe_map(self.pure, outvals)
    if primitive.multiple_results:
      return out_tracers
    return out_tracers[0]

  def handle_sow(self, *tracers, name, tag, mode, tree):
    vals = [t.val for t in tracers]
    context = trace_util.get_dynamic_context(self)
    return safe_map(
        self.pure,
        context.handle_sow(vals, name=name, tag=tag, mode=mode, tree=tree))

  def process_call(self, call_primitive, f, tracers, params):
    return self.process_higher_order_primitive(call_primitive, f, tracers,
                                               params, False)

  def process_map(self, call_primitive, f, tracers, params):
    return self.process_higher_order_primitive(call_primitive, f, tracers,
                                               params, True)

  def process_higher_order_primitive(self, primitive, f, tracers, params,
                                     is_map):
    name = params.pop('name', f.__name__)
    tracers = safe_map(self.instantiate_const, tracers)
    vals = [t.val for t in tracers]
    context = trace_util.get_dynamic_context(self)
    active_tag = context.settings.tag
    plants = context.plants
    if primitive is nest_p:
      plants = plants.get(params['scope'], {})
    if is_map:
      # TODO(sharadmv): figure out if invars are mapped or unmapped
      params = params.copy()
      new_params = dict(
          params,
          mapped_invars=(True,) * len(tree_util.tree_leaves(plants)) +
          params['mapped_invars'])
    else:
      new_params = dict(params)
    all_args, all_tree = tree_util.tree_flatten((plants, vals))
    num_plants = len(all_args) - len(vals)
    if 'donated_invars' in params:
      new_params['donated_invars'] = ((False,) * num_plants
                                      + params['donated_invars'])
    f, aux = harvest_eval(f, self, context.settings, all_tree)
    out_flat = primitive.bind(
        f, *all_args, **new_params, name=jax_util.wrap_name(name, 'harvest'))
    out_tree = aux()
    out, reaps = tree_util.tree_unflatten(out_tree, out_flat)
    out_tracers = safe_map(self.pure, out)
    reap_tracers = tree_util.tree_map(self.pure, reaps)
    if primitive is nest_p and reap_tracers:
      flat_tracers, tree = tree_util.tree_flatten(reap_tracers)
      self.handle_sow(
          *flat_tracers,
          name=params['scope'],
          tag=active_tag,
          mode='strict',
          tree=tree)
    else:
      for name, reap_tracer in reap_tracers.items():
        flat_tracers, tree = tree_util.tree_flatten(reap_tracer)
        self.handle_sow(
            *flat_tracers, name=name, tag=active_tag, mode='strict', tree=tree)
    return out_tracers

  def post_process_call(self, call_primitive, out_tracers, params):
    vals = tuple(t.val for t in out_tracers)
    master = self.master

    def todo(x):
      trace = HarvestTrace(master, jax_core.cur_sublevel())
      return safe_map(jax_util.partial(HarvestTracer, trace), x)

    return vals, todo

  post_process_map = post_process_call


class HarvestTracer(jax_core.Tracer):
  """A HarvestTracer just encapsulates a single value."""

  def __init__(self, trace: HarvestTrace, val):
    self._trace = trace
    self.val = val

  @property
  def aval(self):
    return abstract_arrays.raise_to_shaped(jax_core.get_aval(self.val))

  def full_lower(self):
    return self


@lu.transformation
def harvest_function(master: jax_core.MasterTrace, settings: HarvestSettings,
                     in_tree, args: Iterable[Any]):
  """A JAX linear_util transformation that runs a HarvestTrace."""
  trace = HarvestTrace(master, jax_core.cur_sublevel())
  plants, args = tree_util.tree_unflatten(in_tree, args)
  in_tracers = safe_map(trace.pure, args)
  context = HarvestContext(settings, {}, plants)
  with trace_util.new_dynamic_context(master, context):
    ans = yield in_tracers, {}
    out_tracers = safe_map(trace.full_raise, ans)
    reaps = tree_util.tree_map(trace.full_raise, context.reaps)
    del master
  reaped_tracers = {}
  for key, reaped_tracer in reaps.items():
    if isinstance(reaped_tracer, HarvestList):
      reaped_tracers[key] = reaped_tracer.as_array()
    else:
      reaped_tracers[key] = reaped_tracer
  yield ([t.val for t in out_tracers],
         tree_util.tree_map(lambda t: t.val, reaped_tracers))


def harvest_eval(f: lu.WrappedFun, trace: HarvestTrace,
                 settings: HarvestSettings,
                 all_tree) -> Tuple[lu.WrappedFun, Callable[[], Any]]:
  f = harvest_function(f, trace.master, settings, all_tree)
  return harvest_wrapper(f, trace)


@lu.transformation_with_aux
def harvest_wrapper(trace: HarvestTrace, *args):
  del trace
  out, reaps = yield (args,), {}
  out_flat, out_tree = tree_util.tree_flatten((out, reaps))
  yield out_flat, out_tree


sow_p = jax_core.Primitive('sow')
sow_p.multiple_results = True


def _sow_impl(*args, **_):
  return args


sow_p.def_impl(_sow_impl)


def _sow_abstract_eval(*avals, **_):
  return avals


sow_p.def_abstract_eval(_sow_abstract_eval)


def _sow_transpose(cts_in, *_, **__):
  return cts_in


ad.deflinear(sow_p, _sow_transpose)


def _sow_batch_rule(batched_args, batch_dims, **params):
  outs = sow_p.bind(*batched_args, **params)
  return outs, batch_dims


batching.primitive_batchers[sow_p] = _sow_batch_rule
xla.translations[sow_p] = lambda c, *args, **params: xc.ops.Tuple(c, args)

nest_p = jax_core.CallPrimitive('nest')


def _nest_impl(f, *args, **_):
  return f.call_wrapped(*args)


nest_p.def_impl(_nest_impl)


def _nest_translation_rule(*args, backend, name, call_jaxpr, scope, **_):
  return xla._xla_call_translation_rule(  # pylint: disable=protected-access
      *args,
      name=jax_util.wrap_name(name, f'nest[{scope}]'),
      backend=backend,
      call_jaxpr=call_jaxpr,
      donated_invars=(False,) * len(args))


xla.call_translations[nest_p] = _nest_translation_rule


def _nest_transpose_rule(*args, **kwargs):
  return ad.call_transpose(nest_p, *args, **kwargs)


ad.primitive_transposes[nest_p] = _nest_transpose_rule


def sow(value, *, tag, name, mode='strict', key=None):
  """Marks a value with a name and a tag.

  Args:
    value: A JAX value to be tagged and named.
    tag (str): a string representing the tag of the sown value.
    name (str): a string representing the name to sow the value with.
    mode (str): The mode by which to sow the value. There are three options: 1.
      strict - if another value is sown with the same name and tag in the same
      context, harvest will throw an error. 2. clobber - if another is value is
      sown with the same name and tag, it will replace this value 3. append -
      sown values of the same name and tag are appended to a growing list.
      Append mode assumes some ordering on the values being sown defined by
      data-dependence.
    key: an optional JAX value that will be tied into the sown value.

  Returns:
    The original `value` that was passed in.
  """
  if key is not None:
    value = prim.tie_in(key, value)
  flat_args, in_tree = tree_util.tree_flatten(value)
  out_flat = sow_p.bind(*flat_args, name=name, tag=tag, mode=mode, tree=in_tree)
  return tree_util.tree_unflatten(in_tree, out_flat)


def harvest(f,
            *,
            tag: str,
            allowlist: Union[Iterable[str], None] = None,
            blocklist: Iterable[str] = frozenset()):
  """Transforms a function into a "functionalized" version.

  Sown values are namespaced using string "tags", where a value is sown (using
  `sow`) with a tag, and `harvest` will ignore any sown values that don't match
  its input tag. Harvest will take a function `f :: X -> Y` that has sown values
  and converts it into a function `g :: Plants -> X -> (Y, Reaps)`.

  The additional input to the function, called `plants`, are values that are
  injected into the function. `plants` is a dictionary mapping string names
  to values, and while `f` is being run, if a key in `plants` matches the
  name of a sown value, the value in `plants` is used instead of the sown value.

  The additional output of the function, called `reaps`, are values that are
  collected from the function. `reaps` is a dictionary mapping string names
  to values, and while `f` is being run, if the name of a sown value is not
  in `plants`, it is added to the `reaps` dictionary and returned along
  with the original output of the function. A value can only be reaped if it
  is not also planted.

  Args:
    f: a function to be transformed.
    tag: `str`, the harvest tag that will be reaped/planted.
    allowlist: an iterable of strings of names that will be planted/reaped where
      other names will be ignored.
    blocklist: an iterable of strings of names that will be ignored while
      planting/reaping.

  Returns:
    A function that takes in an additional initial input (a dictionary mapping
    names to values to be injected) and an additional final output (a
    dictionary mapping names to values that were collected).
  """
  blocklist = frozenset(blocklist)
  if allowlist is not None:
    allowlist = frozenset(allowlist)
  settings = HarvestSettings(tag, blocklist, allowlist)

  def wrapped(plants, *args, **kwargs):
    fun = lu.wrap_init(f, kwargs)
    flat_args, in_tree = tree_util.tree_flatten(args)
    flat_fun, out_tree = api_util.flatten_fun_nokwargs(fun, in_tree)
    all_args, all_tree = tree_util.tree_flatten((plants, flat_args))
    with jax_core.new_master(HarvestTrace) as master:
      flat_fun = harvest_function(flat_fun, master, settings, all_tree)
      out_flat, reaped = flat_fun.call_wrapped(all_args)
      del master
    out = tree_util.tree_unflatten(out_tree(), out_flat)
    return out, reaped

  return wrapped


def reap(f, *, tag, **harvest_kwargs):
  """Collects tagged values from a function.

  Transforms a function to return the original output and intermediate collected
  values. In implementation, returns partial(harvest(f), {}). See `harvest`
  for more details.

  Args:
    f: a function to be transformed
    tag: `str`, the harvest tag that will be reaped.
    **harvest_kwargs: additional keyword arguments that will be passed to
      `harvest`.

  Returns:
    A function that returns tagged values (a dictionary mapping
    names to values that were collected).
  """

  def wrapped(*args, **kwargs):
    return harvest(f, tag=tag, **harvest_kwargs)({}, *args, **kwargs)[1]

  return wrapped


def plant(f, *, tag, **harvest_kwargs):
  """Injects tagged values into a function.

  Transforms a function to one where tagged values can injected. In
  implementation, returns a function that takes plants as an additional
  initial argument.

  Args:
    f: a function to be transformed
    tag: `str`, the harvest tag that will be planted.
    **harvest_kwargs: additional keyword arguments that will be passed to
      `harvest`.

  Returns:
    A function that takes in an additional initial input (a dictionary mapping
    names to values to be injected).
  """

  def wrapped(plants, *args, **kwargs):
    return harvest(f, tag=tag, **harvest_kwargs)(plants, *args, **kwargs)[0]

  return wrapped


def nest(f, *, scope):
  """Wraps a function to create a new scope for harvested values.

  Harvested values live in one dynamic name scope (for a particular tag),
  and in strict mode, values with the same name cannot be collected or injected
  more than once. nest(f, scope=<name>) will take all tagged values in `f` and
  put them into a nested dictionary with key <name>. This enables having
  duplicate names in one namespace provided they are in different scopes. This
  is different from using a separate tag to namespace, as it enables creating
  nested/hierarchical structure within a single tag's namespace.

  Example:
  ```python
  def foo(x):
    return sow(x, tag='test', name='x')
  harvest(foo, tag='test')({}, 1.)  # (1., {'x': 1.})
  harvest(nest(foo, scope='a'), tag='test')({}, 1.)  # (1., {'a': {'x': 1.}})
  ```

  Args:
    f: a function to be transformed
    scope (str): a string that will act as the parent scope of all values tagged
      in `f`.

  Returns:
    A semantically identical function to `f`, but when harvested, uses nested
    values according to the input scope.
  """

  def wrapped(*args, **kwargs):
    fun = lu.wrap_init(f, kwargs)
    flat_args, in_tree = tree_util.tree_flatten(args)
    flat_fun, out_tree = api_util.flatten_fun_nokwargs(fun, in_tree)
    out_flat = nest_p.bind(flat_fun, *flat_args, scope=scope, mode='strict',
                           name=getattr(f, '__name__', '<no name>'))
    return tree_util.tree_unflatten(out_tree(), out_flat)

  return wrapped


def _find_sows(typed_jaxpr: jax_core.TypedJaxpr,
               tag: str) -> List[Dict[str, Any]]:
  sows = []
  for eqn in typed_jaxpr.jaxpr.eqns:
    # TODO(sharadmv): handle nested Jaxprs
    if eqn.primitive is sow_p:
      sow_tag = eqn.params['tag']
      if sow_tag == tag:
        sows.append(eqn.params)
  return sows


def _scan_harvest_rule(trace: HarvestTrace, *tracers, length, reverse, jaxpr,
                       num_consts, num_carry, linear, unroll):
  """Collects and injects values into/from the scan body."""
  context = trace_util.get_dynamic_context(trace)
  settings = context.settings
  values = [t.val for t in tracers]
  consts, init, xs = jax_util.split_list(values, [num_consts, num_carry])

  active_sows = _find_sows(jaxpr, settings.tag)
  active_modes = [params['mode'] for params in active_sows]
  if any(mode == 'strict' for mode in active_modes):
    raise ValueError('Cannot use strict mode in a scan.')
  active_names = [params['name'] for params in active_sows]
  sow_modes = {name: mode for name, mode in zip(active_names, active_modes)}
  carry_plants = {
      name: context.plants[name]
      for name in active_names
      if name in context.plants and sow_modes[name] == 'clobber'
  }
  xs_plants = {
      name: context.plants[name]
      for name in active_names
      if name in context.plants and sow_modes[name] == 'append'
  }

  def jaxpr_fun(carry, x):
    body_out = jax_core.eval_jaxpr(jaxpr.jaxpr, [], *(consts + carry + x))
    carry, y = jax_util.split_list(body_out, [num_carry])
    return carry, y

  harvest_body = harvest(
      jaxpr_fun,
      tag=settings.tag,
      allowlist=settings.allowlist,
      blocklist=settings.blocklist)

  def body(carry, x):
    x_plants, x_vals = x
    (carry, y), reaps = harvest_body({
        **carry_plants,
        **x_plants
    }, carry, x_vals)
    return carry, (y, reaps)

  xs_flat = tree_util.tree_leaves((xs_plants, xs))
  x_avals = []
  for x in xs_flat:
    x_aval = jax_core.get_aval(x)
    if x_aval is jax_core.abstract_unit:
      x_avals.append(x_aval)
    else:
      x_shape, x_dtype = masking.padded_shape_as_value(x.shape[1:]), x.dtype
      x_avals.append(abstract_arrays.ShapedArray(x_shape, x_dtype))
  x_avals = tuple(x_avals)
  init_avals = tuple(
      abstract_arrays.raise_to_shaped(jax_core.get_aval(a)) for a in init)
  in_flat, in_tree = tree_util.tree_flatten((init, (xs_plants, xs)))
  body_jaxpr, new_consts, out_tree = (
      jax.lax.lax_control_flow._initial_style_jaxpr(  # pylint: disable=protected-access
          body, in_tree, init_avals + x_avals))
  new_values = list(new_consts) + in_flat
  num_xs_plants = len(new_values) - len(init) - len(xs) - len(new_consts)
  remaining_linear = linear[num_consts:]
  new_linear = ((False,) * len(new_consts) + remaining_linear[:len(init)] +
                (False,) * num_xs_plants + remaining_linear[len(init):])
  assert len(new_linear) == len(new_values)

  outs = lax.scan_p.bind(
      *new_values,
      length=length,
      reverse=reverse,
      jaxpr=body_jaxpr,
      num_consts=len(new_consts),
      num_carry=num_carry,
      linear=new_linear,
      unroll=unroll)
  outs = safe_map(trace.pure, outs)
  carry, (ys, reaps) = tree_util.tree_unflatten(out_tree, outs)
  out_reaps = {}
  for k, val in reaps.items():
    mode = sow_modes.get(k, 'strict')
    if mode == 'append':
      val = tree_util.tree_map(np.concatenate, val)
    elif mode == 'clobber':
      val = tree_util.tree_map(lambda x: x[-1], val)
    out_reaps[k] = sow(val, tag=settings.tag, name=k, mode='strict')
  (carry, ys) = prim.tie_in(out_reaps, (carry, ys))
  return carry + ys


harvest_custom_rules[lax.scan_p] = _scan_harvest_rule
