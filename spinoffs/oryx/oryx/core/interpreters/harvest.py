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
`f :: (x: X) -> Y` has a set of defined intermediates, or `Sows`. This set
can be divided into intermediates you are "collecting" and intermediates you are
"injecting", or `Reaps` and `Plants` respectively. Functionalizing
`f` now gives you `harvest(f) :: (plants: Plants, x: X) -> Tuple[Y, Reaps]`.
Generally, most users will not need to use `harvest` directly, but will use
wrappers around it.

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

#### Sharp edges

* `harvest` has undefined semantics under autodifferentiation. If a function
  you're taking the gradient of has a `sow`, it might produce unintuitive
  results when harvested. To better control gradient semantics, you can use
  `jax.custom_jvp` or `jax.custom_vjp`. The current implementation sows primals
  and tangents in the JVP but ignore cotangents in the VJP. These particular
  semantics are subject to change.
* Planting values into a `pmap` is partially working. Harvest tries to plant all
  the values, assuming they have a leading map dimension.
"""
import collections
import dataclasses
import functools

from typing import Any, Callable, Dict, FrozenSet, Iterable, List, Optional, Tuple, Union, Hashable

from jax import abstract_arrays
from jax import api_util
from jax import core as jax_core
from jax import lax
from jax import linear_util as lu
from jax import tree_util
from jax import util as jax_util
from jax._src.lax import control_flow as lcf
from jax.interpreters import ad
from jax.interpreters import batching
from jax.interpreters import mlir
from jax.interpreters import partial_eval as pe
from jax.interpreters import xla
from jax.lib import xla_client as xc
import jax.numpy as jnp

from oryx.core import primitive as prim
from oryx.core import trace_util

__all__ = [
    'HarvestTrace',
    'HarvestTracer',
    'call_and_reap',
    'harvest',
    'nest',
    'plant',
    'reap',
    'sow',
]

Value = Any

sow_p = jax_core.Primitive('sow')
sow_p.multiple_results = True


@sow_p.def_impl
def _sow_impl(*args, **_):
  return args


@sow_p.def_abstract_eval
def _sow_abstract_eval(*avals, **_):
  return avals


@functools.partial(ad.deflinear, sow_p)
def _sow_transpose(cts_in, *_, **__):
  return cts_in


def _sow_batch_rule(batched_args, batch_dims, **params):
  outs = sow_p.bind(*batched_args, **params)
  return outs, batch_dims


batching.primitive_batchers[sow_p] = _sow_batch_rule
xla.translations[sow_p] = lambda c, *args, **params: xc.ops.Tuple(c, args)


def sow(value, *, tag: Hashable, name: str, mode: str = 'strict', key=None):
  """Marks a value with a name and a tag.

  Args:
    value: A JAX value to be tagged and named.
    tag: a string representing the tag of the sown value.
    name: a string representing the name to sow the value with.
    mode: The mode by which to sow the value. There are three options: 1.
      `'strict'` - if another value is sown with the same name and tag in the
      same context, harvest will throw an error. 2. `'clobber'` - if another is
      value is sown with the same name and tag, it will replace this value 3.
      `'append'` - sown values of the same name and tag are appended to a
      growing list. Append mode assumes some ordering on the values being sown
      defined by data-dependence.
    key: an optional JAX value that will be tied into the sown value.

  Returns:
    The original `value` that was passed in.
  """
  if key is not None:
    value = prim.tie_in(key, value)
  flat_args, in_tree = tree_util.tree_flatten(value)
  out_flat = sow_p.bind(*flat_args, name=name, tag=tag, mode=mode, tree=in_tree)
  return tree_util.tree_unflatten(in_tree, out_flat)


nest_p = jax_core.CallPrimitive('nest')


def _nest_impl(f, *args, **_):
  with jax_core.new_sublevel():
    return f.call_wrapped(*args)


nest_p.def_impl(_nest_impl)


def _nest_lowering(ctx, *args, name, call_jaxpr, scope, **_):
  return mlir._xla_call_lower(  # pylint: disable=protected-access
      ctx,
      *args,
      name=jax_util.wrap_name(name, f'nest[{scope}]'),
      call_jaxpr=call_jaxpr,
      donated_invars=(False,) * len(args))


mlir.register_lowering(nest_p, _nest_lowering)


def _nest_transpose_rule(*args, **kwargs):
  return ad.call_transpose(nest_p, *args, **kwargs)


ad.primitive_transposes[nest_p] = _nest_transpose_rule


def nest(f, *, scope: str):
  """Wraps a function to create a new scope for harvested values.

  Harvested values live in one dynamic name scope (for a particular tag),
  and in strict mode, values with the same name cannot be collected or injected
  more than once. `nest(f, scope=[name])` will take all tagged values in `f` and
  put them into a nested dictionary with key `[name]`. This enables having
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
    scope: a string that will act as the parent scope of all values tagged
      in `f`.

  Returns:
    A semantically identical function to `f`, but when harvested, uses nested
    values according to the input scope.
  """

  def wrapped(*args, **kwargs):
    fun = lu.wrap_init(f, kwargs)
    flat_args, in_tree = tree_util.tree_flatten(args)
    flat_fun, out_tree = api_util.flatten_fun_nokwargs(fun, in_tree)
    out_flat = nest_p.bind(
        flat_fun,
        *flat_args,
        scope=scope,
        name=getattr(f, '__name__', '<no name>'))
    return tree_util.tree_unflatten(out_tree(), out_flat)

  return wrapped


class HarvestTrace(jax_core.Trace):
  """An evaluating trace that dispatches to a dynamic context."""

  def pure(self, val: Value) -> 'HarvestTracer':
    return HarvestTracer(self, val)

  def sublift(self, tracer: 'HarvestTracer') -> 'HarvestTracer':
    return self.pure(tracer.val)

  def lift(self, val: Value) -> 'HarvestTracer':
    return self.pure(val)

  def process_primitive(
      self, primitive: jax_core.Primitive, tracers: List['HarvestTracer'],
      params: Dict[str, Any]) -> Union['HarvestTracer', List['HarvestTracer']]:
    context = trace_util.get_dynamic_context(self)
    custom_rule = context.get_custom_rule(primitive)
    if custom_rule:
      return custom_rule(self, *tracers, **params)
    return self.default_process_primitive(primitive, tracers, params)

  def default_process_primitive(
      self, primitive: jax_core.Primitive, tracers: List['HarvestTracer'],
      params: Dict[str, Any]) -> Union['HarvestTracer', List['HarvestTracer']]:
    context = trace_util.get_dynamic_context(self)
    vals = [t.val for t in tracers]
    if primitive is sow_p:
      outvals = context.process_sow(*vals, **params)
      return jax_util.safe_map(self.pure, outvals)
    outvals = primitive.bind(*vals, **params)
    if not primitive.multiple_results:
      outvals = [outvals]
    out_tracers = jax_util.safe_map(self.pure, outvals)
    if primitive.multiple_results:
      return out_tracers
    return out_tracers[0]

  def process_call(self, call_primitive: jax_core.Primitive, f: Any,
                   tracers: List['HarvestTracer'], params: Dict[str, Any]):
    context = trace_util.get_dynamic_context(self)
    if call_primitive is nest_p:
      return context.process_nest(self, f, *tracers, **params)
    return context.process_higher_order_primitive(self, call_primitive, f,
                                                  tracers, params, False)

  def post_process_call(self, call_primitive, out_tracers, params):
    vals = tuple(t.val for t in out_tracers)
    master = self.main

    def todo(x):
      trace = HarvestTrace(master, jax_core.cur_sublevel())
      return jax_util.safe_map(functools.partial(HarvestTracer, trace), x)

    return vals, todo

  def process_map(self, call_primitive: jax_core.Primitive, f: Any,
                  tracers: List['HarvestTracer'], params: Dict[str, Any]):
    context = trace_util.get_dynamic_context(self)
    return context.process_higher_order_primitive(self, call_primitive, f,
                                                  tracers, params, True)

  post_process_map = post_process_call

  def process_custom_jvp_call(self, primitive, fun, jvp, tracers):
    # This implementation just drops the custom derivative rule.
    # TODO(mattjj,sharadmv): don't drop the custom derivative rule
    del primitive, jvp  # Unused.
    return fun.call_wrapped(*tracers)

  def process_custom_vjp_call(self, primitive, fun, fwd, bwd, tracers,
                              out_trees):
    # This implementation just drops the custom derivative rule.
    # TODO(mattjj,sharadmv): don't drop the custom derivative rule
    del primitive, fwd, bwd, out_trees  # Unused.
    return fun.call_wrapped(*tracers)


class HarvestTracer(jax_core.Tracer):
  """A `HarvestTracer` just encapsulates a single value."""

  def __init__(self, trace: 'HarvestTrace', val: Value):
    self._trace = trace
    self.val = val

  @property
  def aval(self):
    return abstract_arrays.raise_to_shaped(jax_core.get_aval(self.val))

  def full_lower(self):
    return self


@dataclasses.dataclass(frozen=True)
class HarvestSettings:
  """Contains the settings for a HarvestTrace."""
  tag: str
  blocklist: FrozenSet[str]
  allowlist: Union[FrozenSet[str], None]
  exclusive: bool


@dataclasses.dataclass
class HarvestContext:
  """A context that handles `sow`s and `nest`s in a `HarvestTrace`."""
  settings: HarvestSettings

  def process_sow(self, *values, name, tag, mode, tree):
    """Handles a `sow` primitive in a `HarvestTrace`."""
    if mode not in {'strict', 'append', 'clobber'}:
      raise ValueError(f'Invalid mode: {mode}')
    if tag != self.settings.tag:
      if self.settings.exclusive:
        return values
      return sow_p.bind(*values, name=name, tag=tag, tree=tree, mode=mode)
    if (self.settings.allowlist is not None and
        name not in self.settings.allowlist):
      return values
    if name in self.settings.blocklist:
      return values
    return self.handle_sow(*values, name=name, tag=tag, tree=tree, mode=mode)

  def get_custom_rule(self, primitive):
    raise NotImplementedError

  def handle_sow(self, *values, name, tag, mode, tree):
    raise NotImplementedError

  def process_nest(self, trace, f, *tracers, scope, name):
    raise NotImplementedError

  def process_higher_order_primitive(self, trace: HarvestTrace,
                                     call_primitive: jax_core.Primitive, f: Any,
                                     tracers: List['HarvestTracer'],
                                     params: Dict[str, Any], is_map: bool):
    raise NotImplementedError


reap_custom_rules = {}


@dataclasses.dataclass
class Reap:
  value: Any
  metadata: Dict[str, Any]


@dataclasses.dataclass
class ReapContext(HarvestContext):
  """Contains the settings and storage for the current trace in the stack."""
  settings: HarvestSettings
  reaps: Dict[str, Reap]

  def get_custom_rule(self, primitive):
    return reap_custom_rules.get(primitive)

  def handle_sow(self, *values, name, tag, tree, mode):
    """Stores a sow in the reaps dictionary."""
    del tag
    if name in self.reaps:
      raise ValueError(f'Variable has already been reaped: {name}')
    avals = tree_util.tree_unflatten(
        tree,
        [abstract_arrays.raise_to_shaped(jax_core.get_aval(v)) for v in values])
    self.reaps[name] = Reap(
        tree_util.tree_unflatten(tree, values), dict(mode=mode, aval=avals))
    return values

  def reap_higher_order_primitive(self, trace, call_primitive, f, tracers,
                                  params, is_map):
    """Wraps the inner function with a reap trace."""
    name = jax_util.wrap_name(params.pop('name', f.__name__), 'reap')
    vals = [t.val for t in tracers]
    f, aux = reap_eval(f, trace, self.settings)

    if is_map:
      out_axes_thunk = params['out_axes_thunk']

      @jax_util.as_hashable_function(closure=('harvest', out_axes_thunk))
      def new_out_axes_thunk():
        out_axes = out_axes_thunk()
        assert all(out_axis == 0 for out_axis in out_axes)
        out_tree, _ = aux()
        return (0,) * out_tree.num_leaves

      params = dict(params, out_axes_thunk=new_out_axes_thunk)
    out_flat = call_primitive.bind(f, *vals, name=name, **params)
    out_tree, metadata = aux()
    out_vals, reaps = tree_util.tree_unflatten(out_tree, out_flat)
    out_tracers = jax_util.safe_map(trace.pure, out_vals)
    reap_tracers = tree_util.tree_map(trace.pure, reaps)
    return out_tracers, reap_tracers, metadata

  def process_nest(self, trace, f, *tracers, scope, name, **params):
    out_tracers, reap_tracers, _ = self.reap_higher_order_primitive(
        trace, nest_p, f, tracers, dict(params, name=name, scope=scope), False)
    tag = self.settings.tag
    if reap_tracers:
      flat_reap_tracers, reap_tree = tree_util.tree_flatten(reap_tracers)
      trace.process_primitive(
          sow_p, flat_reap_tracers,
          dict(name=scope, tag=tag, tree=reap_tree, mode='strict'))
    return out_tracers

  def process_higher_order_primitive(self, trace, call_primitive, f, tracers,
                                     params, is_map):
    out_tracers, reap_tracers, metadata = self.reap_higher_order_primitive(
        trace, call_primitive, f, tracers, params, is_map)
    tag = self.settings.tag
    for k, v in reap_tracers.items():
      flat_reap_tracers, reap_tree = tree_util.tree_flatten(v)
      trace.process_primitive(
          sow_p, flat_reap_tracers,
          dict(name=k, tag=tag, tree=reap_tree, mode=metadata[k]['mode']))
    return out_tracers


@lu.transformation
def reap_function(main: jax_core.MainTrace, settings: HarvestSettings,
                  return_metadata: bool, args: Iterable[Any]):
  """A function transformation that returns reap values."""
  trace = HarvestTrace(main, jax_core.cur_sublevel())
  in_tracers = jax_util.safe_map(trace.pure, args)
  context = ReapContext(settings, {})
  with trace_util.new_dynamic_context(main, context):
    ans = yield in_tracers, {}
    out_tracers = jax_util.safe_map(trace.full_raise, ans)
    reap_tracers = tree_util.tree_map(lambda x: trace.full_raise(x.value),
                                      context.reaps)
    reap_metadata = tree_util.tree_map(lambda x: x.metadata, context.reaps)
    del main
  out_values, reap_values = tree_util.tree_map(lambda x: x.val,
                                               (out_tracers, reap_tracers))
  if return_metadata:
    out = (out_values, reap_values, reap_metadata)
  else:
    out = (out_values, reap_values)
  yield out


def reap_eval(
    f: lu.WrappedFun, trace: HarvestTrace,
    settings: HarvestSettings) -> Tuple[lu.WrappedFun, Callable[[], Any]]:
  f = reap_function(f, trace.main, settings, True)
  return reap_wrapper(f, trace)


@lu.transformation_with_aux
def reap_wrapper(trace: HarvestTrace, *args):
  del trace
  out, reaps, metadata = yield (args,), {}
  out_flat, out_tree = tree_util.tree_flatten((out, reaps))
  yield out_flat, (out_tree, metadata)


def call_and_reap(f,
                  *,
                  tag: str,
                  allowlist: Optional[Iterable[str]] = None,
                  blocklist: Iterable[str] = frozenset(),
                  exclusive: bool = False):
  """Transforms a function into one that additionally returns its sown values.

  Args:
    f: a function to be transformed.
    tag: a string tag; only sown values with `tag` will be reaped.
    allowlist: an optional sequence of string names, which if provided will
      enforce that only sows with names in the allowlist will be reaped.
    blocklist: an optional sequence of string names, which if provided will
      enforce that only no sows with names in the blocklist will be reaped.
    exclusive: determines whether or not to execute in "exclusive" mode
      where other tags are removed during execution.

  Returns:
    A new function that executes the original and returns its sown values as
    an additional return value.
  """
  blocklist = frozenset(blocklist)
  if allowlist is not None:
    allowlist = frozenset(allowlist)
  settings = HarvestSettings(tag, blocklist, allowlist, exclusive)

  def wrapped(*args, **kwargs):
    fun = lu.wrap_init(f, kwargs)
    flat_args, in_tree = tree_util.tree_flatten(args)
    flat_fun, out_tree = api_util.flatten_fun_nokwargs(fun, in_tree)
    with jax_core.new_main(HarvestTrace) as main:
      flat_fun = reap_function(flat_fun, main, settings, False)
      out_flat, reaps = flat_fun.call_wrapped(flat_args)
      del main
    return tree_util.tree_unflatten(out_tree(), out_flat), reaps

  return wrapped


def reap(f,
         *,
         tag: str,
         allowlist: Optional[Iterable[str]] = None,
         blocklist: Iterable[str] = frozenset(),
         exclusive: bool = False):
  """Transforms a function into one that returns its sown values.

  Args:
    f: a function to be transformed.
    tag: a string tag; only sown values with `tag` will be reaped.
    allowlist: an optional sequence of string names, which if provided will
      enforce that only sows with names in the allowlist will be reaped.
    blocklist: an optional sequence of string names, which if provided will
      enforce that only no sows with names in the blocklist will be reaped.
    exclusive: determines whether or not to execute in "exclusive" mode
      where other tags are removed during execution.

  Returns:
    A new function that executes the original and returns its sown values.
  """

  def wrapped(*args, **kwargs):
    return call_and_reap(
        f,
        tag=tag,
        allowlist=allowlist,
        blocklist=blocklist,
        exclusive=exclusive)(*args, **kwargs)[1]

  return wrapped


@lu.transformation_with_aux
def _reap_metadata_wrapper(*args):
  out, reaps, metadata = yield (args,), {}
  yield (out, reaps), metadata


def _get_harvest_metadata(closed_jaxpr, settings, *args):
  """Probes a jaxpr for metadata like its sown values."""
  fun = lu.wrap_init(jax_core.jaxpr_as_fun(closed_jaxpr))
  with jax_core.new_main(HarvestTrace) as main:
    settings = HarvestSettings(settings.tag, settings.blocklist,
                               settings.allowlist, True)
    fun = reap_function(fun, main, settings, True)
    fun, aux = _reap_metadata_wrapper(fun)
    flat_args, in_tree = tree_util.tree_flatten(args)
    flat_fun, out_tree = api_util.flatten_fun_nokwargs(fun, in_tree)
    in_avals = jax_util.safe_map(
        lambda a: abstract_arrays.raise_to_shaped(jax_core.get_aval(a)),
        flat_args)
    pe.trace_to_jaxpr_final(flat_fun, in_avals)
    metadata = aux()
    out_tree()
  return metadata


def _reap_scan_rule(trace: HarvestTrace, *tracers, length, reverse, jaxpr,
                    num_consts, num_carry, linear, unroll):
  """Reaps the body of a scan to pull out `clobber` and `append` sows."""

  const_tracers, carry_tracers, xs_tracers = jax_util.split_list(
      tracers, [num_consts, num_carry])
  _, carry_avals, xs_avals = tree_util.tree_map(
      lambda x: x.aval, (const_tracers, carry_tracers, xs_tracers))
  const_vals, carry_vals, xs_vals = tree_util.tree_map(
      lambda x: x.val, (const_tracers, carry_tracers, xs_tracers))
  context = trace_util.get_dynamic_context(trace)
  settings = context.settings
  x_tracers = [t[0] if hasattr(t, '_getitem') else t for t in xs_tracers]
  x_avals = [t.aval for t in x_tracers]
  x_vals = [t.val for t in x_tracers]
  metadata = _get_harvest_metadata(jaxpr, settings,
                                   *(const_vals + carry_vals + x_vals))

  reap_modes = collections.defaultdict(set)
  reap_carry_avals = {}
  for name, meta in metadata.items():
    mode = meta['mode']
    aval = meta['aval']
    if mode == 'strict':
      raise ValueError(f'Cannot use strict mode for \'{name}\' inside `scan`.')
    reap_modes[mode].add(name)
    if mode == 'clobber':
      reap_carry_avals[name] = aval
  body_fun = jax_core.jaxpr_as_fun(jaxpr)

  reap_carry_flat_avals, _ = tree_util.tree_flatten(reap_carry_avals)

  reap_carry_in_tree = tree_util.tree_structure(
      ((carry_avals, reap_carry_avals), xs_avals))

  def new_body(carry, x):
    carry, _ = carry
    all_values = const_vals + tree_util.tree_leaves((carry, x))
    out, reaps = call_and_reap(
        body_fun,
        tag=settings.tag,
        allowlist=settings.allowlist,
        blocklist=settings.blocklist,
        exclusive=settings.exclusive)(*all_values)
    carry_out, y = jax_util.split_list(out, [num_carry])
    carry_reaps = {
        name: val
        for name, val in reaps.items()
        if name in reap_modes['clobber']
    }
    xs_reaps = {
        name: val for name, val in reaps.items() if name in reap_modes['append']
    }
    return (carry_out, carry_reaps), (y, xs_reaps)

  new_body_jaxpr, consts, out_tree = lcf._initial_style_jaxpr(  # pylint: disable=protected-access
      new_body, reap_carry_in_tree,
      tuple(carry_avals + reap_carry_flat_avals + x_avals))
  dummy_reap_carry_vals = tree_util.tree_map(
      lambda x: jnp.zeros(x.shape, x.dtype), reap_carry_flat_avals)
  out = lax.scan_p.bind(
      *(consts + carry_vals + dummy_reap_carry_vals + xs_vals),
      reverse=reverse,
      length=length,
      jaxpr=new_body_jaxpr,
      num_consts=len(consts),
      num_carry=len(carry_vals + dummy_reap_carry_vals),
      linear=(linear[:len(consts)] + (False,) * len(dummy_reap_carry_vals) +
              linear[len(consts):]),
      unroll=unroll)
  (carry_out,
   carry_reaps), (ys, ys_reaps) = tree_util.tree_unflatten(out_tree, out)
  (carry_out, carry_reaps), (ys, ys_reaps) = tree_util.tree_map(
      trace.pure, ((carry_out, carry_reaps), (ys, ys_reaps)))
  for k, v in {**carry_reaps, **ys_reaps}.items():
    sow(v, tag=settings.tag, mode=metadata[k]['mode'], name=k)
  return carry_out + ys


reap_custom_rules[lcf.scan_p] = _reap_scan_rule


def _reap_while_rule(trace: HarvestTrace, *tracers, cond_jaxpr, body_jaxpr,
                     cond_nconsts, body_nconsts):
  """Reaps the body of a while loop to get the reaps of the final iteration."""
  cond_const_tracers, body_const_tracers, init_tracers = jax_util.split_list(
      tracers, [cond_nconsts, body_nconsts])
  _, init_avals = tree_util.tree_map(lambda x: x.aval,
                                     (body_const_tracers, init_tracers))
  cond_const_vals, body_const_vals, init_vals = tree_util.tree_map(
      lambda x: x.val, (cond_const_tracers, body_const_tracers, init_tracers))
  context = trace_util.get_dynamic_context(trace)
  settings = context.settings
  body_metadata = _get_harvest_metadata(body_jaxpr, settings,
                                        *(body_const_tracers + init_tracers))
  for k, meta in body_metadata.items():
    mode = meta['mode']
    if mode != 'clobber':
      raise ValueError(
          f'Must use clobber mode for \'{k}\' inside of a `while_loop`.')
  reap_avals = {k: v['aval'] for k, v in body_metadata.items()}

  cond_fun = jax_core.jaxpr_as_fun(cond_jaxpr)
  body_fun = jax_core.jaxpr_as_fun(body_jaxpr)
  reap_settings = dict(
      tag=settings.tag,
      allowlist=settings.allowlist,
      blocklist=settings.blocklist,
      exclusive=settings.exclusive)

  def new_cond(carry, _):
    return cond_fun(*(cond_const_vals + carry))

  def new_body(carry, _):
    carry, reaps = call_and_reap(body_fun,
                                 **reap_settings)(*(body_const_vals + carry))
    return (carry, reaps)

  new_in_avals, new_in_tree = tree_util.tree_flatten((init_avals, reap_avals))
  new_cond_jaxpr, cond_consts, _ = lcf._initial_style_jaxpr(  # pylint: disable=protected-access
      new_cond, new_in_tree, tuple(new_in_avals))
  new_body_jaxpr, body_consts, out_tree = lcf._initial_style_jaxpr(  # pylint: disable=protected-access
      new_body, new_in_tree, tuple(new_in_avals))
  dummy_reap_vals = tree_util.tree_map(lambda x: jnp.zeros(x.shape, x.dtype),
                                       reap_avals)
  new_in_vals = tree_util.tree_leaves((init_vals, dummy_reap_vals))
  out = lax.while_p.bind(
      *(cond_consts + body_consts + new_in_vals),
      cond_nconsts=len(cond_consts),
      body_nconsts=len(body_consts),
      cond_jaxpr=new_cond_jaxpr,
      body_jaxpr=new_body_jaxpr)
  out = jax_util.safe_map(trace.pure, out)
  out, reaps = tree_util.tree_unflatten(out_tree, out)
  for k, v in reaps.items():
    sow(v, name=k, tag=settings.tag, mode=body_metadata[k]['mode'])
  return out


reap_custom_rules[lcf.while_p] = _reap_while_rule


def _check_branch_metadata(branch_metadatas):
  """Checks that a set of harvest metadata are consistent with each other."""
  first_branch_meta = branch_metadatas[0]
  for branch_metadata in branch_metadatas[1:]:
    if len(branch_metadata) != len(first_branch_meta):
      raise ValueError('Mismatching number of `sow`s between branches.')
    for name, meta in branch_metadata.items():
      if name not in first_branch_meta:
        raise ValueError(f'Missing sow in branch: \'{name}\'.')
      first_meta_aval = first_branch_meta[name]['aval']
      if meta['aval'].shape != first_meta_aval.shape:
        raise ValueError(f'Mismatched shape between branches: \'{name}\'.')
      if meta['aval'].dtype != first_meta_aval.dtype:
        raise ValueError(f'Mismatched dtype between branches: \'{name}\'.')


def _reap_cond_rule(trace, *tracers, branches, linear):
  """Reaps each path of the `cond`."""
  index_tracer, ops_tracers = tracers[0], tracers[1:]
  index_val, ops_vals = tree_util.tree_map(lambda x: x.val,
                                           (index_tracer, ops_tracers))
  _, ops_avals = tree_util.tree_map(lambda x: x.aval,
                                    (index_tracer, ops_tracers))
  context = trace_util.get_dynamic_context(trace)
  settings = context.settings
  reap_settings = dict(
      tag=settings.tag,
      allowlist=settings.allowlist,
      blocklist=settings.blocklist,
      exclusive=settings.exclusive)
  branch_metadatas = tuple(
      _get_harvest_metadata(branch, settings, *ops_tracers)
      for branch in branches)
  _check_branch_metadata(branch_metadatas)
  branch_funs = tuple(map(jax_core.jaxpr_as_fun, branches))
  reaped_branches = tuple(
      call_and_reap(f, **reap_settings) for f in branch_funs)
  in_tree = tree_util.tree_structure(ops_avals)
  new_branch_jaxprs, consts, out_trees = (
      lcf._initial_style_jaxprs_with_common_consts(  # pylint: disable=protected-access
          reaped_branches, in_tree, ops_avals, lax.cond_p.name))
  out = lax.cond_p.bind(
      index_val,
      *(tuple(consts) + ops_vals),
      branches=tuple(new_branch_jaxprs),
      linear=(False,) * len(tuple(consts) + linear))
  out = jax_util.safe_map(trace.pure, out)
  out, reaps = tree_util.tree_unflatten(out_trees[0], out)
  for k, v in reaps.items():
    sow(v, name=k, tag=settings.tag, mode=branch_metadatas[0][k]['mode'])
  return out


reap_custom_rules[lcf.cond_p] = _reap_cond_rule

plant_custom_rules = {}


@dataclasses.dataclass
class PlantContext(HarvestContext):
  """Contains the settings and storage for the current trace in the stack."""
  settings: HarvestSettings
  plants: Dict[str, Any]

  def __post_init__(self):
    self._already_planted = set()

  def get_custom_rule(self, primitive):
    return plant_custom_rules.get(primitive)

  def handle_sow(self, *values, name, tag, tree, mode):
    """Returns the value stored in the plants dictionary."""
    if name in self._already_planted:
      raise ValueError(f'Variable has already been planted: {name}')
    if name in self.plants:
      self._already_planted.add(name)
      return tree_util.tree_leaves(self.plants[name])
    return sow_p.bind(*values, name=name, tag=tag, mode=mode, tree=tree)

  def process_nest(self, trace, f, *tracers, scope, name, **params):
    return self.process_higher_order_primitive(
        trace, nest_p, f, tracers, dict(params, name=name, scope=scope), False)

  def process_higher_order_primitive(self, trace, call_primitive, f, tracers,
                                     params, is_map):
    del is_map
    name = jax_util.wrap_name(params.pop('name', f.__name__), 'reap')
    context = trace_util.get_dynamic_context(trace)
    vals = [t.val for t in tracers]
    plants = context.plants
    if 'in_axes' in params:
      # TODO(b/199459308): figure out if invars are mapped or unmapped
      params = dict(
          params,
          in_axes=(0,) * len(tree_util.tree_leaves(plants)) + params['in_axes'])
    if 'donated_invars' in params:
      params = dict(params)
      params['donated_invars'] = (
          (False,) * len(tree_util.tree_leaves(plants)) +
          params['donated_invars'])
    elif call_primitive is nest_p:
      plants = plants.get(params['scope'], {})
    all_vals, all_tree = tree_util.tree_flatten((plants, vals))
    f = plant_eval(f, trace, self.settings, all_tree)
    out_vals = call_primitive.bind(f, *all_vals, name=name, **params)
    return jax_util.safe_map(trace.pure, out_vals)


@lu.transformation
def plant_function(main: jax_core.MainTrace, settings: HarvestSettings,
                   in_tree: Any, args: Iterable[Any]):
  """A function transformation that injects values in place of sows."""
  trace = HarvestTrace(main, jax_core.cur_sublevel())
  plants, args = tree_util.tree_unflatten(in_tree, args)
  args = jax_util.safe_map(trace.pure, args)
  context = PlantContext(settings, plants)
  with trace_util.new_dynamic_context(main, context):
    ans = yield args, {}
    out_tracers = jax_util.safe_map(trace.full_raise, ans)
    del main
  yield [t.val for t in out_tracers]


def plant_eval(f: lu.WrappedFun, trace: HarvestTrace, settings: HarvestSettings,
               all_tree: Any) -> Tuple[lu.WrappedFun, Callable[[], Any]]:
  f = plant_function(f, trace.main, settings, all_tree)
  return plant_wrapper(f)


@lu.transformation
def plant_wrapper(*args):
  out = yield (args,), {}
  yield out


def plant(f,
          *,
          tag: str,
          allowlist: Optional[Iterable[str]] = None,
          blocklist: Iterable[str] = frozenset(),
          exclusive: bool = False):
  """Transforms a function into one that injects values in place of sown ones.

  Args:
    f: a function to be transformed.
    tag: a string tag; only sown values with `tag` will be planted.
    allowlist: an optional sequence of string names, which if provided will
      enforce that only sows with names in the allowlist will be planted.
    blocklist: an optional sequence of string names, which if provided will
      enforce that only no sows with names in the blocklist will be planted.
    exclusive: determines whether or not to execute in "exclusive" mode
      where other tags are removed during execution.

  Returns:
    A new function that takes in a dictionary of planted values in addition to
    the original function's inputs, and injects the planted values in place of
    sown values.
  """

  blocklist = frozenset(blocklist)
  if allowlist is not None:
    allowlist = frozenset(allowlist)
  settings = HarvestSettings(tag, blocklist, allowlist, exclusive)

  def wrapped(plants, *args, **kwargs):
    fun = lu.wrap_init(f, kwargs)
    flat_args, in_tree = tree_util.tree_flatten(args)
    flat_fun, out_tree = api_util.flatten_fun_nokwargs(fun, in_tree)
    all_args, all_tree = tree_util.tree_flatten((plants, flat_args))
    with jax_core.new_main(HarvestTrace) as main:
      flat_fun = plant_function(flat_fun, main, settings, all_tree)
      out_flat = flat_fun.call_wrapped(all_args)
      del main
    return tree_util.tree_unflatten(out_tree(), out_flat)

  return wrapped


def _plant_scan_rule(trace: HarvestTrace, *tracers, length, reverse, jaxpr,
                     num_consts, num_carry, linear, unroll):
  """Injects values into a scan according to their sow mode."""

  const_tracers, carry_tracers, xs_tracers = jax_util.split_list(
      tracers, [num_consts, num_carry])
  carry_avals, xs_avals = tree_util.tree_map(lambda x: x.aval,
                                             (carry_tracers, xs_tracers))
  const_vals, carry_vals, xs_vals = tree_util.tree_map(
      lambda x: x.val, (const_tracers, carry_tracers, xs_tracers))
  context = trace_util.get_dynamic_context(trace)
  settings = context.settings
  x_tracers = [t[0] if hasattr(t, '_getitem') else t for t in xs_tracers]
  x_avals = [t.aval for t in x_tracers]
  metadata = _get_harvest_metadata(jaxpr, settings,
                                   *(const_tracers + carry_tracers + x_tracers))

  plants = context.plants
  plant_modes = collections.defaultdict(set)
  plant_xs_avals = {}
  for name, meta in metadata.items():
    mode = meta['mode']
    aval = meta['aval']
    if mode == 'strict':
      raise ValueError(f'Cannot use strict mode for \'{name}\' inside `scan`.')
    plant_modes[mode].add(name)
    if mode == 'append' and name in plants:
      plant_xs_avals[name] = aval
  body_fun = jax_core.jaxpr_as_fun(jaxpr)
  clobber_plants = {
      name: value
      for name, value in plants.items()
      if name in plant_modes['clobber']
  }
  append_plants = {
      name: value
      for name, value in plants.items()
      if name in plant_modes['append']
  }

  plant_xs_flat_avals, _ = tree_util.tree_flatten(plant_xs_avals)

  plant_xs_in_tree = tree_util.tree_structure(
      (carry_avals, (xs_avals, plant_xs_avals)))

  def new_body(carry, x):
    x, plants = x
    all_plants = {**plants, **clobber_plants}
    all_values = const_vals + tree_util.tree_leaves((carry, x))
    out = plant(
        body_fun,
        tag=settings.tag,
        allowlist=settings.allowlist,
        blocklist=settings.blocklist,
        exclusive=settings.exclusive)(all_plants, *all_values)
    carry_out, y = jax_util.split_list(out, [num_carry])
    return carry_out, y

  new_body_jaxpr, consts, _ = lcf._initial_style_jaxpr(  # pylint: disable=protected-access
      new_body, plant_xs_in_tree,
      tuple(carry_avals + x_avals + plant_xs_flat_avals))
  plant_vals = tree_util.tree_leaves(append_plants)
  out = lcf.scan_p.bind(
      *(consts + carry_vals + xs_vals + plant_vals),
      reverse=reverse,
      length=length,
      jaxpr=new_body_jaxpr,
      num_consts=len(consts),
      num_carry=num_carry,
      linear=linear + (False,) * len(plant_vals),
      unroll=unroll)
  return out


plant_custom_rules[lcf.scan_p] = _plant_scan_rule


def _plant_while_rule(trace: HarvestTrace, *tracers, cond_jaxpr, body_jaxpr,
                      cond_nconsts, body_nconsts):
  """Injects values into a while loop, overriding values for all iterations."""
  cond_const_tracers, body_const_tracers, init_tracers = jax_util.split_list(
      tracers, [cond_nconsts, body_nconsts])
  init_avals = tree_util.tree_map(lambda x: x.aval, init_tracers)
  cond_const_vals, body_const_vals, init_vals = tree_util.tree_map(
      lambda x: x.val, (cond_const_tracers, body_const_tracers, init_tracers))
  context = trace_util.get_dynamic_context(trace)
  settings = context.settings
  body_metadata = _get_harvest_metadata(body_jaxpr, settings,
                                        *(body_const_tracers + init_tracers))
  for k, meta in body_metadata.items():
    mode = meta['mode']
    if mode != 'clobber':
      raise ValueError(
          f'Must use clobber mode for \'{k}\' inside of a `while_loop`.')

  body_fun = jax_core.jaxpr_as_fun(body_jaxpr)
  plant_settings = dict(
      tag=settings.tag,
      allowlist=settings.allowlist,
      blocklist=settings.blocklist,
      exclusive=settings.exclusive)
  plants = context.plants

  def new_body(*carry):
    carry = plant(body_fun, **plant_settings)(plants,
                                              *(tuple(body_const_vals) + carry))
    return carry

  in_tree = tree_util.tree_structure(init_avals)
  new_body_jaxpr, new_body_consts, _ = lcf._initial_style_jaxpr(  # pylint: disable=protected-access
      new_body, in_tree, tuple(init_avals))
  out = lcf.while_p.bind(
      *(cond_const_vals + new_body_consts + init_vals),
      cond_nconsts=len(cond_const_vals),
      body_nconsts=len(new_body_consts),
      cond_jaxpr=cond_jaxpr,
      body_jaxpr=new_body_jaxpr)
  return jax_util.safe_map(trace.pure, out)


plant_custom_rules[lcf.while_p] = _plant_while_rule


def _plant_cond_rule(trace, *tracers, branches, linear):
  """Injects the same values into both branches of a conditional."""
  index_tracer, ops_tracers = tracers[0], tracers[1:]
  index_val, ops_vals = tree_util.tree_map(lambda x: x.val,
                                           (index_tracer, ops_tracers))
  ops_avals = tree_util.tree_map(lambda x: x.aval, ops_tracers)
  context = trace_util.get_dynamic_context(trace)
  settings = context.settings
  plant_settings = dict(
      tag=settings.tag,
      allowlist=settings.allowlist,
      blocklist=settings.blocklist,
      exclusive=settings.exclusive)
  branch_metadatas = tuple(
      _get_harvest_metadata(branch, settings, *ops_tracers)
      for branch in branches)
  _check_branch_metadata(branch_metadatas)
  plants = context.plants
  branch_funs = tuple(map(jax_core.jaxpr_as_fun, branches))
  planted_branches = tuple(
      functools.partial(plant(f, **plant_settings), plants)
      for f in branch_funs)
  in_tree = tree_util.tree_structure(ops_avals)
  new_branch_jaxprs, consts, _ = (
      lcf._initial_style_jaxprs_with_common_consts(  # pylint: disable=protected-access
          planted_branches,
          in_tree,
          ops_avals,
          lax.cond_p.name))
  out = lax.cond_p.bind(
      index_val,
      *(tuple(consts) + ops_vals),
      branches=tuple(new_branch_jaxprs),
      linear=(False,) * len(tuple(consts) + linear))
  return jax_util.safe_map(trace.pure, out)


plant_custom_rules[lcf.cond_p] = _plant_cond_rule


def harvest(f,
            *,
            tag: str,
            allowlist: Optional[Iterable[str]] = None,
            blocklist: Iterable[str] = frozenset(),
            exclusive: bool = False):
  kwargs = dict(
      tag=tag, allowlist=allowlist, blocklist=blocklist, exclusive=exclusive)
  return call_and_reap(plant(f, **kwargs), **kwargs)
