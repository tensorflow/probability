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
"""Module for the unzip function transformation.

Unzip is a function transformation that looks
for 'variable' instantiations and pulls out
concretized variables for partial evaluation.
Primitives that return variables are registered
in the unzip_registry.

Unzip returns two functions:
  1. `init` - maps inputs to variables
  2. `apply` - maps variables and inputs to output
"""
import contextlib
import itertools as it

import dataclasses
from jax import abstract_arrays
from jax import api_util
from jax import core as jax_core
from jax import custom_derivatives as cd
from jax import linear_util as lu
from jax import tree_util
from jax import util as jax_util
from jax._src import source_info_util
from jax.interpreters import partial_eval as pe
import numpy as onp

from oryx.core import trace_util
from oryx.core.interpreters import harvest

__all__ = [
    'VariableError',
    'UnzipTrace',
    'UnzipTracer',
    'unzip',
    'unzip_registry',
]

safe_map = jax_core.safe_map
safe_zip = jax_core.safe_zip

unzip_registry = {}
block_registry = set()


def mapped_aval(*args, **kwargs):
  return jax_core.mapped_aval(*args, **kwargs)


class VariableError(Exception):
  """Raised if unable to unzip a function."""


class UnzipCustomRules:
  """defaultdict-like class that defers to pe.custom_partial_eval_rules."""

  def __init__(self, rules):
    self.rules = rules

  def __getitem__(self, key):
    if key not in self.rules:

      def custom_rule(*tracers, **params):
        out_jaxpr_tracers = pe.custom_partial_eval_rules[key](*tracers,
                                                              **params)
        out_tracers = [UnzipTracer(
            out_tracer._trace, out_tracer.pval, out_tracer.recipe,  # pylint: disable=protected-access
            False, None) for out_tracer in out_jaxpr_tracers]
        for out_tracer in out_tracers:
          recipe = out_tracer.recipe
          out_tracer.recipe = pe.new_eqn_recipe(recipe.invars, out_tracers,
                                                recipe.primitive, recipe.params,
                                                recipe.source_info)  # pytype: disable=wrong-arg-types
        return out_tracers

      return custom_rule
    return self.rules[key]

  def __setitem__(self, key, val):
    self.rules[key] = val

  def __contains__(self, key):
    return key in self.rules or key in pe.custom_partial_eval_rules

  def update(self, new_rules):
    return self.rules.update(new_rules)

  def copy(self):
    return UnzipCustomRules(self.rules.copy())


custom_rules = UnzipCustomRules({})
custom_rule_stack = [custom_rules]

current_custom_rules = lambda: custom_rule_stack[-1]


@contextlib.contextmanager
def new_custom_rules(rules):
  new_rules = current_custom_rules().copy()
  new_rules.update(rules)
  custom_rule_stack.append(new_rules)
  yield
  custom_rule_stack.pop(-1)


class VariableRecipe:

  def __init__(self, name, in_tracers, out_tracers):
    self.name = name
    self.in_tracers = in_tracers
    self.out_tracers = out_tracers


@dataclasses.dataclass(frozen=True)
class UnzipSettings:
  tag: str
  block: bool


@dataclasses.dataclass
class UnzipContext:
  settings: UnzipSettings


class UnzipTrace(jax_core.Trace):
  """Contains logic for handling UnzipTracers when tracing a function.

  The UnzipTrace is very similar to jax.interpreters.partial_eval.JaxprTrace,
  where it adds additional recipes into the tracers that track the variables
  produced while tracing. Variables are defined as outputs of the `variable`
  primitive that are also tagged as "keys". Inputs to the trace are designated
  as keys using `trace.new_arg` and if all the inputs to any primitive are
  "keys", the outputs are also "keys".
  """

  def pure(self, val):
    return self.new_const(val)

  def lift(self, val):
    return self.new_const(val)

  def sublift(self, val):
    return UnzipTracer(self, val.pval, pe.FreeVar(val), True)

  def new_const(self, val):
    if isinstance(val, jax_core.Tracer) and val._trace.level == self.level:  # pylint: disable=protected-access
      raise Exception
    return UnzipTracer(self, pe.PartialVal.known(val), jax_core.unit, True)

  def new_instantiated_literal(self, val):
    return UnzipTracer(self,
                       pe.PartialVal.unknown(trace_util.get_shaped_aval(val)),
                       jax_core.Literal(val), True)

  def new_instantiated_const(self, val):
    return UnzipTracer(self,
                       pe.PartialVal.unknown(trace_util.get_shaped_aval(val)),
                       pe.ConstVar(val), True)

  def new_arg(self, pval, key):
    return UnzipTracer(self, pval, pe.LambdaBinding(), key)

  def instantiate_const(self, tracer):
    pv, const = tracer.pval
    if isinstance(pv, jax_core.AbstractValue):
      return tracer
    elif not pv:
      if type(const) in jax_core.literalable_types and not onp.shape(const):  # pylint: disable=unidiomatic-typecheck
        return self.new_instantiated_literal(const)
      else:
        return self.new_instantiated_const(const)
    else:
      raise TypeError(pv)

  def instantiate_const_abstracted(self, tracer):
    pv, const = tracer.pval
    if isinstance(pv, jax_core.AbstractValue):
      return tracer
    elif pv is None:
      aval = abstract_arrays.raise_to_shaped(
          trace_util.get_shaped_aval(const), onp.isscalar(const))
      return UnzipTracer(self, pe.PartialVal.unknown(aval), pe.ConstVar(const),
                         tracer.is_key())
    else:
      raise TypeError(pv)

  def process_primitive(self, primitive, tracers, params):
    if primitive in current_custom_rules():
      return current_custom_rules()[primitive](self, *tracers, **params)
    return self.default_process_primitive(primitive, tracers, params)

  def default_process_primitive(self, primitive, tracers, params):
    """Partially evaluate primitives and saves variable recipes."""
    pvs, consts = jax_util.unzip2(t.pval for t in tracers)
    if all(pv is None for pv in pvs):
      return primitive.bind(*consts, **params)
    settings = trace_util.get_dynamic_context(self).settings
    tracers = safe_map(self.instantiate_const, tracers)
    if any(not isinstance(t, UnzipTracer) for t in tracers):
      assert False
    key = all(t.is_key() for t in tracers)
    avals = [t.aval for t in tracers]
    ans = primitive.abstract_eval(*avals, **params)
    if not primitive.multiple_results:
      ans = [ans]
    out_tracers = [
        UnzipTracer(self, pe.PartialVal((aval, jax_core.unit)), None, key)
        for aval in ans
    ]
    # Passing in UnzipTracer, which pytype does not recognize as JaxprTracer
    eqn = pe.new_eqn_recipe(tracers, out_tracers, primitive, params,
                            source_info_util.current())  # pytype: disable=wrong-arg-types
    for t in out_tracers:
      t.recipe = eqn

    is_variable = (
        key and primitive is harvest.sow_p and params['tag'] == settings.tag)
    # This block is where UnzipTrace mainly differs from pe.JaxprTrace. Where
    # JaxprTrace will just return out_tracers, UnzipTrace will record an
    # additional VariableRecipe into the tracers, which will be used after
    # the trace is complete to construct init/apply Jaxprs.
    if is_variable:
      name, var_in_tracers, var_out_tracers = unzip_registry[primitive](
          tracers, out_tracers, **params)
      variable_recipe = VariableRecipe(name, var_in_tracers, var_out_tracers)
      for t in out_tracers:
        t.variable_recipe = variable_recipe

    if primitive.multiple_results:
      return out_tracers
    return out_tracers[0]

  def process_call(self, call_primitive, f, tracers, params):
    return self.handle_call_primitive(call_primitive, f, tracers, params, False)

  def process_map(self, call_primitive, f, tracers, params):
    return self.handle_call_primitive(call_primitive, f, tracers, params, True)

  def handle_call_primitive(self, call_primitive, f, tracers, params, is_map):
    """Handler for call_primitives, like jit or layer_call.

    When an UnzipTracer hits a call primitive, there is either a variable
    inside of the call primitive, in which case the input
    function needs to be unzipped into two, or there are no variables
    in the function, so the call_primitive is recorded in the trace as-is.

    We use `unzip_eval_wrapper`, which returns whether or not an unzip
    was successful or not. If it was successful, we record two new
    Jaxprs into the trace (one for init, one for apply). Otherwise, we
    just record the Jaxpr corresponding to the function call.

    Args:
      call_primitive: a call primitive like xla_call
      f: a jax.linear_util wrapped function to be called
      tracers: inputs to the function
      params: parameters of the primitives
      is_map: whether or not the primitive is a map primitive (e.g. xla_pmap)

    Returns:
      A list of output tracers
    """
    name = params.get('name', f.__name__)
    settings = trace_util.get_dynamic_context(self).settings
    tracers = safe_map(self.instantiate_const_abstracted, tracers)
    if call_primitive in current_custom_rules():
      return current_custom_rules()[call_primitive](self, f, *tracers, **params)
    if call_primitive in pe.call_partial_eval_rules:
      raise NotImplementedError
    in_pvals = [t.pval for t in tracers]
    if is_map:
      unknown = pe.PartialVal.unknown
      in_pvals = [pval if pval.is_known() or in_axis is None else
                  unknown(mapped_aval(params['axis_size'], in_axis, pval[0]))
                  for pval, in_axis in zip(in_pvals, params['in_axes'])]
      out_axes_thunk = params['out_axes_thunk']
      @jax_util.as_hashable_function(closure=('unzip', out_axes_thunk))
      def new_out_axes_thunk():
        out_axes = out_axes_thunk()
        assert all(out_axis == 0 for out_axis in out_axes)
        _, num_outputs, _ = aux()
        return (0,) * num_outputs
      new_params = dict(params, out_axes_thunk=new_out_axes_thunk)
    else:
      new_params = params
    pvs, in_consts = jax_util.unzip2(t.pval for t in tracers)
    keys = tuple(t.is_key() for t in tracers)
    new_settings = UnzipSettings(settings.tag, call_primitive in block_registry)
    fun, aux = unzip_eval(f, self, keys, tuple(pvs), new_settings)
    out_flat = call_primitive.bind(fun, *in_consts, **new_params)
    success, _, results = aux()
    if not success:
      out_pvs, out_keys, jaxpr, env = results
      out_pv_consts, consts = jax_util.split_list(out_flat, [len(out_pvs)])
      out_tracers = self._bound_output_tracers(call_primitive, new_params,
                                               jaxpr, consts, env, tracers,
                                               out_pvs, out_pv_consts,
                                               out_keys, name, is_map)
      return out_tracers
    init_name = jax_util.wrap_name(name, 'init')
    apply_name = jax_util.wrap_name(name, 'apply')
    init_pvs, num_init_consts, apply_pvs = results[0]
    init_jaxpr, apply_jaxpr = results[1]
    init_env, apply_env = results[2]
    variable_names, variable_tree, apply_keys = results[3]

    key_tracers = [t for t in tracers if t.is_key()]
    abstract_tracers = [t for t in tracers if not t.is_key()]
    all_init_consts, all_apply_consts = jax_util.split_list(
        out_flat, [len(init_pvs) + num_init_consts])
    init_pv_consts, init_consts = jax_util.split_list(all_init_consts,
                                                      [len(init_pvs)])
    apply_pv_consts, apply_consts = jax_util.split_list(all_apply_consts,
                                                        [len(apply_pvs)])

    variable_tracers = self._bound_output_tracers(
        call_primitive, new_params, init_jaxpr, init_consts, init_env,
        key_tracers, init_pvs, init_pv_consts, [True] * len(init_pvs),
        init_name, is_map)

    unflat_variables = tree_util.tree_unflatten(variable_tree, variable_tracers)
    if call_primitive is harvest.nest_p:
      variable_dict = harvest.sow(
          dict(safe_zip(variable_names, unflat_variables)),
          tag=settings.tag,
          name=new_params['scope'],
          mode='strict')
      unflat_variables = tuple(variable_dict[name] for name in variable_names)
    else:
      unflat_variables = [
          harvest.sow(  # pylint: disable=g-complex-comprehension
              unflat_variable,
              tag=settings.tag,
              name=name,
              mode='strict') for unflat_variable, name in safe_zip(
                  unflat_variables, variable_names)
      ]
    variable_tracers = tree_util.tree_leaves(unflat_variables)

    out_tracers = self._bound_output_tracers(
        call_primitive, new_params, apply_jaxpr, apply_consts, apply_env,
        variable_tracers + abstract_tracers, apply_pvs, apply_pv_consts,
        apply_keys, apply_name, is_map)
    return out_tracers

  def _bound_output_tracers(self, primitive, params, jaxpr, consts, env,
                            in_tracers, out_pvs, out_consts, out_keys, name,
                            is_map):
    """Takes a traced function and binds the Jaxpr to output tracers."""
    lifted_jaxpr = pe.convert_constvars_jaxpr(jaxpr)
    const_tracers = safe_map(self.new_instantiated_const, consts)
    env_tracers = safe_map(self.instantiate_const, env)
    out_tracers = [
        UnzipTracer(self, pe.PartialVal((pv, const)), None, key)
        for pv, const, key in safe_zip(out_pvs, out_consts, out_keys)
    ]
    new_params = dict(params, name=name, call_jaxpr=lifted_jaxpr)
    if 'donated_invars' in params:
      new_donated_invars = (
          (False,) * len(const_tracers) + (False,) * len(env_tracers) +
          tuple(v for v, t in zip(params['donated_invars'], in_tracers)
                if not t.pval.is_known()))
      new_params['donated_invars'] = new_donated_invars
    if is_map:
      out_axes = params['out_axes_thunk']()
      assert all(out_axis == 0 for out_axis in out_axes)
      new_params['out_axes'] = (0,) * len(out_tracers)
      del new_params['out_axes_thunk']
    eqn = pe.new_eqn_recipe(
        tuple(const_tracers + env_tracers + in_tracers), out_tracers, primitive,
        new_params, source_info_util.current())  # pytype: disable=wrong-arg-types
    for t in out_tracers:
      t.recipe = eqn
    return out_tracers

  def post_process_call(self, call_primitive, out_tracers, params):
    raise NotImplementedError


def unzip_eval(f, trace, keys, pvs, settings):
  f = unzip_to_init_apply_subjaxprs(f, trace.main, settings, keys)
  return unzip_eval_wrapper(f, pvs)


class UnzipTracer(jax_core.Tracer):
  """Tracer whose state encapsulates if the inputs are keys."""

  def __init__(self, trace, pval, recipe, key, variable_recipe=None):
    self._trace = trace
    self.pval = pval
    self.recipe = recipe
    self.key = key
    self.variable_recipe = variable_recipe

  def is_key(self):
    return self.key

  @property
  def aval(self):
    pv, const = self.pval
    if isinstance(pv, jax_core.AbstractValue):
      return pv
    elif pv is None:
      return trace_util.get_shaped_aval(const)
    else:
      raise TypeError(pv)
    return self.val

  @property
  def parents(self):
    if isinstance(self.recipe, pe.JaxprEqnRecipe):
      return self.recipe.invars
    else:
      return []

  def is_pure(self):
    pv, _ = self.pval
    return pv is None

  def full_lower(self):
    if self.is_pure():
      _, const = self.pval
      return jax_core.full_lower(const)
    return self

  def __repr__(self):
    return 'Traced[{}]<{}:{}>'.format(self.is_key(), self.aval, self._trace)


@lu.transformation_with_aux
def unzip_eval_wrapper(pvs, *consts):
  """Function transformation that returns init/apply jaxprs and metadata."""
  args = (safe_map(pe.PartialVal, safe_zip(pvs, consts)),)
  success, result = yield args, {}
  if success:
    init_out, apply_out, pvals, metadata = result
    init_jaxpr, init_consts, init_env = init_out
    apply_jaxpr, apply_consts, apply_env = apply_out
    init_pvals, apply_pvals = pvals
    init_pvs, init_pv_consts = jax_util.unzip2(init_pvals)
    apply_pvs, apply_pv_consts = jax_util.unzip2(apply_pvals)

    out = (
        tuple(init_pv_consts) + tuple(init_consts) + tuple(apply_pv_consts) +
        tuple(apply_consts))
    yield out, (success, len(out),
                ((init_pvs, len(init_consts), apply_pvs),
                 (init_jaxpr, apply_jaxpr),
                 (init_env, apply_env),
                 metadata))
  else:
    jaxpr, (out_pvals, out_keys, consts, env) = result
    out_pvs, out_consts = jax_util.unzip2(out_pvals)
    out = tuple(out_consts) + tuple(consts)
    yield out, (success, len(out), (out_pvs, out_keys, jaxpr, env))


@lu.transformation
def unzip_to_init_apply_subjaxprs(master, settings, keys, pvals):
  """Function transformation that returns init/apply jaxprs."""
  trace = UnzipTrace(master, jax_core.cur_sublevel())
  # Setting up input UnzipTracer objects
  in_tracers = safe_map(lambda a: trace.new_arg(a[0], a[1]), zip(pvals, keys))
  key_tracers = [t for t in in_tracers if t.key]
  abstract_tracers = [t for t in in_tracers if not t.key]
  # Passing input tracers into function
  # to get output tracers
  context = UnzipContext(settings)
  with trace_util.new_dynamic_context(master, context):
    ans = yield in_tracers, {}
  out_tracers = safe_map(trace.full_raise, safe_map(jax_core.full_lower, ans))
  out_pvals = [t.pval for t in out_tracers]

  all_tracers = jax_util.toposort(out_tracers)
  variable_tracers = [t for t in all_tracers if t.variable_recipe]
  if not settings.block:
    try:
      # This try/catch tests whether or not the variables define a cut of the
      # computation graph. `pe.tracers_to_jaxpr` throws an AssertionError
      # if that is the case.
      old_recipes = [t.recipe for t in variable_tracers]
      for t in variable_tracers:
        t.recipe = pe.LambdaBinding()
      _tracers_to_jaxpr(variable_tracers + abstract_tracers, out_tracers)
    except VariableError:
      success = False
    else:
      success = True
    finally:
      # Restore the old recipes if it fails
      for t, old_recipe in safe_zip(variable_tracers, old_recipes):
        t.recipe = old_recipe
  else:
    success = False
  if not success:
    jaxpr, consts, env = _tracers_to_jaxpr(in_tracers, out_tracers)
    out_keys = [t.is_key() for t in out_tracers]
    yield success, (jaxpr, (out_pvals, out_keys, consts, env))
    return

  variable_recipes = {}
  for t in all_tracers:
    if t.variable_recipe:
      name = t.variable_recipe.name
      if (name in variable_recipes and
          variable_recipes[name] is not t.variable_recipe):
        raise ValueError('Cannot use duplicate variable name: {}'.format(name))
      variable_recipes[name] = t.variable_recipe

  variables = {
      name: (recipe.in_tracers, recipe.out_tracers)
      for name, recipe in variable_recipes.items()
  }
  variable_names, variable_tracers = jax_util.unzip2(variables.items())
  var_in_tracers, var_out_tracers = jax_util.unzip2(variable_tracers)
  flat_var_in_tracers, variable_tree = tree_util.tree_flatten(var_in_tracers)
  var_pvals = [t.pval for t in flat_var_in_tracers]
  flat_var_out_tracers, _ = tree_util.tree_flatten(var_out_tracers)
  init_jaxpr, init_consts, init_env = _tracers_to_jaxpr(key_tracers,
                                                        flat_var_in_tracers)
  for t in flat_var_out_tracers:
    t.recipe = pe.LambdaBinding()
  apply_jaxpr, apply_consts, apply_env = _tracers_to_jaxpr(
      flat_var_out_tracers + abstract_tracers, out_tracers)
  if None in variable_names:
    raise ValueError('Must provide name for variable.')
  out_keys = [t.is_key() for t in out_tracers]
  yield success, ((init_jaxpr, init_consts,
                   init_env), (apply_jaxpr, apply_consts, apply_env),
                  (var_pvals, out_pvals), (variable_names, variable_tree,
                                           out_keys))


def flatten_args_into_keys(avals, key_args):
  """Flattens avals and returns a list indicating which are keys."""
  flat_avals, in_tree = tree_util.tree_flatten(avals)

  def is_key_aval(i):
    return lambda _: i in key_args

  flat_keys, _ = tree_util.tree_flatten([
      tree_util.tree_map(is_key_aval(i), aval) for i, aval in enumerate(avals)
  ])
  return flat_avals, flat_keys, in_tree


def unzip(f, *, tag: str, key_args=0):
  """Unzip function transformation."""
  if tag is None:
    raise ValueError('Must provide sow tag to unzip.')
  if key_args is None:
    key_args = ()
  if isinstance(key_args, int):
    key_args = (key_args,)
  key_args = set(key_args)

  def wrapped(*args, **kwargs):
    """Callable returned by unzip."""
    with jax_core.new_main(UnzipTrace) as master:
      # Preparing args to be traced
      fun = lu.wrap_init(f, kwargs)
      avals = tree_util.tree_map(trace_util.get_shaped_aval, args)
      flat_avals, flat_keys, in_tree = (flatten_args_into_keys(avals, key_args))
      flat_pvals = [pe.PartialVal.unknown(aval) for aval in flat_avals]
      flat_fun, out_tree = api_util.flatten_fun_nokwargs(fun, in_tree)

      # Trace to jaxpr
      settings = UnzipSettings(tag, False)
      fun = unzip_to_init_apply_subjaxprs(flat_fun, master, settings)  # pylint: disable=no-value-for-parameter
      success, results = fun.call_wrapped(flat_keys, flat_pvals)
      if not success:
        raise ValueError('Variables do not cut dependence graph.')
      init_out, apply_out, _, metadata = results
      init_jaxpr, init_consts, init_env = init_out
      assert not init_env

      apply_jaxpr, apply_consts, apply_env = apply_out
      assert not apply_env

      names, variable_tree, _ = metadata
      out_tree = out_tree()

      # Final functions
      def init(*args):
        flat_args, _ = tree_util.tree_flatten(args)
        flat_params = jax_core.eval_jaxpr(init_jaxpr, init_consts, *flat_args)
        flat_variables = tree_util.tree_unflatten(variable_tree, flat_params)
        return {name: var for name, var in safe_zip(names, flat_variables)}

      def apply(params, *args):
        flat_variables, _ = tree_util.tree_flatten(
            [params[name] for name in names])
        flat_args, _ = tree_util.tree_flatten(args)
        out = jax_core.eval_jaxpr(apply_jaxpr, apply_consts,
                                  *(flat_variables + flat_args))
        return tree_util.tree_unflatten(out_tree, out)

      del master
    return init, apply

  return wrapped


def _tracers_to_jaxpr(in_tracers, out_tracers):
  """Constructs Jaxpr given tracers for inputs and outputs.

  Copied from jax.interpreters.partial_eval.tracers_to_jaxpr but modified to
  raise an VariableError when unknown in_tracers are found, rather than the
  default AssertionError.

  Args:
    in_tracers: the tracers that were created for the function inputs
    out_tracers: the tracers that were output by the function.

  Returns:
    a triple of a `Jaxpr`, a list of constant values corresponding to
    the `constvars` in the returned Jaxps, and a list of environment values.
    The vars for the environment values have been pre-pended to the Jaxpr's
    `invars`.

  Raises:
    VariableError: if an unknown input tracer is found
  """
  newvar = jax_core.gensym(None)
  t_to_var = {}

  def getvar(t):
    var = t_to_var.get(id(t))
    if var is None:
      var = newvar(t.pval.get_aval())
      t_to_var[id(t)] = var
    return var

  sorted_tracers = jax_util.toposort(out_tracers)
  invars = safe_map(getvar, in_tracers)
  eqns = []
  env = {}
  consts = {}
  const_to_var = {}

  def getconstvar(c):
    var = const_to_var.get(id(c))
    if var is None:
      var = newvar(jax_core.get_aval(c))
      const_to_var[id(c)] = var
    return var

  processed_eqn_ids = set()
  for t in sorted_tracers:
    recipe = t.recipe
    if isinstance(recipe, pe.JaxprEqnRecipe):
      if recipe.eqn_id not in processed_eqn_ids:
        eqns.append(pe.recipe_to_eqn(getvar, recipe))
        processed_eqn_ids.add(recipe.eqn_id)
    elif isinstance(recipe, pe.LambdaBinding):
      if not any(t is in_tracer for in_tracer in in_tracers):
        raise VariableError(f'Found unknown input tracer: {t}')
      assert in_tracers, 'Lambda binding with no args'
    elif isinstance(recipe, pe.FreeVar):
      env[getvar(t)] = recipe.val
    elif isinstance(recipe, pe.ConstVar):
      v = t_to_var[id(t)] = getconstvar(recipe.val)
      consts[v] = recipe.val
    elif isinstance(recipe, jax_core.Literal):
      t_to_var[id(t)] = recipe
    elif recipe is jax_core.unit:
      t_to_var[id(t)] = jax_core.unitvar
    else:
      raise TypeError(recipe)

  env_vars, env_vals = jax_util.unzip2(env.items())
  const_vars, const_vals = jax_util.unzip2(consts.items())
  # The env_vars are pre-pended to the invars
  jaxpr = jax_core.Jaxpr(const_vars, list(it.chain(env_vars, invars)),
                         safe_map(getvar, out_tracers), eqns)
  return jaxpr, const_vals, env_vals


def sow_unzip(in_tracers, out_tracers, name=None, tree=None, tag=None, **_):
  del tag
  if tree:
    in_tracers = tree_util.tree_unflatten(tree, in_tracers)
    out_tracers = tree_util.tree_unflatten(tree, out_tracers)
  return name, in_tracers, out_tracers


unzip_registry[harvest.sow_p] = sow_unzip


def _custom_jvp_call_unzip(trace, fun, *tracers, **params):
  del trace
  return custom_jvp_call_jaxpr(fun, params['jvp'], *tracers)


custom_rules[cd.custom_jvp_call_p] = _custom_jvp_call_unzip


def _custom_vjp_call_unzip(trace, fun, *tracers, **params):
  del trace
  return custom_vjp_call_jaxpr(fun, params['fwd'], params['bwd'], *tracers,
                               **params)


custom_rules[cd.custom_vjp_call_p] = _custom_vjp_call_unzip


def custom_jvp_call_jaxpr(fun, jvp, *args):
  """A convenience wrapper to apply the custom_jvp_call_jaxpr primitive."""
  in_avals = [
      abstract_arrays.raise_to_shaped(jax_core.get_aval(x)) for x in args
  ]
  fun_jaxpr, consts = cd._initial_style_jaxpr(  # pylint: disable=protected-access
      fun, in_avals)  # consts can be tracers!
  closed_fun_jaxpr = jax_core.ClosedJaxpr(
      pe.convert_constvars_jaxpr(fun_jaxpr), ())
  jvp_jaxpr_thunk = pe._memoize(  # pylint: disable=protected-access
      lambda: cd._initial_style_jaxpr(jvp, in_avals * 2))  # pylint: disable=protected-access
  return cd.custom_jvp_call_jaxpr_p.bind(
      *consts,
      *args,
      fun_jaxpr=closed_fun_jaxpr,
      jvp_jaxpr_thunk=jvp_jaxpr_thunk,
      num_consts=len(consts))


def custom_vjp_call_jaxpr(fun, fwd, bwd, *args, out_trees):
  in_avals = [
      abstract_arrays.raise_to_shaped(jax_core.get_aval(x)) for x in args
  ]
  fun_jaxpr, consts = cd._initial_style_jaxpr(  # pylint: disable=protected-access
      fun, in_avals)  # consts can be tracers!
  closed_fun_jaxpr = jax_core.ClosedJaxpr(
      pe.convert_constvars_jaxpr(fun_jaxpr), ())
  fwd_jaxpr_thunk = pe._memoize(lambda: cd._initial_style_jaxpr(fwd, in_avals))  # pylint: disable=protected-access
  return cd.custom_vjp_call_jaxpr_p.bind(
      *consts,
      *args,
      fun_jaxpr=closed_fun_jaxpr,
      fwd_jaxpr_thunk=fwd_jaxpr_thunk,
      bwd=bwd,
      out_trees=out_trees,
      num_consts=len(consts))
