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
"""Module for transforming functions into FunctionModules.

In order to `init` functions, we need to define a `Module` subclass for them,
which is the `FunctionModule`. The `FunctionModule` encapsulates a Jaxpr that
is evaluated to execute the function, with special handling for keyword
arguments. This is useful for neural networks, where a keyword argument such as
`training` may change the semantics of a function. The next will be
adding a rule into the `kwargs_rules` dictionary, which is used in the
custom Jaxpr evaluator in `FunctionModule`. The `kwargs_rules` enables having
implementations for primitives that can change depending on the value of a
keyword argument. An example would be a neural network layer like dropout,
which has different behavior while training and not.

We also register functions with `api.init`. The `init` for functions first
inspects if the input function has a keyword argument `init_key`, and only if
that is the case does it harvest the function. This results in an opt-in
behavior for functions to be stateful.

To see documentation of `init`/`spec`/`call_and_update` and an example,
see api.py.
"""
import functools
import types

from typing import Any, Dict, Iterable, Optional

import jax
from jax import core as jax_core
from jax import linear_util as lu
from jax import random
from jax import tree_util
from jax import util as jax_util

from oryx.core import kwargs_util
from oryx.core import trace_util
from oryx.core.interpreters import harvest
from oryx.core.interpreters.inverse import custom_inverse
from oryx.core.state import api
from oryx.core.state import module

__all__ = [
    'FunctionModule',
]

safe_map = jax_util.safe_map
safe_zip = jax_util.safe_zip
Key = Any

kwargs_rules = {}


def _function_register(generic_func):
  """Registers function types with a single-dispatch function."""

  def wrapped(f):
    for pytype in [
        types.FunctionType, types.MethodType, types.LambdaType,
        types.BuiltinMethodType, types.BuiltinFunctionType, functools.partial,
        custom_inverse.CustomInverse, jax.custom_jvp, jax.custom_vjp
    ]:
      generic_func.register(pytype, f)
    return f

  return wrapped


def lift_from_kwargs(func, key, *args, **kwargs):
  """Moves a kwarg to the beginning of args."""
  kwargs = kwargs.copy()
  elem = kwargs.pop(key, None)
  if elem is None:

    def wrapped(*args, **kwargs):
      return func(*args, **kwargs)
  else:

    def wrapped(el, *args, **kwargs):
      wrapped_kwargs = kwargs.copy()
      wrapped_kwargs[key] = el
      return func(*args, **wrapped_kwargs)

  return wrapped, elem, args, kwargs


def eval_jaxpr_with_kwargs(jaxpr: jax_core.Jaxpr, consts: Iterable[Any], *args,
                           **kwargs):
  """Evals a jaxpr while passing kwargs into registered primitives."""

  def read(v):
    if isinstance(v, jax_core.Literal):
      return v.val
    else:
      return env[v]

  def write(v, val):
    env[v] = val

  env = {}
  write(jax_core.unitvar, jax_core.unit)
  safe_map(write, jaxpr.constvars, consts)
  safe_map(write, jaxpr.invars, args)
  for eqn in jaxpr.eqns:
    in_vals = safe_map(read, eqn.invars)
    subjaxpr, params = jax_core.extract_call_jaxpr(eqn.primitive, eqn.params)
    if subjaxpr:
      subfuns = [
          lu.wrap_init(
              functools.partial(eval_jaxpr_with_kwargs, subjaxpr, (), **kwargs))
      ]
    else:
      subfuns = []
      params = dict(eqn.params)
    if eqn.primitive in kwargs_rules:
      new_kwargs = dict(params.pop('kwargs', {}), **kwargs)
      ans = kwargs_rules[eqn.primitive](
          *(subfuns + in_vals), kwargs=new_kwargs, **params)
    else:
      ans = eqn.primitive.bind(*(subfuns + in_vals), **params)
    if eqn.primitive.multiple_results:
      safe_map(write, eqn.outvars, ans)
    else:
      write(eqn.outvars[0], ans)
  return safe_map(read, jaxpr.outvars)


class _EqJaxpr:
  """A class that automatically assumes equality between Jaxprs.

  This class exists to avoid Pytree comparison errors in JAX transformations and
  control-flow functions. Sometimes when we construct modules (which store
  Jaxprs as metadata), the equality comparison of Pytrees at the beginning
  and end of control flow functions will fail. This ensures that it will always
  pass for semantically equivalent `Modules`.
  """

  def __init__(self, jaxpr):
    self.jaxpr = jaxpr

  def __eq__(self, other):
    return True


class FunctionModule(module.Module):
  """Encapsulates a staged function."""

  def __init__(self,
               variables: Dict[str, Any],
               jaxpr: jax_core.ClosedJaxpr,
               in_tree: Any,
               out_tree: Any,
               *,
               name: Optional[str] = None):
    super().__init__(name=name)
    self._variables = variables
    self._jaxpr = jaxpr
    self._in_tree = in_tree
    self._out_tree = out_tree

  def replace(self, *, variables=None):
    variables = variables if variables is not None else self._variables
    return FunctionModule(
        variables, self._jaxpr, self._in_tree, self._out_tree, name=self.name)

  def call(self, *args, **kwargs):
    return self.call_and_update(*args, **kwargs)[0]

  def update(self, *args, **kwargs):
    return self.call_and_update(*args, **kwargs)[1]

  def _cau_jaxpr(self, *args, **kwargs):
    flat_args = tree_util.tree_leaves(args)
    out_flat = eval_jaxpr_with_kwargs(self._jaxpr.jaxpr, self._jaxpr.literals,
                                      *flat_args, **kwargs)
    return tree_util.tree_unflatten(self._out_tree, out_flat)

  def call_and_update(self, *args, **kwargs):
    out, assigns = self._cau_jaxpr(self._variables, *args, **kwargs)
    for name in assigns.keys():
      if name not in self._variables:
        raise ValueError(f'No variable declared for assign: {name}')
    return out, self.replace(variables={**self._variables, **assigns})

  def variables(self):
    return self._variables

  def flatten(self):
    variable_keys, variable_vals = jax_util.unzip2(self._variables.items())
    return variable_vals, (variable_keys, _EqJaxpr(self._jaxpr), self._in_tree,
                           self._out_tree, self.name)

  @classmethod
  def unflatten(cls, data, variable_vals):
    variable_keys, eq_jaxpr, in_tree, out_tree, name = data
    return cls(
        dict(zip(variable_keys, variable_vals)),
        eq_jaxpr.jaxpr,
        in_tree,
        out_tree,
        name=name)

  def __getattr__(self, attr_name):
    if attr_name in self._variables:
      return self._variables[attr_name]

  def __repr__(self):
    if self.name:
      return f'FunctionModule[{self.name}]({self._variables.keys()})'
    return f'FunctionModule({self._variables.keys()})'


api.call_and_update.register(FunctionModule)(FunctionModule.call_and_update)


@_function_register(api.init)
def function_init(f, *, name=None, init_keyword='init_key'):
  """Transforms a function into one that initializes a FunctionModule.

  `function_init` takes in a function `f` and an optional name and inspects
  `f` to see if it has a keyword argument (by default `init_key`).
  The keyword argument is a magic name for the stateful function API that
  enables `function_init` to be aware if a function is stateful or not.

  If the input function is stateful, then `function_init` will `reap` it
  with the `VARIABLE` tag to obtain the initial set of variables for a module.
  The `apply_f` is just `harvest(plant(f, VARIABLE), ASSIGN)` and is a function
  of the form `f(variables, x) -> y, assigns`, acting as a `call_and_update`
  function.  We then stage out the `call_and_update` function into a Jaxpr and
  instantiate a `FunctionModule`. If a name is provided to `function_init`, we
  tag the entire module as a variable, and if not, we re-tag its variables.

  Args:
    f: a function to be transformed.
    name: a string name for the returned module
      Default value: `None`
    init_keyword: a string name for keyword for the initialization PRNGKey.
      Default value: `'init_key'`

  Returns:
    A function that when given a key and arguments, returns a FunctionModule.
  """

  def wrapped(init_key: Key, *args, **kwargs) -> FunctionModule:
    has_init_key = kwargs_util.check_in_kwargs(f, init_keyword)
    if not has_init_key:

      def init_f(init_key, *args, **kwargs):
        del init_key, args, kwargs
        return {}

      def cau_f(variables, *args, **kwargs):
        return f(*args, **kwargs), variables
    else:

      def f_(init_key, *args, **kwargs):
        return f(*args, **kwargs, init_key=init_key)

      def init_f(init_key, *args, **kwargs):
        return harvest.reap(
            f_, tag=module.VARIABLE, exclusive=True)(init_key, *args, **kwargs)

      def apply_f(variables, *args, **kwargs):
        return harvest.plant(
            f_, tag=module.VARIABLE)(variables, random.PRNGKey(0), *args,
                                     **kwargs)

      cau_f = functools.partial(harvest.harvest(apply_f, tag=module.ASSIGN), {})
    variables = init_f(init_key, *args, **kwargs)
    cau_jaxpr, (in_tree, out_tree) = trace_util.stage(
        cau_f, dynamic=True)(variables, *args, **kwargs)
    if name is None:
      variables = {
          k: module.variable(val, name=k, key=init_key)
          for k, val in variables.items()
      }
      return FunctionModule(variables, cau_jaxpr, in_tree, out_tree, name=name)
    else:
      mod = FunctionModule(variables, cau_jaxpr, in_tree, out_tree, name=name)
      if variables:
        return module.variable(mod, name=name, key=init_key)
      return mod

  return wrapped


@_function_register(api.spec)
def function_spec(f):
  """Returns a function that computes the output spec for function types."""

  def wrapped(*args, **kwargs):
    has_init_key = kwargs_util.check_in_kwargs(f, 'init_key')
    if has_init_key:

      def fun(init_key, *args, **kwargs):
        return f(*args, **kwargs, init_key=init_key)

      args = (random.PRNGKey(0),) + args
    else:
      fun = f
    in_specs = tree_util.tree_map(api.make_array_spec, args)
    out_specs = jax.eval_shape(fun, *in_specs, **kwargs)
    return tree_util.tree_map(api.make_array_spec, out_specs)

  return wrapped
