# Copyright 2021 The TensorFlow Probability Authors.
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
"""Enables writing custom effect handlers for probabilistic programs.

# Background

Oryx's PPL system comes with built-in transformations such as `joint_sample`
and `log_prob` which enable manipulating probabilistic programs. However,
the built-in transformations do not support many types of program manipulation
and transformation.

Consider, for example, a noncentering program transformation
which takes location-scale random variables in a program and adjusts them to
sample from a zero-one location-scale and then shift and scale the resulting
sample. This transformation can be important for well-conditioned MCMC and VI
but cannot be expressed as some combination of the built-in ones --
transformations like `joint_sample` and `log_prob` operate using tagged values
in a program but cannot actually adjust what happens at a particular sample
site.

# Custom interpreters for effect handling

A custom interpreter for a probabilistic program is the most general purpose
tool for building a transformation. A custom interpreter first traces a program
into a JAXpr and then executes the JAXpr with modified execution rules -- for
example, a (silly) custom interpreter could execute a program exactly as written
but could add 1 to the result of every call to `exp`. A more complicated
custom interpreter could apply a noncentering transformation to each
location-scale family random variable.

Writing a custom interpreter from scratch for each possible transformation,
however, can be tedious. Instead, in this module, we provide a simpler API to a
more restrictive, but still useful, set of custom interpreters. In particular,
we're interested in intercepting certain operations (effects) and
handling them by perhaps inserting new ones in their place (e.g. we could
intercept all normal sampling operations and replace them with their noncentered
versions) or updating a running state (e.g. accumulating `log_prob`s at each
sample site). This technique is called *effect handling* and is used by
libraries such as [Edward2](https://github.com/google/edward2) and
[Pyro](https://github.com/pyro-ppl/pyro).

# Usage

The main function is `make_effect_handler` which takes in a dictionary,
`handlers`, that maps JAX primitives to a handler function. The handler function
takes in the handler state and the usual arguments to the JAX primitive. It
can then return a new value (intercepting the regular call to the JAX primitive)
and an updated state which will be passed into later handlers. The
`effect_handler`'s default behavior when a handler is not provided for a
primitive is to execute the primitive normally and return an unchanged state.

The result of `make_effect_handler` is a function transformation that
when applied to a function `f` returns a transformed function that takes in an
initial state argument and returns the final state as an additional output.

## Example: add 1 to `exp`

This handler adds 1 to the output of each call to `exp`.
```
def exp_rule(state, value):
  # Leave state unchanged.
  return jnp.exp(value) + 1., state
add_one_exp = make_effect_handler({lax.exp_p: exp_rule})

# We can transform functions with `add_one_exp`.
def f(x):
  return jnp.exp(x) * 2.
add_one_exp(f)(None, 2.) # Executes (exp(x) + 1.) * 2.
```

## Example: count number of `add`s

This handler counts the number of times `add` is called.
```
def add_rule(count, x, y):
  # Increment count
  return x + y, count + 1
count_adds = make_effect_handler({lax.add_p: add_rule})

# We can transform functions with `count_adds`. The first argument to
# `count_adds` is input state, and it returns the final state as the second
# output.
def f(x):
  return x + 1. + 2.
count_adds(f)(0, 2.) # Returns (5., 2)
```
"""
import functools

from typing import Any, Dict, Callable, List, Sequence, Tuple, Union

from jax import api_util
from jax import core as jax_core
from jax import linear_util as lu
from jax import tree_util
from jax import util as jax_util
from jax.interpreters import xla

from oryx.core import trace_util

__all__ = [
    'make_effect_handler',
]

Value = Any
EffectHandler = Callable[..., Any]
VarOrLiteral = Union[jax_core.Var, jax_core.Literal]
Rules = Dict[jax_core.Primitive, EffectHandler]

_effect_handler_call_rules: Rules = {}


def register_call_rule(primitive: jax_core.Primitive,
                       rule: EffectHandler) -> None:
  _effect_handler_call_rules[primitive] = rule


class Environment:
  """Tracks variables and their values during a JAXpr's execution.

  JAXprs are an edge-list representation of a computation graph, with each
  node either being a `Var` or `Literal` and each edge being a `JaxprEqn`.
  A JAXpr interpreter keeps track of the values of intermediate `Var`s in an
  `Environment`; `Literal`s have fixed values regardless of the inputs to the
  JAXpr, so they do not need to be stored.

  Follows the environment convention found in `jax.core.eval_jaxpr`.
  """

  def __init__(self):
    self.env = {jax_core.unitvar: jax_core.unit}

  def read(self, var: VarOrLiteral) -> Value:
    """Reads a value from an environment."""
    if isinstance(var, jax_core.Literal):
      return var.val
    if var not in self.env:
      raise ValueError(f'Couldn\'t find {var} in environment: {self.env}')
    return self.env[var]

  def write(self, var: VarOrLiteral, val: Value) -> None:
    """Writes a value to an environment."""
    if isinstance(var, jax_core.Literal):
      return
    self.env[var] = val


def eval_jaxpr_with_state(jaxpr: jax_core.Jaxpr, rules: Rules,
                          consts: Sequence[Value], state: Value,
                          *args: Value) -> Tuple[List[Value], Value]:
  """Interprets a JAXpr and manages an input state with primitive rules.

  The implementation follows `jax.core.eval_jaxpr` closely; the main differences
  are:
  1. Rather than always calling `primitive.bind`, `eval_jaxpr_with_state`
     looks up a rule for the primitive in the provided `rules` dictionary first.
  2. A `state` value is provided that is threaded through the execution of the
     JAXpr and whose final value is returned as an additional output.

  Args:
    jaxpr: The JAXpr to be interpreted.
    rules: A `dict` that maps JAX primitives to functions that take in `(state,
      *args)` and return `(output, new_state)`.
    consts: A list of constant values corresponding to the JAXpr's constvars.
    state: The initial state for the interpreter.
    *args: A list of values that correspond to the JAXpr's invars.

  Returns:
    A list of outputs from the JAXpr and the final state.
  """
  env = Environment()

  jax_util.safe_map(env.write, jaxpr.constvars, consts)
  jax_util.safe_map(env.write, jaxpr.invars, args)

  for eqn in jaxpr.eqns:
    invals = jax_util.safe_map(env.read, eqn.invars)
    call_jaxpr, params = jax_core.extract_call_jaxpr(eqn.primitive, eqn.params)
    if call_jaxpr:
      call_rule = _effect_handler_call_rules.get(
          eqn.primitive,
          functools.partial(default_call_interpreter_rule, eqn.primitive))
      ans, state = call_rule(rules, state, invals, call_jaxpr, **params)
    elif eqn.primitive in rules:
      ans, state = rules[eqn.primitive](state, *invals, **params)
    else:
      ans = eqn.primitive.bind(*invals, **params)
    if eqn.primitive.multiple_results:
      jax_util.safe_map(env.write, eqn.outvars, ans)
    else:
      env.write(eqn.outvars[0], ans)
  return jax_util.safe_map(env.read, jaxpr.outvars), state


def default_call_interpreter_rule(primitive: jax_core.CallPrimitive,
                                  rules: Rules, state: Value,
                                  invals: Sequence[Value],
                                  call_jaxpr: jax_core.Jaxpr,
                                  **params: Any) -> Tuple[Value, Value]:
  """Handles simple call primitives like `jax_core.call_p`.

  When evaluating call primitives, the input `state` needs to be an additional
  input to the call primitive and it also needs to return an additional output
  `state`. After flattening the state along with the regular inputs, this
  handler recursively calls `eval_jaxpr_with_state` on the primitive's
  `call_jaxpr`. The output state from the recursive call is returned from the
  call primitive.

  Args:
    primitive: A `jax_core.CallPrimitive` such as `jax_core.call_p`.
    rules: A `dict` that maps JAX primitives to functions that take in `(state,
      *args)` and return `(output, new_state)`.
    state: The interpreter `state` value at the time of calling evaluating the
      call primitive.
    invals: The input values to the call primitive.
    call_jaxpr: The `jax_core.Jaxpr` that corresponds to the body of the call
      primitive.
    **params: The parameters of the call primitive.

  Returns:
    A tuple of the output of the call primitive and its output state.
  """
  # Recursively use the effect handler for the call primitive's JAXpr.
  fun = lu.wrap_init(
      functools.partial(eval_jaxpr_with_state, call_jaxpr, rules, []))

  state_invals, state_invals_tree = tree_util.tree_flatten((state, *invals))
  flat_fun, out_tree = api_util.flatten_fun_nokwargs(fun, state_invals_tree)
  ans_state = primitive.bind(flat_fun, *state_invals, **params)
  return tree_util.tree_unflatten(out_tree(), ans_state)


def xla_call_interpreter_rule(rules: Rules, state: Value,
                              invals: Sequence[Value],
                              call_jaxpr: jax_core.Jaxpr, *,
                              donated_invars: Tuple[bool],
                              **params: Any) -> Tuple[Value, Value]:
  """Handles the `jax.jit` call primitive.

  To handle the call primitive that appears when functions are JIT-ted, the rule
  needs to do some additional bookkeeping for the `donated_invars`, which is a
  tuple of `bool`s indicating whether or not the input value's buffer has been
  donated. Since the recursive call adds additional inputs to the call primitive
  the `donated_invars` need to be updated to indicate that the buffers have not
  been donated. Otherwise, this rule behaves exactly like the
  `default_call_interpreter_rule`.

  Args:
    rules: A `dict` that maps JAX primitives to functions that take in `(state,
      *args)` and return `(output, new_state)`.
    state: The interpreter `state` value at the time of calling evaluating the
      call primitive.
    invals: The input values to the call primitive.
    call_jaxpr: The `jax_core.Jaxpr` that corresponds to the body of the call
      primitive.
    donated_invars: A tuple of `bool`s for each input to the call primitive
      indicating if that input's buffer has been donated. See the documentation
      for `jax.jit` for more details.
    **params: The parameters of the call primitive.

  Returns:
    A tuple of the output of the call primitive and its output state.
  """
  # Adjust the `donated_invars` parameter to include `False` for each
  # of the new inputs to the call primitive.
  num_state = len(tree_util.tree_leaves(state))
  return default_call_interpreter_rule(
      xla.xla_call_p,
      rules,
      state,
      invals,
      call_jaxpr,
      donated_invars=(False,) * num_state + donated_invars,
      **params)


register_call_rule(xla.xla_call_p, xla_call_interpreter_rule)


def make_effect_handler(
    handlers: Dict[jax_core.Primitive, EffectHandler]
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
  """Returns a function transformation that applies a provided set of handlers.

  Args:
    handlers: A `dict` that maps JAX primitives to callback functions that take
      in `(state, *args)` and return `(output, new_state)`. When running the the
      transformed function, the execution of primitives in `handlers` will be
      delegated to the callback functions rather than their default execution
      rules.

  Returns:
    A function transformation that applies the handlers to an input function.
    The transformation takes in an input function and returns a transformed
    function that takes an additional initial `state` argument and returns
    an additional output `state`.
  """

  def transform(f: Callable[..., Any]) -> Callable[..., Any]:
    """Transforms a function according to a set of JAX primitive handlers."""

    def forward(state: Value, *args: Value, **kwargs) -> Tuple[Value, Value]:
      # First, trace the function into a JAXpr.
      closed_jaxpr, (_, out_tree) = trace_util.stage(
          f, dynamic=True)(*args, **kwargs)
      flat_args = tree_util.tree_leaves(args)
      # Interpret the JAXpr according to `handlers`.
      out, state = eval_jaxpr_with_state(closed_jaxpr.jaxpr, handlers,
                                         closed_jaxpr.literals, state,
                                         *flat_args)
      return tree_util.tree_unflatten(out_tree, out), state

    return forward

  return transform
