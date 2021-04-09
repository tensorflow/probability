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
"""Contains utilities for writing JAX patterns and rewriting JAX functions.

This module enables taking JAX functions (i.e. functions that take in JAX arrays
and manipulate them with JAX primitives) and using pattern matching and term
rewriting to transform them into new JAX functions.

# Primer: JAXprs

In order to pattern match against and rewrite JAX functions, we first convert a
JAX function into a JAXpr (JAX expression). JAXprs are an intermediate
representation that JAX often uses for function transformation. A JAXpr is an
[A-normal form](https://en.wikipedia.org/wiki/A-normal_form) representation of a
function where a set of equations operate on named values in an environment
sequentially. For example, the JAXpr corresponding to the JAX function
```python
def f(x):
  return jnp.exp(x) + 1.
```
is
```
{ lambda  ; a.
  let b = exp a
      c = add b 1.0
  in (c,) }
```
where `a` is the input to the JAXpr and `c` is the output. We can convert
functions into JAXprs by tracing the function with abstract values, as done in
`jax.make_jaxpr` or `oryx.core.trace_util.stage`.

# JAX expressions

We can think of JAXprs as an edge-list representation of a computation graph;
we convert them into an expression tree representation more amenable to pattern
matching and rewriting. We do this with a custom JAXpr interpreter (see
`jaxpr_to_expressions`) that returns a `JaxExpression` for each of the outputs
of the JAXpr. `JaxExpression`s are registered with the pattern matching and term
rewriting system, so we can write rules to transform them (see `rules`).

The basic elements of a `JaxExpression` parallel abstract values in JAX, a.k.a.
a `shape` and a `dtype`. We can also "evaluate" `JaxExpression`s using the
`evaluate` function, which will take in an expression and an environment (a
binding of names to JAX values) and produce a JAX array result. Evaluating
enables us to transform JAX functions into expressions and back into JAX
functions.

We'll quickly go over the core `JaxExpression`s that may comprise an expression
tree.

## `JaxVar`

A `JaxVar` corresponds to a named input to a JAXpr. Evaluating a `JaxVar`
is just looking up a name in the evaluation environment.

Example:
```python
a = JaxVar('x', shape=(), dtype=jnp.float32)
evaluate(a, {'x': 5.}) # ==> 5.
```

## `Literal`

A `Literal` corresponds to a literal value in a JAXpr, or values in a JAXpr that
are inlined into the equations, like scalars. Evaluating a `Literal` involves
returning the literal value it was instantiated with.

Example:
```python
a = Literal(1.)
evaluate(a, {}) # ==> 1.
```

## `Primitive`

Perhaps the most important expression is a `Primitive`, which corresponds to an
equation in a JAXpr. A `Primitive` is a combination of a JAX primitive, a tuple
of expression operands, and an instance of `Params`, which correspond to the
parameters of the JAX primitive. Evaluating a `Primitive` involves first
recursively evaluating the operands, then calling
`primitive.bind(*operands, **params)` to get the resulting JAX value.

Example:
```python
a = Primitive(lax.exp_p, (Literal(0.),), Params())
evaluate(a, {}) # ==> 1.
b = Primitive(lax.exp_p, (JaxVar('x', (), jnp.float32),), Params())
evaluate(b, {'x': 0.}) # ==> 1.
```

## `CallPrimitive`

JAXprs can contain other JAXprs by means of "call primitives", which correspond
to transformations like `jax.jit` and `jax.pmap`. These call primitives
encapsulate another JAXpr, which is evaluated when the call primitive is
evaluated. A `CallPrimitive` expression recursively converts the nested JAXpr
into an expression and is evaluated by rebinding names and recursively
evaluating its containing expression.

Example:
```python
expr = CallPrimitive(
    primitive=xla.xla_call_p,
    operands=(JaxVar('a', (), jnp.float32),),
    expression=(
        Primitive(lax.exp_p, (JaxVar('b', (), jnp.float32),), Params()),),
    params=Params(),
    variable_names=['b'])

evaluate(expr, {'a': 0.}) # ==> 1.
```

## Other relevant expressions

* `Part` - used to handle primitives with multiple outputs. `Part(expr, i)`
  corresponds to indexing into the multi-part expression `expr` with index `i`.

* `BoundExpression` - used to pre-bind some names to values in an expression
  so the names don't have to be bound when calling `evaluate`. For example, this
  is used to encapsulate a JAXpr and any constant values in it.
"""
import functools

from typing import Any, Dict, Iterator, Sequence, Tuple, Union

import dataclasses
import jax
from jax import core as jax_core
from jax import linear_util as lu
from jax import util as jax_util
import jax.numpy as jnp
import numpy as np

from oryx.experimental.matching import matcher
from oryx.experimental.matching import rules

Expr = matcher.Expr
Bindings = matcher.Bindings
Continuation = matcher.Continuation
Success = matcher.Success
Env = Dict[str, Any]


@functools.singledispatch
def evaluate(obj, env: Env) -> Any:
  """Evaluates expressions into JAX values.

  `evaluate` is a single-dispatch function whose default implementation is to
  raise a `NotImplementedError`. To register evaluation functions for particular
  types, use `evaluate.register`.

  Args:
    obj: an object to evaluate.
    env: a dictionary mapping string names to JAX values

  Returns:
    The JAX value that is the result of evaluating the object.
  """
  del env
  raise NotImplementedError(f'{type(obj)} not registered with `evaluate`.')


@evaluate.register(float)
@evaluate.register(int)
@evaluate.register(np.ndarray)
@evaluate.register(jax.xla.DeviceArray)
@evaluate.register(jax_core.Tracer)
def evaluate_value(value, env: Env) -> Any:
  """Default evaluate function for numerical types."""
  del env
  return value


@evaluate.register(tuple)
def evaluate_tuple(expr: Tuple[Any], env: Env) -> Tuple[Any]:
  return tuple(map(lambda e: evaluate(e, env), expr))


class JaxExpression(rules.Expression):
  """A node in an expression tree.

  `JAXExpression`s are subclasses of `rules.Expression` so if the `tree_map`
  and `tree_children` methods are implemented, it is compatible with the
  rewrite interface in `rules`. To be compatible with the pattern matching
  interface in `matcher`, the `match` method needs to be implemented too.
  """

  @property
  def shape(self):
    raise NotImplementedError

  @property
  def dtype(self):
    raise NotImplementedError

  def evaluate(self, env: Env) -> Any:
    raise NotImplementedError


@evaluate.register(JaxExpression)
def evaluate_jax_expression(jax_expr, env: Env) -> Any:
  """Evaluates a `JaxExpression` in an environment."""
  return jax_expr.evaluate(env)


@dataclasses.dataclass(frozen=True)
class Literal(JaxExpression):
  """A expression that evaluates to a provided scalar literal value.

  Attributes:
    value: a scalar literal value
  """
  value: Any

  @property
  def shape(self):
    return ()

  @property
  def dtype(self):
    return jnp.array(self.value).dtype

  def match(self, expr: Expr, bindings: Bindings,
            succeed: Continuation) -> Success:
    if not isinstance(expr, Literal):
      return
    yield from matcher.matcher(self.value)(expr.value, bindings, succeed)

  def tree_map(self, _):
    return self

  def tree_children(self):
    yield from ()

  def evaluate(self, env: Env) -> Any:
    del env
    return self.value

  def __str__(self):
    return f'{self.value}'

  def __hash__(self):
    return object.__hash__(self)

  def __eq__(self, other):
    if not isinstance(other, Literal):
      return False
    return self.value == other.value


class JaxVar(JaxExpression):
  """An expression that looks up a provided name in the environment.

  Attributes:
    name: The string name for the `JaxVar` used to look up the `JaxVar`'s value
      in an environment.
  """

  def __init__(self, name: str, shape: Tuple[Any], dtype: jnp.dtype):
    """Constructor for `JaxVar`.

    Args:
      name: a string name corresponding the key used to lookup this `JaxVar`'s
        value in an environment.
      shape: a tuple of integers corresponding to the shape of the `JaxVar`.
      dtype: a JAX `dtype` object for this `JaxVar`'s dtype.
    """
    super().__init__()
    self.name = name
    self._shape = shape
    self._dtype = dtype

  @property
  def shape(self):
    return self._shape

  @property
  def dtype(self):
    return self._dtype

  def match(self, expr: Expr, bindings: Bindings,
            succeed: Continuation) -> Success:
    if not isinstance(expr, JaxVar):
      return
    jaxvar_matcher = matcher.matcher((self.name, self.shape, self.dtype))
    yield from jaxvar_matcher((expr.name, expr.shape, expr.dtype), bindings,
                              succeed)

  def tree_map(self, _) -> 'JaxVar':
    return self

  def tree_children(self) -> Iterator[Expr]:
    yield from ()

  def evaluate(self, env: Env) -> Any:
    if self.name not in env:
      raise ValueError(f'Cannot find name `{self.name}` in environment: {env}')
    return env[self.name]

  def __hash__(self):
    return hash((self.name, self.shape, self.dtype))

  def __str__(self):
    return self.name


class Params(matcher.Pattern):
  """An immutable dictionary used to represent parameters of JAX primitives."""

  def __init__(self, params: Dict[str, Any] = None, **kwargs: Any):
    """The constructor for a `Params` object.

    A `Params` object is an immutable dictionary, meant to encapsulate the
    parameters to JAX primitives. It is duck-typed to behave like a dictionary,
    but is hashable and can be used in pattern matching.

    Args:
      params: an optional dictionary that the `Params` object will be
        initialized from.
      **kwargs: keyword arguments that will populate the `Params`.
    """
    if params is None:
      params = {}
    params = dict(params, **kwargs)
    self.sorted_keys = tuple(sorted(params.keys()))
    self.sorted_values = tuple(params[k] for k in self.sorted_keys)
    self._dict = params

  def __getitem__(self, key):
    return self._dict[key]

  def __contains__(self, key):
    return key in self._dict

  def __setitem__(self, key, value):
    raise ValueError('`Params` are immutable.')

  def __iter__(self):
    yield from self.sorted_keys

  def keys(self):
    yield from self.sorted_keys

  def match(self, expr: Expr, bindings: Bindings,
            succeed: Continuation) -> Success:
    if not isinstance(expr, Params):
      return
    yield from matcher.matcher((self.sorted_keys, self.sorted_values))(
        (expr.sorted_keys, expr.sorted_values), bindings, succeed)

  def __eq__(self, other):
    return (self.sorted_keys == other.sorted_keys and
            self.sorted_values == other.sorted_values)

  def __hash__(self):
    return hash((self.sorted_keys, self.sorted_values))

  def __str__(self):
    return str(self._dict)

  def __repr__(self):
    return repr(self._dict)


@dataclasses.dataclass
class Primitive(JaxExpression):
  """A `JAXExpression` corresponding to a `jax.core.Primitive`.

  A `Primitive` encapsulates a `jax.core.Primitive`, its arguments and its
  parameters.

  Attributes:
    primitive: A JAX primitive.
    operands: A tuple of expressions that are evaluated and passed into the
      primitive when a `Primitive` is evaluated.
    params: A `Params` object that holds the parameters passed into the JAX
      primitive when a `Primitive` is evaluated.
  """
  primitive: Union[matcher.Pattern, jax_core.Primitive]
  operands: Tuple[Any]
  params: Params

  def __post_init__(self):
    self._shape_dtype = None

  @property
  def shape_dtype(self):
    if self._shape_dtype is not None:
      return self._shape_dtype
    fun = functools.partial(self.primitive.bind, **self.params)
    self._shape_dtype = jax.eval_shape(fun, *self.operands)
    return self._shape_dtype

  @property
  def shape(self):
    if not isinstance(self.primitive, jax_core.Primitive):
      raise ValueError('Cannot compute shape of pattern.')
    if self.primitive.multiple_results:
      return tuple(a.shape for a in self.shape_dtype)
    return self.shape_dtype.shape

  @property
  def dtype(self):
    if not isinstance(self.primitive, jax_core.Primitive):
      raise ValueError('Cannot compute dtype of pattern.')
    if self.primitive.multiple_results:
      return tuple(a.dtype for a in self.shape_dtype)
    return self.shape_dtype.dtype

  def match(self, expr: Expr, bindings: Bindings,
            succeed: Continuation) -> Success:
    if not isinstance(expr, Primitive):
      return
    yield from matcher.matcher((self.primitive, self.operands, self.params))(
        (expr.primitive, expr.operands, expr.params), bindings, succeed)

  def tree_map(self, fn) -> 'Primitive':
    return Primitive(self.primitive, tuple(map(fn, self.operands)), self.params)

  def tree_children(self) -> Iterator[Expr]:
    yield from self.operands

  def evaluate(self, env: Env) -> Any:
    operands = evaluate(self.operands, env)
    if not isinstance(self.primitive, jax_core.Primitive):
      raise ValueError('Cannot evaluate expression with patterns.')
    return self.primitive.bind(*operands, **self.params)

  def __str__(self):
    return f'({self.primitive} {" ".join(map(str, self.operands))})'

  def __eq__(self, other):
    return (self.primitive == other.primitive and
            self.operands == other.operands and self.params == other.params)

  def __hash__(self):
    return hash((self.primitive, self.operands, self.params))


@dataclasses.dataclass(frozen=True)
class Part(JaxExpression):
  """Used to select the outputs of JAX primitives with multiple outputs.

  When a JAX primitive has `multiple_results = True`, it returns several outputs
  when called. To represent multiple outputs in an expression tree, we wrap
  the output of a multiple-output primitive with `Part` with an index for each
  of its outputs. `Part` is primarily used with `CallPrimitive`s, which always
  have multiple outputs.

  Attributes:
    operand: An expression that can be indexed into with an integer
      i.e. `operand[i]`.
    index: The index that is used when accessing the operand.
  """
  operand: Expr
  index: int

  @property
  def shape(self):
    return self.operand.shape[self.index]

  @property
  def dtype(self):
    return self.operand.dtype[self.index]

  def match(self, expr: Expr, bindings: Bindings,
            succeed: Continuation) -> Success:
    if not isinstance(expr, Part):
      return
    yield from matcher.matcher((self.operand, self.index))(
        (expr.operand, expr.index), bindings, succeed)

  def tree_map(self, fn) -> 'Part':
    return Part(fn(self.operand), self.index)

  def tree_children(self) -> Iterator[Expr]:
    yield self.operand

  def evaluate(self, env: Env) -> Any:
    return evaluate(self.operand, env)[self.index]

  def __hash__(self):
    return hash((self.operand, self.index))

  def __str__(self):
    return f'{self.operand}[{self.index}]'

  __repr__ = __str__


@dataclasses.dataclass(frozen=True)
class BoundExpression(JaxExpression):
  """Represents JAX expressions with closed over constants.

  A `BoundExpression` enables pinning `JaxVar`s in an expression to fixed
  values, removing the need to bind them to values when the `BoundExpression` is
  evaluated. Conceptually this is equivalent to a `jax.core.ClosedJaxpr`.

  Attributes:
    expressions: A sequence of expressions that are evaluated to produce the
      result of this `BoundExpression`.
    consts: A dictionary mapping string names (corresponding to `JaxVar`s in
      `expressions`) to their JAX values.
  """
  expressions: Sequence[Expr]
  consts: Dict[str, Expr]

  @property
  def shape(self):
    return tuple(a.shape for a in self.expressions)

  @property
  def dtype(self):
    return tuple(a.dtype for a in self.expressions)

  def match(self, expr: Expr, bindings: Bindings,
            succeed: Continuation) -> Success:
    if not isinstance(expr, BoundExpression):
      return
    yield from matcher.matcher(self.expressions)(expr.expressions, bindings,
                                                 succeed)

  def tree_map(self, fn) -> 'BoundExpression':
    return BoundExpression(tuple(map(fn, self.expressions)), self.consts)

  def tree_children(self) -> Iterator[Expr]:
    yield from self.expressions

  def evaluate(self, env: Env) -> Any:
    """Evaluates using an environment augmented with constants."""
    return evaluate(self.expressions, dict(self.consts, **env))

  def __str__(self):
    return f'({" ".join(map(str, self.expressions))})'

  def __hash__(self):
    return hash((self.expressions, tuple(sorted(self.consts.keys()))))

  __repr__ = __str__


@dataclasses.dataclass(frozen=True)
class CallPrimitive(JaxExpression):
  """Encapsulates JAX `CallPrimitive`s like `jit` and `pmap`.

  Attributes:
    primitive: A JAX call primitive.
    operands: A sequence of expressions that are evaluated and passed as inputs
      to the primitive when the `CallPrimitive` is evaluated.
    expression: The expression that corresponds to the body of the call
      primitive. The `operands` are bound to the `variable_names` in an
      environment and the expression is evaluated with that environment.
    params: A `Params` object corresponding to the parameters of the call
      primitive.
    variable_names: A sequence of string names that are used as keys for the
      operands in the environment `expression` is evaluated in.
  """
  primitive: jax_core.Primitive
  operands: Sequence[Any]
  expression: Any
  params: Params
  variable_names: Sequence[str]

  @property
  def shape(self):
    if self.primitive.multiple_results:
      return tuple(a.shape for a in self.expression)
    return self.expression.shape

  @property
  def dtype(self):
    if self.primitive.multiple_results:
      return tuple(a.dtype for a in self.expression)
    return self.expression.dtype

  def match(self, expr: Expr, bindings: Bindings,
            succeed: Continuation) -> Success:
    if not isinstance(expr, CallPrimitive):
      return
    yield from matcher.matcher(
        (self.primitive, self.operands, self.expression, self.params,
         self.variable_names))((expr.primitive, expr.operands, expr.expression,
                                expr.params, expr.variable_names), bindings,
                               succeed)

  def tree_map(self, fn) -> 'CallPrimitive':
    return CallPrimitive(self.primitive, tuple(map(fn, self.operands)),
                         fn(self.expression), self.params, self.variable_names)

  def tree_children(self) -> Iterator[Expr]:
    yield from self.operands

  def evaluate(self, env: Env) -> Any:
    operands = evaluate(self.operands, env)

    def f(*args):
      sub_env = dict(jax_util.safe_zip(self.variable_names, args))
      return evaluate(self.expression, sub_env)

    fun = lu.wrap_init(f)
    return self.primitive.bind(fun, *operands, **self.params)

  def __str__(self):
    return f'({self.primitive} {self.expression} {self.operands})'

  def __hash__(self):
    return hash((self.primitive, self.operands, self.expression, self.params,
                 self.variable_names))
