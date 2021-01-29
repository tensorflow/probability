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
"""A basic pattern matching system.

The following is a self-contained pattern matching system based on a pattern
matcher written for
[MIT class 6.945](https://groups.csail.mit.edu/mac/users/gjs/6.945/) in 2009.
The API closesly mirrors the Scheme API found in
[`rules`](https://github.com/axch/rules) by Alexey Radul. The star matching
implementation  is based on the one found in
[Autoconj](https://github.com/google-research/autoconj)
by Matthew D. Hoffman, Matthew J. Johnson, and Dustin Tran. Specifically, the
signature for `matcher` resembles the corresponding signature in `rules` and
`autoconj`, but instead of returning a sequences matches from `matcher`
directly, this library uses generators to return an iterator over matches.

This library is related to the
[`matchpy`](https://github.com/HPAC/matchpy) library, but additionally supports
the `Choice` and `Star` patterns which aren't yet supported in `matchpy`, along
with a different approach to extensibility (single-dispatch functions). However,
`matchpy` additionally supports faster many-to-one matchers and native support
for associative and commutative operators.

# Usage

By default, expressions are matched by Python equality. To build more
sophisticated patterns that can match expressions that are not just equal
to each other, we provide match combinators, or "higher-order patterns", like
`Not`, `Star` and `Choice`.

## `match`

The `match` method takes in a pattern and an expression to match.
If the expression does not match the pattern, a `MatchError` is thrown.
`match` returns a dictionary of "bindings", where a binding is a mapping of a
name to a value. The bindings can be substituted into the pattern in order to
make the pattern and expression equal. The `Var` pattern described below
demonstrates how to create bindings in a match.


```python
match(1, 1)  # ==> {}
match('abc', 'abc')  # ==> {}
match((1, 2), (1, 2))  # ==> {}
match({'a': 1, 'b': 2}, {'a': 1, 'b': 2})  # ==> {}
match(1, 2) # ==> MatchError!
```

A `tuple` pattern matches a `tuple` expression if each of its elements matches.
Similarly, a `dict` pattern matches a `dict` expression if their keys are
the same and the corresponding values match.

Also provided are `is_match(pattern, expr)`, which returns `True` if the
expression matches the pattern and `False` if not, and
`match_all(pattern, expr)`, which returns an iterator over all possible
bindings that successfully make the pattern and expression equal.

## `Var`

We can use `Var`s to define wildcard values that can match any expression.
If a pattern containing `Var`s matches an expression, `match` will return a
dictionary of bindings that maps the name of the `Var`s to the expressions that
makes the pattern and expression equal. If the same `Var` appears multiple times
in a pattern, the value needs to be the same across each of the instances for a
successful match.

```python
match(Var('x'), 1)  # ==> {'x': 1}
match((Var('x'), Var('x')), (1, 1))  # ==> {'x': 1}
match((Var('x'), Var('x')), (1, 2))  # ==> MatchError!
match({'a': 1, 'b': Var('x'}), {'a': 1, 'b': 2})  # ==> {'x': 2}
```

We can apply restrictions to `Var`s to condition the types of values they can
match.

```python
is_even = lambda x: x % 2 == 0
x = Var('x', restrictions=[is_even])
match(x, 2) # ==> {'x': 2}
match(x, 3) # ==> MatchError!
```

`Dot` is a `Var` that will match any expression without binding a name
to a value. It is equivalent to `Var(None)`.

```python
match(Dot, 2) # ==> {}
match(Dot, (1, 2, 3)) # ==> {}
```

## `Not`

`Not` takes a pattern and successfully matches an expression if the provided
pattern does not match it.

```python
match(Not(1), 2) # ==> {}
match(Not(2), 2) # ==> MatchError!
match(Not(Var('x')), 2) # ==> MatchError!
```

## `Choice`

`Choice` takes in a sequence of patterns and tries matching each of them to an
expression, and on first success, returns the successful match. `Choice`
also takes in an optional name that is used when binding the successful match.

```python
match(Choice(1, 2), 1) # ==> {}
match(Choice(1, 2), 2) # ==> {}
match(Choice(Var('x'), 2), 2) # ==> {'x': 2}
match(Choice((1, 2), (2, 1), name='x'), (2, 1)) # ==> {'x': (2, 1)}
```
# Extending the system

The pattern matching library is based on a single-dispatch function, `matcher`.
`matcher` takes in a "pattern", which can be any Python object. It returns a
function `match(expr, bindings, succeed)`, where `expr` is an expression to
match, `bindings` is a dictionary mapping names to values, and `succeed`
is a function that takes in `bindings` (that could be extended by the match)
and returns an iterator over results.

To extend the system to include new patterns, there are two options:

1.  We can register a custom type with `matcher`. `matcher` is a
single-dispatch function, so we can override its behavior for custom types.
By default, `matcher` returns a `match` that only succeeds if the pattern and
expression are equal according to Python equality. To extend it for a custom
type `Foo` that we'd like to match against, we can call
`matcher.register(Foo)(custom_matcher)` to define a custom matcher for `Foo`
objects. This is how we define matchers for tuples and dictionaries.

2. Alternatively, we provide the `Pattern` class which is the parent class
of the various combinators (`Choice`, `Not`, etc.). By subclassing
`Pattern` and implementing the  `match(self, expr, bindings, succeed)` method,
the class is automatically registered with `matcher`.
"""
import functools

from typing import Any, Callable, Dict, Iterator, Optional, Sequence, Tuple, TypeVar

import dataclasses

__all__ = [
    'Choice',
    'Dot',
    'is_match',
    'match',
    'match_all',
    'matcher',
    'MatchError',
    'Not',
    'Pattern',
    'Var',
]

T = TypeVar('T')
Expr = Any
Bindings = Dict[str, Expr]
Success = Iterator[T]
Continuation = Callable[[Bindings], Success]
TupleSegment = Tuple[Any]
StarContinuation = Callable[[Bindings, TupleSegment], Success]
Matcher = Callable[[Expr, Bindings, Continuation], Success]

id_success = lambda x: (yield x)


@functools.singledispatch
def matcher(pattern: Any) -> Matcher:
  """Returns a function that determines if an expression matches the given pattern.

  `matcher` is a single-dispatch function so its behavior can be overridden
  for any type. Its default implementation uses Python equality to determine if
  a pattern matches an expression.

  Args:
    pattern: A Python object that describes a set of expressions to match.

  Returns:
    A function that takes in three arguments: 1. an expression `expr`
    to be matched against, 2. a dictionary mapping names to values `bindings`
    that encapsulate the results of matches made up to this point, and 3.
    `succeed`, a continuation function that takes in `bindings` and returns an
    iterator. The function returns an iterator over bindings. By default, the
    match is successful if `expr` matches `pattern` using Python equality. For
    more details, refer to the `matcher` module docstring.
  """

  def default_match(expr: Expr, bindings: Bindings,
                    succeed: Continuation) -> Success:
    if expr == pattern:
      yield from succeed(bindings)

  return default_match


class MatchError(Exception):
  """Raised when unable to find a pattern match."""


def is_match(pattern: Any, expr: Expr) -> bool:
  """Returns whether or not an expression matches a pattern."""
  for _ in matcher(pattern)(expr, {}, id_success):
    return True
  return False


def match(pattern: Any, expr: Expr) -> Bindings:
  """Returns a single match for pattern and expression or errors otherwise."""
  for bindings in matcher(pattern)(expr, {}, id_success):
    return bindings
  raise MatchError(f'No match found. Pattern: {pattern}, Expression: {expr}')


def match_all(pattern: Any, expr: Expr) -> Iterator[Bindings]:
  """Returns an iterator over all bindings matching a pattern to an expression."""
  yield from matcher(pattern)(expr, {}, id_success)


class Pattern:
  """Objects that are auto-registered with `matcher`."""

  def match(self, expr: Expr, bindings: Bindings,
            succeed: Continuation) -> Success:
    raise NotImplementedError


@matcher.register(Pattern)
def pattern_matcher(pattern):
  return pattern.match


@dataclasses.dataclass(frozen=True)
class Not(Pattern):
  """Matches successfully if the input pattern is not matched."""
  pattern: Any

  def match(self, expr: Expr, bindings: Bindings,
            succeed: Continuation) -> Success:
    for _ in matcher(self.pattern)(expr, bindings, id_success):
      return
    yield from succeed(bindings)

  def __str__(self):
    return f'~{self.pattern}'


@dataclasses.dataclass(frozen=True)
class Var(Pattern):
  """Adds a named "wildcard" pattern that can match anything.

  Attributes:
    name: A `str` name for the `Var`. When the `Var` is successfully matched
      to an expression, a new binding (name-expression pair) will be created
      in the bindings for subsequent matches. If `name` is `None`, no binding
      will be created.
    restrictions: An optional sequence of predicate functions that take in the
      expression to be matched against and return a `bool`. The match is only
      successful if all predicate functions return `True`. By default,
      `restrictions` is empty, so a `Var` will match any expression.
  """
  name: Optional[str]
  restrictions: Sequence[Callable[[Expr], bool]] = ()

  def ok(self, expr: Expr) -> bool:
    return all(restriction(expr) for restriction in self.restrictions)

  def match(self, expr: Expr, bindings: Bindings,
            succeed: Continuation) -> Success:
    if not self.ok(expr):
      return
    if self.name is None:
      yield from succeed(bindings)
      return
    # Name is not yet in bindings, so we add it.
    if self.name not in bindings:
      yield from succeed(dict(bindings, **{self.name: expr}))
      return
    bound_value = bindings[self.name]
    if bound_value == expr:
      yield from succeed(bindings)

  def __str__(self):
    if self.name is None:
      return '?'
    return f'?{self.name}'


Dot = Var(None)


class Choice(Pattern):
  """Tries to match any of a sequence of patterns."""

  def __init__(self, *patterns: Any, name: Optional[str] = None):
    self.patterns = patterns
    self.name = name

  def match(self, expr: Expr, bindings: Bindings,
            succeed: Continuation) -> Success:
    if self.name is not None:
      bindings = dict(bindings, **{self.name: expr})
    for pattern in self.patterns:
      matches = tuple(matcher(pattern)(expr, bindings, succeed))
      if matches:
        yield from matches
        return

  def __hash__(self):
    return hash(self.patterns)

  def __eq__(self, other):
    if not isinstance(other, Choice):
      return False
    return self.patterns == other.patterns

  def __str__(self):
    return f'(choice {" ".join(map(str, self.patterns))})'


@matcher.register(tuple)
def tuple_matcher(pattern: Tuple[Any]):
  """Returns a matcher for a given tuple pattern."""

  def tuple_match(expr: Expr, bindings: Bindings,
                  succeed: Continuation) -> Success:
    if not isinstance(expr, tuple):
      return
    if not expr:
      if not pattern:
        # Both expr and pattern are empty, so a successful match
        yield from succeed(bindings)
      return
    if not pattern:
      # If pattern is () and expression is not, a failed match
      return

    # Match the first element and then recursively match the rest
    def rest_succeed(bindings):
      rest_match = matcher(pattern[1:])
      yield from rest_match(expr[1:], bindings, succeed)

    first_match = matcher(pattern[0])
    yield from first_match(expr[0], bindings, rest_succeed)

  return tuple_match


@matcher.register(dict)
def dict_matcher(pattern: Dict[Any, Any]):
  """Returns a matcher for dictionaries treating them as sorted item tuples."""
  pattern_keys = tuple(sorted(pattern.keys()))
  pattern_values = tuple(pattern[k] for k in pattern_keys)

  def dict_match(expr: Expr, bindings: Bindings,
                 succeed: Continuation) -> Success:
    if not isinstance(expr, dict):
      return
    expr_keys = tuple(sorted(expr.keys()))
    if expr_keys != pattern_keys:
      return
    expr_values = tuple(expr[k] for k in expr_keys)
    yield from matcher(pattern_values)(expr_values, bindings, succeed)

  return dict_match
