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

A `Sequence` pattern matches a `Sequence` expression if each of its elements
matches.
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

## `Star`, `Plus`, and `Segment`

The `Star`, `Plus` and `Segment` patterns are used when matching sequences.
Specifically, they can match variable-length slices of a sequence. `Star` takes
in a pattern and matches a sequence slice if all of its elements match the
pattern.
`Star` must be used inside of a sequence.

```python
match((Star(1),), ()) # ==> {}
match((Star(1),), (1,)) # ==> {}
match((Star(1),), (1, 1)) # ==> {}
match((Star(1),), (1, 2)) # ==> MatchError!
match((3, Star(1), 2), (3, 1, 1, 1, 2)) # ==> {}
```

`Plus` is exactly like `Star`, but will not match zero-length slices.

```python
match((Plus(1),), (1,)) # ==> {}
match((Plus(1),), ()) # ==> MatchError!
```

`Star` and `Plus` optionally take in a `name` that will be used to bind the
matched slice. Like `Var`, if the same name appears in multiple places, the same
bound value must be able to be substituted into the pattern in each element
of the slice to make the expression match.

```python
match((Star(1, name='x'),), (1, 1, 1)) # ==> {'x': (1, 1, 1)}
match((Star(1, name='x'), 2), (1, 1, 2)) # ==> {'x': (1, 1)}
match((Star(Dot, name='x'), 4), (1, 2, 3, 4))  # ==> {'x': (1, 2, 3)}
```

If the pattern inside of a `Star` has any names in it (from `Var`s or
nested `Star`s), the same value needs to match for the entire slice. If name
`'x'` needs to be bound to `1` to make an element of a slice match, it needs to
be `1` for every subsequent element of the sequence. For example, in the
following
snippets, `x` cannot be bound to multiple values to make the match succeed.

```python
match((Star(Var('x')),), (1, 1)) # ==> {'x': 1}
match((Star(Var('x')),), (1, 2)) # ==> MatchError!
```


Alternatively, we can "accumulate" the bindings for a name inside of a `Star`
by providing the name to the `accumulate` argument to `Star`. `Star` tries to
match each element of a slice to its pattern. For `Var`s inside of a `Star`,
this means that the value bound to its name must match across the slice.
When we accumulate for a name inside of a `Star`, rather than enforcing that the
match is the same across the slice, we collect the individual matches into a
sequence and bind the name to the sequence.
```python
match((Star(Var('x'), accumulate=['x']),), (1, 2, 3)) # ==> {'x': (1, 2, 3)}
match((Star((Var('x'), Var('y')), accumulate=['y']),), ((1, 2), (1, 3)))
# ==> {'x': 1, 'y': (2, 3)}
```

Another example of using an accumulating `Star` is to match expressions that
exhibit the distributive property, i.e. `a * b + a * c`.

```
Mul = lambda *args: ('*', *args)
Add = lambda *args: ('+', *args)
distributive = Add(Star(Mul(Var('x'), Var('y')), accumulate=['y']))
match(distributive, Add(Mul('a', 'b'), Mul('a', 'c')))
# ==> {'x': 'a', 'y': ('b', 'c')}
match(distributive, Add(Mul('a', 'b'), Mul('c', 'd'))) # MatchError!
```

A `Segment` is shorthand for a named `Star` that has the `Dot` pattern,
meaning it matches slices of a sequence regardless of the individual values.
Specifically, `Segment(name)` matches the same expressions as
`Star(Dot, name=name)`.
```python
match((Segment('a'),), (1, 1, 1)) # ==> {'a': (1, 1, 1)}
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
objects. This is how we define matchers for sequences and dictionaries.

2. Alternatively, we provide the `Pattern` class which is the parent class
of the various combinators (`Choice`, `Not`, `Star`, etc.). By subclassing
`Pattern` and implementing the  `match(self, expr, bindings, succeed)` method,
the class is automatically registered with `matcher`.
"""
import functools

from typing import Any, Callable, Dict, Iterator, Optional, Sequence, TypeVar

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
    'Plus',
    'Segment',
    'Star',
    'Var',
]

T = TypeVar('T')
Expr = Any
Bindings = Dict[str, Expr]
Success = Iterator[T]
Continuation = Callable[[Bindings], Success]
SequenceSegment = Sequence[Any]
StarContinuation = Callable[[Bindings, SequenceSegment], Success]
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
  if isinstance(pattern, Star):
    raise ValueError('`Star` pattern must be inside of a sequence.')
  for _ in matcher(pattern)(expr, {}, id_success):
    return True
  return False


def match(pattern: Any, expr: Expr) -> Bindings:
  """Returns a single match for pattern and expression or errors otherwise."""
  if isinstance(pattern, Star):
    raise ValueError('`Star` pattern must be inside of a sequence.')
  for bindings in matcher(pattern)(expr, {}, id_success):
    return bindings
  raise MatchError(f'No match found. Pattern: {pattern}, Expression: {expr}')


def match_all(pattern: Any, expr: Expr) -> Iterator[Bindings]:
  """Returns an iterator over all bindings matching a pattern to an expression."""
  if isinstance(pattern, Star):
    raise ValueError('`Star` pattern must be inside of a sequence.')
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
    name: A `str` name for the `Var`. When the `Var` is successfully matched to
      an expression, a new binding (name-expression pair) will be created in the
      bindings for subsequent matches. If `name` is `None`, no binding will be
      created.
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


@dataclasses.dataclass(frozen=True)
class Star(Pattern):
  """A pattern for repeated sub-patterns inside of a sequence.

  Attributes:
    pattern: an object that will be matched against elements of a sequence.
    name: an optional `str` name to bind the result of the star match.
    accumulate: a sequence of `str` names corresponding to `Var`s in `pattern`
      that will be accumulated into a sequence instead of having to match across
      the elements of the sequence.
    greedy: a `bool` that sets whether or not the `Star` greedily matches a
      sequence. A greedy `Star` will try to match slices starting from the
      largest possible and then trying smaller ones. A non-greedy `Star` will
      match slices starting from the smallest possible.
      Default: `False`.
    plus: a `bool`, if `True` requires matches of length one or more and if
      `False` allows zero-length matches.
      Default: `False`
  """
  pattern: Any
  name: Optional[str] = None
  accumulate: Sequence[str] = ()
  greedy: bool = False
  plus: bool = False

  def bind(self, bindings: Bindings, value: Any) -> Bindings:
    if self.name is not None:
      return dict(bindings, **{self.name: value})
    return bindings

  def accumulate_value(self, bindings: Bindings, name: str,
                       value: Any) -> Bindings:
    accum = bindings.get(name, ())
    accum += (value,)
    return dict(bindings, **{name: accum})

  def accumulate_match(self, expr: Expr, bindings: Bindings,
                       succeed: Continuation) -> Success:
    """Matches each element of a sequence to this `Star`'s pattern.

    Iteratively matches each element of `expr` with `self.pattern`. For any
    created as the result of each match, they are accumulated if the names
    are in the `Star` pattern's `accumulate` property. If not, the bindings need
    to match over all elemnets of `expr`.

    Args:
      expr: An expression to match.
      bindings: A dictionary mapping string names to values representing the
        results of previous matches.
      succeed: A function that when passed in `bindings` returns a generator
        over results.

    Yields:
      The results of the `succeed` continuation function, augmented with
      bindings corresponding to matches made over the course of the accumulate
      match.
    """
    if not expr:
      yield from succeed(bindings)
      return
    elem, rest = expr[0], expr[1:]
    non_acc_bindings = {
        k: v for k, v in bindings.items() if k not in self.accumulate
    }
    for m in matcher(self.pattern)(elem, non_acc_bindings, id_success):
      new_bindings = bindings
      for k, v in m.items():
        if k in self.accumulate:
          new_bindings = self.accumulate_value(new_bindings, k, v)
        else:
          new_bindings = dict(new_bindings, **{k: v})
      yield from self.accumulate_match(rest, new_bindings, succeed)

  def match(self, expr: Expr, bindings: Bindings,
            succeed: StarContinuation) -> Success:
    """Matches the `Star` pattern against an expression.

    Constructs all splits of the expression and performs an `accumulate_match`
    on each of the left halves. The right half is matched using sequence
    matching.

    Args:
      expr: An expression to match.
      bindings: A dictionary mapping string names to values representing the
        results of previous matches.
      succeed: A function that when passed in `bindings` returns a generator
        over results.

    Yields:
      The results of the `succeed` continuation function, augmented with
      bindings corresponding to matches made over the course of the Star match.
    """
    if not isinstance(expr, Sequence):
      return
    # If name appears in bindings, we have already matched and need to verify
    # that bound value matches the current expression
    if self.name in bindings:
      bound_value = bindings[self.name]
      if bound_value == expr[:len(bound_value)]:
        yield from succeed(bindings, expr[len(bound_value):])
      return

    slice_indices = range(int(self.plus), len(expr) + 1)
    if self.greedy:
      slice_indices = slice_indices[::-1]
    for i in slice_indices:
      # For each possible slicing of the expression, we try to match the
      # subpattern for each element, while accumulating matched bindings.
      segment, rest = expr[:i], expr[i:]
      for m in self.accumulate_match(segment, bindings, id_success):
        # If the segment matches, we can bind it to the environment and continue
        yield from succeed(dict(m, **self.bind(bindings, segment)), rest)

  def __str__(self):
    if self.pattern != Dot:
      if self.name:
        return f'*[?{self.name}]{self.pattern}'
      return f'*{self.pattern}'
    if self.name:
      return f'*?{self.name}'
    return '*'

  __repr__ = __str__


class Plus(Star):
  """A pattern for repeated sub-patterns inside of a sequence.

  Attributes:
    pattern: an object that will be matched against elements of a sequence.
    name: an optional `str` name to bind the result of the star match.
    accumulate: a sequence of `str` names corresponding to `Var`s in `pattern`
      that will be accumulated into a sequence instead of having to match across
      the elements of the sequence.
    greedy: a `bool` that sets whether or not the `Plus` greedily matches a
      sequence. A greedy `Plus` will try to match slices starting from the
      largest possible and then trying smaller ones. A non-greedy `Plus` will
      match slices starting from the smallest possible.
      Default: `False`.
  """

  def __init__(self,
               pattern: Any,
               name: Optional[str] = None,
               accumulate: Sequence[str] = (),
               greedy: bool = False):
    super().__init__(
        pattern, name=name, accumulate=accumulate, plus=True, greedy=greedy)


class Segment(Star):
  """Matches any slice of a sequence.

  Attributes:
    name: a `str` name to bind the result of the segment match. If `name` is
      `None`, a match produces no binding.
    accumulate: a sequence of `str` names corresponding to `Var`s in `pattern`
      that will be accumulated into a sequence instead of having to match across
      the elements of the sequence.
    greedy: a `bool` that sets whether or not the `Segment` greedily matches a
      sequence. A greedy `Segment` will try to match slices starting from the
      largest possible and then trying smaller ones. A non-greedy `Segment` will
      match slices starting from the smallest possible.
      Default: `False`.
    plus: a `bool`, if `True` requires matches of length one or more and if
      `False` allows zero-length matches.
      Default: `False`
  """

  def __init__(self,
               name: Optional[str],
               accumulate: Sequence[str] = (),
               greedy: bool = False,
               plus: bool = False):
    super().__init__(
        Dot, name=name, accumulate=accumulate, plus=plus, greedy=greedy)


@matcher.register(Sequence)
def sequence_matcher(pattern: Sequence[Any]):
  """Returns a matcher for a given sequence pattern."""

  def sequence_match(expr: Expr, bindings: Bindings,
                     succeed: Continuation) -> Success:
    """Matches a sequence expression against a sequence pattern.

    Matches each element of the sequence pattern against each element of the
    sequence expression. When there is a `Star` in the sequence pattern, the
    sequence  matcher calls the `Star` pattern's matcher, which calls a special
    success function that takes in the remaining part of the sequence to match.

    Args:
      expr: A sequence to match.
      bindings: A dictionary mapping string names to values representing the
        results of previous matches.
      succeed: A function that when passed in `bindings` returns a generator
        over results.

    Yields:
      The results of the `succeed` continuation function, augmented with
      bindings corresponding to matches made over the course of the sequence
      match.
    """
    if not isinstance(expr, Sequence):
      return
    # Special case Star here
    if pattern and isinstance(pattern[0], Star):
      star_pattern, rest_pattern = pattern[0], pattern[1:]

      # A star continuation takes in an additional `remaining` argument that
      # contains the remaining slice of the sequence to be matched.
      def star_succeed(bindings: Bindings, remaining: Sequence[Any]) -> Success:
        post_star_match = matcher(rest_pattern)
        yield from post_star_match(remaining, bindings, succeed)

      star_match = matcher(star_pattern)
      yield from star_match(expr, bindings, star_succeed)  # pytype: disable=wrong-arg-types
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

  return sequence_match


@matcher.register(str)
def str_matcher(pattern: Any):
  """Overrides default sequence matcher for strings to avoid infinite recursion.

  Strings are a tricky case of sequence because indexing into a string returns
  a length-1 string. This, by default, triggers an infinite recursion in the
  sequence matcher. To avoid this, we special-case 1-length strings to do a
  manual match and use the sequence matcher for other strings.

  Args:
    pattern: A pattern used to match a string.
  Returns:
    A pattern matcher for string expressions.
  """
  def str_match(expr: Expr,
                bindings: Bindings,
                succeed: Continuation) -> Success:
    if not isinstance(expr, str):
      return
    if len(expr) == 1 and isinstance(pattern, str):
      if pattern == expr:
        yield from succeed(bindings)
      return
    yield from sequence_matcher(pattern)(expr, bindings, succeed)
  return str_match


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
