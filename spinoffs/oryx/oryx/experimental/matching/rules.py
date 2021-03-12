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
"""A simple term-rewriting system.

This module is a Python implementation of the rules combinator library found in
[`rules`](https://github.com/axch/rules) by Alexey Radul.

# Expressions

Term-rewriting involves searching expression trees for patterns and replacing
those patterns with new expressions. In order to represent expression trees,
this module provides a single-dispatch API to enable registering both builtin
Python types like `tuple`s and `dict`s, and registering custom classes as
nodes in an expression tree.

The two methods are:
1. `tree_map(expr, fn)`, which takes in an expression `expr` and a function `fn`
   and returns `expr` but with `fn` mapped over its children.
2. `tree_children(expr)`, which takes in an expression `expr` and returns an
   iterator over its children. It acts similar to Python's `iter`, but we can
   provide a custom iterator over builtin types like `dict`s, as we'd like to
   iterate over the values in the dictionary, not the keys.

Both methods are single-dispatch and can be overridden for custom types.
Alternatively, this module provides the `Expression` class, which has methods
`Expression.tree_map` and `Expression.tree_children` and has been already
registered with the two functions. If a custom type subclasses `Expression` it
will thus be auto-registered with `tree_map` and `tree_children`.

# Rules

A rule is a function that maps an input expression to an output expression.
We can either construct rules using `make_rule` or with "rule combinators".

## `make_rule`

A rule can be constructed from a `pattern` (see `matcher`) and a `handler`
function. The `rule` function checks if the input expression matches the
`pattern`, and if so, it passes the resulting bindings into `handler`, producing
the output expression. If the input expression does not match `pattern`, it is
returned as-is.

```python
rule = make_rule(1, lambda: 2) # Replaces 1 with 2
rule(1) # ==> 2
rule(3) # ==> 3
```

We can use matcher combinators to construct more complex rules. For example,
we can use a `matcher.Var` to add one to any input expression.

```python
add_one = make_rule(Var('x'), lambda x: x + 1)
add_one(1) # ==> 2
add_one(2) # ==> 3
```

Alternatively, we can use `matcher.Choice` to add one only if the number is 1 or
2.

```python
maybe_add_one = make_rule(Choice(1, 2, name='x'), lambda x: x + 1)
maybe_add_one(1) # ==> 2
maybe_add_one(2) # ==> 3
maybe_add_one(3) # ==> 3
```

## Rule combinators

Another way to build more complex rules is to use "rule combinators", or
higher-order rules, i.e. functions that take in a rule and return a new rule.

### `rule_list`

`rule_list` takes in a variable number of rules and tries applying them in
sequence until the expression changes, and then immediately returns the changed
expression. If none of the input rules change the expression, the expression is
returned unchanged. Even if a particular pattern matches, if the handler leaves
the expression unchanged, `rule_list` will continue to the next rule. For
example, the following rule,
`rule_list(make_rule(0, lambda: 0), make_rule(Var('x'), lambda x: 1/x))`, will
still divide by zero when called on `0`.

Here are some more examples of using `rule_list`:
```python
one_to_three = make_rule(1, lambda: 3)
three_to_one = make_rule(3, lambda: 1)
rule = rule_list(one_to_three, three_to_one)

rule(1) # ==> 3
rule(3) # ==> 1
rule(2) # ==> 2
```

### `in_order`

`in_order` takes in a variable number of rules and applies them in sequence
regardless of if the expression changes or not, and returns the final
expression.

```python
one_to_three = make_rule(1, lambda: 3)
three_to_one = make_rule(3, lambda: 1)
rule = in_order(one_to_three, three_to_one)

rule(1) # ==> 1
rule(3) # ==> 1
rule(2) # ==> 2
```

### `iterated`

`iterated` takes in a rule and applies it to an expression over and over again
until the expression does not change. Note that is possible to create an
infinite loop with `iterated` if the rules cause the expression to cycle between
values.

```python
add_one_until_five = iterated(make_rule(
                                Var('x', restrictions=[lambda x: x < 5]),
                                lambda x: x + 1))

add_one_until_five(1) # ==> 5
add_one_until_five(-10) # ==> 5
add_one_until_five(7) # ==> 7

one_to_three = make_rule(1, lambda: 3)
three_to_one = make_rule(3, lambda: 1)
bad_rule = iterated(rule_list(one_to_three, three_to_one))

bad_rule(1) # ==> Infinite loop!
```

### `rewrite_subexpressions`

`rewrite_subexpressions` takes in a rule and just applies it to the children of
the input expression using `tree_map`. Types like `int`s and `str`s have no
children, so `rewrite_subexpressions` will leave them unchanged, but types like
`dict`s and `tuple`s do have children, so `rewrite_subexpressions` will apply
a rule to their contained values. Note that `rewrite_subexpressions` does *not*
recurse into the children of children, and only applies the rule to the direct
children of the expression.

```python
one_to_three = make_rule(1, lambda: 3)
three_to_one = make_rule(3, lambda: 1)
rule = rewrite_subexpressions(rule_list(one_to_three, three_to_one))

rule(1) # ==> 1
rule((1, 3)) # ==> (3, 1)
rule(((1, 3),)) # ==> ((1, 3),)
```

### `bottom_up`

`bottom_up` recursively applies a rule to an expression, rewriting children
first, resulting in a bottom-up rewrite of the expression tree (i.e. one that
begins at the leaves and ends with the root). Unlike `rewrite_subexpressions`,
it does rewrite the provided expression, and does recurse through all children.

```python
one_to_three = make_rule(1, lambda: 3)
three_to_one = make_rule(3, lambda: 1)
rule = bottom_up(rule_list(one_to_three, three_to_one))

rule(1) # ==> 3
rule((1, 3)) # ==> (3, 1)
rule(((1, 3),)) # ==> ((3, 1),)
```

To see how the ordering can affect the rewrite, see the following example
where we add one to the children of a tuple before taking its sum:
```python
Integer = lambda name: Var(name, restrictions=[
    lambda x: isinstance(x, int)])
Tuple = lambda name: Var(name, restrictions=[lambda x: isinstance(x, tuple)])

rule = bottom_up(in_order(
         make_rule(Integer('a'), lambda a: a + 1.),
         make_rule(Tuple('t'), lambda t: sum(t))
       ))
rule((1., 2., 3.)) # ==> 9
```

### `top_down`

`top_down` recursively applies a rule to an expression, rewriting the root
expression first and then recursing down into the children, resulting in a
top-down rewrite of the expression tree (i.e. one that begins at the root and
ends with the leaves). Unlike `rewrite_subexpressions`, it does rewrite the
provided expression, and does recurse through all children.

```python
one_to_three = make_rule(1, lambda: 3)
three_to_one = make_rule(3, lambda: 1)
rule = top_down(rule_list(one_to_three, three_to_one))

rule(1) # ==> 3
rule((1, 3)) # ==> (3, 1)
rule(((1, 3),)) # ==> ((3, 1),)
```

To see how the ordering can affect the rewrite, see the following example
where we add one to the children of a tuple after taking its sum:
```python
Integer = lambda name: Var(name, restrictions=[
    lambda x: isinstance(x, int)])
Tuple = lambda name: Var(name, restrictions=[lambda x: isinstance(x, tuple)])

rule = top_down(in_order(
         make_rule(Integer('a'), lambda a: a + 1.),
         make_rule(Tuple('t'), lambda t: sum(t))
       ))
rule((1., 2., 3.)) # ==> 7
```

### `term_rewriter`

The `term_rewriter` combinator composes `bottom_up`, `iterated` and `rule_list`,
resulting in a rule combinator that takes in a list of rules and recursively
applies it on an input expression over and over again until the expression does
not change anymore.

It is provided as a convenience for term-rewriting applications like algebraic
simplification that involve applying series of general rewrite rules to all
parts of an expression.

```python
Positive = lambda name: Var(name, restrictions=[
    lambda x: isinstance(x, int), lambda x: x > 0])
rule = term_rewriter(make_rule(Positive('a'), lambda a: a - 1))

rule(1) # ==> 0
rule(5) # ==> 0
rule((2, 3)) # ==> (0, 0)
rule(((1, 2), 3)) # ==> ((0, 0), 0)
```
"""

import functools
from typing import Any, Callable, Dict, Iterator, Tuple, TypeVar

from oryx.experimental.matching import matcher


__all__ = [
    'tree_children',
    'tree_map',
    'Expression',
    'make_rule',
    'rule_list',
    'in_order',
    'iterated',
    'rewrite_subexpressions',
    'bottom_up',
    'top_down',
    'term_rewriter',
]

Expr = matcher.Expr
T = TypeVar('T')
Rule = Callable[[Expr], Expr]


@functools.singledispatch
def tree_map(expr: T, fn) -> T:
  """Shallow maps a function over the children of an expression."""
  raise NotImplementedError(f'{type(expr)} not registered as a tree node.')


@functools.singledispatch
def tree_children(expr: Expr) -> Iterator[Expr]:
  """Returns the children of an expression."""
  raise NotImplementedError(f'{type(expr)} not registered as a tree node.')


@tree_map.register(float)
@tree_map.register(int)
@tree_map.register(str)
def primitive_tree_map(expr: Expr, fn) -> Expr:
  del fn
  return expr


@tree_children.register(float)
@tree_children.register(int)
@tree_children.register(str)
def primitive_tree_children(expr: Expr) -> Iterator[Expr]:
  yield from ()


@tree_map.register(tuple)
def tuple_tree_map(expr: Tuple[Expr], fn) -> Tuple[Expr]:
  return tuple(map(fn, expr))


@tree_children.register(tuple)
def tuple_tree_children(expr: Tuple[Expr]) -> Iterator[Expr]:
  return iter(expr)


@tree_map.register(dict)
def dict_tree_map(expr: Dict[Any, Expr], fn) -> Dict[Any, Expr]:
  return {k: fn(v) for k, v in expr.items()}


@tree_children.register(dict)
def dict_tree_children(expr: Dict[Any, Expr]) -> Iterator[Expr]:
  return (v for k, v in sorted(expr.items()))


class Expression(matcher.Pattern):
  """A class that is auto-registered with `tree_map` and `tree_children`.

  Subclassing `Expression` and implementing `Expression.tree_map` and
  `Expression.tree_children` will result in a class whose instances that can
  be recursively rewritten with rule combinators.
  """

  def tree_map(self, fn):
    raise NotImplementedError

  def tree_children(self):
    raise NotImplementedError


@tree_map.register(Expression)
def expression_tree_map(expr: Expression, fn) -> Expression:
  return expr.tree_map(fn)


@tree_children.register(Expression)
def expression_tree_children(expr: Expression) -> Iterator[Expr]:
  return expr.tree_children()


def make_rule(pattern: Any, handler) -> Rule:
  """Constructs a rewrite rule from a pattern and handler function.

  Args:
    pattern: Any object registered as a matcher.
    handler: A function that takes in bindings as keyword arguments and returns
      an expression.

  Returns:
    A `rule` function that rewrites an expression if it matches `pattern` to the
    expression returned by `handler` with the matched bindings. If there is
    no match, it returns the expression unchanged.
  """

  @functools.lru_cache(None)
  def rule(expr: Expr) -> Expr:
    try:
      return handler(**matcher.match(pattern, expr))
    except matcher.MatchError:
      return expr

  return rule


# Rule combinators


def rule_list(*rules: Rule) -> Rule:
  """Returns a rule that tries applying series of rules until one succeeds."""

  def rule(expr: Expr) -> Expr:
    for r in rules:
      rewritten = r(expr)
      if rewritten != expr:
        # The expression changed, so we return here.
        return rewritten
    return expr

  return rule


def in_order(*rules: Rule) -> Rule:
  """Returns a rule that applies a series of rules in order."""

  def rule(expr: Expr) -> Expr:
    for r in rules:
      expr = r(expr)
    return expr

  return rule


def iterated(rule: Rule) -> Rule:
  """Returns a rule that iteratively applies a rule until convergence."""

  def iterated_rule(expr: Expr) -> Expr:
    while True:
      prev, expr = expr, rule(expr)
      if prev == expr:
        return expr
    return expr

  return iterated_rule


def rewrite_subexpressions(rule: Rule) -> Rule:
  """Returns a rule that applies rewrites subexpressions of an expression."""

  def subexpr_rule(expr: Expr) -> Expr:
    return tree_map(expr, rule)

  return subexpr_rule


def bottom_up(rule: Rule) -> Rule:
  """Returns a rule that recursively rewrites expressions bottom up."""

  def subexpr_rule(expr: Expr) -> Expr:
    return rule(rewrite_subexpressions(subexpr_rule)(expr))

  return subexpr_rule


def top_down(rule: Rule) -> Rule:
  """Returns a rule that recursively rewrites expressions top down."""

  def subexpr_rule(expr: Expr) -> Expr:
    return rewrite_subexpressions(subexpr_rule)(rule(expr))

  return subexpr_rule


def term_rewriter(*rules: Rule) -> Rule:
  """Returns a rule that rewrites expressions iteratively from the bottom-up."""
  return iterated(bottom_up(rule_list(*rules)))
