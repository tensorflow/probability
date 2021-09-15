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
"""Contains the `AddN` expression.

The `AddN` expression represents a sum of operands. JAX only has a binary
`add` primitive, meaning a sequence of adds is represented as an expression
tree of `add` primitives. In `autoconj`, we'd like to roll all the `add`s into
a single expression to simplify rewrite rules and to represent a canonicalized
density function. Thus we use `AddN` to represent a flat sum of operands.
"""
import dataclasses
import functools
import operator

from typing import Any, Dict, Iterator, Tuple, Union

import jax
import jax.numpy as jnp

from oryx.experimental.matching import jax_rewrite as jr
from oryx.experimental.matching import matcher

__all__ = [
    'AddN',
]

Bindings = matcher.Bindings
Continuation = matcher.Continuation
Expr = matcher.Expr
Pattern = matcher.Pattern
Success = matcher.Success


@dataclasses.dataclass(frozen=True)
class AddN(jr.JaxExpression):
  """Adds several children expressions.

  JAX's `add` primitive is binary so adding several terms must be represented
  as a tree of `add`s. `AddN` is a "flat" expression representation of adding
  several subexpressions which is more convenient for pattern matching and
  term rewriting.

  Attributes:
    operands: A tuple of expressions to be added together when evaluating
      the `AddN` expression.
  """
  operands: Union[Pattern, Tuple[Any, ...]]

  @functools.lru_cache(None)
  def shape_dtype(self) -> jax.ShapeDtypeStruct:
    """Computes the shape and dtype of the result of this `AddN`.

    Returns:
      A `jax.ShapeDtypeStruct` object describing the shape and dtype of the
      `AddN`.
    """
    operand_shape_dtypes = tuple(
        jax.ShapeDtypeStruct(operand.shape, operand.dtype)
        for operand in self.operands)

    def _eval_fun(*args):
      return functools.reduce(operator.add, args)

    return jax.eval_shape(_eval_fun, *operand_shape_dtypes)

  @property
  def shape(self) -> Tuple[int, ...]:
    return self.shape_dtype().shape

  @property
  def dtype(self) -> jnp.dtype:
    return self.shape_dtype().dtype

  # Matching methods

  def match(self, expr: Expr, bindings: Bindings,
            succeed: Continuation) -> Success:
    """Matches the formula and operands of an `AddN`."""
    if not isinstance(expr, AddN):
      return
    yield from matcher.matcher(self.operands)(expr.operands, bindings, succeed)

  # Rules methods

  def tree_map(self, fn) -> 'AddN':
    """Maps a function across the operands of an `AddN`."""
    return AddN(tuple(map(fn, self.operands)))

  def tree_children(self) -> Iterator[Any]:
    """Returns an iterator over the operands of an `AddN`."""
    yield from self.operands

  # JAX rewriting methods

  def evaluate(self, env: Dict[str, Any]) -> Any:
    """Evaluates an `AddN` in an environment."""
    operands = jr.evaluate(self.operands, env)
    return functools.reduce(operator.add, operands)

  # Builtin methods

  def __str__(self) -> str:
    return f'(addn {self.operands})'
