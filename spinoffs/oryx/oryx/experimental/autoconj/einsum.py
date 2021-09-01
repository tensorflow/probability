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
"""Contains the `Einsum` expression and utilities.

The `Einsum` pattern is used for pattern matching and term rewriting in JAX.
JAX does not have an underlying einsum primitive; a call to `jnp.einsum` turns
into its component `dot_general`, `broadcast`, and `transpose` primitives and
therefore einsums do not directly appear in JAXprs. Autoconj is based on
rewriting expressions and combining expressions into large einsums and because
JAX does not have a primitive representation, we need to create our own.

Along with the `Einsum` pattern we include utilities for manipulating
einsums. For example, the `compose_einsums` function contains the logic for
taking two nested einsums and combining them into a single one. These utilities
are needed for systems such as
[Autoconj](https://papers.nips.cc/paper/2018/hash/9b89bedda1fc8a2d88c448e361194f02-Abstract.html),
which aim to create large, monolothic einsums in a program. The functions are
based on their [implementations in
Autoconj](https://github.com/google-research/autoconj/blob/master/autoconj/rewrites.py).
"""
import collections
import dataclasses
import functools

from typing import Any, Dict, Iterator, List, Sequence, Tuple, Union

import jax
import jax.numpy as jnp

from oryx.experimental.matching import jax_rewrite as jr
from oryx.experimental.matching import matcher

__all__ = [
    'compose_einsums',
    'Einsum',
    'einsum_letters',
]

Bindings = matcher.Bindings
Continuation = matcher.Continuation
Expr = matcher.Expr
Pattern = matcher.Pattern
Success = matcher.Success

_EINSUM_RANGE = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'


def einsum_letters() -> Iterator[str]:
  """Returns an iterator over valid einsum index names."""
  yield from _EINSUM_RANGE


@dataclasses.dataclass(frozen=True)
class Einsum(jr.JaxExpression):
  """An expression that executes a JAX einsum on its operands.

  JAX offers a `jax.numpy.einsum` function but is executed as a series of JAX
  primitive operations including `dot_general` and `broadcast`. This means that
  when a function with an `Einsum` is traced, the `Einsum` does not explicitly
  show up in the resulting JAXpr. For the purposes of term rewriting, we
  therefore need our own `Einsum` representation that can be constructed from
  JAX primitives using rewrite rules.

  Attributes:
    formula: A string describing the `Einsum`'s operation. See the [NumPy
      documentation](
      https://numpy.org/doc/stable/reference/generated/numpy.einsum.html) for
        more information.
    operands: The inputs to the Einsum.
  """
  formula: Union[Pattern, str]
  operands: Union[Pattern, Tuple[Any, ...]]

  @functools.lru_cache(None)
  def shape_dtype(self) -> jax.ShapeDtypeStruct:
    """Computes the shape and dtype of the result of this `Einsum`.

    This function traces the JAX execution and does not incur any FLOPs. To
    avoid retracing, however, we are safe to cache the result of this function
    because `Einsum`s are immutable.

    Returns:
      A `jax.ShapeDtypeStruct` object describing the shape and dtype of the
      `Einsum`.
    """
    # We can trace the evaluation without incurring any FLOPs.
    operand_shape_dtypes = tuple(
        jax.ShapeDtypeStruct(operand.shape, operand.dtype)
        for operand in self.operands)

    def _eval_fun(*args):
      return jnp.einsum(self.formula, *args)

    return jax.eval_shape(_eval_fun, *operand_shape_dtypes)

  @property
  def shape(self) -> Tuple[int]:
    return self.shape_dtype().shape

  @property
  def dtype(self) -> jnp.dtype:
    return self.shape_dtype().dtype

  # Matching methods

  def match(self, expr: Expr, bindings: Bindings,
            succeed: Continuation) -> Success:
    """Matches the formula and operands of an `Einsum`."""
    if not isinstance(expr, Einsum):
      return
    yield from matcher.matcher((self.operands, self.formula))(
        (expr.operands, expr.formula), bindings, succeed)

  # Rules methods

  def tree_map(self, fn) -> 'Einsum':
    """Maps a function across the formula and operands of an `Einsum`."""
    return Einsum(self.formula, tuple(map(fn, self.operands)))

  def tree_children(self) -> Iterator[Any]:
    """Returns an iterator over the operands of an `Einsum`."""
    yield from self.operands

  # JAX rewriting methods

  def evaluate(self, env: Dict[str, Any]) -> Any:
    """Evaluates an `Einsum` in an environment."""
    operands = jr.evaluate(self.operands, env)
    return jnp.einsum(self.formula, *operands)

  # Builtin methods

  def __str__(self) -> str:
    return f'(einsum[{self.formula}] {" ".join(map(str, self.operands))})'


def split_einsum_formula(formula: str) -> Tuple[List[str], str]:
  """Splits an einsum formula string into its component axis names."""
  input_formula, output_formula = formula.split('->')
  return input_formula.split(','), output_formula


def reconstitute_einsum_formula(input_formulas: Sequence[str],
                                output_formula: str) -> str:
  """Joins einsum input formulas and output formula into a complete formula."""
  joined_input_formula = ','.join(input_formulas)
  return f'{joined_input_formula}->{output_formula}'


def compose_einsums(parent_formula: str, left_args: Tuple[Any, ...],
                    child_einsum: Einsum, right_args: Tuple[Any,
                                                            ...]) -> Einsum:
  """Combines nested einsums into a single einsum.

  Einsums are linear functions and thus the composition of two (or more) einsums
  can be represented as a single one. Composed einsums often come up during the
  term rewriting phase of Autoconj, where a series of linear operations (a
  matrix multiplication followed by a transpose, for example) need to be
  folded together into a single einsum in order to represent a function as a
  sum of einsums. This function takes a composition of einsums (an einsum with
  an einsum as one its arguments) and returns a flattened, single einsum.

  As an example use-case, suppose we have matrices `w, x, y` and `z` along with
  the following `Einsum`s:
  ```python
  child_op = Einsum('ab,bc->ac', (x, y))
  parent_op = Einsum('ab,bc,cd->ad', (w, child_op, z))
  ```

  These two operations can be combined to form the single `Einsum`
  ```python
  combined_op = Einsum('ab,bc,cd,de->ae', (w, x, y, z))
  ```

  Implementation based on `_compose_einsums` in
  [Autoconj](https://github.com/google-research/autoconj/blob/master/autoconj/rewrites.py).

  Args:
    parent_formula: The formula of the parent einsum.
    left_args: The sequence of arguments to the left of the child einsum.
    child_einsum: An `Einsum` that is an argument in the `parent_formula`.
    right_args: The sequence of arguments to the right of the child einsum.

  Returns:
    A single un-nested `Einsum` that computes the same quantity as the nested
    einsums.
  """
  parent_in_formulas, parent_out_formula = split_einsum_formula(parent_formula)
  child_formula, child_args = child_einsum.formula, child_einsum.operands
  child_in_formulas, child_out_formula = split_einsum_formula(child_formula)
  num_left = len(left_args)
  # Number of output dimensions of child einsum should match number of
  # dimensions in parent einsum.
  if len(child_out_formula) != len(parent_in_formulas[num_left]):
    raise ValueError(f'Child output formula {child_out_formula} and '
                     f'parent formula {parent_in_formulas[num_left]} have'
                     ' inconsistent size.')
  str_iterator = einsum_letters()
  # Creates a dictionary where each time we access a new element, we generate
  # a new letter.
  subs_map = collections.defaultdict(lambda: next(str_iterator))
  # Splices out the old input formula
  old_in_formula = parent_in_formulas[num_left]
  parent_in_formulas = (
      parent_in_formulas[:num_left] + parent_in_formulas[num_left + 1:])
  # Canonicalizes input and output formulas (optional, for cleanliness)
  parent_in_formulas = [
      ''.join(subs_map[idx] for idx in subs) for subs in parent_in_formulas
  ]
  out_formula = ''.join(subs_map[idx] for idx in parent_out_formula)
  # Maps child output indices with corresponding parent indices
  subs_map.update((pidx + '_child', subs_map[idx])
                  for pidx, idx in zip(child_out_formula, old_in_formula))
  # Updates the child input formulas to use parent mappings
  child_in_formulas = [
      ''.join(subs_map[idx + '_child']
              for idx in subs)
      for subs in child_in_formulas
  ]
  # Concatenates the formulas and arguments
  new_in_formulas = (
      parent_in_formulas[:num_left] + child_in_formulas +
      parent_in_formulas[num_left:])
  new_args = left_args + child_args + right_args
  new_formula = reconstitute_einsum_formula(new_in_formulas, out_formula)
  return Einsum(new_formula, tuple(new_args))
