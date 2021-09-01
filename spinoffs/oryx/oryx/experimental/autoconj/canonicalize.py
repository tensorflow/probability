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
"""Contains rules for rewriting expressions into a canonical form.

In Autoconj, we aim to rewrite expressions into a form that makes it easy to
marginalize out latent variables and compute complete conditional forms, which
is called "canonical form". Specifically, the log joint probability
conjugate-exponential model can be expressed as a sum of linear functions of the
sufficient statistics of the random variables (note that we can express any
linear function as an `Einsum`). The goal of the `canonicalize` module is to
provide rewrite rules that take JAX expressions and rewrite them as a sum of
`Einsum`s.

Because JAX does not have an einsum primitive, some rules are dedicated to
converting existing linear function primitives in JAX (like dot products,
transposes, etc.) into into `Einsum`s.
"""
import itertools as it

from typing import Any, Callable

from jax import lax

from oryx.experimental.autoconj import einsum
from oryx.experimental.matching import jax_rewrite as jr
from oryx.experimental.matching import matcher
from oryx.experimental.matching import rules

__all__ = [
    'CANONICALIZATION_RULES',
    'canonicalize',
]


Einsum = einsum.Einsum
Var = matcher.Var
Segment = matcher.Segment
Primitive = jr.Primitive
JaxExpression = jr.JaxExpression
Params = jr.Params


def register_rule(pattern: matcher.Pattern) -> Callable[..., rules.Rule]:
  """A decorator that associates a rule function with a pattern."""

  def register(handler: Callable[..., Any]) -> rules.Rule:
    return rules.make_rule(pattern, handler)

  return register


_transpose_pattern = Primitive(lax.transpose_p, (Var('x'),), Var('params'))


@register_rule(_transpose_pattern)
def transpose_as_einsum(x: JaxExpression, params: Params) -> Einsum:
  """Converts a transpose into an `Einsum`."""
  x_ndim = len(x.shape)
  x_dims = ''.join(it.islice(einsum.einsum_letters(), x_ndim))
  out_dims = ''.join([x_dims[dim] for dim in params['permutation']])
  return Einsum(f'{x_dims}->{out_dims}', (x,))


_squeeze_pattern = Primitive(lax.squeeze_p, (Var('x'),), Var('params'))


@register_rule(_squeeze_pattern)
def squeeze_as_einsum(x: JaxExpression, params: Params) -> Einsum:
  """Converts a squeeze into an `Einsum`."""
  dimensions = params['dimensions']
  x_ndim = len(x.shape)
  x_dims = ''.join(it.islice(einsum.einsum_letters(), x_ndim))
  out_dims = ''.join([x_dims[i] for i in range(x_ndim) if i not in dimensions])
  return Einsum(f'{x_dims}->{out_dims}', (x,))


_dot_pattern = Primitive(lax.dot_general_p, (Var('x'), Var('y')), Var('params'))


@register_rule(_dot_pattern)
def dot_as_einsum(x: JaxExpression, y: JaxExpression, params: Params) -> Einsum:
  """Converts a dot product into an `Einsum`."""
  dimension_numbers = params['dimension_numbers']
  (x_contract, y_contract), (x_batch, y_batch) = dimension_numbers
  x_ndim, y_ndim = len(x.shape), len(y.shape)
  letter_iter = einsum.einsum_letters()
  x_dims = ''.join(it.islice(letter_iter, x_ndim))
  y_dims = list(it.islice(letter_iter, y_ndim))
  for x_dim, y_dim in zip(x_contract + x_batch, y_contract + y_batch):
    y_dims[y_dim] = x_dims[x_dim]
  y_dims = ''.join(y_dims)
  out_batch_dims = [x_dims[dim] for dim in x_batch]
  out_dims = out_batch_dims + ([xd for xd in x_dims if xd not in y_dims] +
                               [yd for yd in y_dims if yd not in x_dims])
  out_dims = ''.join(out_dims)
  return Einsum(f'{x_dims},{y_dims}->{out_dims}', (x, y))


_reduce_sum_pattern = Primitive(lax.reduce_sum_p, (Var('x'),), Var('params'))


@register_rule(_reduce_sum_pattern)
def reduce_sum_as_einsum(x: JaxExpression, params: Params) -> Einsum:
  """Converts a reduce sum into an `Einsum`."""
  axis = params['axes']
  x_shape = x.shape
  x_dims = ''.join(it.islice(einsum.einsum_letters(), len(x_shape)))
  out_dims = ''.join([x_dims[i] for i in range(len(x_shape)) if i not in axis])
  formula = f'{x_dims}->{out_dims}'
  return Einsum(formula, (x,))


_einsum_of_einsum_pattern = Einsum(
    Var('parent_formula'),
    (Segment('left_args'), Einsum(
        Var('child_formula'), (Segment('child_args'),)), Segment('right_args')))


@register_rule(_einsum_of_einsum_pattern)
def compose_einsums(parent_formula, left_args, child_formula, child_args,
                    right_args):
  child_einsum = Einsum(child_formula, child_args)
  return einsum.compose_einsums(parent_formula, left_args, child_einsum,
                                right_args)


CANONICALIZATION_RULES = (
    transpose_as_einsum,
    dot_as_einsum,
    squeeze_as_einsum,
    reduce_sum_as_einsum,
    compose_einsums,
)

canonicalize = rules.term_rewriter(*CANONICALIZATION_RULES)
