# Copyright 2023 The TensorFlow Probability Authors.
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
"""Routines for making tree-structured BNNs."""

from typing import Iterable, List

from flax import linen as nn
import jax
import jax.numpy as jnp
from tensorflow_probability.spinoffs.autobnn import bnn
from tensorflow_probability.spinoffs.autobnn import kernels
from tensorflow_probability.spinoffs.autobnn import operators
from tensorflow_probability.spinoffs.autobnn import util

Array = jnp.ndarray


NON_PERIODIC_KERNELS = [
    kernels.ExponentiatedQuadraticBNN,
    kernels.LinearBNN,
    kernels.QuadraticBNN,
    # Don't use Matern, Exponential or OneLayer BNN's in the leaves because
    # they all give very similar predictions to ExponentiatedQuadratic.
]

LEAVES = NON_PERIODIC_KERNELS + [kernels.PeriodicBNN]

OPERATORS = [
    operators.Multiply,
    operators.Add,
    operators.WeightedSum,
    operators.ChangePoint,
    operators.LearnableChangePoint
]


def list_of_all(
    time_series_xs: Array,
    depth: int = 2,
    width: int = 50,
    periods: Iterable[float] = (),
    parent_is_multiply: bool = False,
    include_sums: bool = True,
    include_changepoints: bool = True,
    only_safe_products: bool = False
) -> List[bnn.BNN]:
  """Return a list of all BNNs of the given depth."""
  all_bnns = []
  if depth == 0:
    all_bnns.extend(k(width=width, going_to_be_multiplied=parent_is_multiply)
                    for k in NON_PERIODIC_KERNELS)
    for p in periods:
      all_bnns.append(kernels.PeriodicBNN(
          width=width, period=p, going_to_be_multiplied=parent_is_multiply))
    return all_bnns

  multiply_children = list_of_all(
      time_series_xs, depth-1, width, periods, True)
  if parent_is_multiply:
    non_multiply_children = multiply_children
  else:
    non_multiply_children = list_of_all(
        time_series_xs, depth-1, width, periods, False)

  # Abelian operators that aren't Multiply.
  if include_sums:
    for i, c1 in enumerate(non_multiply_children):
      for j in range(i):
        c2 = non_multiply_children[j]
        # Add is also abelian, but WeightedSum is more general.
        all_bnns.append(
            operators.WeightedSum(
                bnns=(c1.clone(_deep_clone=True), c2.clone(_deep_clone=True))
            )
        )

  if parent_is_multiply:
    # Remaining operators don't expose .penultimate() method.
    return all_bnns

  # Multiply
  for i, c1 in enumerate(multiply_children):
    if only_safe_products:
      # The only safe kernels to multiply by are Linear and Quadratic.
      if not isinstance(c1, kernels.PolynomialBNN):
        continue
    for j in range(i+1):
      c2 = multiply_children[j]
      all_bnns.append(operators.Multiply(bnns=(
          c1.clone(_deep_clone=True), c2.clone(_deep_clone=True))))

  # Non-abelian operators
  if include_changepoints:
    for c1 in non_multiply_children:
      for c2 in non_multiply_children:
        # ChangePoint is also non-abelian, but requires that we know
        # what the change point is.
        all_bnns.append(operators.LearnableChangePoint(
            bnns=(c1.clone(_deep_clone=True), c2.clone(_deep_clone=True)),
            time_series_xs=time_series_xs))

  return all_bnns


def weighted_sum_of_all(time_series_xs: Array,
                        time_series_ys: Array,
                        depth: int = 2, width: int = 50,
                        alpha: float = 1.0) -> bnn.BNN:
  """Return a weighted sum of all BNNs of the given depth."""
  periods = util.suggest_periods(time_series_ys)

  all_bnns = list_of_all(time_series_xs, depth, width, periods, False)

  return operators.WeightedSum(bnns=tuple(all_bnns), alpha=alpha)


def random_tree(key: jax.Array, depth: int, width: int, period: float,
                parent_is_multiply: bool = False) -> nn.Module:
  """Return a random complete tree BNN of the given depth.

  Args:
    key: Random number key.
    depth: Return a BNN of this tree depth.  Zero based, so depth=0 returns
      a leaf BNN.
    width: The number of hidden nodes in the leaf layers.
    period: The period of any PeriodicBNN kernels in the tree.
    parent_is_multiply: If true, don't create a weight layer after the hidden
      nodes of any leaf kernels and only use addition as an internal node.

  Returns:
    A BNN of the specified tree depth.
  """
  if depth == 0:
    c = jax.random.choice(key, len(LEAVES))
    return LEAVES[c](
        width=width, going_to_be_multiplied=parent_is_multiply,
        period=period)

  key1, key2, key3 = jax.random.split(key, 3)
  if parent_is_multiply:
    c = 1  # Can't multiply Multiply or ChangePoints
    is_multiply = True
  else:
    c = jax.random.choice(key1, len(OPERATORS))
    is_multiply = (c == 0)

  sub1 = random_tree(key2, depth - 1, width, period, is_multiply)
  sub2 = random_tree(key3, depth - 1, width, period, is_multiply)

  return OPERATORS[c](bnns=(sub1, sub2))
