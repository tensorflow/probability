# Copyright 2019 The TensorFlow Probability Authors.
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
"""Compute einsums in log space."""

import opt_einsum as oe
import tensorflow.compat.v1 as tf


# pylint: disable=no-member


def _execute_contract_path(contract_path, tensors):
  """Carry out sequence of contractions in order specified by opt_einsum."""

  for positions, _, formula, _, _ in contract_path:

    args = [tensors.pop(x) for x in positions]

    if len(args) == 1:
      new_tensor = _unary_logeinsumexp(formula, *args)
    elif len(args) == 2:
      new_tensor = _binary_einslogsumexp(formula, *args)
    else:
      raise ValueError("`_execute_contract_path` can't execute n-ary"
                       "contraction for n > 1")

    tensors.append(new_tensor)

  return tensors[0]


def logeinsumexp(formula, *tensors):
  """Computing `einsum` in logarithmic space.

  Args:
    formula: a formula compatible with `einsum`. (But note that
      ellipses are not supported.)
    *tensors: tensors to which contractions will be applied.

  Returns:
    `logeinsumexp(formula, *tensors)` is equivalent to
    `tf.math.log(einsum(formula, *map(tf.math.exp, tensors)))` except
    that it is more numerically stable.

  Notes:
    This implementation of `logeinsumexp` is primarily intended for internal
    use in the `marginalize` library and assumes the formula is well-formed.
  """

  _, path = oe.contract_path(formula, *tensors)
  return _execute_contract_path(path.contraction_list, list(tensors))


# Rewrite using `tf.transpose`
def rearrange(src, dst, t):
  """Reorder dimensions of tensor according to formula."""

  new_indices = ''
  for i in dst:
    if i not in src:
      new_indices += i
  new_src = src + new_indices
  new_t = tf.reshape(t, tf.concat(
      [tf.shape(t), tf.ones(len(new_indices), dtype=tf.int32)], axis=0))
  formula = '{}->{}'.format(new_src, dst)
  # It is safe to use ordinary `einsum` here as no summations
  # are performed.
  return tf.einsum(formula, new_t)


def _extract_diagonals(formula_a, a):
  """Perform diagonal extraction in einsum formula.

  Args:
    formula_a: a string of indices compatible with the given tensor
    a: a tensor compatible with the given indices. For example
      consider the case
        `formula_a == ijik` and `a.shape == [2, 3, 2, 4]`.
      These are compatible because the tensor has rank 4 and
      the string has 4 index letters. Additionally the first and
      third index letters are the same corresponding to the first
      and third dimensions of the tensor shape being equal.

  Returns:
    If there are any repeated indices in the index string then
    these correspond to entries in `a` along a generalized
    diagonal. For example the 'ii' entries in a 3x3 matrix `m`
    correspond to the 3 diagonal entries of `m`.
    `_extract_diagonals('ii', m)` returns a pair consisting of
    a 3-vector representing the diagonal of `m` as well as the
    index string `i` indicating that the extraction was
    equivalent to computing `einsum('ii->i', m)`.
    If we have an einsum formula like 'ijik,j,k->jk'
    then we can reduce it to an einsum of the form
    'ijk,j,k->jk', removing the repeated indices,
    by applying `_extract_diagonals` to the corresponding tensor.
    `_extract_diagonals` also returns the string 'ijk'.
  """

  extracted_a = ''
  for i in formula_a:
    if i not in extracted_a:
      extracted_a += i
  if formula_a != extracted_a:
    # It is safe to use ordinary `einsum` here as no summations
    # are performed.
    a = tf.einsum('{}->{}'.format(formula_a, extracted_a), a)
    formula_a = extracted_a

  return formula_a, a


def _internal_contract(formula_a, a, other):
  """Perform contractions contained completely within a single tensor.

  Args:
    formula_a: a string of indices compatible with the given tensor.
    a: a tensor compatible with the given indices.
    other: a string of indices.

  Returns:
    If an einsum formula contains indices that correspond solely to
    dimensions of one tensor and appear nowhere among the indices
    for the other input arguments or the output then these describe
    a summation or contraction within that tensor.
    For example an einsum formula like 'ij,kj,lk->...'
    we can completely eliminate all occurrences of 'i' from the
    formula by applying `_internal_contract` to the appropriate tensor.
    `_internal_contract` also updates the appropriate part of the formula
    reflect the eliminations.
    For example if `m` is a 10x3x3 tensor considered as an array
    of 10 3-vectors, `_internal_contract('ji', m)` will return
    a vector consisting of the sums of the 10 vectors along with the
    string 'j' indicating the index 'j' was not reduced or contracted over.
    Note that this function sums in log space so it in fact uses
    `reduce_logsumexp` to compute the sum.
    Note also that this function is written on the assumption that repeated
    indices have already been removed by `_extract_diagonals`.
  """

  reducing_a = ''
  keep_a = ''
  for i in formula_a:
    if i in other:
      # Keep an index if it's used elsewhere
      keep_a += i
    else:
      # Collect indices that appear only for this tensor
      reducing_a += i
  if formula_a != keep_a:
    axes_a = map(formula_a.index, reducing_a)
    # This reduction is only valid if there are no repeated indices
    # in `formula_a`.
    a = tf.reduce_logsumexp(a, axis=tuple(axes_a))
    formula_a = keep_a

  return formula_a, a


def _unary_logeinsumexp(formula, a):
  """Compute `einsum` in log space for the case of one tensor argument."""
  lhs_a, rhs = formula.split('->')

  lhs_a, a = _extract_diagonals(lhs_a, a)
  lhs_a, a = _internal_contract(lhs_a, a, rhs)
  a = rearrange(lhs_a, rhs, a)

  return a


def _binary_einslogsumexp(formula, a, b):
  """Compute `einsum` in log space for the case of two tensor arguments."""
  lhs, rhs = formula.split('->')
  lhs_a, lhs_b = lhs.split(',')

  # First extract any 'diagonals'. These are repeated indices within `lhs_a`
  # or `lhs_b`.
  lhs_a, a = _extract_diagonals(lhs_a, a)
  lhs_b, b = _extract_diagonals(lhs_b, b)

  # At this point there should be no repeated indices in `lhs_a` or
  # `lhs_b`.

  # Now do any simple reductions. These correspond to indices that appear
  # in just one of `lhs_a` or `lhs_b` and not in the final result.
  lhs_a, a = _internal_contract(lhs_a, a, lhs_b + rhs)
  lhs_b, b = _internal_contract(lhs_b, b, lhs_a + rhs)

  intermediates = lhs_a
  for i in lhs_b:
    if i not in lhs_a:
      intermediates += i
  reductions = ''
  for i in lhs_a:
    if i not in rhs:
      reductions += i
  for i in lhs_b:
    if i not in rhs:
      reductions += i
  reduced = ''
  for i in intermediates:
    if i in rhs:
      reduced += i

  a = rearrange(lhs_a, intermediates, a)
  b = rearrange(lhs_b, intermediates, b)

  # Note that the intermediate `a + b` will likely be materialized
  # at this point. This makes the implementation of `logeinsumexp` less
  # efficient in its use of memory than it could be.
  # It is possible that future versions of TensorFlow will fuse this
  # operation so that the intermediate isn't materialised.
  # Note however that we are only materializing the sum for binary
  # `einsum`s and that we extract diagonals and perform internal
  # contractions to minimize the size of the materialized array.
  #
  # A proposed way to avoid the intermediate is presented
  # here:
  # https://stackoverflow.com/questions/23630277/numerically-stable-way-to-multiply-log-probability-matrices-in-numpy
  # That method can result in -inf entries even though all of the
  # numbers are well within a domain in which `logeinsumexp` shouldn't
  # underflow.
  almost = tf.reduce_logsumexp(a + b,
                               axis=tuple(map(intermediates.index, reductions)))

  return rearrange(reduced, rhs, almost)
