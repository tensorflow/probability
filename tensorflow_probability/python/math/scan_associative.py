# Copyright 2020 The TensorFlow Probability Authors.
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
"""Utilities for parallel calculation of prefix sums."""

import numpy as np
import tensorflow.compat.v2 as tf

from tensorflow_probability.python.internal import prefer_static


def _interleave(a, b):
  """Interleaves two `Tensor`s along their first axis."""
  # [a b c ...] [d e f ...] -> [a d b e c f ...]
  num_elems_a = prefer_static.shape(a)[0]
  num_elems_b = prefer_static.shape(b)[0]

  def _interleave_with_b(a):
    return tf.reshape(
        tf.stack([a, b], axis=1),
        prefer_static.concat([[2 * num_elems_b],
                              prefer_static.shape(a)[1:]], axis=0))
  return prefer_static.cond(
      prefer_static.equal(num_elems_a, num_elems_b + 1),
      lambda: tf.concat([_interleave_with_b(a[:-1]), a[-1:]], axis=0),
      lambda: _interleave_with_b(a))


def _validate_elem_length(max_num_levels, elems_flat):
  """Checks that elems all have the same length, and returns that length."""
  assertions = []

  elem_length = prefer_static.shape(elems_flat[0])[0]

  # The default size limit will overflow a 32-bit int, so make sure we're
  # using 64-bit.
  size_limit = 2**(prefer_static.cast(max_num_levels, np.int64) + 1)
  enough_levels = prefer_static.less(
      prefer_static.cast(elem_length, np.int64), size_limit)
  enough_levels_ = tf.get_static_value(enough_levels)
  if enough_levels_ is None:
    assertions.append(
        tf.debugging.assert_equal(
            enough_levels, True,
            message='Input `Tensor`s must have first axis dimension less than'
            ' `2**(max_num_levels + 1)`'
            ' (saw: {} which is not less than 2**{} == {})'.format(
                elem_length,
                max_num_levels,
                size_limit)))
  elif not enough_levels_:
    raise ValueError(
        'Input `Tensor`s must have first axis dimension less than'
        ' `2**(max_num_levels + 1)`'
        ' (saw: {} which is not less than 2**{} == {})'.format(
            elem_length,
            max_num_levels,
            size_limit))

  is_consistent = prefer_static.reduce_all([
      prefer_static.equal(
          prefer_static.shape(elem)[0], elem_length)
      for elem in elems_flat[1:]])

  is_consistent_ = tf.get_static_value(is_consistent)
  if is_consistent_ is None:
    assertions.append(
        tf.debugging.assert_equal(
            is_consistent, True,
            message='Input `Tensor`s must have the same first dimension.'
            ' (saw: {})'.format([elem.shape for elem in elems_flat])))
  elif not is_consistent_:
    raise ValueError(
        'Input `Tensor`s must have the same first dimension.'
        ' (saw: {})'.format([elem.shape for elem in elems_flat]))
  return elem_length, assertions


def scan_associative(fn, elems, max_num_levels=48,
                     validate_args=False, name=None):
  """Perform a scan with an associative binary operation, in parallel.

  The associative scan operation computes the cumulative sum, or
  [all-prefix sum](https://en.wikipedia.org/wiki/Prefix_sum), of a set of
  elements under an associative binary operation [1]. For example, using the
  ordinary addition operator `fn = lambda a, b: a + b`, this is equivalent to
  the ordinary cumulative sum `tf.math.cumsum` along axis 0. This method
  supports the general case of arbitrary associative binary operations operating
  on `Tensor`s or structures of `Tensor`s:

  ```python
  scan_associative(fn, elems) = tf.stack([
    elems[0],
    fn(elems[0], elems[1]),
    fn(elems[0], fn(elems[1], elems[2])),
    ...
    fn(elems[0], fn(elems[1], fn(..., fn(elems[-2], elems[-1]))),
  ], axis=0)
  ```

  The associative structure allows the computation to be decomposed
  and executed by parallel reduction. Where a naive sequential
  implementation would loop over all `N` elements, this method requires
  only a logarithmic number (`2 * ceil(log_2 N)`) of sequential steps, and
  can thus yield substantial performance speedups from hardware-accelerated
  vectorization. The total number of invocations of the binary operation
  (including those performed in parallel) is
  `2 * (N / 2 + N / 4 + ... + 1) = 2N - 2`
  --- i.e., approximately twice as many as a naive approach.

  [1] Blelloch, Guy E.
      [Prefix sums and their applications](
      https://www.cs.cmu.edu/~guyb/papers/Ble93.pdf)
      Technical Report CMU-CS-90-190,
      School of Computer Science,
      Carnegie Mellon University, 1990.

  Args:
    fn: Python callable implementing an associative binary operation with
      signature `r = fn(a, b)`. This must satisfy associativity:
      `fn(a, fn(b, c)) == fn(fn(a, b), c)`. The inputs and result are
      (possibly nested structures of) `Tensor`(s), matching `elems`. Each
      `Tensor` has a leading batch dimension in place of `elem_length`; the `fn`
      is expected to map over this dimension. The result `r` has the same shape
      (and structure) as the two inputs `a` and `b`.
    elems: A (possibly nested structure of) `Tensor`(s), each with leading
      dimension `elem_length`. Note that `elem_length` determines the number
      of recursive steps required to perform the scan: if, in graph mode,
      this is not statically available, then ops will be created to
      handle any `elem_length` up to the maximum dimension of a `Tensor`.
    max_num_levels: Python `int`. The size
      of the first dimension of the tensors in `elems` must be less than
      `2**(max_num_levels + 1)`. The default value is sufficiently large
      for most needs. Lowering this value can reduce graph-building time when
      `scan_associative` is used with inputs of unknown shape.
      Default value: `48`.
    validate_args: Python `bool`. When `True`, runtime checks
      for invalid inputs are performed. This may carry a performance cost.
      Default value: `False`.
    name: Python `str` name prefixed to ops created by this function.
  Returns:
    result: A (possibly nested structure of) `Tensor`(s) of the same shape
      and structure as `elems`, in which the `k`th element is the result of
      recursively applying `fn` to combine the first `k` elements of
      `elems`. For example, given `elems = [a, b, c, ...]`, the result
      would be `[a, fn(a, b), fn(fn(a, b), c), ...]`.

  #### Examples

  ```python
  import tensorflow as tf
  import tensorflow_probability as tfp
  import operator

  # Example 1: Partials sums of numbers.

  tfp.math.scan_associative(operator.add, tf.range(0, 4))
  # ==> [ 0, 1, 3, 6]

  # Example 2: Partial products of random matrices.

  dist = tfp.distributions.Normal(loc=0., scale=1.)
  matrices = dist.sample(sample_shape=[100, 2, 2])
  tfp.math.scan_associative(tf.matmul, matrices)
  ```
  """

  def lowered_fn(a, b):
    # Lower `fn` to operate on flattened sequences of elems.
    with tf.name_scope('fn'):
      return tf.nest.flatten(
          fn(tf.nest.pack_sequence_as(elems, a),
             tf.nest.pack_sequence_as(elems, b)))

  elems_flat = [tf.convert_to_tensor(elem) for elem in tf.nest.flatten(elems)]

  # Summary of algorithm:
  #
  # Consider elements of `_scan(elems)` at odd indices. That's the same as first
  # summing successive pairs of elements of `elems` and performing a scan on
  # that half sized tensor. We perform the latter scan by recursion.
  #
  # Now consider the even elements of `_scan(elems)`. These can be computed
  # from the odd elements of `_scan(elems)` by adding each odd element of
  # `_scan(elems)` to the matching even element in the original `elems`.
  #
  # We return the odd and even elements interleaved.
  #
  # For the base case of the recursion we return the first element
  # of `elems` followed by the sum of the first two elements computed as
  # a (small two-down-to-one) reduction step.

  # The following is a pictorial representation of the algorithm using the
  # variables in the code below. The operator '+' is used to represent
  # the binary operation.
  # Note how the recursive call to `_scan` operates on a reduced form of the
  # input array in which successive pairs have already been summed.

  # elems     x0         x1   x2         x3    x4         x5    ...
  #           |\         /    | \        /     | \        /
  #           | \       /     |  \      /      |  \      /
  #           |  \     /      |   \    /       |   \    /
  #           |   \   /       |    \  /        |    \  /
  # reduced   |  x0+x1        |   x2+x3        |    x4+x5       ...
  # _elems    |    |          |     |          |       |
  #           |    |          |     |          |       |
  #           |    |          |     |          |       |
  # _scan(..) |    |          |     |          |       |
  #        +--|----+----------|-----+----------|-------+----    ...
  #        |  |               |                |
  #        |  |               |                |
  #        +--|----+----------|-----+----------|-------+----    ...
  #           |    |          |     |          |       |
  # odd       |  x0+x1        |   x0+...+x3    |     x0+..+x5   ...
  # _elems    |    | \        |     |      \   |       |
  #           |    |  \       |     |       \  |       |
  # even      |    |   \      |     |        \ |       |
  # _elems    x0   |   x0+...+x2    |       x0+...+x4  |        ...
  #           |    |          |     |          |       |
  # inter     |    |          |     |          |       |
  # leave(..) |    |          |     |          |       |
  #           x0 x0+x1 x0+...+x2  x0+...+x3 x0+...+x4 x0+...+x5 ...

  # TODO(b/150374456): if the sizes of all of the tensors can be determined
  # statically then we don't need a `level` parameter.
  def _scan(level, elems):
    """Perform scan on `elems`."""
    elem_length = prefer_static.shape(elems[0])[0]

    # Apply `fn` to reduce adjacent pairs to a single entry.
    a = [elem[0:-1:2] for elem in elems]
    b = [elem[1::2] for elem in elems]
    reduced_elems = lowered_fn(a, b)

    def handle_base_case_elem_length_two():
      return [tf.concat([elem[0:1], reduced_elem], axis=0)
              for (reduced_elem, elem) in zip(reduced_elems, elems)]

    def handle_base_case_elem_length_three():
      reduced_reduced_elems = lowered_fn(
          reduced_elems, [elem[2:3] for elem in elems])
      return [
          tf.concat([elem[0:1], reduced_elem, reduced_reduced_elem], axis=0)
          for (reduced_reduced_elem, reduced_elem, elem)
          in zip(reduced_reduced_elems, reduced_elems, elems)]

    # Base case of recursion: assumes `elem_length` is 2 or 3.
    at_base_case = prefer_static.logical_or(
        prefer_static.equal(elem_length, 2),
        prefer_static.equal(elem_length, 3))
    base_value = lambda: prefer_static.cond(  # pylint: disable=g-long-lambda
        prefer_static.equal(elem_length, 2),
        handle_base_case_elem_length_two,
        handle_base_case_elem_length_three)

    if level <= 0:
      return base_value()

    def recursive_case():
      """Evaluate the next step of the recursion."""
      odd_elems = _scan(level - 1, reduced_elems)

      def even_length_case():
        return lowered_fn([odd_elem[:-1] for odd_elem in odd_elems],
                          [elem[2::2] for elem in elems])

      def odd_length_case():
        return lowered_fn([odd_elem for odd_elem in odd_elems],
                          [elem[2::2] for elem in elems])

      results = prefer_static.cond(
          prefer_static.equal(elem_length % 2, 0),
          even_length_case,
          odd_length_case)

      # The first element of a scan is the same as the first element
      # of the original `elems`.
      even_elems = [tf.concat([elem[0:1], result], axis=0)
                    for (elem, result) in zip(elems, results)]
      return list(map(_interleave, even_elems, odd_elems))

    return prefer_static.cond(at_base_case, base_value, recursive_case)

  with tf.name_scope(name if name else 'scan_associative'):
    elem_length, assertions = _validate_elem_length(max_num_levels, elems_flat)

  with tf.control_dependencies(assertions if validate_args else []):
    return prefer_static.cond(
        elem_length < 2,
        lambda: elems,
        lambda: (tf.nest.pack_sequence_as(  # pylint: disable=g-long-lambda
            elems, _scan(max_num_levels - 1, elems_flat))))
