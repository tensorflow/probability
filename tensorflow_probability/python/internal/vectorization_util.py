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
"""Utilities for vectorizing code."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

import tensorflow.compat.v2 as tf
from tensorflow_probability.python.internal import prefer_static
from tensorflow.python.util import nest  # pylint: disable=g-direct-tensorflow-import


# TODO(b/145252136): merge `make_rank_polymorphic` into core TensorFlow.
def make_rank_polymorphic(fn, core_ndims, validate_args=False, name=None):
  """Lift a function to one that vectorizes across arbitrary-rank inputs.

  Args:
    fn: Python `callable` `result = fn(*args)` where all arguments
      and the returned result(s) are (structures of) Tensors.
    core_ndims: structure matching `args` of `int` Tensors, containing the
      expected rank of each argument in an unvectorized call to `fn`. May
      alternately be a single scalar Tensor `int` applicable to all `args`.
    validate_args: whether to add runtime checks.
      Default value: `False`.
    name: Python `str` name prefixed to ops created by `vectorized_fn`.
  Returns:
    vectorized_fn: a new function, equivalent to `fn`, but which automatically
      accepts arguments of any (combination of) ranks above the `core_ndims`,
      and returns a value with the broadcast batch shape of its arguments.

  #### Example

  ```python
  def add(a, b):
    return a + b
  add(tf.constant([1., 2.]), tf.constant(3.))
    # ==> Returns [4., 5.]

  # Naively passing a batch of three values for `b` raises an error.
  add(tf.constant([1., 2.]), tf.constant([3., 4., 5.]))
    # ==> Raises InvalidArgumentError: Incompatible shapes.

  # By annotating that `b` is scalar, we clarify how to batch the results.
  add_vector_to_scalar = make_rank_polymorphic(add, core_ndims=(1, 0))
  add_vector_to_scalar(tf.constant([1., 2.]), tf.constant([3., 4., 5.]))
    # ==> Returns [[4., 5.], [5., 6.], [6., 7.]]
  ```

  #### Limitations

  The current form of this function has several limitations due to
  `vectorize_map`'s requirement that all inputs and outputs share a common batch
  dimension. When an automatically vectorized function is called, its inputs are
  padded up to the common batch shape, and all outputs are returned with the
  full broadcast batch shape. This can waste memory when a function has multiple
  outputs or multiple, differently-sized inputs. For example, if we define a
  simple function of two arguments:

  ```python
  def silly_increment(a, b):
    return a + 1., b + 1.
  vectorized_increment = make_rank_polymorphic(silly_increment, core_ndims=0)
  ```

  and call it with one very large argument (in this case, a billion entries):

  ```python
  a1, b1 = vectorized_increment(0., tf.ones([1000, 1000, 1000]))
  print(a1.shape, b1.shape)
    # ==> [1000, 1000, 1000], [1000, 1000, 1000]
  ```

  then both of the returned results will have a billion entries, even though
  the first result `a1` could have been computed as a scalar. In addition,
  another unnecessary billion-entry tensor will be created internally
  representing the vectorized input `a`. In this case, the vectorization was
  inefficient because not all outputs depended on all inputs; it would have
  been more efficient to exploit this by incrementing `a` and `b` separately,
  in different calls.

  """

  def vectorized_fn(*args):
    """Vectorized version of `fn` that accepts arguments of any rank."""
    with tf.name_scope(name or 'make_rank_polymorphic'):
      assertions = []

      # If we got a single value for core_ndims, tile it across all args.
      core_ndims_structure = (
          core_ndims
          if nest.is_nested(core_ndims)
          else nest.map_structure(lambda _: core_ndims, args))

      # Build flat lists of all argument parts and their corresponding core
      # ndims.
      flat_core_ndims = nest.flatten(core_ndims_structure)
      parts = tf.nest.flatten(nest.map_structure_up_to(
          core_ndims_structure, tf.convert_to_tensor, args, check_types=False))
      if len(parts) != len(flat_core_ndims):
        raise ValueError('Number of args does not match `core_ndims` '
                         '({} vs {}). Saw argument parts {}; core '
                         'ndims {}.'.format(len(parts), len(flat_core_ndims),
                                            parts, flat_core_ndims))

      # `vectorized_map` requires all inputs to have a single, common batch
      # dimension `[n]`. So we broadcast all input parts to a common
      # batch shape, then flatten it down to a single dimension.

      # First, compute how many 'extra' (batch) ndims each part has. This must
      # be nonnegative.
      part_shapes = [tf.shape(part) for part in parts]
      batch_ndims = [
          prefer_static.rank_from_shape(part_shape) - nd
          for (part_shape, nd) in zip(part_shapes, flat_core_ndims)]
      static_ndims = [tf.get_static_value(nd) for nd in batch_ndims]
      if any([nd and nd < 0 for nd in static_ndims]):
        raise ValueError('Cannot broadcast a Tensor having lower rank than the '
                         'specified `core_ndims`! (saw input ranks {}, '
                         '`core_ndims` {}).'.format(
                             tf.nest.map_structure(
                                 prefer_static.rank_from_shape, part_shapes),
                             flat_core_ndims))
      if validate_args:
        for nd, part, core_nd in zip(batch_ndims, parts, flat_core_ndims):
          assertions.append(tf.debugging.assert_non_negative(
              nd, message='Cannot broadcast a Tensor having lower rank than '
              'the specified `core_ndims`! (saw {} vs minimum rank {}).'.format(
                  part, core_nd)))

      # Next, split each part's shape into batch and core shapes, and
      # broadcast the batch shapes.
      with tf.control_dependencies(assertions):
        batch_shapes, core_shapes = zip(*[
            (part_shape[:nd], part_shape[nd:])
            for (part_shape, nd) in zip(part_shapes, batch_ndims)])
        broadcast_batch_shape = functools.reduce(
            prefer_static.broadcast_shape, batch_shapes, [])

      # Flatten all of the batch dimensions into one.
      n = tf.cast(prefer_static.reduce_prod(broadcast_batch_shape), tf.int32)
      static_n = tf.get_static_value(n)
      if static_n == 1:
        result = fn(*args)
      else:
        # Pad all input parts to the common shape, then flatten
        # into the single leading dimension `[n]`.
        # TODO(b/145227909): If/when vmap supports broadcasting, use nested vmap
        # when batch rank is static so that we can exploit broadcasting.
        broadcast_parts = [
            tf.broadcast_to(part, prefer_static.concat([broadcast_batch_shape,
                                                        core_shape], axis=0))
            for (part, core_shape) in zip(parts, core_shapes)]
        parts_with_flattened_batch_dim = [
            tf.reshape(part, prefer_static.concat([[n], core_shape], axis=0))
            for (part, core_shape) in zip(broadcast_parts, core_shapes)]

        # Run the vectorized computation
        batched_result = tf.vectorized_map(lambda args: fn(*args),
                                           nest.pack_sequence_as(
                                               args,
                                               parts_with_flattened_batch_dim))

        # Unflatten the result
        result = nest.map_structure(
            lambda x: tf.reshape(x, prefer_static.concat([  # pylint: disable=g-long-lambda
                broadcast_batch_shape, prefer_static.shape(x)[1:]], axis=0)),
            batched_result)
    return result

  return vectorized_fn

