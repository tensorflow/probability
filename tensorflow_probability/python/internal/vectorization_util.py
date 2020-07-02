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
import warnings

import numpy as np

import tensorflow.compat.v2 as tf
from tensorflow_probability.python.internal import distribution_util
from tensorflow_probability.python.internal import prefer_static
from tensorflow_probability.python.internal import samplers
from tensorflow_probability.python.util import SeedStream
from tensorflow_probability.python.util.seed_stream import TENSOR_SEED_MSG_PREFIX
from tensorflow.python.ops import parallel_for  # pylint: disable=g-direct-tensorflow-import
from tensorflow.python.util import nest  # pylint: disable=g-direct-tensorflow-import

__all__ = [
    'iid_sample',
    'make_rank_polymorphic'
]

# Cause all warnings to always be triggered.
# Not having this means subsequent calls wont trigger the warning.
warnings.filterwarnings('always',
                        module='tensorflow_probability.*vectorization_util',
                        append=True)  # Don't override user-set filters.


def iid_sample(sample_fn, sample_shape):
  """Lift a sampling function to one that draws multiple iid samples.

  Args:
    sample_fn: Python `callable` that returns a (possibly nested) structure of
      `Tensor`s. May optionally take a `seed` named arg: if so, any `int`
      seeds (for stateful samplers) are passed through directly, while any
      pair-of-`int` seeds (for stateless samplers) are split into independent
      seeds for each sample.
    sample_shape: `int` `Tensor` shape of iid samples to draw.
  Returns:
    iid_sample_fn: Python `callable` taking the same arguments as `sample_fn`
      and returning iid samples. Each returned `Tensor` will have shape
      `concat([sample_shape, shape_of_original_returned_tensor])`.
  """
  sample_shape = distribution_util.expand_to_vector(
      prefer_static.cast(sample_shape, np.int32), tensor_name='sample_shape')
  n = prefer_static.cast(
      prefer_static.reduce_prod(sample_shape), dtype=np.int32)

  def unflatten(x):
    unflattened_shape = prefer_static.cast(
        prefer_static.concat([
            sample_shape, prefer_static.shape(x)[1:]], axis=0),
        dtype=np.int32)
    return tf.reshape(x, unflattened_shape)

  def iid_sample_fn(*args, **kwargs):
    """Draws iid samples from `fn`."""

    pfor_loop_body = lambda _: sample_fn(*args, **kwargs)

    seed = kwargs.pop('seed', None)
    try:  # Assume that `seed` is a valid stateful seed (Python `int`).
      kwargs = dict(kwargs, seed=SeedStream(seed, salt='iid_sample')())
      pfor_loop_body = lambda _: sample_fn(*args, **kwargs)
    except TypeError as e:
      # If a stateless seed arg is passed, split it into `n` different stateless
      # seeds, so that we don't just get a bunch of copies of the same sample.
      if TENSOR_SEED_MSG_PREFIX not in str(e):
        raise
      warnings.warn('Saw non-`int` seed {}, implying stateless sampling. '
                    'Autovectorized functions that use stateless sampling '
                    'may be quite slow because the current implementation '
                    'falls back to an explicit loop. This will be fixed in the '
                    'future. For now, you will likely see better performance '
                    'from stateful sampling, which you can invoke by passing a'
                    'traditional Python `int` seed.'.format(seed))
      seed = samplers.split_seed(seed, n=n, salt='iid_sample_stateless')
      pfor_loop_body = (
          lambda i: sample_fn(*args, seed=tf.gather(seed, i), **kwargs))

    draws = parallel_for.pfor(pfor_loop_body, n)
    return tf.nest.map_structure(unflatten, draws, expand_composites=True)

  return iid_sample_fn


def _lock_in_non_vectorized_args(fn, arg_structure, flat_core_ndims, flat_args):
  """Wraps `fn` to take only those args with non-`None` core ndims."""

  # Extract the indices and values of args where core_ndims is not `None`.
  (vectorized_arg_indices,
   vectorized_arg_core_ndims,
   vectorized_args) = [], [], []
  if any(nd is not None for nd in flat_core_ndims):
    vectorized_arg_indices, vectorized_arg_core_ndims, vectorized_args = zip(*[
        (i, nd, tf.convert_to_tensor(t))
        for i, (nd, t) in enumerate(zip(flat_core_ndims, flat_args))
        if nd is not None])

  vectorized_arg_index_set = set(vectorized_arg_indices)

  def fn_of_vectorized_args(vectorized_args):
    vectorized_args_by_index = dict(
        zip(vectorized_arg_indices, vectorized_args))
    # Substitute the vectorized args into the original argument
    # structure.
    new_args_with_original_structure = nest.pack_sequence_as(
        arg_structure, [vectorized_args_by_index[i]
                        if i in vectorized_arg_index_set
                        else v for (i, v) in enumerate(flat_args)])
    return fn(*new_args_with_original_structure)

  return (vectorized_arg_core_ndims,
          vectorized_args,
          fn_of_vectorized_args)


# TODO(b/145252136): merge `make_rank_polymorphic` into core TensorFlow.
def make_rank_polymorphic(fn, core_ndims, validate_args=False, name=None):
  """Lift a function to one that vectorizes across arbitrary-rank inputs.

  Args:
    fn: Python `callable` `result = fn(*args)` where all arguments
      and the returned result(s) are (structures of) Tensors. Non-`Tensor`
      arguments may also be passed through by specifying a value of `None`
      in `core_ndims`.
    core_ndims: structure of `int` Tensors and/or `None` values, of the same
      structure as `args`. Each `int` contains the
      expected rank of the corresponding `Tensor` argument in an unvectorized
      call to `fn`; `None` values denote arguments that should not be vectorized
      (e.g., non-`Tensor` arguments). May alternately be a single scalar
      Tensor `int` applicable to all `args`.
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

  Lifted functions may accept non-`Tensor` args, denoted by `None` values in
  `core_ndims`. These values will be passed unmodified to the underlying
  function. For example, we could generalize `add` above to take an
  arbitrary binary operation, specified as a Python callable.

  ```python
  import operator
  def apply_binop(fn, a, b):
    return fn(a, b)
  apply_binop(operator.mul, tf.constant([1., 2.]), tf.constant(3.))
    # ==> Returns [3., 6.]

  # Batching, we pass the `fn` arg by specifying `core_ndims` of `None`.
  apply_binop_to_vector_and_scalar = make_rank_polymorphic(
    apply_binop, core_ndims=(None, 1, 0))
  apply_binop_to_vector_and_scalar(
    operator.mul, tf.constant([1., 2.]), tf.constant([3., 4., 5.]))
    # ==> Returns [[3., 6.], [4., 8.], [5., 10.]]
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
      flat_args = nest.flatten_up_to(
          core_ndims_structure, args, check_types=False)

      # Filter to only the `Tensor`-valued args (taken to be those with `None`
      # values for `core_ndims`). Other args will be passed through to `fn`
      # unmodified.
      (vectorized_arg_core_ndims,
       vectorized_args,
       fn_of_vectorized_args) = _lock_in_non_vectorized_args(
           fn,
           arg_structure=core_ndims_structure,
           flat_core_ndims=flat_core_ndims,
           flat_args=flat_args)

      # `vectorized_map` requires all inputs to have a single, common batch
      # dimension `[n]`. So we broadcast all input parts to a common
      # batch shape, then flatten it down to a single dimension.

      # First, compute how many 'extra' (batch) ndims each part has. This must
      # be nonnegative.
      vectorized_arg_shapes = [tf.shape(arg) for arg in vectorized_args]
      batch_ndims = [
          prefer_static.rank_from_shape(arg_shape) - nd
          for (arg_shape, nd) in zip(
              vectorized_arg_shapes, vectorized_arg_core_ndims)]
      static_ndims = [tf.get_static_value(nd) for nd in batch_ndims]
      if any([nd and nd < 0 for nd in static_ndims]):
        raise ValueError('Cannot broadcast a Tensor having lower rank than the '
                         'specified `core_ndims`! (saw input ranks {}, '
                         '`core_ndims` {}).'.format(
                             tf.nest.map_structure(
                                 prefer_static.rank_from_shape,
                                 vectorized_arg_shapes),
                             vectorized_arg_core_ndims))
      if validate_args:
        for nd, part, core_nd in zip(
            batch_ndims, vectorized_args, vectorized_arg_core_ndims):
          assertions.append(tf.debugging.assert_non_negative(
              nd, message='Cannot broadcast a Tensor having lower rank than '
              'the specified `core_ndims`! (saw {} vs minimum rank {}).'.format(
                  part, core_nd)))

      # Next, split each part's shape into batch and core shapes, and
      # broadcast the batch shapes.
      with tf.control_dependencies(assertions):
        empty_shape = np.zeros([0], dtype=np.int32)
        batch_shapes, core_shapes = empty_shape, empty_shape
        if vectorized_arg_shapes:
          batch_shapes, core_shapes = zip(*[
              (arg_shape[:nd], arg_shape[nd:])
              for (arg_shape, nd) in zip(vectorized_arg_shapes, batch_ndims)])
        broadcast_batch_shape = (
            functools.reduce(prefer_static.broadcast_shape, batch_shapes, []))

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
        broadcast_vectorized_args = [
            tf.broadcast_to(part, prefer_static.concat(
                [broadcast_batch_shape, core_shape], axis=0))
            for (part, core_shape) in zip(vectorized_args, core_shapes)]
        vectorized_args_with_flattened_batch_dim = [
            tf.reshape(part, prefer_static.concat([[n], core_shape], axis=0))
            for (part, core_shape) in zip(
                broadcast_vectorized_args, core_shapes)]
        batched_result = tf.vectorized_map(
            fn_of_vectorized_args, vectorized_args_with_flattened_batch_dim)

        # Unflatten any `Tensor`s in the result.
        unflatten = lambda x: tf.reshape(x, prefer_static.concat([  # pylint: disable=g-long-lambda
            broadcast_batch_shape, prefer_static.shape(x)[1:]], axis=0))
        result = nest.map_structure(
            lambda x: unflatten(x) if tf.is_tensor(x) else x, batched_result,
            expand_composites=True)
    return result

  return vectorized_fn

