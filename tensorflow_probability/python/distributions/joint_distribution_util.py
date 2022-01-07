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
"""Tests for JointDistribution utilities."""

import collections
import functools

import tensorflow.compat.v2 as tf

from tensorflow_probability.python.distributions import independent
from tensorflow_probability.python.distributions import joint_distribution_auto_batched
from tensorflow_probability.python.distributions import joint_distribution_named
from tensorflow_probability.python.distributions import joint_distribution_sequential
from tensorflow_probability.python.internal import distribution_util as dist_util
from tensorflow_probability.python.internal import prefer_static as ps

from tensorflow.python.util import nest  # pylint: disable=g-direct-tensorflow-import

__all__ = [
    'independent_joint_distribution_from_structure',
]


def independent_joint_distribution_from_structure(structure_of_distributions,
                                                  batch_ndims=None,
                                                  validate_args=False):
  """Turns a (potentially nested) structure of dists into a single dist.

  Args:
    structure_of_distributions: instance of `tfd.Distribution`, or nested
      structure (tuple, list, dict, etc.) in which all leaves are
      `tfd.Distribution` instances.
    batch_ndims: Optional integer `Tensor` number of leftmost batch dimensions
      shared across all members of the input structure. If this is specified,
      the returned joint distribution will be an autobatched distribution with
      the given batch rank, and all other dimensions absorbed into the event.
    validate_args: Python `bool`. Whether the joint distribution should validate
      input with asserts. This imposes a runtime cost. If `validate_args` is
      `False`, and the inputs are invalid, correct behavior is not guaranteed.
      Default value: `False`.
  Returns:
    distribution: instance of `tfd.Distribution` such that
      `distribution.sample()` is equivalent to
      `tf.nest.map_structure(lambda d: d.sample(), structure_of_distributions)`.
      If `structure_of_distributions` was indeed a structure (as opposed to
      a single `Distribution` instance), this will be a `JointDistribution`
      with the corresponding structure.
  Raises:
    TypeError: if any leaves of the input structure are not `tfd.Distribution`
      instances.
  """
  # If input is already a Distribution, just return it.
  if dist_util.is_distribution_instance(structure_of_distributions):
    dist = structure_of_distributions
    if batch_ndims is not None:
      excess_ndims = ps.rank_from_shape(dist.batch_shape_tensor()) - batch_ndims
      if tf.get_static_value(excess_ndims) != 0:  # Static value may be None.
        dist = independent.Independent(dist,
                                       reinterpreted_batch_ndims=excess_ndims)
    return dist

  # If this structure contains other structures (ie, has elements at depth > 1),
  # recursively turn them into JDs.
  element_depths = nest.map_structure_with_tuple_paths(
      lambda path, x: len(path), structure_of_distributions)
  if max(tf.nest.flatten(element_depths)) > 1:
    next_level_shallow_structure = nest.get_traverse_shallow_structure(
        traverse_fn=lambda x: min(tf.nest.flatten(x)) <= 1,
        structure=element_depths)
    structure_of_distributions = nest.map_structure_up_to(
        next_level_shallow_structure,
        functools.partial(independent_joint_distribution_from_structure,
                          batch_ndims=batch_ndims,
                          validate_args=validate_args),
        structure_of_distributions)

  jdnamed = joint_distribution_named.JointDistributionNamed
  jdsequential = joint_distribution_sequential.JointDistributionSequential
  # Use an autobatched JD if a specific batch rank was requested.
  if batch_ndims is not None:
    jdnamed = functools.partial(
        joint_distribution_auto_batched.JointDistributionNamedAutoBatched,
        batch_ndims=batch_ndims, use_vectorized_map=False)
    jdsequential = functools.partial(
        joint_distribution_auto_batched.JointDistributionSequentialAutoBatched,
        batch_ndims=batch_ndims, use_vectorized_map=False)

  # Otherwise, build a JD from the current structure.
  if (hasattr(structure_of_distributions, '_asdict') or
      isinstance(structure_of_distributions, collections.abc.Mapping)):
    return jdnamed(structure_of_distributions, validate_args=validate_args)
  return jdsequential(structure_of_distributions, validate_args=validate_args)
