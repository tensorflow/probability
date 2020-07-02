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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

import tensorflow.compat.v2 as tf

from tensorflow_probability.python.distributions import joint_distribution_named
from tensorflow_probability.python.distributions import joint_distribution_sequential
from tensorflow_probability.python.internal import distribution_util as dist_util

from tensorflow.python.util import nest  # pylint: disable=g-direct-tensorflow-import

__all__ = [
    'independent_joint_distribution_from_structure',
]


def independent_joint_distribution_from_structure(structure_of_distributions):
  """Turns a (potentially nested) structure of dists into a single dist.

  Args:
    structure_of_distributions: instance of `tfd.Distribution`, or nested
      structure (tuple, list, dict, etc.) in which all leaves are
      `tfd.Distribution` instances.
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
    return structure_of_distributions

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
        independent_joint_distribution_from_structure,
        structure_of_distributions)

  # Otherwise, build a JD from the current structure.
  if (hasattr(structure_of_distributions, '_asdict') or
      isinstance(structure_of_distributions, collections.Mapping)):
    return joint_distribution_named.JointDistributionNamed(
        structure_of_distributions)
  return joint_distribution_sequential.JointDistributionSequential(
      structure_of_distributions)
