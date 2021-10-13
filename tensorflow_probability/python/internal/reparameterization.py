# Copyright 2018 The TensorFlow Probability Authors.
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
"""ReparameterizationType class implementation."""

__all__ = [
    "ReparameterizationType",
    "FULLY_REPARAMETERIZED",
    "NOT_REPARAMETERIZED",
]


class ReparameterizationType(object):
  """Instances of this class represent how sampling is reparameterized.

  Two static instances exist in the distributions library, signifying
  one of two possible properties for samples from a distribution:

  `FULLY_REPARAMETERIZED`: Samples from the distribution are fully
    reparameterized, and straight-through gradients are supported.

  `NOT_REPARAMETERIZED`: Samples from the distribution are not fully
    reparameterized, and straight-through gradients are either partially
    unsupported or are not supported at all. In this case, for purposes of
    e.g. RL or variational inference, it is generally safest to wrap the
    sample results in a `stop_gradients` call and use policy
    gradients / surrogate loss instead.
  """

  def __init__(self, rep_type):
    self._rep_type = rep_type

  def __repr__(self):
    return "<Reparameterization Type: %s>" % self._rep_type

  def __hash__(self):
    return hash(self._rep_type)

  def __eq__(self, other):
    """Determine if this `ReparameterizationType` is equal to another.

    Since RepaparameterizationType instances are constant static global
    instances, equality checks if two instances' id() values are equal.

    Args:
      other: Object to compare against.

    Returns:
      `self is other`.
    """
    return self is other


# Fully reparameterized distribution: samples from a fully
# reparameterized distribution support straight-through gradients with
# respect to all parameters.
FULLY_REPARAMETERIZED = ReparameterizationType("FULLY_REPARAMETERIZED")


# Not reparameterized distribution: samples from a non-
# reparameterized distribution do not support straight-through gradients for
# at least some of the parameters.
NOT_REPARAMETERIZED = ReparameterizationType("NOT_REPARAMETERIZED")
