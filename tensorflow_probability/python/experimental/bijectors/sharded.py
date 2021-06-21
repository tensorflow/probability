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
"""Bijectors for distributed computations."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v2 as tf

from tensorflow_probability.python.bijectors import bijector as bijector_lib
from tensorflow_probability.python.internal import distribute_lib

JAX_MODE = False


class Sharded(bijector_lib.Bijector):
  """A meta-bijector meant for use in an SPMD distributed context.

  `Sharded` is a meta-bijector that enables distributions to be used in SPMD
  programs. A `Sharded` bijector represents a bijector which acts on elements
  are split across a set of devices. The number of shards is the number of
  devices in the current TensorFlow DistributionStrategy or the provided JAX
  pmap axis.
  """

  def __init__(self, bijector, *, shard_axis_name, name=None):
    """Constructs a `Sharded` bijector.

    Args:
      bijector: The base bijector instance to transform. Typically an instance
        of `Bijector`.
      shard_axis_name: `str` for axis name for use in JAX backend.
      name: The name for ops managed by the distribution.
        Default value: `None` (i.e., `'Sharded' + bijector.name`).
    """
    if JAX_MODE and shard_axis_name is None:
      raise ValueError('Cannot provide a `None` axis name in JAX backend.')

    with tf.name_scope(name or 'Sharded' + bijector.name) as name:
      super().__init__(
          name=name,
          forward_min_event_ndims=bijector.forward_min_event_ndims,
          inverse_min_event_ndims=bijector.inverse_min_event_ndims)
      self._bijector = bijector
      self._shard_axis_name = shard_axis_name

  @property
  def bijector(self):
    return self._bijector

  @property
  def shard_axis_name(self):
    return self._shard_axis_name

  def _forward(self, x):
    return self.bijector.forward(x)

  def _inverse(self, y):
    return self.bijector.inverse(y)

  def _forward_log_det_jacobian(self, x, **kwargs):

    return distribute_lib.psum(
        self.bijector.forward_log_det_jacobian(x, **kwargs),
        named_axis=self.shard_axis_name)

  def _inverse_log_det_jacobian(self, y, **kwargs):

    return distribute_lib.psum(
        self.bijector.inverse_log_det_jacobian(y, **kwargs),
        named_axis=self.shard_axis_name)
