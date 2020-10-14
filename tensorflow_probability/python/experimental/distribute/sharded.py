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
"""Distributions for distributed computations."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports

import tensorflow.compat.v2 as tf

from tensorflow_probability.python.distributions import independent as independent_lib
from tensorflow_probability.python.distributions import sample as sample_lib
from tensorflow_probability.python.internal import prefer_static as ps
from tensorflow_probability.python.internal import samplers


class ShardedSample(sample_lib.Sample):
  """A version of `tfd.Sample` that shards its output across devices."""

  def __init__(self,
               distribution,
               sample_shape=(),
               shard_axis=0,
               validate_args=False,
               name=None):
    """Construct the `ShardedSample` distribution.

    Args:
      distribution: The base distribution instance to transform. Typically an
        instance of `Distribution`.
      sample_shape: `int` scalar or vector `Tensor` representing the shape of a
        single sample.
      shard_axis: `int` representing which axis of `sample_shape` will be
        sharded across devices.
      validate_args: Python `bool`.  Whether to validate input with asserts. If
        `validate_args` is `False`, and the inputs are invalid, correct behavior
        is not guaranteed.
      name: The name for ops managed by the distribution.
        Default value: `None` (i.e., `'Sample' + distribution.name`).
    """
    with tf.name_scope(name or 'ShardedSample' + distribution.name) as name:
      self._shard_axis = shard_axis

      super(ShardedSample, self).__init__(
          distribution,
          validate_args=validate_args,
          sample_shape=sample_shape,
          name=name)

  @property
  def sample_shape(self):
    sample_shape = ps.reshape(self._sample_shape, shape=[-1])
    shard_axis_size = sample_shape[self.shard_axis]
    num_devices = self.num_devices
    if shard_axis_size % num_devices != 0:
      raise ValueError('Does not shard evenly.')
    shard_size = shard_axis_size // num_devices
    sample_shape = ps.concat([
        sample_shape[:self.shard_axis], [shard_size],
        sample_shape[self.shard_axis + 1:]
    ],
                             axis=0)
    return sample_shape

  @property
  def shard_axis(self):
    return self._shard_axis

  @property
  def replica_id(self):
    ctx = tf.distribute.get_replica_context()
    return ctx.replica_id_in_sync_group

  @property
  def num_devices(self):
    ctx = tf.distribute.get_replica_context()
    return ctx.num_replicas_in_sync

  def _sample_n(self, n, seed, **kwargs):
    seed = samplers.sanitize_seed(seed, salt='sharded_sample_sample')
    return super(ShardedSample, self)._sample_n(n, seed + self.replica_id,
                                                **kwargs)


class ShardedIndependent(independent_lib.Independent):
  """A version of `tfd.Independent` that folds device id into its randomness."""

  @property
  def replica_id(self):
    ctx = tf.distribute.get_replica_context()
    return ctx.replica_id_in_sync_group

  def _sample_n(self, n, seed, **kwargs):
    seed = samplers.sanitize_seed(seed, salt='sharded_independent_sample')
    return super(ShardedIndependent, self)._sample_n(n, seed + self.replica_id,
                                                     **kwargs)
