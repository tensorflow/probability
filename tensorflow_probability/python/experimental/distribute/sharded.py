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
from tensorflow_probability.python.distributions import log_prob_ratio
from tensorflow_probability.python.distributions import sample as sample_lib
from tensorflow_probability.python.experimental.distribute import distribute_lib
from tensorflow_probability.python.internal import prefer_static as ps
from tensorflow_probability.python.internal import samplers


JAX_MODE = False


class ShardedSample(sample_lib.Sample):
  """A version of `tfd.Sample` that shards its output across devices."""

  def __init__(self,
               distribution,
               sample_shape=(),
               shard_axis=0,
               shard_axis_name=None,
               validate_args=False,
               experimental_use_kahan_sum=False,
               name=None):
    """Construct the `ShardedSample` distribution.

    Args:
      distribution: The base distribution instance to transform. Typically an
        instance of `Distribution`.
      sample_shape: `int` scalar or vector `Tensor` representing the shape of a
        single sample.
      shard_axis: `int` representing which axis of `sample_shape` will be
        sharded across devices.
      shard_axis_name: `str` for axis name for use in JAX backend.
      validate_args: Python `bool`.  Whether to validate input with asserts. If
        `validate_args` is `False`, and the inputs are invalid, correct behavior
        is not guaranteed.
      experimental_use_kahan_sum: Python `bool`. When `True`, we use Kahan
        summation to aggregate independent underlying log_prob values, which
        improves against the precision of a naive float32 sum. This can be
        noticeable in particular for large dimensions in float32. See CPU caveat
        on `tfp.math.reduce_kahan_sum`.
      name: The name for ops managed by the distribution.
        Default value: `None` (i.e., `'ShardedSample' + distribution.name`).
    """
    parameters = dict(locals())

    with tf.name_scope(name or 'ShardedSample' + distribution.name) as name:
      self._shard_axis = shard_axis
      self._shard_axis_name = shard_axis_name
      super(ShardedSample, self).__init__(
          distribution,
          validate_args=validate_args,
          sample_shape=sample_shape,
          experimental_use_kahan_sum=experimental_use_kahan_sum,
          name=name)
      self._parameters = parameters

  is_sharded = True

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
    ], axis=0)
    return sample_shape

  @property
  def shard_axis_name(self):
    return self._shard_axis_name

  @property
  def shard_axis(self):
    return self._shard_axis

  @property
  def replica_id(self):
    return distribute_lib.get_replica_id(axis_name=self.shard_axis_name)

  @property
  def num_devices(self):
    return distribute_lib.get_num_replicas(axis_name=self.shard_axis_name)

  def _sample_n(self, n, seed, **kwargs):
    seed = samplers.sanitize_seed(seed, salt='sharded_sample_sample')
    seed = samplers.fold_in(seed, tf.cast(self.replica_id, tf.int32))
    return super(ShardedSample, self)._sample_n(n, seed, **kwargs)

  def _log_prob(self, value, reduce_over_shards=True, **kwargs):
    out_log_prob = super(ShardedSample, self)._log_prob(value, **kwargs)
    if reduce_over_shards:
      return distribute_lib.psum(out_log_prob, axis_name=self.shard_axis_name)
    return out_log_prob

  def _parameter_control_dependencies(self, is_init=False):
    if not self.validate_args:
      return []
    return super(ShardedSample, self)._parameter_control_dependencies(
        is_init=is_init)


@log_prob_ratio.RegisterLogProbRatio(ShardedSample)
def _sharded_sample_log_prob_ratio(p, x, q, y, reduce_over_shards=True):
  """Distributed log-prob ratio for ShardedSample."""
  if p.shard_axis_name != q.shard_axis_name:
    raise ValueError(
        f'Mismatched axis names "{p.shard_axis_name}" vs "{q.shard_axis_name}"')
  underlying = sample_lib._sample_log_prob_ratio(p, x, q, y)  # pylint: disable=protected-access
  if reduce_over_shards:
    return distribute_lib.psum(underlying, axis_name=p.shard_axis_name)
  return underlying


class ShardedIndependent(independent_lib.Independent):
  """A version of `tfd.Independent` that folds device id into its randomness."""

  def __init__(self,
               distribution,
               reinterpreted_batch_ndims=None,
               validate_args=False,
               shard_axis_name=None,
               experimental_use_kahan_sum=False,
               name=None):
    """Construct a `ShardedIndependent` distribution.

    Args:
      distribution: The base distribution instance to transform. Typically an
        instance of `Distribution`.
      reinterpreted_batch_ndims: Scalar, integer number of rightmost batch dims
        which will be regarded as event dims. When `None` all but the first
        batch axis (batch axis 0) will be transferred to event dimensions
        (analogous to `tf.layers.flatten`).
      validate_args: Python `bool`.  Whether to validate input with asserts. If
        `validate_args` is `False`, and the inputs are invalid, correct behavior
        is not guaranteed.
      shard_axis_name: `str` for axis name for use in JAX backend.
      experimental_use_kahan_sum: Python `bool`. When `True`, we use Kahan
        summation to aggregate independent underlying log_prob values, which
        improves against the precision of a naive float32 sum. This can be
        noticeable in particular for large dimensions in float32. See CPU caveat
        on `tfp.math.reduce_kahan_sum`.
      name: The name for ops managed by the distribution.
        Default value: `'ShardedIndependent' + distribution.name`.

    Raises:
      ValueError: if `reinterpreted_batch_ndims` exceeds
        `distribution.batch_ndims`
    """
    parameters = dict(locals())

    with tf.name_scope(name or
                       'ShardedIndependent' + distribution.name) as name:
      self._shard_axis_name = shard_axis_name
      super(ShardedIndependent, self).__init__(
          distribution,
          reinterpreted_batch_ndims=reinterpreted_batch_ndims,
          validate_args=validate_args,
          experimental_use_kahan_sum=experimental_use_kahan_sum,
          name=name)
      self._parameters = parameters

  is_sharded = True

  @property
  def shard_axis_name(self):
    return self._shard_axis_name

  def _log_prob(self, value, reduce_over_shards=True, **kwargs):
    out_log_prob = super(ShardedIndependent, self)._log_prob(value, **kwargs)
    if reduce_over_shards:
      return distribute_lib.psum(out_log_prob, axis_name=self.shard_axis_name)
    return out_log_prob

  @property
  def replica_id(self):
    return distribute_lib.get_replica_id(axis_name=self.shard_axis_name)

  @property
  def num_devices(self):
    return distribute_lib.get_num_replicas(axis_name=self.shard_axis_name)

  def _sample_n(self, n, seed, **kwargs):
    seed = samplers.sanitize_seed(seed, salt='sharded_independent_sample')
    seed = samplers.fold_in(seed, tf.cast(self.replica_id, tf.int32))
    return super(ShardedIndependent, self)._sample_n(n, seed, **kwargs)

  def _parameter_control_dependencies(self, is_init):
    if JAX_MODE:
      return []
    return super(ShardedIndependent, self)._parameter_control_dependencies(
        is_init=is_init)


@log_prob_ratio.RegisterLogProbRatio(ShardedIndependent)
def _sharded_independent_log_prob_ratio(p, x, q, y, reduce_over_shards=True):
  """Distributed log-prob ratio for ShardedIndependent."""
  if p.shard_axis_name != q.shard_axis_name:
    raise ValueError(
        f'Mismatched axis names "{p.shard_axis_name}" vs "{q.shard_axis_name}"')
  underlying = independent_lib._independent_log_prob_ratio(p, x, q, y)  # pylint: disable=protected-access
  if reduce_over_shards:
    return distribute_lib.psum(underlying, axis_name=p.shard_axis_name)
  return underlying
