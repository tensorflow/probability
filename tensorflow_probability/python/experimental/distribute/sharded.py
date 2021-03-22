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

from tensorflow_probability.python.distributions import distribution as distribution_lib
from tensorflow_probability.python.distributions import log_prob_ratio
from tensorflow_probability.python.experimental.distribute import distribute_lib
from tensorflow_probability.python.internal import samplers


JAX_MODE = False


class Sharded(distribution_lib.Distribution):
  """A meta-distribution meant for use in an SPMD distributed context.

  `Sharded` is a meta-distribution enables distributions to be used in SPMD
  programs. A `Sharded` distribution represents a random variable that has
  been split across a set of devices. The number of shards is the number of
  devices in the current TensorFlow DistributionStrategy or the provided JAX
  pmap axis.

  In practice, `Sharded` modifies its input distribution in two ways.
  First, when a `Sharded` distribution is sampled, it first folds the current
  device ID into the input random seed, resulting in different samples on each
  device. Second, when computing the `log_prob` of a value, a `Sharded`
  distribution aggregates the log-prob over all devices, resulting in the same
  synchronized value.
  """

  def __init__(self, distribution, shard_axis_name=None, validate_args=False,
               name=None):

    """Constructs a `Sharded` distribution.

    Args:
      distribution: The base distribution instance to transform. Typically an
        instance of `Distribution`.
      shard_axis_name: `str` for axis name for use in JAX backend.
      validate_args: Python `bool`.  Whether to validate input with asserts. If
        `validate_args` is `False`, and the inputs are invalid, correct behavior
        is not guaranteed.
      name: The name for ops managed by the distribution.
        Default value: `None` (i.e., `'Sharded' + distribution.name`).
    """
    parameters = dict(locals())

    with tf.name_scope(name or 'Sharded' + distribution.name) as name:
      self._distribution = distribution
      self._shard_axis_name = shard_axis_name
      super(Sharded, self).__init__(
          dtype=self._distribution.dtype,
          validate_args=validate_args,
          allow_nan_stats=self._distribution.allow_nan_stats,
          reparameterization_type=self._distribution.reparameterization_type,
          parameters=parameters,
          name=name)

  experimental_is_sharded = True

  @property
  def distribution(self):
    return self._distribution

  @property
  def shard_axis_name(self):
    return self._shard_axis_name

  @property
  def replica_id(self):
    return distribute_lib.get_replica_id(axis_name=self.shard_axis_name)

  @property
  def num_devices(self):
    return distribute_lib.get_num_replicas(axis_name=self.shard_axis_name)

  def _sample_n(self, n, seed, **kwargs):
    seed = samplers.sanitize_seed(seed, salt='sharded_sample')
    seed = samplers.fold_in(seed, tf.cast(self.replica_id, tf.int32))
    return self.distribution.sample(sample_shape=n, seed=seed, **kwargs)

  def _log_prob(self, x, reduce_over_shards=True, **kwargs):

    def log_prob_fn(value):
      return self.distribution.log_prob(value, **kwargs)

    if reduce_over_shards:
      log_prob_fn = distribute_lib.make_sharded_log_prob_parts(
          log_prob_fn, is_sharded=True, axis_name=self.shard_axis_name)
    return log_prob_fn(x)

  def _batch_shape_tensor(self):
    return self.distribution.batch_shape_tensor()

  def _batch_shape(self):
    return self.distribution.batch_shape

  def _event_shape_tensor(self):
    return self.distribution.event_shape_tensor()

  def _event_shape(self):
    return self.distribution.event_shape

  def _parameter_control_dependencies(self, is_init):
    if JAX_MODE:
      return []
    return self.distribution._parameter_control_dependencies(is_init=is_init)  # pylint: disable=protected-access

  def _default_event_space_bijector(self, *args, **kwargs):
    # TODO(b/175084455): This should likely be wrapped in a `tfb.Sharded`-like
    # construct.
    return self.distribution.experimental_default_event_space_bijector(
        *args, **kwargs)

  _composite_tensor_nonshape_params = ('distribution',)


@log_prob_ratio.RegisterLogProbRatio(Sharded)
def _sharded_log_prob_ratio(p, x, q, y, name=None, reduce_over_shards=True):
  """Distributed log-prob ratio for Sharded."""
  with tf.name_scope(name or 'sharded_log_prob_ratio'):
    if p.shard_axis_name != q.shard_axis_name:
      raise ValueError('Mismatched axis names '
                       f'"{p.shard_axis_name}" vs "{q.shard_axis_name}"')

    def log_prob_ratio_fn(x_y):
      return log_prob_ratio.log_prob_ratio(p.distribution, x_y[0],
                                           q.distribution, x_y[1])

    if reduce_over_shards:
      return distribute_lib.make_sharded_log_prob_parts(
          # Stack, because make_sharded_log_prob_parts expects inputs/outputs to
          # be 1 to 1. TODO(b/175084455): revisit this after the distributed
          # bijectors are done, as it is likely that make_sharded_log_prob_parts
          # will be adjusted then to not have this limitation.
          log_prob_ratio_fn,
          is_sharded=True,
          axis_name=p.shard_axis_name)(
              tf.stack([x, y], axis=0))
    return log_prob_ratio_fn([x, y])
