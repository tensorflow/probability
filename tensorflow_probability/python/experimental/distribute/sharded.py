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
from tensorflow_probability.python.experimental.bijectors import sharded as sharded_bij
from tensorflow_probability.python.internal import distribute_lib
from tensorflow_probability.python.internal import parameter_properties
from tensorflow_probability.python.internal import samplers


JAX_MODE = False


def _implement_sharded_lp_fn(fn_name):
  """Implements log_prob or unnormalized_log_prob."""
  def lp_fn(self, x):
    lp = getattr(self.distribution, fn_name)(x)
    return distribute_lib.psum(lp, self.experimental_shard_axis_names)

  lp_fn.__name__ = f'_{fn_name}'
  return lp_fn


class Sharded(distribution_lib.Distribution):
  """A meta-distribution meant for use in an SPMD distributed context.

  `Sharded` is a meta-distribution that enables distributions to be used in SPMD
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
      shard_axis_name: `str` or a list of strings for axis name(s). An empty
        list means that no sharding is actually done. This can be `None` under
        the TensorFlow backend (meaning a sharded axis is present, but
        anonymous). Only the JAX backend supports multiple axes names.
      validate_args: Python `bool`.  Whether to validate input with asserts. If
        `validate_args` is `False`, and the inputs are invalid, correct behavior
        is not guaranteed.
      name: The name for ops managed by the distribution.
        Default value: `None` (i.e., `'Sharded' + distribution.name`).
    """
    parameters = dict(locals())

    if shard_axis_name is None:
      if JAX_MODE:
        # In JAX, axes names matter and we don't know which axis name the user
        # might intend, so we bail.
        raise ValueError('Cannot provide a `None` axis name in JAX backend.')
      else:
        # In TF, there are no axes names, so we can pick a reasonable default.
        shard_axis_name = [True]

    # Use inner axes before outer axes
    full_shard_axis_name = (
        distribution.experimental_shard_axis_names +
        distribute_lib.canonicalize_named_axis(shard_axis_name))

    if not JAX_MODE:
      if len(full_shard_axis_name) > 1:
        raise ValueError(
            'TensorFlow backend does not support multiple shard axes:\n'
            'inner shard_axis_names: '
            f'{list(distribution.experimental_shard_axis_names)}\n'
            f'outer shard_axis_names: {list(shard_axis_name)}')

    if len(set(full_shard_axis_name)) != len(full_shard_axis_name):
      duplicates = set()
      seen = set()
      for axis_name in full_shard_axis_name:
        if axis_name in seen:
          duplicates.add(axis_name)
        seen.add(axis_name)
      raise ValueError(
          'Found duplicate axis name(s).\n'
          'inner shard_axis_names: '
          f'{list(distribution.experimental_shard_axis_names)}\n'
          f'outer shard_axis_names: {shard_axis_name}\n'
          f'duplicates: {list(duplicates)}')

    with tf.name_scope(name or 'Sharded' + distribution.name) as name:
      self._distribution = distribution
      self._shard_axis_name = full_shard_axis_name
      super(Sharded, self).__init__(
          dtype=self._distribution.dtype,
          validate_args=validate_args,
          allow_nan_stats=self._distribution.allow_nan_stats,
          reparameterization_type=self._distribution.reparameterization_type,
          parameters=parameters,
          name=name)

  @property
  def experimental_shard_axis_names(self):
    return self._shard_axis_name

  @classmethod
  def _parameter_properties(cls, dtype, num_classes=None):
    return dict(
        distribution=parameter_properties.BatchedComponentProperties())

  @property
  def distribution(self):
    return self._distribution

  def _sample_n(self, n, seed, **kwargs):
    seed = samplers.sanitize_seed(seed, salt='sharded_sample')
    seed = distribute_lib.fold_in_axis_index(
        seed, self.experimental_shard_axis_names)
    return self.distribution.sample(sample_shape=n, seed=seed, **kwargs)

  def _variance(self):
    return self.distribution.variance()

  _log_prob = _implement_sharded_lp_fn('log_prob')
  _unnormalized_log_prob = _implement_sharded_lp_fn('unnormalized_log_prob')

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
    bij = self.distribution.experimental_default_event_space_bijector(
        *args, **kwargs)
    if bij is None:
      return None
    return sharded_bij.Sharded(
        bij, shard_axis_name=self.experimental_shard_axis_names)


@log_prob_ratio.RegisterLogProbRatio(Sharded)
def _sharded_log_prob_ratio(p, x, q, y, name=None):
  """Distributed log-prob ratio for Sharded."""
  with tf.name_scope(name or 'sharded_log_prob_ratio'):
    if p.experimental_shard_axis_names != q.experimental_shard_axis_names:
      raise ValueError(
          'Mismatched axis names '
          f'"{p.experimental_shard_axis_names}" vs "'
          f'"{q.experimental_shard_axis_names}"')

    def log_prob_ratio_fn(x, y):
      return log_prob_ratio.log_prob_ratio(p.distribution, x,
                                           q.distribution, y)

    axes = p.experimental_shard_axis_names
    return distribute_lib.make_psum_function(
        log_prob_ratio_fn, in_axes=(axes, axes), out_axes=axes,
        out_dtype=x)(x, y)
