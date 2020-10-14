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
"""Contains sharding-aware versions of tfd.JointDistributions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v2 as tf
from tensorflow_probability.python import distributions as distribution_lib

from tensorflow_probability.python.experimental.distribute import distribute_lib
from tensorflow_probability.python.experimental.distribute import sharded


class JointDistributionDistributedMixin(object):
  """A JDMixin that shards the log_prob calculation."""

  def get_sharded_distributions(self):
    """Indicates for each part distribution whether or not it is sharded."""
    ds = self._get_single_sample_distributions()
    return self._model_unflatten((
        isinstance(d, (sharded.ShardedIndependent, sharded.ShardedSample))
        for d in ds))

  def _map_measure_over_dists(self, attr, value):
    """Overrides the default implementation to shard its log_prob calculation."""
    if any(x is None for x in tf.nest.flatten(value)):
      raise ValueError('No `value` part can be `None`; saw: {}.'.format(value))
    if attr == 'log_prob' and any(self.get_sharded_distributions()):

      def inner_log_prob_parts(flat_value):
        unflat_value = self._model_unflatten(flat_value)
        ds, xs = self._call_flat_sample_distributions(
            value=unflat_value, seed=42)
        # We need to flatten and unflatten here to ensure the output structure
        # matches `flat_sharded_distributions`.
        vals = self._model_unflatten(
            [getattr(d, attr)(x) for d, x in zip(ds, xs)])
        return self._model_flatten(vals)

      flat_value = self._model_flatten(value)
      flat_sharded_distributions = self._model_flatten(
          self.get_sharded_distributions())
      flat_xs = distribute_lib.make_sharded_log_prob_parts(
          inner_log_prob_parts, flat_sharded_distributions)(
              flat_value)
      return iter(flat_xs)
    ds, xs = self._call_flat_sample_distributions(value=value, seed=42)
    return (getattr(d, attr)(x) for d, x in zip(ds, xs))


class JointDistributionSequential(JointDistributionDistributedMixin,
                                  distribution_lib.JointDistributionSequential):
  pass


class JointDistributionNamed(JointDistributionDistributedMixin,
                             distribution_lib.JointDistributionNamed):
  pass


class JointDistributionCoroutine(JointDistributionDistributedMixin,
                                 distribution_lib.JointDistributionCoroutine):
  pass
