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
from tensorflow_probability.python.distributions import log_prob_ratio as lp_ratio
from tensorflow_probability.python.internal import distribute_lib
from tensorflow_probability.python.internal import samplers

from tensorflow.python.util import nest  # pylint: disable=g-direct-tensorflow-import


class JointDistributionDistributedMixin(object):
  """A JDMixin that shards the log_prob calculation."""

  def _map_measure_over_dists(self, attr, value):
    """Override the default implementation to shard its log_prob calculation."""
    if any(x is None for x in tf.nest.flatten(value)):
      raise ValueError('No `value` part can be `None`; saw: {}.'.format(value))
    if (attr in ('log_prob', 'unnormalized_log_prob')) and any(
        self.experimental_shard_axis_names):

      def inner_log_prob_parts(value):
        ds, xs = self._call_flat_sample_distributions(
            value=value, seed=samplers.zeros_seed())
        # We need to flatten and unflatten here to ensure the output structure
        # matches `flat_sharded_distributions`.
        return self._model_unflatten(
            [getattr(d, attr)(x) for d, x in zip(ds, xs)])

      axis_names = self.experimental_shard_axis_names
      # Individual distributions will apply psum in their `log_prob` methods
      # so we need to pbroadcast `value` according to `axis_names` to provide
      # correct gradients. We are safe to add pbroadcasts to functions with
      # psums already in them.
      log_prob_parts = distribute_lib.make_pbroadcast_function(
          inner_log_prob_parts, (axis_names,), axis_names,
          out_dtype=value)(value)
      return iter(tf.nest.flatten(log_prob_parts))
    ds, xs = self._call_flat_sample_distributions(
        value=value, seed=samplers.zeros_seed())
    return (getattr(d, attr)(x) for d, x in zip(ds, xs))


class JointDistributionSequential(JointDistributionDistributedMixin,
                                  distribution_lib.JointDistributionSequential):
  """A sharding-aware JointDistributionSequential."""

  _composite_tensor_nonshape_params = ('model',)


class JointDistributionNamed(JointDistributionDistributedMixin,
                             distribution_lib.JointDistributionNamed):
  """A sharding-aware JointDistributionNamed."""

  _composite_tensor_nonshape_params = ('model',)


class JointDistributionCoroutine(JointDistributionDistributedMixin,
                                 distribution_lib.JointDistributionCoroutine):
  """A sharding-aware JointDistributionCoroutine."""


@lp_ratio.RegisterLogProbRatio(JointDistributionSequential)
@lp_ratio.RegisterLogProbRatio(JointDistributionNamed)
@lp_ratio.RegisterLogProbRatio(JointDistributionCoroutine)
def _dist_jd_log_prob_ratio(p, x, q, y, name=None):
  """Distributed log-prob ratio for JDs."""
  with tf.name_scope(name or 'dist_jd_log_prob_ratio'):
    tf.nest.assert_same_structure(x, y)

    p_axis_names = p.experimental_shard_axis_names
    q_axis_names = q.experimental_shard_axis_names
    if p_axis_names != q_axis_names:
      raise ValueError('p and q must use the same sharding. '
                       f'Saw: p: {p}, {p_axis_names}, q: {q}, {q_axis_names}')

    def log_prob_ratio_parts_fn(x, y):
      p_dists = p.sample_distributions(value=x, seed=samplers.zeros_seed())[0]
      q_dists = q.sample_distributions(value=y, seed=samplers.zeros_seed())[0]
      return nest.map_structure_up_to(
          p_dists,
          lp_ratio.log_prob_ratio,
          p_dists, x, q_dists, y)

    return tf.add_n(
        tf.nest.flatten(
            distribute_lib.make_pbroadcast_function(
                log_prob_ratio_parts_fn,
                in_axes=(p_axis_names, p_axis_names),
                out_axes=p_axis_names,
                out_dtype=x)(x, y)))
