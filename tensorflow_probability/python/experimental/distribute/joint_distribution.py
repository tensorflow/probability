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

import functools

import tensorflow.compat.v2 as tf
from tensorflow_probability.python import distributions as distribution_lib
from tensorflow_probability.python.distributions import log_prob_ratio as lp_ratio
from tensorflow_probability.python.experimental.distribute import distribute_lib
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

      def inner_log_prob_parts(flat_value):
        unflat_value = self._model_unflatten(flat_value)
        ds, xs = self._call_flat_sample_distributions(
            value=unflat_value, seed=samplers.zeros_seed())
        # For sharded distributions, we need to make sure not to do an
        # all-reduce.
        axis_names = self._model_flatten(self.experimental_shard_axis_names)
        log_prob_fns = [
            functools.partial(getattr(d, attr), reduce_over_shards=False)
            if axis_name else getattr(d, attr)
            for d, axis_name in zip(ds, axis_names)
        ]
        # We need to flatten and unflatten here to ensure the output structure
        # matches `flat_sharded_distributions`.
        vals = self._model_unflatten(
            [log_prob_fn(x) for log_prob_fn, x in zip(log_prob_fns, xs)])
        return self._model_flatten(vals)

      flat_value = self._model_flatten(value)
      flat_axis_names = self._model_flatten(self.experimental_shard_axis_names)
      flat_xs = distribute_lib.make_sharded_log_prob_parts(
          inner_log_prob_parts, flat_axis_names)(
              flat_value)
      return iter(flat_xs)
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

    def log_prob_ratio_parts_fn(x_y):
      x = tf.nest.map_structure(lambda part: part[0], x_y)
      y = tf.nest.map_structure(lambda part: part[1], x_y)
      p_dists = p.sample_distributions(value=x, seed=samplers.zeros_seed())[0]
      q_dists = q.sample_distributions(value=y, seed=samplers.zeros_seed())[0]
      # Ensure sharded distributions defer reductions.
      kwds = lambda a: {'reduce_over_shards': False} if a else {}
      return nest.map_structure_up_to(
          p_dists,
          lambda p, x, q, y, s: lp_ratio.log_prob_ratio(p, x, q, y, **kwds(s)),
          p_dists, x, q_dists, y, p_axis_names)

    return tf.add_n(
        tf.nest.flatten(
            distribute_lib.make_sharded_log_prob_parts(
                log_prob_ratio_parts_fn,
                # Stack, because make_sharded_log_prob_parts expects
                # inputs/outputs to be 1 to 1. TODO(b/175084455): revisit this
                # after the distributed bijectors are done, as it is likely that
                # make_sharded_log_prob_parts will be adjusted then to not have
                # this limitation.
                p_axis_names)(tf.nest.map_structure(
                    lambda x, y: tf.stack([x, y], axis=0), x, y))))
