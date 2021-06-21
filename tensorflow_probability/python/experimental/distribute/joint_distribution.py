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
from tensorflow_probability.python.bijectors import identity as identity_bijector
from tensorflow_probability.python.distributions import joint_distribution as jd_lib
from tensorflow_probability.python.distributions import log_prob_ratio as lp_ratio
from tensorflow_probability.python.internal import distribute_lib
from tensorflow_probability.python.internal import samplers


def pbroadcast_value(value, value_axis_names, output_axis_names):
  value_axis_names = distribute_lib.canonicalize_named_axis(value_axis_names)
  pbroadcast_axes = [
      axis_name for axis_name in output_axis_names
      if axis_name not in value_axis_names
  ]
  return distribute_lib.pbroadcast(value, named_axis=pbroadcast_axes)


def _maybe_substitute_or_add_value_in_tuple(value_tuple, index, value):
  if index > len(value_tuple):
    raise ValueError('Cannot add value to tuple without available slot.')
  if index == len(value_tuple):
    return value_tuple + (value,)
  curr_value = value_tuple[index]
  if curr_value is not None:
    return value_tuple
  return value_tuple[:index] + (value,) + value_tuple[index + 1:]


class JointDistributionDistributedMixin(object):
  """A JDMixin that shards the log_prob calculation."""

  def _call_execute_model(
      self,
      sample_shape=(),
      seed=None,
      value=None,
      sample_and_trace_fn=jd_lib.trace_distributions_and_values):
    return self._distribute_execute_model(
        sample_shape=sample_shape,
        seed=seed,
        value=value if value is None else self._model_flatten(value),
        sample_and_trace_fn=sample_and_trace_fn)

  def _distribute_execute_model(
      self,
      sample_shape=(),
      seed=None,
      value=None,
      sample_and_trace_fn=jd_lib.trace_distributions_and_values):
    """Executes a model, adding `pbroadcasts` to ensure correct gradients."""
    shard_axis_names = self._model_flatten(self.experimental_shard_axis_names)
    final_values_out = []
    if value is None:
      value = ()

    def sample_and_trace_value_fn(dist,
                                  sample_shape,
                                  seed,
                                  value=None):
      value, traced = sample_and_trace_fn(
          dist=dist, sample_shape=sample_shape, seed=seed, value=value)
      # We trace `next_value` here so we can pass it back in as part of `value`
      # in the next iteration of the coroutine.
      return value, (value, traced)

    for output_index, output_axes in enumerate(shard_axis_names):
      # We pbroadcast all values according to the difference between the current
      # `output_axes` and their own active axes.
      previous_shard_axes = shard_axis_names[:len(value)]
      pbroadcasted_value = tuple(
          pbroadcast_value(v, v_axis_names, output_axes)
          for v, v_axis_names in zip(value, previous_shard_axes)
      )
      pbroadcasted_values, traced_values = zip(*super()._execute_model(
          sample_shape=sample_shape,
          seed=seed,
          value=pbroadcasted_value + (None,),
          stop_index=output_index + 1,
          sample_and_trace_fn=sample_and_trace_value_fn))
      value = _maybe_substitute_or_add_value_in_tuple(
          value, output_index, pbroadcasted_values[output_index])
      final_values_out.append(traced_values[output_index])
    return final_values_out

  def _default_event_space_bijector(self, *args, **kwargs):
    if args or kwargs:
      return _DefaultJointBijector(self.experimental_pin(*args, **kwargs))
    return _DefaultJointBijector(self)


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
    return jd_lib._jd_log_prob_ratio(p, x, q, y, name=name)  # pylint: disable=protected-access


class _DefaultJointBijector(jd_lib._DefaultJointBijector):  # pylint: disable=protected-access
  """Sharding-compatible event space bijector for JDs."""

  def _conditioned_bijectors(self, samples, constrained=False):
    if samples is None:
      return self.bijectors

    def sample_and_trace_fn(dist, value, **_):
      bij = self._bijector_fn(dist)
      if bij is None:
        bij = identity_bijector.Identity()

      # If the RV is not yet constrained, transform it.
      value = value if constrained else bij.forward(value)
      return jd_lib.ValueWithTrace(value=value, traced=bij)

    return self._jd._call_execute_model(  # pylint: disable=protected-access
        sample_shape=(),
        value=samples,
        seed=samplers.zeros_seed(),
        sample_and_trace_fn=sample_and_trace_fn)
