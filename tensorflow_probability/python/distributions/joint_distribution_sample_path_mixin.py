# Copyright 2019 The TensorFlow Probability Authors.
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
"""`JointDistribution` mixin class implementing sample-path semantics."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

import tensorflow.compat.v2 as tf

from tensorflow_probability.python import math as tfp_math
from tensorflow_probability.python.distributions import joint_distribution as jd_lib
from tensorflow_probability.python.internal import assert_util
from tensorflow_probability.python.internal import prefer_static


def _get_reduction_axes(x, nd):
  """Enumerates the final `nd` axis indices of `x`."""
  x_rank = prefer_static.rank_from_shape(prefer_static.shape(x))
  return prefer_static.range(x_rank - 1, x_rank - nd -1, -1)


class JointDistributionSamplePathMixin(object):
  """Mixin endowing a JointDistribution with sample-path semantics.

  This class provides alternate vectorization semantics for
  `tfd.JointDistribution`, which in many cases eliminate the need to
  explicitly account for batch shapes in the model specification.
  Instead of simply summing the `log_prob`s of component distributions
  (which may have different shapes), it first reduces the component `log_prob`s
  to ensure that `jd.log_prob(jd.sample())` always returns a scalar, unless
  otherwise specified.

  Sample-path semantics can be summarized as follows:

  - An `event` of a sample-path JointDistribution is the structure of
    tensors produced by `.sample()`; thus, the `event_shape` is the
    structure containing the shapes of sampled tensors. These combine both
    the event and batch dimensions of the component distributions. By contrast,
    the event shape of a base `JointDistribution`s does not include batch
    dimensions of component distributions.
  - The `batch_shape` is a global property of the entire model, rather
    than a per-component property as in base `JointDistribution`s.
    The global batch shape must be a prefix of the batch shapes of
    each component; the length of this prefix is specified by an optional
    argument `batch_ndims`. If `batch_ndims` is not specified, the model has
    batch shape `[]`.

  For further discussion and examples, see
  `tfp.distributions.JointDistributionCoroutineAutoBatched`,
  `tfp.distributions.JointDistributionNamedAutoBatched`,
  and `tfp.distributions.JointDistributionSequentialAutoBatched`.

  """

  def __init__(self, *args, **kwargs):
    self._batch_ndims = kwargs.pop('batch_ndims', 0)
    self._experimental_use_kahan_sum = kwargs.pop(
        'experimental_use_kahan_sum', False)
    super(JointDistributionSamplePathMixin, self).__init__(*args, **kwargs)

  @property
  def batch_ndims(self):
    return self._batch_ndims

  def _batch_shape_parts(self):
    static_batch_ndims = tf.get_static_value(self.batch_ndims)
    static_batch_shapes = self._get_static_distribution_attributes().batch_shape
    if static_batch_ndims is None:
      return [tf.TensorShape(None)] * len(static_batch_shapes)
    return [batch_shape[:static_batch_ndims]
            for batch_shape in static_batch_shapes]

  def _batch_shape(self):
    # Caching will not leak graph Tensors since this is a static attribute.
    reduce_fn = ((lambda a, b: a.merge_with(b)) if self.validate_args
                 else tf.broadcast_static_shape)  # Allows broadcasting.
    return functools.reduce(reduce_fn, self._batch_shape_parts())

  def _batch_shape_tensor_parts(self):
    return [d.batch_shape_tensor()[:self.batch_ndims]
            for d in self._get_single_sample_distributions()]

  def _batch_shape_tensor(self):
    return tf.convert_to_tensor(functools.reduce(
        prefer_static.broadcast_shape, self._batch_shape_tensor_parts()))

  def _event_shape(self):
    """Static shape of a single sample from a single batch."""
    if not hasattr(self, '_cached_event_shape'):
      part_attrs = self._get_static_distribution_attributes()
      static_batch_ndims = tf.get_static_value(self.batch_ndims)
      if static_batch_ndims is None:
        flat_joint_event_shape = [tf.TensorShape(None)] * len(part_attrs.dtype)
      else:
        flat_joint_event_shape = [
            # Recurse over joint component dists.
            tf.nest.map_structure(  # pylint: disable=g-complex-comprehension
                part_batch_shape[static_batch_ndims:].concatenate,
                part_event_shape)
            for part_event_shape, part_batch_shape in zip(
                part_attrs.event_shape, part_attrs.batch_shape)]
      self._cached_event_shape = flat_joint_event_shape
    return self._model_unflatten(self._cached_event_shape)

  def _event_shape_tensor(self):
    """Shape of a single sample from a single batch."""
    component_shapes = []
    for d in self._get_single_sample_distributions():
      iid_event_shape = d.batch_shape_tensor()[self.batch_ndims:]
      # Recurse over the (potentially joint) component distribution's event.
      component_shapes.append(tf.nest.map_structure(
          lambda a, b=iid_event_shape: prefer_static.concat([b, a], axis=0),
          d.event_shape_tensor()))
    return self._model_unflatten(component_shapes)

  def _reduce_measure_over_dists(self, xs, reduce_fn):
    num_trailing_batch_dims_treated_as_event = [
        prefer_static.rank_from_shape(
            d.batch_shape_tensor()) - self._batch_ndims
        for d in self._get_single_sample_distributions()]

    with tf.control_dependencies(self._maybe_check_batch_shape()):
      return [reduce_fn(unreduced_x, axis=_get_reduction_axes(unreduced_x, nd))
              for (unreduced_x, nd) in zip(
                  xs, num_trailing_batch_dims_treated_as_event)]

  def _maybe_check_batch_shape(self):
    assertions = []
    if self.validate_args:
      parts = self._batch_shape_tensor_parts()
      for s in parts[1:]:
        assertions.append(assert_util.assert_equal(
            parts[0], s, message='Component batch shapes are inconsistent.'))
    return assertions

  def _reduce_log_probs_over_dists(self, lps):
    if self._experimental_use_kahan_sum:
      return sum(jd_lib.maybe_check_wont_broadcast(
          self._reduce_measure_over_dists(
              lps, reduce_fn=tfp_math.reduce_kahan_sum),
          self.validate_args)).total
    else:
      return sum(jd_lib.maybe_check_wont_broadcast(
          self._reduce_measure_over_dists(lps, reduce_fn=tf.reduce_sum),
          self.validate_args))

  def _sample_and_log_prob(self, sample_shape, seed, value=None, **kwargs):
    xs, lps = zip(
        *self._call_execute_model(
            sample_shape,
            seed=seed,
            value=self._resolve_value(value=value,
                                      allow_partially_specified=True,
                                      **kwargs),
            sample_and_trace_fn=jd_lib.trace_values_and_log_probs))
    return self._model_unflatten(xs), self._reduce_log_probs_over_dists(lps)

  def _log_prob(self, value):
    return self._reduce_log_probs_over_dists(
        self._map_measure_over_dists('log_prob', value))

  def log_prob_parts(self, value, name='log_prob_parts'):
    """Log probability density/mass function, part-wise.

    Args:
      value: `list` of `Tensor`s in `distribution_fn` order for which we compute
        the `log_prob_parts` and to parameterize other ("downstream")
        distributions.
      name: name prepended to ops created by this function.
        Default value: `"log_prob_parts"`.

    Returns:
      log_prob_parts: a `tuple` of `Tensor`s representing the `log_prob` for
        each `distribution_fn` evaluated at each corresponding `value`.
    """
    with self._name_and_control_scope(name):
      sum_fn = tf.reduce_sum
      if self._experimental_use_kahan_sum:
        sum_fn = lambda x, axis: tfp_math.reduce_kahan_sum(x, axis=axis).total
      return self._model_unflatten(
          self._reduce_measure_over_dists(
              self._map_measure_over_dists('log_prob', value),
              sum_fn))

  def _unnormalized_log_prob(self, value):
    return self._reduce_log_probs_over_dists(
        self._map_measure_over_dists('unnormalized_log_prob', value))

  def unnormalized_log_prob_parts(self, value, name=None):
    """Unnormalized log probability density/mass function, part-wise.

    Args:
      value: `list` of `Tensor`s in `distribution_fn` order for which we compute
        the `unnormalized_log_prob_parts` and to parameterize other
        ("downstream") distributions.
      name: name prepended to ops created by this function.
        Default value: `"unnormalized_log_prob_parts"`.

    Returns:
      log_prob_parts: a `tuple` of `Tensor`s representing the `log_prob` for
        each `distribution_fn` evaluated at each corresponding `value`.
    """
    with self._name_and_control_scope(name or 'unnormalized_log_prob_parts'):
      sum_fn = tf.reduce_sum
      if self._experimental_use_kahan_sum:
        sum_fn = lambda x, axis: tfp_math.reduce_kahan_sum(x, axis=axis).total
      return self._model_unflatten(
          self._reduce_measure_over_dists(
              self._map_measure_over_dists('unnormalized_log_prob', value),
              sum_fn))

  def prob_parts(self, value, name='prob_parts'):
    """Log probability density/mass function.

    Args:
      value: `list` of `Tensor`s in `distribution_fn` order for which we compute
        the `prob_parts` and to parameterize other ("downstream") distributions.
      name: name prepended to ops created by this function.
        Default value: `"prob_parts"`.

    Returns:
      prob_parts: a `tuple` of `Tensor`s representing the `prob` for
        each `distribution_fn` evaluated at each corresponding `value`.
    """
    with self._name_and_control_scope(name):
      return self._model_unflatten(
          self._reduce_measure_over_dists(
              self._map_measure_over_dists('prob', value),
              tf.reduce_prod))

  def is_scalar_batch(self, name='is_scalar_batch'):
    """Indicates that `batch_shape == []`.

    Args:
      name: Python `str` prepended to names of ops created by this function.

    Returns:
      is_scalar_batch: `bool` scalar `Tensor`.
    """
    with self._name_and_control_scope(name):
      return self._is_scalar_helper(self.batch_shape, self.batch_shape_tensor())
