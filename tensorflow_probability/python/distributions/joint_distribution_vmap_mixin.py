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
"""`JointDistribution` mixin class implementing automatic vectorization."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v2 as tf

from tensorflow_probability.python.distributions import joint_distribution as joint_distribution_lib
from tensorflow_probability.python.internal import prefer_static as ps
from tensorflow_probability.python.internal import vectorization_util


JAX_MODE = False


def _might_have_nonzero_size(sample_shape):
  static_size = tf.get_static_value(tf.size(sample_shape))
  return (static_size is None) or static_size >= 1


def _might_have_excess_ndims(flat_value, flat_core_ndims):
  for v, nd in zip(flat_value, flat_core_ndims):
    static_excess_ndims = (
        0 if v is None else
        tf.get_static_value(ps.convert_to_shape_tensor(ps.rank(v) - nd)))
    if static_excess_ndims is None or static_excess_ndims > 0:
      return True
  return False


def _pad_value_to_full_length(value, dtype):
  """Fills a partial `value` structure with `None`s for any unspecified RVs."""
  # If dtype is dict-like, set missing values to `None`.
  if hasattr(dtype, 'keys'):
    return type(dtype)({k: value.get(k, None) for k in dtype.keys()})

  # Otherwise, dtype is a sequence, so append `None`s.
  return tf.nest.pack_sequence_as(dtype,
                                  [value[i] if i < len(value) else None
                                   for i in range(len(dtype))])


# Lint doesn't know that docstrings are defined in the base JD class.
# pylint: disable=missing-docstring
class JointDistributionVmapMixin(object):
  """A joint distribution with automatically vectorized sample and log-prob.

  Auto-vectorized variants of JointDistribution treat the underlying
  model as describing a single possible world, or equivalently, as
  specifying the process of generating a single sample from the model.
  Drawing multiple samples, and computing batched log-probs, is accomplished
  using `tf.vectorized_map`. In many cases this allows for significant
  simplication of the model. For example, the following
  manually-vectorized `tfd.JointDistributionCoroutine` model:

  ```python
  def model_fn():
    x = yield tfd.JointDistributionCoroutine.Root(
      tfd.Normal(0., tf.ones([3])))
    y = yield tfd.JointDistributionCoroutine.Root(
      tfd.Normal(0., 1.)))
    z = yield tfd.Normal(x[..., :2] + y[..., tf.newaxis], 1.)

  can be written in auto-vectorized form as

  ```python
  def model_fn():
    x = yield tfd.Normal(0., tf.ones([3]))
    y = yield tfd.Normal(0., 1.))
    z = yield tfd.Normal(x[:2] + y, 1.)
  ```

  in which we were able to drop the specification of `Root` nodes and to
  avoid explicitly accounting for batch dimensions when indexing and slicing
  computed quantities in the third line.

  Note: auto-vectorization is still experimental and some TensorFlow ops may
  be unsupported.

  A limitation relative to standard `JointDistribution`s is that the
  `sample_distributions()` method does not currently support (nontrivial) sample
  shapes.
  """

  def __init__(self, *args, **kwargs):
    self._use_vectorized_map = kwargs.pop('use_vectorized_map', True)
    super(JointDistributionVmapMixin, self).__init__(*args, **kwargs)

  # TODO(b/166658748): Drop this (make it always True).
  _stateful_to_stateless = JAX_MODE

  @property
  def use_vectorized_map(self):
    return self._use_vectorized_map

  @property
  def _single_sample_ndims(self):
    """Computes the rank of values produced by executing the base model."""
    result = []
    for d in self._get_single_sample_distributions():
      batch_ndims = ps.rank_from_shape(d.batch_shape_tensor, d.batch_shape)
      result.append(tf.nest.map_structure(
          lambda a, b, nd=batch_ndims: nd + ps.rank_from_shape(a, b),
          d.event_shape_tensor(),
          d.event_shape))
    return result

  def sample_distributions(self, sample_shape=(), seed=None, value=None,
                           name='sample_distributions'):
    with self._name_and_control_scope(name):

      value_might_have_sample_dims = False
      if value is not None:
        value = _pad_value_to_full_length(value, self.dtype)
        value = tf.nest.map_structure(
            lambda v: v if v is None else tf.convert_to_tensor(v), value)
        value_might_have_sample_dims = _might_have_excess_ndims(
            flat_value=self._model_flatten(value),
            flat_core_ndims=self._single_sample_ndims)

      # TODO(b/157953455): Return distributions as CompositeTensors once
      # vectorized_map supports this.
      if self.use_vectorized_map and (
          _might_have_nonzero_size(sample_shape) or
          value_might_have_sample_dims):
        raise NotImplementedError('`sample_distributions` with nontrivial '
                                  'sample shape is not yet supported '
                                  'for autovectorized JointDistributions.')
      else:
        ds, xs = self._call_flat_sample_distributions(
            sample_shape=sample_shape, seed=seed, value=value)
      return self._model_unflatten(ds), self._model_unflatten(xs)

  def _sample_n(self, sample_shape, seed, value=None, **kwargs):

    value_might_have_sample_dims = False
    if value is not None:
      value = _pad_value_to_full_length(value, self.dtype)
      value = tf.nest.map_structure(
          lambda v: v if v is None else tf.convert_to_tensor(v), value)
      value_might_have_sample_dims = _might_have_excess_ndims(
          flat_value=self._model_flatten(value),
          flat_core_ndims=self._single_sample_ndims)

    if not self.use_vectorized_map or not (
        _might_have_nonzero_size(sample_shape) or
        value_might_have_sample_dims):
      # No need to auto-vectorize.
      xs = self._call_flat_sample_distributions(
          sample_shape=sample_shape, seed=seed, value=value, **kwargs)[1]
      return self._model_unflatten(xs)

    # Set up for autovectorized sampling. To support the `value` arg, we need to
    # first understand which dims are from the model itself, then wrap
    # `_call_flat_sample_distributions` to batch over all remaining dims.
    value_core_ndims = None
    if value is not None:
      value_core_ndims = tf.nest.map_structure(
          lambda v, nd: None if v is None else nd,
          value, self._model_unflatten(self._single_sample_ndims),
          check_types=False)
    batch_flat_sample = vectorization_util.make_rank_polymorphic(
        lambda v, seed: self._call_flat_sample_distributions(  # pylint: disable=g-long-lambda
            sample_shape=(), seed=seed, value=v)[1],
        core_ndims=[value_core_ndims, None],
        validate_args=self.validate_args)

    # Draw samples.
    vectorized_flat_sample = vectorization_util.iid_sample(
        # Redefine the polymorphic fn to hack around `make_rank_polymorphic`
        # not currently supporting keyword args.
        lambda v, seed: batch_flat_sample(v, seed), sample_shape)  # pylint: disable=unnecessary-lambda
    xs = vectorized_flat_sample(value, seed=seed)
    return self._model_unflatten(xs)

  # Redefine `_map_measure_over_dists` to autovectorize the measure if needed.
  def _map_measure_over_dists(self, attr, value):
    if any(x is None for x in self._model_flatten(value)):
      raise ValueError('No `value` part can be `None`; saw: {}.'.format(value))
    if value is not None:
      value = self._model_flatten(value)

    def map_measure_fn(value):
      # We always provide a seed, since _flat_sample_distributions will
      # unconditionally split the seed.
      with tf.name_scope('map_measure_fn'):
        constant_seed = joint_distribution_lib.dummy_seed()
        return [getattr(d, attr)(x) for (d, x) in zip(
            *self._flat_sample_distributions(value=value, seed=constant_seed))]
    if self.use_vectorized_map:
      map_measure_fn = vectorization_util.make_rank_polymorphic(
          map_measure_fn,
          core_ndims=[self._single_sample_ndims],
          validate_args=self.validate_args)

    return map_measure_fn(value)
