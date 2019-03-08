# Copyright 2018 The TensorFlow Probability Authors.
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
"""Slicing utility for tfd.Distribution."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import warnings

import six
import tensorflow as tf

__all__ = ['batch_slice']


# We track the provenance of a sliced or copied distribution all the way back to
# the arguments initially provided to the first tfd.Distribution constructor.
# This allows us to ensure that sub-sliced and copied distributions retain the
# gradient back to any source variables provided up-front. e.g. we want the
# following to work:
#    v = tf.compat.v2.Variable(tf.random.uniform([]))
#    dist = tfd.Normal(v, 1)
#    with tf.GradientTape() as tape:
#      lp = dist[...].log_prob(0.)
#    dlpdv = tape.gradient(lp, v)
# dlpdv should not be None.
PROVENANCE_ATTR = '_tfp_batch_slice_provenance'

ALL_SLICE = slice(None)


def _slice_single_param(param, param_event_ndims, slices, dist_batch_shape):
  """Slices a single parameter of a distribution.

  Args:
    param: A `Tensor`, the original parameter to slice.
    param_event_ndims: `int` event parameterization rank for this parameter.
    slices: A `tuple` of normalized slices.
    dist_batch_shape: The distribution's batch shape `Tensor`.

  Returns:
    new_param: A `Tensor`, batch-sliced according to slices.
  """
  # Extend param shape with ones on the left to match dist_batch_shape.
  param_shape = tf.shape(input=param)
  insert_ones = tf.ones(
      [tf.size(input=dist_batch_shape) + param_event_ndims - tf.rank(param)],
      dtype=param_shape.dtype)
  new_param_shape = tf.concat([insert_ones, param_shape], axis=0)
  full_batch_param = tf.reshape(param, new_param_shape)
  param_slices = []
  # We separately track the batch axis from the parameter axis because we want
  # them to align for positive indexing, and be offset by param_event_ndims for
  # negative indexing.
  param_dim_idx = 0
  batch_dim_idx = 0
  for slc in slices:
    if slc is tf.newaxis:
      param_slices.append(slc)
      continue
    if slc is Ellipsis:
      if batch_dim_idx < 0:
        raise ValueError('Found multiple `...` in slices {}'.format(slices))
      param_slices.append(slc)
      # Switch over to negative indexing for the broadcast check.
      num_remaining_non_newaxis_slices = sum(
          [s is not tf.newaxis for s in slices[slices.index(Ellipsis) + 1:]])
      batch_dim_idx = -num_remaining_non_newaxis_slices
      param_dim_idx = batch_dim_idx - param_event_ndims
      continue
    # Find the batch dimension sizes for both parameter and distribution.
    param_dim_size = new_param_shape[param_dim_idx]
    batch_dim_size = dist_batch_shape[batch_dim_idx]
    is_broadcast = batch_dim_size > param_dim_size
    # Slices are denoted by start:stop:step.
    if isinstance(slc, slice):
      start, stop, step = slc.start, slc.stop, slc.step
      if start is not None:
        start = tf.where(is_broadcast, 0, start)
      if stop is not None:
        stop = tf.where(is_broadcast, 1, stop)
      if step is not None:
        step = tf.where(is_broadcast, 1, step)
      param_slices.append(slice(start, stop, step))
    else:  # int, or int Tensor, e.g. d[d.batch_shape_tensor()[0] // 2]
      param_slices.append(tf.where(is_broadcast, 0, slc))
    param_dim_idx += 1
    batch_dim_idx += 1
  param_slices.extend([ALL_SLICE] * param_event_ndims)
  return full_batch_param.__getitem__(param_slices)


def _slice_params_to_dict(dist, params_event_ndims, slices):
  """Computes the override dictionary of sliced parameters.

  Args:
    dist: The tfd.Distribution being batch-sliced.
    params_event_ndims: Per-event parameter ranks, a `str->int` `dict`.
    slices: Slices as received by __getitem__.

  Returns:
    overrides: `str->Tensor` `dict` of batch-sliced parameter overrides.
  """
  override_dict = {}
  for param_name, param_event_ndims in six.iteritems(params_event_ndims):
    # Verify that either None or a legit value is in the parameters dict.
    if param_name not in dist.parameters:
      raise ValueError('Distribution {} is missing advertised '
                       'parameter {}'.format(dist, param_name))
    param = dist.parameters[param_name]
    if param is None:
      # some distributions have multiple possible parameterizations; this
      # param was not provided
      continue
    dtype = None
    if hasattr(dist, param_name):
      attr = getattr(dist, param_name)
      dtype = getattr(attr, 'dtype', None)
    if dtype is None:
      dtype = dist.dtype
      warnings.warn('Unable to find property getter for parameter Tensor {} '
                    'on {}, falling back to Distribution.dtype {}'.format(
                        param_name, dist, dtype))
    param = tf.convert_to_tensor(value=param, dtype=dtype)
    override_dict[param_name] = _slice_single_param(param, param_event_ndims,
                                                    slices,
                                                    dist.batch_shape_tensor())
  return override_dict


def _apply_single_step(dist, params_event_ndims, slices, params_overrides):
  """Applies a single slicing step to `dist`, returning a new instance."""
  if len(slices) == 1 and slices[0] == Ellipsis:
    # The path used by Distribution.copy: batch_slice(...args..., Ellipsis)
    override_dict = {}
  else:
    override_dict = _slice_params_to_dict(dist, params_event_ndims, slices)
  override_dict.update(params_overrides)
  parameters = dict(dist.parameters, **override_dict)
  new_dist = type(dist)(**parameters)
  return new_dist


def _apply_slice_sequence(dist, params_event_ndims, slice_overrides_seq):
  """Applies a sequence of slice or copy-with-overrides operations to `dist`."""
  for slices, overrides in slice_overrides_seq:
    dist = _apply_single_step(dist, params_event_ndims, slices, overrides)
  return dist


def batch_slice(dist, params_event_ndims, params_overrides, slices):
  """Slices `dist` along its batch dimensions. Helper for tfd.Distribution.

  Args:
    dist: A `tfd.Distribution` instance.
    params_event_ndims: A `dict` of `str->int` indicating the number of
      dimensions of a given parameter required to parameterize a single event.
    params_overrides: A `dict` of parameter overrides. (e.g. from
      `Distribution.copy`).
    slices: A `slice` or `int` or `int` `Tensor` or `tf.newaxis` or `tuple`
      thereof. (e.g. the argument of a `__getitem__` method).

  Returns:
    new_dist: A batch-sliced `tfd.Distribution`.
  """
  if not isinstance(slices, collections.Sequence):
    slices = (slices,)
  # We track the history of slice and copy(**param_overrides) in order to trace
  # back to the original distribution's source variables.
  orig_dist, slice_overrides_seq = getattr(dist, PROVENANCE_ATTR, (dist, []))
  slice_overrides_seq += [(slices, params_overrides)]
  # Re-doing the full sequence of slice+copy override work here enables
  # gradients all the way back to the original distribution's arguments.
  dist = _apply_slice_sequence(orig_dist, params_event_ndims,
                               slice_overrides_seq)
  setattr(dist, PROVENANCE_ATTR, (orig_dist, slice_overrides_seq))
  return dist
