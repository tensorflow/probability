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
"""Slicing utility for TFP objects with batch shape (tfd.Distribution, etc)."""

import collections
import functools


import tensorflow.compat.v2 as tf
from tensorflow_probability.python.internal import batch_shape_lib
from tensorflow_probability.python.internal import prefer_static as ps


__all__ = ['batch_slice']


# We track the provenance of a sliced or copied object all the way back to
# the arguments initially provided to the first constructor.
# This allows us to ensure that sub-sliced and copied objects retain the
# gradient back to any source variables provided up-front. e.g. we want the
# following to work:
#    v = tf.compat.v2.Variable(tf.random.uniform([]))
#    dist = tfd.Normal(v, 1)
#    with tf.GradientTape() as tape:
#      lp = dist[...].log_prob(0.)
#    dlpdv = tape.gradient(lp, v)
# dlpdv should not be None.
PROVENANCE_ATTR = '_tfp_batch_slice_provenance'


def _sanitize_slices(slices, intended_shape, deficient_shape):
  """Restricts slices to avoid overflowing size-1 (broadcast) dimensions.

  Args:
    slices: iterable of slices received by `__getitem__`.
    intended_shape: int `Tensor` shape for which the slices were intended.
    deficient_shape: int `Tensor` shape to which the slices will be applied.
      Must have the same rank as `intended_shape`.
  Returns:
    sanitized_slices: Python `list` of
  """
  sanitized_slices = []
  idx = 0
  for slc in slices:
    if slc is Ellipsis:  # Switch over to negative indexing.
      if idx < 0:
        raise ValueError('Found multiple `...` in slices {}'.format(slices))
      num_remaining_non_newaxis_slices = sum(
          s is not tf.newaxis for s in slices[slices.index(Ellipsis) + 1:])
      idx = -num_remaining_non_newaxis_slices
    elif slc is tf.newaxis:
      pass
    else:
      is_broadcast = intended_shape[idx] > deficient_shape[idx]
      if isinstance(slc, slice):
        # Slices are denoted by start:stop:step.
        start, stop, step = slc.start, slc.stop, slc.step
        if start is not None:
          start = ps.where(is_broadcast, 0, start)
        if stop is not None:
          stop = ps.where(is_broadcast, 1, stop)
        if step is not None:
          step = ps.where(is_broadcast, 1, step)
        slc = slice(start, stop, step)
      else:  # int, or int Tensor, e.g. d[d.batch_shape_tensor()[0] // 2]
        slc = ps.where(is_broadcast, 0, slc)
      idx += 1
    sanitized_slices.append(slc)
  return sanitized_slices


def _slice_single_param(param, param_event_ndims, slices, batch_shape):
  """Slices into the batch shape of a single parameter.

  Args:
    param: The original parameter to slice; either a `Tensor` or an object
      with batch shape (Distribution, Bijector, etc).
    param_event_ndims: `int` event rank of this parameter. For non-Tensor
      parameters, this is the number of this param's batch dimensions used by
      an event of the parent object.
    slices: iterable of slices received by `__getitem__`.
    batch_shape: The parameterized object's batch shape `Tensor`.

  Returns:
    new_param: Instance of the same type as `param`, batch-sliced according to
      `slices`.
  """
  # Broadcast the parmameter to have full batch rank.
  param = batch_shape_lib.broadcast_parameter_with_batch_shape(
      param, param_event_ndims, ps.ones_like(batch_shape))
  param_batch_shape = batch_shape_lib.get_batch_shape_tensor_part(
      param, param_event_ndims)
  # At this point the param should have full batch rank, *unless* it's an
  # atomic object like `tfb.Identity()` incapable of having any batch rank.
  if (tf.get_static_value(ps.rank_from_shape(batch_shape)) != 0 and
      tf.get_static_value(ps.rank_from_shape(param_batch_shape)) == 0):
    return param
  param_slices = _sanitize_slices(slices,
                                  intended_shape=batch_shape,
                                  deficient_shape=param_batch_shape)

  # Bijectors (which have no fixed batch shape) handle `param_event_ndims` in
  # the recursive call.
  if hasattr(param, 'forward_min_event_ndims'):
    return batch_slice(param,
                       params_overrides={},
                       slices=tuple(param_slices),
                       bijector_x_event_ndims=param_event_ndims)

  # Otherwise, extend `param_slices` (which represents slicing into the
  # parameter's batch shape) with the parameter's event ndims. For example, if
  # `params_event_ndims == 1`, then `[i, ..., j]` would become `[i, ..., j, :]`.
  if param_event_ndims > 0:
    if Ellipsis not in [slc for slc in slices if not tf.is_tensor(slc)]:
      param_slices.append(Ellipsis)
    param_slices += [slice(None)] * param_event_ndims
  return param.__getitem__(tuple(param_slices))


def _slice_params_to_dict(batch_object, slices, bijector_x_event_ndims=None):
  """Computes the override dictionary of sliced parameters.

  Args:
    batch_object: The tfd.Distribution being batch-sliced.
    slices: Slices as received by __getitem__.
    bijector_x_event_ndims: If `batch_object` is a bijector, this is the
      (structure of) integer(s) value of `x_event_ndims` in the current context
      (for example, as passed to `experimental_batch_shape`). Otherwise, this
      argument should be `None`.
      Default value: `None`.

  Returns:
    overrides: `str->Tensor` `dict` of batch-sliced parameter overrides.
  """

  if bijector_x_event_ndims is None:
    batch_shape = batch_object.batch_shape_tensor()
  else:
    batch_shape = batch_object.experimental_batch_shape_tensor(
        x_event_ndims=bijector_x_event_ndims)
  return batch_shape_lib.map_fn_over_parameters_with_event_ndims(
      batch_object,
      functools.partial(_slice_single_param,
                        slices=slices,
                        batch_shape=batch_shape),
      bijector_x_event_ndims=bijector_x_event_ndims)


def _apply_single_step(
    batch_object, slices, params_overrides, bijector_x_event_ndims=None):
  """Applies a single slicing step to `batch_object`, returning a new instance."""
  if len(slices) == 1 and slices[0] is Ellipsis:
    # The path used by Distribution.copy: batch_slice(...args..., Ellipsis)
    override_dict = {}
  else:
    override_dict = _slice_params_to_dict(
        batch_object, slices, bijector_x_event_ndims=bijector_x_event_ndims)
  override_dict.update(params_overrides)
  parameters = dict(batch_object.parameters, **override_dict)
  return type(batch_object)(**parameters)


def _apply_slice_sequence(
    batch_object, slice_overrides_seq, bijector_x_event_ndims=None):
  """Applies a sequence of slice or copy-with-overrides operations to `batch_object`."""
  for slices, overrides in slice_overrides_seq:
    batch_object = _apply_single_step(
        batch_object,
        slices,
        overrides,
        bijector_x_event_ndims=bijector_x_event_ndims)
  return batch_object


def batch_slice(batch_object,
                params_overrides,
                slices,
                bijector_x_event_ndims=None):
  """Slices `batch_object` along its batch dimensions.

  Args:
    batch_object: A `tfd.Distribution` instance.
    params_overrides: A `dict` of parameter overrides. (e.g. from
      `Distribution.copy`).
    slices: A `slice` or `int` or `int` `Tensor` or `tf.newaxis` or `tuple`
      thereof. (e.g. the argument of a `__getitem__` method).
    bijector_x_event_ndims: If `batch_object` is a bijector, this is the
      (structure of) integer(s) value of `x_event_ndims` in the current context
      (for example, as passed to `experimental_batch_shape`). Otherwise, this
      argument should be `None`.
      Default value: `None`.

  Returns:
    new_batch_object: A batch-sliced `tfd.Distribution`.
  """
  if not isinstance(slices, collections.abc.Sequence):
    slices = (slices,)
  # We track the history of slice and copy(**param_overrides) in order to trace
  # back to the original object's source variables.
  #
  # NOTE: We must not modify `slice_overrides_seq`, as it could be an attribute
  # of `batch_object and this method should not modify `batch_object`.
  orig_batch_object, slice_overrides_seq = getattr(
      batch_object, PROVENANCE_ATTR, (batch_object, []))
  slice_overrides_seq = slice_overrides_seq + [(slices, params_overrides)]
  # Re-doing the full sequence of slice+copy override work here enables
  # gradients all the way back to the original batch_objectribution's arguments.
  batch_object = _apply_slice_sequence(
      orig_batch_object,
      slice_overrides_seq,
      bijector_x_event_ndims=bijector_x_event_ndims)
  setattr(batch_object,
          PROVENANCE_ATTR,
          batch_object._no_dependency((orig_batch_object, slice_overrides_seq)))  # pylint: disable=protected-access
  return batch_object
