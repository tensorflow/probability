# Copyright 2021 The TensorFlow Probability Authors.
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
"""Inference of distribution and bijector batch shapes from their parameters."""

import functools

from absl import logging
import numpy as np
import tensorflow.compat.v2 as tf

from tensorflow_probability.python.internal import prefer_static as ps
from tensorflow_probability.python.internal import tensorshape_util

from tensorflow.python.util import nest  # pylint: disable=g-direct-tensorflow-import


__all__ = [
    'inferred_batch_shape',
    'inferred_batch_shape_tensor',
]


def inferred_batch_shape(batch_object, additional_event_ndims=0):
  """Infers an object's batch shape from its  parameters.

  Each parameter contributes a batch shape of
  `base_shape(parameter)[:-event_ndims(parameter)]`, where a parameter's
  `base_shape` is its batch shape if it defines one (e.g., if it is a
  Distribution, LinearOperator, etc.), and its Tensor shape otherwise,
  and `event_ndims` is as annotated by
  `batch_object.parameter_properties()[parameter_name].event_ndims`.
  Parameters with structured batch shape
  (in particular, non-autobatched JointDistributions) are not currently
  supported.

  Args:
    batch_object: Python object, typically a `tfd.Distribution` or
      `tfb.Bijector`. This must implement the method
      `batched_object.parameter_properties()` and expose a dict
      `batched_object.parameters` of the parameters passed to its
      constructor.
    additional_event_ndims: Optional integer value to add to the
      annotated `event_ndims` property for all parameters. For Bijectors, this
      is the unique difference between the user-provided `event_ndims` and
      the bijector's inherent `min_event_ndims` (on which the parameter
      annotations are based).
      Default value: `0`.
  Returns:
    batch_shape: `tf.TensorShape` broadcast batch shape of all parameters; may
        be partially defined or unknown.
  """
  batch_shapes = batch_shape_parts(
      batch_object,
      additional_event_ndims=additional_event_ndims,
      get_base_shape_fn=get_base_shape,
      slice_batch_shape_fn=slice_batch_shape)
  return functools.reduce(tf.broadcast_static_shape,
                          tf.nest.flatten(batch_shapes),
                          tf.TensorShape([]))


def inferred_batch_shape_tensor(batch_object,
                                additional_event_ndims=0,
                                **parameter_kwargs):
  """Infers an object's batch shape from its  parameters.

  Each parameter contributes a batch shape of
  `base_shape(parameter)[:-event_ndims(parameter)]`, where a parameter's
  `base_shape` is its batch shape if it defines one (e.g., if it is a
  Distribution, LinearOperator, etc.), and its Tensor shape otherwise,
  and `event_ndims` is as annotated by
  `batch_object.parameter_properties()[parameter_name].event_ndims`.
  Parameters with structured batch shape
  (in particular, non-autobatched JointDistributions) are not currently
  supported.

  Args:
    batch_object: Python object, typically a `tfd.Distribution` or
      `tfb.Bijector`. This must implement the method
      `batched_object.parameter_properties()` and expose a dict
      `batched_object.parameters` of the parameters passed to its
      constructor.
    additional_event_ndims: Optional integer value to add to the
      annotated `event_ndims` property for all parameters. For Bijectors, this
      is the unique difference between the user-provided `event_ndims` and
      the bijector's inherent `min_event_ndims` (on which the parameter
      annotations are based).
      Default value: `0`.
    **parameter_kwargs: Optional keyword arguments overriding parameter
      values in `batch_object.parameters`. Typically this is used to avoid
      multiple Tensor conversions of the same value.
  Returns:
    batch_shape_tensor: `Tensor` broadcast batch shape of all parameters.
  """
  return functools.reduce(
      ps.broadcast_shape,
      tf.nest.flatten(
          batch_shape_parts(
              batch_object,
              additional_event_ndims=additional_event_ndims,
              get_base_shape_fn=get_base_shape_tensor,
              slice_batch_shape_fn=slice_batch_shape_tensor,
              **parameter_kwargs)),
      [])


def get_base_shape_tensor(x):
  """Extracts an object's runtime (Tensor) shape for batch shape inference."""
  if hasattr(x, 'batch_shape_tensor'):  # `x` is a distribution or linop.
    return x.batch_shape_tensor()
  elif hasattr(x, 'shape') and tensorshape_util.is_fully_defined(x.shape):
    return x.shape
  return tf.shape(x)


def get_base_shape(x):
  """Extracts an object's shape for batch shape inference."""
  if hasattr(x, 'batch_shape'):  # `x` is a distribution or linear operator.
    return tf.TensorShape(x.batch_shape)
  elif hasattr(x, 'shape'):  # `x` is a Tensor or ndarray.
    return tf.TensorShape(x.shape)
  # `x` is a Python list, tuple, or literal.
  return tf.TensorShape(np.array(x).shape)


def slice_batch_shape_tensor(base_shape, event_ndims):
  base_shape = ps.convert_to_shape_tensor(base_shape, dtype_hint=np.int32)
  event_ndims = ps.convert_to_shape_tensor(event_ndims, dtype_hint=np.int32)
  base_rank = ps.rank_from_shape(base_shape)
  return base_shape[:(base_rank -
                      # Don't try to slice away more ndims than the parameter
                      # actually has, if that's fewer than `event_ndims` (i.e.,
                      # if it relies on broadcasting).
                      ps.minimum(event_ndims, base_rank))]


def slice_batch_shape(base_shape, event_ndims):
  if tensorshape_util.rank(base_shape) is None:
    return tf.TensorShape(None)
  if tf.is_tensor(event_ndims):
    event_ndims = tf.get_static_value(event_ndims)
    if event_ndims is None:
      return tf.TensorShape(None)
  return base_shape[:(len(base_shape) -
                      # Don't try to slice away more ndims than the parameter
                      # actually has, if that's fewer than `event_ndims` (i.e.,
                      # if it relies on broadcasting).
                      min(event_ndims, len(base_shape)))]


def batch_shape_parts(batch_object,
                      additional_event_ndims=0,
                      get_base_shape_fn=get_base_shape,
                      slice_batch_shape_fn=slice_batch_shape,
                      **parameter_kwargs):
  """Returns a dict mapping parameter names to their inferred batch shapes.

  An object's batch shape is (with rare exceptions) derived by broadcasting the
  batch shapes contributed by its parameters:

  ```
  batch_shape = functools.reduce(ps.broadcast_shape,
                                 batch_shape_parts(self).values())
  ```

  Each parameter contributes a batch shape part
  `base_shape(parameter)[:-event_ndims(parameter)]`, where a parameter's
  `base_shape` is its batch shape if it defines one (e.g., if it is a
  Distribution, LinearOperator, etc.), and its Tensor shape otherwise,
  and `event_ndims` is as annotated by `batch_object.parameter_properties()`.

  Args:
    batch_object: Python object, typically a `tfd.Distribution` or
      `tfb.Bijector`. This must implement the method
      `batched_object.parameter_properties()` and expose a dict
      `batched_object.parameters` of the parameters passed to its
      constructor.
    additional_event_ndims: Optional integer value to add to the
      annotated `event_ndims` property for all parameters. For Bijectors, this
      is the unique difference between the user-provided `event_ndims` and
      the bijector's inherent `min_event_ndims` (on which the parameter
      annotations are based).
      Default value: `0`.
    get_base_shape_fn: Optional `callable` taking a parameter value (which
      may be a `Tensor`, or an instance of `tfd.Distribution`, `tfb.Bijector`,
      etc.) and returning its batch shape, if it has one, and otherwise its
      Tensor shape. Expected to be one of `get_base_shape` (which returns static
      shapes, which may be partially defined) or `get_base_shape_tensor`
      (which returns Tensor shapes).
      Default value: `get_base_shape`.
    slice_batch_shape_fn: Optional `callable` with signature
      `batch_shape = slice_batch_shape_fn(base_shape, event_ndims)`.
      Expected to be one of `slice_batch_shape`, which slices static shapes,
      or `slice_batch_shape_tensor`, which slices Tensor shapes. Must be
      compatible with base shapes returned by the provided `get_base_shape_fn`.
      Default value: `slice_batch_shape`.
    **parameter_kwargs: Optional keyword arguments overriding the parameter
        values in `batch_object.parameters`. Typically this is used to avoid
        multiple Tensor conversions of the same value.
  Returns:
    batch_shape_parts: Dictionary with parameter names (Python `str` values) as
      keys, with values corresponding to the batch shape contributed to
      `batch_object` by each parameter's value, as derived from property
      annotations. Parameters that do not contribute batch shape may be omitted.
      The shapes will be of the type returned by `get_base_shape_fn` and
      `slice_batch_shape_fn`.
  """
  batch_shapes = {}
  parameter_properties = type(batch_object).parameter_properties()

  for kwarg in parameter_kwargs:
    if kwarg not in parameter_properties:
      logging.warning(
          '`batch_shape_parts` received unrecognized keyword argument `%s`, '
          'which will be ignored. Expected any of the parameters: %s. Any of '
          'these not passed as arguments will be read from '
          '`self.parameters`, resulting in redundant Tensor '
          'conversions if they were not already specified as Tensors.',
          kwarg, parameter_properties.keys())

  for param_name, param in dict(batch_object.parameters,
                                **parameter_kwargs).items():
    if param is None:
      continue
    if param_name not in parameter_properties:
      continue

    # Ndims of base shape used for a *minimal* event.
    properties = parameter_properties[param_name]
    event_ndims = properties.instance_event_ndims(batch_object)
    if event_ndims is None:
      continue

    batch_shapes[param_name] = nest.map_structure_up_to(
        event_ndims,
        lambda p, nd: slice_batch_shape_fn(  # pylint: disable=g-long-lambda
            base_shape=get_base_shape_fn(p),
            event_ndims=nd + additional_event_ndims),
        param,
        event_ndims)

  return batch_shapes
