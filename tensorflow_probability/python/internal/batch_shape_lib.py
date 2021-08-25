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
from tensorflow_probability.python.internal import tensor_util
from tensorflow_probability.python.internal import tensorshape_util

from tensorflow.python.util import nest  # pylint: disable=g-direct-tensorflow-import

__all__ = [
    'inferred_batch_shape',
    'inferred_batch_shape_tensor',
]


def inferred_batch_shape(batch_object, bijector_x_event_ndims=None):
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
      `batched_object.parameters` of the parameters passed to its constructor.
    bijector_x_event_ndims: If `batch_object` is a bijector, this is the
      (structure of) integer(s) value of `x_event_ndims` in the current context
      (for example, as passed to `experimental_batch_shape`). Otherwise, this
      argument should be `None`.
      Default value: `None`.

  Returns:
    batch_shape: `tf.TensorShape` broadcast batch shape of all parameters; may
        be partially defined or unknown.
  """
  batch_shapes = map_fn_over_parameters_with_event_ndims(
      batch_object,
      _get_batch_shape_part,
      require_static=True,
      bijector_x_event_ndims=bijector_x_event_ndims)
  return functools.reduce(tf.broadcast_static_shape,
                          tf.nest.flatten(batch_shapes),
                          tf.TensorShape([]))


def inferred_batch_shape_tensor(batch_object,
                                bijector_x_event_ndims=None,
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
      `batched_object.parameters` of the parameters passed to its constructor.
    bijector_x_event_ndims: If `batch_object` is a bijector, this is the
      (structure of) integer(s) value of `x_event_ndims` in the current context
      (for example, as passed to `experimental_batch_shape`). Otherwise, this
      argument should be `None`.
      Default value: `None`.
    **parameter_kwargs: Optional keyword arguments overriding parameter values
      in `batch_object.parameters`. Typically this is used to avoid multiple
      Tensor conversions of the same value.

  Returns:
    batch_shape_tensor: `Tensor` broadcast batch shape of all parameters.
  """
  batch_shapes = map_fn_over_parameters_with_event_ndims(
      batch_object,
      _get_batch_shape_tensor_part,
      bijector_x_event_ndims=bijector_x_event_ndims,
      require_static=False,
      **parameter_kwargs)
  return functools.reduce(ps.broadcast_shape, tf.nest.flatten(batch_shapes), [])


def _get_batch_shape_tensor_part(x, event_ndims):
  """Extracts an object's runtime (Tensor) shape for batch shape inference."""
  if hasattr(x, 'experimental_batch_shape_tensor'):  # `x` is a bijector
    try:
      return x.experimental_batch_shape_tensor(x_event_ndims=event_ndims)
    except NotImplementedError:
      # Backwards compatibility with bijector-like instances that don't
      # inherit from tfb.Bijector (e.g., Distrax bijectors) and/or don't
      # implement `_parameter_properties`.
      return []
  if hasattr(x, 'batch_shape_tensor'):  # `x` is a distribution or linop.
    base_shape = x.batch_shape_tensor()
    if tf.nest.is_nested(base_shape):
      # Attempt to collapse non-autobatched JDs to a coherent batch shape.
      base_shape = functools.reduce(ps.broadcast_shape,
                                    tf.nest.flatten(base_shape))
  elif hasattr(x, 'shape') and tensorshape_util.is_fully_defined(x.shape):
    base_shape = x.shape
  else:
    base_shape = tf.shape(x)
  return _truncate_shape_tensor(base_shape, event_ndims)


def _get_batch_shape_part(x, event_ndims):
  """Extracts an object's shape for batch shape inference."""
  if hasattr(x, 'experimental_batch_shape'):  # `x` is a bijector
    if event_ndims is None:
      # If `event_ndims` is None, then we don't know the rank of the event for
      # inputs to `x`, so we cannot determine the batch shape of bijector `x`.
      return tf.TensorShape(None)
    try:
      return x.experimental_batch_shape(x_event_ndims=event_ndims)
    except NotImplementedError:
      # Backwards compatibility with bijector-like instances that don't
      # inherit from tfb.Bijector (e.g., Distrax bijectors) and/or don't
      # implement `_parameter_properties`.
      return tf.TensorShape([])
  if hasattr(x, 'batch_shape'):  # `x` is a distribution or linear operator.
    base_shape = x.batch_shape
    # Hack to attempt to collapse non-autobatched JDs to a coherent batch shape.
    # Distrax distributions have tuple-valued batch shapes, which are
    # always spuriously 'nested', so to avoid confusion we check that the
    # distribution is actually a joint distribution (Distrax
    # distributions are always single-part).
    if tf.nest.is_nested(x.dtype):
      base_shape = functools.reduce(tf.broadcast_static_shape,
                                    tf.nest.flatten(base_shape))
    base_shape = tf.TensorShape(base_shape)
  elif hasattr(x, 'shape'):  # `x` is a Tensor or ndarray.
    base_shape = tf.TensorShape(x.shape)
    # `x` is a Python list, tuple, or literal.
  else:
    base_shape = tf.TensorShape(np.array(x).shape)
  return _truncate_shape(base_shape, event_ndims)


def _truncate_shape_tensor(shape, ndims_to_truncate):
  shape = ps.convert_to_shape_tensor(shape, dtype_hint=np.int32)
  ndims_to_truncate = ps.convert_to_shape_tensor(
      ndims_to_truncate, dtype_hint=np.int32)
  base_rank = ps.rank_from_shape(shape)
  return shape[:(
      base_rank -
      # Don't try to slice away more ndims than the parameter
      # actually has, if that's fewer than `event_ndims` (i.e.,
      # if it relies on broadcasting).
      ps.minimum(ndims_to_truncate, base_rank))]


def _truncate_shape(shape, ndims_to_truncate):
  if tensorshape_util.rank(shape) is None or ndims_to_truncate is None:
    return tf.TensorShape(None)
  if tf.is_tensor(ndims_to_truncate):
    event_ndims = tf.get_static_value(ndims_to_truncate)
    if event_ndims is None:
      return tf.TensorShape(None)
  return shape[:(len(shape) -
                 # Don't try to slice away more ndims than the parameter
                 # actually has, if that's fewer than `event_ndims` (i.e.,
                 # if it relies on broadcasting).
                 min(ndims_to_truncate, len(shape)))]


def batch_shape_parts(batch_object,
                      bijector_x_event_ndims=None,
                      **parameter_kwargs):
  """Returns a dict mapping parameter names to their inferred batch shapes."""
  return map_fn_over_parameters_with_event_ndims(
      batch_object,
      _get_batch_shape_tensor_part,
      bijector_x_event_ndims=bijector_x_event_ndims,
      **parameter_kwargs)


def broadcast_parameters_with_batch_shape(batch_object,
                                          batch_shape,
                                          bijector_x_event_ndims=None):
  """Broadcasts each parameter's batch shape with the given `batch_shape`.

  This returns a dict of parameters to `batch_object` broadcast with the given
  batch shape. It can be understood as a pseudo-inverse operation to batch
  slicing:

  ```python
  dist = tfd.Normal(0., 1.)
  # ==> `dist.batch_shape == []`
  broadcast_dist = dist._broadcast_parameters_with_batch_shape([3])
  # ==> `broadcast_dist.batch_shape == [3]`
  #     `broadcast_dist.loc.shape == [3]`
  #     `broadcast_dist.scale.shape == [3]`
  sliced_dist = broadcast_dist[0]
  # ==> `sliced_dist.batch_shape == []`.
  ```

  Args:
    batch_object: Python object, typically a `tfd.Distribution` or
      `tfb.Bijector`. This must implement the method
      `batched_object.parameter_properties()` and expose a dict
      `batched_object.parameters` of the parameters passed to its constructor.
    batch_shape: Integer `Tensor` batch shape.
    bijector_x_event_ndims: If `batch_object` is a bijector, this is the
      (structure of) integer(s) value of `x_event_ndims` in the current context
      (for example, as passed to `experimental_batch_shape`). Otherwise, this
      argument should be `None`.
      Default value: `None`.

  Returns:
    updated_parameters: Python `dict` mapping names of parameters from
      `batch_object.parameter_properties()` to broadcast values.
  """
  return map_fn_over_parameters_with_event_ndims(
      batch_object,
      functools.partial(
          _broadcast_parameter_with_batch_shape, batch_shape=batch_shape),
      bijector_x_event_ndims=bijector_x_event_ndims)


def _broadcast_parameter_with_batch_shape(param,
                                          param_event_ndims,
                                          batch_shape):
  """Broadcasts `param` with the given batch shape, recursively."""
  if hasattr(param, 'forward_min_event_ndims'):
    # Bijector-valued params are responsible for handling any structure in
    # their event ndims.
    return param._broadcast_parameters_with_batch_shape(  # pylint: disable=protected-access
        batch_shape, x_event_ndims=param_event_ndims)

  # Otherwise, param_event_ndims is a single integer, corresponding to the
  # number of batch dimensions of the parameter that were
  # treated as event dimensions by the outer context.
  base_shape = ps.concat([batch_shape,
                          ps.ones([param_event_ndims], dtype=np.int32)],
                         axis=0)
  if hasattr(param, '_broadcast_parameters_with_batch_shape'):
    return param._broadcast_parameters_with_batch_shape(base_shape)  # pylint: disable=protected-access
  elif hasattr(param, 'matmul'):
    # TODO(davmre): support broadcasting LinearOperator parameters.
    return param
  return tf.broadcast_to(param, ps.broadcast_shape(base_shape, ps.shape(param)))


def map_fn_over_parameters_with_event_ndims(batch_object,
                                            fn,
                                            bijector_x_event_ndims=None,
                                            require_static=False,
                                            **parameter_kwargs):
  """Maps `fn` over an object's parameters and corresponding param event_ndims.

  Args:
    batch_object: Python object, typically a `tfd.Distribution` or
      `tfb.Bijector`. This must implement the method
      `batched_object.parameter_properties()` and expose a dict
      `batched_object.parameters` of the parameters passed to its constructor.
    fn: Python `callable` with signature `result = fn(param, param_event_ndims)`
      to be applied to the parameters of `batch_object`.
    bijector_x_event_ndims: If `batch_object` is a bijector, this is the
      (structure of) integer(s) value of `x_event_ndims` in the current context
      (for example, as passed to `experimental_batch_shape`). Otherwise, this
      argument should be `None`.
      Default value: `None`.
    require_static: Python `bool`, whether to use only statically available
      `event_ndims` information. If `True`, this function will perform
      no Tensor operations (other than any performed by `fn` itself), but may
      invoke `fn` with `param_event_ndims=None`.
      Default value: `False`.
    **parameter_kwargs: Optional keyword arguments overriding the parameter
      values in `batch_object.parameters`. Typically this is used to avoid
      multiple Tensor conversions of the same value.

  Returns:
    results: Dictionary with parameter names (Python `str` values) as
      keys, containing results returned by `fn`. Parameters whose value is
      `None` or that have undefined `event_ndims` will be omitted.
  """
  results = {}
  parameter_properties = type(batch_object).parameter_properties()

  for kwarg in parameter_kwargs:
    if kwarg not in parameter_properties:
      logging.warning(
          'Keyword argument `%s` was not expected and will be ignored. '
          'Expected any of the parameters: %s. Any of '
          'these not passed as arguments will be read from '
          '`self.parameters`, resulting in redundant Tensor '
          'conversions if they were not already specified as Tensors.', kwarg,
          parameter_properties.keys())

  for param_name, param in dict(batch_object.parameters,
                                **parameter_kwargs).items():
    if param is None:
      continue
    if param_name not in parameter_properties:
      continue

    # Ndims of base shape used for a *minimal* event.
    properties = parameter_properties[param_name]
    if properties.event_ndims is None:
      continue
    if bijector_x_event_ndims is not None:
      param_event_ndims = properties.bijector_instance_event_ndims(
          batch_object,
          x_event_ndims=bijector_x_event_ndims,
          require_static=require_static)
    else:
      param_event_ndims = properties.instance_event_ndims(
          batch_object,
          require_static=require_static)

    if param_name not in parameter_kwargs:
      # Values from `batch_object.parameters` have not been converted to
      # Tensor, so may be lists or other literals with unexpected structure.
      # First, try to get the param's public attribute, since this will
      # typically have been converted to Tensor if applicable.
      if hasattr(batch_object, param_name):
        param = getattr(batch_object, param_name)
      elif (properties.is_tensor
            and not tf.is_tensor(param)
            and not tf.nest.is_nested(param_event_ndims)):
        # As a last resort, try an explicit conversion.
        param = tensor_util.convert_nonref_to_tensor(param, name=param_name)

    results[param_name] = nest.map_structure_up_to(
        param, fn, param, param_event_ndims)
  return results
