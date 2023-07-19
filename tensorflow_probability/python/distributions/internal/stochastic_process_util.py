# Copyright 2023 The TensorFlow Probability Authors.
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

"""Internal helper libraries for stochastic processes."""

import functools

import tensorflow.compat.v2 as tf
from tensorflow_probability.python.internal import distribution_util
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import prefer_static as ps
from tensorflow_probability.python.internal import tensorshape_util
from tensorflow_probability.python.math.psd_kernels.internal import util as psd_kernels_util
from tensorflow.python.util import nest  # pylint: disable=g-direct-tensorflow-import


def is_empty_observation_data(
    feature_ndims, observation_index_points, observations):
  """Returns `True` if given observation data is empty.

  "Empty" means either
    1. Both `observation_index_points` and `observations` are `None`, or
    2. the "number of observations" shape is 0. The shape of
    `observation_index_points` (or each of its components, if nested) is
    `[..., N, f1, ..., fF]`, where `N` is the number of observations and the
    `f`s are feature dims. Thus, we look at the shape element just to the
    left of the leftmost feature dim. If that shape is zero, we consider the
    data empty.

  We don't check the shape of observations; validations are checked elsewhere in
  the calling code, to ensure these shapes are consistent.

  Args:
    feature_ndims: the number of feature dims, as reported by the GP kernel.
    observation_index_points: the observation data locations in the index set.
    observations: the observation data.

  Returns:
    is_empty: True if the data were deemed to be empty.
  """
  # If both input locations and observations are `None`, we consider this
  # "empty" observation data.
  if observation_index_points is None and observations is None:
    return True
  num_obs = tf.nest.map_structure(
      lambda t, nd: tf.compat.dimension_value(t.shape[-(nd + 1)]),
      observation_index_points, feature_ndims)
  if all(n is not None and n == 0 for n in tf.nest.flatten(num_obs)):
    return True
  return False


def validate_observation_data(
    kernel, observation_index_points, observations):
  """Ensure that observation data and locations have consistent shapes.

  This basically means that the batch shapes are broadcastable. We can only
  ensure this when those shapes are fully statically defined.


  Args:
    kernel: The GP kernel.
    observation_index_points: the observation data locations in the index set.
    observations: the observation data.

  Raises:
    ValueError: if the observations' batch shapes are not broadcastable.
  """
  # Check that observation index points and observation counts broadcast.
  ndims = kernel.feature_ndims

  def _validate(t, nd):
    if nd > 0:
      shape = t.shape[:-nd]
    else:
      shape = t.shape
    if (tensorshape_util.is_fully_defined(shape)
        and tensorshape_util.is_fully_defined(observations.shape)):
      index_point_count = shape
      observation_count = observations.shape
      try:
        tf.broadcast_static_shape(index_point_count, observation_count)
      except ValueError:
        # Re-raise with our own more contextual error message.
        raise ValueError(  # pylint:disable=raise-missing-from
            'Observation index point and observation counts are not '
            'broadcastable: {} and {}, respectively.'.format(
                index_point_count, observation_count))

  tf.nest.map_structure(_validate, observation_index_points, ndims)


def check_nested_index_points(kernel, index_points):
  """Ensures that the example dimensions are the same or broadcastable."""
  num_index_points = tf.nest.map_structure(
      lambda x, nd: tf.compat.dimension_value(x.shape[-(nd + 1)]),
      index_points, kernel.feature_ndims)
  flat_num_index_points = tf.nest.flatten(num_index_points)
  static_non_singleton_num_points = set(
      n for n in flat_num_index_points if n is not None and n != 1)
  if len(static_non_singleton_num_points) > 1:
    raise ValueError(
        'Nested components of `index_points` must contain the same or '
        'broadcastable numbers of examples. Saw components with '
        f'{", ".join(list(str(n) for n in static_non_singleton_num_points))} '
        'examples.')


def add_diagonal_shift(matrix, shift):
  broadcast_shape = distribution_util.get_broadcast_shape(
      matrix, shift[..., tf.newaxis])
  matrix = tf.broadcast_to(matrix, broadcast_shape)
  return tf.linalg.set_diag(
      matrix, tf.linalg.diag_part(matrix) + shift, name='add_diagonal_shift')


def event_shape_tensor(kernel, index_points):
  """Get shape of number of index poins."""
  # The examples index is one position to the left of the feature dims.
  example_shape = tf.nest.map_structure(
      lambda t, nd: ps.shape(t)[ps.rank(t) - (nd + 1):ps.rank(t) - nd],
      index_points, kernel.feature_ndims)
  return functools.reduce(ps.broadcast_shape,
                          tf.nest.flatten(example_shape), [])


def event_shape(kernel, index_points):
  """Get shape of number of index poins."""
  check_nested_index_points(kernel, index_points)
  # The examples index is one position to the left of the feature dims.
  example_shape = tf.nest.map_structure(
      lambda t, nd: tf.TensorShape(t.shape[-(nd + 1)]),
      index_points, kernel.feature_ndims)
  flat_shapes = nest.flatten_up_to(kernel.feature_ndims, example_shape)

  if None in [tensorshape_util.rank(s) for s in flat_shapes]:
    return tf.TensorShape([None])
  return functools.reduce(
      tf.broadcast_static_shape, flat_shapes, tf.TensorShape([]))


def multitask_event_shape_tensor(kernel, index_points):
  example_shape = tf.nest.map_structure(
      lambda t, nd: ps.shape(t)[-(nd + 1):-nd],
      index_points, kernel.feature_ndims)
  shape = functools.reduce(ps.broadcast_shape,
                           tf.nest.flatten(example_shape), [])
  return ps.concat([shape, [kernel.num_tasks]], axis=0)


def multitask_event_shape(kernel, index_points):
  check_nested_index_points(kernel, index_points)
  example_shape = tf.nest.map_structure(
      lambda t, nd: tf.TensorShape(t.shape[-(nd + 1):-nd]),
      index_points, kernel.feature_ndims)
  flat_shapes = nest.flatten_up_to(kernel.feature_ndims, example_shape)
  if None in [tensorshape_util.rank(s) for s in flat_shapes]:
    return tf.TensorShape([None, kernel.num_tasks])
  shape = functools.reduce(
      tf.broadcast_static_shape, flat_shapes, tf.TensorShape([]))
  return tensorshape_util.concatenate(shape, [kernel.num_tasks])


def maybe_create_mean_fn(mean_fn, dtype):
  """Create a default mean function if one is not provided."""
  if mean_fn is not None:
    if not callable(mean_fn):
      raise ValueError('`mean_fn` must be a Python callable')
    return mean_fn

  # Default to a constant zero function, borrowing the dtype from
  # index_points to ensure consistency.
  return lambda _: tf.zeros([1], dtype=dtype)


def maybe_create_multitask_mean_fn(mean_fn, kernel, dtype):
  """Create a default mean function if one is not provided."""
  if mean_fn is not None:
    if not callable(mean_fn):
      raise ValueError('`mean_fn` must be a Python callable')
    return mean_fn
  def _mean_fn(x):
    # Shape B1 + [E, N], where E is the number of index points, and N is
    # the number of tasks.
    flat_shapes = tf.nest.flatten(
        tf.nest.map_structure(lambda z, d: ps.shape(z)[:-d],
                              x, kernel.feature_ndims))
    bcast_shape = functools.reduce(ps.broadcast_shape, flat_shapes, [])
    return tf.zeros(ps.concat(
        [bcast_shape, [kernel.num_tasks]], axis=0), dtype=dtype)
  return _mean_fn


def get_loc_and_kernel_matrix(
    kernel,
    index_points,
    observation_noise_variance,
    mean_fn,
    is_missing=None,
    mask_loc=True):
  """Compute location and kernel matrix from inputs, possibly masking them."""
  if is_missing is not None:
    # Mask the index_points to avoid NaN gradients. NOTE: We are assuming the
    # 0. vector is a valid sample. TODO(b/276969724): Mask out missing index
    # points to something in the support of the kernel.
    pad_shapes = lambda nd: psd_kernels_util.pad_shape_with_ones(  # pylint:disable=g-long-lambda
        is_missing, nd, start=-1)
    mask_is_missing = tf.nest.map_structure(pad_shapes, kernel.feature_ndims)

    mask = lambda m, x: tf.where(m, dtype_util.as_numpy_dtype(x.dtype)(0), x)
    index_points = tf.nest.map_structure(mask, mask_is_missing, index_points)

  kernel_matrix = compute_kernel_matrix(
      kernel, index_points, observation_noise_variance)

  loc = mean_fn(index_points)
  if is_missing is not None:
    if mask_loc:
      loc = tf.where(is_missing, 0., loc)
    kernel_matrix = psd_kernels_util.mask_matrix(kernel_matrix, is_missing)
  return loc, kernel_matrix


def compute_kernel_matrix(kernel, index_points, observation_noise_variance):
  kernel_matrix = kernel.matrix(index_points, index_points)
  observation_noise_variance = tf.convert_to_tensor(observation_noise_variance)
  # We are compute K + obs_noise_variance * I. The shape of this matrix is
  # going to be a broadcast of the shapes of K and obs_noise_variance * I.
  broadcast_shape = distribution_util.get_broadcast_shape(
      kernel_matrix,
      # We pad with two single dimension since this represents a batch of
      # scaled identity matrices.
      observation_noise_variance[..., tf.newaxis, tf.newaxis])
  kernel_matrix = tf.broadcast_to(kernel_matrix, broadcast_shape)
  return add_diagonal_shift(
      kernel_matrix, observation_noise_variance[..., tf.newaxis])
