# Lint as: python3
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
"""Implementation of the VectorModel."""

import collections
import functools

import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp

from tensorflow_probability.python.internal import prefer_static as ps
from inference_gym.targets import model as model_lib

tfb = tfp.bijectors

__all__ = [
    'VectorModel',
]


class VectorModel(model_lib.Model):
  """An adapter to convert an existing model to have a vector-valued support.

  This adapter makes it convenient to use the Inference Gym models with
  inference algorithms which cannot handle structured events. It does so by
  reshaping individual event Tensors and concatenating them into a single
  vector.

  The resultant vector-valued model has updated properties and sample
  transformations which reflect the transformation above. By default, the sample
  transformations will still return structured values. This behavior can be
  altered via the `flatten_sample_transformations` argument.

  There are only two restrictions on the models that can be handled by this
  class:

  1. The individual Tensors in an event must all have the same dtype.
  2. The `default_event_space_bijector` must apply to a single tensor at time.

  The second restriction will be lifted soon.

  #### Example

  ```
  base_model = gym.targets.SyntheticItemResponseTheory()
  vec_model = gym.targets.VectorModel(base_model)

  base_model.dtype
  # ==> {
  #         'mean_student_ability': tf.float32,
  #         'centered_student_ability': tf.float32,
  #         'question_difficulty': tf.float32,
  #     }

  vec_model.dtype
  # ==> tf.float32

  base_model.event_shape
  # ==> {
  #         'mean_student_ability': [],
  #         'centered_student_ability': [400],
  #         'question_difficulty': [100],
  #     }

  vec_model.event_shape
  # ==> [501]
  ```

  """

  def __init__(self, model, flatten_sample_transformations=False):
    """Constructs the adapter.

    Args:
      model: An Inference Gym model.
      flatten_sample_transformations: Python bool. Whether to flatten and
        concatenate the outputs of sample transformations.

    Raises:
      TypeError: If `model` has more than one unique Tensor dtype.
      TypeError: If `flatten_sample_transformations` is `True` and there is a
        sample transformation that has more than one unique Tensor dtype.
    """
    self._model = model

    super(VectorModel, self).__init__(
        default_event_space_bijector=_make_vector_event_space_bijector(
            self._model),
        event_shape=_get_vector_event_shape(self._model),
        dtype=_get_unique_dtype(
            self._model.dtype,
            'Model must have only one Tensor dtype, saw: {}'.format),
        name='vector_' + self._model.name,
        pretty_name=str(self._model),
        sample_transformations=_make_vector_sample_transformations(
            self._model, flatten_sample_transformations),
    )

  def _unnormalized_log_prob(self, value):
    return self._model.unnormalized_log_prob(
        _split_and_reshape_event(value, self._model))


def _flatten_and_concat(x, batch_shape, dtype):
  """Flattens and concatenates a structured `x`."""
  # For convenience.
  if x is None:
    return x

  def _reshape_part(part):
    part = tf.cast(part, dtype)
    new_shape = ps.concat(
        [batch_shape, [-1]],
        axis=-1,
    )
    return tf.reshape(part, ps.cast(new_shape, tf.int32))

  x = tf.nest.map_structure(_reshape_part, x)
  return tf.concat(tf.nest.flatten(x), axis=-1)


def _get_unique_dtype(dtype, error_message_fn):
  """Gets unique singleton dtype from a structure."""
  dtypes = set(tf.nest.flatten(tf.nest.map_structure(tf.as_dtype, dtype)))
  if len(dtypes) > 1:
    raise TypeError(error_message_fn(dtype))
  return dtypes.pop()


def _make_vector_event_space_bijector(model):
  """Creates a vector bijector that constrains like the structured model."""

  # TODO(siege): Make this work with multi-part default_event_bijector.
  def _make_reshaped_bijector(b, s):
    return tfb.Chain([
        tfb.Reshape(event_shape_in=s, event_shape_out=[ps.reduce_prod(s)]),
        b,
        tfb.Reshape(
            event_shape_in=[ps.reduce_prod(b.inverse_event_shape(s))],
            event_shape_out=b.inverse_event_shape(s)),
    ])

  reshaped_bijector = tf.nest.map_structure(_make_reshaped_bijector,
                                            model.default_event_space_bijector,
                                            model.event_shape)

  return tfb.Blockwise(
      bijectors=tf.nest.flatten(reshaped_bijector),
      block_sizes=tf.nest.flatten(
          tf.nest.map_structure(
              lambda b, s: ps.reduce_prod(b.inverse_event_shape(s)),  # pylint: disable=g-long-lambda
              model.default_event_space_bijector,
              model.event_shape)))


def _get_vector_event_shape(model):
  """Returns the size of the vector corresponding to the flattened event."""
  event_sizes = tf.nest.map_structure(ps.reduce_prod, model.event_shape)
  return tf.TensorShape([sum(tf.nest.flatten(event_sizes))])


def _split_and_reshape_event(x, model):
  """Splits and reshapes a flat event `x` to match the structure of `model`."""
  splits = [
      ps.maximum(1, ps.reduce_prod(s))
      for s in tf.nest.flatten(model.event_shape)
  ]
  x = tf.nest.pack_sequence_as(model.event_shape, tf.split(x, splits, axis=-1))

  def _reshape_part(part, dtype, event_shape):
    part = tf.cast(part, dtype)
    new_shape = ps.concat([ps.shape(part)[:-1], event_shape], axis=-1)
    return tf.reshape(part, ps.cast(new_shape, tf.int32))

  x = tf.nest.map_structure(_reshape_part, x, model.dtype, model.event_shape)
  return x


def _make_vector_sample_transformations(model, flatten_sample_transformations):
  """Makes `model`'s sample transformations compatible with vector events."""
  sample_transformations = collections.OrderedDict()

  def flattened_transform(x, dtype, transform):
    res = transform(_split_and_reshape_event(x, model))
    if not flatten_sample_transformations:
      return res
    batch_shape = ps.shape(x)[:-1]
    return _flatten_and_concat(res, batch_shape, dtype)

  # We yank this out to avoid capturing the loop variable.
  def make_flattened_transform(transform_name, transform):
    if flatten_sample_transformations:
      dtype = _get_unique_dtype(
          transform.dtype,
          lambda d:  # pylint: disable=g-long-lambda
          'Sample transformation \'{}\' must have only one Tensor dtype, '
          'saw: {}'.format(transform_name, d))
      transform = transform._replace(
          ground_truth_mean=_flatten_and_concat(
              transform.ground_truth_mean, (), dtype),
          ground_truth_standard_deviation=_flatten_and_concat(
              transform.ground_truth_standard_deviation, (), dtype),
          ground_truth_mean_standard_error=_flatten_and_concat(
              transform.ground_truth_mean_standard_error, (), dtype),
          ground_truth_standard_deviation_standard_error=_flatten_and_concat(
              transform.ground_truth_standard_deviation_standard_error, (),
              dtype),
      )
    else:
      dtype = None
    return transform._replace(
        fn=functools.partial(
            flattened_transform, dtype=dtype, transform=transform))

  for key, transform in model.sample_transformations.items():
    sample_transformations[key] = make_flattened_transform(key, transform)
  return sample_transformations
