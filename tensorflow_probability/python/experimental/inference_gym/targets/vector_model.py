# Lint as: python2, python3
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

import tensorflow.compat.v2 as tf

from tensorflow_probability.python import bijectors as tfb
from tensorflow_probability.python.experimental.inference_gym.targets import model as model_lib
from tensorflow_probability.python.internal import prefer_static as ps

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
  transformations which reflect the transformation above. Note that the sample
  transformations will still return structured values, as those generally cannot
  be as readily flattened.

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
  #         'student_ability': tf.float32,
  #         'question_difficulty': tf.float32,
  #     }

  vec_model.dtype
  # ==> tf.float32

  base_model.event_shape
  # ==> {
  #         'mean_student_ability': [],
  #         'student_ability': [400],
  #         'question_difficulty': [100],
  #     }

  vec_model.event_shape
  # ==> [501]
  ```

  """

  def __init__(self, model):
    """Constructs the adapter.

    Args:
      model: An Inference Gym model.

    Raises:
      TypeError: If `model` has more than one unique Tensor dtype.
    """
    self._model = model
    dtypes = set(
        tf.nest.flatten(tf.nest.map_structure(tf.as_dtype, self._model.dtype)))
    if len(dtypes) > 1:
      raise TypeError('Model must have only one Tensor dtype, saw: {}'.format(
          self._model.dtype))
    dtype = dtypes.pop()

    # TODO(siege): Make this work with multi-part default_event_bijector.
    def _make_reshaped_bijector(b, s):
      return tfb.Chain([
          tfb.Reshape(event_shape_in=s, event_shape_out=[ps.reduce_prod(s)]),
          b,
          tfb.Reshape(event_shape_out=b.inverse_event_shape(s)),
      ])

    reshaped_bijector = tf.nest.map_structure(
        _make_reshaped_bijector, self._model.default_event_space_bijector,
        self._model.event_shape)

    bijector = tfb.Blockwise(
        bijectors=tf.nest.flatten(reshaped_bijector),
        block_sizes=tf.nest.flatten(
            tf.nest.map_structure(
                lambda b, s: ps.reduce_prod(b.inverse_event_shape(s)),  # pylint: disable=g-long-lambda
                self._model.default_event_space_bijector,
                self._model.event_shape)))

    event_sizes = tf.nest.map_structure(
        lambda b, s: ps.reduce_prod(b.inverse_event_shape(s)),
        self._model.default_event_space_bijector, self._model.event_shape)
    event_shape = tf.TensorShape([sum(tf.nest.flatten(event_sizes))])

    sample_transformations = collections.OrderedDict()

    def make_flattened_transform(transform):
      # We yank this out to avoid capturing the loop variable.
      return transform._replace(
          fn=lambda x: transform(self._split_and_reshape_event(x)))

    for key, transform in self._model.sample_transformations.items():
      sample_transformations[key] = make_flattened_transform(transform)

    super(VectorModel, self).__init__(
        default_event_space_bijector=bijector,
        event_shape=event_shape,
        dtype=dtype,
        name='vector_' + self._model.name,
        pretty_name=str(self._model),
        sample_transformations=sample_transformations,
    )

  def _unnormalized_log_prob(self, value):
    return self._model.unnormalized_log_prob(
        self._split_and_reshape_event(value))

  def _flatten_and_concat_event(self, x):

    def _reshape_part(part, event_shape):
      part = tf.cast(part, self.dtype)
      new_shape = ps.concat(
          [
              ps.shape(part)[:ps.size(ps.shape(part)) - ps.size(event_shape)],
              [-1]
          ],
          axis=-1,
      )
      return tf.reshape(part, ps.cast(new_shape, tf.int32))

    x = tf.nest.map_structure(_reshape_part, x, self._model.event_shape)
    return tf.concat(tf.nest.flatten(x), axis=-1)

  def _split_and_reshape_event(self, x):
    splits = [
        ps.maximum(1, ps.reduce_prod(s))
        for s in tf.nest.flatten(self._model.event_shape)
    ]
    x = tf.nest.pack_sequence_as(self._model.event_shape,
                                 tf.split(x, splits, axis=-1))

    def _reshape_part(part, dtype, event_shape):
      part = tf.cast(part, dtype)
      new_shape = ps.concat([ps.shape(part)[:-1], event_shape], axis=-1)
      return tf.reshape(part, ps.cast(new_shape, tf.int32))

    x = tf.nest.map_structure(_reshape_part, x, self._model.dtype,
                              self._model.event_shape)
    return x
