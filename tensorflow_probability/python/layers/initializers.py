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
"""Keras initializers useful for TFP Keras layers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
import numpy as np

import tensorflow as tf


class BlockwiseInitializer(tf.keras.initializers.Initializer):
  """Initializer which concats other intializers."""

  def __init__(self, initializers, sizes, validate_args=False):
    """Creates the `BlockwiseInitializer`.

    Arguments:
      initializers: `list` of Keras initializers, e.g., `"glorot_uniform"` or
        `tf.keras.initializers.Constant(0.5413)`.
      sizes: `list` of `int` scalars representing the number of elements
        associated with each initializer in `initializers`.
      validate_args: Python `bool` indicating we should do (possibly expensive)
        graph-time assertions, if necessary.
    """
    self._initializers = initializers
    self._sizes = sizes
    self._validate_args = validate_args

  @property
  def initializers(self):
    return self._initializers

  @property
  def sizes(self):
    return self._sizes

  @property
  def validate_args(self):
    return self._validate_args

  def __call__(self, shape, dtype=None):
    """Returns a tensor object initialized as specified by the initializer.

    Args:
      shape: Shape of the tensor.
      dtype: Optional dtype of the tensor. If not provided will return tensor
       of `tf.float32`.
    """
    dtype = tf.as_dtype(dtype or tf.keras.backend.floatx())
    if isinstance(shape, tf.TensorShape):
      shape_dtype = tf.int32
      shape_ = np.int32(shape)
    else:
      if not tf.is_tensor(shape):
        shape = tf.convert_to_tensor(
            value=shape, dtype_hint=tf.int32, name='shape')
      shape_dtype = shape.dtype.base_dtype
      shape_ = tf.get_static_value(shape, partial=True)

    sizes_ = tf.get_static_value(self.sizes)
    if sizes_ is not None:
      sizes_ = np.array(sizes_, shape_dtype.as_numpy_dtype)

    assertions = []
    message = 'Rightmost dimension of shape must equal `sum(sizes)`.'
    n = shape[-1] if shape_ is None or shape_[-1] is None else shape_[-1]
    if sizes_ is not None and not tf.is_tensor(n):
      if sum(sizes_) != n:
        raise ValueError(message)
    elif self.validate_args:
      assertions.append(tf.compat.v1.assert_equal(
          shape[-1], tf.reduce_sum(input_tensor=self.sizes), message=message))

    s = (shape[:-1]
         if shape_ is None or any(s is None for s in shape_[:-1])
         else shape_[:-1])
    if sizes_ is not None and isinstance(s, (np.ndarray, np.generic)):
      return tf.concat([
          tf.keras.initializers.get(init)(np.concatenate([
              s, np.array([e], shape_dtype.as_numpy_dtype)], axis=-1), dtype)
          for init, e in zip(self.initializers, sizes_.tolist())
      ], axis=-1)

    sizes = tf.split(self.sizes, len(self.initializers))
    return tf.concat([
        tf.keras.initializers.get(init)(tf.concat([s, e], axis=-1), dtype)
        for init, e in zip(self.initializers, sizes)
    ], axis=-1)

  def get_config(self):
    """Returns initializer configuration as a JSON-serializable dict."""
    return {
        'initializers': [
            tf.compat.v2.initializers.serialize(
                tf.keras.initializers.get(init))
            for init in self.initializers
        ],
        'sizes': self.sizes,
        'validate_args': self.validate_args,
    }

  @classmethod
  def from_config(cls, config):
    """Instantiates an initializer from a configuration dictionary."""
    return cls(**{
        'initializers': [tf.compat.v2.initializers.deserialize(init)
                         for init in config.get('initializers', [])],
        'sizes': config.get('sizes', []),
        'validate_args': config.get('validate_args', False),
    })


tf.keras.utils.get_custom_objects()[
    'BlockwiseInitializer'] = BlockwiseInitializer
