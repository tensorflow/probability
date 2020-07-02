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
"""Builds a Bayesian ResNet18 Model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v1 as tf
import tensorflow_probability as tfp


def bayesian_resnet(input_shape,
                    num_classes=10,
                    kernel_posterior_scale_mean=-9.0,
                    kernel_posterior_scale_stddev=0.1,
                    kernel_posterior_scale_constraint=0.2):
  """Constructs a ResNet18 model.

  Args:
    input_shape: A `tuple` indicating the Tensor shape.
    num_classes: `int` representing the number of class labels.
    kernel_posterior_scale_mean: Python `int` number for the kernel
      posterior's scale (log variance) mean. The smaller the mean the closer
      is the initialization to a deterministic network.
    kernel_posterior_scale_stddev: Python `float` number for the initial kernel
      posterior's scale stddev.
      ```
      q(W|x) ~ N(mu, var),
      log_var ~ N(kernel_posterior_scale_mean, kernel_posterior_scale_stddev)
      ````
    kernel_posterior_scale_constraint: Python `float` number for the log value
      to constrain the log variance throughout training.
      i.e. log_var <= log(kernel_posterior_scale_constraint).

  Returns:
    tf.keras.Model.
  """

  filters = [64, 128, 256, 512]
  kernels = [3, 3, 3, 3]
  strides = [1, 2, 2, 2]

  def _untransformed_scale_constraint(t):
    return tf.clip_by_value(t, -1000,
                            tf.math.log(kernel_posterior_scale_constraint))

  kernel_posterior_fn = tfp.layers.default_mean_field_normal_fn(
      untransformed_scale_initializer=tf.compat.v1.initializers.random_normal(
          mean=kernel_posterior_scale_mean,
          stddev=kernel_posterior_scale_stddev),
      untransformed_scale_constraint=_untransformed_scale_constraint)

  image = tf.keras.layers.Input(shape=input_shape, dtype='float32')
  x = tfp.layers.Convolution2DFlipout(
      64,
      3,
      strides=1,
      padding='same',
      kernel_posterior_fn=kernel_posterior_fn)(image)

  for i in range(len(kernels)):
    x = _resnet_block(
        x,
        filters[i],
        kernels[i],
        strides[i],
        kernel_posterior_fn)

  x = tf.keras.layers.BatchNormalization()(x)
  x = tf.keras.layers.Activation('relu')(x)
  x = tf.keras.layers.AveragePooling2D(4, 1)(x)
  x = tf.keras.layers.Flatten()(x)

  x = tfp.layers.DenseFlipout(
      num_classes,
      kernel_posterior_fn=kernel_posterior_fn)(x)

  model = tf.keras.Model(inputs=image, outputs=x, name='resnet18')
  return model


def _resnet_block(x, filters, kernel, stride, kernel_posterior_fn):
  """Network block for ResNet."""
  x = tf.keras.layers.BatchNormalization()(x)
  x = tf.keras.layers.Activation('relu')(x)

  if stride != 1 or filters != x.shape[1]:
    shortcut = _projection_shortcut(x, filters, stride, kernel_posterior_fn)
  else:
    shortcut = x

  x = tfp.layers.Convolution2DFlipout(
      filters,
      kernel,
      strides=stride,
      padding='same',
      kernel_posterior_fn=kernel_posterior_fn)(x)
  x = tf.keras.layers.BatchNormalization()(x)
  x = tf.keras.layers.Activation('relu')(x)

  x = tfp.layers.Convolution2DFlipout(
      filters,
      kernel,
      strides=1,
      padding='same',
      kernel_posterior_fn=kernel_posterior_fn)(x)
  x = tf.keras.layers.add([x, shortcut])
  return x


def _projection_shortcut(x, out_filters, stride, kernel_posterior_fn):
  x = tfp.layers.Convolution2DFlipout(
      out_filters,
      1,
      strides=stride,
      padding='valid',
      kernel_posterior_fn=kernel_posterior_fn)(x)
  return x
