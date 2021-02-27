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
"""MonteCarloDropout layer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


__all__ = [
    'MonteCarloDropout',
]


class MonteCarloDropout(tf.keras.layers.Layer):
  """A Monte Carlo dropout layer.

    Following [Kendall et all. (2017)][1] a Monte Carlo dropout is a type of
    dropout that is not only applied at training time but also at test time.
    In other words, this layer is similar to `tf.keras.layers.Dropout` but
    it ignores the training flag set to true by `model.fit` and to false by
    `model.predict`. Unlike `tf.keras.layers.Dropout`, this layer does not
    return the input unchanged if `training=False`, but always randomly drops a
    fraction (`rate`) of the input nodes.

    #### References

    [1]: Alex Kendall, Yarin Gal, What Uncertainties Do We Need in Bayesian
         Deep Learning for Computer Vision. In _Neural Information Processing
         Systems_, 2017. https://arxiv.org/abs/1703.04977
  """

  def __init__(self, rate, noise_shape=None, seed=None, **kwargs):
    """Creates the `MonteCarloDropout` layer.

    Args:
      rate: Float between 0 and 1. Fraction of the input units to drop.
      noise_shape: 1D integer tensor representing the shape of the binary
        dropout mask that will be multiplied with the input.
        For instance, if your input have shape
        `(batch_size, timesteps, features)` and you want the dropout mask to be
        the same for all timesteps, you can use
        `noise_shape=(batch_size, 1, features)`.
      seed: A Python Integer to use as a random seed.
      **kwargs: Extra arguments forwarded to `tf.keras.layers.Layer`.
    """
    super(MonteCarloDropout, self).__init__(**kwargs)
    self.rate = rate
    self.noise_shape = noise_shape
    self.seed = seed

  def _get_noise_shape(self, inputs):
    if self.noise_shape is None:
      return None

    input_shape = tf.shape(inputs)
    noise_shape = []
    for noise_value, input_value in zip(self.noise_shape, input_shape):
      noise_shape.append(input_value if noise_value is None else noise_value)
    return tf.convert_to_tensor(noise_shape)

  def call(self, inputs):
    return tf.nn.dropout(inputs,
                         self.rate,
                         noise_shape=self.noise_shape,
                         seed=self.seed)

  def compute_output_shape(self, input_shape):
    """Computes the output shape of the layer.

    Args:
      input_shape: Shape tuple (tuple of integers) or list of shape tuples
        (one per output tensor of the layer). Shape tuples can include None for
        free dimensions, instead of an integer.

    Returns:
      output_shape: A tuple representing the output shape.
    """
    return input_shape

  def get_config(self):
    """Returns the config of the layer.

    A layer config is a Python dictionary (serializable) containing the
    configuration of a layer. The same layer can be reinstantiated later
    (without its trained weights) from this configuration.

    Returns:
      config: A Python dictionary of class keyword arguments and their
        serialized values.
    """
    base_config = super(MonteCarloDropout, self).get_config()
    config = {
      'rate': self.rate,
      'noise_shape': self.noise_shape,
      'seed': self.seed
    }
    return base_config.update(config)
