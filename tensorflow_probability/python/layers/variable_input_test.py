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
"""Tests for tensorflow_probability.layers.VariableInputLayer."""

import tensorflow.compat.v2 as tf
from tensorflow_probability.python.distributions import independent
from tensorflow_probability.python.distributions import normal
from tensorflow_probability.python.internal import test_util
from tensorflow_probability.python.layers import distribution_layer
from tensorflow_probability.python.layers import variable_input


@test_util.test_all_tf_execution_regimes
class VariableInputLayerTest(test_util.TestCase):

  def test_sequential_api(self):
    # Create a trainable distribution using the Sequential API.
    model = tf.keras.models.Sequential([
        variable_input.VariableLayer(
            shape=[2, 3, 4],
            dtype=tf.float64,
            trainable=False),  # You'd probably never want this in IRL.
        # The Dense serves no real purpose; it will change the event_shape.
        tf.keras.layers.Dense(5, use_bias=False, dtype=tf.float64),
        distribution_layer.DistributionLambda(
            lambda t: independent.Independent(  # pylint: disable=g-long-lambda
                normal.Normal(loc=t[0], scale=t[1]),
                reinterpreted_batch_ndims=1),
            dtype=tf.float64),

    ])

    # Instantiate the model (as a TFP distribution).
    dist = model(tf.zeros([]))

    # Check the weights.
    self.assertEqual(2, len(model.weights))

    # Check the VariableLayer layer.
    self.assertIs(tf.float64, tf.as_dtype(model.weights[0].dtype))
    self.assertEqual((2, 3, 4), model.layers[0].weights[0].shape)
    self.assertFalse(model.layers[0].trainable)
    self.assertFalse(model.layers[0].weights[0].trainable)

    # Check the Dense layer.
    self.assertIs(tf.float64, tf.as_dtype(model.weights[1].dtype))
    self.assertEqual((4, 5), model.layers[1].weights[0].shape)
    self.assertTrue(model.layers[1].trainable)
    self.assertTrue(model.layers[1].weights[0].trainable)

    # Check the distribution.
    self.assertIsInstance(dist.tensor_distribution, independent.Independent)
    self.assertIs(tf.float64, dist.dtype)
    self.assertEqual((3,), dist.batch_shape)
    self.assertEqual((5,), dist.event_shape)

  def test_functional_api(self):
    # Create a trainable distribution using the functional API.
    dummy_input = tf.keras.Input(shape=())
    x = variable_input.VariableLayer(
        shape=[2, 3, 4],
        dtype=tf.float64,
        trainable=False,  # You'd probably never want this in IRL.
    )(dummy_input)
    # The Dense serves no real purpose; it will change the event_shape.
    x = tf.keras.layers.Dense(5, use_bias=False, dtype=tf.float64)(x)
    x = distribution_layer.DistributionLambda(
        lambda t: independent.Independent(normal.Normal(loc=t[0], scale=t[1]),  # pylint: disable=g-long-lambda
                                          reinterpreted_batch_ndims=1),
        dtype=tf.float64)(x)
    model = tf.keras.Model(dummy_input, x)

    # Instantiate the model (as a TFP distribution).
    dist = model(tf.zeros([]))

    # Check the weights.
    self.assertEqual(2, len(model.weights))

    # Check the VariableLayer layer.
    self.assertIs(tf.float64, tf.as_dtype(model.weights[0].dtype))
    self.assertEqual((2, 3, 4), model.layers[1].weights[0].shape)
    self.assertFalse(model.layers[1].trainable)
    self.assertFalse(model.layers[1].weights[0].trainable)

    # Check the Dense layer.
    self.assertIs(tf.float64, tf.as_dtype(model.weights[1].dtype))
    self.assertEqual((4, 5), model.layers[2].weights[0].shape)
    self.assertTrue(model.layers[2].trainable)
    self.assertTrue(model.layers[2].weights[0].trainable)

    # Check the distribution.
    self.assertIsInstance(dist.tensor_distribution, independent.Independent)
    self.assertIs(tf.float64, dist.dtype)
    self.assertEqual((3,), dist.batch_shape)
    self.assertEqual((5,), dist.event_shape)


if __name__ == '__main__':
  test_util.main()
