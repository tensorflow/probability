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
"""Tests for tensorflow_probability.layers.DenseVariational."""

# Dependency imports

import numpy as np
import tensorflow.compat.v2 as tf
from tensorflow_probability.python.distributions import distribution
from tensorflow_probability.python.distributions import independent
from tensorflow_probability.python.distributions import normal
from tensorflow_probability.python.internal import test_util
from tensorflow_probability.python.layers import dense_variational_v2
from tensorflow_probability.python.layers import distribution_layer
from tensorflow_probability.python.layers import variable_input


def create_dataset():
  np.random.seed(43)
  n = 150
  w0 = 0.125
  b0 = 5.
  x_range = [-20, 60]
  x_tst = np.linspace(*x_range).astype(np.float32)

  def s(x):
    g = (x - x_range[0]) / (x_range[1] - x_range[0])
    return 3 * (0.25 + g**2.)

  x = (x_range[1] - x_range[0]) * np.random.rand(n) + x_range[0]
  eps = np.random.randn(n) * s(x)
  y = (w0 * x * (1. + np.sin(x)) + b0) + eps
  x = x[..., np.newaxis]
  x_tst = x_tst[..., np.newaxis]

  return y, x, x_tst


def posterior_mean_field(kernel_size, bias_size=0, dtype=None):
  n = kernel_size + bias_size
  c = np.log(np.expm1(1.))
  return tf.keras.Sequential([
      variable_input.VariableLayer(2 * n, dtype=dtype),
      distribution_layer.DistributionLambda(lambda t: independent.Independent(  # pylint: disable=g-long-lambda
          normal.Normal(loc=t[..., :n],
                        scale=1e-5 + tf.nn.softplus(c + t[..., n:])),
          reinterpreted_batch_ndims=1)),
  ])


def prior_trainable(kernel_size, bias_size=0, dtype=None):
  n = kernel_size + bias_size
  return tf.keras.Sequential([
      variable_input.VariableLayer(n, dtype=dtype),
      distribution_layer.DistributionLambda(
          lambda t: independent.Independent(normal.Normal(loc=t, scale=1),  # pylint: disable=g-long-lambda
                                            reinterpreted_batch_ndims=1)),
  ])


negloglik = lambda y, rv_y: -rv_y.log_prob(y)


@test_util.test_all_tf_execution_regimes
class DenseVariationalLayerTest(test_util.TestCase):

  def test_end_to_end(self):
    # Get dataset.
    y, x, x_tst = create_dataset()

    layer = dense_variational_v2.DenseVariational(1, posterior_mean_field,
                                                  prior_trainable)

    model = tf.keras.Sequential([
        layer,
        distribution_layer.DistributionLambda(
            lambda t: normal.Normal(loc=t, scale=1))
    ])

    if tf.__internal__.tf2.enabled() and tf.executing_eagerly():
      optimizer = tf.keras.optimizers.Adam(learning_rate=0.05)
    else:
      optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=0.05)
    # Do inference.
    model.compile(optimizer=optimizer,
                  loss=negloglik)
    model.fit(x, y, epochs=2, verbose=False)

    self.assertLen(model.layers, 2)
    self.assertLen(model.layers[0].variables, 2)
    self.assertEmpty(model.layers[1].variables)

    posterior, prior = model.layers[0].variables
    self.assertNotEqual(prior.name, posterior.name)
    self.assertContainsSubsequence(posterior.name, '/posterior/')
    self.assertContainsSubsequence(prior.name, '/prior/')

    # Check the output_shape.
    expected_output_shape = layer.compute_output_shape(
        (None, x.shape[-1])).as_list()
    self.assertAllEqual(expected_output_shape, (None, 1))

    # Profit.
    yhat = model(x_tst)
    self.assertIsInstance(yhat, distribution.Distribution)


if __name__ == '__main__':
  test_util.main()
