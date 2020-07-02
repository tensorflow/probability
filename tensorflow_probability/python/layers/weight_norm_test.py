# Copyright 2019 The TensorFlow Probability Authors.
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
"""Tests for tensorflow_probability.layers.WeightNorm."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tempfile

# Dependency imports

from absl.testing import parameterized

import numpy as np
import tensorflow.compat.v2 as tf

from tensorflow_probability.python.internal import test_util
from tensorflow_probability.python.layers import weight_norm
tfk = tf.keras

tfkl = tf.keras.layers


# TODO(b/143642032): Figure out how to get this working with
# @test_util.test_all_tf_execution_regimes
@test_util.test_graph_and_eager_modes
class WeightNormTest(test_util.TestCase):

  def setUp(self):
    super(WeightNormTest, self).setUp()
    self.data_dim = 2
    self.batch_size = 16
    self.num_hidden = 5
    self.random_input = np.random.rand(
        self.batch_size, self.data_dim).astype(np.float32)
    self.random_targets = np.random.rand(self.batch_size, 1).astype(np.float32)

    data_dim = (32, 32, 3)
    self.conv_random_input = np.random.rand(
        self.batch_size, *data_dim).astype(np.float32)
    self.num_conv_filters = 5

  def _define_model(
      self, model_type, data_dim, num_hidden, kernel_initializer='ones',
      use_weight_norm=True, data_init=True):

    if use_weight_norm:
      base_layer = tfkl.Dense(num_hidden, kernel_initializer=kernel_initializer)
      layer = weight_norm.WeightNorm(
          base_layer, data_init=data_init, name='maybe_norm_layer')
    else:
      layer = tfkl.Dense(num_hidden, kernel_initializer=kernel_initializer,
                         name='maybe_norm_layer')

    if model_type == 'layer':
      return layer
    elif model_type == 'sequential':
      return tfk.Sequential(
          [tfkl.InputLayer((data_dim,)), layer,
           tfkl.Dense(1, kernel_initializer=kernel_initializer)])
    elif model_type == 'sequential_no_input':
      return tfk.Sequential(
          [layer, tfkl.Dense(1, kernel_initializer=kernel_initializer)])
    elif model_type == 'functional':
      inputs = tfkl.Input(shape=(data_dim,))
      net = layer(inputs)
      outputs = tfkl.Dense(1, kernel_initializer=kernel_initializer)(net)
      return tfk.Model(inputs=inputs, outputs=outputs)
    else:
      raise ValueError('{} is not a valid model type'.format(model_type))

  def _get_maybe_norm_layer(self, model):
    if model.name == 'maybe_norm_layer':
      return model
    else:
      return model.get_layer('maybe_norm_layer')

  def _calculate_true_initial_variables_dense(
      self, inputs, data_dim, num_hidden):
    """Calculate values of initialized vars from Salimans and Kingma (2016)."""
    init_kernel = np.ones((data_dim, num_hidden))
    normalized_features = (np.matmul(inputs, init_kernel) /
                           np.linalg.norm(init_kernel, axis=0))
    true_init_g = 1. / np.sqrt(np.var(normalized_features, axis=0) + 1e-10)
    true_init_bias = -np.mean(normalized_features) * true_init_g
    return true_init_g, true_init_bias

  # Calculate expected initial variables
  def _calculate_true_initial_variables_conv(
      self, inputs, num_filters, transpose=False):
    """Calculate values of initialized vars from Salimans and Kingma (2016)."""

    data_dim = inputs.shape[1:]
    norm_axes_out = (0, 1, 2)
    if transpose:
      conv = tfkl.Conv2DTranspose
      norm_axes_kernel = (0, 1, 3)
    else:
      conv = tfkl.Conv2D
      norm_axes_kernel = norm_axes_out

    # Initialize a convolutional layer and normalize the kernel
    conv_layer = conv(
        filters=num_filters, kernel_size=(2, 2), kernel_initializer='ones')
    conv_layer.build(input_shape=(None,) + data_dim)

    self.evaluate([v.initializer for v in conv_layer.weights])
    conv_layer.kernel = tf.nn.l2_normalize(
        conv_layer.kernel, axis=norm_axes_kernel)

    # Calculate initial parameter values (Equation 6 of Salimans and Kingma
    # (2016))
    x_init = self.evaluate(conv_layer(inputs))
    m_init = np.mean(x_init, axis=norm_axes_out)
    v_init = np.var(x_init, axis=norm_axes_out)
    scale_init = 1. / np.sqrt(v_init + 1e-10)
    true_init_g = scale_init
    true_init_bias = -m_init * scale_init
    return true_init_g, true_init_bias

  @parameterized.parameters(
      ['layer', 'sequential', 'sequential_no_input', 'functional'])
  def testTrainableVariablesAreCreatedByCall(self, model_type):
    model = self._define_model(model_type, self.data_dim, self.num_hidden)
    model(self.random_input)
    layer = self._get_maybe_norm_layer(model)
    self.assertLen(layer.trainable_variables, 3)

  @parameterized.parameters(
      ['layer', 'sequential', 'sequential_no_input', 'functional'])
  def testTrainableVariablesAreCreatedByBuild(self, model_type):
    model = self._define_model(model_type, self.data_dim, self.num_hidden)
    model.build(self.random_input.shape)
    layer = self._get_maybe_norm_layer(model)
    self.assertLen(layer.trainable_variables, 3)

  def testVariableCreationNoBias(self):
    conv_layer = tfkl.Conv2D(filters=self.num_conv_filters, kernel_size=(2, 2),
                             kernel_initializer='ones', use_bias=False)
    norm_layer = weight_norm.WeightNorm(conv_layer, name='norm_layer')
    norm_layer.build(self.conv_random_input.shape)
    self.assertLen(norm_layer.trainable_variables, 2)

  @parameterized.parameters(
      ['layer', 'sequential', 'sequential_no_input', 'functional'])
  def testCorrectInitialValues(self, model_type):
    true_init_g, true_init_bias = self._calculate_true_initial_variables_dense(
        self.random_input, self.data_dim, self.num_hidden)

    # Build the model and call it on a batch of data to trigger initialization
    model = self._define_model(model_type, self.data_dim, self.num_hidden)
    model.build(self.random_input.shape)
    self.evaluate([v.initializer for v in model.weights])
    norm_layer = self._get_maybe_norm_layer(model)
    self.evaluate(model(self.random_input))

    self.assertAllClose(true_init_g, self.evaluate(norm_layer.g))
    self.assertAllClose(true_init_bias, self.evaluate(norm_layer.layer.bias))
    self.assertTrue(self.evaluate(norm_layer.initialized))

    # Make sure variables are not re-initialized with a second batch of data
    new_input = np.random.rand(self.batch_size, self.data_dim) + 5.
    self.evaluate(model(new_input))
    self.assertAllClose(true_init_g, self.evaluate(norm_layer.g))
    self.assertAllClose(true_init_bias, self.evaluate(norm_layer.layer.bias))

  @parameterized.parameters(
      ['layer', 'sequential', 'sequential_no_input', 'functional'])
  def testGradientValues(self, model_type):
    model = self._define_model(model_type, self.data_dim, self.num_hidden)
    model.build(self.random_input.shape)
    self.evaluate([v.initializer for v in model.weights])
    norm_layer = self._get_maybe_norm_layer(model)
    self.evaluate(model(self.random_input))

    # Calculate gradients of a loss function with respect to `g` and `v`
    with tf.GradientTape(persistent=True) as tape:
      out = model(self.random_input)
      loss = tf.norm(out)
    grad_loss_v = self.evaluate(tape.gradient(loss, norm_layer.v))
    grad_loss_g = self.evaluate(tape.gradient(loss, norm_layer.g))

    # Build an identical model with no weight normalization
    model_no_norm = self._define_model(
        model_type, self.data_dim, self.num_hidden, use_weight_norm=False)
    model_no_norm.build(self.random_input.shape)
    self.evaluate([v.initializer for v in model_no_norm.weights])
    layer = self._get_maybe_norm_layer(model_no_norm)
    self.evaluate(model_no_norm(self.random_input))

    # Calculate gradients of the loss function with respect to the kernel
    kernel = tf.nn.l2_normalize(
        norm_layer.v, axis=norm_layer.kernel_norm_axes) * norm_layer.g
    with tf.GradientTape() as tape:
      with tf.control_dependencies(
          [layer.kernel.assign(kernel),
           layer.bias.assign(norm_layer.layer.bias)]):
        out_no_norm = model_no_norm(self.random_input)
        loss_no_norm = tf.norm(out_no_norm)
    grad_loss_w = self.evaluate(tape.gradient(loss_no_norm, layer.kernel))

    # Check that gradient equations (3) in Salimans and Kingma (2016) hold.
    g = self.evaluate(norm_layer.g)
    v = self.evaluate(norm_layer.v)
    v_norm = self.evaluate(tf.sqrt(
        tf.reduce_sum(tf.square(v), axis=norm_layer.kernel_norm_axes)))

    grad_loss_v_calculated = ((g / v_norm) * grad_loss_w -
                              g * grad_loss_g * v / tf.square(v_norm))
    grad_loss_g_calculated = np.diag(
        np.matmul(np.transpose(grad_loss_w), v) / v_norm)

    self.assertAllClose(grad_loss_g, grad_loss_g_calculated)
    self.assertAllClose(grad_loss_v, grad_loss_v_calculated)

  @parameterized.parameters(['sequential', 'sequential_no_input', 'functional'])
  def testTrainableVariableInitializationInModelFit(self, model_type):
    sgd = tf.keras.optimizers.SGD(lr=0.)
    model = self._define_model(model_type, self.data_dim, self.num_hidden)
    model.compile(optimizer=sgd, loss='mse')
    model.fit(
        x=self.random_input,
        y=self.random_targets,
        batch_size=4,
        shuffle=False,
        epochs=1,
        )

    norm_layer = self._get_maybe_norm_layer(model)
    self.assertLen(norm_layer.trainable_variables, 3)
    self.assertTrue(self.evaluate(norm_layer.initialized))

    # Verify initialization is correct
    true_init_g, true_init_bias = self._calculate_true_initial_variables_dense(
        self.random_input[:4], self.data_dim, self.num_hidden)
    self.assertAllClose(true_init_g, self.evaluate(norm_layer.g))
    self.assertAllClose(true_init_bias, self.evaluate(norm_layer.layer.bias))

    # Ensure model isn't re-initialized when called on new data
    new_input = np.random.rand(self.batch_size, self.data_dim)
    model.fit(
        x=new_input,
        y=self.random_targets,
        batch_size=4,
        shuffle=False,
        epochs=1,
        )

    self.assertAllClose(true_init_g, self.evaluate(norm_layer.g))
    self.assertAllClose(true_init_bias, self.evaluate(norm_layer.layer.bias))

    model(new_input)
    self.assertAllClose(true_init_g, self.evaluate(norm_layer.g))
    self.assertAllClose(true_init_bias, self.evaluate(norm_layer.layer.bias))

    # Ensure variables are updated when learning rate is nonzero
    sgd = tf.keras.optimizers.SGD(lr=0.01)
    model.compile(optimizer=sgd, loss='mse')
    model.fit(
        x=self.random_input,
        y=self.random_targets,
        batch_size=4,
        shuffle=False,
        epochs=1,
        )

    norm_layer = self._get_maybe_norm_layer(model)
    self.assertNotAllClose(true_init_g, self.evaluate(norm_layer.g))
    self.assertNotAllClose(true_init_bias, self.evaluate(norm_layer.layer.bias))

  @parameterized.parameters(
      ['layer', 'sequential', 'sequential_no_input', 'functional'])
  def testNoDataDependentInitialization(self, model_type):
    init_kernel = np.ones((self.data_dim, self.num_hidden))
    init_norm = tf.reshape(tf.norm(
        tf.reshape(init_kernel, [-1, self.num_hidden]), axis=0),
                           (self.num_hidden,))

    model = self._define_model(model_type, self.data_dim, self.num_hidden,
                               data_init=False)
    model.build(self.random_input.shape)
    self.evaluate([v.initializer for v in model.weights])
    norm_layer = self._get_maybe_norm_layer(model)
    self.evaluate(model(self.random_input))

    self.assertAllClose(init_norm, self.evaluate(norm_layer.g))

  @parameterized.named_parameters(('conv', False), ('conv_transpose', True))
  def testConv2DInitializedCorrectly(self, transpose):

    conv = tfkl.Conv2DTranspose if transpose else tfkl.Conv2D
    conv_layer = conv(filters=self.num_conv_filters, kernel_size=(2, 2),
                      kernel_initializer='ones')
    norm_layer = weight_norm.WeightNorm(conv_layer, name='norm_layer')

    norm_layer.build(self.conv_random_input.shape)
    self.evaluate([v.initializer for v in norm_layer.weights])
    self.evaluate(norm_layer(self.conv_random_input))

    true_init_g, true_init_bias = self._calculate_true_initial_variables_conv(
        self.conv_random_input, self.num_conv_filters, transpose=transpose)

    self.assertAllClose(true_init_g, self.evaluate(norm_layer.g))
    self.assertAllClose(true_init_bias, self.evaluate(norm_layer.layer.bias))

  def testCheckpoint(self):
    model = self._define_model('sequential', self.data_dim, self.num_hidden)
    self.evaluate([v.initializer for v in model.weights])
    checkpoint = tf.train.Checkpoint(model=model)
    model_dir = tempfile.mkdtemp()
    checkpoint.save(file_prefix=model_dir)

if __name__ == '__main__':
  tf.test.main()
