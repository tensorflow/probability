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
"""Tests for Pixel CNN++ distribution."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports

import numpy as np
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp

from tensorflow_probability.python.internal import test_util

tfd = tfp.distributions


# Not decorating with `test_util.test_all_tf_execution_regimes` since the
# `WeightNorm` layer wrapper fails in Eager without `tf.function`.
@test_util.test_graph_and_eager_modes
class PixelCnnTest(test_util.TestCase):

  def setUp(self):
    super(PixelCnnTest, self).setUp()
    self.batch_shape = (2,)
    self.image_shape = tf.TensorShape((8, 8, 3))
    self.high = 10
    self.low = 0
    self.h_shape = None
    self.num_logistic_mix = 1

  def _make_pixel_cnn(self, use_weight_norm_and_data_init=False):
    return tfd.PixelCNN(
        image_shape=self.image_shape,
        conditional_shape=self.h_shape,
        num_resnet=2,
        num_hierarchies=2,
        num_filters=1,
        num_logistic_mix=self.num_logistic_mix,
        high=self.high,
        low=self.low,
        use_weight_norm=use_weight_norm_and_data_init,
        use_data_init=use_weight_norm_and_data_init)

  def _make_fake_images(self):
    return np.random.randint(
        self.low,
        self.high,
        size=self.batch_shape + self.image_shape).astype(np.float32)

  def _make_fake_conditional(self):
    return

  def _make_fake_inputs(self):
    return self._make_fake_images()

  def _make_input_layers(self):
    return tf.keras.layers.Input(self.image_shape)

  def _get_single_pixel_logit_gradients(self, dist, logit_ind, pixel_ind):

    h = self._make_fake_conditional()
    def g(x):
      x = (2. * (x - self.low) / (self.high - self.low)) - 1.
      inputs = [x, h] if h is not None else x
      params = dist.network(inputs)
      out = self._apply_channel_conditioning(dist, x, *params)
      return out[logit_ind][pixel_ind]

    image_input = self._make_fake_images()
    _, grads = tfp.math.value_and_gradient(g, image_input)
    return self.evaluate(grads)

  def _grads_assertions(self, grads, i, j):
    # Gradients of pixel output wrt pixels below are 0
    self.assertEqual((grads[0][i+1:] != 0).sum(), 0)

    # Gradients of pixel output wrt pixels above are nonzero
    self.assertEqual((grads[0][:i] == 0).sum(), 0)

    # Gradients of pixels output wrt pixels to the right in the same row are 0
    self.assertEqual((grads[0][i][j+1:] != 0.).sum(), 0)

    # Gradients wrt pixels to the left in the same row is nonzero
    self.assertEqual((grads[0][i][:j] == 0.).sum(), 0)

    # Gradients wrt pixels in different batches are zero
    self.assertEqual((grads[1] != 0.).sum(), 0)

  def _apply_channel_conditioning(
      self, dist, image_input, component_logits, locs, scales, coeffs=None):
    """Apply conditional dependencies to the channels."""
    num_channels = dist.event_shape[-1]
    num_coeffs = num_channels * (num_channels - 1) // 2
    loc_tensors = tf.split(locs, num_channels, axis=-1)
    coef_tensors = tf.split(coeffs, num_coeffs, axis=-1)
    channel_tensors = tf.split(image_input, num_channels, axis=-1)

    coef_count = 0
    for i in range(num_channels):
      channel_tensors[i] = tf.expand_dims(channel_tensors[i], axis=-2)
      for j in range(i):
        loc_tensors[i] += channel_tensors[j] * coef_tensors[coef_count]
        coef_count += 1
    return [component_logits, tf.concat(loc_tensors, axis=-1), scales]

  def testLogProb(self):
    dist = self._make_pixel_cnn()
    images = self._make_fake_images()
    h = self._make_fake_conditional()

    self.evaluate([v.initializer for v in dist.network.weights])
    log_prob_values = self.evaluate(dist.log_prob(images, conditional_input=h))

    self.assertAllEqual(self.batch_shape, log_prob_values.shape)
    prob = tf.exp(log_prob_values)
    prob_ = self.evaluate(prob)
    self.assertAllInRange(prob_, 0., 1.)

  def testSample(self):

    num_samples = 2
    h = self._make_fake_conditional()

    # Weight normalization and data-dependent initialization work only in Eager
    # so we enable them only if executing eagerly. We use them only in this
    # test, since they add significantly to run time, and using them in
    # additional tests wouldn't meaningfully increase coverage.
    dist = self._make_pixel_cnn(
        use_weight_norm_and_data_init=tf.executing_eagerly())
    self.evaluate([v.initializer for v in dist.network.weights])
    sample_shape = ((num_samples,) if h is None
                    else (num_samples,) + self.batch_shape)
    samples = self.evaluate(dist.sample(sample_shape, conditional_input=h))

    self.assertAllEqual(sample_shape + self.image_shape,
                        self.evaluate(tf.shape(samples)))
    self.assertAllInRange(samples, self.low, self.high)
    self.assertLessEqual(np.unique(samples).size, self.high+1)

    sample_rng_1 = dist.sample(
        self.batch_shape, conditional_input=h, seed=test_util.test_seed())
    sample_rng_2 = dist.sample(
        self.batch_shape, conditional_input=h, seed=test_util.test_seed())
    self.assertAllEqual(
        self.evaluate(sample_rng_1), self.evaluate(sample_rng_2))

  def testAutoregression(self):

    inputs = self._make_input_layers()
    dist = self._make_pixel_cnn()

    if isinstance(inputs, list):
      log_prob = dist.log_prob(inputs[0], conditional_input=inputs[1])
    else:
      log_prob = dist.log_prob(inputs)

    # Build/fit a model to activate autoregressive kernel constraints
    model = tf.keras.Model(inputs=inputs, outputs=log_prob)
    model.add_loss(-tf.reduce_mean(log_prob))

    model.compile()
    train_data = self._make_fake_inputs()
    model.fit(x=train_data)

    ### Test gradients of logistic means
    i, j = 2, 3
    pixel_ind = (0, i, j, 0, 1)
    grads = self._get_single_pixel_logit_gradients(dist, 1, pixel_ind)
    self._grads_assertions(grads, i, j)

    # Since we're testing a green pixel: Gradients wrt the green and blue
    # components of the input pixel are 0
    self.assertEqual((grads[0][i][j][1:] != 0.).sum(), 0)

    # Gradients with respect to the red pixel are nonzero
    self.assertNotEqual(grads[0][i][j][0], 0)

    ### Test gradients of logistic scales
    i, j = 2, 2
    pixel_ind = (0, i, j, 0, 1)
    grads = self._get_single_pixel_logit_gradients(dist, 2, pixel_ind)
    self._grads_assertions(grads, i, j)

    ### Test gradients of mixture components
    i, j = 3, 3
    pixel_ind = (0, i, j, 0)
    grads = self._get_single_pixel_logit_gradients(dist, 0, pixel_ind)
    self._grads_assertions(grads, i, j)

  def testNetworkOutputShapes(self):
    dist = self._make_pixel_cnn()
    inputs = self._make_fake_inputs()
    out = dist.network(inputs)
    batch_image_shape = self.batch_shape + self.image_shape

    num_channels = self.image_shape[-1]
    num_coeffs = num_channels * (num_channels - 1) // 2

    self.assertAllEqual(
        batch_image_shape[:-1]+(self.num_logistic_mix,), out[0].shape)
    self.assertAllEqual(
        batch_image_shape[:-1] + (self.num_logistic_mix, num_coeffs),
        out[3].shape)
    for i in [1, 2]:
      self.assertAllEqual(
          batch_image_shape[:-1] + (self.num_logistic_mix, num_channels),
          out[i].shape)

  def testDtype(self):
    inputs = self._make_fake_images()
    h = self._make_fake_conditional()
    dist = self._make_pixel_cnn()
    num_samples = 4

    sample_shape = (num_samples if h is None
                    else (num_samples,) + self.batch_shape)

    samples = dist.sample(sample_shape, conditional_input=h)
    log_prob = dist.log_prob(inputs, conditional_input=h)

    self.assertEqual(dist.dtype, tf.float32)
    self.assertEqual(dist.dtype, samples.dtype)
    self.assertEqual(dist.dtype, log_prob.dtype)

    dist64 = tfd.PixelCNN(
        image_shape=self.image_shape,
        conditional_shape=self.h_shape,
        num_resnet=2,
        num_hierarchies=2,
        num_filters=3,
        num_logistic_mix=2,
        dtype=tf.float64,
        high=self.high,
        low=self.low,
        use_weight_norm=False,
        use_data_init=False)

    self.assertEqual(dist64.dtype, tf.float64)
    self.assertEqual(dist64.dtype,
                     dist64.sample(sample_shape, conditional_input=h).dtype)


# As with `PixelCnnTest`, we do not use
# `test_util.test_all_tf_execution_regimes` since the `WeightNorm` layer wrapper
# fails in Eager without `tf.function`.
@test_util.test_graph_and_eager_modes
class ConditionalPixelCnnTest(PixelCnnTest):

  def setUp(self):
    super(ConditionalPixelCnnTest, self).setUp()
    self.h_shape = (5,)

  def _make_fake_conditional(self):
    return np.random.randint(10, size=self.batch_shape + self.h_shape
                            ).astype(np.float32)

  def _make_fake_inputs(self):
    return [self._make_fake_images(), self._make_fake_conditional()]

  def _make_input_layers(self):
    return [tf.keras.layers.Input(shape=self.image_shape),
            tf.keras.layers.Input(shape=self.h_shape)]

  def testScalarConditional(self):
    dist = tfd.PixelCNN(
        image_shape=self.image_shape,
        conditional_shape=(),
        num_resnet=2,
        num_hierarchies=2,
        num_filters=3,
        num_logistic_mix=2,
        high=self.high,
        low=self.low)

    self.evaluate([v.initializer for v in dist.network.weights])
    self.evaluate(
        dist.log_prob(
            dist.sample(conditional_input=1.),
            conditional_input=0.))

if __name__ == '__main__':
  tf.test.main()
