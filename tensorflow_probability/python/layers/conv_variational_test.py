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
"""Tests for convolutional variational layers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
import numpy as np

import tensorflow as tf
import tensorflow_probability as tfp

from tensorflow.python.keras._impl.keras import testing_utils
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops.distributions import util as distribution_util


tfd = tf.contrib.distributions


class Counter(object):
  """Helper class to manage incrementing a counting `int`."""

  def __init__(self):
    self._value = -1

  @property
  def value(self):
    return self._value

  def __call__(self):
    self._value += 1
    return self._value


class MockDistribution(tfd.Independent):
  """Monitors layer calls to the underlying distribution."""

  def __init__(self, result_sample, result_log_prob, loc=None, scale=None):
    self.result_sample = result_sample
    self.result_log_prob = result_log_prob
    self.result_loc = loc
    self.result_scale = scale
    self.result_distribution = tfd.Normal(loc=0.0, scale=1.0)
    if loc is not None and scale is not None:
      self.result_distribution = tfd.Normal(loc=self.result_loc,
                                            scale=self.result_scale)
    self.called_log_prob = Counter()
    self.called_sample = Counter()
    self.called_loc = Counter()
    self.called_scale = Counter()

  def log_prob(self, *args, **kwargs):
    self.called_log_prob()
    return self.result_log_prob

  def sample(self, *args, **kwargs):
    self.called_sample()
    return self.result_sample

  @property
  def distribution(self):  # for dummy check on Independent(Normal)
    return self.result_distribution

  @property
  def loc(self):
    self.called_loc()
    return self.result_loc

  @property
  def scale(self):
    self.called_scale()
    return self.result_scale


class MockKLDivergence(object):
  """Monitors layer calls to the divergence implementation."""

  def __init__(self, result):
    self.result = result
    self.args = []
    self.called = Counter()

  def __call__(self, *args, **kwargs):
    self.called()
    self.args.append(args)
    return self.result


class ConvVariational(tf.test.TestCase):

  def _testKerasLayer(self, layer_class):
    def kernel_posterior_fn(dtype, shape, name, trainable, add_variable_fn):
      """Set trivially. The function is required to instantiate layer."""
      del name, trainable, add_variable_fn  # unused
      # Deserialized Keras objects do not perform lexical scoping. Any modules
      # that the function requires must be imported within the function.
      import tensorflow as tf  # pylint: disable=g-import-not-at-top,redefined-outer-name
      tfd = tf.contrib.distributions  # pylint: disable=redefined-outer-name

      loc = tf.zeros(shape, dtype=dtype)
      scale = tf.ones(shape, dtype=dtype)
      return tfd.Independent(tfd.Normal(loc=loc, scale=scale))

    if layer_class in (tfp.layers.Convolution1DReparameterization,
                       tfp.layers.Convolution1DFlipout):
      input_shape = (2, 3, 1)
    elif layer_class in (tfp.layers.Convolution2DReparameterization,
                         tfp.layers.Convolution2DFlipout):
      input_shape = (2, 3, 3, 1)
    elif layer_class in (tfp.layers.Convolution3DReparameterization,
                         tfp.layers.Convolution3DFlipout):
      input_shape = (2, 3, 3, 3, 1)

    with tf.keras.utils.CustomObjectScope({layer_class.__name__: layer_class}):
      with self.test_session():
        testing_utils.layer_test(
            layer_class,
            kwargs={'filters': 2,
                    'kernel_size': 3,
                    'kernel_posterior_fn': kernel_posterior_fn,
                    'kernel_prior_fn': None,
                    'bias_posterior_fn': None,
                    'bias_prior_fn': None},
            input_shape=input_shape)

  def _testKLPenaltyKernel(self, layer_class):
    with self.test_session():
      layer = layer_class(filters=2, kernel_size=3)
      if layer_class in (tfp.layers.Convolution1DReparameterization,
                         tfp.layers.Convolution1DFlipout):
        inputs = tf.random_uniform([2, 3, 1], seed=1)
      elif layer_class in (tfp.layers.Convolution2DReparameterization,
                           tfp.layers.Convolution2DFlipout):
        inputs = tf.random_uniform([2, 3, 3, 1], seed=1)
      elif layer_class in (tfp.layers.Convolution3DReparameterization,
                           tfp.layers.Convolution3DFlipout):
        inputs = tf.random_uniform([2, 3, 3, 3, 1], seed=1)

      # No keys.
      losses = layer.get_losses_for(inputs=None)
      self.assertEqual(len(losses), 0)
      self.assertListEqual(layer.losses, losses)

      _ = layer(inputs)

      # Yes keys.
      losses = layer.get_losses_for(inputs=None)
      self.assertEqual(len(losses), 1)
      self.assertListEqual(layer.losses, losses)

  def _testKLPenaltyBoth(self, layer_class):
    def _make_normal(dtype, shape, *dummy_args):
      return tfd.Independent(tfd.Normal(
          loc=tf.zeros(shape, dtype), scale=dtype.as_numpy_dtype(1.)))
    with self.test_session():
      layer = layer_class(
          filters=2,
          kernel_size=3,
          bias_posterior_fn=tfp.layers.default_mean_field_normal_fn(),
          bias_prior_fn=_make_normal)
      if layer_class in (tfp.layers.Convolution1DReparameterization,
                         tfp.layers.Convolution1DFlipout):
        inputs = tf.random_uniform([2, 3, 1], seed=1)
      elif layer_class in (tfp.layers.Convolution2DReparameterization,
                           tfp.layers.Convolution2DFlipout):
        inputs = tf.random_uniform([2, 3, 3, 1], seed=1)
      elif layer_class in (tfp.layers.Convolution3DReparameterization,
                           tfp.layers.Convolution3DFlipout):
        inputs = tf.random_uniform([2, 3, 3, 3, 1], seed=1)

      # No keys.
      losses = layer.get_losses_for(inputs=None)
      self.assertEqual(len(losses), 0)
      self.assertListEqual(layer.losses, losses)

      _ = layer(inputs)

      # Yes keys.
      losses = layer.get_losses_for(inputs=None)
      self.assertEqual(len(losses), 2)
      self.assertListEqual(layer.losses, losses)

  def _testConvSetUp(self, layer_class, batch_size, depth=None,
                     height=None, width=None, channels=None, filters=None,
                     **kwargs):
    seed = Counter()
    if layer_class in (tfp.layers.Convolution1DReparameterization,
                       tfp.layers.Convolution1DFlipout):
      inputs = tf.random_uniform(
          [batch_size, width, channels], seed=seed())
      kernel_size = (2,)
    elif layer_class in (tfp.layers.Convolution2DReparameterization,
                         tfp.layers.Convolution2DFlipout):
      inputs = tf.random_uniform(
          [batch_size, height, width, channels], seed=seed())
      kernel_size = (2, 2)
    elif layer_class in (tfp.layers.Convolution3DReparameterization,
                         tfp.layers.Convolution3DFlipout):
      inputs = tf.random_uniform(
          [batch_size, depth, height, width, channels], seed=seed())
      kernel_size = (2, 2, 2)

    kernel_shape = kernel_size + (channels, filters)
    kernel_posterior = MockDistribution(
        loc=tf.random_uniform(kernel_shape, seed=seed()),
        scale=tf.random_uniform(kernel_shape, seed=seed()),
        result_log_prob=tf.random_uniform(kernel_shape, seed=seed()),
        result_sample=tf.random_uniform(kernel_shape, seed=seed()))
    kernel_prior = MockDistribution(
        result_log_prob=tf.random_uniform(kernel_shape, seed=seed()),
        result_sample=tf.random_uniform(kernel_shape, seed=seed()))
    kernel_divergence = MockKLDivergence(
        result=tf.random_uniform(kernel_shape, seed=seed()))

    bias_size = (filters,)
    bias_posterior = MockDistribution(
        result_log_prob=tf.random_uniform(bias_size, seed=seed()),
        result_sample=tf.random_uniform(bias_size, seed=seed()))
    bias_prior = MockDistribution(
        result_log_prob=tf.random_uniform(bias_size, seed=seed()),
        result_sample=tf.random_uniform(bias_size, seed=seed()))
    bias_divergence = MockKLDivergence(
        result=tf.random_uniform(bias_size, seed=seed()))

    layer = layer_class(
        filters=filters,
        kernel_size=kernel_size,
        padding='SAME',
        kernel_posterior_fn=lambda *args: kernel_posterior,
        kernel_posterior_tensor_fn=lambda d: d.sample(seed=42),
        kernel_prior_fn=lambda *args: kernel_prior,
        kernel_divergence_fn=kernel_divergence,
        bias_posterior_fn=lambda *args: bias_posterior,
        bias_posterior_tensor_fn=lambda d: d.sample(seed=43),
        bias_prior_fn=lambda *args: bias_prior,
        bias_divergence_fn=bias_divergence,
        **kwargs)

    outputs = layer(inputs)

    kl_penalty = layer.get_losses_for(inputs=None)
    return (kernel_posterior, kernel_prior, kernel_divergence,
            bias_posterior, bias_prior, bias_divergence,
            layer, inputs, outputs, kl_penalty, kernel_shape)

  def _testConvReparameterization(self, layer_class):
    batch_size, depth, height, width, channels, filters = 2, 4, 4, 4, 3, 5
    with self.test_session() as sess:
      (kernel_posterior, kernel_prior, kernel_divergence,
       bias_posterior, bias_prior, bias_divergence, layer, inputs,
       outputs, kl_penalty, kernel_shape) = self._testConvSetUp(
           layer_class, batch_size,
           depth=depth, height=height, width=width, channels=channels,
           filters=filters)

      convolution_op = nn_ops.Convolution(
          tf.TensorShape(inputs.shape),
          filter_shape=tf.TensorShape(kernel_shape),
          padding='SAME')
      expected_outputs = convolution_op(inputs, kernel_posterior.result_sample)
      expected_outputs = tf.nn.bias_add(expected_outputs,
                                        bias_posterior.result_sample,
                                        data_format='NHWC')

      [
          expected_outputs_, actual_outputs_,
          expected_kernel_, actual_kernel_,
          expected_kernel_divergence_, actual_kernel_divergence_,
          expected_bias_, actual_bias_,
          expected_bias_divergence_, actual_bias_divergence_,
      ] = sess.run([
          expected_outputs, outputs,
          kernel_posterior.result_sample, layer.kernel_posterior_tensor,
          kernel_divergence.result, kl_penalty[0],
          bias_posterior.result_sample, layer.bias_posterior_tensor,
          bias_divergence.result, kl_penalty[1],
      ])

      self.assertAllClose(
          expected_kernel_, actual_kernel_,
          rtol=1e-6, atol=0.)
      self.assertAllClose(
          expected_bias_, actual_bias_,
          rtol=1e-6, atol=0.)
      self.assertAllClose(
          expected_outputs_, actual_outputs_,
          rtol=1e-6, atol=0.)
      self.assertAllClose(
          expected_kernel_divergence_, actual_kernel_divergence_,
          rtol=1e-6, atol=0.)
      self.assertAllClose(
          expected_bias_divergence_, actual_bias_divergence_,
          rtol=1e-6, atol=0.)

      self.assertAllEqual(
          [[kernel_posterior.distribution,
            kernel_prior.distribution,
            kernel_posterior.result_sample]],
          kernel_divergence.args)

      self.assertAllEqual(
          [[bias_posterior.distribution,
            bias_prior.distribution,
            bias_posterior.result_sample]],
          bias_divergence.args)

  def _testConvFlipout(self, layer_class):
    batch_size, depth, height, width, channels, filters = 2, 4, 4, 4, 3, 5
    with self.test_session() as sess:
      (kernel_posterior, kernel_prior, kernel_divergence,
       bias_posterior, bias_prior, bias_divergence, layer, inputs,
       outputs, kl_penalty, kernel_shape) = self._testConvSetUp(
           layer_class, batch_size,
           depth=depth, height=height, width=width, channels=channels,
           filters=filters, seed=44)

      convolution_op = nn_ops.Convolution(
          tf.TensorShape(inputs.shape),
          filter_shape=tf.TensorShape(kernel_shape),
          padding='SAME')

      expected_kernel_posterior_affine = tfd.Normal(
          loc=tf.zeros_like(kernel_posterior.result_loc),
          scale=kernel_posterior.result_scale)
      expected_kernel_posterior_affine_tensor = (
          expected_kernel_posterior_affine.sample(seed=42))

      expected_outputs = convolution_op(
          inputs, kernel_posterior.distribution.loc)

      input_shape = tf.shape(inputs)
      output_shape = tf.shape(expected_outputs)
      batch_shape = tf.expand_dims(input_shape[0], 0)
      channels = input_shape[-1]
      rank = len(inputs.get_shape()) - 2

      sign_input = tf.random_uniform(
          tf.concat([batch_shape,
                     tf.expand_dims(channels, 0)], 0),
          minval=0,
          maxval=2,
          dtype=tf.int32,
          seed=layer.seed)
      sign_input = tf.cast(2 * sign_input - 1, inputs.dtype)
      sign_output = tf.random_uniform(
          tf.concat([batch_shape,
                     tf.expand_dims(filters, 0)], 0),
          minval=0,
          maxval=2,
          dtype=tf.int32,
          seed=distribution_util.gen_new_seed(
              layer.seed, salt='conv_flipout'))
      sign_output = tf.cast(2 * sign_output - 1, inputs.dtype)
      for _ in range(rank):
        sign_input = tf.expand_dims(sign_input, 1)  # 2D ex: (B, 1, 1, C)
        sign_output = tf.expand_dims(sign_output, 1)

      sign_input = tf.tile(  # tile for element-wise op broadcasting
          sign_input,
          [1] + [input_shape[i + 1] for i in range(rank)] + [1])
      sign_output = tf.tile(
          sign_output,
          [1] + [output_shape[i + 1] for i in range(rank)] + [1])

      perturbed_inputs = convolution_op(
          inputs * sign_input, expected_kernel_posterior_affine_tensor)
      perturbed_inputs *= sign_output

      expected_outputs += perturbed_inputs
      expected_outputs = tf.nn.bias_add(expected_outputs,
                                        bias_posterior.result_sample,
                                        data_format='NHWC')

      [
          expected_outputs_, actual_outputs_,
          expected_kernel_divergence_, actual_kernel_divergence_,
          expected_bias_, actual_bias_,
          expected_bias_divergence_, actual_bias_divergence_,
      ] = sess.run([
          expected_outputs, outputs,
          kernel_divergence.result, kl_penalty[0],
          bias_posterior.result_sample, layer.bias_posterior_tensor,
          bias_divergence.result, kl_penalty[1],
      ])

      self.assertAllClose(
          expected_bias_, actual_bias_,
          rtol=1e-6, atol=0.)
      self.assertAllClose(
          expected_outputs_, actual_outputs_,
          rtol=1e-6, atol=0.)
      self.assertAllClose(
          expected_kernel_divergence_, actual_kernel_divergence_,
          rtol=1e-6, atol=0.)
      self.assertAllClose(
          expected_bias_divergence_, actual_bias_divergence_,
          rtol=1e-6, atol=0.)

      self.assertAllEqual(
          [[kernel_posterior.distribution, kernel_prior.distribution, None]],
          kernel_divergence.args)

      self.assertAllEqual(
          [[bias_posterior.distribution,
            bias_prior.distribution,
            bias_posterior.result_sample]],
          bias_divergence.args)

  def _testRandomConvFlipout(self, layer_class):
    batch_size, depth, height, width, channels, filters = 2, 4, 4, 4, 3, 5
    with self.test_session() as sess:
      seed = Counter()
      if layer_class in (tfp.layers.Convolution1DReparameterization,
                         tfp.layers.Convolution1DFlipout):
        inputs = tf.random_uniform(
            [batch_size, width, channels], seed=seed())
        kernel_size = (2,)
      elif layer_class in (tfp.layers.Convolution2DReparameterization,
                           tfp.layers.Convolution2DFlipout):
        inputs = tf.random_uniform(
            [batch_size, height, width, channels], seed=seed())
        kernel_size = (2, 2)
      elif layer_class in (tfp.layers.Convolution3DReparameterization,
                           tfp.layers.Convolution3DFlipout):
        inputs = tf.random_uniform(
            [batch_size, depth, height, width, channels], seed=seed())
        kernel_size = (2, 2, 2)

      kernel_shape = kernel_size + (channels, filters)
      bias_size = (filters,)

      kernel_posterior = MockDistribution(
          loc=tf.random_uniform(
              kernel_shape, seed=seed()),
          scale=tf.random_uniform(
              kernel_shape, seed=seed()),
          result_log_prob=tf.random_uniform(
              kernel_shape, seed=seed()),
          result_sample=tf.random_uniform(
              kernel_shape, seed=seed()))
      bias_posterior = MockDistribution(
          loc=tf.random_uniform(
              bias_size, seed=seed()),
          scale=tf.random_uniform(
              bias_size, seed=seed()),
          result_log_prob=tf.random_uniform(
              bias_size, seed=seed()),
          result_sample=tf.random_uniform(
              bias_size, seed=seed()))
      layer_one = layer_class(
          filters=filters,
          kernel_size=kernel_size,
          padding='SAME',
          kernel_posterior_fn=lambda *args: kernel_posterior,
          kernel_posterior_tensor_fn=lambda d: d.sample(seed=42),
          bias_posterior_fn=lambda *args: bias_posterior,
          bias_posterior_tensor_fn=lambda d: d.sample(seed=43),
          seed=44)
      layer_two = layer_class(
          filters=filters,
          kernel_size=kernel_size,
          padding='SAME',
          kernel_posterior_fn=lambda *args: kernel_posterior,
          kernel_posterior_tensor_fn=lambda d: d.sample(seed=42),
          bias_posterior_fn=lambda *args: bias_posterior,
          bias_posterior_tensor_fn=lambda d: d.sample(seed=43),
          seed=45)

      outputs_one = layer_one(inputs)
      outputs_two = layer_two(inputs)

      outputs_one_, outputs_two_ = sess.run([
          outputs_one, outputs_two])

      self.assertLess(np.sum(np.isclose(outputs_one_, outputs_two_)),
                      np.prod(outputs_one_.shape))

  def testKerasLayerConvolution1DReparameterization(self):
    self._testKerasLayer(tfp.layers.Convolution1DReparameterization)

  def testKerasLayerConvolution2DReparameterization(self):
    self._testKerasLayer(tfp.layers.Convolution2DReparameterization)

  def testKerasLayerConvolution3DReparameterization(self):
    self._testKerasLayer(tfp.layers.Convolution3DReparameterization)

  def testKerasLayerConvolution1DFlipout(self):
    self._testKerasLayer(tfp.layers.Convolution1DFlipout)

  def testKerasLayerConvolution2DFlipout(self):
    self._testKerasLayer(tfp.layers.Convolution2DFlipout)

  def testKerasLayerConvolution3DFlipout(self):
    self._testKerasLayer(tfp.layers.Convolution3DFlipout)

  def testKLPenaltyKernelConvolution1DReparameterization(self):
    self._testKLPenaltyKernel(tfp.layers.Convolution1DReparameterization)

  def testKLPenaltyKernelConvolution2DReparameterization(self):
    self._testKLPenaltyKernel(tfp.layers.Convolution2DReparameterization)

  def testKLPenaltyKernelConvolution3DReparameterization(self):
    self._testKLPenaltyKernel(tfp.layers.Convolution3DReparameterization)

  def testKLPenaltyKernelConvolution1DFlipout(self):
    self._testKLPenaltyKernel(tfp.layers.Convolution1DFlipout)

  def testKLPenaltyKernelConvolution2DFlipout(self):
    self._testKLPenaltyKernel(tfp.layers.Convolution2DFlipout)

  def testKLPenaltyKernelConvolution3DFlipout(self):
    self._testKLPenaltyKernel(tfp.layers.Convolution3DFlipout)

  def testKLPenaltyBothConvolution1DReparameterization(self):
    self._testKLPenaltyBoth(tfp.layers.Convolution1DReparameterization)

  def testKLPenaltyBothConvolution2DReparameterization(self):
    self._testKLPenaltyBoth(tfp.layers.Convolution2DReparameterization)

  def testKLPenaltyBothConvolution3DReparameterization(self):
    self._testKLPenaltyBoth(tfp.layers.Convolution3DReparameterization)

  def testKLPenaltyBothConvolution1DFlipout(self):
    self._testKLPenaltyBoth(tfp.layers.Convolution1DFlipout)

  def testKLPenaltyBothConvolution2DFlipout(self):
    self._testKLPenaltyBoth(tfp.layers.Convolution2DFlipout)

  def testKLPenaltyBothConvolution3DFlipout(self):
    self._testKLPenaltyBoth(tfp.layers.Convolution3DFlipout)

  def testConvolution1DReparameterization(self):
    self._testConvReparameterization(tfp.layers.Convolution1DReparameterization)

  def testConvolution2DReparameterization(self):
    self._testConvReparameterization(tfp.layers.Convolution2DReparameterization)

  def testConvolution3DReparameterization(self):
    self._testConvReparameterization(tfp.layers.Convolution3DReparameterization)

  def testConvolution1DFlipout(self):
    self._testConvFlipout(tfp.layers.Convolution1DFlipout)

  def testConvolution2DFlipout(self):
    self._testConvFlipout(tfp.layers.Convolution2DFlipout)

  def testConvolution3DFlipout(self):
    self._testConvFlipout(tfp.layers.Convolution3DFlipout)

  def testRandomConvolution1DFlipout(self):
    self._testRandomConvFlipout(tfp.layers.Convolution1DFlipout)


if __name__ == '__main__':
  tf.test.main()
