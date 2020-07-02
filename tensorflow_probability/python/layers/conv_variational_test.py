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

import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp

from tensorflow_probability.python.internal import test_util
from tensorflow.python.keras import testing_utils
from tensorflow.python.layers import utils as tf_layers_util
from tensorflow.python.ops import nn_ops

tfd = tfp.distributions


def channels_last_to_first(shape):
  """Given a shape with channels last, returns a shape with channels first.

  Assumes we're dealing with the standard shapes as used by
  [TF's convolution operators][1].

  [1]: https://www.tensorflow.org/api_docs/python/tf/nn/convolution

  Args:
    shape: A sequence with each element corresponding to an axis in a shape.

  Returns:
    The transposed shape.
  """
  return shape[:1] + shape[-1:] + shape[1:-1]


def channels_first_to_last(shape):
  """Given a shape with channels first, returns a shape with channels last.

  Assumes we're dealing with the standard shapes as used by
  [TF's convolution operators][1].

  [1]: https://www.tensorflow.org/api_docs/python/tf/nn/convolution

  Args:
    shape: A sequence with each element corresponding to an axis in a shape.

  Returns:
    The transposed shape.
  """
  return shape[:1] + shape[2:] + shape[1:2]


class CPUConvolution(nn_ops.Convolution):
  """A wrapper over Convolution that works on CPU for all data formats.

  Unfortunately 'channels_first' data format is not implemented on the CPU, so
  for testing purposes we override Convolution to this class, which performs a
  number of transpositions to avoid internal errors.
  """

  def __init__(self, input_shape, data_format=None, **kwargs):
    """Creates the convolution operator.

    Args:
      input_shape: A `TensorShape` input shape.
      data_format: A string describing the data format, see
        https://www.tensorflow.org/api_docs/python/tf/nn/convolution.
      **kwargs: Passed to `nn_ops.Convolution`.
    """
    self._standardize_to_channels_last = (
        data_format is not None and data_format.startswith('NC'))
    self._rank = input_shape.rank
    if self._standardize_to_channels_last:
      data_format = channels_first_to_last(data_format)
      input_shape = tf.TensorShape(
          channels_first_to_last(input_shape.as_list()))
    super(CPUConvolution, self).__init__(
        input_shape, data_format=data_format, **kwargs)

  def __call__(self, inp, filter_):
    if self._standardize_to_channels_last:
      order = channels_first_to_last(list(range(self._rank)))
      inp = tf.transpose(a=inp, perm=order)
    ret = super(CPUConvolution, self).__call__(inp, filter_)
    if self._standardize_to_channels_last:
      order = channels_last_to_first(list(range(self._rank)))
      ret = tf.transpose(a=ret, perm=order)
    return ret


nn_ops.Convolution = CPUConvolution


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


@test_util.test_all_tf_execution_regimes
class ConvVariational(object):

  def maybe_transpose_tensor(self, tensor):
    if self.data_format == 'channels_first':
      order = channels_last_to_first(list(range(tensor.shape.rank)))
      return tf.transpose(a=tensor, perm=order)
    else:
      return tensor

  def _testKerasLayer(self, layer_class):  # pylint: disable=invalid-name
    def kernel_posterior_fn(dtype, shape, name, trainable, add_variable_fn):
      """Set trivially. The function is required to instantiate layer."""
      del name, trainable, add_variable_fn  # unused
      # Deserialized Keras objects do not perform lexical scoping. Any modules
      # that the function requires must be imported within the function.
      import tensorflow.compat.v2 as tf  # pylint: disable=g-import-not-at-top,redefined-outer-name,reimported
      import tensorflow_probability as tfp  # pylint: disable=g-import-not-at-top,redefined-outer-name,reimported
      tfd = tfp.distributions  # pylint: disable=redefined-outer-name

      dist = tfd.Normal(loc=tf.zeros(shape, dtype),
                        scale=dtype.as_numpy_dtype(1))
      batch_ndims = tf.size(dist.batch_shape_tensor())
      return tfd.Independent(dist, reinterpreted_batch_ndims=batch_ndims)

    if layer_class in (tfp.layers.Convolution1DReparameterization,
                       tfp.layers.Convolution1DFlipout):
      input_shape = (2, 3, 1)
    elif layer_class in (tfp.layers.Convolution2DReparameterization,
                         tfp.layers.Convolution2DFlipout):
      input_shape = (2, 3, 3, 1)
    elif layer_class in (tfp.layers.Convolution3DReparameterization,
                         tfp.layers.Convolution3DFlipout):
      input_shape = (2, 3, 3, 3, 1)

    if self.data_format == 'channels_first':
      input_shape = channels_last_to_first(input_shape)

    with tf.keras.utils.CustomObjectScope({layer_class.__name__: layer_class}):
      with self.cached_session():
        testing_utils.layer_test(
            layer_class,
            kwargs={'filters': 2,
                    'kernel_size': 3,
                    'kernel_posterior_fn': kernel_posterior_fn,
                    'kernel_prior_fn': None,
                    'bias_posterior_fn': None,
                    'bias_prior_fn': None,
                    'data_format': self.data_format},
            input_shape=input_shape)

  def _testKLPenaltyKernel(self, layer_class):  # pylint: disable=invalid-name
    with self.cached_session():
      layer = layer_class(
          filters=2, kernel_size=3, data_format=self.data_format)
      if layer_class in (tfp.layers.Convolution1DReparameterization,
                         tfp.layers.Convolution1DFlipout):
        inputs = tf.random.uniform([2, 3, 1], seed=1)
      elif layer_class in (tfp.layers.Convolution2DReparameterization,
                           tfp.layers.Convolution2DFlipout):
        inputs = tf.random.uniform([2, 3, 3, 1], seed=1)
      elif layer_class in (tfp.layers.Convolution3DReparameterization,
                           tfp.layers.Convolution3DFlipout):
        inputs = tf.random.uniform([2, 3, 3, 3, 1], seed=1)
      inputs = self.maybe_transpose_tensor(inputs)

      # No keys.
      input_dependent_losses = layer.get_losses_for(inputs=None)
      self.assertEqual(len(layer.losses), 0)
      self.assertListEqual(layer.losses, input_dependent_losses)

      _ = layer(inputs)

      # Yes keys.
      input_dependent_losses = layer.get_losses_for(inputs=None)
      self.assertEqual(len(layer.losses), 1)
      self.assertEqual(layer.losses[0].shape, ())
      self.assertListEqual(layer.losses, input_dependent_losses)

  def _testKLPenaltyBoth(self, layer_class):  # pylint: disable=invalid-name
    with self.cached_session():
      layer = layer_class(
          filters=2,
          kernel_size=3,
          bias_posterior_fn=tfp.layers.default_mean_field_normal_fn(),
          bias_prior_fn=tfp.layers.default_multivariate_normal_fn,
          data_format=self.data_format)
      if layer_class in (tfp.layers.Convolution1DReparameterization,
                         tfp.layers.Convolution1DFlipout):
        inputs = tf.random.uniform([2, 3, 1], seed=1)
      elif layer_class in (tfp.layers.Convolution2DReparameterization,
                           tfp.layers.Convolution2DFlipout):
        inputs = tf.random.uniform([2, 3, 3, 1], seed=1)
      elif layer_class in (tfp.layers.Convolution3DReparameterization,
                           tfp.layers.Convolution3DFlipout):
        inputs = tf.random.uniform([2, 3, 3, 3, 1], seed=1)
      inputs = self.maybe_transpose_tensor(inputs)

      # No keys.
      input_dependent_losses = layer.get_losses_for(inputs=None)
      self.assertEqual(len(layer.losses), 0)
      self.assertListEqual(layer.losses, input_dependent_losses)

      _ = layer(inputs)

      # Yes keys.
      input_dependent_losses = layer.get_losses_for(inputs=None)
      self.assertEqual(len(layer.losses), 2)
      self.assertEqual(layer.losses[0].shape, ())
      self.assertEqual(layer.losses[1].shape, ())
      self.assertListEqual(layer.losses, input_dependent_losses)

  def _testConvSetUp(self, layer_class, batch_size, depth=None,
                     height=None, width=None, channels=None, filters=None,
                     **kwargs):  # pylint: disable=invalid-name
    seed = Counter()
    if layer_class in (tfp.layers.Convolution1DReparameterization,
                       tfp.layers.Convolution1DFlipout):
      inputs = tf.random.uniform([batch_size, width, channels], seed=seed())
      kernel_size = (2,)
    elif layer_class in (tfp.layers.Convolution2DReparameterization,
                         tfp.layers.Convolution2DFlipout):
      inputs = tf.random.uniform([batch_size, height, width, channels],
                                 seed=seed())
      kernel_size = (2, 2)
    elif layer_class in (tfp.layers.Convolution3DReparameterization,
                         tfp.layers.Convolution3DFlipout):
      inputs = tf.random.uniform([batch_size, depth, height, width, channels],
                                 seed=seed())
      kernel_size = (2, 2, 2)
    inputs = self.maybe_transpose_tensor(inputs)

    kernel_shape = kernel_size + (channels, filters)
    kernel_posterior = MockDistribution(
        loc=tf.random.uniform(kernel_shape, seed=seed()),
        scale=tf.random.uniform(kernel_shape, seed=seed()),
        result_log_prob=tf.random.uniform(kernel_shape, seed=seed()),
        result_sample=tf.random.uniform(kernel_shape, seed=seed()))
    kernel_prior = MockDistribution(
        result_log_prob=tf.random.uniform(kernel_shape, seed=seed()),
        result_sample=tf.random.uniform(kernel_shape, seed=seed()))
    kernel_divergence = MockKLDivergence(
        result=tf.random.uniform([], seed=seed()))

    bias_size = (filters,)
    bias_posterior = MockDistribution(
        result_log_prob=tf.random.uniform(bias_size, seed=seed()),
        result_sample=tf.random.uniform(bias_size, seed=seed()))
    bias_prior = MockDistribution(
        result_log_prob=tf.random.uniform(bias_size, seed=seed()),
        result_sample=tf.random.uniform(bias_size, seed=seed()))
    bias_divergence = MockKLDivergence(
        result=tf.random.uniform([], seed=seed()))

    tf.random.set_seed(5995)
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
        data_format=self.data_format,
        **kwargs)

    outputs = layer(inputs)

    kl_penalty = layer.get_losses_for(inputs=None)
    return (kernel_posterior, kernel_prior, kernel_divergence,
            bias_posterior, bias_prior, bias_divergence,
            layer, inputs, outputs, kl_penalty, kernel_shape)

  def _testConvReparameterization(self, layer_class):  # pylint: disable=invalid-name
    batch_size, depth, height, width, channels, filters = 2, 4, 4, 4, 3, 5
    with self.cached_session() as sess:
      (kernel_posterior, kernel_prior, kernel_divergence,
       bias_posterior, bias_prior, bias_divergence, layer, inputs,
       outputs, kl_penalty, kernel_shape) = self._testConvSetUp(
           layer_class, batch_size,
           depth=depth, height=height, width=width, channels=channels,
           filters=filters)

      convolution_op = nn_ops.Convolution(
          tf.TensorShape(inputs.shape),
          filter_shape=tf.TensorShape(kernel_shape),
          padding='SAME',
          data_format=tf_layers_util.convert_data_format(
              self.data_format, inputs.shape.rank))
      expected_outputs = convolution_op(inputs, kernel_posterior.result_sample)
      expected_outputs = tf.nn.bias_add(
          expected_outputs,
          bias_posterior.result_sample,
          data_format=tf_layers_util.convert_data_format(self.data_format, 4))

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

      self.assertAllClose(expected_kernel_, actual_kernel_, rtol=1e-6)
      self.assertAllClose(expected_bias_, actual_bias_, rtol=1e-6)
      self.assertAllClose(expected_outputs_, actual_outputs_, rtol=1e-6)
      self.assertAllClose(
          expected_kernel_divergence_, actual_kernel_divergence_, rtol=1e-6)
      self.assertAllClose(
          expected_bias_divergence_, actual_bias_divergence_, rtol=1e-6)

      expected_args = [kernel_posterior,
                       kernel_prior,
                       kernel_posterior.result_sample]
      # We expect that there was one call to kernel_divergence, with the above
      # args; MockKLDivergence appends the list of args to a list, so the above
      # args should be in the 0th position of that list.
      actual_args = kernel_divergence.args[0]
      # Test for identity with 'is'. TensorFlowTestCase.assertAllEqual actually
      # coerces the inputs to numpy arrays, so we can't use that to assert that
      # the arguments (which are a mixture of Distributions and Tensors) are
      # equal.
      for a, b in zip(expected_args, actual_args):
        self.assertIs(a, b)

      # Same story as above.
      expected_args = [bias_posterior, bias_prior, bias_posterior.result_sample]
      actual_args = bias_divergence.args[0]
      for a, b in zip(expected_args, actual_args):
        self.assertIs(a, b)

  def _testConvFlipout(self, layer_class):  # pylint: disable=invalid-name
    batch_size, depth, height, width, channels, filters = 2, 4, 4, 4, 3, 5
    with self.cached_session() as sess:
      (kernel_posterior, kernel_prior, kernel_divergence,
       bias_posterior, bias_prior, bias_divergence, layer, inputs,
       outputs, kl_penalty, kernel_shape) = self._testConvSetUp(
           layer_class, batch_size,
           depth=depth, height=height, width=width, channels=channels,
           filters=filters, seed=44)

      tf.random.set_seed(5995)

      convolution_op = nn_ops.Convolution(
          tf.TensorShape(inputs.shape),
          filter_shape=tf.TensorShape(kernel_shape),
          padding='SAME',
          data_format=tf_layers_util.convert_data_format(
              self.data_format, inputs.shape.rank))

      expected_kernel_posterior_affine = tfd.Normal(
          loc=tf.zeros_like(kernel_posterior.result_loc),
          scale=kernel_posterior.result_scale)
      expected_kernel_posterior_affine_tensor = (
          expected_kernel_posterior_affine.sample(seed=42))

      expected_outputs = convolution_op(
          inputs, kernel_posterior.distribution.loc)

      input_shape = tf.shape(inputs)
      batch_shape = tf.expand_dims(input_shape[0], 0)
      if self.data_format == 'channels_first':
        channels = input_shape[1]
      else:
        channels = input_shape[-1]
      rank = len(inputs.shape) - 2

      seed_stream = tfp.util.SeedStream(layer.seed, salt='ConvFlipout')

      sign_input = tf.random.uniform(
          tf.concat([batch_shape, tf.expand_dims(channels, 0)], 0),
          minval=0,
          maxval=2,
          dtype=tf.int64,
          seed=seed_stream())
      sign_input = tf.cast(2 * sign_input - 1, inputs.dtype)
      sign_output = tf.random.uniform(
          tf.concat([batch_shape, tf.expand_dims(filters, 0)], 0),
          minval=0,
          maxval=2,
          dtype=tf.int64,
          seed=seed_stream())
      sign_output = tf.cast(2 * sign_output - 1, inputs.dtype)

      if self.data_format == 'channels_first':
        for _ in range(rank):
          sign_input = tf.expand_dims(sign_input, -1)  # 2D ex: (B, C, 1, 1)
          sign_output = tf.expand_dims(sign_output, -1)
      else:
        for _ in range(rank):
          sign_input = tf.expand_dims(sign_input, 1)  # 2D ex: (B, 1, 1, C)
          sign_output = tf.expand_dims(sign_output, 1)

      perturbed_inputs = convolution_op(
          inputs * sign_input, expected_kernel_posterior_affine_tensor)
      perturbed_inputs *= sign_output

      expected_outputs += perturbed_inputs
      expected_outputs = tf.nn.bias_add(
          expected_outputs,
          bias_posterior.result_sample,
          data_format=tf_layers_util.convert_data_format(self.data_format, 4))

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

      self.assertAllClose(expected_bias_, actual_bias_, rtol=1e-6)
      self.assertAllClose(expected_outputs_, actual_outputs_, rtol=1e-6)
      self.assertAllClose(
          expected_kernel_divergence_, actual_kernel_divergence_, rtol=1e-6)
      self.assertAllClose(
          expected_bias_divergence_, actual_bias_divergence_, rtol=1e-6)

      expected_args = [kernel_posterior,
                       kernel_prior,
                       None]
      # We expect that there was one call to kernel_divergence, with the above
      # args; MockKLDivergence appends the list of args to a list, so the above
      # args should be in the 0th position of that list.
      actual_args = kernel_divergence.args[0]
      # Test for identity with 'is'. TensorFlowTestCase.assertAllEqual actually
      # coerces the inputs to numpy arrays, so we can't use that to assert that
      # the arguments (which are a mixture of Distributions and Tensors) are
      # equal.
      for a, b in zip(expected_args, actual_args):
        self.assertIs(a, b)

      # Same story as above.
      expected_args = [bias_posterior, bias_prior, bias_posterior.result_sample]
      actual_args = bias_divergence.args[0]
      for a, b in zip(expected_args, actual_args):
        self.assertIs(a, b)

  def _testRandomConvFlipout(self, layer_class):  # pylint: disable=invalid-name
    batch_size, depth, height, width, channels, filters = 2, 4, 4, 4, 3, 5
    with self.cached_session() as sess:
      seed = Counter()
      if layer_class in (tfp.layers.Convolution1DReparameterization,
                         tfp.layers.Convolution1DFlipout):
        inputs = tf.random.uniform([batch_size, width, channels], seed=seed())
        kernel_size = (2,)
      elif layer_class in (tfp.layers.Convolution2DReparameterization,
                           tfp.layers.Convolution2DFlipout):
        inputs = tf.random.uniform([batch_size, height, width, channels],
                                   seed=seed())
        kernel_size = (2, 2)
      elif layer_class in (tfp.layers.Convolution3DReparameterization,
                           tfp.layers.Convolution3DFlipout):
        inputs = tf.random.uniform([batch_size, depth, height, width, channels],
                                   seed=seed())
        kernel_size = (2, 2, 2)
      inputs = self.maybe_transpose_tensor(inputs)

      kernel_shape = kernel_size + (channels, filters)
      bias_size = (filters,)

      kernel_posterior = MockDistribution(
          loc=tf.random.uniform(kernel_shape, seed=seed()),
          scale=tf.random.uniform(kernel_shape, seed=seed()),
          result_log_prob=tf.random.uniform(kernel_shape, seed=seed()),
          result_sample=tf.random.uniform(kernel_shape, seed=seed()))
      bias_posterior = MockDistribution(
          loc=tf.random.uniform(bias_size, seed=seed()),
          scale=tf.random.uniform(bias_size, seed=seed()),
          result_log_prob=tf.random.uniform(bias_size, seed=seed()),
          result_sample=tf.random.uniform(bias_size, seed=seed()))
      layer_one = layer_class(
          filters=filters,
          kernel_size=kernel_size,
          padding='SAME',
          kernel_posterior_fn=lambda *args: kernel_posterior,
          kernel_posterior_tensor_fn=lambda d: d.sample(seed=42),
          kernel_divergence_fn=None,
          bias_posterior_fn=lambda *args: bias_posterior,
          bias_posterior_tensor_fn=lambda d: d.sample(seed=43),
          bias_divergence_fn=None,
          seed=44,
          data_format=self.data_format)
      layer_two = layer_class(
          filters=filters,
          kernel_size=kernel_size,
          padding='SAME',
          kernel_posterior_fn=lambda *args: kernel_posterior,
          kernel_posterior_tensor_fn=lambda d: d.sample(seed=42),
          kernel_divergence_fn=None,
          bias_posterior_fn=lambda *args: bias_posterior,
          bias_posterior_tensor_fn=lambda d: d.sample(seed=43),
          bias_divergence_fn=None,
          seed=45,
          data_format=self.data_format)

      outputs_one = layer_one(inputs)
      outputs_two = layer_two(inputs)

      outputs_one_, outputs_two_ = sess.run([
          outputs_one, outputs_two])

      self.assertLess(np.sum(np.isclose(outputs_one_, outputs_two_)),
                      np.prod(outputs_one_.shape))

  def _testLayerInSequential(self, layer_class):  # pylint: disable=invalid-name
    if layer_class in (tfp.layers.Convolution1DReparameterization,
                       tfp.layers.Convolution1DFlipout):
      inputs = tf.random.uniform([2, 3, 1])
      outputs = tf.random.uniform([2, 1, 2])
    elif layer_class in (tfp.layers.Convolution2DReparameterization,
                         tfp.layers.Convolution2DFlipout):
      inputs = tf.random.uniform([2, 3, 3, 1])
      outputs = tf.random.uniform([2, 1, 1, 2])
    elif layer_class in (tfp.layers.Convolution3DReparameterization,
                         tfp.layers.Convolution3DFlipout):
      inputs = tf.random.uniform([2, 3, 3, 3, 1])
      outputs = tf.random.uniform([2, 1, 1, 1, 2])
    inputs = self.maybe_transpose_tensor(inputs)
    outputs = self.maybe_transpose_tensor(outputs)

    net = tf.keras.Sequential([
        layer_class(filters=2, kernel_size=3, data_format=self.data_format),
        layer_class(filters=2, kernel_size=1, data_format=self.data_format)])

    net.compile(loss='mse', optimizer='adam')
    net.fit(inputs, outputs, batch_size=2, epochs=3, steps_per_epoch=2)

    batch_output = self.evaluate(net(inputs))
    self.assertAllEqual(outputs.shape, batch_output.shape)

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

  def testSequentialConvolution1DReparameterization(self):
    self._testLayerInSequential(tfp.layers.Convolution1DReparameterization)

  def testSequentialConvolution2DReparameterization(self):
    self._testLayerInSequential(tfp.layers.Convolution2DReparameterization)

  def testSequentialConvolution3DReparameterization(self):
    self._testLayerInSequential(tfp.layers.Convolution3DReparameterization)

  def testSequentialConvolution1DFlipout(self):
    self._testLayerInSequential(tfp.layers.Convolution1DFlipout)

  def testSequentialConvolution2DFlipout(self):
    self._testLayerInSequential(tfp.layers.Convolution2DFlipout)

  def testSequentialConvolution3DFlipout(self):
    self._testLayerInSequential(tfp.layers.Convolution3DFlipout)

  def testGradients(self):
    net = tf.keras.Sequential([
        tfp.layers.Convolution1DFlipout(1, 1, data_format=self.data_format),
        tfp.layers.Convolution1DReparameterization(
            1, 1, data_format=self.data_format),
    ])
    with tf.GradientTape() as tape:
      y = net(tf.zeros([1, 1, 1]))
    grads = tape.gradient(y, net.trainable_variables)
    self.assertLen(grads, 6)
    self.assertAllNotNone(grads)


@test_util.test_all_tf_execution_regimes
class ConvVariationalTestChannelsFirst(test_util.TestCase, ConvVariational):
  data_format = 'channels_first'


@test_util.test_all_tf_execution_regimes
class ConvVariationalTestChannelsLast(test_util.TestCase, ConvVariational):
  data_format = 'channels_last'

if __name__ == '__main__':
  tf.test.main()
