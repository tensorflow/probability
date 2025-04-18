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
"""Tests for dense variational layers."""
# pylint: disable=g-import-not-at-top
try:
  from keras.testing_infra import test_utils as keras_test_utils
except ImportError:
  keras_test_utils = None
import numpy as np

import tensorflow.compat.v2 as tf
from tensorflow_probability.python.distributions import independent
from tensorflow_probability.python.distributions import normal
from tensorflow_probability.python.internal import test_util
from tensorflow_probability.python.internal import tf_keras
from tensorflow_probability.python.layers import dense_variational
from tensorflow_probability.python.layers import util
from tensorflow_probability.python.random import random_ops
from tensorflow_probability.python.util import seed_stream


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


class MockDistribution(independent.Independent):
  """Monitors layer calls to the underlying distribution."""

  def __init__(self, result_sample, result_log_prob, loc=None, scale=None):
    self.result_sample = result_sample
    self.result_log_prob = result_log_prob
    self.result_loc = loc
    self.result_scale = scale
    self.result_distribution = normal.Normal(loc=0.0, scale=1.0)
    if loc is not None and scale is not None:
      self.result_distribution = normal.Normal(loc=self.result_loc,
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
class DenseVariational(test_util.TestCase):

  def _testKerasLayer(self, layer_class):
    def kernel_posterior_fn(dtype, shape, name, trainable, add_variable_fn):
      """Set trivially. The function is required to instantiate layer."""
      del name, trainable, add_variable_fn  # unused
      # Deserialized Keras objects do not perform lexical scoping. Any modules
      # that the function requires must be imported within the function.
      import tensorflow.compat.v2 as tf  # pylint: disable=g-import-not-at-top,redefined-outer-name,reimported
      from tensorflow_probability.python.distributions import independent  # pylint: disable=g-import-not-at-top,redefined-outer-name,reimported
      from tensorflow_probability.python.distributions import normal  # pylint: disable=g-import-not-at-top,redefined-outer-name,reimported

      dist = normal.Normal(
          loc=tf.zeros(shape, dtype), scale=tf.ones(shape, dtype))
      batch_ndims = tf.size(dist.batch_shape_tensor())
      return independent.Independent(
          dist, reinterpreted_batch_ndims=batch_ndims)

    kwargs = {'units': 3,
              'kernel_posterior_fn': kernel_posterior_fn,
              'kernel_prior_fn': None,
              'bias_posterior_fn': None,
              'bias_prior_fn': None}
    with tf_keras.utils.CustomObjectScope({layer_class.__name__: layer_class}):
      # TODO(scottzhu): reenable the test when the repo switch change reach
      # the TF PIP package.
      self.skipTest('Skip the test until the TF and Keras has a new PIP.')
      with self.cached_session():
        keras_test_utils.layer_test(
            layer_class,
            kwargs=kwargs,
            input_shape=(3, 2))
        keras_test_utils.layer_test(
            layer_class,
            kwargs=kwargs,
            input_shape=(None, None, 2))

  def _testKLPenaltyKernel(self, layer_class):
    with self.cached_session():
      layer = layer_class(units=2)
      inputs = tf.random.uniform([2, 3], seed=1)

      # No keys.
      self.assertEqual(len(layer.losses), 0)

      _ = layer(inputs)

      # Yes keys.
      self.assertEqual(len(layer.losses), 1)
      self.assertEqual(layer.losses[0].shape, ())

  def _testKLPenaltyBoth(self, layer_class):
    with self.cached_session():
      layer = layer_class(
          units=2,
          bias_posterior_fn=util.default_mean_field_normal_fn(),
          bias_prior_fn=util.default_multivariate_normal_fn)
      inputs = tf.random.uniform([2, 3], seed=1)

      # No keys.
      self.assertEqual(len(layer.losses), 0)

      _ = layer(inputs)

      # Yes keys.
      self.assertEqual(len(layer.losses), 2)
      self.assertEqual(layer.losses[0].shape, ())
      self.assertEqual(layer.losses[1].shape, ())

  def _testDenseSetUp(self, layer_class, batch_size, in_size, out_size,
                      **kwargs):
    seed = Counter()
    inputs = tf.random.uniform([batch_size, in_size], seed=seed())

    kernel_size = [in_size, out_size]
    kernel_posterior = MockDistribution(
        loc=tf.random.uniform(kernel_size, seed=seed()),
        scale=tf.random.uniform(kernel_size, seed=seed()),
        result_log_prob=tf.random.uniform(kernel_size, seed=seed()),
        result_sample=tf.random.uniform(kernel_size, seed=seed()))
    kernel_prior = MockDistribution(
        result_log_prob=tf.random.uniform(kernel_size, seed=seed()),
        result_sample=tf.random.uniform(kernel_size, seed=seed()))
    kernel_divergence = MockKLDivergence(
        result=tf.random.uniform([], seed=seed()))

    bias_size = [out_size]
    bias_posterior = MockDistribution(
        result_log_prob=tf.random.uniform(bias_size, seed=seed()),
        result_sample=tf.random.uniform(bias_size, seed=seed()))
    bias_prior = MockDistribution(
        result_log_prob=tf.random.uniform(bias_size, seed=seed()),
        result_sample=tf.random.uniform(bias_size, seed=seed()))
    bias_divergence = MockKLDivergence(
        result=tf.random.uniform([], seed=seed()))

    layer = layer_class(
        units=out_size,
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

    kl_penalty = layer.losses
    return (kernel_posterior, kernel_prior, kernel_divergence,
            bias_posterior, bias_prior, bias_divergence,
            layer, inputs, outputs, kl_penalty)

  def testKerasLayerReparameterization(self):
    self._testKerasLayer(dense_variational.DenseReparameterization)

  def testKerasLayerLocalReparameterization(self):
    self._testKerasLayer(dense_variational.DenseLocalReparameterization)

  def testKerasLayerFlipout(self):
    self._testKerasLayer(dense_variational.DenseFlipout)

  def testKLPenaltyKernelReparameterization(self):
    self._testKLPenaltyKernel(dense_variational.DenseReparameterization)

  def testKLPenaltyKernelLocalReparameterization(self):
    self._testKLPenaltyKernel(dense_variational.DenseLocalReparameterization)

  def testKLPenaltyKernelFlipout(self):
    self._testKLPenaltyKernel(dense_variational.DenseFlipout)

  def testKLPenaltyBothReparameterization(self):
    self._testKLPenaltyBoth(dense_variational.DenseReparameterization)

  def testKLPenaltyBothLocalReparameterization(self):
    self._testKLPenaltyBoth(dense_variational.DenseLocalReparameterization)

  def testKLPenaltyBothFlipout(self):
    self._testKLPenaltyBoth(dense_variational.DenseFlipout)

  def testDenseReparameterization(self):
    batch_size, in_size, out_size = 2, 3, 4
    with self.cached_session() as sess:
      (kernel_posterior, kernel_prior, kernel_divergence,
       bias_posterior, bias_prior, bias_divergence, layer, inputs,
       outputs, kl_penalty) = self._testDenseSetUp(
           dense_variational.DenseReparameterization,
           batch_size, in_size, out_size)

      expected_outputs = (
          tf.matmul(inputs, kernel_posterior.result_sample) +
          bias_posterior.result_sample)

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

  def testDenseLocalReparameterization(self):
    batch_size, in_size, out_size = 2, 3, 4
    with self.cached_session() as sess:
      tf.random.set_seed(9068)
      (kernel_posterior, kernel_prior, kernel_divergence,
       bias_posterior, bias_prior, bias_divergence, layer, inputs,
       outputs, kl_penalty) = self._testDenseSetUp(
           dense_variational.DenseLocalReparameterization,
           batch_size, in_size, out_size)

      tf.random.set_seed(9068)
      expected_kernel_posterior_affine = normal.Normal(
          loc=tf.matmul(inputs, kernel_posterior.result_loc),
          scale=tf.matmul(
              inputs**2., kernel_posterior.result_scale**2)**0.5)
      expected_kernel_posterior_affine_tensor = (
          expected_kernel_posterior_affine.sample(seed=42))
      expected_outputs = (expected_kernel_posterior_affine_tensor +
                          bias_posterior.result_sample)

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

  def testDenseFlipout(self):
    batch_size, in_size, out_size = 2, 3, 4
    with self.cached_session() as sess:
      tf.random.set_seed(9069)
      (kernel_posterior, kernel_prior, kernel_divergence,
       bias_posterior, bias_prior, bias_divergence, layer, inputs,
       outputs, kl_penalty) = self._testDenseSetUp(
           dense_variational.DenseFlipout,
           batch_size, in_size, out_size, seed=44)

      tf.random.set_seed(9069)
      expected_kernel_posterior_affine = normal.Normal(
          loc=tf.zeros_like(kernel_posterior.result_loc),
          scale=kernel_posterior.result_scale)
      expected_kernel_posterior_affine_tensor = (
          expected_kernel_posterior_affine.sample(seed=42))

      seed = seed_stream.SeedStream(layer.seed, salt='DenseFlipout')

      sign_input = random_ops.rademacher(
          [batch_size, in_size],
          dtype=inputs.dtype,
          seed=seed())
      sign_output = random_ops.rademacher(
          [batch_size, out_size],
          dtype=inputs.dtype,
          seed=seed())
      perturbed_inputs = tf.matmul(
          inputs * sign_input, expected_kernel_posterior_affine_tensor)
      perturbed_inputs *= sign_output

      expected_outputs = tf.matmul(inputs, kernel_posterior.result_loc)
      expected_outputs += perturbed_inputs
      expected_outputs += bias_posterior.result_sample

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

  def testRandomDenseFlipout(self):
    batch_size, in_size, out_size = 2, 3, 4
    with self.cached_session() as sess:
      seed = Counter()
      inputs = tf.random.uniform([batch_size, in_size], seed=seed())

      kernel_posterior = MockDistribution(
          loc=tf.random.uniform([in_size, out_size], seed=seed()),
          scale=tf.random.uniform([in_size, out_size], seed=seed()),
          result_log_prob=tf.random.uniform([in_size, out_size], seed=seed()),
          result_sample=tf.random.uniform([in_size, out_size], seed=seed()))
      bias_posterior = MockDistribution(
          loc=tf.random.uniform([out_size], seed=seed()),
          scale=tf.random.uniform([out_size], seed=seed()),
          result_log_prob=tf.random.uniform([out_size], seed=seed()),
          result_sample=tf.random.uniform([out_size], seed=seed()))
      layer_one = dense_variational.DenseFlipout(
          units=out_size,
          kernel_posterior_fn=lambda *args: kernel_posterior,
          kernel_posterior_tensor_fn=lambda d: d.sample(seed=42),
          kernel_divergence_fn=None,
          bias_posterior_fn=lambda *args: bias_posterior,
          bias_posterior_tensor_fn=lambda d: d.sample(seed=43),
          bias_divergence_fn=None,
          seed=44)
      layer_two = dense_variational.DenseFlipout(
          units=out_size,
          kernel_posterior_fn=lambda *args: kernel_posterior,
          kernel_posterior_tensor_fn=lambda d: d.sample(seed=42),
          kernel_divergence_fn=None,
          bias_posterior_fn=lambda *args: bias_posterior,
          bias_posterior_tensor_fn=lambda d: d.sample(seed=43),
          bias_divergence_fn=None,
          seed=45)

      outputs_one = layer_one(inputs)
      outputs_two = layer_two(inputs)

      outputs_one_, outputs_two_ = sess.run([
          outputs_one, outputs_two])

      self.assertLess(np.sum(np.isclose(outputs_one_, outputs_two_)), out_size)

  def testDenseLayersInSequential(self):
    data_size, batch_size, in_size, out_size = 10, 2, 3, 4
    x = np.random.uniform(-1., 1., size=(data_size, in_size)).astype(np.float32)
    y = np.random.uniform(
        -1., 1., size=(data_size, out_size)).astype(np.float32)

    model = tf_keras.Sequential([
        dense_variational.DenseReparameterization(6, activation=tf.nn.relu),
        dense_variational.DenseFlipout(6, activation=tf.nn.relu),
        dense_variational.DenseLocalReparameterization(out_size)
    ])

    model.compile(loss='mse', optimizer='adam')
    model.fit(x, y, batch_size=batch_size, epochs=3)

    batch_input = tf.random.uniform([batch_size, in_size])
    batch_output = self.evaluate(model(batch_input))
    self.assertAllEqual(batch_output.shape, [batch_size, out_size])

  def testGradients(self):
    net = tf_keras.Sequential([
        dense_variational.DenseReparameterization(1),
        dense_variational.DenseFlipout(1),
        dense_variational.DenseLocalReparameterization(1)
    ])
    with tf.GradientTape() as tape:
      y = net(tf.zeros([1, 1]))
    grads = tape.gradient(y, net.trainable_variables)
    self.assertLen(grads, 9)
    self.assertAllNotNone(grads)

if __name__ == '__main__':
  test_util.main()
