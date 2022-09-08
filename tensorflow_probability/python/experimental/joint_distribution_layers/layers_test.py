# Copyright 2021 The TensorFlow Probability Authors.
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
"""Tests for layers."""

from absl.testing import parameterized
import tensorflow.compat.v2 as tf

from tensorflow_probability.python.bijectors import chain
from tensorflow_probability.python.bijectors import real_nvp
from tensorflow_probability.python.bijectors import scale
from tensorflow_probability.python.bijectors import shift
from tensorflow_probability.python.distributions import distribution
from tensorflow_probability.python.distributions import joint_distribution_coroutine as jdc
from tensorflow_probability.python.distributions import lognormal
from tensorflow_probability.python.experimental.joint_distribution_layers import layers as jdlayers
from tensorflow_probability.python.internal import custom_gradient
from tensorflow_probability.python.internal import prefer_static as ps
from tensorflow_probability.python.internal import test_util
from tensorflow_probability.python.math import gradient

Root = jdc.JointDistributionCoroutine.Root


@test_util.test_all_tf_execution_regimes
class _JDLayersTestBase(test_util.TestCase):

  @test_util.numpy_disable_gradient_test
  def testBijectorIntegration(self):

    @jdc.JointDistributionCoroutine
    def bijector_model():

      bijectors = []
      for i in range(3):
        bijector_net = yield Root(
            jdlayers.Sequential(
                jdlayers.Affine(5, 5),
                tf.tanh,
                jdlayers.Affine(10, 5),
                jdlayers.Lambda(lambda x: chain.Chain(  # pylint: disable=g-long-lambda
                    [
                        shift.Shift(x[..., :5]),
                        scale.Scale(log_scale=x[..., 5:])
                    ])),
                name=f'net{i}',
            ))

        bijectors.append(
            real_nvp.RealNVP(
                fraction_masked=0.5 * (-1)**i,
                bijector_fn=lambda x, _, bn=bijector_net: bn(x)))

      yield jdlayers.Lambda(lambda: chain.Chain(bijectors))

    *params, _ = bijector_model.sample(
        2, seed=test_util.test_seed(sampler_type='stateless'))

    def ldj_fn(params):
      bijector_fn = bijector_model.sample(
          value=params, seed=test_util.test_seed(sampler_type='stateless'))[-1]
      return bijector_fn().forward_log_det_jacobian(tf.ones([2, 10]), 1)

    ldj, (params_grad,) = gradient.value_and_gradient(ldj_fn, (params,))
    self.assertEqual([2], ldj.shape)
    self.assertAllAssertsNested(
        lambda x, g: self.assertTrue(x.shape[-1] == 0 or custom_gradient.  # pylint: disable=g-long-lambda
                                     is_valid_gradient(g)),
        params,
        params_grad)

  @parameterized.named_parameters(
      ('Affine', lambda dtype: jdlayers.Affine(4, 3, dtype=dtype)),
      (
          'Conv2D',
          lambda dtype: jdlayers.Conv2D(4, (3, 4), 3, dtype=dtype),
          True,
      ),
      ('Lambda', lambda dtype: jdlayers.Lambda(tf.nn.softplus, dtype=dtype)),
      (
          'SequentialAffine',
          lambda dtype: jdlayers.Sequential(  # pylint: disable=g-long-lambda
              jdlayers.Affine(2, 4, dtype=dtype),
              jdlayers.Lambda(tf.nn.softplus, dtype=dtype),
          ),
      ),
      (
          'SequentialConv',
          lambda dtype: jdlayers.Sequential(  # pylint: disable=g-long-lambda
              jdlayers.Conv2D(2, 3, 4, dtype=dtype),
              jdlayers.Lambda(tf.nn.softplus, dtype=dtype),
          ),
          True),
  )
  def testIsDistribution(self, dist_fn, has_conv=False):
    """Instantiates a layer distribution and exercises its methods."""
    dist = dist_fn(self.dtype)
    if has_conv and test_util.is_numpy_not_jax_mode():
      self.skipTest('tf.nn.conv not implemented in NumPy.')
    self.assertIsInstance(dist, distribution.Distribution)
    dtype = dist.dtype
    tf.nest.assert_same_structure(dtype, dist.batch_shape)
    tf.nest.assert_same_structure(dtype, dist.event_shape)
    tf.nest.assert_same_structure(dtype, dist.batch_shape_tensor())
    tf.nest.assert_same_structure(dtype, dist.event_shape_tensor())
    sample = dist.sample([3],
                         seed=test_util.test_seed(sampler_type='stateless'))
    tf.nest.assert_same_structure(dtype, sample)

    # Make sure we can use bijectors too.
    bijector = dist.experimental_default_event_space_bijector()
    unconstrained_sample = bijector.inverse(sample)
    unconstrained_sample = tf.nest.map_structure(lambda x: x + 0.,
                                                 unconstrained_sample)
    sample = bijector.forward(unconstrained_sample)

    lp = dist.log_prob(sample)
    expected_lp_shape = [3]
    self.assertEqual(expected_lp_shape, lp.shape)

  @parameterized.named_parameters(
      ('LayerBatch', [7], []),
      ('InputBatch', [], [7]),
      ('BothBatch', [7], [7]),
  )
  def testAffineBatching(self, layer_batch, input_batch):
    dist = jdlayers.Affine(4, 3, dtype=self.dtype)
    layer = dist.sample(
        layer_batch, seed=test_util.test_seed(sampler_type='stateless'))
    # Validate that we can map the layer.
    layer = tf.nest.map_structure(lambda x: x + 0., layer)
    x = tf.ones(input_batch + [3], dtype=self.dtype)
    y = layer(x)
    self.assertAllEqual(
        list(ps.broadcast_shape(layer_batch, input_batch)) + [4], y.shape)
    self.assertEqual(self.dtype, y.dtype)

  def testAffineCustomParamsDist(self):

    def params_model_fn(out_units, in_units, dtype):
      yield Root(
          lognormal.LogNormal(
              tf.zeros([out_units, in_units], dtype), 1., name='weights'))
      yield Root(
          lognormal.LogNormal(tf.zeros([out_units], dtype), 1., name='bias'))

    dist = jdlayers.Affine(
        5, 4, dtype=self.dtype, params_model_fn=params_model_fn)
    layer = dist.sample(seed=test_util.test_seed(sampler_type='stateless'))
    x = tf.ones([4], self.dtype)
    y = layer(x)
    self.assertAllEqual([5], y.shape)
    self.assertEqual(self.dtype, y.dtype)
    # Since the parameters are log-normal, the outputs will be positive for
    # positive inputs.
    self.assertAllTrue(y > 0.)

  @test_util.disable_test_for_backend(
      disable_numpy=True, reason='tf.nn.conv not implemented in NumPy.')
  @parameterized.named_parameters(
      ('LayerBatch', [7], []),
      ('InputBatch', [], [7]),
      ('BothBatch', [7], [7]),
  )
  def testConv2DBatching(self, layer_batch, input_batch):
    dist = jdlayers.Conv2D(4, (3, 3), 3, dtype=self.dtype)
    layer = dist.sample(
        layer_batch, seed=test_util.test_seed(sampler_type='stateless'))
    # Validate that we can map the layer.
    layer = tf.nest.map_structure(lambda x: x + 0., layer)
    x = tf.ones(input_batch + [9, 5, 5, 3], dtype=self.dtype)
    y = layer(x)
    self.assertAllEqual(
        list(ps.broadcast_shape(layer_batch, input_batch)) + [9, 5, 5, 4],
        y.shape)
    self.assertEqual(self.dtype, y.dtype)

  @test_util.disable_test_for_backend(
      disable_numpy=True, reason='tf.nn.conv not implemented in NumPy.')
  @parameterized.named_parameters(
      ('SameNoStrides', 'SAME', [1, 1], [7, 11]),
      ('ValidNoStrides', 'VALID', [1, 1], [5, 9]),
      ('SameWithStrides', 'SAME', [2, 2], [4, 6]),
      ('ValidWithStrides', 'VALID', [2, 2], [3, 5]),
  )
  def testConv2DParams(self, padding, strides, expected_output_size):
    dist = jdlayers.Conv2D(
        4, (3, 3), 3, strides=strides, padding=padding, dtype=self.dtype)
    layer = dist.sample(seed=test_util.test_seed(sampler_type='stateless'))
    # Validate that we can map the layer.
    layer = tf.nest.map_structure(lambda x: x + 0., layer)
    x = tf.ones([5, 7, 11, 3], dtype=self.dtype)
    y = layer(x)
    self.assertAllEqual([5] + expected_output_size + [4], y.shape)
    self.assertEqual(self.dtype, y.dtype)

  @test_util.disable_test_for_backend(
      disable_numpy=True, reason='tf.nn.conv not implemented in NumPy.')
  def testConvCustomParamsDist(self):

    def params_model_fn(out_channels, size, in_channels, dtype):
      yield Root(
          lognormal.LogNormal(
              tf.zeros(list(size) + [in_channels, out_channels], dtype),
              1.,
              name='kernel'))

    dist = jdlayers.Conv2D(
        5, (3, 3), 4, dtype=self.dtype, params_model_fn=params_model_fn)
    layer = dist.sample(seed=test_util.test_seed(sampler_type='stateless'))
    x = tf.ones([3, 5, 7, 4], self.dtype)
    y = layer(x)
    self.assertAllEqual([3, 5, 7, 5], y.shape)
    self.assertEqual(self.dtype, y.dtype)
    # Since the parameters are log-normal, the outputs will be positive for
    # positive inputs.
    self.assertAllTrue(y > 0.)

  def testLambda(self):
    dist = jdlayers.Lambda(tf.square)
    layer = dist.sample(seed=test_util.test_seed(sampler_type='stateless'))
    self.assertAllClose(4, layer(2))

  def testSequential(self):
    dist = jdlayers.Sequential(
        jdlayers.Affine(5, 3),
        tf.nn.softplus,
    )
    layer = dist.sample(seed=test_util.test_seed(sampler_type='stateless'))
    x = tf.ones([4, 3])
    y = layer(x)
    self.assertAllEqual([4, 5], y.shape)
    self.assertAllTrue(y > 0.)
    self.assertAllEqual([], dist.log_prob(layer).shape)


class JDLayersTest32(_JDLayersTestBase):
  dtype = tf.float32


class JDLayersTest64(_JDLayersTestBase):
  dtype = tf.float64


del _JDLayersTestBase

if __name__ == '__main__':
  test_util.main()
