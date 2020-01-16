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
"""Tests for the Disentangled Sequential Variational Autoencoder."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow.compat.v1 as tf

from tensorflow_probability.examples import disentangled_vae

from tensorflow.python.framework import test_util  # pylint: disable=g-direct-tensorflow-import


class _DisentangledVAETest(object):
  """Base test class."""

  def setUp(self):
    self.samples = 2
    self.batch_size = 3
    self.length = 4
    self.latent_size = 5
    self.dimensions = self.latent_size
    self.hidden_size = 6
    self.channels = 3

  def assertDistShape(self, dist, event_shape, batch_shape):
    self.assertEqual(dist.event_shape, event_shape)
    self.assertEqual(dist.batch_shape, batch_shape)


@test_util.run_all_in_graph_and_eager_modes
class DisentangledVAEComponentsTest(_DisentangledVAETest, tf.test.TestCase):
  """Test class for the individual model components."""

  def testLearnableMultivariateNormalDiagClass(self):
    dist_model = disentangled_vae.LearnableMultivariateNormalDiag(
        self.dimensions)
    dist = dist_model()
    self.assertDistShape(dist, (self.dimensions,), ())

  def testLearnableMultivariateNormalDiagCellClassNoBatch(self):
    prior = disentangled_vae.LearnableMultivariateNormalDiagCell(
        self.dimensions, self.hidden_size)

    # Zero state.
    dynamic_previous_output, state = prior.zero_state()
    self.assertEqual(dynamic_previous_output.shape, (self.dimensions,))
    for tensor in state:
      self.assertEqual(tensor.shape, (1, self.hidden_size))
    h0, c0 = state

    # First timestep.
    dist_z1, state_z1 = prior(dynamic_previous_output, state)
    self.assertDistShape(dist_z1, (self.dimensions,), ())
    for tensor in state_z1:
      self.assertEqual(tensor.shape, (1, self.hidden_size))
    if not tf.executing_eagerly():
      self.evaluate(tf.compat.v1.global_variables_initializer())
    h1, c1 = state_z1
    self.assertTrue(np.allclose(self.evaluate(h1), self.evaluate(h0)))
    self.assertTrue(np.allclose(self.evaluate(c1), self.evaluate(c0)))

    # Second timestep.
    dist_z2, state = prior(dist_z1.sample(), state_z1)
    self.assertDistShape(dist_z2, (self.dimensions,), ())
    for tensor in state:
      self.assertEqual(tensor.shape, (1, self.hidden_size))
    h2, c2 = state
    self.assertFalse(np.allclose(self.evaluate(h2), self.evaluate(h1)))
    self.assertFalse(np.allclose(self.evaluate(c2), self.evaluate(c1)))

    # Second timestep with sample shape.
    dist_z2, state = prior(dist_z1.sample(2), state_z1)
    self.assertDistShape(dist_z2, (self.dimensions,), (2))
    for tensor in state:
      self.assertEqual(tensor.shape, (2, self.hidden_size))

  def testLearnableMultivariateNormalDiagCellClassBatch(self):
    prior = disentangled_vae.LearnableMultivariateNormalDiagCell(
        self.dimensions, self.hidden_size)

    # Zero state with complex batch shape.
    dynamic_previous_output, state = prior.zero_state(
        (self.samples, self.batch_size))
    self.assertEqual(dynamic_previous_output.shape,
                     (self.samples, self.batch_size, self.dimensions))
    for tensor in state:
      self.assertEqual(tensor.shape, (1, self.hidden_size))
    h0, c0 = state

    # First timestep.
    dist_z1, state_z1 = prior(dynamic_previous_output, state)
    self.assertDistShape(dist_z1, (self.dimensions,),
                         (self.samples, self.batch_size))
    for tensor in state_z1:
      self.assertEqual(
          tensor.shape, (self.samples, self.batch_size, self.hidden_size))
    if not tf.executing_eagerly():
      self.evaluate(tf.compat.v1.global_variables_initializer())
    h1, c1 = state_z1
    self.assertTrue(np.allclose(self.evaluate(h1), self.evaluate(h0)))
    self.assertTrue(np.allclose(self.evaluate(c1), self.evaluate(c0)))

    # Second timestep.
    dist_z2, state = prior(dist_z1.sample(), state_z1)
    self.assertDistShape(dist_z2, (self.dimensions,),
                         (self.samples, self.batch_size))
    for tensor in state:
      self.assertEqual(
          tensor.shape, (self.samples, self.batch_size, self.hidden_size))
    h2, c2 = state
    self.assertFalse(np.allclose(self.evaluate(h2), self.evaluate(h1)))
    self.assertFalse(np.allclose(self.evaluate(c2), self.evaluate(c1)))

    # Second timestep with sample shape.
    dist_z2, state = prior(dist_z1.sample(2), state_z1)
    self.assertDistShape(dist_z2, (self.dimensions,),
                         (2, self.samples, self.batch_size))
    for tensor in state:
      self.assertEqual(
          tensor.shape, (2, self.samples, self.batch_size, self.hidden_size))

  def testDecoderClassNoSampleShape(self):
    decoder = disentangled_vae.Decoder(20, self.channels)

    z = tf.random.normal([self.batch_size, self.length, 10])
    f = tf.random.normal([self.batch_size, 10])
    dist = decoder((z, f))
    self.assertDistShape(dist, (64, 64, self.channels),
                         (self.batch_size, self.length))

  def testDecoderClassWithSampleShape(self):
    decoder = disentangled_vae.Decoder(20, self.channels)

    # Using sample shape.
    z = tf.random.normal([self.samples, self.batch_size, self.length, 10])
    f = tf.random.normal([self.samples, self.batch_size, 10])
    dist = decoder((z, f))
    self.assertDistShape(dist, (64, 64, self.channels),
                         (self.samples, self.batch_size, self.length))

  def testCompressorClassNoSampleShape(self):
    encoder = disentangled_vae.Compressor(self.hidden_size)

    xt = tf.random.normal([self.batch_size, self.length, 64, 64, 3])
    out = encoder(xt)
    self.assertEqual(out.shape,
                     (self.batch_size, self.length, self.hidden_size))

  def testCompressorClassWithSampleShape(self):
    encoder = disentangled_vae.Compressor(self.hidden_size)

    xt = tf.random.normal(
        [self.samples, self.batch_size, self.length, 64, 64, 3])
    out = encoder(xt)
    self.assertEqual(out.shape,
                     (self.samples, self.batch_size, self.length,
                      self.hidden_size))

  def testEncoderStaticClassNoSamples(self):
    encoder = disentangled_vae.EncoderStatic(self.latent_size, self.hidden_size)

    input_features = tf.random.normal(
        [self.batch_size, self.length, self.hidden_size])
    dist = encoder(input_features)
    self.assertDistShape(dist, (self.latent_size,), (self.batch_size,))

  def testEncoderStaticClassSamples(self):
    encoder = disentangled_vae.EncoderStatic(self.latent_size, self.hidden_size)

    input_features = tf.random.normal(
        [self.samples, self.batch_size, self.length, self.hidden_size])
    dist = encoder(input_features)
    self.assertDistShape(dist, (self.latent_size,),
                         (self.samples, self.batch_size))

  def testEncoderDynamicFactorizedClassNoSamples(self):
    encoder = disentangled_vae.EncoderDynamicFactorized(
        self.latent_size, self.hidden_size)

    input_features = tf.random.normal(
        [self.batch_size, self.length, self.hidden_size])
    dist = encoder(input_features)
    self.assertDistShape(dist, (self.latent_size,),
                         (self.batch_size, self.length))

  def testEncoderDynamicFactorizedClassSamples(self):
    encoder = disentangled_vae.EncoderDynamicFactorized(
        self.latent_size, self.hidden_size)

    input_features = tf.random.normal(
        [self.samples, self.batch_size, self.length, self.hidden_size])
    dist = encoder(input_features)
    self.assertDistShape(dist, (self.latent_size,),
                         (self.samples, self.batch_size, self.length))

  def testEncoderDynamicFullClassNoSamples(self):
    encoder = disentangled_vae.EncoderDynamicFull(
        self.latent_size, self.hidden_size)

    input_features = tf.random.normal(
        [self.batch_size, self.length, self.hidden_size])
    f = tf.random.normal([self.batch_size, self.latent_size])
    dist = encoder((input_features, f))
    self.assertDistShape(dist, (self.latent_size,),
                         (self.batch_size, self.length))

  def testEncoderDynamicFullClassStaticSamples(self):
    encoder = disentangled_vae.EncoderDynamicFull(
        self.latent_size, self.hidden_size)

    input_features = tf.random.normal(
        [self.batch_size, self.length, self.hidden_size])
    f = tf.random.normal([self.samples, self.batch_size, self.latent_size])
    dist = encoder((input_features, f))
    self.assertDistShape(dist, (self.latent_size,),
                         (self.samples, self.batch_size, self.length))

  def testEncoderDynamicFullClassInputSamples(self):
    encoder = disentangled_vae.EncoderDynamicFull(
        self.latent_size, self.hidden_size)

    input_features = tf.random.normal(
        [self.samples, self.batch_size, self.length, self.hidden_size])
    f = tf.random.normal([self.batch_size, self.latent_size])
    dist = encoder((input_features, f))
    self.assertDistShape(dist, (self.latent_size,),
                         (self.samples, self.batch_size, self.length))

  def testEncoderDynamicFullClassBothSamples(self):
    encoder = disentangled_vae.EncoderDynamicFull(
        self.latent_size, self.hidden_size)

    input_features = tf.random.normal(
        [self.samples, self.batch_size, self.length, self.hidden_size])
    f = tf.random.normal([self.samples, self.batch_size, self.latent_size])
    dist = encoder((input_features, f))
    self.assertDistShape(dist, (self.latent_size,),
                         (self.samples, self.batch_size, self.length))

  def testEncoderDynamicFullClassComplexStaticSampleShape(self):
    encoder = disentangled_vae.EncoderDynamicFull(
        self.latent_size, self.hidden_size)

    input_features = tf.random.normal(
        [self.batch_size, self.length, self.hidden_size])
    f = tf.random.normal(
        [self.samples * 2, self.samples, self.batch_size, self.latent_size])
    dist = encoder((input_features, f))
    self.assertDistShape(
        dist, (self.latent_size,),
        (self.samples*2, self.samples, self.batch_size, self.length))


@test_util.run_all_in_graph_and_eager_modes
class DisentangledSequentialVAETest(_DisentangledVAETest, tf.test.TestCase):
  """Test class for the DisentangledSequentialVAE model."""

  def setUp(self):
    super(DisentangledSequentialVAETest, self).setUp()
    self.latent_size_static = 10
    self.latent_size_dynamic = 11
    self.model_factorized = disentangled_vae.DisentangledSequentialVAE(
        self.latent_size_static, self.latent_size_dynamic, self.hidden_size,
        self.channels, "factorized")
    self.model_full = disentangled_vae.DisentangledSequentialVAE(
        self.latent_size_static, self.latent_size_dynamic, self.hidden_size,
        self.channels, "full")
    self.inputs = tf.random.normal(
        [self.batch_size, self.length, 64, 64, self.channels])

  def testGenerateFactorizedFixStatic(self):
    dist = self.model_factorized.generate(self.batch_size, self.length,
                                          samples=self.samples, fix_static=True)
    self.assertDistShape(dist, (64, 64, self.channels),
                         (self.samples, self.batch_size, self.length))

  def testGenerateFullFixStatic(self):
    dist = self.model_full.generate(self.batch_size, self.length,
                                    samples=self.samples, fix_static=True)
    self.assertDistShape(dist, (64, 64, self.channels),
                         (self.samples, self.batch_size, self.length))

  def testGenerateFactorizedFixDynamic(self):
    dist = self.model_factorized.generate(self.batch_size, self.length,
                                          samples=self.samples,
                                          fix_dynamic=True)
    self.assertDistShape(dist, (64, 64, self.channels),
                         (self.samples, self.batch_size, self.length))

  def testGenerateFullFixDynamic(self):
    dist = self.model_full.generate(self.batch_size, self.length,
                                    samples=self.samples, fix_dynamic=True)
    self.assertDistShape(dist, (64, 64, self.channels),
                         (self.samples, self.batch_size, self.length))

  def testGenerateFactorizedFixBoth(self):
    dist = self.model_factorized.generate(self.batch_size, self.length,
                                          samples=self.samples, fix_static=True,
                                          fix_dynamic=True)
    self.assertDistShape(dist, (64, 64, self.channels),
                         (self.samples, self.batch_size, self.length))

  def testGenerateFullFixBoth(self):
    dist = self.model_full.generate(self.batch_size, self.length,
                                    samples=self.samples, fix_static=True,
                                    fix_dynamic=True)
    self.assertDistShape(dist, (64, 64, self.channels),
                         (self.samples, self.batch_size, self.length))

  def testReconstructFactorized(self):
    dist = self.model_factorized.reconstruct(self.inputs, self.samples)
    self.assertDistShape(dist, (64, 64, self.channels),
                         (self.samples, self.batch_size, self.length))

  def testReconstructFull(self):
    dist = self.model_full.reconstruct(self.inputs, self.samples)
    self.assertDistShape(dist, (64, 64, self.channels),
                         (self.samples, self.batch_size, self.length))

  def testReconstructFactorizedSampleStatic(self):
    dist = self.model_factorized.reconstruct(self.inputs, self.samples,
                                             sample_static=True)
    self.assertDistShape(dist, (64, 64, self.channels),
                         (self.samples, self.batch_size, self.length))

  def testReconstructFullSampleStatic(self):
    dist = self.model_full.reconstruct(self.inputs, self.samples,
                                       sample_static=True)
    self.assertDistShape(dist, (64, 64, self.channels),
                         (self.samples, self.batch_size, self.length))

  def testReconstructFactorizedSampleStaticFixed(self):
    dist = self.model_factorized.reconstruct(self.inputs, self.samples,
                                             sample_static=True,
                                             fix_static=True)
    self.assertDistShape(dist, (64, 64, self.channels),
                         (self.samples, self.batch_size, self.length))

  def testReconstructFullSampleStaticFixed(self):
    dist = self.model_full.reconstruct(self.inputs, self.samples,
                                       sample_static=True, fix_static=True)
    self.assertDistShape(dist, (64, 64, self.channels),
                         (self.samples, self.batch_size, self.length))

  def testReconstructFactorizedSampleDynamic(self):
    dist = self.model_factorized.reconstruct(self.inputs, self.samples,
                                             sample_dynamic=True)
    self.assertDistShape(dist, (64, 64, self.channels),
                         (self.samples, self.batch_size, self.length))

  def testReconstructFullSampleDynamic(self):
    dist = self.model_full.reconstruct(self.inputs, self.samples,
                                       sample_dynamic=True)
    self.assertDistShape(dist, (64, 64, self.channels),
                         (self.samples, self.batch_size, self.length))

  def testReconstructFactorizedSampleDynamicFixed(self):
    dist = self.model_factorized.reconstruct(self.inputs, self.samples,
                                             sample_dynamic=True,
                                             fix_dynamic=True)
    self.assertDistShape(dist, (64, 64, self.channels),
                         (self.samples, self.batch_size, self.length))

  def testReconstructFullSampleDynamicFixed(self):
    dist = self.model_full.reconstruct(self.inputs, self.samples,
                                       sample_dynamic=True, fix_dynamic=True)
    self.assertDistShape(dist, (64, 64, self.channels),
                         (self.samples, self.batch_size, self.length))

  def testReconstructFactorizedSampleBoth(self):
    dist = self.model_factorized.reconstruct(self.inputs, self.samples,
                                             sample_static=True,
                                             sample_dynamic=True)
    self.assertDistShape(dist, (64, 64, self.channels),
                         (self.samples, self.batch_size, self.length))

  def testReconstructFullSampleBoth(self):
    dist = self.model_full.reconstruct(self.inputs, self.samples,
                                       sample_static=True, sample_dynamic=True)
    self.assertDistShape(dist, (64, 64, self.channels),
                         (self.samples, self.batch_size, self.length))

  def testReconstructFactorizedSwapStatic(self):
    dist = self.model_factorized.reconstruct(self.inputs, self.samples,
                                             swap_static=True)
    self.assertDistShape(dist, (64, 64, self.channels),
                         (self.samples, self.batch_size, self.length))

  def testReconstructFullSwapStatic(self):
    dist = self.model_full.reconstruct(self.inputs, self.samples,
                                       swap_static=True)
    self.assertDistShape(dist, (64, 64, self.channels),
                         (self.samples, self.batch_size, self.length))

  def testReconstructFactorizedSwapDynamic(self):
    dist = self.model_factorized.reconstruct(self.inputs, self.samples,
                                             swap_dynamic=True)
    self.assertDistShape(dist, (64, 64, self.channels),
                         (self.samples, self.batch_size, self.length))

  def testReconstructFullSwapDynamic(self):
    dist = self.model_full.reconstruct(self.inputs, self.samples,
                                       swap_dynamic=True)
    self.assertDistShape(dist, (64, 64, self.channels),
                         (self.samples, self.batch_size, self.length))

  def testSampleStaticPriorFactorized(self):
    sample, dist = self.model_factorized.sample_static_prior(self.samples,
                                                             self.batch_size)
    self.assertEqual(sample.shape, (self.samples, self.batch_size,
                                    self.latent_size_static))
    self.assertDistShape(dist, (self.latent_size_static,), ())

  def testSampleStaticPriorFull(self):
    sample, dist = self.model_full.sample_static_prior(self.samples,
                                                       self.batch_size)
    self.assertEqual(sample.shape, (self.samples, self.batch_size,
                                    self.latent_size_static))
    self.assertDistShape(dist, (self.latent_size_static,), ())

  def testSampleStaticPriorFactorizedFixed(self):
    sample, dist = self.model_factorized.sample_static_prior(
        self.samples, self.batch_size, fixed=True)
    self.assertEqual(sample.shape, (self.samples, self.batch_size,
                                    self.latent_size_static))
    self.assertDistShape(dist, (self.latent_size_static,), ())

  def testSampleStaticPriorFullFixed(self):
    sample, dist = self.model_full.sample_static_prior(
        self.samples, self.batch_size, fixed=True)
    self.assertEqual(sample.shape, (self.samples, self.batch_size,
                                    self.latent_size_static))
    self.assertDistShape(dist, (self.latent_size_static,), ())

  def testSampleStaticPosteriorFactorized(self):
    features = self.model_factorized.compressor(self.inputs)
    sample, dist = self.model_factorized.sample_static_posterior(features,
                                                                 self.samples)
    self.assertEqual(sample.shape, (self.samples, self.batch_size,
                                    self.latent_size_static))
    self.assertDistShape(dist, (self.latent_size_static,), (self.batch_size,))

  def testSampleStaticPosteriorFull(self):
    features = self.model_full.compressor(self.inputs)
    sample, dist = self.model_full.sample_static_posterior(features,
                                                           self.samples)
    self.assertEqual(sample.shape, (self.samples, self.batch_size,
                                    self.latent_size_static))
    self.assertDistShape(dist, (self.latent_size_static,), (self.batch_size,))

  def testSampleDynamicPriorFactorized(self):
    sample, dist = self.model_factorized.sample_dynamic_prior(
        self.samples, self.batch_size, self.length)
    self.assertEqual(sample.shape, (self.samples, self.batch_size, self.length,
                                    self.latent_size_dynamic))
    self.assertDistShape(dist, (self.latent_size_dynamic,),
                         (self.samples, self.batch_size, self.length))

  def testSampleDynamicPriorFull(self):
    sample, dist = self.model_full.sample_dynamic_prior(
        self.samples, self.batch_size, self.length)
    self.assertEqual(sample.shape, (self.samples, self.batch_size, self.length,
                                    self.latent_size_dynamic))
    self.assertDistShape(dist, (self.latent_size_dynamic,),
                         (self.samples, self.batch_size, self.length))

  def testSampleDynamicPriorFactorizedFixed(self):
    sample, dist = self.model_factorized.sample_dynamic_prior(
        self.samples, self.batch_size, self.length, fixed=True)
    self.assertEqual(sample.shape, (self.samples, self.batch_size, self.length,
                                    self.latent_size_dynamic))
    self.assertDistShape(dist, (self.latent_size_dynamic,),
                         (self.samples, 1, self.length))

  def testSampleDynamicPriorFullFixed(self):
    sample, dist = self.model_full.sample_dynamic_prior(
        self.samples, self.batch_size, self.length, fixed=True)
    self.assertEqual(sample.shape, (self.samples, self.batch_size, self.length,
                                    self.latent_size_dynamic))
    self.assertDistShape(dist, (self.latent_size_dynamic,),
                         (self.samples, 1, self.length))

  def testSampleDynamicPosteriorFactorized(self):
    features = self.model_factorized.compressor(self.inputs)
    sample, dist = self.model_factorized.sample_dynamic_posterior(
        features, self.samples)
    self.assertEqual(sample.shape, (self.samples, self.batch_size, self.length,
                                    self.latent_size_dynamic))
    self.assertDistShape(dist, (self.latent_size_dynamic,),
                         (self.batch_size, self.length))

  def testSampleDynamicPosteriorFull(self):
    features = self.model_full.compressor(self.inputs)
    static_sample, dist = self.model_full.sample_static_posterior(features,
                                                                  self.samples)
    sample, dist = self.model_full.sample_dynamic_posterior(
        features, self.samples, static_sample)
    self.assertEqual(sample.shape, (self.samples, self.batch_size, self.length,
                                    self.latent_size_dynamic))
    self.assertDistShape(dist, (self.latent_size_dynamic,),
                         (self.samples, self.batch_size, self.length))


if __name__ == "__main__":
  tf.test.main()
