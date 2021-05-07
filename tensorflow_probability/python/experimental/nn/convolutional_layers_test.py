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
"""Tests for convolution_layers.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
# Dependency imports
import numpy as np
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp

from tensorflow_probability.python.internal import test_util


tfb = tfp.bijectors
tfd = tfp.distributions
tfn = tfp.experimental.nn


class BnnEndToEnd(object):

  def run_bnn_test(self, make_conv):
    # 1  Prepare Dataset

    train_size = 128
    batch_size = 2
    evidence_shape = [28, 28, 1]
    target_shape = [10]

    train_dataset = tf.data.Dataset.from_tensor_slices((
        tf.random.uniform([train_size] + evidence_shape,
                          maxval=1, dtype=tf.float32),
        tf.math.softmax(tf.random.normal([train_size] + target_shape)),
    ))
    train_dataset = tfn.util.tune_dataset(
        train_dataset,
        batch_size=batch_size,
        shuffle_size=int(train_size / 7))

    # 2  Specify Model

    scale = tfp.util.TransformedVariable(1., tfb.Softplus())
    bnn = tfn.Sequential([
        make_conv(  # [b, 14, 14, 32]
            evidence_shape[-1], 32, filter_shape=7, strides=2),
        tfn.util.flatten_rightmost(ndims=3),    # [b, 14 * 14 * 32]
        tfn.AffineVariationalReparameterization(
            14 * 14 * 32, np.prod(target_shape) - 1),   # [b, 9]
        lambda loc: tfb.SoftmaxCentered()(  # pylint: disable=g-long-lambda
            tfd.Independent(tfd.Normal(loc, scale),
                            reinterpreted_batch_ndims=1)),  # [b, 10]
    ], name='bayesian_autoencoder')

    self.evaluate([v.initializer for v in bnn.trainable_variables])

    # 3  Train.
    n = tf.cast(train_size, tf.float32)
    train_iter = iter(train_dataset)
    def loss_fn():
      x, y = next(train_iter)
      nll = -tf.reduce_mean(bnn(x).log_prob(y), axis=-1)
      kl = tfn.losses.compute_extra_loss(bnn) / n
      return nll + kl, (nll, kl)
    opt = tf.optimizers.Adam()
    fit_op = tfn.util.make_fit_op(loss_fn, opt, bnn.trainable_variables)
    for _ in range(2):
      loss, (nll, kl) = fit_op()  # pylint: disable=unused-variable


@test_util.test_all_tf_execution_regimes
class ConvolutionTest(test_util.TestCase):

  def test_works_correctly(self):
    pass

  def test_non_bayesian_conv(self):
    x = tfn.Convolution(
        1, 5, 5, strides=2, padding='same')(tf.random.normal([3, 28, 28, 1]))
    self.assertShapeEqual(np.zeros([3, 14, 14, 5]), x)

  def test_bayesian_conv(self):
    def sample(kernel_shape,
               bias_shape,
               kernel_initializer=None,  # pylint: disable=unused-argument
               bias_initializer=None,  # pylint: disable=unused-argument
               kernel_batch_ndims=0,  # pylint: disable=unused-argument
               bias_batch_ndims=0,  # pylint: disable=unused-argument
               dtype=tf.float32,  # pylint: disable=unused-argument
               kernel_prior_scale_scale=None,  # pylint: disable=unused-argument
               kernel_scale_init_value=None,  # pylint: disable=unused-argument
               bias_prior_scale=None,  # pylint: disable=unused-argument
               bias_scale_init_value=None,  # pylint: disable=unused-argument
               kernel_prior=None):  # pylint: disable=unused-argument
      k = tfd.Normal(0., 1.).sample(kernel_shape)
      b = tfd.Normal(0., 1.).sample(bias_shape)
      return k, b
    def tile_for_batch(x, batch_shape):
      x_shape = tf.shape(x)
      x_ndims = tf.rank(x)
      x = tf.reshape(x, shape=tf.concat([
          x_shape[:-1],
          tf.ones_like(batch_shape),
          x_shape[-1:],
          ], axis=0))
      return tf.tile(x, multiples=tf.pad(
          batch_shape,
          paddings=[[x_ndims - 1, 1]],
          constant_values=1))

    nn = tfn.Convolution(1, 5, 5, strides=2, padding='same', batch_shape=[10],
                         make_kernel_bias_fn=sample)
    x = nn(tile_for_batch(tf.random.normal([3, 28, 28, 1]), [10]))
    self.assertShapeEqual(np.zeros([3, 14, 14, 10, 5]), x)


@test_util.test_all_tf_execution_regimes
class ConvolutionVariationalReparameterizationTest(
    test_util.TestCase, BnnEndToEnd):

  def test_works_correctly(self):
    if not tf.executing_eagerly():
      self.skipTest('Skipping graph mode test since we simulate user behavior.')
    make_conv = functools.partial(
        tfn.ConvolutionVariationalReparameterization,
        rank=2,
        padding='same',
        kernel_initializer=tfn.initializers.he_uniform(),
        activation_fn=tf.nn.elu)
    self.run_bnn_test(make_conv)


@test_util.test_all_tf_execution_regimes
class ConvolutionVariationalFlipoutTest(test_util.TestCase, BnnEndToEnd):

  def test_works_correctly(self):
    if not tf.executing_eagerly():
      self.skipTest('Skipping graph mode test since we simulate user behavior.')
    make_conv = functools.partial(
        tfn.ConvolutionVariationalFlipout,
        rank=2,
        padding='same',
        kernel_initializer=tfn.initializers.he_uniform(),
        activation_fn=tf.nn.elu)
    self.run_bnn_test(make_conv)


if __name__ == '__main__':
  tf.test.main()
