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
"""Tests for convolutional_transpose_layers.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
# Dependency imports
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp

from tensorflow_probability.python.internal import test_util


tfb = tfp.bijectors
tfd = tfp.distributions
tfn = tfp.experimental.nn


class BnnEndToEnd(object):

  def run_bnn_test(self, make_conv, make_deconv):
    # 1  Prepare Dataset

    train_size = 128
    batch_size = 2
    train_dataset = tf.data.Dataset.from_tensor_slices(
        tf.random.uniform([train_size, 28, 28, 1],
                          maxval=1,
                          dtype=tf.float32))
    train_dataset = tfn.util.tune_dataset(
        train_dataset,
        batch_size=batch_size,
        shuffle_size=int(train_size / 7))
    train_iter = iter(train_dataset)
    x = next(train_iter)
    input_channels = int(x.shape[-1])

    # 2  Specify Model

    bottleneck_size = 2

    scale = tfp.util.TransformedVariable(1., tfb.Softplus())

    bnn = tfn.Sequential([
        make_conv(input_channels, 32, filter_shape=5,
                  strides=2),                                # [b, 14, 14, 32]
        tfn.util.flatten_rightmost(ndims=3),                 # [b, 14 * 14 * 32]
        tfn.AffineVariationalReparameterization(
            14 * 14 * 32, bottleneck_size),                  # [b, 2]
        lambda x: x[..., tf.newaxis, tf.newaxis, :],         # [b, 1, 1, 2]
        make_deconv(2, 64, filter_shape=7, strides=1,
                    padding='valid'),                        # [b, 7, 7, 64]
        make_deconv(64, 32, filter_shape=4, strides=4),      # [2, 28, 28, 32]
        make_conv(32, 1, filter_shape=2, strides=1),         # [2, 28, 28, 1]
        tfn.Lambda(eval_fn=lambda loc: tfd.Independent(  # pylint: disable=g-long-lambda
            tfb.Sigmoid()(tfd.Normal(loc, scale)),
            reinterpreted_batch_ndims=3), also_track=scale),  # [b, 28, 28, 1]
    ], name='bayesian_autoencoder')

    # 3  Train.

    def loss_fn():
      x = next(train_iter)
      nll = -tf.reduce_mean(bnn(x).log_prob(x), axis=-1)
      kl = bnn.extra_loss / tf.cast(train_size, tf.float32)
      loss = nll + kl
      return loss, (nll, kl)
    opt = tf.optimizers.Adam()
    fit_op = tfn.util.make_fit_op(loss_fn, opt, bnn.trainable_variables)
    for _ in range(2):
      loss, (nll, kl) = fit_op()  # pylint: disable=unused-variable


@test_util.test_all_tf_execution_regimes
class ConvolutionTransposeTest(test_util.TestCase):

  def test_works_correctly(self):
    pass


@test_util.test_all_tf_execution_regimes
class ConvolutionTransposeVariationalReparameterizationTest(
    test_util.TestCase, BnnEndToEnd):

  def test_works_correctly(self):
    if not tf.executing_eagerly():
      self.skipTest('Skipping graph mode test since we simulate user behavior.')
    make_conv = functools.partial(
        tfn.ConvolutionVariationalReparameterization,
        rank=2,
        padding='same',
        filter_shape=5,
        init_kernel_fn=tfn.initializers.he_uniform(),
        activation_fn=tf.nn.elu)
    make_deconv = functools.partial(
        tfn.ConvolutionTransposeVariationalReparameterization,
        rank=2,
        padding='same',
        filter_shape=5,
        init_kernel_fn=tfn.initializers.he_uniform(),
        activation_fn=tf.nn.elu)
    self.run_bnn_test(make_conv, make_deconv)


@test_util.test_all_tf_execution_regimes
class ConvolutionTransposeVariationalFlipoutTest(
    test_util.TestCase, BnnEndToEnd):

  def test_works_correctly(self):
    if not tf.executing_eagerly():
      self.skipTest('Skipping graph mode test since we simulate user behavior.')
    make_conv = functools.partial(
        tfn.ConvolutionVariationalFlipout,
        rank=2,
        padding='same',
        filter_shape=5,
        init_kernel_fn=tfn.initializers.he_uniform(),
        activation_fn=tf.nn.elu)
    make_deconv = functools.partial(
        tfn.ConvolutionTransposeVariationalFlipout,
        rank=2,
        padding='same',
        filter_shape=5,
        init_kernel_fn=tfn.initializers.he_uniform(),
        activation_fn=tf.nn.elu)
    self.run_bnn_test(make_conv, make_deconv)


if __name__ == '__main__':
  tf.test.main()
