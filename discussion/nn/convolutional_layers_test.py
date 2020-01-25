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

from discussion import nn
from tensorflow_probability.python.internal import test_util


tfb = tfp.bijectors
tfd = tfp.distributions


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
    train_dataset = nn.util.tune_dataset(
        train_dataset,
        batch_size=batch_size,
        shuffle_size=int(train_size / 7))

    # 2  Specify Model

    scale = tfp.util.TransformedVariable(1., tfb.Softplus())
    n = tf.cast(train_size, tf.float32)
    bnn = nn.Sequential([
        make_conv(evidence_shape[-1], 32, filter_shape=7, strides=2,
                  penalty_weight=1. / n),
        tf.nn.elu,
        # nn.util.trace('conv1'),    # [b, 14, 14, 32]
        nn.util.flatten_rightmost,
        # nn.util.trace('flat1'),    # [b, 14 * 14 * 32]
        nn.AffineVariationalReparameterization(
            14 * 14 * 32, np.prod(target_shape) - 1,
            penalty_weight=1. / n),
        # nn.util.trace('affine1'),  # [b, 9]
        nn.Lambda(
            eval_fn=lambda loc: tfb.SoftmaxCentered()(  # pylint: disable=g-long-lambda
                tfd.Independent(tfd.Normal(loc, scale),
                                reinterpreted_batch_ndims=1)),
            also_track=scale),
        # nn.util.trace('head'),     # [b, 10]
    ], name='bayesian_autoencoder')

    self.evaluate([v.initializer for v in bnn.trainable_variables])

    # 3  Train.

    train_iter = iter(train_dataset)
    def loss_fn():
      x, y = next(train_iter)
      nll = -tf.reduce_mean(bnn(x).log_prob(y), axis=-1)
      kl = bnn.extra_loss  # Already normalized.
      return nll + kl, (nll, kl)
    opt = tf.optimizers.Adam()
    fit_op = nn.util.make_fit_op(loss_fn, opt, bnn.trainable_variables)
    for _ in range(2):
      loss, (nll, kl) = fit_op()  # pylint: disable=unused-variable


@test_util.test_all_tf_execution_regimes
class ConvolutionTest(test_util.TestCase):

  def test_works_correctly(self):
    pass


@test_util.test_all_tf_execution_regimes
class ConvolutionVariationalReparameterizationTest(
    test_util.TestCase, BnnEndToEnd):

  def test_works_correctly(self):
    if not tf.executing_eagerly():
      self.skipTest('Skipping graph mode test since we simulate user behavior.')
    make_conv = functools.partial(
        nn.ConvolutionVariationalReparameterization,
        rank=2,
        padding='same',
        init_kernel_fn=tf.initializers.he_normal())
    self.run_bnn_test(make_conv)


@test_util.test_all_tf_execution_regimes
class ConvolutionVariationalFlipoutTest(test_util.TestCase, BnnEndToEnd):

  def test_works_correctly(self):
    if not tf.executing_eagerly():
      self.skipTest('Skipping graph mode test since we simulate user behavior.')
    make_conv = functools.partial(
        nn.ConvolutionVariationalFlipout,
        rank=2,
        padding='same',
        init_kernel_fn=tf.initializers.he_normal())
    self.run_bnn_test(make_conv)


if __name__ == '__main__':
  tf.test.main()
