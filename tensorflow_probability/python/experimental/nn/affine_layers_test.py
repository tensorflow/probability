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
"""Tests for affine_layers.py."""

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

  def run_bnn_test(self, make_affine):
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
    n = tf.cast(train_size, tf.float32)
    bnn = tfn.Sequential([
        tfn.ConvolutionVariationalReparameterization(
            evidence_shape[-1], 32, filter_shape=7,
            rank=2, strides=2, padding='same',
            init_kernel_fn=tfn.initializers.he_uniform(),
            penalty_weight=1. / n,
            activation_fn=tf.nn.elu),        # [b, 14, 14, 32]
        tfn.util.flatten_rightmost(ndims=3),  # [b, 14 * 14 * 32]
        make_affine(
            14 * 14 * 32, np.prod(target_shape) - 1,
            penalty_weight=1. / n),          # [b, 9]
        tfn.Lambda(
            eval_fn=lambda loc: tfb.SoftmaxCentered()(  # pylint: disable=g-long-lambda
                tfd.Independent(tfd.Normal(loc, scale),
                                reinterpreted_batch_ndims=1)),
            also_track=scale),               # [b, 10]
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
    fit_op = tfn.util.make_fit_op(loss_fn, opt, bnn.trainable_variables)
    for _ in range(2):
      loss, (nll, kl) = fit_op()  # pylint: disable=unused-variable


@test_util.test_all_tf_execution_regimes
class AffineTest(test_util.TestCase):

  def test_works_correctly(self):
    pass


@test_util.test_all_tf_execution_regimes
class AffineVariationalReparameterizationTest(test_util.TestCase, BnnEndToEnd):

  def test_works_correctly(self):
    if not tf.executing_eagerly():
      self.skipTest('Skipping graph mode test since we simulate user behavior.')
    make_affine = functools.partial(
        tfn.AffineVariationalReparameterization,
        init_kernel_fn=tfn.initializers.he_uniform(),
        init_bias_fn=tfn.initializers.he_uniform())
    self.run_bnn_test(make_affine)


@test_util.test_all_tf_execution_regimes
class AffineVariationalFlipoutTest(test_util.TestCase, BnnEndToEnd):

  def test_works_correctly(self):
    if not tf.executing_eagerly():
      self.skipTest('Skipping graph mode test since we simulate user behavior.')
    make_affine = functools.partial(
        tfn.AffineVariationalFlipout,
        init_kernel_fn=tfn.initializers.he_uniform(),
        init_bias_fn=tfn.initializers.he_uniform())
    self.run_bnn_test(make_affine)


@test_util.test_all_tf_execution_regimes
class AffineVariationalReparameterizationLocalTest(
    test_util.TestCase, BnnEndToEnd):

  def test_works_correctly(self):
    if not tf.executing_eagerly():
      self.skipTest('Skipping graph mode test since we simulate user behavior.')
    make_affine = functools.partial(
        tfn.AffineVariationalReparameterizationLocal,
        init_kernel_fn=tfn.initializers.he_uniform(),
        init_bias_fn=tfn.initializers.he_uniform())
    self.run_bnn_test(make_affine)


if __name__ == '__main__':
  tf.test.main()
