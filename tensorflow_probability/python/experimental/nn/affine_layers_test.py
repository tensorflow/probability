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

from absl.testing import parameterized

import numpy as np
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp

from tensorflow_probability.python.internal import prefer_static as ps
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
    bnn = tfn.Sequential([
        tfn.ConvolutionVariationalReparameterization(
            evidence_shape[-1], 32, filter_shape=7,
            rank=2, strides=2, padding='same',
            kernel_initializer=tfn.initializers.he_uniform(),
            activation_fn=tf.nn.elu),        # [b, 14, 14, 32]
        tfn.util.flatten_rightmost(ndims=3),  # [b, 14 * 14 * 32]
        make_affine(
            14 * 14 * 32, np.prod(target_shape) - 1),          # [b, 9]
        lambda loc: tfb.SoftmaxCentered()(  # pylint: disable=g-long-lambda
            tfd.Independent(tfd.Normal(loc, scale),
                            reinterpreted_batch_ndims=1))  # [b, 10]
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
class AffineTest(test_util.TestCase):
  # Note that some of these kwargs are unused for the tests, but they are
  # passed from the layer to make_kernel_bias_fn so we need to take them in.

  def make_bayesian_kernel_bias(self,
                                kernel_shape,
                                bias_shape,
                                kernel_initializer=None,
                                bias_initializer=None,
                                kernel_batch_ndims=0,
                                bias_batch_ndims=0,
                                dtype=tf.float32):

    kernel_dist = tfd.Independent(
        tfd.Normal(tf.zeros(kernel_shape, dtype=dtype),
                   tf.ones(kernel_shape, dtype=dtype)),
        reinterpreted_batch_ndims=tf.size(kernel_shape)-kernel_batch_ndims)

    bias_dist = tfd.Independent(
        tfd.Normal(tf.zeros(bias_shape, dtype=dtype),
                   tf.ones(bias_shape, dtype=dtype)),
        reinterpreted_batch_ndims=tf.size(bias_shape)-bias_batch_ndims)

    wk = kernel_dist.sample(self.num_samples)
    wb = bias_dist.sample(self.num_samples)

    return wk, wb

  def test_dnn_nobayes(self):
    affine = tfn.Affine(5, 10, batch_shape=())
    inputs = tf.zeros([4, 5])
    output = affine(inputs)
    self.assertShapeEqual(np.zeros([4, 10]), output)

  def test_dnn_bayes(self):
    self.num_samples = 3
    self.num_components = None
    inputs = tf.zeros([4, 5])
    affine = tfn.Affine(5, 10,
                        make_kernel_bias_fn=self.make_bayesian_kernel_bias)
    output = affine(inputs[:, tf.newaxis, :])
    self.assertShapeEqual(np.zeros([4, 3, 10]), output)

  def test_dnn_bayes_mixture(self):
    self.num_samples = 3
    self.num_components = 6
    inputs = tf.zeros([4, 5])
    affine = tfn.Affine(5, 10,
                        make_kernel_bias_fn=self.make_bayesian_kernel_bias,
                        batch_shape=self.num_components)
    output = affine(inputs[:, tf.newaxis, tf.newaxis, :])
    self.assertShapeEqual(np.zeros([4, 3, 6, 10]), output)

  @parameterized.parameters(
      (12, 3, (), (1,)),            # scalar input batch, scalar kernel batch
      (6, 4, (2, 3), (1,)),         # non-scalar kernel batch
      (3, 5, (), (2, 2)),           # non-scalar input batch
      (3, 3, (2, 3), (8, 1, 1)),    # broadcasting kernel and input batch shapes
      (3, 3, (2, 2), (4, 2, 2)))    # same kernel and input batch shapes
  def test_works_correctly(
      self,
      input_size,
      output_size,
      kernel_batch_shape,
      input_batch_shape):
    affine = tfn.Affine(
        input_size,
        output_size=output_size,
        batch_shape=kernel_batch_shape)
    x = tf.ones((input_batch_shape + (input_size,)), dtype=tf.float32)
    y = affine(x)
    self.assertAllEqual(
        y.shape,
        ps.broadcast_shape(
            kernel_batch_shape, input_batch_shape).concatenate(output_size))


@test_util.test_all_tf_execution_regimes
class AffineVariationalReparameterizationTest(test_util.TestCase, BnnEndToEnd):

  def test_works_correctly(self):
    if not tf.executing_eagerly():
      self.skipTest('Skipping graph mode test since we simulate user behavior.')
    make_affine = functools.partial(
        tfn.AffineVariationalReparameterization,
        kernel_initializer=tfn.initializers.he_uniform(),
        bias_initializer=tfn.initializers.he_uniform())
    self.run_bnn_test(make_affine)


@test_util.test_all_tf_execution_regimes
class AffineVariationalFlipoutTest(test_util.TestCase, BnnEndToEnd):

  def test_works_correctly(self):
    if not tf.executing_eagerly():
      self.skipTest('Skipping graph mode test since we simulate user behavior.')
    make_affine = functools.partial(
        tfn.AffineVariationalFlipout,
        kernel_initializer=tfn.initializers.he_uniform(),
        bias_initializer=tfn.initializers.he_uniform())
    self.run_bnn_test(make_affine)


@test_util.test_all_tf_execution_regimes
class AffineVariationalReparameterizationLocalTest(
    test_util.TestCase, BnnEndToEnd):

  def test_works_correctly(self):
    if not tf.executing_eagerly():
      self.skipTest('Skipping graph mode test since we simulate user behavior.')
    make_affine = functools.partial(
        tfn.AffineVariationalReparameterizationLocal,
        kernel_initializer=tfn.initializers.he_uniform(),
        bias_initializer=tfn.initializers.he_uniform())
    self.run_bnn_test(make_affine)


if __name__ == '__main__':
  test_util.main()
