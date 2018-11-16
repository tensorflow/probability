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
"""Tests for Trainable Distributions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
import numpy as np

import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
tfe = tf.contrib.eager


@tfe.run_all_tests_in_graph_and_eager_modes
class TestMVNTriL(tf.test.TestCase):

  def setUp(self):
    np.random.seed(142)

  def testDefaultsYieldCorrectShapesAndValues(self):
    batch_shape = [4, 3]
    x_size = 3
    mvn_size = 5
    x_ = np.random.randn(*np.concatenate([batch_shape, [x_size]]))

    x = tf.constant(x_)
    mvn = tfp.trainable_distributions.multivariate_normal_tril(x, dims=mvn_size)
    scale = mvn.scale.to_dense()
    scale_upper = tf.matrix_set_diag(
        tf.matrix_band_part(scale, num_lower=0, num_upper=-1),
        tf.zeros(np.concatenate([batch_shape, [mvn_size]]), scale.dtype))
    scale_diag = tf.matrix_diag_part(scale)

    self.evaluate(tf.global_variables_initializer())
    [
        batch_shape_,
        event_shape_,
        scale_diag_,
        scale_upper_,
    ] = self.evaluate([
        mvn.batch_shape_tensor(),
        mvn.event_shape_tensor(),
        scale_diag,
        scale_upper,
    ])

    self.assertAllEqual(batch_shape, mvn.batch_shape)
    self.assertAllEqual(batch_shape, batch_shape_)

    self.assertAllEqual([mvn_size], mvn.event_shape)
    self.assertAllEqual([mvn_size], event_shape_)

    self.assertAllEqual(np.ones_like(scale_diag_, dtype=np.bool),
                        scale_diag_ > 0.)
    self.assertAllEqual(np.zeros_like(scale_upper_), scale_upper_)

  def testNonDefaultsYieldCorrectShapesAndValues(self):
    batch_shape = [4, 3]
    x_size = 3
    mvn_size = 5
    x_ = np.random.randn(*np.concatenate([batch_shape, [x_size]]))

    x = tf.constant(x_)
    mvn = tfp.trainable_distributions.multivariate_normal_tril(
        x,
        dims=mvn_size,
        loc_fn=tf.zeros_like,
        scale_fn=lambda x: tfd.fill_triangular(tf.ones_like(x)))
    scale = mvn.scale.to_dense()
    expected_scale = tf.matrix_band_part(
        tf.ones(np.concatenate([batch_shape, [mvn_size, mvn_size]]),
                scale.dtype),
        num_lower=-1,
        num_upper=0)

    self.evaluate(tf.global_variables_initializer())
    [
        batch_shape_,
        event_shape_,
        loc_,
        scale_,
        expected_scale_,
    ] = self.evaluate([
        mvn.batch_shape_tensor(),
        mvn.event_shape_tensor(),
        mvn.loc,
        scale,
        expected_scale,
    ])

    self.assertAllEqual(batch_shape, mvn.batch_shape)
    self.assertAllEqual(batch_shape, batch_shape_)

    self.assertAllEqual([mvn_size], mvn.event_shape)
    self.assertAllEqual([mvn_size], event_shape_)

    self.assertAllEqual(np.zeros_like(loc_), loc_)
    self.assertAllEqual(expected_scale_, scale_)


@tfe.run_all_tests_in_graph_and_eager_modes
class TestBernoulli(tf.test.TestCase):

  def setUp(self):
    np.random.seed(142)

  def testDefaultsYieldCorrectShape(self):
    batch_shape = [4, 3]
    x_size = 3
    x_ = np.random.randn(*np.concatenate([batch_shape, [x_size]]))

    x = tf.constant(x_)
    bernoulli = tfp.trainable_distributions.bernoulli(x)

    self.evaluate(tf.global_variables_initializer())
    [
        batch_shape_,
        event_shape_,
    ] = self.evaluate([
        bernoulli.batch_shape_tensor(),
        bernoulli.event_shape_tensor(),
    ])

    self.assertAllEqual(batch_shape, bernoulli.batch_shape)
    self.assertAllEqual(batch_shape, batch_shape_)

    self.assertAllEqual([], bernoulli.event_shape)
    self.assertAllEqual([], event_shape_)

  def testNonDefaultsYieldCorrectShapeAndValues(self):
    batch_shape = [4, 3]
    x_size = 3
    x_ = np.random.randn(*np.concatenate([batch_shape, [x_size]]))

    x = tf.constant(x_)
    bernoulli = tfp.trainable_distributions.bernoulli(
        x,
        layer_fn=lambda x, _: tf.reduce_sum(x, axis=-1, keepdims=True))

    [
        batch_shape_,
        event_shape_,
        logits_,
    ] = self.evaluate([
        bernoulli.batch_shape_tensor(),
        bernoulli.event_shape_tensor(),
        bernoulli.logits,
    ])

    self.assertAllEqual(batch_shape, bernoulli.batch_shape)
    self.assertAllEqual(batch_shape, batch_shape_)

    self.assertAllEqual([], bernoulli.event_shape)
    self.assertAllEqual([], event_shape_)

    self.assertAllClose(np.sum(x_, axis=-1), logits_, atol=0, rtol=1e-3)


@tfe.run_all_tests_in_graph_and_eager_modes
class TestNormal(tf.test.TestCase):

  def setUp(self):
    np.random.seed(142)

  def testDefaultsYieldCorrectShape(self):
    batch_shape = [4, 3]
    x_size = 3
    x_ = np.random.randn(*np.concatenate([batch_shape, [x_size]]))

    x = tf.constant(x_)
    normal = tfp.trainable_distributions.normal(x)

    self.evaluate(tf.global_variables_initializer())
    [
        batch_shape_,
        event_shape_,
    ] = self.evaluate([
        normal.batch_shape_tensor(),
        normal.event_shape_tensor(),
    ])

    self.assertAllEqual(batch_shape, normal.batch_shape)
    self.assertAllEqual(batch_shape, batch_shape_)

    self.assertAllEqual([], normal.event_shape)
    self.assertAllEqual([], event_shape_)

  def testNonDefaultsYieldCorrectShapeAndValues(self):
    batch_shape = [4, 3]
    x_size = 3
    x_ = np.random.randn(*np.concatenate([batch_shape, [x_size]]))

    x = tf.constant(x_)
    normal = tfp.trainable_distributions.normal(
        x,
        layer_fn=lambda x, _: tf.reduce_sum(x, axis=-1, keepdims=True))

    [
        batch_shape_,
        event_shape_,
        loc_,
    ] = self.evaluate([
        normal.batch_shape_tensor(),
        normal.event_shape_tensor(),
        normal.loc,
    ])

    self.assertAllEqual(batch_shape, normal.batch_shape)
    self.assertAllEqual(batch_shape, batch_shape_)

    self.assertAllEqual([], normal.event_shape)
    self.assertAllEqual([], event_shape_)

    self.assertAllClose(np.sum(x_, axis=-1), loc_, atol=0, rtol=1e-3)


@tfe.run_all_tests_in_graph_and_eager_modes
class TestPoisson(tf.test.TestCase):

  def setUp(self):
    np.random.seed(142)

  def testDefaultsYieldCorrectShape(self):
    batch_shape = [4, 3]
    x_size = 3
    x_ = np.random.randn(*np.concatenate([batch_shape, [x_size]]))

    x = tf.constant(x_)
    poisson = tfp.trainable_distributions.poisson(x)

    self.evaluate(tf.global_variables_initializer())
    [
        batch_shape_,
        event_shape_,
    ] = self.evaluate([
        poisson.batch_shape_tensor(),
        poisson.event_shape_tensor(),
    ])

    self.assertAllEqual(batch_shape, poisson.batch_shape)
    self.assertAllEqual(batch_shape, batch_shape_)

    self.assertAllEqual([], poisson.event_shape)
    self.assertAllEqual([], event_shape_)

  def testNonDefaultsYieldCorrectShapeAndValues(self):
    batch_shape = [4, 3]
    x_size = 3
    x_ = np.random.randn(*np.concatenate([batch_shape, [x_size]]))

    x = tf.constant(x_)
    poisson = tfp.trainable_distributions.poisson(
        x,
        layer_fn=lambda x, _: tf.reduce_sum(x, axis=-1, keepdims=True))

    [
        batch_shape_,
        event_shape_,
        log_rate_,
    ] = self.evaluate([
        poisson.batch_shape_tensor(),
        poisson.event_shape_tensor(),
        poisson.log_rate,
    ])

    self.assertAllEqual(batch_shape, poisson.batch_shape)
    self.assertAllEqual(batch_shape, batch_shape_)

    self.assertAllEqual([], poisson.event_shape)
    self.assertAllEqual([], event_shape_)

    self.assertAllClose(np.sum(x_, axis=-1), log_rate_, atol=0, rtol=1e-3)


@tfe.run_all_tests_in_graph_and_eager_modes
class TestMakePositiveFunctions(tf.test.TestCase):

  def softplus(self, x):
    return np.log1p(np.exp(x))

  def testPositiveTriLWorks(self):
    x_ = np.float32(np.arange(6) - 3)
    y = tfp.trainable_distributions.tril_with_diag_softplus_and_shift(
        x_, diag_shift=1.)
    y_ = self.evaluate(y)
    # Recall that:
    # self.evaluate(tfd.fill_triangular(np.arange(6) - 3))
    # ==> array([[ 0,  0,  0],
    #            [ 2,  1,  0],
    #            [-1, -2, -3]])
    # I.e., fill_triangular fills in a counter-clockwise spiral, starting from
    # the last row. Hence,
    expected_y = np.float32([
        [self.softplus(0) + 1, 0, 0],
        [1 + 1, self.softplus(1) + 1, 0],
        [-2 + 1, -3 + 1, self.softplus(-3) + 1]])
    self.assertAllClose(expected_y, y_, atol=1e-5, rtol=1e-5)

  def testPositiveWorks(self):
    x_ = np.float32(np.arange(6) - 3)
    y = tfp.trainable_distributions.softplus_and_shift(x_, shift=1.)
    y_ = self.evaluate(y)
    self.assertAllClose(self.softplus(x_) + 1, y_, atol=1e-5, rtol=1e-5)


if __name__ == '__main__':
  tf.test.main()
