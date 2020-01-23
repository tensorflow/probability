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
"""Tests for Positive-Semidefinite Kernels utilities."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
from absl.testing import parameterized
import numpy as np
import tensorflow.compat.v1 as tf1
import tensorflow.compat.v2 as tf

from tensorflow_probability.python.internal import tensorshape_util
from tensorflow_probability.python.internal import test_util
from tensorflow_probability.python.math.gradient import value_and_gradient
from tensorflow_probability.python.math.psd_kernels.internal import util


class UtilTest(test_util.TestCase):

  def testPadShapeRightWithOnes(self):
    # Test nominal behavior.
    x = np.ones([3], np.float32)
    self.assertAllEqual(
        self.evaluate(util.pad_shape_with_ones(x, 3)).shape,
        [3, 1, 1, 1])

  def testPadShapeStartWithOnes(self):
    # Test nominal behavior.
    x = np.ones([3], np.float32)
    self.assertAllEqual(
        self.evaluate(util.pad_shape_with_ones(x, 3, start=-2)).shape,
        [1, 1, 1, 3])

  def testPadShapeMiddleWithOnes(self):
    # Test nominal behavior.
    x = np.ones([2, 3, 5], np.float32)
    self.assertAllEqual(
        self.evaluate(util.pad_shape_with_ones(x, 3)).shape,
        [2, 3, 5, 1, 1, 1])

    self.assertAllEqual(
        self.evaluate(util.pad_shape_with_ones(x, 3, start=-2)).shape,
        [2, 3, 1, 1, 1, 5])

    self.assertAllEqual(
        self.evaluate(util.pad_shape_with_ones(x, 3, start=-3)).shape,
        [2, 1, 1, 1, 3, 5])

  def testPadShapeRightWithOnesDynamicShape(self):
    if tf.executing_eagerly(): return
    # Test partially unknown shape
    x = tf1.placeholder_with_default(np.ones([3], np.float32), [None])
    expanded = util.pad_shape_with_ones(x, 3)
    self.assertAllEqual((1, 1, 1), expanded.shape[1:])
    self.assertAllEqual((3, 1, 1, 1), self.evaluate(expanded).shape)

    expanded = util.pad_shape_with_ones(x, 3, start=-2)
    self.assertAllEqual((1, 1, 1), expanded.shape[:-1])
    self.assertAllEqual(self.evaluate(expanded).shape, [1, 1, 1, 3])
    self.assertAllEqual((1, 1, 1, 3), self.evaluate(expanded).shape)

    # Test totally unknown shape
    x = tf1.placeholder_with_default(np.ones([3], np.float32), None)
    expanded = util.pad_shape_with_ones(x, 3)
    self.assertAllEqual([3, 1, 1, 1], self.evaluate(expanded).shape)

  @test_util.jax_disable_test_missing_functionality(
      'Graphs do not exist in Jax.')
  def testPadShapeRightWithOnesCanBeGraphNoop(self):
    # First ensure graph actually *is* changed when we use non-trivial ndims.
    # Use an explicitly created graph, to make sure no whacky test fixture graph
    # reuse is going on in the background.
    g = tf.Graph()
    with g.as_default():
      x = tf.constant(np.ones([3], np.float32))
      graph_def = g.as_graph_def()
      x = util.pad_shape_with_ones(x, 3)
      self.assertNotEqual(graph_def, g.as_graph_def())

    # Now verify that graphdef is unchanged (no extra ops) when we pass ndims=0.
    g = tf.Graph()
    with g.as_default():
      x = tf.constant(np.ones([3], np.float32))
      graph_def = g.as_graph_def()
      x = util.pad_shape_with_ones(x, 0)
      self.assertEqual(graph_def, g.as_graph_def())

  def testSumRightmostNdimsPreservingShapeStaticRank(self):
    x = np.ones((5, 4, 3, 2))
    self.assertAllEqual(
        util.sum_rightmost_ndims_preserving_shape(x, ndims=2).shape,
        [5, 4])

    x = tf1.placeholder_with_default(
        np.ones((5, 4, 3, 2)), shape=[5, 4, None, None])
    if not tf.executing_eagerly():
      return
    y = util.sum_rightmost_ndims_preserving_shape(x, ndims=1)
    self.assertAllEqual((5, 4, 3), y.shape)

  def testSumRightmostNdimsPreservingShapeDynamicRank(self):
    if tf.executing_eagerly(): return
    x = tf1.placeholder_with_default(np.ones((5, 4, 3, 2)), shape=None)
    y = util.sum_rightmost_ndims_preserving_shape(x, ndims=2)
    self.assertIsNone(tensorshape_util.rank(y.shape))
    self.assertAllEqual(
        (5, 4),
        self.evaluate(
            util.sum_rightmost_ndims_preserving_shape(x, ndims=2)).shape)

  def testSqrtWithFiniteGradsHasCorrectValues(self):
    self.assertTrue(np.isnan(self.evaluate(util.sqrt_with_finite_grads(-1.))))
    xs = np.linspace(0., 10., 100)
    self.assertAllEqual(
        self.evaluate(tf.sqrt(xs)),
        self.evaluate(util.sqrt_with_finite_grads(xs)))

  def testSqrtWithFiniteGradsHasCorrectGradients(self):
    self.assertTrue(np.isnan(self.evaluate(util.sqrt_with_finite_grads(-1.))))
    xs = tf.constant(np.linspace(1e-10, 10., 100))
    _, grad_tf_sqrt = value_and_gradient(tf.sqrt, xs)
    _, grad_safe_sqrt = value_and_gradient(
        util.sqrt_with_finite_grads, xs)
    self.assertAllEqual(*self.evaluate([grad_tf_sqrt, grad_safe_sqrt]))

    zero = tf.constant(0.)
    _, grad_tf_sqrt = value_and_gradient(tf.sqrt, zero)
    _, grad_safe_sqrt = value_and_gradient(
        util.sqrt_with_finite_grads, zero)
    self.assertNotEqual(*self.evaluate([grad_tf_sqrt, grad_safe_sqrt]))

  def testSqrtWithFiniteGradsBackpropsCorrectly(self):
    # Part of implementing a tf.custom_gradient is correctly handling the
    # `grad_ys` value that is propagating back from downstream ops. This test
    # checks that we got this right, in a particular case where our sqrt
    # function is squashed between a couple of other functions.
    def f(x):
      return x ** 2

    def g(x):
      return util.sqrt_with_finite_grads(x)

    def h(x):
      return tf.sin(x) ** 2

    # We only test away from zero, since we know the values don't match there.
    xs = tf.constant(np.linspace(1e-10, 10., 100))
    _, grad_tf_sqrt = value_and_gradient(
        lambda xs_: f(tf.sqrt(h(xs_))), xs)
    _, grad_safe_sqrt = value_and_gradient(
        lambda xs_: f(g(h(xs_))), xs)
    self.assertAllClose(*self.evaluate([grad_tf_sqrt, grad_safe_sqrt]),
                        rtol=1e-10)

  def testSqrtWithFiniteGradsWithDynamicShape(self):
    x = tf1.placeholder_with_default([1.], shape=[None])
    _, grad_tf_sqrt = value_and_gradient(tf.sqrt, x)
    _, grad_safe_sqrt = value_and_gradient(
        util.sqrt_with_finite_grads, x)
    self.assertAllEqual(*self.evaluate([grad_tf_sqrt, grad_safe_sqrt]))

  @parameterized.parameters(
      {'feature_ndims': 1, 'dims': 3},
      {'feature_ndims': 1, 'dims': 4},
      {'feature_ndims': 2, 'dims': 2},
      {'feature_ndims': 2, 'dims': 3},
      {'feature_ndims': 3, 'dims': 2},
      {'feature_ndims': 3, 'dims': 3})
  def testPairwiseSquareDistanceMatrix(self, feature_ndims, dims):
    batch_shape = [2, 3]
    seed_stream = test_util.test_seed_stream('pairwise_square_distance')
    x1 = tf.random.normal(
        dtype=np.float64, shape=batch_shape + [dims] * feature_ndims,
        seed=seed_stream())
    x2 = tf.random.normal(
        dtype=np.float64, shape=batch_shape + [dims] * feature_ndims,
        seed=seed_stream())
    pairwise_square_distance = util.pairwise_square_distance_matrix(
        x1, x2, feature_ndims)

    x1_pad = util.pad_shape_with_ones(
        x1, ndims=1, start=-(feature_ndims + 1))
    x2_pad = util.pad_shape_with_ones(
        x2, ndims=1, start=-(feature_ndims + 2))
    actual_square_distance = util.sum_rightmost_ndims_preserving_shape(
        tf.math.squared_difference(x1_pad, x2_pad), feature_ndims)
    pairwise_square_distance_, actual_square_distance_ = self.evaluate([
        pairwise_square_distance, actual_square_distance])
    self.assertAllClose(pairwise_square_distance_, actual_square_distance_)


if __name__ == '__main__':
  tf.test.main()
