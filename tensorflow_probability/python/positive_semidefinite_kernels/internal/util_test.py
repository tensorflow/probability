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
import numpy as np
import tensorflow as tf

from tensorflow_probability.python.math.gradient import value_and_gradient
from tensorflow_probability.python.positive_semidefinite_kernels.internal import util


class UtilTest(tf.test.TestCase):

  def testPadShapeRightWithOnes(self):
    # Test nominal behavior.
    x = np.ones([3], np.float32)
    self.assertAllEqual(
        self.evaluate(util.pad_shape_right_with_ones(x, 3)).shape,
        [3, 1, 1, 1])

  def testPadShapeRightWithOnesDynamicShape(self):
    if tf.executing_eagerly(): return
    # Test partially unknown shape
    x = tf.compat.v1.placeholder_with_default(np.ones([3], np.float32), [None])
    expanded = util.pad_shape_right_with_ones(x, 3)
    self.assertAllEqual(expanded.shape.as_list(), [None, 1, 1, 1])
    self.assertAllEqual(self.evaluate(expanded).shape, [3, 1, 1, 1])

    # Test totally unknown shape
    x = tf.compat.v1.placeholder_with_default(np.ones([3], np.float32), None)
    expanded = util.pad_shape_right_with_ones(x, 3)
    self.assertIsNone(expanded.shape.ndims)
    self.assertAllEqual(self.evaluate(expanded).shape, [3, 1, 1, 1])

  def testPadShapeRightWithOnesCanBeGraphNoop(self):
    # First ensure graph actually *is* changed when we use non-trivial ndims.
    # Use an explicitly created graph, to make sure no whacky test fixture graph
    # reuse is going on in the background.
    g = tf.Graph()
    with g.as_default():
      x = tf.constant(np.ones([3], np.float32))
      graph_def = g.as_graph_def()
      x = util.pad_shape_right_with_ones(x, 3)
      self.assertNotEqual(graph_def, g.as_graph_def())

    # Now verify that graphdef is unchanged (no extra ops) when we pass ndims=0.
    g = tf.Graph()
    with g.as_default():
      x = tf.constant(np.ones([3], np.float32))
      graph_def = g.as_graph_def()
      x = util.pad_shape_right_with_ones(x, 0)
      self.assertEqual(graph_def, g.as_graph_def())

  def testSumRightmostNdimsPreservingShapeStaticRank(self):
    x = np.ones((5, 4, 3, 2))
    self.assertAllEqual(
        util.sum_rightmost_ndims_preserving_shape(x, ndims=2).shape,
        [5, 4])

    x = tf.compat.v1.placeholder_with_default(
        np.ones((5, 4, 3, 2)), shape=[5, 4, None, None])
    self.assertAllEqual(
        util.sum_rightmost_ndims_preserving_shape(x, ndims=1).shape.as_list(),
        [5, 4, 3 if tf.executing_eagerly() else None])

  def testSumRightmostNdimsPreservingShapeDynamicRank(self):
    if tf.executing_eagerly(): return
    x = tf.compat.v1.placeholder_with_default(np.ones((5, 4, 3, 2)), shape=None)
    self.assertIsNone(
        util.sum_rightmost_ndims_preserving_shape(x, ndims=2).shape.ndims)
    self.assertAllEqual(
        self.evaluate(
            util.sum_rightmost_ndims_preserving_shape(x, ndims=2)).shape,
        [5, 4])

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
    x = tf.compat.v1.placeholder_with_default([1.], shape=[None])
    _, grad_tf_sqrt = value_and_gradient(tf.sqrt, x)
    _, grad_safe_sqrt = value_and_gradient(
        util.sqrt_with_finite_grads, x)
    self.assertAllEqual(*self.evaluate([grad_tf_sqrt, grad_safe_sqrt]))


if __name__ == '__main__':
  tf.test.main()
