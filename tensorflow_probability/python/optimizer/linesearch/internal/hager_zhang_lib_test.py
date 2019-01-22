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
"""Tests for hager_zhang_lib."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import tensorflow as tf
from tensorflow_probability.python.optimizer.linesearch.internal import hager_zhang_lib as hzl

tfe = tf.contrib.eager


def test_function_x_y(x, y):
  """Builds a function that passes through the given points.

  Args:
    x: A tf.Tensor of shape [n].
    y: A tf.Tensor of shape [n] or [b, n] if batching is desired.

  Returns:
    A callable that takes a tf.Tensor `t` as input and returns as output the
    value and derivative of the interpolated function at `t`.
  """
  if len(y.shape) == 1:  # No batches.
    y = tf.expand_dims(y, axis=0)
  b, n = y.shape
  y = tf.expand_dims(y, axis=-1)
  x = tf.reshape(tf.tile(x, [b]), (b, n, 1))  # Repeat x on all batches.

  def f(t):
    t = tf.convert_to_tensor(t)
    while len(t.shape) < 3:
      t = tf.expand_dims(t, axis=-1)
    with tf.GradientTape() as g:
      g.watch(t)
      p = tf.contrib.image.interpolate_spline(x, y, t, 2)
    return tf.squeeze(p), tf.squeeze(g.gradient(p, t))

  return f


def test_function_x_y_dy(x, y, dy, eps=0.1):
  """Builds a polynomial with (approx) given values and derivatives."""
  x1 = x + eps
  y1 = y + eps * dy
  x2 = x - eps
  y2 = y - eps * dy
  return test_function_x_y(tf.concat([x1, x2], -1), tf.concat([y1, y2], -1))


class HagerZhangLibTest(tf.test.TestCase):

  @tfe.run_test_in_graph_and_eager_modes
  def test_bisect_simple(self):
    """Tests that _bisect works on a 1 variable scalar valued function."""
    wolfe_threshold = 1e-6
    x = tf.constant([0.0, 0.5, 1.0])
    y = tf.constant([1.0, 0.6, 1.2])
    dy = tf.constant([-0.8, 0.6, -0.7])
    fun = test_function_x_y_dy(x, y, dy)

    val_a = hzl._apply(fun, 0.0)  # Value at zero.
    val_b = hzl._apply(fun, 1.0)  # Value at initial step.
    f_lim = val_a.f + (wolfe_threshold * tf.abs(val_a.f))

    result = self.evaluate(hzl.bisect(fun, val_a, val_b, f_lim))
    self.assertEqual(result.right.x, 0.5)

  @tfe.run_test_in_graph_and_eager_modes
  def test_bisect_batching(self):
    """Tests that _bisect works in batching mode."""
    wolfe_threshold = 1e-6
    # Let's build our example function with 4 batches, each evaluating a
    # different poly. They all have negative slopes both on 0.0 and 1.0,
    # but different slopes (positive, negative) and values (low enough, too
    # high) on their midpoint.
    x = tf.constant([0.0, 0.5, 1.0])
    y = tf.constant([[1.0, 0.6, 1.2],
                     [1.0, 0.6, 1.2],
                     [1.0, 1.6, 1.2],
                     [1.0, 1.6, 1.2]])
    dy = tf.constant([[-0.8, 0.6, -0.7],
                      [-0.8, -0.4, -0.7],
                      [-0.8, 0.8, -0.7],
                      [-0.8, -0.4, -0.7]])
    fun = test_function_x_y_dy(x, y, dy, eps=0.1)

    val_a = hzl._apply(fun, tf.zeros(4))  # Values at zero.
    val_b = hzl._apply(fun, tf.ones(4))  # Values at initial step.
    f_lim = val_a.f + (wolfe_threshold * tf.abs(val_a.f))

    expected_left = np.array([0.0, 0.5, 0.0, 0.0])
    expected_right = np.array([0.5, 0.75, 0.5, 0.25])

    result = self.evaluate(hzl.bisect(fun, val_a, val_b, f_lim))
    self.assertTrue(np.all(result.stopped))
    self.assertTrue(np.all(~result.failed))
    self.assertTrue(np.all(result.left.df < 0))
    self.assertTrue(np.all(result.right.df >= 0))
    self.assertArrayNear(result.left.x, expected_left, 1e-5)
    self.assertArrayNear(result.right.x, expected_right, 1e-5)


if __name__ == '__main__':
  tf.test.main()
