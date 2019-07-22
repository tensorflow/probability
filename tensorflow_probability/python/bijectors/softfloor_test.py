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
"""Tests for Bijector."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
import numpy as np

import tensorflow as tf

from tensorflow_probability.python import bijectors as tfb
from tensorflow_probability.python.bijectors import bijector_test_util
from tensorflow_probability.python.internal import tensorshape_util
from tensorflow.python.framework import test_util  # pylint: disable=g-direct-tensorflow-import,g-import-not-at-top


def _sigmoid(x):
  return 1 / (1 + np.exp(-x))


def _softfloor_grad_np(x, t):
  x -= 0.5
  frac_part = x - np.floor(x)
  inner_part = (frac_part - 0.5) / t
  return _sigmoid(inner_part) * (1 - _sigmoid(inner_part)) / (t * (
      1 - 2. * _sigmoid(-0.5 / t)))


@test_util.run_all_in_graph_and_eager_modes
class _SoftFloorBijectorBase(object):
  """Tests correctness of the floor transformation."""

  def setUp(self):
    self._rng = np.random.RandomState(42)

  def testBijectorApproximatesFloorLowTemperature(self):
    # Let's make this look floor.
    floor = tfb.Softfloor(self.dtype(1e-4))
    # We chose a high temperature, and truncated range so that
    # we are likely to be retrieving 2.
    pos_values = np.linspace(2.1, 2.9, 50).astype(self.dtype)
    neg_values = np.linspace(-2.9, -2.1, 50).astype(self.dtype)
    self.assertAllClose(
        self.evaluate(floor.forward(pos_values)),
        np.floor(pos_values))
    self.assertAllClose(
        self.evaluate(floor.forward(neg_values)),
        np.floor(neg_values))

  def testShapeGetters(self):
    x = tf.TensorShape([4])
    y = tf.TensorShape([4])
    bijector = tfb.Softfloor(self.dtype(1.), validate_args=True)
    self.assertAllEqual(y, bijector.forward_event_shape(x))
    self.assertAllEqual(
        tensorshape_util.as_list(y),
        self.evaluate(
            bijector.forward_event_shape_tensor(tensorshape_util.as_list(x))))
    self.assertAllEqual(x, bijector.inverse_event_shape(y))
    self.assertAllEqual(
        tensorshape_util.as_list(x),
        self.evaluate(
            bijector.inverse_event_shape_tensor(tensorshape_util.as_list(y))))

  def testBijectiveAndFiniteHighTemperature(self):
    floor = tfb.Softfloor(self.dtype(70.))
    x = np.sort(5 * self._rng.randn(3, 10), axis=-1).astype(self.dtype)
    y = 5 * self._rng.randn(3, 10).astype(self.dtype)
    bijector_test_util.assert_bijective_and_finite(
        floor, x, y, eval_func=self.evaluate, event_ndims=1)

  def testBijectiveAndFiniteMediumTemperature(self):
    floor = tfb.Softfloor(self.dtype(5.))
    x = np.sort(5 * self._rng.randn(3, 10), axis=-1).astype(self.dtype)
    y = 5 * self._rng.randn(3, 10).astype(self.dtype)
    bijector_test_util.assert_bijective_and_finite(
        floor, x, y, eval_func=self.evaluate, event_ndims=1)

  def testBijectiveAndFiniteLowTemperature(self):
    floor = tfb.Softfloor(self.dtype(1e-1))
    x = np.sort(5 * self._rng.randn(3, 10), axis=-1).astype(self.dtype)
    y = 5 * self._rng.randn(3, 10).astype(self.dtype)
    bijector_test_util.assert_bijective_and_finite(
        floor, x, y, eval_func=self.evaluate, event_ndims=1)

  def testBijectorScalarCongruencyHighTemperature(self):
    floor = tfb.Softfloor(self.dtype(1000.))
    bijector_test_util.assert_scalar_congruency(
        floor, self.dtype(-1.1), self.dtype(1.1), eval_func=self.evaluate)

  def testBijectorScalarCongruencyMediumTemperature(self):
    floor = tfb.Softfloor(self.dtype(5.))
    bijector_test_util.assert_scalar_congruency(
        floor, self.dtype(-1.1), self.dtype(1.1), eval_func=self.evaluate)

  def testBijectorScalarCongruencyLowTemperature(self):
    floor = tfb.Softfloor(self.dtype(0.5))
    bijector_test_util.assert_scalar_congruency(
        floor, self.dtype(-1.1), self.dtype(1.1),
        eval_func=self.evaluate, rtol=0.05)

  # Check that we have a gradient for the forward, and it
  # matches the numpy gradient.
  def testBijectorForwardGradient(self):
    x_np = np.array([0.1, 2.23, 4.1], dtype=self.dtype)
    x = tf.constant(x_np)
    with tf.GradientTape() as tape:
      tape.watch(x)
      value = tfb.Softfloor(self.dtype(1.2)).forward(x)
    grad = tape.gradient(value, x)
    self.assertAllClose(_softfloor_grad_np(x_np, 1.2), grad)


class SoftFloor32Test(_SoftFloorBijectorBase, tf.test.TestCase):
  dtype = np.float32


class SoftFloor64Test(_SoftFloorBijectorBase, tf.test.TestCase):
  dtype = np.float64


if __name__ == "__main__":
  tf.test.main()
