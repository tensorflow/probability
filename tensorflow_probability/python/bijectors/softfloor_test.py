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
import tensorflow.compat.v2 as tf
from tensorflow_probability.python import bijectors as tfb
from tensorflow_probability.python.bijectors import bijector_test_util
from tensorflow_probability.python.internal import tensorshape_util
from tensorflow_probability.python.internal import test_util
from tensorflow_probability.python.math import value_and_gradient


def _sigmoid(x):
  return 1 / (1 + np.exp(-x))


def _softfloor_grad_np(x, t):
  x -= 0.5
  frac_part = x - np.floor(x)
  inner_part = (frac_part - 0.5) / t
  return _sigmoid(inner_part) * (1 - _sigmoid(inner_part)) / (t * (
      1 - 2. * _sigmoid(-0.5 / t)))


@test_util.test_all_tf_execution_regimes
class _SoftFloorBijectorBase(object):
  """Tests correctness of the floor transformation."""

  def setUp(self):
    self._rng = np.random.RandomState(42)

  def testBijectorApproximatesFloorLowTemperature(self):
    # Let's make this look floor.
    floor = tfb.Softfloor(self.dtype(1e-4))
    # We chose a high temperature, and truncated range so that
    # we are likely to be retrieving 2.
    pos_values = np.linspace(2.01, 2.99, 100).astype(self.dtype)
    neg_values = np.linspace(-2.99, -2.01, 100).astype(self.dtype)
    self.assertAllClose(
        self.evaluate(floor.forward(pos_values)),
        np.floor(pos_values))
    self.assertAllClose(
        self.evaluate(floor.forward(neg_values)),
        np.floor(neg_values))

  def testBijectorEndpointsAtLimit(self):
    # Check that we don't get NaN at half-integer and the floor matches.
    floor = tfb.Softfloor(self.dtype(1e-5))
    half_integers = np.linspace(0.5, 10.5, 11).astype(self.dtype)
    self.assertAllClose(
        self.evaluate(floor.forward(half_integers)),
        np.floor(half_integers))
    self.assertAllFinite(self.evaluate(floor.inverse(half_integers)))
    # At integer values, check we don't have NaN's, and this is close to
    # integer - 0.5 (which is only true in the limit!).
    integers = np.linspace(0., 10., 11).astype(self.dtype)
    self.assertAllClose(
        self.evaluate(floor.forward(integers)),
        integers - 0.5, rtol=3e-3)
    self.assertAllFinite(self.evaluate(floor.inverse(integers)))

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
        floor, x, y, eval_func=self.evaluate, event_ndims=1, rtol=2e-5)

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
        floor, self.dtype(-1.1), self.dtype(1.1), eval_func=self.evaluate)

  def testBijectorScalarCongruencyLowerTemperature(self):
    floor = tfb.Softfloor(self.dtype(0.1))
    bijector_test_util.assert_scalar_congruency(
        floor, self.dtype(-1.1), self.dtype(1.1), eval_func=self.evaluate,
        rtol=5e-2)

  # Check that we have a gradient for the forward, and it
  # matches the numpy gradient.
  @test_util.numpy_disable_gradient_test
  def testBijectorForwardGradient(self):
    x_np = np.array([0.1, 2.23, 4.1], dtype=self.dtype)
    x = tf.constant(x_np)
    grad = value_and_gradient(tfb.Softfloor(self.dtype(1.2)).forward, x)[1]
    self.assertAllClose(_softfloor_grad_np(x_np, 1.2), grad)

  def testVariableTemperature(self):
    temperature = tf.Variable(1.)
    b = tfb.Softfloor(temperature, validate_args=True)
    self.evaluate(temperature.initializer)
    self.assertIs(temperature, b.temperature)
    self.assertEqual((), self.evaluate(b.forward(0.5)).shape)
    with self.assertRaisesOpError(
        'Argument `temperature` was not positive.'):
      with tf.control_dependencies([temperature.assign(0.)]):
        self.evaluate(b.forward(0.5))


class SoftFloor32Test(_SoftFloorBijectorBase, test_util.TestCase):
  dtype = np.float32


class SoftFloor64Test(_SoftFloorBijectorBase, test_util.TestCase):
  dtype = np.float64


if __name__ == '__main__':
  tf.test.main()
