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
"""Tests for SoftClip Bijector."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports

from absl.testing import parameterized

import numpy as np
import tensorflow.compat.v1 as tf1
import tensorflow.compat.v2 as tf
from tensorflow_probability.python import bijectors as tfb
from tensorflow_probability.python.internal import test_util as tfp_test_util

from tensorflow.python.framework import test_util  # pylint: disable=g-direct-tensorflow-import,g-import-not-at-top


@test_util.run_all_in_graph_and_eager_modes
class _SoftClipBijectorBase(tfp_test_util.TestCase):
  """Tests correctness of the softclip transformation."""

  @parameterized.named_parameters(
      {'testcase_name': 'narrow', 'low': -2.,
       'high': 2., 'hinge_softness': 1.},
      {'testcase_name': 'wide', 'low': -200.,
       'high': 4., 'hinge_softness': 0.1},
      {'testcase_name': 'low_only', 'low': 100.,
       'high': None, 'hinge_softness': 10.},
      {'testcase_name': 'high_only', 'low': None,
       'high': 0.7, 'hinge_softness': 0.01},
      {'testcase_name': 'unconstrained', 'low': None,
       'high': None, 'hinge_softness': 1.})
  def test_constraints_are_satisfied(self, low, high, hinge_softness):
    xs = tf.convert_to_tensor(
        [-1e6, -100., -3., 0., 1.3, 4., 123., 1e6], self.dtype)

    low_tensor = (tf.convert_to_tensor(low, self.dtype)
                  if low is not None else low)
    high_tensor = (tf.convert_to_tensor(high, self.dtype)
                   if high is not None else high)
    hinge_softness = tf.convert_to_tensor(hinge_softness, self.dtype)
    b = tfb.SoftClip(low_tensor, high_tensor, hinge_softness)
    ys = self.evaluate(b.forward(xs))
    if low is not None:
      self.assertAllGreaterEqual(ys, low)
      self.assertAllClose(ys[0], low)
    if high is not None:
      self.assertAllLessEqual(ys, high)
      self.assertAllClose(ys[-1], high)

  def test_is_nearly_identity_within_range(self):
    xs = tf.convert_to_tensor(np.linspace(-3., 3., 20), dtype=self.dtype)
    b = tfb.SoftClip(tf.convert_to_tensor(-5., self.dtype),
                     tf.convert_to_tensor(5., self.dtype),
                     hinge_softness=0.01)
    ys = self.evaluate(b.forward(xs))
    self.assertAllClose(ys, xs)
    xs_inverted = self.evaluate(b.inverse(ys))
    self.assertAllClose(ys, xs_inverted)

  @tfp_test_util.jax_disable_test_missing_functionality('TF Exceptions')
  @tfp_test_util.numpy_disable_test_missing_functionality('TF Exceptions')
  def test_raises_exception_on_invalid_params(self):
    with self.assertRaisesRegex(tf.errors.InvalidArgumentError,
                                'Argument `high` must be greater than `low`'):
      b = tfb.SoftClip(5., 3., validate_args=True)

    with self.assertRaisesOpError('Argument `high` must be greater than `low`'):
      low = tf.Variable(0.)
      self.evaluate(low.initializer)
      b = tfb.SoftClip(low, 3., validate_args=True)
      with tf.control_dependencies([low.assign(5.)]):
        self.evaluate(b.forward(4.))

  @tfp_test_util.jax_disable_test_missing_functionality('TF Exceptions')
  @tfp_test_util.numpy_disable_test_missing_functionality('TF Exceptions')
  def test_raises_exception_on_invalid_input(self):
    b = tfb.SoftClip(3., 5., validate_args=True)
    with self.assertRaisesRegex(tf.errors.InvalidArgumentError,
                                'Input must be greater than `low`'):
      b.inverse(2.)

    with self.assertRaisesOpError('Input must be less than `high`'):
      dynamic_y = tf1.placeholder_with_default(input=7., shape=[])
      x = b.inverse(dynamic_y)
      self.evaluate(x)

  @tfp_test_util.jax_disable_variable_test
  def test_variable_gradients(self):
    b = tfb.SoftClip(low=tf.Variable(2.), high=tf.Variable(6.))
    with tf.GradientTape() as tape:
      y = b.forward(.1)
    self.assertAllNotNone(tape.gradient(y, b.trainable_variables))


class SoftClip32Test(_SoftClipBijectorBase):
  dtype = np.float32


class SoftClip64Test(_SoftClipBijectorBase):
  dtype = np.float64

del _SoftClipBijectorBase

if __name__ == '__main__':
  tf.test.main()
