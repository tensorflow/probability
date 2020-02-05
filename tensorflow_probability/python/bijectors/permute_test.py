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
"""Tests for Permute bijector."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports

import numpy as np
import tensorflow.compat.v1 as tf1
import tensorflow.compat.v2 as tf
from tensorflow_probability.python import bijectors as tfb
from tensorflow_probability.python.bijectors import bijector_test_util
from tensorflow_probability.python.internal import test_util


@test_util.test_all_tf_execution_regimes
class PermuteBijectorTest(test_util.TestCase):
  """Tests correctness of the Permute bijector."""

  def assertRaisesError(self, msg):
    return self.assertRaisesRegexp(Exception, msg)

  def setUp(self):
    self._rng = np.random.RandomState(42)

  def testBijector(self):
    expected_permutation = np.int32([2, 0, 1])
    expected_x = np.random.randn(4, 2, 3)
    expected_y = expected_x[..., expected_permutation]

    permutation_ph = tf1.placeholder_with_default(
        expected_permutation, shape=None)
    bijector = tfb.Permute(permutation=permutation_ph, validate_args=True)
    [
        permutation_,
        x_,
        y_,
        fldj,
        ildj,
    ] = self.evaluate([
        bijector.permutation,
        bijector.inverse(expected_y),
        bijector.forward(expected_x),
        bijector.forward_log_det_jacobian(expected_x, event_ndims=1),
        bijector.inverse_log_det_jacobian(expected_y, event_ndims=1),
    ])
    self.assertStartsWith(bijector.name, 'permute')
    self.assertAllEqual(expected_permutation, permutation_)
    self.assertAllClose(expected_y, y_, rtol=1e-6, atol=0)
    self.assertAllClose(expected_x, x_, rtol=1e-6, atol=0)
    self.assertAllClose(0., fldj, rtol=1e-6, atol=0)
    self.assertAllClose(0., ildj, rtol=1e-6, atol=0)

  def testBijectiveAndFinite(self):
    permutation = np.int32([2, 0, 1])
    x = np.random.randn(4, 2, 3)
    y = x[..., permutation]
    bijector = tfb.Permute(permutation=permutation, validate_args=True)
    bijector_test_util.assert_bijective_and_finite(
        bijector, x, y, eval_func=self.evaluate, event_ndims=1, rtol=1e-6,
        atol=0)

  def testBijectiveAndFiniteAxis(self):
    permutation = np.int32([1, 0])
    x = np.random.randn(4, 2, 3)
    y = x[..., permutation, :]
    bijector = tfb.Permute(
        permutation=permutation,
        axis=-2,
        validate_args=True)
    bijector_test_util.assert_bijective_and_finite(
        bijector, x, y, eval_func=self.evaluate, event_ndims=2, rtol=1e-6,
        atol=0)

  @test_util.jax_disable_test_missing_functionality(
      'Test specific to Keras with losing shape information.')
  def testPreservesShape(self):
    # TODO(b/131157549, b/131124359): Test should not be needed. Consider
    # deleting when underlying issue with constant eager tensors is fixed.
    permutation = [2, 1, 0]
    x = tf.keras.Input((3,), batch_size=None)
    bijector = tfb.Permute(permutation=permutation, axis=-1, validate_args=True)

    y = bijector.forward(x)
    self.assertAllEqual(y.shape.as_list(), [None, 3])

    inverse_y = bijector.inverse(x)
    self.assertAllEqual(inverse_y.shape.as_list(), [None, 3])

  def testNonPermutationAssertion(self):
    message = 'must contain exactly one of each of'
    with self.assertRaisesRegexp(Exception, message):
      permutation = np.int32([1, 0, 1])
      bijector = tfb.Permute(permutation=permutation, validate_args=True)
      x = np.random.randn(4, 2, 3)
      _ = self.evaluate(bijector.forward(x))

  def testVariableNonPermutationAssertion(self):
    message = 'must contain exactly one of each of'
    permutation = tf.Variable(np.int32([1, 0, 1]))
    self.evaluate(permutation.initializer)
    with self.assertRaisesRegexp(Exception, message):
      bijector = tfb.Permute(permutation=permutation, validate_args=True)
      x = np.random.randn(4, 2, 3)
      _ = self.evaluate(bijector.forward(x))

  def testModifiedVariableNonPermutationAssertion(self):
    message = 'must contain exactly one of each of'
    permutation = tf.Variable(np.int32([1, 0, 2]))
    self.evaluate(permutation.initializer)
    bijector = tfb.Permute(permutation=permutation, validate_args=True)
    with self.assertRaisesRegexp(Exception, message):
      with tf.control_dependencies([permutation.assign([1, 0, 1])]):
        x = np.random.randn(4, 2, 3)
        _ = self.evaluate(bijector.forward(x))

  def testPermutationTypeAssertion(self):
    message = 'should be `int`-like'
    with self.assertRaisesRegexp(Exception, message):
      permutation = np.float32([2, 0, 1])
      bijector = tfb.Permute(permutation=permutation, validate_args=True)
      x = np.random.randn(4, 2, 3)
      _ = self.evaluate(bijector.forward(x))


if __name__ == '__main__':
  tf.test.main()
