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
import tensorflow as tf
from tensorflow_probability.python import bijectors as tfb

from tensorflow_probability.python.bijectors import bijector_test_util


class PermuteBijectorTest(tf.test.TestCase):
  """Tests correctness of the Permute bijector."""

  def setUp(self):
    self._rng = np.random.RandomState(42)

  def testBijector(self):
    expected_permutation = np.int32([2, 0, 1])
    expected_x = np.random.randn(4, 2, 3)
    expected_y = expected_x[..., expected_permutation]

    with self.test_session() as sess:
      permutation_ph = tf.placeholder(dtype=tf.int32)
      bijector = tfb.Permute(permutation=permutation_ph, validate_args=True)
      [
          permutation_,
          x_,
          y_,
          fldj,
          ildj,
      ] = sess.run([
          bijector.permutation,
          bijector.inverse(expected_y),
          bijector.forward(expected_x),
          bijector.forward_log_det_jacobian(expected_x, event_ndims=1),
          bijector.inverse_log_det_jacobian(expected_y, event_ndims=1),
      ], feed_dict={permutation_ph: expected_permutation})
      self.assertEqual("permute", bijector.name)
      self.assertAllEqual(expected_permutation, permutation_)
      self.assertAllClose(expected_y, y_, rtol=1e-6, atol=0)
      self.assertAllClose(expected_x, x_, rtol=1e-6, atol=0)
      self.assertAllClose(0., fldj, rtol=1e-6, atol=0)
      self.assertAllClose(0., ildj, rtol=1e-6, atol=0)

  def testRaisesOpError(self):
    with self.test_session() as sess:
      with self.assertRaisesOpError("Permutation over `d` must contain"):
        permutation_ph = tf.placeholder(dtype=tf.int32)
        bijector = tfb.Permute(permutation=permutation_ph, validate_args=True)
        sess.run(bijector.inverse([1.]),
                 feed_dict={permutation_ph: [1, 2]})

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


if __name__ == "__main__":
  tf.test.main()
