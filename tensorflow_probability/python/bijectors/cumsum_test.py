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
"""Cumsum Tests."""

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
class _CumsumBijectorTest(test_util.TestCase):
  """Tests correctness of the cumsum bijector."""

  def testInvalidAxis(self):
    with self.assertRaisesRegexp(ValueError,
                                 r'Argument `axis` must be negative.*'):
      tfb.Cumsum(axis=0, validate_args=True)
    with self.assertRaisesRegexp(TypeError,
                                 r'Argument `axis` is not an `int` type\.*'):
      tfb.Cumsum(axis=-1., validate_args=True)

  def testBijector(self):
    self._checkBijectorInAllDims(np.arange(5.))
    self._checkBijectorInAllDims(np.reshape([np.arange(5.)] * 2, [5, 2]))
    self._checkBijectorInAllDims(np.reshape([np.arange(5.)] * 4, [5, 2, 2]))

  def testBijectiveAndFinite(self):
    bijector = tfb.Cumsum(validate_args=True)
    x = np.linspace(-10, 10, num=10).astype(np.float32)
    y = np.cumsum(x, axis=-1)
    bijector_test_util.assert_bijective_and_finite(
        bijector, x, y, eval_func=self.evaluate, event_ndims=1)

  @test_util.numpy_disable_gradient_test
  def testJacobian(self):
    self._checkEqualTheoreticalFldj(np.expand_dims(np.arange(5.), -1))
    self._checkEqualTheoreticalFldj(np.reshape([np.arange(5.)] * 2, [5, 2]))
    self._checkEqualTheoreticalFldj(
        np.expand_dims(np.reshape([np.arange(5.)] * 2, [5, 2]), -1))
    self._checkEqualTheoreticalFldj(np.reshape([np.arange(5.)] * 4, [5, 2, 2]))
    self._checkEqualTheoreticalFldj(
        np.expand_dims(np.reshape([np.arange(5.)] * 4, [5, 2, 2]), -1))

  def _checkBijectorInAllDims(self, x):
    """Helper for `testBijector`."""
    x = self._build_tensor(x)
    for axis in range(-self.evaluate(tf.rank(x)), 0):
      bijector = tfb.Cumsum(axis=axis, validate_args=True)
      self.assertStartsWith(bijector.name, 'cumsum')

      y = tf.cumsum(x, axis=axis)
      self.assertAllClose(y, self.evaluate(bijector.forward(x)))
      self.assertAllClose(x, self.evaluate(bijector.inverse(y)))

  def _checkEqualTheoreticalFldj(self, x):
    """Helper for `testJacobian`."""
    event_ndims = len(x.shape) - 1
    self.assertGreaterEqual(event_ndims, 1)

    bijector = tfb.Cumsum(axis=-event_ndims, validate_args=True)
    fldj = bijector.forward_log_det_jacobian(
        self._build_tensor(x), event_ndims=event_ndims)
    fldj_theoretical = bijector_test_util.get_fldj_theoretical(
        bijector, x, event_ndims=event_ndims)
    fldj_, fldj_theoretical_ = self.evaluate([fldj, fldj_theoretical])
    self.assertAllEqual(np.zeros_like(fldj_), fldj_)
    self.assertAllClose(np.zeros_like(fldj_theoretical_), fldj_theoretical_)

  def _build_tensor(self, ndarray, dtype=None):
    # Enforce parameterized dtype and static/dynamic testing.
    ndarray = np.asarray(ndarray).astype(
        dtype if dtype is not None else self.dtype)
    return tf1.placeholder_with_default(
        ndarray, shape=ndarray.shape if self.use_static_shape else None)


class CumsumBijectorTestWithStaticShape(_CumsumBijectorTest):
  dtype = np.float32
  use_static_shape = True


class CumsumBijectorTestWithDynamicShape(_CumsumBijectorTest):
  dtype = np.float32
  use_static_shape = False


del _CumsumBijectorTest  # Don't run tests for the base class.

if __name__ == '__main__':
  tf.test.main()
