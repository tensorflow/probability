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
"""Tests for dtype_util."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

# Dependency imports
from absl.testing import parameterized
import numpy as np
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp

from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import test_util


@test_util.test_all_tf_execution_regimes
class DtypeUtilTest(test_util.TestCase):

  def testIsInteger(self):
    self.assertFalse(dtype_util.is_integer(np.float64))

  def testNoModifyArgsList(self):
    x = tf.ones(3, tf.float32)
    y = tf.zeros(4, tf.float32)
    lst = [x, y]
    self.assertEqual(tf.float32, dtype_util.common_dtype(lst))
    self.assertLen(lst, 2)

  def testCommonDtypeAcceptsNone(self):
    self.assertEqual(
        tf.float16, dtype_util.common_dtype(
            [None], dtype_hint=tf.float16))

    x = tf.ones(3, tf.float16)
    self.assertEqual(
        tf.float16, dtype_util.common_dtype(
            [x, None], dtype_hint=tf.float32))

    fake_tensor = collections.namedtuple('fake_tensor', ['dtype'])
    self.assertEqual(
        tf.float16, dtype_util.common_dtype(
            [fake_tensor(dtype=None), None, x], dtype_hint=tf.float32))

  def testCommonDtypeFromLinop(self):
    x = tf.linalg.LinearOperatorDiag(tf.ones(3, tf.float16))
    self.assertEqual(
        tf.float16, dtype_util.common_dtype([x], dtype_hint=tf.float32))

  def testCommonDtypeFromEdRV(self):
    # Only test Edward2 if it's able to be imported (not possible in jax/numpy
    # modes).
    try:
      ed = tfp.experimental.edward2
    except AttributeError:
      self.skipTest('No edward2 module present in jax/numpy modes.')
    # As in tensorflow_probability github issue #221
    x = ed.Dirichlet(np.ones(3, dtype='float64'))
    self.assertEqual(
        tf.float64, dtype_util.common_dtype([x], dtype_hint=tf.float32))

  @parameterized.named_parameters(
      dict(testcase_name='Float32', dtype=tf.float32,
           expected_minval=np.float32(-3.4028235e+38)),
      dict(testcase_name='Float64', dtype=tf.float64,
           expected_minval=np.float64(-1.7976931348623157e+308)),
  )
  def testMin(self, dtype, expected_minval):
    self.assertEqual(dtype_util.min(dtype), expected_minval)

  @parameterized.named_parameters(
      dict(testcase_name='Float32', dtype=tf.float32,
           expected_maxval=np.float32(3.4028235e+38)),
      dict(testcase_name='Float64', dtype=tf.float64,
           expected_maxval=np.float64(1.7976931348623157e+308)),
  )
  def testMax(self, dtype, expected_maxval):
    self.assertEqual(dtype_util.max(dtype), expected_maxval)


@test_util.test_all_tf_execution_regimes
class FloatDTypeTest(test_util.TestCase):

  def test_assert_same_float_dtype(self):
    self.assertIs(tf.float32, dtype_util.assert_same_float_dtype(None, None))
    self.assertIs(tf.float32, dtype_util.assert_same_float_dtype([], None))
    self.assertIs(
        tf.float32, dtype_util.assert_same_float_dtype([], tf.float32))
    self.assertIs(
        tf.float32, dtype_util.assert_same_float_dtype(None, tf.float32))
    self.assertIs(
        tf.float32, dtype_util.assert_same_float_dtype([None, None], None))
    self.assertIs(
        tf.float32,
        dtype_util.assert_same_float_dtype([None, None], tf.float32))

    const_float = tf.constant(3.0, dtype=tf.float32)
    self.assertIs(
        tf.float32,
        dtype_util.assert_same_float_dtype([const_float], tf.float32))
    self.assertRaises(ValueError, dtype_util.assert_same_float_dtype,
                      [const_float], tf.int32)

    if not hasattr(tf, 'SparseTensor'):
      # No SparseTensor in numpy/jax mode.
      return
    sparse_float = tf.SparseTensor(
        tf.constant([[111], [232]], tf.int64),
        tf.constant([23.4, -43.2], tf.float32),
        tf.constant([500], tf.int64))
    self.assertIs(
        tf.float32,
        dtype_util.assert_same_float_dtype([sparse_float], tf.float32))
    self.assertRaises(ValueError, dtype_util.assert_same_float_dtype,
                      [sparse_float], tf.int32)
    self.assertRaises(ValueError, dtype_util.assert_same_float_dtype,
                      [const_float, None, sparse_float], tf.float64)

    self.assertIs(
        tf.float32,
        dtype_util.assert_same_float_dtype([const_float, sparse_float]))
    self.assertIs(
        tf.float32,
        dtype_util.assert_same_float_dtype(
            [const_float, sparse_float], tf.float32))

    const_int = tf.constant(3, dtype=tf.int32)
    self.assertRaises(ValueError, dtype_util.assert_same_float_dtype,
                      [sparse_float, const_int])
    self.assertRaises(ValueError, dtype_util.assert_same_float_dtype,
                      [sparse_float, const_int], tf.int32)
    self.assertRaises(ValueError, dtype_util.assert_same_float_dtype,
                      [sparse_float, const_int], tf.float32)
    self.assertRaises(ValueError, dtype_util.assert_same_float_dtype,
                      [const_int])

  def test_size(self):
    self.assertEqual(dtype_util.size(tf.int32), 4)
    self.assertEqual(dtype_util.size(tf.int64), 8)
    self.assertEqual(dtype_util.size(tf.float32), 4)
    self.assertEqual(dtype_util.size(tf.float64), 8)

    self.assertEqual(dtype_util.size(np.int32), 4)
    self.assertEqual(dtype_util.size(np.int64), 8)
    self.assertEqual(dtype_util.size(np.float32), 4)
    self.assertEqual(dtype_util.size(np.float64), 8)


if __name__ == '__main__':
  tf.test.main()
