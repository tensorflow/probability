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

import dataclasses

# Dependency imports
from absl.testing import parameterized
import numpy as np
import tensorflow.compat.v2 as tf

from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import test_util


NUMPY_MODE = False
JAX_MODE = False


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

    fake_tensor = dataclasses.make_dataclass('fake_tensor', ['dtype'])
    self.assertEqual(
        tf.float16, dtype_util.common_dtype(
            [fake_tensor(dtype=None), None, x], dtype_hint=tf.float32))
    self.assertEqual(
        tf.float32, dtype_util.common_dtype(
            [fake_tensor(dtype=tf.float32), None]))

  def testCommonDtypeFromLinop(self):
    x = tf.linalg.LinearOperatorDiag(tf.ones(3, tf.float16))
    self.assertEqual(
        tf.float16, dtype_util.common_dtype([x], dtype_hint=tf.float32))

  def testCommonStructuredDtype(self):
    structured_dtype_obj = dataclasses.make_dataclass(
        'structured_dtype_obj', ['dtype'])
    x = structured_dtype_obj({'a': tf.float32, 'b': (None, None)})
    y = structured_dtype_obj({'a': None, 'b': (None, tf.float64)})
    z = structured_dtype_obj({'a': None, 'b': (None, None)})
    w = structured_dtype_obj(None)

    # Check that structured dtypes unify correctly.
    self.assertAllEqualNested(
        dtype_util.common_dtype([w, x, y, z]),
        {'a': tf.float32, 'b': (None, tf.float64)})

    # Check that dict `args` works and that `dtype_hint` works.
    dtype_hint = {'a': tf.int32, 'b': (tf.int32, None)}
    self.assertAllEqualNested(
        dtype_util.common_dtype(
            {'x': x, 'y': y, 'z': z}, dtype_hint=dtype_hint),
        {'a': tf.float32, 'b': (tf.int32, tf.float64)})
    self.assertAllEqualNested(
        dtype_util.common_dtype([w], dtype_hint=dtype_hint),
        dtype_hint)

    # Check that non-nested dtype_hint broadcasts.
    self.assertAllEqualNested(
        dtype_util.common_dtype([y, z], dtype_hint=tf.int32),
        {'a': tf.int32, 'b': (tf.int32, tf.float64)})

    # Check that structured `dtype_hint` behaves as expected.
    s = {'a': [tf.ones([3], tf.float32), 4.],
         'b': (np.float64(2.), None)}
    self.assertAllEqualNested(
        dtype_util.common_dtype([x, s], dtype_hint=z.dtype),
        {'a': tf.float32, 'b': (tf.float64, None)})
    self.assertAllEqualNested(
        dtype_util.common_dtype([y, s], dtype_hint=z.dtype),
        {'a': tf.float32, 'b': (tf.float64, tf.float64)})

    t = {'a': [[1., 2., 3.]], 'b': {'c': np.float64(1.), 'd': np.float64(2.)}}
    self.assertAllEqualNested(
        dtype_util.common_dtype(
            [w, t],
            dtype_hint={'a': tf.float32, 'b': tf.float32}),
        {'a': tf.float32, 'b': tf.float64})

    with self.assertRaisesRegex(ValueError, 'the same nested structure'):
      dtype_util.common_dtype([x, s])

    with self.assertRaisesRegex(ValueError, 'the same nested structure'):
      dtype_util.common_dtype([x, t], dtype_hint={'a': None, 'b': None})

    with self.assertRaisesRegex(
        ValueError, 'must have a shallow structure that matches'):
      dtype_util.common_dtype([{'a': np.float64(3.), 'b': [3, 2], 'c': 3.}],
                              dtype_hint=dtype_hint)

    if not dtype_util.SKIP_DTYPE_CHECKS:
      with self.assertRaisesRegex(
          TypeError, 'contains elements with incompatible dtypes'):
        dtype_util.common_dtype(
            [{'a': np.float64(3.),
              'b': (tf.ones([3], dtype=tf.float32),
                    tf.ones([2], dtype=tf.float16))}],
            dtype_hint={'a': None, 'b': None})

      with self.assertRaisesRegex(TypeError, 'Found incompatible dtypes'):
        dtype_util.common_dtype([x, {'a': np.float64(3.), 'b': (3, 2)}],
                                dtype_hint=dtype_hint)

      with self.assertRaisesRegex(TypeError, 'Found incompatible dtypes'):
        dtype_util.common_dtype([x, structured_dtype_obj(dtype_hint)])

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

  @parameterized.parameters(
      ([1], None, None),
      ([1], tf.float32, None),
      ([1], None, tf.float32),
      ([1], tf.float32, tf.float64),
      (np.int64, None, None),
      (np.int64, None, tf.float32),
      (np.int64, tf.float32, None),
      (np.int64, tf.float32, tf.float64),
      (tf.int64, None, None),
      (tf.int64, None, tf.float32),
      (tf.float32, tf.float32, None),
      (tf.float32, tf.float32, tf.float64))
  @test_util.disable_test_for_backend(
      disable_numpy=True,
      reason='`convert_to_tensor` respects array dtypes in numpy backend.')
  def testConvertToDtype(self, tensor_or_dtype, dtype, dtype_hint):
    if np.issctype(tensor_or_dtype):
      example_tensor = np.zeros([], tensor_or_dtype)
    elif isinstance(tensor_or_dtype, tf.DType):
      example_tensor = tf.zeros([], tensor_or_dtype)
    else:
      example_tensor = tensor_or_dtype

    # Try with the original argument.
    self.assertEqual(
        tf.convert_to_tensor(example_tensor, dtype, dtype_hint).dtype,
        dtype_util.convert_to_dtype(tensor_or_dtype, dtype, dtype_hint))
    # Try with a concrete value.
    self.assertEqual(
        tf.convert_to_tensor(example_tensor, dtype, dtype_hint).dtype,
        dtype_util.convert_to_dtype(example_tensor, dtype, dtype_hint))

  @parameterized.parameters(
      (tf.int64, tf.float32, None),
      (tf.int64, tf.float32, tf.float64))
  @test_util.disable_test_for_backend(
      disable_numpy=True,
      disable_jax=True,
      reason='`convert_to_tensor` only raises in TF mode.')
  def testConvertToDTypeRaises(self, tensor_or_dtype, dtype, dtype_hint):
    if np.issctype(tensor_or_dtype):
      example_tensor = np.zeros([], tensor_or_dtype)
    elif isinstance(tensor_or_dtype, tf.DType):
      example_tensor = tf.zeros([], tensor_or_dtype)
    else:
      example_tensor = tensor_or_dtype

    with self.assertRaisesRegex(TypeError, 'Found incompatible dtypes'):
      dtype_util.convert_to_dtype(tensor_or_dtype, dtype, dtype_hint)
    with self.assertRaisesRegex(TypeError, 'Found incompatible dtypes'):
      dtype_util.convert_to_dtype(example_tensor, dtype, dtype_hint)

  @parameterized.parameters(np.float16, np.float32, np.float64, tf.float16,
                            tf.float32, tf.float64)
  def testEps(self, dtype):
    self.assertEqual(
        dtype_util.eps(dtype).dtype, dtype_util.as_numpy_dtype(dtype))


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

  @parameterized.named_parameters(
      ('float32', tf.float32, True),
      ('bfloat16', 'bfloat16', True),
      ('not_int8', tf.int8, False))
  def test_is_floating(self, dtype, expected):
    if NUMPY_MODE and dtype == 'bfloat16':
      self.skipTest('No bfloat16 in numpy')
    self.assertEqual(dtype_util.is_floating(dtype), expected)

if __name__ == '__main__':
  test_util.main()
