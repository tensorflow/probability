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
"""Tests for sparse ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
import numpy as np

import tensorflow.compat.v1 as tf1
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp


from tensorflow_probability.python.internal import test_util


def _assert_sparse_tensor_value(test_case_instance, expected, actual):
  test_case_instance.assertEqual(np.int64, np.array(actual.indices).dtype)
  test_case_instance.assertAllEqual(expected.indices, actual.indices)

  test_case_instance.assertEqual(
      np.array(expected.values).dtype, np.array(actual.values).dtype)
  test_case_instance.assertAllEqual(expected.values, actual.values)

  test_case_instance.assertEqual(np.int64, np.array(actual.dense_shape).dtype)
  test_case_instance.assertAllEqual(expected.dense_shape, actual.dense_shape)


@test_util.test_all_tf_execution_regimes
class SparseTest(test_util.TestCase):
  # Copied (with modifications) from:
  # tensorflow/contrib/layers/python/ops/sparse_ops.py.

  def test_dense_to_sparse_1d(self):
    st = tfp.math.dense_to_sparse([1, 0, 2, 0])
    result = self.evaluate(st)
    self.assertEqual(result.indices.dtype, np.int64)
    self.assertEqual(result.values.dtype, np.int32)
    self.assertEqual(result.dense_shape.dtype, np.int64)
    self.assertAllEqual([[0], [2]], result.indices)
    self.assertAllEqual([1, 2], result.values)
    self.assertAllEqual([4], result.dense_shape)

  def test_dense_to_sparse_1d_float(self):
    st = tfp.math.dense_to_sparse([1.5, 0.0, 2.3, 0.0])
    result = self.evaluate(st)
    self.assertEqual(result.indices.dtype, np.int64)
    self.assertEqual(result.values.dtype, np.float32)
    self.assertEqual(result.dense_shape.dtype, np.int64)
    self.assertAllEqual([[0], [2]], result.indices)
    self.assertAllClose([1.5, 2.3], result.values)
    self.assertAllEqual([4], result.dense_shape)

  def test_dense_to_sparse_1d_bool(self):
    st = tfp.math.dense_to_sparse([True, False, True, False])
    result = self.evaluate(st)
    self.assertEqual(result.indices.dtype, np.int64)
    self.assertEqual(result.values.dtype, np.bool)
    self.assertEqual(result.dense_shape.dtype, np.int64)
    self.assertAllEqual([[0], [2]], result.indices)
    self.assertAllEqual([True, True], result.values)
    self.assertAllEqual([4], result.dense_shape)

  def test_dense_to_sparse_1d_str(self):
    st = tfp.math.dense_to_sparse([b'qwe', b'', b'ewq', b''])
    result = self.evaluate(st)
    self.assertEqual(result.indices.dtype, np.int64)
    self.assertEqual(result.values.dtype, np.object)
    self.assertEqual(result.dense_shape.dtype, np.int64)
    self.assertAllEqual([[0], [2]], result.indices)
    self.assertAllEqual([b'qwe', b'ewq'], result.values)
    self.assertAllEqual([4], result.dense_shape)

  def test_dense_to_sparse_1d_str_special_ignore(self):
    st = tfp.math.dense_to_sparse(
        [b'qwe', b'', b'ewq', b''], ignore_value=b'qwe')
    result = self.evaluate(st)
    self.assertEqual(result.indices.dtype, np.int64)
    self.assertEqual(result.values.dtype, np.object)
    self.assertEqual(result.dense_shape.dtype, np.int64)
    self.assertAllEqual([[1], [2], [3]], result.indices)
    self.assertAllEqual([b'', b'ewq', b''], result.values)
    self.assertAllEqual([4], result.dense_shape)

  def test_dense_to_sparse_2d(self):
    st = tfp.math.dense_to_sparse([[1, 2, 0, 0], [3, 4, 5, 0]])
    result = self.evaluate(st)
    self.assertAllEqual([[0, 0], [0, 1], [1, 0], [1, 1], [1, 2]],
                        result.indices)
    self.assertAllEqual([1, 2, 3, 4, 5], result.values)
    self.assertAllEqual([2, 4], result.dense_shape)

  def test_dense_to_sparse_3d(self):
    st = tfp.math.dense_to_sparse(
        [[[1, 2, 0, 0],
          [3, 4, 5, 0]],
         [[7, 8, 0, 0],
          [9, 0, 0, 0]]])
    result = self.evaluate(st)
    self.assertAllEqual(
        [[0, 0, 0],
         [0, 0, 1],
         [0, 1, 0],
         [0, 1, 1],
         [0, 1, 2],
         [1, 0, 0],
         [1, 0, 1],
         [1, 1, 0]],
        result.indices)
    self.assertAllEqual([1, 2, 3, 4, 5, 7, 8, 9], result.values)
    self.assertAllEqual([2, 2, 4], result.dense_shape)

  def test_dense_to_sparse_unknown_1d_shape(self):
    tensor = tf1.placeholder_with_default(
        np.array([0, 100, 0, 3], np.int32), shape=[None])
    st = tfp.math.dense_to_sparse(tensor)
    result = self.evaluate(st)
    self.assertAllEqual([[1], [3]], result.indices)
    self.assertAllEqual([100, 3], result.values)
    self.assertAllEqual([4], result.dense_shape)

  def test_dense_to_sparse_unknown_3d_shape(self):
    tensor = tf1.placeholder_with_default(
        np.array([[[1, 2, 0, 0], [3, 4, 5, 0]], [[7, 8, 0, 0], [9, 0, 0, 0]]],
                 np.int32),
        shape=[None, None, None])
    st = tfp.math.dense_to_sparse(tensor)
    result = self.evaluate(st)
    self.assertAllEqual(
        [[0, 0, 0],
         [0, 0, 1],
         [0, 1, 0],
         [0, 1, 1],
         [0, 1, 2],
         [1, 0, 0],
         [1, 0, 1],
         [1, 1, 0]],
        result.indices)
    self.assertAllEqual([1, 2, 3, 4, 5, 7, 8, 9], result.values)
    self.assertAllEqual([2, 2, 4], result.dense_shape)

  def test_dense_to_sparse_unknown_rank(self):
    ph = tf1.placeholder_with_default(
        np.array([[1, 2, 0, 0], [3, 4, 5, 0]], np.int32), shape=None)
    st = tfp.math.dense_to_sparse(ph)
    result = self.evaluate(st)
    self.assertAllEqual(
        [[0, 0],
         [0, 1],
         [1, 0],
         [1, 1],
         [1, 2]],
        result.indices)
    self.assertAllEqual([1, 2, 3, 4, 5], result.values)
    self.assertAllEqual([2, 4], result.dense_shape)


if __name__ == '__main__':
  tf.test.main()
