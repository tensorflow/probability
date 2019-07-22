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
"""Tests for TensorFlow Probability TestCase class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
import numpy as np
import tensorflow as tf

from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import test_case
from tensorflow.python.framework import test_util  # pylint: disable=g-direct-tensorflow-import


class _TestCaseTest(object):

  def setUp(self):
    np.random.seed(932)

  def test_assert_all_finite_input_finite(self):
    minval = tf.constant(dtype_util.min(self.dtype), dtype=self.dtype)
    maxval = tf.constant(dtype_util.max(self.dtype), dtype=self.dtype)

    # This tests if the minimum value for the dtype is detected as finite.
    self.assertAllFinite(minval)

    # This tests if the maximum value for the dtype is detected as finite.
    self.assertAllFinite(maxval)

    # This tests if a rank 3 `Tensor` with entries in the range
    # [0.4*minval, 0.4*maxval] is detected as finite.
    # The choice of range helps to avoid overflows or underflows
    # in tf.linspace calculations.
    num_elem = 1000
    shape = (10, 10, 10)
    a = tf.reshape(tf.linspace(0.4*minval, 0.4*maxval, num_elem), shape)
    self.assertAllFinite(a)

  def test_assert_all_finite_input_nan(self):
    # This tests if np.nan is detected as non-finite.
    num_elem = 1000
    shape = (10, 10, 10)
    a = np.linspace(0., 1., num_elem)
    a[50] = np.nan
    a = tf.reshape(tf.convert_to_tensor(value=a, dtype=self.dtype), shape)
    with self.assertRaisesRegexp(AssertionError, "Arrays are not equal"):
      self.assertAllFinite(a)

  def test_assert_all_finite_input_inf(self):
    # This tests if np.inf is detected as non-finite.
    num_elem = 1000
    shape = (10, 10, 10)
    a = np.linspace(0., 1., num_elem)
    a[100] = np.inf
    a = tf.reshape(tf.convert_to_tensor(value=a, dtype=self.dtype), shape)
    with self.assertRaisesRegexp(AssertionError, "Arrays are not equal"):
      self.assertAllFinite(a)

  def test_assert_all_finite_input_py_literal(self):
    # This tests if finite Python literals are detected as finite.
    for a in [1, 3., -7.1e-12]:
      self.assertAllFinite(a)

    b = [0, 1.45e17, 0x2a]
    self.assertAllFinite(b)

    c = (1, 2., 3, 4)
    self.assertAllFinite(c)

  def test_assert_all_nan_input_all_nan(self):
    a = tf.convert_to_tensor(
        value=np.full((10, 10, 10), np.nan), dtype=self.dtype)
    self.assertAllNan(a)

  def test_assert_all_nan_input_some_nan(self):
    a = np.random.rand(10, 10, 10)
    a[1, :, :] = np.nan
    a = tf.convert_to_tensor(value=a, dtype=self.dtype)
    with self.assertRaisesRegexp(AssertionError, "Arrays are not equal"):
      self.assertAllNan(a)

  def test_assert_all_nan_input_numpy_rand(self):
    a = np.random.rand(10, 10, 10).astype(dtype_util.as_numpy_dtype(self.dtype))
    with self.assertRaisesRegexp(AssertionError, "Arrays are not equal"):
      self.assertAllNan(a)

  def test_assert_all_nan_input_inf(self):
    a = tf.convert_to_tensor(
        value=np.full((10, 10, 10), np.inf), dtype=self.dtype)
    with self.assertRaisesRegexp(AssertionError, "Arrays are not equal"):
      self.assertAllNan(a)

  def test_assert_all_nan_input_placeholder_with_default(self):
    all_nan = np.full((10, 10, 10),
                      np.nan).astype(dtype_util.as_numpy_dtype(self.dtype))
    a = tf.compat.v1.placeholder_with_default(
        input=all_nan, shape=all_nan.shape)
    self.assertAllNan(a)

  def test_assert_all_are_not_none(self):
    no_nones = [1, 2, 3]
    self.assertAllNotNone(no_nones)

    has_nones = [1, 2, None]
    with self.assertRaisesRegexp(
        AssertionError,
        r"Expected no entry to be `None` but found `None` in positions \[2\]"):
      self.assertAllNotNone(has_nones)


@test_util.run_all_in_graph_and_eager_modes
class TestCaseTestFloat32(test_case.TestCase, _TestCaseTest):
  dtype = tf.float32


@test_util.run_all_in_graph_and_eager_modes
class TestCaseTestFloat64(test_case.TestCase, _TestCaseTest):
  dtype = tf.float64


if __name__ == "__main__":
  tf.test.main()
