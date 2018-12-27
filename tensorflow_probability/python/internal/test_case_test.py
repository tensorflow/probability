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

from tensorflow_probability.python.internal import test_case
tfe = tf.contrib.eager


@tfe.run_all_tests_in_graph_and_eager_modes
class _TestCaseTest(object):

  def setUp(self):
    np.random.seed(932)

  def test_assert_all_finite_input_finite(self):
    minval = tf.constant(self.dtype.min, dtype=self.dtype)
    maxval = tf.constant(self.dtype.max, dtype=self.dtype)

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
    a = tf.reshape(tf.convert_to_tensor(a, dtype=self.dtype), shape)
    with self.assertRaisesRegexp(AssertionError, "Arrays are not equal"):
      self.assertAllFinite(a)

  def test_assert_all_finite_input_inf(self):
    # This tests if np.inf is detected as non-finite.
    num_elem = 1000
    shape = (10, 10, 10)
    a = np.linspace(0., 1., num_elem)
    a[100] = np.inf
    a = tf.reshape(tf.convert_to_tensor(a, dtype=self.dtype), shape)
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
        np.full((10, 10, 10), np.nan), dtype=self.dtype)
    self.assertAllNan(a)

  def test_assert_all_nan_input_some_nan(self):
    a = np.random.rand(10, 10, 10)
    a[1, :, :] = np.nan
    a = tf.convert_to_tensor(a, dtype=self.dtype)
    with self.assertRaisesRegexp(AssertionError, "Arrays are not equal"):
      self.assertAllNan(a)

  def test_assert_all_nan_input_numpy_rand(self):
    a = np.random.rand(10, 10, 10).astype(self.dtype.as_numpy_dtype)
    with self.assertRaisesRegexp(AssertionError, "Arrays are not equal"):
      self.assertAllNan(a)

  def test_assert_all_nan_input_inf(self):
    a = tf.convert_to_tensor(
        np.full((10, 10, 10), np.inf), dtype=self.dtype)
    with self.assertRaisesRegexp(AssertionError, "Arrays are not equal"):
      self.assertAllNan(a)

  def test_assert_all_nan_input_placeholder_with_default(self):
    all_nan = np.full((10, 10, 10), np.nan).astype(self.dtype.as_numpy_dtype)
    a = tf.placeholder_with_default(input=all_nan, shape=all_nan.shape)
    self.assertAllNan(a)

  def test_compute_gradients_no_initial_gradients(self):
    x_ = np.random.rand(1000, 100).astype(self.dtype.as_numpy_dtype)
    x = tf.placeholder_with_default(x_, shape=x_.shape)
    y_ = np.random.rand(100, 1).astype(self.dtype.as_numpy_dtype)
    y = tf.placeholder_with_default(y_, shape=y_.shape)
    f = lambda x, y: tf.matmul(x, y)  # pylint: disable=unnecessary-lambda
    dfdx, dfdy = self.compute_gradients(f, [x, y])
    expected_dfdx = np.transpose(y_) + np.zeros_like(x_)
    expected_dfdy = np.transpose(np.sum(x_, axis=0, keepdims=True))
    self.assertAllClose(dfdx, expected_dfdx, atol=0., rtol=1e-4)
    self.assertAllClose(dfdy, expected_dfdy, atol=0., rtol=1e-4)

  def test_compute_gradients_with_initial_gradients(self):
    x_ = np.random.rand(1000, 100).astype(self.dtype.as_numpy_dtype)
    x = tf.placeholder_with_default(x_, shape=x_.shape)
    y_ = np.random.rand(100, 1).astype(self.dtype.as_numpy_dtype)
    y = tf.placeholder_with_default(y_, shape=y_.shape)
    f = lambda x, y: tf.matmul(x, y)  # pylint: disable=unnecessary-lambda
    init_grad_ = np.random.rand(1000, 1).astype(self.dtype.as_numpy_dtype)
    init_grad = tf.placeholder_with_default(init_grad_, shape=init_grad_.shape)
    dfdx, dfdy = self.compute_gradients(f, [x, y], grad_ys=init_grad)
    expected_dfdx = (np.transpose(y_) + np.zeros_like(x_)) * init_grad_
    expected_dfdy = np.transpose(np.sum(x_ * init_grad_, axis=0, keepdims=True))
    self.assertAllClose(dfdx, expected_dfdx, atol=0., rtol=1e-4)
    self.assertAllClose(dfdy, expected_dfdy, atol=0., rtol=1e-4)


class TestCaseTestFloat32(test_case.TestCase, _TestCaseTest):
  dtype = tf.float32


class TestCaseTestFloat64(test_case.TestCase, _TestCaseTest):
  dtype = tf.float64


if __name__ == "__main__":
  tf.test.main()
