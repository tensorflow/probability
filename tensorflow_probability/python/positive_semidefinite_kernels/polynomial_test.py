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
"""Tests for Polynomial and Linear."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from absl.testing import parameterized
import numpy as np


import tensorflow as tf
from tensorflow_probability import positive_semidefinite_kernels as psd_kernels
from tensorflow.python.framework import test_util


@test_util.run_all_in_graph_and_eager_modes
class PolynomialTest(tf.test.TestCase, parameterized.TestCase):
  """Test the Polynomial kernel."""

  def test_mismatched_float_types_are_bad(self):
    with self.assertRaises(TypeError):
      psd_kernels.Polynomial(
          bias_variance=np.float32(1.),
          slope_variance=np.float64(1.),
          exponent=1.
      )

  def testFloat32Fallback(self):
    # Should be OK (float32 fallback).
    self.polynomial = psd_kernels.Polynomial(
        bias_variance=0, slope_variance=1, exponent=1)
    # Should be OK.
    psd_kernels.Polynomial(
        bias_variance=np.float32(1.),
        slope_variance=1.,
        exponent=1.)

  def testValidateArgsNonPositiveAreBad(self):
    with self.assertRaisesOpError('Condition x > 0 did not hold'):
      k = psd_kernels.Polynomial(
          bias_variance=-1.,
          validate_args=True)
      self.evaluate(k.bias_variance)
    with self.assertRaisesOpError('Condition x > 0 did not hold'):
      k = psd_kernels.Polynomial(
          slope_variance=-1.,
          validate_args=True)
      self.evaluate(k.slope_variance)
    with self.assertRaisesOpError('Condition x > 0 did not hold'):
      k = psd_kernels.Polynomial(
          exponent=-1.,
          validate_args=True)
      self.evaluate(k.exponent)

  def testValidateArgsNoneIsOk(self):
    # No exception expected
    k = psd_kernels.Polynomial(
        bias_variance=None,
        slope_variance=None,
        exponent=None,
        validate_args=True)
    self.evaluate(k.apply([[1.]], [[1.]]))

  def testNoneShapes(self):
    k = psd_kernels.Polynomial(
        bias_variance=np.reshape(np.arange(12.), [2, 3, 2]))
    self.assertEqual([2, 3, 2], k.batch_shape.as_list())

  @parameterized.named_parameters(
      ('Shape [] kernel', 2., 2., 2., []),
      ('Shape [1] kernel', [2.], [2.], [2.], [1]),
      ('Shape [2] kernel', [1., 2.], [1., 2.], [1., 2.], [2]),
      ('Shape [2, 1] kernel', [[1.], [2.]], [[1.], [2.]], [[1.], [2.]],
       [2, 1]),
      ('Shape [2, 1] broadcast kernel', 2., [2.], [[1.], [2.]], [2, 1]))
  def testBatchShape(self, bias_variance, slope_variance,
                           exponent, shape):
    k = psd_kernels.Polynomial(
        bias_variance=bias_variance,
        slope_variance=slope_variance,
        exponent=exponent,
        validate_args=True)
    self.assertAllEqual(k.batch_shape.as_list(), shape)
    self.assertAllEqual(self.evaluate(k.batch_shape_tensor()), shape)

  def testFloat32(self):
    # No exception expected
    k = psd_kernels.Polynomial(
        bias_variance=0.,
        slope_variance=1.,
        exponent=1.,
        feature_ndims=1)
    x = np.ones([5, 3], np.float32)
    y = np.ones([5, 3], np.float32)
    k.apply(x, y)

  def testFloat64(self):
    # No exception expected
    k = psd_kernels.Polynomial(
        bias_variance=np.float64(0.),
        slope_variance=np.float64(1.),
        exponent=np.float64(1.),
        feature_ndims=1)
    x = np.ones([5, 3], np.float64)
    y = np.ones([5, 3], np.float64)
    k.apply(x, y)

  @parameterized.named_parameters(
      ('1 feature dimension', 1, (5, 3), (5, 3), (5,)),
      ('2 feature dimensions', 2, (5, 3, 2), (5, 3, 2), (5,)))
  def testShapesAreCorrectApply(self, feature_ndims,
                                x_shape, y_shape, apply_shape):
    k = psd_kernels.Polynomial(
        bias_variance=0.,
        slope_variance=1.,
        exponent=1.,
        feature_ndims=feature_ndims)
    x = np.ones(x_shape, np.float32)
    y = np.ones(y_shape, np.float32)
    self.assertAllEqual(
        k.apply(x, y).shape, apply_shape)

  @parameterized.named_parameters(
      ('1 feature dimension, 1 batch dimension',
       1, (5, 3), (4, 3), (5, 4)),
      ('1 feature dimension, 2 batch dimensions',
       1, (10, 5, 3), (10, 4, 3), (10, 5, 4)),
      ('2 feature dimensions, 1 batch dimension',
       2, (5, 3, 2), (4, 3, 2), (5, 4)))
  def testShapesAreCorrectMatrix(self, feature_ndims,
                                 x_shape, y_shape, matrix_shape):
    k = psd_kernels.Polynomial(
        bias_variance=0.,
        slope_variance=1.,
        exponent=1.,
        feature_ndims=feature_ndims)
    x = np.ones(x_shape, np.float32)
    y = np.ones(y_shape, np.float32)
    self.assertAllEqual(
        k.matrix(x, y).shape, matrix_shape)

  def testShapesAreCorrectBroadcast(self):
    k = psd_kernels.Polynomial(
        bias_variance=np.ones([2, 1, 1], np.float32),
        slope_variance=np.ones([1, 3, 1], np.float32))
    self.assertAllEqual(
        k.matrix(
            np.ones([2, 4, 3], np.float32),
            np.ones([2, 5, 3], np.float32)
        ).shape, [2, 3, 2, 4, 5])
    #             `--'  |  `--'
    #               |   |    `- matrix shape
    #               |   `- from input batch shapes
    #               `- from broadcasting kernel params

  def testValuesAreCorrect(self):
    bias_variance = 1.5
    slope_variance = 0.5
    exponent = 2
    k = psd_kernels.Polynomial(
        bias_variance=bias_variance,
        slope_variance=slope_variance,
        exponent=exponent
    )
    x = np.random.uniform(-1, 1, size=[5, 3]).astype(np.float32)
    y = np.random.uniform(-1, 1, size=[4, 3]).astype(np.float32)
    self.assertAllClose(
        self.evaluate(k.matrix(x, y)),
        bias_variance**2 + slope_variance**2 * (x.dot(y.T))**exponent
    )


@test_util.run_all_in_graph_and_eager_modes
class LinearTest(tf.test.TestCase, parameterized.TestCase):
  """Test the Linear kernel."""

  def testIsPolynomial(self):
    # Linear kernel is subclass of Polynomial kernel
    self.assertIsInstance(psd_kernels.Linear(), psd_kernels.Polynomial)

  def testValuesAreCorrect(self):
    k = psd_kernels.Linear()
    x = np.random.uniform(-1, 1, size=[5, 3]).astype(np.float32)
    y = np.random.uniform(-1, 1, size=[4, 3]).astype(np.float32)
    self.assertAllClose(
        self.evaluate(k.matrix(x, y)), x.dot(y.T))


if __name__ == '__main__':
  tf.test.main()
