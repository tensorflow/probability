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

import tensorflow.compat.v2 as tf

import tensorflow_probability as tfp

from tensorflow_probability.python.internal import test_util


@test_util.test_all_tf_execution_regimes
class PolynomialTest(test_util.TestCase):
  """Test the Polynomial kernel."""

  def test_mismatched_float_types_are_bad(self):
    with self.assertRaises(TypeError):
      tfp.math.psd_kernels.Polynomial(
          bias_variance=np.float32(1.),
          slope_variance=np.float64(1.),
          shift=0.,
          exponent=1.)

  def testFloat32Fallback(self):
    # Should be OK (float32 fallback).
    self.polynomial = tfp.math.psd_kernels.Polynomial(
        bias_variance=0, slope_variance=1, shift=0, exponent=1)
    # Should be OK.
    tfp.math.psd_kernels.Polynomial(
        bias_variance=np.float32(1.), slope_variance=1., shift=0., exponent=1.)

  def testValidateArgsNonPositiveAreBad(self):
    with self.assertRaisesOpError('Condition x > 0 did not hold'):
      k = tfp.math.psd_kernels.Polynomial(bias_variance=-1., validate_args=True)
      self.evaluate(k.apply([1.], [1.]))
    with self.assertRaisesOpError('Condition x > 0 did not hold'):
      k = tfp.math.psd_kernels.Polynomial(
          slope_variance=-1., validate_args=True)
      self.evaluate(k.apply([1.], [1.]))
    with self.assertRaisesOpError('Condition x > 0 did not hold'):
      k = tfp.math.psd_kernels.Polynomial(exponent=-1., validate_args=True)
      self.evaluate(k.apply([1.], [1.]))

  def testShifttNonPositiveIsOk(self):
    # No exception expected
    k = tfp.math.psd_kernels.Polynomial(shift=-1., validate_args=True)
    self.evaluate(k.apply([1.], [1.]))

  def testValidateArgsNoneIsOk(self):
    # No exception expected
    k = tfp.math.psd_kernels.Polynomial(
        bias_variance=None,
        slope_variance=None,
        shift=None,
        exponent=None,
        validate_args=True)
    self.evaluate(k.apply([[1.]], [[1.]]))

  def testNoneShapes(self):
    k = tfp.math.psd_kernels.Polynomial(
        bias_variance=np.reshape(np.arange(12.), [2, 3, 2]))
    self.assertAllEqual([2, 3, 2], k.batch_shape)

  @parameterized.named_parameters(
      dict(
          testcase_name='Shape [] kernel',
          bias_variance=2.,
          slope_variance=2.,
          shift=2.,
          exponent=2.,
          shape=[]),
      dict(
          testcase_name='Shape [1] kernel',
          bias_variance=[2.],
          slope_variance=[2.],
          shift=[2.],
          exponent=[2.],
          shape=[1]),
      dict(
          testcase_name='Shape [2] kernel',
          bias_variance=[1., 2.],
          slope_variance=[1., 2.],
          shift=[1., 2.],
          exponent=[1., 2.],
          shape=[2]),
      dict(
          testcase_name='Shape [2, 1] kernel',
          bias_variance=[[1.], [2.]],
          slope_variance=[[1.], [2.]],
          shift=[[1.], [2.]],
          exponent=[[1.], [2.]],
          shape=[2, 1]),
      dict(
          testcase_name='Shape [2, 1] broadcast kernel',
          bias_variance=None,
          slope_variance=2.,
          shift=[2.],
          exponent=[[1.], [2.]],
          shape=[2, 1]))
  def testBatchShape(self, bias_variance, slope_variance,
                     shift, exponent, shape):
    k = tfp.math.psd_kernels.Polynomial(
        bias_variance=bias_variance,
        slope_variance=slope_variance,
        shift=shift,
        exponent=exponent,
        validate_args=True)
    self.assertAllEqual(shape, k.batch_shape)
    self.assertAllEqual(shape, self.evaluate(k.batch_shape_tensor()))

  def testFloat32(self):
    # No exception expected
    k = tfp.math.psd_kernels.Polynomial(
        bias_variance=0.,
        slope_variance=1.,
        shift=0.,
        exponent=1.,
        feature_ndims=1)
    x = np.ones([5, 3], np.float32)
    y = np.ones([5, 3], np.float32)
    k.apply(x, y)

  def testFloat64(self):
    # No exception expected
    k = tfp.math.psd_kernels.Polynomial(
        bias_variance=np.float64(0.),
        slope_variance=np.float64(1.),
        shift=np.float64(0.),
        exponent=np.float64(1.),
        feature_ndims=1)
    x = np.ones([5, 3], np.float64)
    y = np.ones([5, 3], np.float64)
    k.apply(x, y)

  @parameterized.named_parameters(
      dict(
          testcase_name='1 feature dimension',
          feature_ndims=1,
          x_shape=(5, 3),
          y_shape=(5, 3),
          apply_shape=(5,),
      ),
      dict(
          testcase_name='2 feature dimension',
          feature_ndims=2,
          x_shape=(5, 3, 2),
          y_shape=(5, 3, 2),
          apply_shape=(5,),
      ))
  def testShapesAreCorrectApply(self, feature_ndims,
                                x_shape, y_shape, apply_shape):
    k = tfp.math.psd_kernels.Polynomial(
        bias_variance=0.,
        slope_variance=1.,
        shift=0.,
        exponent=1.,
        feature_ndims=feature_ndims)
    x = np.ones(x_shape, np.float32)
    y = np.ones(y_shape, np.float32)
    self.assertAllEqual(
        apply_shape, k.apply(x, y).shape)

  @parameterized.named_parameters(
      dict(
          testcase_name='1 feature dimension, 1 batch dimension',
          feature_ndims=1,
          x_shape=(5, 3),
          y_shape=(4, 3),
          matrix_shape=(5, 4),
      ),
      dict(
          testcase_name='1 feature dimension, 2 batch dimensions',
          feature_ndims=1,
          x_shape=(10, 5, 3),
          y_shape=(10, 4, 3),
          matrix_shape=(10, 5, 4),
      ),
      dict(
          testcase_name='2 feature dimensions, 1 batch dimension',
          feature_ndims=2,
          x_shape=(5, 3, 2),
          y_shape=(4, 3, 2),
          matrix_shape=(5, 4),
      ))
  def testShapesAreCorrectMatrix(self, feature_ndims,
                                 x_shape, y_shape, matrix_shape):
    k = tfp.math.psd_kernels.Polynomial(
        bias_variance=0.,
        slope_variance=1.,
        shift=0.,
        exponent=1.,
        feature_ndims=feature_ndims)
    x = np.ones(x_shape, np.float32)
    y = np.ones(y_shape, np.float32)
    self.assertAllEqual(matrix_shape, k.matrix(x, y).shape)

  def testShapesAreCorrectBroadcast(self):
    k = tfp.math.psd_kernels.Polynomial(
        bias_variance=np.ones([2, 1, 1], np.float32),
        slope_variance=np.ones([1, 3, 1], np.float32))
    self.assertAllEqual(
        [2, 3, 2, 4, 5],
        #`--'  |  `--'
        #  |   |    `- matrix shape
        #  |   `- from input batch shapes
        #  `- from broadcasting kernel params
        k.matrix(
            np.ones([2, 4, 3], np.float32),
            np.ones([2, 5, 3], np.float32)
        ).shape)

  def testValuesAreCorrect(self):
    bias_variance = 1.5
    slope_variance = 0.5
    shift = 1.
    exponent = 2
    k = tfp.math.psd_kernels.Polynomial(
        bias_variance=bias_variance,
        slope_variance=slope_variance,
        shift=shift,
        exponent=exponent)
    x = np.random.uniform(-1, 1, size=[5, 3]).astype(np.float32)
    y = np.random.uniform(-1, 1, size=[4, 3]).astype(np.float32)
    self.assertAllClose(
        (bias_variance ** 2 + slope_variance ** 2 *
         ((x - shift).dot((y - shift).T)) ** exponent),
        self.evaluate(k.matrix(x, y))
    )


@test_util.test_all_tf_execution_regimes
class LinearTest(test_util.TestCase):
  """Test the Linear kernel."""

  def testIsPolynomial(self):
    # Linear kernel is subclass of Polynomial kernel
    self.assertIsInstance(tfp.math.psd_kernels.Linear(),
                          tfp.math.psd_kernels.Polynomial)

  def testValuesAreCorrect(self):
    k = tfp.math.psd_kernels.Linear()
    x = np.random.uniform(-1, 1, size=[5, 3]).astype(np.float32)
    y = np.random.uniform(-1, 1, size=[4, 3]).astype(np.float32)
    self.assertAllClose(x.dot(y.T), self.evaluate(k.matrix(x, y)))


if __name__ == '__main__':
  tf.test.main()
