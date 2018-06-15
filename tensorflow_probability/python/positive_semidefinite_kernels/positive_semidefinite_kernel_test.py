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
"""PositiveSemidefiniteKernel Tests."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import functools
import operator
from absl.testing import parameterized
import tensorflow as tf
import tensorflow_probability as tfp

kernels_lib = tfp.positive_semidefinite_kernels
kernels_util = tfp.positive_semidefinite_kernels.util


class IncompletelyDefinedKernel(kernels_lib.PositiveSemidefiniteKernel):

  def __init__(self):
    super(IncompletelyDefinedKernel, self).__init__(feature_ndims=1)


class TestKernel(kernels_lib.PositiveSemidefiniteKernel):
  """A PositiveSemidefiniteKernel implementation just for testing purposes.

  k(x, y) = m * sum(x + y)

  Not at all positive semidefinite, but we don't care about this here.
  """

  def __init__(self, multiplier):
    self._multiplier = tf.convert_to_tensor(multiplier)
    super(TestKernel, self).__init__(feature_ndims=1)

  def _batch_shape(self):
    return self._multiplier.shape

  def _batch_shape_tensor(self):
    return tf.shape(self._multiplier)

  def _apply(self, x1, x2, param_expansion_ndims=0):
    x1 = tf.convert_to_tensor(x1)
    x2 = tf.convert_to_tensor(x2)

    multiplier = kernels_util.pad_shape_right_with_ones(
        self._multiplier, param_expansion_ndims)

    return multiplier * tf.reduce_sum(x1 + x2, axis=-1)


class PositiveSemidefiniteKernelTest(tf.test.TestCase, parameterized.TestCase):
  """Test the abstract base class behaviors."""

  def setUp(self):
    self.x = tf.placeholder_with_default(
        [[1., 1., 1.], [2., 2., 2.], [3., 3., 3.]], shape=[3, 3])
    self.y = tf.placeholder_with_default(
        [[4., 4., 4.], [5., 5., 5.], [6., 6., 6.]], shape=[3, 3])
    self.z = tf.placeholder_with_default(
        [[4., 4., 4.], [5., 5., 5.], [6., 6., 6.], [7., 7., 7.]], shape=[4, 3])

    self.batch_x = tf.stack([self.x] * 5)
    self.batch_y = tf.stack([self.y] * 5)
    self.batch_z = tf.stack([self.z] * 5)

    # Kernel params of various shapes
    self.params_0 = 2.
    self.params_1 = [2.]
    self.params_2 = [1., 2.]
    self.params_21 = [[1.], [2.]]

    # Kernel params with dynamic shapes
    self.params_2_dynamic = tf.placeholder_with_default([1., 2.], shape=None)
    self.params_21_dynamic = tf.placeholder_with_default(
        [[1.], [2.]], shape=None)

  def testNotImplementedExceptions(self):
    k = IncompletelyDefinedKernel()
    with self.assertRaises(NotImplementedError):
      k.apply(self.x, self.x)

    with self.assertRaises(NotImplementedError):
      _ = k.batch_shape

    with self.assertRaises(NotImplementedError):
      _ = k.batch_shape_tensor()

  @parameterized.named_parameters(
      ('String feature_ndims', 'non-integer'),
      ('Float feature_ndims', 4.2),
      ('Zero feature_ndims', 0),
      ('Negative feature_ndims', -3))
  def testFeatureNdimsExceptions(self, feature_ndims):
    class FeatureNdimsKernel(kernels_lib.PositiveSemidefiniteKernel):

      def __init__(self):
        super(FeatureNdimsKernel, self).__init__(feature_ndims)
    with self.assertRaises(ValueError):
      FeatureNdimsKernel()

  @parameterized.named_parameters(
      ('Shape [] kernel', 2., []),
      ('Shape [1] kernel', [2.], [1]),
      ('Shape [2] kernel', [1., 2.], [2]),
      ('Shape [2, 1] kernel', [[1.], [2.]], [2, 1]))
  def testStaticBatchShape(self, params, shape):
    k = TestKernel(params)
    self.assertAllEqual(k.batch_shape.as_list(), shape)
    self.assertAllEqual(self.evaluate(k.batch_shape_tensor()), shape)

  @parameterized.named_parameters(
      ('Dynamic-shape [2] kernel',
       tf.placeholder_with_default([1., 2.], shape=None), [2]),
      ('Dynamic-shape [2, 1] kernel',
       tf.placeholder_with_default([[1.], [2.]], shape=None), [2, 1]))
  def testDynamicBatchShape(self, params, shape):
    k = TestKernel(params)
    self.assertIsNone(k.batch_shape.ndims)
    self.assertAllEqual(self.evaluate(k.batch_shape_tensor()), shape)

  def testApplyOutputWithStaticShapes(self):
    k = TestKernel(self.params_0)  # batch_shape = []
    self.assertAllEqual(k.apply(self.x, self.y).shape, [3])

    k = TestKernel(self.params_2)  # batch_shape = [2]
    with self.assertRaises(ValueError):
      # Param batch shape [2] won't broadcast with the input batch shape, [3].
      k.apply(
          self.x,  # shape [3, 3]
          self.y)  # shape [3, 3]

    k = TestKernel(self.params_21)  # batch_shape = [2, 1]
    self.assertAllEqual(k.apply(
        self.x,  # shape [3, 3]
        self.y   # shape [3, 3]
    ).shape, [2, 3])

  def testApplyOutputWithDynamicShapes(self):
    k = TestKernel(self.params_2_dynamic)
    apply_op = k.apply(
        self.x,  # shape [3, 3]
        self.y   # shape [3, 3]
    )  # No exception yet
    self.assertEqual(apply_op.shape.ndims, None)
    with self.assertRaises(tf.errors.InvalidArgumentError):
      self.evaluate(apply_op)

    k = TestKernel(self.params_21_dynamic)
    apply_op = k.apply(
        self.x,  # shape [3, 3]
        self.y)  # shape [3, 3]
    self.assertEqual(apply_op.shape.ndims, None)
    self.assertAllEqual(self.evaluate(apply_op).shape, [2, 3])

  def testMatrixOutputWithStaticShapes(self):
    k = TestKernel(self.params_0)  # batch_shape = []
    self.assertAllEqual(k.matrix(
        self.x,  # shape [3, 3]
        self.z   # shape [4, 3]
    ).shape, [3, 4])

    k = TestKernel(self.params_2)  # batch_shape = [2]
    self.assertAllEqual(k.matrix(
        self.x,  # shape [3, 3]
        self.z   # shape [4, 3]
    ).shape, [2, 3, 4])

    k = TestKernel(self.params_2)  # batch_shape = [2]
    with self.assertRaises(ValueError):
      k.matrix(
          self.batch_x,  # shape = [5, 3, 3]
          self.batch_z)  # shape = [5, 4, 3]

    k = TestKernel(self.params_21)  # batch_shape = [2, 1]
    self.assertAllEqual(
        k.matrix(
            self.batch_x,  # shape = [5, 3, 3]
            self.batch_z   # shape = [5, 4, 3]
        ).shape,
        [2, 5, 3, 4])

  def testMatrixOutputWithDynamicShapes(self):
    k = TestKernel(self.params_2_dynamic)  # batch_shape [2]
    apply_op = k.matrix(
        self.x,  # shape [3, 3]
        self.z)  # shape [4, 3]
    self.assertEqual(apply_op.shape.ndims, None)
    self.assertAllEqual(self.evaluate(apply_op).shape, [2, 3, 4])

    k = TestKernel(self.params_21_dynamic)  # shape [2, 1]
    apply_op = k.matrix(
        self.batch_x,  # shape [5, 3, 3]
        self.batch_z)  # shape [5, 4, 3]
    self.assertEqual(apply_op.shape.ndims, None)
    self.assertAllEqual(self.evaluate(apply_op).shape, [2, 5, 3, 4])

  def testOperatorOverloads(self):
    k0 = TestKernel(self.params_0)
    sum_kernel = k0 + k0 + k0
    self.assertEqual(len(sum_kernel.kernels), 3)
    sum_kernel += k0 + k0
    self.assertEqual(len(sum_kernel.kernels), 5)

    product_kernel = k0 * k0 * k0
    self.assertEqual(len(product_kernel.kernels), 3)
    product_kernel *= k0 * k0
    self.assertEqual(len(product_kernel.kernels), 5)

  def testStaticShapesAndValuesOfSum(self):
    k0 = TestKernel(self.params_0)
    k1 = TestKernel(self.params_1)
    k2 = TestKernel(self.params_2)
    k21 = TestKernel(self.params_21)

    sum_kernel = k0 + k1 + k2 + k21
    self.assertAllEqual(sum_kernel.batch_shape, [2, 2])
    self.assertAllEqual(
        self.evaluate(sum_kernel.matrix(self.x, self.y)),
        sum([self.evaluate(k.matrix(self.x, self.y))
             for k in sum_kernel.kernels]))

  def testDynamicShapesAndValuesOfSum(self):
    k2 = TestKernel(self.params_2_dynamic)
    k21 = TestKernel(self.params_21_dynamic)

    sum_kernel = k2 + k21
    self.assertIsNone(sum_kernel.batch_shape.ndims)
    self.assertAllEqual(self.evaluate(sum_kernel.batch_shape_tensor()), [2, 2])
    self.assertAllEqual(
        self.evaluate(sum_kernel.matrix(self.x, self.y)),
        sum([self.evaluate(k.matrix(self.x, self.y))
             for k in sum_kernel.kernels]))

  def testStaticShapesAndValuesOfProduct(self):
    k0 = TestKernel(self.params_0)
    k1 = TestKernel(self.params_1)
    k2 = TestKernel(self.params_2)
    k21 = TestKernel(self.params_21)

    product_kernel = k0 * k1 * k2 * k21
    self.assertAllEqual(product_kernel.batch_shape, [2, 2])
    self.assertAllEqual(
        self.evaluate(product_kernel.matrix(self.x, self.y)),
        functools.reduce(
            operator.mul,
            [self.evaluate(k.matrix(self.x, self.y))
             for k in product_kernel.kernels]))

  def testDynamicShapesAndValuesOfProduct(self):
    k2 = TestKernel(self.params_2_dynamic)
    k21 = TestKernel(self.params_21_dynamic)

    product_kernel = k2 * k21
    self.assertIsNone(product_kernel.batch_shape.ndims)
    self.assertAllEqual(
        self.evaluate(product_kernel.batch_shape_tensor()), [2, 2])
    self.assertAllEqual(
        self.evaluate(product_kernel.matrix(self.x, self.y)),
        functools.reduce(
            operator.mul,
            [self.evaluate(k.matrix(self.x, self.y))
             for k in product_kernel.kernels]))

if __name__ == '__main__':
  tf.test.main()

