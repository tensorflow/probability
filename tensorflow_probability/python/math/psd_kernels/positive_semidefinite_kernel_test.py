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
import numpy as np
import tensorflow.compat.v1 as tf1
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp

from tensorflow_probability.python.internal import test_util
from tensorflow_probability.python.math.psd_kernels.internal import util as kernels_util


JAX_MODE = False
if JAX_MODE:
  ERROR_TYPES = TypeError  # pylint: disable=invalid-name
else:
  ERROR_TYPES = (ValueError, tf.errors.InvalidArgumentError)  # pylint: disable=invalid-name

PARAMS_0 = np.array(2.).astype(np.float32)
PARAMS_1 = np.array([2.]).astype(np.float32)
PARAMS_2 = np.array([1., 2.]).astype(np.float32)
PARAMS_21 = np.array([[1.], [2.]]).astype(np.float32)


class IncompletelyDefinedKernel(tfp.math.psd_kernels.PositiveSemidefiniteKernel
                               ):

  def __init__(self):
    super(IncompletelyDefinedKernel, self).__init__(feature_ndims=1)


class TestKernel(tfp.math.psd_kernels.PositiveSemidefiniteKernel):
  """A PositiveSemidefiniteKernel implementation just for testing purposes.

  k(x, y) = m * sum(x + y)

  Not at all positive semidefinite, but we don't care about this here.
  """

  def __init__(self, multiplier=None, feature_ndims=1):
    self._multiplier = (None if multiplier is None else
                        tf.convert_to_tensor(multiplier))
    dtype = None if multiplier is None else self._multiplier.dtype
    super(TestKernel, self).__init__(feature_ndims=feature_ndims,
                                     dtype=dtype)

  @property
  def multiplier(self):
    return self._multiplier

  def _batch_shape(self):
    return [] if self.multiplier is None else self._multiplier.shape

  def _batch_shape_tensor(self):
    return (tf.TensorShape([]) if self.multiplier is None else
            tf.shape(self._multiplier))

  def _apply(self, x1, x2, example_ndims=0):
    x1 = tf.convert_to_tensor(x1)
    x2 = tf.convert_to_tensor(x2)

    value = tf.reduce_sum(x1 + x2, axis=-1)
    if self.multiplier is not None:
      multiplier = kernels_util.pad_shape_with_ones(
          self._multiplier, example_ndims)
      value = value * multiplier

    return value


@test_util.test_all_tf_execution_regimes
class PositiveSemidefiniteKernelTest(test_util.TestCase):
  """Test the abstract base class behaviors."""

  def createKernelInputs(self, batched=False):
    x = tf1.placeholder_with_default(np.float32(
        [[1., 1., 1.], [2., 2., 2.], [3., 3., 3.]]), shape=[3, 3])
    y = tf1.placeholder_with_default(np.float32(
        [[4., 4., 4.], [5., 5., 5.], [6., 6., 6.]]), shape=[3, 3])
    z = tf1.placeholder_with_default(np.float32(
        [[4., 4., 4.], [5., 5., 5.], [6., 6., 6.], [7., 7., 7.]]), shape=[4, 3])
    if not batched:
      return x, y, z

    x = tf.stack([x] * 5)
    y = tf.stack([y] * 5)
    z = tf.stack([z] * 5)

    return x, y, z

  def testStrUnknownBatchShape(self):
    if tf.executing_eagerly(): return
    k32_batch_unk = TestKernel(
        tf1.placeholder_with_default([2., 3], shape=None))
    self.assertEqual(
        'tfp.math.psd_kernels.TestKernel('
        '"TestKernel", feature_ndims=1, dtype=float32)', str(k32_batch_unk))

  def testStr(self):
    k32_batch2 = TestKernel(tf.cast([123., 456.], dtype=tf.float32))
    k64_batch2x1 = TestKernel(tf.cast([[123.], [456.]], dtype=tf.float64))
    k_fdim3 = TestKernel(tf.cast(123., dtype=tf.float32), feature_ndims=3)
    self.assertEqual(
        'tfp.math.psd_kernels.TestKernel('
        '"TestKernel", batch_shape=(2,), feature_ndims=1, dtype=float32)',
        str(k32_batch2))
    self.assertEqual(
        'tfp.math.psd_kernels.TestKernel('
        '"TestKernel", batch_shape=(2, 1), feature_ndims=1, dtype=float64)',
        str(k64_batch2x1))
    self.assertEqual(
        'tfp.math.psd_kernels.TestKernel('
        '"TestKernel", batch_shape=(), feature_ndims=3, dtype=float32)',
        str(k_fdim3))

  def testReprUnknownBatchShape(self):
    if tf.executing_eagerly(): return
    k32_batch_unk = TestKernel(
        tf1.placeholder_with_default([2., 3], shape=None))
    self.assertEqual(
        '<tfp.math.psd_kernels.TestKernel '
        '\'TestKernel\' batch_shape=<unknown> feature_ndims=1 dtype=float32>',
        repr(k32_batch_unk))

  def testRepr(self):
    k32_batch2 = TestKernel(tf.cast([123., 456.], dtype=tf.float32))
    k64_batch2x1 = TestKernel(tf.cast([[123.], [456.]], dtype=tf.float64))
    k_fdim3 = TestKernel(tf.cast(123., dtype=tf.float32), feature_ndims=3)
    self.assertEqual(
        '<tfp.math.psd_kernels.TestKernel '
        '\'TestKernel\' batch_shape=(2,) feature_ndims=1 dtype=float32>',
        repr(k32_batch2))
    self.assertEqual(
        '<tfp.math.psd_kernels.TestKernel '
        '\'TestKernel\' batch_shape=(2, 1) feature_ndims=1 dtype=float64>',
        repr(k64_batch2x1))
    self.assertEqual(
        '<tfp.math.psd_kernels.TestKernel '
        '\'TestKernel\' batch_shape=() feature_ndims=3 dtype=float32>',
        repr(k_fdim3))

  def testNotImplementedExceptions(self):
    k = IncompletelyDefinedKernel()
    x, _, _ = self.createKernelInputs()
    with self.assertRaises(NotImplementedError):
      k.apply(x, x)

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

    class FeatureNdimsKernel(tfp.math.psd_kernels.PositiveSemidefiniteKernel):

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
    self.assertAllEqual(shape, k.batch_shape)
    self.assertAllEqual(shape, self.evaluate(k.batch_shape_tensor()))

  @parameterized.named_parameters(
      ('Dynamic-shape [2] kernel', [1., 2.], [2]),
      ('Dynamic-shape [2, 1] kernel', [[1.], [2.]], [2, 1]))
  def testDynamicBatchShape(self, params, shape):
    tensor_params = tf1.placeholder_with_default(params, shape=None)
    k = TestKernel(tensor_params)
    self.assertAllEqual(shape, self.evaluate(k.batch_shape_tensor()))

  def testApplyOutputWithStaticShapes(self):
    k = TestKernel(PARAMS_0)  # batch_shape = []
    x, y, _ = self.createKernelInputs()
    self.assertAllEqual([3], k.apply(x, y).shape)

    k = TestKernel(PARAMS_2)  # batch_shape = [2]
    with self.assertRaises(ERROR_TYPES):
      # Param batch shape [2] won't broadcast with the input batch shape, [3].
      k.apply(
          x,  # shape [3, 3]
          y)  # shape [3, 3]

    k = TestKernel(PARAMS_21)  # batch_shape = [2, 1]
    self.assertAllEqual(
        [2, 3],
        k.apply(
            x,  # shape [3, 3]
            y   # shape [3, 3]
        ).shape)

  def testApplyOutputWithDynamicShapes(self):
    params_2_dynamic = tf1.placeholder_with_default([1., 2.], shape=None)
    k = TestKernel(params_2_dynamic)
    x, y, _ = self.createKernelInputs()
    if JAX_MODE:
      error_types = TypeError
    else:
      error_types = tf.errors.InvalidArgumentError
    with self.assertRaises(error_types):
      apply_op = k.apply(
          x,  # shape [3, 3]
          y   # shape [3, 3]
      )  # No exception yet
      self.evaluate(apply_op)

    params_21_dynamic = tf1.placeholder_with_default([[1.], [2.]], shape=None)
    k = TestKernel(params_21_dynamic)
    apply_op = k.apply(
        x,  # shape [3, 3]
        y)  # shape [3, 3]
    self.assertAllEqual([2, 3], self.evaluate(apply_op).shape)

  def testMatrixOutputWithStaticShapes(self):
    k = TestKernel(PARAMS_0)  # batch_shape = []
    x, _, z = self.createKernelInputs()
    self.assertAllEqual(
        [3, 4],
        k.matrix(
            x,  # shape [3, 3]
            z   # shape [4, 3]
        ).shape)

    k = TestKernel(PARAMS_2)  # batch_shape = [2]
    self.assertAllEqual(
        [2, 3, 4],
        k.matrix(
            x,  # shape [3, 3]
            z   # shape [4, 3]
        ).shape)

    k = TestKernel(PARAMS_2)  # batch_shape = [2]
    batch_x, _, batch_z = self.createKernelInputs(batched=True)

    with self.assertRaises(ERROR_TYPES):
      k.matrix(
          batch_x,  # shape = [5, 3, 3]
          batch_z)  # shape = [5, 4, 3]

    k = TestKernel(PARAMS_21)  # batch_shape = [2, 1]
    self.assertAllEqual(
        [2, 5, 3, 4],
        k.matrix(
            batch_x,  # shape = [5, 3, 3]
            batch_z   # shape = [5, 4, 3]
        ).shape)

  def testMatrixOutputWithDynamicShapes(self):
    params_2_dynamic = tf1.placeholder_with_default([1., 2.], shape=None)
    k = TestKernel(params_2_dynamic)  # batch_shape [2]
    x, _, z = self.createKernelInputs()
    apply_op = k.matrix(
        x,  # shape [3, 3]
        z)  # shape [4, 3]
    self.assertAllEqual([2, 3, 4], self.evaluate(apply_op).shape)

    params_21_dynamic = tf1.placeholder_with_default([[1.], [2.]], shape=None)
    k = TestKernel(params_21_dynamic)  # shape [2, 1]
    batch_x, _, batch_z = self.createKernelInputs(batched=True)
    apply_op = k.matrix(
        batch_x,  # shape [5, 3, 3]
        batch_z)  # shape [5, 4, 3]
    self.assertAllEqual([2, 5, 3, 4], self.evaluate(apply_op).shape)

  def testOperatorOverloads(self):
    k0 = TestKernel(PARAMS_0)
    sum_kernel = k0 + k0 + k0
    self.assertLen(sum_kernel.kernels, 3)
    sum_kernel += k0 + k0
    self.assertLen(sum_kernel.kernels, 5)

    product_kernel = k0 * k0 * k0
    self.assertLen(product_kernel.kernels, 3)
    product_kernel *= k0 * k0
    self.assertLen(product_kernel.kernels, 5)

  def testStaticShapesAndValuesOfSum(self):
    k0 = TestKernel(PARAMS_0)
    k1 = TestKernel(PARAMS_1)
    k2 = TestKernel(PARAMS_2)
    k21 = TestKernel(PARAMS_21)

    x, y, _ = self.createKernelInputs()
    sum_kernel = k0 + k1 + k2 + k21
    self.assertAllEqual([2, 2], sum_kernel.batch_shape)
    self.assertEqual(tf.float32, sum_kernel.dtype)
    self.assertAllEqual(
        sum([self.evaluate(k.matrix(x, y))
             for k in sum_kernel.kernels]),
        self.evaluate(sum_kernel.matrix(x, y)))

  def testDynamicShapesAndValuesOfSum(self):
    params_2_dynamic = tf1.placeholder_with_default(np.float32([1., 2.]),
                                                    shape=None)
    params_21_dynamic = tf1.placeholder_with_default(np.float32([[1.], [2.]]),
                                                     shape=None)
    k2 = TestKernel(params_2_dynamic)
    k21 = TestKernel(params_21_dynamic)

    x, y, _ = self.createKernelInputs()
    sum_kernel = k2 + k21
    self.assertEqual(tf.float32, sum_kernel.dtype)
    self.assertAllEqual([2, 2], self.evaluate(sum_kernel.batch_shape_tensor()))
    self.assertAllEqual(
        sum([self.evaluate(k.matrix(x, y))
             for k in sum_kernel.kernels]),
        self.evaluate(sum_kernel.matrix(x, y)))

  def testStaticShapesAndValuesOfProduct(self):
    k0 = TestKernel(PARAMS_0)
    k1 = TestKernel(PARAMS_1)
    k2 = TestKernel(PARAMS_2)
    k21 = TestKernel(PARAMS_21)

    x, y, _ = self.createKernelInputs()
    product_kernel = k0 * k1 * k2 * k21
    self.assertEqual(tf.float32, product_kernel.dtype)
    self.assertAllEqual([2, 2], product_kernel.batch_shape)
    self.assertAllEqual(
        functools.reduce(
            operator.mul,
            [self.evaluate(k.matrix(x, y))
             for k in product_kernel.kernels]),
        self.evaluate(product_kernel.matrix(x, y)))

  def testDynamicShapesAndValuesOfProduct(self):
    params_2_dynamic = tf1.placeholder_with_default(np.float32([1., 2.]),
                                                    shape=None)
    params_21_dynamic = tf1.placeholder_with_default(np.float32([[1.], [2.]]),
                                                     shape=None)
    k2 = TestKernel(params_2_dynamic)
    k21 = TestKernel(params_21_dynamic)

    x, y, _ = self.createKernelInputs()
    product_kernel = k2 * k21
    self.assertEqual(tf.float32, product_kernel.dtype)
    self.assertAllEqual(
        [2, 2], self.evaluate(product_kernel.batch_shape_tensor()))
    self.assertAllEqual(
        functools.reduce(
            operator.mul,
            [self.evaluate(k.matrix(x, y))
             for k in product_kernel.kernels]),
        self.evaluate(product_kernel.matrix(x, y)))

  def testSumOfKernelsWithNoneDtypes(self):
    none_kernel = TestKernel()
    float32_kernel = TestKernel(np.float32(1))
    float64_kernel = TestKernel(np.float64(1))

    # Should all be fine.
    _ = none_kernel + none_kernel
    _ = none_kernel + float32_kernel
    _ = none_kernel + float64_kernel

    # Should not be fine.
    with self.assertRaises(TypeError):
      _ = float32_kernel + float64_kernel

  def testProductOfKernelsWithNoneDtypes(self):
    none_kernel = TestKernel()
    float32_kernel = TestKernel(np.float32(1))
    float64_kernel = TestKernel(np.float64(1))

    # Should all be fine.
    _ = none_kernel * none_kernel
    _ = none_kernel * float32_kernel
    _ = none_kernel * float64_kernel

    # Should not be fine.
    with self.assertRaises(TypeError):
      _ = float32_kernel * float64_kernel

if __name__ == '__main__':
  tf.test.main()
