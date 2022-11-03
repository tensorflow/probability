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


import functools
import operator

from absl.testing import parameterized
import numpy as np
import tensorflow.compat.v1 as tf1
import tensorflow.compat.v2 as tf

from tensorflow_probability.python.internal import parameter_properties
from tensorflow_probability.python.internal import test_util
from tensorflow_probability.python.math.psd_kernels import positive_semidefinite_kernel as psd_kernel
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


class IncompletelyDefinedKernel(psd_kernel.PositiveSemidefiniteKernel):

  def __init__(self):
    super(IncompletelyDefinedKernel, self).__init__(feature_ndims=1)


class TestKernel(psd_kernel.PositiveSemidefiniteKernel):
  """A PositiveSemidefiniteKernel implementation just for testing purposes.

  k(x, y) = m * sum(x + y)

  Not at all positive semidefinite, but we don't care about this here.
  """

  def __init__(self, multiplier=None, feature_ndims=1):
    parameters = dict(locals())
    self._multiplier = (None if multiplier is None else
                        tf.convert_to_tensor(multiplier))
    dtype = None if multiplier is None else self._multiplier.dtype
    super(TestKernel, self).__init__(feature_ndims=feature_ndims,
                                     dtype=dtype,
                                     parameters=parameters,
                                     validate_args=True)

  @classmethod
  def _parameter_properties(cls, dtype):
    return dict(multiplier=parameter_properties.ParameterProperties())

  @property
  def multiplier(self):
    return self._multiplier

  def _apply(self, x1, x2, example_ndims=0):
    x1 = tf.convert_to_tensor(x1)
    x2 = tf.convert_to_tensor(x2)

    value = tf.reduce_sum(x1 + x2, axis=-1)
    if self.multiplier is not None:
      multiplier = kernels_util.pad_shape_with_ones(
          self._multiplier, example_ndims)
      value = value * multiplier

    return value


class CompositeTensorTestKernel(
    TestKernel, psd_kernel.AutoCompositeTensorPsdKernel):
  pass


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

  @parameterized.named_parameters(
      ('String feature_ndims', 'non-integer'),
      ('Float feature_ndims', 4.2),
      ('Zero feature_ndims', 0),
      ('Negative feature_ndims', -3))
  def testFeatureNdimsExceptions(self, feature_ndims):

    class FeatureNdimsKernel(psd_kernel.PositiveSemidefiniteKernel):

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
  @test_util.disable_test_for_backend(
      disable_numpy=True, reason='No dynamic shapes.')
  def testDynamicBatchShape(self, params, shape):
    tensor_params = tf1.placeholder_with_default(params, shape=None)
    k = TestKernel(tensor_params)
    self.assertAllEqual(shape, self.evaluate(k.batch_shape_tensor()))

  @parameterized.named_parameters(
      ('Shape [] kernel', 2., [4]),
      ('Shape [1] kernel', [2.], [3, 1]),
      ('Shape [2] kernel', [1., 2.], [2]),
      ('Shape [2, 1] kernel', [[1.], [2.]], [2, 3]))
  def testBroadcastParametersWithBatchShape(self, params, broadcast_shape):
    k = TestKernel(params)._broadcast_parameters_with_batch_shape(
        broadcast_shape)
    self.assertAllEqual(broadcast_shape, k.batch_shape)
    self.assertAllEqual(broadcast_shape, self.evaluate(k.batch_shape_tensor()))

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

  @test_util.disable_test_for_backend(
      disable_numpy=True, reason='No dynamic shapes in numpy.')
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
    k0_ct = CompositeTensorTestKernel(PARAMS_0)
    sum_kernel = k0 + k0 + k0_ct
    self.assertLen(sum_kernel.kernels, 3)
    sum_kernel += k0 + k0
    self.assertLen(sum_kernel.kernels, 5)
    self.assertNotIsInstance(sum_kernel, tf.__internal__.CompositeTensor)

    product_kernel = k0 * k0 * k0_ct
    self.assertLen(product_kernel.kernels, 3)
    product_kernel *= k0 * k0
    self.assertLen(product_kernel.kernels, 5)
    self.assertNotIsInstance(product_kernel, tf.__internal__.CompositeTensor)

    # This works because we special-case (0 + kernel), which is how sum is
    # initialized.
    sum_of_list_of_kernels = sum([k0, k0, k0])
    self.assertLen(sum_of_list_of_kernels.kernels, 3)

  def testOperatorOverloadsCompositeTensor(self):
    k0 = CompositeTensorTestKernel(PARAMS_0)
    sum_kernel = k0 + k0 + k0
    self.assertLen(sum_kernel.kernels, 3)
    sum_kernel += k0 + k0
    self.assertLen(sum_kernel.kernels, 5)
    self.assertIsInstance(sum_kernel, tf.__internal__.CompositeTensor)

    product_kernel = k0 * k0 * k0
    self.assertLen(product_kernel.kernels, 3)
    product_kernel *= k0 * k0
    self.assertLen(product_kernel.kernels, 5)
    self.assertIsInstance(product_kernel, tf.__internal__.CompositeTensor)

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

    self.assertAllEqual([2], sum_kernel[..., 1].batch_shape)
    self.assertAllEqual(
        sum([self.evaluate(k.matrix(x, y))
             for k in sum_kernel[..., 1].kernels]),
        self.evaluate(sum_kernel[..., 1].matrix(x, y)))

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

    self.assertAllEqual(
        [1, 2], self.evaluate(sum_kernel[:1].batch_shape_tensor()))
    self.assertAllEqual(
        sum([self.evaluate(k.matrix(x, y))
             for k in sum_kernel[:1].kernels]),
        self.evaluate(sum_kernel[:1].matrix(x, y)))

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

    self.assertAllEqual([1], product_kernel[0, 1:].batch_shape)
    self.assertAllEqual(
        functools.reduce(
            operator.mul,
            [self.evaluate(k.matrix(x, y))
             for k in product_kernel[0, 1:].kernels]),
        self.evaluate(product_kernel[0, 1:].matrix(x, y)))

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

    self.assertAllEqual(
        [2, 1], self.evaluate(product_kernel[:, :1].batch_shape_tensor()))
    self.assertAllEqual(
        functools.reduce(
            operator.mul,
            [self.evaluate(k.matrix(x, y))
             for k in product_kernel[:, :1].kernels]),
        self.evaluate(product_kernel[:, :1].matrix(x, y)))

  @test_util.disable_test_for_backend(
      disable_numpy=True, reason='DType mismatch not caught in numpy.')
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

  @test_util.disable_test_for_backend(
      disable_numpy=True, reason='DType mismatch not caught in numpy.')
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

  @test_util.disable_test_for_backend(
      disable_numpy=True,
      reason='Numpy has no notion of CompositeTensor/Pytree.')
  def testInputOutputOfJittedFunction(self):
    @tf.function(jit_compile=True)
    def create_kernel(m):
      return CompositeTensorTestKernel(multiplier=m)

    create_kernel(np.array([1., 2., 3.]))

  @parameterized.parameters(
      {'foo': 1, 'bar': 1},
      {'foo': 1, 'bar': 2},
      {'foo': 3, 'bar': 2})
  def testMultipartKernelFeatureNdims(self, **feature_ndims):
    k0 = TestKernel(multiplier=3.)
    mk0 = test_util.MultipartKernel(k0, feature_ndims=feature_ndims)
    k1 = TestKernel(multiplier=2.)
    mk1 = test_util.MultipartKernel(k1, feature_ndims=feature_ndims)

    # Define inputs to the multipart kernel. `x` has batch shape, `y` does not.
    x = {'foo': np.random.normal(
        size=[3, 4] + [5] * feature_ndims['foo']).astype(np.float32),
         'bar': np.random.normal(
             size=[4] + [6] * feature_ndims['bar']).astype(np.float32)}
    y = {'foo': np.random.normal(
        size=[5] * feature_ndims['foo']).astype(np.float32),
         'bar': np.random.normal(
             size=[6] * feature_ndims['bar']).astype(np.float32)}

    # Reshape and concatenate the inputs for the simple kernel.
    x_ = tf.concat(
        [tf.reshape(x['foo'], [3, 4, -1]),
         tf.stack([tf.reshape(x['bar'], [4, -1])] * 3, axis=0)], axis=-1)
    y_ = tf.concat(
        [tf.reshape(y['foo'], [-1]), tf.reshape(y['bar'], [-1])], axis=-1)

    self.assertAllClose(k0.apply(x_, y_), mk0.apply(x, y), rtol=1e-5)
    self.assertAllClose((k0 + k1).apply(x_, y_), (mk0 + mk1).apply(x, y),
                        rtol=1e-5)
    self.assertAllClose((k0 * k1).apply(x_, y_), (mk0 * mk1).apply(x, y),
                        rtol=1e-5)

  def testMultipartKernelBatchExampleShapesBroadcast(self):
    k = TestKernel(multiplier=3.)
    mk = test_util.MultipartKernel(k, feature_ndims={'foo': 2, 'bar': 1})

    # x has broadcasted batch and example dims [2, 3, 4].
    x = {'foo': tf.ones([3, 4, 2, 2]), 'bar': tf.ones([2, 3, 1, 5])}

    # y has broadcasted batch and example dims [1, 3, 1]
    y = {'foo': tf.ones([1, 3, 1, 2, 2]), 'bar': tf.ones([1, 1, 5])}

    # All extra dimensions are in the batch shape and are broadcasted together.
    self.assertAllEqual(mk.apply(x, y).shape, [2, 3, 4])

    # The first two extra dimensions are broadcasted together, and the remaining
    # extra dimensions come in sequence as example dims.
    self.assertAllEqual(mk.matrix(x, y).shape, [2, 3, 4, 1])
    self.assertAllEqual(mk.matrix(x, x).shape, [2, 3, 4, 4])

    # The first extra dimension is the batch dimension, and the remaining extra
    # dimensions come in sequence as example dimensions.
    self.assertAllEqual(
        mk.tensor(x, y, x1_example_ndims=2, x2_example_ndims=2).shape,
        [2, 3, 4, 3, 1])
    self.assertAllEqual(
        mk.tensor(x, y, x1_example_ndims=2, x2_example_ndims=3).shape,
        [2, 3, 4, 1, 3, 1])


if __name__ == '__main__':
  test_util.main()
