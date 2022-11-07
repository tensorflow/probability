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
"""Tests for Matern kernels."""

from absl.testing import parameterized

import numpy as np
import scipy.special as scipy_special
import tensorflow.compat.v2 as tf

from tensorflow_probability.python.internal import test_util
from tensorflow_probability.python.math import gradient
from tensorflow_probability.python.math.psd_kernels import matern


class _MaternTestCase(test_util.TestCase):
  """Mixin test for Matern type kernels.

  Subclasses must specify _kernel_type and _numpy_kernel.
  """

  @test_util.disable_test_for_backend(
      disable_numpy=True, reason='DType mismatch not caught in numpy.')
  def testMismatchedFloatTypesAreBad(self):
    self._kernel_type(1., 1)  # Should be OK (float32 fallback).
    self._kernel_type(1., np.float64(1.))  # Should be OK.
    with self.assertRaises(TypeError):
      self._kernel_type(np.float32(1.), np.float64(1.))

    with self.assertRaises(TypeError):
      self._kernel_type(
          amplitude=np.float32(1.), inverse_length_scale=np.float64(1.))

  def testBatchShape(self):
    amplitude = np.random.uniform(2, 3., size=[3, 1, 2]).astype(np.float32)
    length_scale = np.random.uniform(2, 3., size=[1, 3, 1]).astype(np.float32)
    k = self._kernel_type(amplitude, length_scale)
    self.assertAllEqual(tf.TensorShape([3, 3, 2]), k.batch_shape)
    self.assertAllEqual([3, 3, 2], self.evaluate(k.batch_shape_tensor()))

  def testBatchShapeWithNone(self):
    amplitude = np.random.uniform(2, 3., size=[3, 1, 2]).astype(np.float32)
    k = self._kernel_type(amplitude, None)
    self.assertAllEqual(tf.TensorShape([3, 1, 2]), k.batch_shape)
    self.assertAllEqual([3, 1, 2], self.evaluate(k.batch_shape_tensor()))

  def testValidateArgs(self):
    with self.assertRaisesOpError('amplitude must be positive'):
      k = self._kernel_type(amplitude=-1., length_scale=1., validate_args=True)
      self.evaluate(k.apply([[1.]], [[1.]]))

    if not tf.executing_eagerly():
      with self.assertRaisesOpError('length_scale must be positive'):
        k = self._kernel_type(
            amplitude=1., length_scale=-1., validate_args=True)
        self.evaluate(k.apply([[1.]], [[1.]]))

    # But `None`'s are ok
    k = self._kernel_type(amplitude=None, length_scale=None, validate_args=True)
    self.evaluate(k.apply([1.], [1.]))

  @parameterized.parameters({
      'feature_ndims': 1,
      'dtype': np.float32,
      'dims': 3
  }, {
      'feature_ndims': 1,
      'dtype': np.float32,
      'dims': 4
  }, {
      'feature_ndims': 2,
      'dtype': np.float32,
      'dims': 2
  }, {
      'feature_ndims': 2,
      'dtype': np.float64,
      'dims': 3
  }, {
      'feature_ndims': 3,
      'dtype': np.float64,
      'dims': 2
  }, {
      'feature_ndims': 3,
      'dtype': np.float64,
      'dims': 3
  })
  def testValuesAreCorrect(self, feature_ndims, dtype, dims):
    amplitude = np.array(5., dtype=dtype)
    length_scale = np.array(.2, dtype=dtype)

    np.random.seed(42)
    k = self._kernel_type(
        amplitude=amplitude,
        length_scale=length_scale,
        feature_ndims=feature_ndims)
    shape = [dims] * feature_ndims
    for _ in range(5):
      x = np.random.uniform(-1, 1, size=shape).astype(dtype)
      y = np.random.uniform(-1, 1, size=shape).astype(dtype)
      self.assertAllClose(
          self._numpy_kernel(amplitude, length_scale, x, y),
          self.evaluate(k.apply(x, y)))

  @parameterized.parameters({
      'feature_ndims': 1,
      'dtype': np.float32,
      'dims': 3
  }, {
      'feature_ndims': 1,
      'dtype': np.float32,
      'dims': 4
  }, {
      'feature_ndims': 2,
      'dtype': np.float32,
      'dims': 2
  }, {
      'feature_ndims': 2,
      'dtype': np.float64,
      'dims': 3
  }, {
      'feature_ndims': 3,
      'dtype': np.float64,
      'dims': 2
  }, {
      'feature_ndims': 3,
      'dtype': np.float64,
      'dims': 3
  })
  def testValuesAreCorrectInverseLengthScale(self, feature_ndims, dtype, dims):
    amplitude = np.array(5., dtype=dtype)
    inverse_length_scale = np.linspace(1., 20., 20, dtype=dtype)

    np.random.seed(42)
    k = self._kernel_type(
        amplitude=amplitude,
        inverse_length_scale=inverse_length_scale,
        feature_ndims=feature_ndims)
    k_ls = self._kernel_type(
        amplitude=amplitude,
        length_scale=1. / inverse_length_scale,
        feature_ndims=feature_ndims)
    shape = [dims] * feature_ndims
    x = np.random.uniform(-1, 1, size=shape).astype(dtype)
    y = np.random.uniform(-1, 1, size=shape).astype(dtype)
    self.assertAllClose(
        self.evaluate(k.apply(x, y)),
        self.evaluate(k_ls.apply(x, y)))

  def testShapesAreCorrect(self):
    k = self._kernel_type(amplitude=1., length_scale=1.)

    x = np.ones([4, 3], np.float32)
    y = np.ones([5, 3], np.float32)

    self.assertAllEqual([4, 5], k.matrix(x, y).shape)
    self.assertAllEqual(
        k.matrix(tf.stack([x] * 2), tf.stack([y] * 2)).shape, [2, 4, 5])

    k = self._kernel_type(
        amplitude=np.ones([2, 1, 1], np.float32),
        length_scale=np.ones([1, 3, 1], np.float32))
    self.assertAllEqual(
        [2, 3, 2, 4, 5],
        #`--'  |  `--'
        #  |   |    `- matrix shape
        #  |   `- from input batch shapes
        #  `- from broadcasting kernel params
        k.matrix(
            tf.stack([x] * 2),  # shape [2, 4, 3]
            tf.stack([y] * 2)  # shape [2, 5, 3]
        ).shape)

  @test_util.numpy_disable_gradient_test
  def testGradsAtIdenticalInputsAreZeroNotNaN(self):
    k = self._kernel_type()
    x = tf.constant(np.arange(3 * 5, dtype=np.float32).reshape(3, 5))

    grads = [
        gradient.value_and_gradient(lambda x: k.apply(x, x)[i], x)[1]  # pylint: disable=cell-var-from-loop
        for i in range(3)
    ]

    self.assertAllEqual(
        [np.zeros(np.int32(grad.shape), np.float32) for grad in grads],
        self.evaluate(grads))


@test_util.test_all_tf_execution_regimes
class GeneralizedMaternTest(_MaternTestCase):

  def _kernel_type(
      self,
      amplitude=None,
      length_scale=None,
      feature_ndims=1,
      inverse_length_scale=None,
      validate_args=None,
      df=None):
    # Make sure tests work for scalar DF. Non-scalar DF is handled below.
    if df is None:
      df = np.pi
    return matern.GeneralizedMatern(
        df=df,
        amplitude=amplitude,
        length_scale=length_scale,
        inverse_length_scale=inverse_length_scale,
        feature_ndims=feature_ndims,
        validate_args=validate_args)

  def _numpy_kernel(self, amplitude, length_scale, x, y, df=None):
    if df is None:
      df = np.pi
    norm = np.sqrt(2 * df * np.sum((x - y)**2)) / length_scale
    return (amplitude**2 * 2 ** (1 - df) / scipy_special.gamma(df) *
            norm ** df * scipy_special.kv(df, norm))

  def testValidateDFPositive(self):
    with self.assertRaisesOpError('df must be positive'):
      k = matern.GeneralizedMatern(
          df=-1., amplitude=1., length_scale=1., validate_args=True)
      self.evaluate(k.apply([[1.]], [[1.]]))

  def testDFBatchShape(self):
    df = np.random.uniform(2, 3., size=[5, 1, 3, 2]).astype(np.float32)
    amplitude = np.random.uniform(2, 3., size=[1, 3, 1, 2]).astype(np.float32)
    length_scale = np.random.uniform(
        2, 3., size=[5, 1, 1, 2]).astype(np.float32)
    k = matern.GeneralizedMatern(df, amplitude, length_scale)
    self.assertAllEqual(tf.TensorShape([5, 3, 3, 2]), k.batch_shape)
    self.assertAllEqual([5, 3, 3, 2], self.evaluate(k.batch_shape_tensor()))

  @parameterized.parameters({
      'feature_ndims': 1,
      'dtype': np.float32,
      'dims': 3
  }, {
      'feature_ndims': 2,
      'dtype': np.float32,
      'dims': 2
  }, {
      'feature_ndims': 2,
      'dtype': np.float64,
      'dims': 3
  }, {
      'feature_ndims': 3,
      'dtype': np.float64,
      'dims': 2
  })
  def testReplicateMaternOneHalf(self, feature_ndims, dtype, dims):
    amplitude = np.array(5., dtype=dtype)
    length_scale = np.array(.2, dtype=dtype)

    k = matern.GeneralizedMatern(
        df=0.5,
        amplitude=amplitude,
        length_scale=length_scale,
        feature_ndims=feature_ndims)
    matern_one_half = matern.MaternOneHalf(
        amplitude=amplitude,
        length_scale=length_scale,
        feature_ndims=feature_ndims)
    shape = [dims] * feature_ndims
    for _ in range(5):
      x = np.random.uniform(-1, 1, size=shape).astype(dtype)
      y = np.random.uniform(-1, 1, size=shape).astype(dtype)
      self.assertAllClose(
          self.evaluate(matern_one_half.apply(x, y)),
          self.evaluate(k.apply(x, y)))

      self.assertAllClose(
          self.evaluate(matern_one_half.apply(x, x)),
          self.evaluate(k.apply(x, x)))

  @parameterized.parameters({
      'feature_ndims': 1,
      'dtype': np.float32,
      'dims': 3
  }, {
      'feature_ndims': 2,
      'dtype': np.float32,
      'dims': 2
  }, {
      'feature_ndims': 2,
      'dtype': np.float64,
      'dims': 3
  }, {
      'feature_ndims': 3,
      'dtype': np.float64,
      'dims': 2
  })
  def testReplicateMaternThreeHalves(self, feature_ndims, dtype, dims):
    amplitude = np.array(5., dtype=dtype)
    length_scale = np.array(.2, dtype=dtype)

    k = matern.GeneralizedMatern(
        df=1.5,
        amplitude=amplitude,
        length_scale=length_scale,
        feature_ndims=feature_ndims)
    matern_three_halves = matern.MaternThreeHalves(
        amplitude=amplitude,
        length_scale=length_scale,
        feature_ndims=feature_ndims)
    shape = [dims] * feature_ndims
    for _ in range(5):
      x = np.random.uniform(-1, 1, size=shape).astype(dtype)
      y = np.random.uniform(-1, 1, size=shape).astype(dtype)
      self.assertAllClose(
          self.evaluate(matern_three_halves.apply(x, y)),
          self.evaluate(k.apply(x, y)))

      self.assertAllClose(
          self.evaluate(matern_three_halves.apply(x, x)),
          self.evaluate(k.apply(x, x)))

  @parameterized.parameters({
      'feature_ndims': 1,
      'dtype': np.float32,
      'dims': 3
  }, {
      'feature_ndims': 2,
      'dtype': np.float32,
      'dims': 2
  }, {
      'feature_ndims': 2,
      'dtype': np.float64,
      'dims': 3
  }, {
      'feature_ndims': 3,
      'dtype': np.float64,
      'dims': 2
  })
  def testReplicateMaternFiveHalves(self, feature_ndims, dtype, dims):
    amplitude = np.array(5., dtype=dtype)
    length_scale = np.array(.2, dtype=dtype)

    k = matern.GeneralizedMatern(
        df=2.5,
        amplitude=amplitude,
        length_scale=length_scale,
        feature_ndims=feature_ndims)
    matern_five_halves = matern.MaternFiveHalves(
        amplitude=amplitude,
        length_scale=length_scale,
        feature_ndims=feature_ndims)
    shape = [dims] * feature_ndims
    for _ in range(5):
      x = np.random.uniform(-1, 1, size=shape).astype(dtype)
      y = np.random.uniform(-1, 1, size=shape).astype(dtype)
      self.assertAllClose(
          self.evaluate(matern_five_halves.apply(x, y)),
          self.evaluate(k.apply(x, y)))

      self.assertAllClose(
          self.evaluate(matern_five_halves.apply(x, x)),
          self.evaluate(k.apply(x, x)))

  @test_util.jax_disable_test_missing_functionality(
      'BesselKve does not have gradients with respect to `df` parameter.')
  @test_util.numpy_disable_gradient_test
  def testGradsAtIdenticalInputsAreZeroNotNaN(self):
    k = self._kernel_type()
    x = tf.constant(np.arange(3 * 5, dtype=np.float32).reshape(3, 5))

    grads = [
        gradient.value_and_gradient(lambda x: k.apply(x, x)[i], x)[1]  # pylint: disable=cell-var-from-loop
        for i in range(3)
    ]

    self.assertAllEqual(
        [np.zeros(np.int32(grad.shape), np.float32) for grad in grads],
        self.evaluate(grads))

  @test_util.jax_disable_test_missing_functionality(
      'BesselKve does not have gradients with respect to `df` parameter.')
  @test_util.numpy_disable_gradient_test
  def testGradsNoneDfParameter(self):
    x = tf.constant([5.], dtype=np.float32)
    y = tf.constant([-2.], dtype=np.float32)
    def _apply(df):
      k = matern.GeneralizedMatern(
          df=df, amplitude=1., length_scale=1., feature_ndims=1)
      return k.apply(x, y)
    df = tf.constant(2.5, dtype=np.float32)

    self.assertIsNone(gradient.value_and_gradient(_apply, df)[1])


@test_util.test_all_tf_execution_regimes
class MaternOneHalfTest(_MaternTestCase):

  _kernel_type = matern.MaternOneHalf

  def _numpy_kernel(self, amplitude, length_scale, x, y):
    norm = np.sqrt(np.sum((x - y)**2)) / length_scale
    return amplitude**2 * np.exp(-norm)


@test_util.test_all_tf_execution_regimes
class MaternThreeHalvesTest(_MaternTestCase):

  _kernel_type = matern.MaternThreeHalves

  def _numpy_kernel(self, amplitude, length_scale, x, y):
    norm = np.sqrt(3. * np.sum((x - y)**2)) / length_scale
    return amplitude**2 * (1 + norm) * np.exp(-norm)


@test_util.test_all_tf_execution_regimes
class MaternFiveHalvesTest(_MaternTestCase):

  _kernel_type = matern.MaternFiveHalves

  def _numpy_kernel(self, amplitude, length_scale, x, y):
    norm = np.sqrt(5. * np.sum((x - y)**2)) / length_scale
    return amplitude**2 * (1 + norm + norm**2 / 3.) * np.exp(-norm)


del _MaternTestCase
# Otherwise, the abstract test case will fail for lack of _kernel_type


if __name__ == '__main__':
  test_util.main()
