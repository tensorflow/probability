# Copyright 2022 The TensorFlow Probability Authors.
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
"""Tests for PSD kernel linop."""

import functools

# Dependency imports

from absl.testing import parameterized
import numpy as np
import tensorflow.compat.v1 as tf1
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp

from tensorflow_probability.python.internal import test_util

tfpk = tfp.math.psd_kernels

JAX_MODE = False


@test_util.test_all_tf_execution_regimes
class _LinearOperatorInterpolatedPSDKernelTest(test_util.TestCase):
  use_static_shape = True
  dtype = tf.float32

  # pyformat: disable
  @parameterized.named_parameters((  # pylint: disable=g-complex-comprehension
      (f'(x={x_batch}, diag_shift={diag_shift_batch}, '
       f'length_scale={length_scale_batch}, mat2={mat2_batch}, '
       f'is_square={is_square})'),
      x_batch,
      diag_shift_batch,
      length_scale_batch,
      mat2_batch,
      is_square,
  )
                                  for x_batch in ((), (3,))
                                  for length_scale_batch in ((), (3,))
                                  for mat2_batch in ((), (3,))
                                  for is_square in (True, False)
                                  for diag_shift_batch in (
                                      ((), (3,)) if is_square else ((),)
                                  ))
  # pyformat: enable
  def testBatchedInputs(self, x_batch, diag_shift_batch, length_scale_batch,
                        mat2_batch, is_square):
    """Self-consistency checks with possibly batched inputs."""
    if not self.use_static_shape and tf.executing_eagerly():
      self.skipTest('Cannot test dynamic shape in Eager execution.')
    # Arrange inputs.
    ndims = 2
    x1 = np.stack([np.linspace(-5., 5., 5)] * ndims, axis=-1)
    x1_length = x1.shape[-2]
    if is_square:
      x2 = None
      x2_length = x1.shape[-2]
    else:
      x2 = np.stack([np.linspace(-5., 5., 3)] * ndims, axis=-1)
      x2_length = x2.shape[-2]
    mat2 = tf.random.normal(
        mat2_batch + (x2_length, 4),
        dtype=self.dtype,
        seed=test_util.test_seed())
    bounds_min = [-5.] * ndims
    bounds_max = [5.] * ndims
    num_interpolation_points = 7
    diag_shift = 1. if is_square else None
    length_scale = 3.

    def _add_batch(v, batch):
      v = np.array(v)
      return np.tile(
          np.reshape(v, (1,) * len(batch) + tuple(v.shape)),
          batch + (1,) * len(v.shape))

    if x_batch:
      x1 = _add_batch(x1, x_batch)
    if diag_shift_batch:
      diag_shift = _add_batch(diag_shift, diag_shift_batch)
    if length_scale_batch:
      length_scale = _add_batch(length_scale, length_scale_batch)

    # Construct linop, do some operations on it.
    linop = tfp.experimental.linalg.LinearOperatorInterpolatedPSDKernel(
        kernel=tfpk.MaternThreeHalves(
            length_scale=self.make_input(length_scale, [])),
        bounds_min=tf.constant(bounds_min, self.dtype),
        bounds_max=tf.constant(bounds_max, self.dtype),
        num_interpolation_points=num_interpolation_points,
        x1=self.make_input(x1, [-2, -1]),
        x2=x2 if x2 is None else self.make_input(x2, [-2, -1]),
        diag_shift=diag_shift if diag_shift is None else self.make_input(
            diag_shift, []))

    dense = linop.to_dense()
    rows = linop.row([0, 1, 2])
    if is_square:
      diag_part = linop.diag_part()
    matmul = linop.matmul(self.make_input(mat2, [-2, -1]))
    dense_matmul = dense @ mat2

    # Perform the tests.
    expected_batch = functools.reduce(
        tf.broadcast_static_shape,
        map(tf.TensorShape, (x_batch, diag_shift_batch, length_scale_batch)))
    expected_mat_shape = tuple(expected_batch) + (x1_length, x2_length)
    expected_diag_shape = tuple(expected_batch) + (x1_length,)
    expected_row_shape = tuple(expected_batch) + (3,) + (x2_length,)
    expected_matmul_shape = (
        tuple(
            tf.broadcast_static_shape(
                expected_batch, tf.TensorShape(mat2_batch))) + (x1_length, 4))

    with self.subTest('shapes'):
      if self.use_static_shape:
        self.assertAllEqual(expected_mat_shape, dense.shape)
        self.assertAllEqual(expected_mat_shape, linop.shape)
        if is_square:
          self.assertAllEqual(expected_diag_shape, diag_part.shape)
        self.assertAllEqual(expected_row_shape, rows.shape)
        self.assertAllEqual(expected_matmul_shape, matmul.shape)

      self.assertAllEqual(expected_mat_shape, self.evaluate(tf.shape(dense)))
      self.assertAllEqual(expected_mat_shape,
                          self.evaluate(linop.shape_tensor()))
      if is_square:
        self.assertAllEqual(expected_diag_shape,
                            self.evaluate(tf.shape(diag_part)))
      self.assertAllEqual(expected_row_shape, self.evaluate(tf.shape(rows)))
      self.assertAllEqual(expected_matmul_shape,
                          self.evaluate(tf.shape(matmul)))

    with self.subTest('dtypes'):
      self.assertEqual(self.dtype, dense.dtype)
      self.assertEqual(self.dtype, linop.dtype)
      if is_square:
        self.assertEqual(self.dtype, diag_part.dtype)
      self.assertEqual(self.dtype, rows.dtype)
      self.assertEqual(self.dtype, matmul.dtype)

    with self.subTest('self_consistency'):
      dense_ = self.evaluate(dense)
      self.assertAllClose(rows, dense_[..., [0, 1, 2], :])
      if is_square:
        expected_diag_part_ = self.evaluate(tf.linalg.diag_part(dense))
        self.assertAllClose(diag_part, expected_diag_part_)
      self.assertAllClose(dense_matmul, matmul)

  def make_input(self, value, static_dims=None):
    value = tf.convert_to_tensor(value, dtype=self.dtype)
    if self.use_static_shape:
      return value
    else:
      if static_dims is None:
        shape = None
      else:
        shape = value.shape
        static_dims = [(s + len(shape)) % len(shape) for s in static_dims]
        shape = [
            shape[i] if i in static_dims else None for i in range(len(shape))
        ]
      return tf1.placeholder_with_default(value, shape=shape)


class _LinearOperatorInterpolatedPSDKernelTestStaticTest(
    _LinearOperatorInterpolatedPSDKernelTest):
  use_static_shape = True

  @parameterized.named_parameters(
      ('Square', True),
      ('Rect', False),
  )
  def test3D(self, is_square):
    """Performs some minimal self-consistency checks with 3D interpolation."""
    # Arrange inputs.
    ndims = 3
    x1 = np.stack([np.linspace(-5., 5., 5)] * ndims, axis=-1)
    if is_square:
      x2 = None
      x2_length = x1.shape[-2]
    else:
      x2 = np.stack([np.linspace(-5., 5., 3)] * ndims, axis=-1)
      x2_length = x2.shape[-2]
    mat2 = tf.random.normal((x2_length, 4),
                            dtype=self.dtype,
                            seed=test_util.test_seed())
    bounds_min = [-5.] * ndims
    bounds_max = [5.] * ndims
    num_interpolation_points = 7
    diag_shift = 1. if is_square else None
    length_scale = 3.

    # Construct linop, do some operations on it.
    linop = tfp.experimental.linalg.LinearOperatorInterpolatedPSDKernel(
        kernel=tfpk.MaternThreeHalves(
            length_scale=self.make_input(length_scale, [])),
        bounds_min=tf.constant(bounds_min, self.dtype),
        bounds_max=tf.constant(bounds_max, self.dtype),
        num_interpolation_points=num_interpolation_points,
        x1=self.make_input(x1, [-2, -1]),
        x2=x2 if x2 is None else self.make_input(x2, [-2, -1]),
        diag_shift=diag_shift if diag_shift is None else self.make_input(
            diag_shift, []))

    dense = linop.to_dense()
    rows = linop.row([0, 1, 2])
    if is_square:
      diag_part = linop.diag_part()
    matmul = linop.matmul(self.make_input(mat2, [-2, -1]))
    dense_matmul = dense @ mat2

    # Perform the tests.
    dense_ = self.evaluate(dense)
    self.assertAllClose(rows, dense_[..., [0, 1, 2], :])
    if is_square:
      expected_diag_part_ = self.evaluate(tf.linalg.diag_part(dense))
      self.assertAllClose(diag_part, expected_diag_part_)
    self.assertAllClose(dense_matmul, matmul)

  @parameterized.named_parameters(
      ('Square', True),
      ('Rect', False),
  )
  def testLinearKernel(self, is_square):
    """Checks outputs against a linear kernel, where interpolation is exact."""
    # Arrange inputs.
    ndims = 3
    x1 = np.stack([np.linspace(-5., 5., 5)] * ndims, axis=-1)
    x1_length = x1.shape[-2]
    if is_square:
      x2 = None
      x2_length = x1.shape[-2]
    else:
      x2 = np.stack([np.linspace(-5., 5., 3)] * ndims, axis=-1)
      x2_length = x2.shape[-2]
    bounds_min = [-5.] * ndims
    bounds_max = [5.] * ndims
    num_interpolation_points = 3
    diag_shift = 1. if is_square else None
    shift = 1.

    kernel = tfpk.Linear(shift=self.make_input(shift, []))
    # Construct linop, do some operations on it.
    linop = tfp.experimental.linalg.LinearOperatorInterpolatedPSDKernel(
        kernel=kernel,
        bounds_min=tf.constant(bounds_min, self.dtype),
        bounds_max=tf.constant(bounds_max, self.dtype),
        num_interpolation_points=num_interpolation_points,
        x1=self.make_input(x1, [-2, -1]),
        x2=x2 if x2 is None else self.make_input(x2, [-2, -1]),
        diag_shift=diag_shift if diag_shift is None else self.make_input(
            diag_shift, []))

    dense = linop.to_dense()
    kernel_dense = kernel.matrix(x1, x1 if x2 is None else x2)
    if diag_shift is not None:
      kernel_dense += diag_shift * tf.eye(
          x1_length, x2_length, dtype=self.dtype)
    self.assertAllClose(kernel_dense, dense)


class LinearOperatorInterpolatedPSDKernelStatic32Test(
    _LinearOperatorInterpolatedPSDKernelTestStaticTest):
  dtype = tf.float32


class LinearOperatorInterpolatedPSDKernelStatic64Test(
    _LinearOperatorInterpolatedPSDKernelTestStaticTest):
  dtype = tf.float64


if not JAX_MODE:

  class LinearOperatorInterpolatedPSDKernelDynamic32Test(
      _LinearOperatorInterpolatedPSDKernelTest):
    use_static_shape = False
    dtype = tf.float32

  class LinearOperatorInterpolatedPSDKernelDynamic64Test(
      _LinearOperatorInterpolatedPSDKernelTest):
    use_static_shape = False
    dtype = tf.float64


del _LinearOperatorInterpolatedPSDKernelTest
del _LinearOperatorInterpolatedPSDKernelTestStaticTest

if __name__ == '__main__':
  test_util.main()
