# Copyright 2019 The TensorFlow Probability Authors.
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


# Dependency imports

from absl.testing import parameterized
import numpy as np
import tensorflow.compat.v2 as tf

from tensorflow_probability.python.experimental.linalg import linear_operator_psd_kernel
from tensorflow_probability.python.internal import test_util
from tensorflow_probability.python.math.psd_kernels import exponentiated_quadratic
from tensorflow_probability.python.math.psd_kernels import polynomial


def skip_if_no_xla(skip_test_fn):
  try:
    tf.function(lambda: tf.constant(0), jit_compile=True)()
  except tf.errors.UnimplementedError as e:
    if 'Could not find compiler' in str(e):
      skip_test_fn('XLA not available')


@test_util.test_all_tf_execution_regimes
class LinearOperatorPSDKernelTest(test_util.TestCase):
  """Tests for linear_operator_psd_kernel.LinearOperatorPSDKernel."""

  def test_shape(self):
    kernel = exponentiated_quadratic.ExponentiatedQuadratic(
        amplitude=tf.random.uniform([17, 1, 1]), feature_ndims=2)
    x1 = tf.random.normal([1, 11, 5, 2, 13])
    x2 = tf.random.normal([7, 1, 3, 2, 13])
    linop = linear_operator_psd_kernel.LinearOperatorPSDKernel(kernel, x1, x2)
    self.assertAllEqual((17, 7, 11, 5, 3), linop.shape)
    self.assertAllEqual((17, 7, 11), linop.batch_shape)

  def test_diag_part(self):
    kernel = exponentiated_quadratic.ExponentiatedQuadratic()
    x1 = tf.random.normal([7, 3, 5, 2])  # square matrix 5x5
    linop = linear_operator_psd_kernel.LinearOperatorPSDKernel(kernel, x1)
    expected, actual = self.evaluate([
        tf.linalg.diag_part(kernel.matrix(x1, x1)),
        linop.diag_part()
    ])
    self.assertAllClose(expected, actual)

    x2 = tf.random.normal([3, 11, 2])  # wide matrix 5x11
    linop = linear_operator_psd_kernel.LinearOperatorPSDKernel(kernel, x1, x2)
    expected, actual = self.evaluate([
        tf.linalg.diag_part(kernel.matrix(x1, x2)),
        linop.diag_part()
    ])
    self.assertAllClose(expected, actual)

    x2 = tf.random.normal([2, 2])  # tall matrix 5x2
    linop = linear_operator_psd_kernel.LinearOperatorPSDKernel(kernel, x1, x2)
    expected, actual = self.evaluate([
        tf.linalg.diag_part(kernel.matrix(x1, x2)),
        linop.diag_part()
    ])
    self.assertAllClose(expected, actual)

  def test_diag_part_xla(self):
    skip_if_no_xla(self.skipTest)
    if not tf.executing_eagerly(): return  # jit_compile is eager-only.
    kernel = exponentiated_quadratic.ExponentiatedQuadratic()
    x1 = tf.random.normal([7, 3, 5, 2])  # square matrix 5x5
    linop = linear_operator_psd_kernel.LinearOperatorPSDKernel(kernel, x1)
    expected, actual = self.evaluate([
        tf.linalg.diag_part(kernel.matrix(x1, x1)),
        tf.function(linop.diag_part, jit_compile=True)()
    ])
    self.assertAllClose(expected, actual)

    x2 = tf.random.normal([3, 11, 2])  # wide matrix 5x11
    linop = linear_operator_psd_kernel.LinearOperatorPSDKernel(kernel, x1, x2)
    expected, actual = self.evaluate([
        tf.linalg.diag_part(kernel.matrix(x1, x2)),
        tf.function(linop.diag_part, jit_compile=True)()
    ])
    self.assertAllClose(expected, actual)

    x2 = tf.random.normal([2, 2])  # tall matrix 5x2
    linop = linear_operator_psd_kernel.LinearOperatorPSDKernel(kernel, x1, x2)
    expected, actual = self.evaluate([
        tf.linalg.diag_part(kernel.matrix(x1, x2)),
        tf.function(linop.diag_part, jit_compile=True)()
    ])
    self.assertAllClose(expected, actual)

  def test_row_scalar(self):
    kernel = exponentiated_quadratic.ExponentiatedQuadratic()
    x1 = tf.random.normal([5, 2])
    x2 = tf.random.normal([7, 3, 5, 2])
    linop = linear_operator_psd_kernel.LinearOperatorPSDKernel(kernel, x1, x2)
    i = np.random.randint(0, 5)
    expected, actual = self.evaluate(
        [kernel.matrix(x1, x2)[..., i, :], linop.row(i)])
    self.assertAllClose(expected, actual)

  def test_row_batch(self):
    kernel = exponentiated_quadratic.ExponentiatedQuadratic()
    x1 = tf.random.normal([7, 1, 5, 2])
    x2 = tf.random.normal([1, 3, 4, 2])
    linop = linear_operator_psd_kernel.LinearOperatorPSDKernel(kernel, x1, x2)
    i = np.random.randint(0, 5, size=(7, 3))
    cov = kernel.matrix(x1, x2)
    expected, actual = self.evaluate([
        tf.gather(cov, i[..., tf.newaxis], batch_dims=2)[..., 0, :],
        linop.row(i)
    ])
    self.assertAllClose(expected, actual)

  def test_col_scalar(self):
    kernel = exponentiated_quadratic.ExponentiatedQuadratic()
    x1 = tf.random.normal([5, 2])
    x2 = tf.random.normal([7, 3, 5, 2])
    linop = linear_operator_psd_kernel.LinearOperatorPSDKernel(kernel, x1, x2)
    j = np.random.randint(0, 5)
    expected, actual = self.evaluate(
        [kernel.matrix(x1, x2)[..., j], linop.col(j)])
    self.assertAllClose(expected, actual)

  def test_col_batch(self):
    kernel = exponentiated_quadratic.ExponentiatedQuadratic()
    x1 = tf.random.normal([3, 5, 2])
    x2 = tf.random.normal([7, 1, 4, 2])
    linop = linear_operator_psd_kernel.LinearOperatorPSDKernel(kernel, x1, x2)
    j = np.random.randint(0, 4, size=(7, 3))
    cov = kernel.matrix(x1, x2)
    transpose = tf.linalg.matrix_transpose
    # Gather with batch_dims wants all the batch dims adjacent and leading, so
    # transpose-gather-transpose is easier to write than injecting a
    # range(nrows) column into the gather indices.
    expected, actual = self.evaluate([
        transpose(tf.gather(transpose(cov), j[..., tf.newaxis], batch_dims=2)
                 )[..., 0],
        linop.col(j)
    ])
    self.assertAllClose(expected, actual)

  def test_matmul(self):
    kernel = exponentiated_quadratic.ExponentiatedQuadratic()
    x1 = tf.random.normal([3, 2, 11])
    x2 = tf.random.normal([5, 1, 4, 11])
    linop = linear_operator_psd_kernel.LinearOperatorPSDKernel(kernel, x1, x2)
    cov = kernel.matrix(x1, x2)
    x = tf.random.normal([4, 3])
    expected, actual = self.evaluate([tf.matmul(cov, x), linop.matmul(x)])
    self.assertAllClose(expected, actual)

  def test_matmul_chunked(self):
    kernel = exponentiated_quadratic.ExponentiatedQuadratic()
    x1 = tf.random.normal([3, 2, 11])
    x2 = tf.random.normal([5, 1, 14, 11])
    linop = linear_operator_psd_kernel.LinearOperatorPSDKernel(
        kernel, x1, x2, num_matmul_parts=7)
    cov = kernel.matrix(x1, x2)
    x = tf.random.normal([14, 3])
    expected, actual = self.evaluate([tf.matmul(cov, x), linop.matmul(x)])
    self.assertAllClose(expected, actual)

  @parameterized.named_parameters(
      (dict(testcase_name='_{}chunk'.format(n), nchunks=n) for n in (2, 5)))
  def test_matmul_chunked_with_remainder(self, nchunks):
    kernel = exponentiated_quadratic.ExponentiatedQuadratic()
    x1 = tf.random.normal([3, 2, 11])
    x2 = tf.random.normal([5, 1, 17, 11])
    linop = linear_operator_psd_kernel.LinearOperatorPSDKernel(
        kernel, x1, x2, num_matmul_parts=nchunks)
    cov = kernel.matrix(x1, x2)
    x = tf.random.normal([17, 3])
    expected, actual = self.evaluate([tf.matmul(cov, x), linop.matmul(x)])
    self.assertAllClose(expected, actual)

  def test_matmul_chunked_grad(self):
    kernel = exponentiated_quadratic.ExponentiatedQuadratic()
    x1 = tf.random.normal([5, 3])
    x2 = tf.random.normal([7, 3])
    linop = linear_operator_psd_kernel.LinearOperatorPSDKernel(
        kernel, x1, x2, num_matmul_parts=3)
    x = tf.random.normal([7, 2])
    with tf.GradientTape() as tape:
      tape.watch((x1, x2, x))
      y = linop.matmul(x)

    out_grad = tf.random.normal(tf.shape(y))
    actuals = tape.gradient(y, (x1, x2, x), output_gradients=out_grad)

    with tf.GradientTape() as tape:
      tape.watch((x1, x2, x))
      y = tf.matmul(kernel.matrix(x1, x2), x)
    expecteds = tape.gradient(y, (x1, x2, x), output_gradients=out_grad)

    expecteds, actuals = self.evaluate([expecteds, actuals])

    self.assertEqual(len(expecteds), len(actuals))
    for expected, actual in zip(expecteds, actuals):
      self.assertAllClose(expected, actual)

  def test_matmul_xla(self):
    skip_if_no_xla(self.skipTest)
    if not tf.executing_eagerly(): return  # jit_compile is eager-only.
    kernel = exponentiated_quadratic.ExponentiatedQuadratic()
    x1 = tf.random.normal([5, 3])
    x2 = tf.random.normal([7, 3])
    linop = linear_operator_psd_kernel.LinearOperatorPSDKernel(
        kernel, x1, x2, num_matmul_parts=3)
    x = tf.random.normal([7, 2])

    @tf.function(jit_compile=True)
    def f():
      return linop.matmul(x)

    actual = f()
    expected = tf.matmul(kernel.matrix(x1, x2), x)

    expected, actual = self.evaluate([expected, actual])
    self.assertAllClose(expected, actual)

  def test_matmul_grad_xla(self):
    skip_if_no_xla(self.skipTest)
    if not tf.executing_eagerly(): return  # jit_compile is eager-only.
    kernel = exponentiated_quadratic.ExponentiatedQuadratic()
    x1 = tf.random.normal([5, 3])
    x2 = tf.random.normal([7, 3])
    linop = linear_operator_psd_kernel.LinearOperatorPSDKernel(
        kernel, x1, x2, num_matmul_parts=3)
    x = tf.random.normal([7, 2])

    @tf.function(jit_compile=True)
    def f():
      with tf.GradientTape() as tape:
        tape.watch((x1, x2, x))
        y = linop.matmul(x)

      out_grad = tf.random.normal(tf.shape(y))
      actuals = tape.gradient(y, (x1, x2, x), output_gradients=out_grad)
      return y, actuals, out_grad

    y, actuals, out_grad = f()

    with tf.GradientTape() as tape:
      tape.watch((x1, x2, x))
      y = tf.matmul(kernel.matrix(x1, x2), x)
    expecteds = tape.gradient(y, (x1, x2, x), output_gradients=out_grad)

    expecteds, actuals = self.evaluate([expecteds, actuals])

    self.assertEqual(len(expecteds), len(actuals))
    for expected, actual in zip(expecteds, actuals):
      self.assertAllClose(expected, actual)

  def test_matmul_grad_xla_kernelparams(self):
    skip_if_no_xla(self.skipTest)
    if not tf.executing_eagerly(): return  # jit_compile is eager-only.
    feature_dim = 3

    def kernel_fn(eq_params, poly_params):
      return (exponentiated_quadratic.ExponentiatedQuadratic(*eq_params) *
              polynomial.Polynomial(bias_amplitude=poly_params[0],
                                    shift=poly_params[1]))

    # TODO(b/284106340): Return this to a dictionary.
    kernel_args = (
        (tf.random.uniform([], 1.5, 2.5, dtype=tf.float64),  # amplitude
         tf.random.uniform([], .5, 1.5, dtype=tf.float64)),  # length_scale
        (tf.random.uniform([feature_dim], .5, 1.5,  # bias_amplitude
                           dtype=tf.float64),
         tf.random.normal([feature_dim], dtype=tf.float64)))  # shift

    x1 = tf.random.normal([5, feature_dim], dtype=tf.float64)
    x2 = tf.random.normal([7, feature_dim], dtype=tf.float64)
    linop = linear_operator_psd_kernel.LinearOperatorPSDKernel(
        kernel_fn, x1, x2, kernel_args=kernel_args, num_matmul_parts=3)
    x = tf.random.normal([7, 2], dtype=tf.float64)

    @tf.function(jit_compile=True)
    def f():
      with tf.GradientTape() as tape:
        tape.watch((x1, x2, x, kernel_args))
        y = linop.matmul(x)

      out_grad = tf.random.normal(tf.shape(y), dtype=tf.float64)
      actuals = tape.gradient(y, (x1, x2, x, kernel_args),
                              output_gradients=out_grad)
      return y, actuals, out_grad

    y, actuals, out_grad = f()

    with tf.GradientTape() as tape:
      tape.watch((x1, x2, x, kernel_args))
      y = tf.matmul(kernel_fn(*kernel_args).matrix(x1, x2), x)
    expecteds = tape.gradient(y, (x1, x2, x, kernel_args),
                              output_gradients=out_grad)

    expecteds, actuals = self.evaluate([expecteds, actuals])

    tf.nest.assert_same_structure(expecteds, actuals)
    for expected, actual in zip(tf.nest.flatten(expecteds),
                                tf.nest.flatten(actuals)):
      self.assertAllClose(expected, actual)


if __name__ == '__main__':
  test_util.main()
