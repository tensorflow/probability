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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


# Dependency imports

import numpy as np
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp

from tensorflow_probability.python.internal import test_util as tfp_test_util

from tensorflow.python.framework import test_util  # pylint: disable=g-direct-tensorflow-import

tfb = tfp.bijectors
tfd = tfp.distributions


@test_util.run_all_in_graph_and_eager_modes
class LinearOperatorPSDKernelTest(tfp_test_util.TestCase):
  """Tests for tfp.experimental.linalg.LinearOperatorPSDKernel."""

  def test_shape(self):
    kernel = tfp.math.psd_kernels.ExponentiatedQuadratic(
        amplitude=tf.random.uniform([17, 1, 1]),
        feature_ndims=2)
    x1 = tf.random.normal([1, 11, 5, 2, 13])
    x2 = tf.random.normal([7, 1, 3, 2, 13])
    linop = tfp.experimental.linalg.LinearOperatorPSDKernel(kernel, x1, x2)
    self.assertAllEqual((17, 7, 11, 5, 3), linop.shape)
    self.assertAllEqual((17, 7, 11), linop.batch_shape)

  def test_diag_part(self):
    kernel = tfp.math.psd_kernels.ExponentiatedQuadratic()
    x1 = tf.random.normal([7, 3, 5, 2])  # square matrix 5x5
    linop = tfp.experimental.linalg.LinearOperatorPSDKernel(kernel, x1)
    expected, actual = self.evaluate([
        tf.linalg.diag_part(kernel.matrix(x1, x1)),
        linop.diag_part()
    ])
    self.assertAllClose(expected, actual)

    x2 = tf.random.normal([3, 11, 2])  # wide matrix 5x11
    linop = tfp.experimental.linalg.LinearOperatorPSDKernel(kernel, x1, x2)
    expected, actual = self.evaluate([
        tf.linalg.diag_part(kernel.matrix(x1, x2)),
        linop.diag_part()
    ])
    self.assertAllClose(expected, actual)

    x2 = tf.random.normal([2, 2])  # tall matrix 5x2
    linop = tfp.experimental.linalg.LinearOperatorPSDKernel(kernel, x1, x2)
    expected, actual = self.evaluate([
        tf.linalg.diag_part(kernel.matrix(x1, x2)),
        linop.diag_part()
    ])
    self.assertAllClose(expected, actual)

  def test_diag_part_xla(self):
    try:
      tf.function(lambda: tf.constant(0), experimental_compile=True)()
    except tf.errors.UnimplementedError as e:
      if 'Could not find compiler' in str(e):
        self.skipTest('XLA not available')

    kernel = tfp.math.psd_kernels.ExponentiatedQuadratic()
    x1 = tf.random.normal([7, 3, 5, 2])  # square matrix 5x5
    linop = tfp.experimental.linalg.LinearOperatorPSDKernel(kernel, x1)
    expected, actual = self.evaluate([
        tf.linalg.diag_part(kernel.matrix(x1, x1)),
        tf.function(linop.diag_part, experimental_compile=True)()
    ])
    self.assertAllClose(expected, actual)

    x2 = tf.random.normal([3, 11, 2])  # wide matrix 5x11
    linop = tfp.experimental.linalg.LinearOperatorPSDKernel(kernel, x1, x2)
    expected, actual = self.evaluate([
        tf.linalg.diag_part(kernel.matrix(x1, x2)),
        tf.function(linop.diag_part, experimental_compile=True)()
    ])
    self.assertAllClose(expected, actual)

    x2 = tf.random.normal([2, 2])  # tall matrix 5x2
    linop = tfp.experimental.linalg.LinearOperatorPSDKernel(kernel, x1, x2)
    expected, actual = self.evaluate([
        tf.linalg.diag_part(kernel.matrix(x1, x2)),
        tf.function(linop.diag_part, experimental_compile=True)()
    ])
    self.assertAllClose(expected, actual)

  def test_row_scalar(self):
    kernel = tfp.math.psd_kernels.ExponentiatedQuadratic()
    x1 = tf.random.normal([5, 2])
    x2 = tf.random.normal([7, 3, 5, 2])
    linop = tfp.experimental.linalg.LinearOperatorPSDKernel(kernel, x1, x2)
    i = np.random.randint(0, 5)
    expected, actual = self.evaluate(
        [kernel.matrix(x1, x2)[..., i, :], linop.row(i)])
    self.assertAllClose(expected, actual)

  def test_row_batch(self):
    kernel = tfp.math.psd_kernels.ExponentiatedQuadratic()
    x1 = tf.random.normal([7, 1, 5, 2])
    x2 = tf.random.normal([1, 3, 4, 2])
    linop = tfp.experimental.linalg.LinearOperatorPSDKernel(kernel, x1, x2)
    i = np.random.randint(0, 5, size=(7, 3))
    cov = kernel.matrix(x1, x2)
    expected, actual = self.evaluate([
        tf.gather(cov, i[..., tf.newaxis], batch_dims=2)[..., 0, :],
        linop.row(i)
    ])
    self.assertAllClose(expected, actual)

  def test_col_scalar(self):
    kernel = tfp.math.psd_kernels.ExponentiatedQuadratic()
    x1 = tf.random.normal([5, 2])
    x2 = tf.random.normal([7, 3, 5, 2])
    linop = tfp.experimental.linalg.LinearOperatorPSDKernel(kernel, x1, x2)
    j = np.random.randint(0, 5)
    expected, actual = self.evaluate(
        [kernel.matrix(x1, x2)[..., j], linop.col(j)])
    self.assertAllClose(expected, actual)

  def test_col_batch(self):
    kernel = tfp.math.psd_kernels.ExponentiatedQuadratic()
    x1 = tf.random.normal([3, 5, 2])
    x2 = tf.random.normal([7, 1, 4, 2])
    linop = tfp.experimental.linalg.LinearOperatorPSDKernel(kernel, x1, x2)
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


if __name__ == '__main__':
  tf.test.main()
