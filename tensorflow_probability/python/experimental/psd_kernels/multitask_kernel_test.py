# Copyright 2021 The TensorFlow Probability Authors.
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
"""Tests for MultiTask kernel."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow.compat.v2 as tf

from tensorflow_probability.python.experimental.psd_kernels import multitask_kernel
from tensorflow_probability.python.internal import test_util
from tensorflow_probability.python.math.psd_kernels import exponentiated_quadratic


@test_util.test_all_tf_execution_regimes
class _MultiTaskKernelTest(object):

  def testIndependentShape(self):
    amplitude = np.random.uniform(2, 3., size=[3, 1, 2]).astype(self.dtype)
    length_scale = np.random.uniform(2, 3., size=[1, 3, 1]).astype(self.dtype)
    base_kernel = exponentiated_quadratic.ExponentiatedQuadratic(
        amplitude, length_scale)
    kernel = multitask_kernel.Independent(num_tasks=5, base_kernel=base_kernel)
    self.assertAllEqual([3, 3, 2], self.evaluate(kernel.batch_shape_tensor()))
    matrix_over_all_tasks = kernel.matrix_over_all_tasks(
        np.ones([4, 1, 1, 1, 3, 2], dtype=self.dtype),
        np.ones([5, 1, 1, 1, 1, 4, 2], dtype=self.dtype))
    self.assertAllEqual([5, 4, 3, 3, 2, 15, 20], self.evaluate(
        matrix_over_all_tasks.shape_tensor()))

  def testSeparableShape(self):
    amplitude = np.random.uniform(2, 3., size=[3, 1, 2]).astype(self.dtype)
    length_scale = np.random.uniform(2, 3., size=[1, 3, 1]).astype(self.dtype)
    base_kernel = exponentiated_quadratic.ExponentiatedQuadratic(
        amplitude, length_scale)
    task_kernel_matrix_linop = tf.linalg.LinearOperatorIdentity(
        5, batch_shape=[4, 1, 1, 1], dtype=self.dtype)

    kernel = multitask_kernel.Separable(
        num_tasks=5,
        base_kernel=base_kernel,
        task_kernel_matrix_linop=task_kernel_matrix_linop)
    self.assertAllEqual(
        [4, 3, 3, 2], self.evaluate(kernel.batch_shape_tensor()))
    matrix_over_all_tasks = kernel.matrix_over_all_tasks(
        np.ones([4, 1, 1, 1, 3, 2], dtype=self.dtype),
        np.ones([5, 1, 1, 1, 1, 4, 2], dtype=self.dtype))
    self.assertAllEqual([5, 4, 3, 3, 2, 15, 20], self.evaluate(
        matrix_over_all_tasks.shape_tensor()))


class MultiTaskKernelTestFloat32(test_util.TestCase, _MultiTaskKernelTest):
  dtype = np.float32


class MultiTaskKernelTestFloat64(test_util.TestCase, _MultiTaskKernelTest):
  dtype = np.float64


if __name__ == '__main__':
  test_util.main()
