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

import tensorflow_probability as tfp

from tensorflow_probability.python.experimental import psd_kernels as tfpk
from tensorflow_probability.python.internal import test_util


@test_util.test_all_tf_execution_regimes
class MultiTaskKernelTest(test_util.TestCase):

  def testIndependentShape(self):
    amplitude = np.random.uniform(2, 3., size=[3, 1, 2]).astype(np.float32)
    length_scale = np.random.uniform(2, 3., size=[1, 3, 1]).astype(np.float32)
    base_kernel = tfp.math.psd_kernels.ExponentiatedQuadratic(
        amplitude, length_scale)
    kernel = tfpk.Independent(num_tasks=5, base_kernel=base_kernel)
    self.assertAllEqual([3, 3, 2], self.evaluate(kernel.batch_shape_tensor()))
    matrix_over_all_tasks = kernel.matrix_over_all_tasks(
        tf.ones([4, 1, 1, 1, 3, 2]), tf.ones([5, 1, 1, 1, 1, 4, 2]))
    self.assertAllEqual([5, 4, 3, 3, 2, 15, 20], self.evaluate(
        matrix_over_all_tasks.shape_tensor()))

  def testSeparableShape(self):
    amplitude = np.random.uniform(2, 3., size=[3, 1, 2]).astype(np.float32)
    length_scale = np.random.uniform(2, 3., size=[1, 3, 1]).astype(np.float32)
    base_kernel = tfp.math.psd_kernels.ExponentiatedQuadratic(
        amplitude, length_scale)
    task_kernel_matrix_linop = tf.linalg.LinearOperatorIdentity(
        5, batch_shape=[4, 1, 1, 1])

    kernel = tfpk.Separable(
        num_tasks=5,
        base_kernel=base_kernel,
        task_kernel_matrix_linop=task_kernel_matrix_linop)
    self.assertAllEqual(
        [4, 3, 3, 2], self.evaluate(kernel.batch_shape_tensor()))
    matrix_over_all_tasks = kernel.matrix_over_all_tasks(
        tf.ones([4, 1, 1, 1, 3, 2]), tf.ones([5, 1, 1, 1, 1, 4, 2]))
    self.assertAllEqual([5, 4, 3, 3, 2, 15, 20], self.evaluate(
        matrix_over_all_tasks.shape_tensor()))


if __name__ == '__main__':
  tf.test.main()
