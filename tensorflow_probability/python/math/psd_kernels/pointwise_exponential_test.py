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
"""Tests for exponential."""

import numpy as np
import tensorflow.compat.v2 as tf
from tensorflow_probability.python.internal import test_util
from tensorflow_probability.python.math.psd_kernels import matern
from tensorflow_probability.python.math.psd_kernels import parabolic
from tensorflow_probability.python.math.psd_kernels import pointwise_exponential


@test_util.test_all_tf_execution_regimes
class PointwiseExponentialTest(test_util.TestCase):

  def testValuesAreCorrect(self):
    original_kernel = parabolic.Parabolic()
    exponential_kernel = pointwise_exponential.PointwiseExponential(
        original_kernel)
    x1 = [[1.0]]
    x2 = [[2.0]]
    original_output = original_kernel.apply(x1, x2)
    exponential_output = exponential_kernel.apply(x1, x2)

    self.assertAllEqual(
        self.evaluate(tf.math.exp(original_output)),
        self.evaluate(exponential_output))

  def testBatchShape(self):
    amplitude = np.random.uniform(2, 3., size=[3, 1, 2]).astype(np.float32)
    length_scale = np.random.uniform(2, 3., size=[1, 3, 1]).astype(np.float32)
    original_kernel = matern.GeneralizedMatern(
        df=np.pi, amplitude=amplitude, length_scale=length_scale)
    exponential_kernel = pointwise_exponential.PointwiseExponential(
        original_kernel)
    self.assertAllEqual(original_kernel.batch_shape,
                        exponential_kernel.batch_shape)
    self.assertAllEqual(
        self.evaluate(original_kernel.batch_shape_tensor()),
        self.evaluate(exponential_kernel.batch_shape_tensor()))


if __name__ == '__main__':
  test_util.main()
