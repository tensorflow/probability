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
"""Tests for Constant Kernel."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized

import numpy as np
import tensorflow.compat.v2 as tf

import tensorflow_probability as tfp

from tensorflow_probability.python.internal import test_util

@test_util.test_all_tf_execution_regimes
class ConstantTest(test_util.TestCase):

  @parameterized.parameters(
      {'feature_ndims': 1, 'dims': 3},
      {'feature_ndims': 1, 'dims': 4},
      {'feature_ndims': 2, 'dims': 2},
      {'feature_ndims': 2, 'dims': 3},
      {'feature_ndims': 3, 'dims': 2},
      {'feature_ndims': 3, 'dims': 3})
  def testValuesAreCorrect(self, feature_ndims, dims):
    coef = 3.

    np.random.seed(42)
    k = tfp.math.psd_kernels.Constant(coef, feature_ndims)
    shape = [dims] * feature_ndims
    for _ in range(5):
      x = np.random.uniform(-1, 1, size=shape).astype(np.float32)
      y = np.random.uniform(-1, 1, size=shape).astype(np.float32)
      self.assertAllClose(
          np.ones(x.shape[:-feature_ndims]),
          self.evaluate(k.apply(x, y)))

  def testShapesAreCorrect(self):
    k = tfp.math.psd_kernels.Constant(coef=1.0)

    x = np.ones([4, 3], np.float32)
    y = np.ones([5, 3], np.float32)

    self.assertAllEqual([4, 5], k.matrix(x, y).shape)
    self.assertAllEqual(
        [2, 4, 5],
        k.matrix(tf.stack([x]*2), tf.stack([y]*2)).shape)

    k = tfp.math.psd_kernels.Constant(coef=np.ones([2, 1, 1], np.float32))
    self.assertAllEqual(
        [2, 1, 2, 4, 5],
        #`--'  |  `--'
        #  |   |    `- matrix shape
        #  |   `- from input batch shapes
        #  `- from broadcasting kernel params
        k.matrix(
            tf.stack([x]*2),  # shape [2, 4, 3]
            tf.stack([y]*2)   # shape [2, 5, 3]
        ).shape)

  def testValidateArgs(self):
    with self.assertRaisesOpError('must be positive'):
      k = tfp.math.psd_kernels.Constant(-1., validate_args=True)
      self.evaluate(k.apply([1.], [1.]))

    # But `None`'s are ok
    k = tfp.math.psd_kernels.Constant(
        None, validate_args=True)
    self.evaluate(k.apply([1.], [1.]))

  @test_util.jax_disable_variable_test
  def testValidateVariableArgs(self):
    coef = tf.Variable(1.)
    k = tfp.math.psd_kernels.Constant(coef, validate_args=True)
    self.evaluate([v.initializer for v in k.variables])

    with self.assertRaisesOpError('must be positive'):
      with tf.control_dependencies([coef.assign(-1.)]):
        self.evaluate(k.apply([1.], [1.]))
