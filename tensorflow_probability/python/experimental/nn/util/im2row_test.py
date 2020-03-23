# Copyright 2020 The TensorFlow Probability Authors.
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
"""Tests for im2col."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp

from tensorflow_probability.python.internal import test_util

tfn = tfp.experimental.nn


@test_util.test_all_tf_execution_regimes
class Im2ColTest(test_util.TestCase):

  def test_works_like_conv2d(self):
    x = tf.constant([[
        [[2], [1], [2], [0], [1]],
        [[1], [3], [2], [2], [3]],
        [[1], [1], [3], [3], [0]],
        [[2], [2], [0], [1], [1]],
        [[0], [0], [3], [1], [2]],
    ]], tf.float32)  # shape=[1, 5, 5, 1]
    x = tf.concat([x, x], axis=-1)
    k = tf.constant([
        [[[2, 0.1]], [[3, 0.2]]],
        [[[0, 0.3]], [[1, 0.4]]],
    ], tf.float32)  # shape=[2, 2, 1, 2]
    k = tf.concat([k, k], axis=-2)
    strides = [1, 2]
    im2row_x = tfn.util.im2row(
        x,
        block_shape=k.shape[:2],
        slice_step=strides,
        padding='VALID')
    y_expected = tf.nn.conv2d(x, k, strides=strides, padding='VALID')
    y_actual = tf.matmul(im2row_x, tf.reshape(k, [-1, k.shape[-1]]))
    [y_expected_, y_actual_] = self.evaluate([y_expected, y_actual])
    self.assertAllClose(y_expected_, y_actual_, rtol=1e-5, atol=0)


if __name__ == '__main__':
  tf.test.main()
