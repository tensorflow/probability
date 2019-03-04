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
"""Tests for dtype_util."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from tensorflow_probability.python.internal import dtype_util

ed = tfp.edward2


class DtypeUtilTest(tf.test.TestCase):

  def testNoModifyArgsList(self):
    x = tf.ones(3, tf.float32)
    y = tf.zeros(4, tf.float32)
    lst = [x, y]
    self.assertEqual(tf.float32, dtype_util.common_dtype(lst))
    self.assertLen(lst, 2)

  def testCommonDtypeFromLinop(self):
    x = tf.linalg.LinearOperatorDiag(tf.ones(3, tf.float16))
    self.assertEqual(
        tf.float16, dtype_util.common_dtype([x], preferred_dtype=tf.float32))

  def testCommonDtypeFromEdRV(self):
    # As in tensorflow_probability github issue #221
    x = ed.Dirichlet(np.ones(3, dtype='float64'))
    self.assertEqual(
        tf.float64, dtype_util.common_dtype([x], preferred_dtype=tf.float32))


if __name__ == '__main__':
  tf.test.main()
