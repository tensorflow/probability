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
"""Tests for tensorshape_util."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports

import numpy as np
import tensorflow.compat.v2 as tf

from tensorflow_probability.python.internal import tensorshape_util
from tensorflow_probability.python.internal import test_util


@test_util.test_all_tf_execution_regimes
class TensorShapeUtilTestTest(test_util.TestCase):

  def test_with_rank_list_tuple(self):
    with self.assertRaises(ValueError):
      tensorshape_util.with_rank([2], 2)

    with self.assertRaises(ValueError):
      tensorshape_util.with_rank((2,), 2)

    self.assertAllEqual(
        (2, 1),
        tensorshape_util.with_rank((2, 1), 2))
    self.assertAllEqual(
        [2, 1],
        tensorshape_util.with_rank([2, 1], 2))

    self.assertAllEqual(
        (2, 3, 4),
        tensorshape_util.with_rank_at_least((2, 3, 4), 2))
    self.assertAllEqual(
        [2, 3, 4],
        tensorshape_util.with_rank_at_least([2, 3, 4], 2))

  def test_with_rank_ndarray(self):
    x = np.array([2], dtype=np.int32)
    with self.assertRaises(ValueError):
      tensorshape_util.with_rank(x, 2)

    x = np.array([2, 3, 4], dtype=np.int32)
    y = tensorshape_util.with_rank(x, 3)
    self.assertAllEqual(x, y)

    x = np.array([2, 3, 4, 5], dtype=np.int32)
    y = tensorshape_util.with_rank_at_least(x, 3)
    self.assertAllEqual(x, y)


if __name__ == '__main__':
  tf.test.main()
