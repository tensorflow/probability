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
"""Tests for StructuralTimeSeries utilities."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow_probability.python.sts.internal import util as sts_util

from tensorflow.python.framework import test_util
from tensorflow.python.platform import test


class UtilityTests(test.TestCase):

  @test_util.run_in_graph_and_eager_modes
  def test_maybe_expand_trailing_dim(self):

    # static inputs
    self.assertEqual(
        sts_util.maybe_expand_trailing_dim(tf.zeros([4, 3])).shape,
        tf.TensorShape([4, 3, 1]))
    self.assertEqual(
        sts_util.maybe_expand_trailing_dim(tf.zeros([4, 3, 1])).shape,
        tf.TensorShape([4, 3, 1]))

    # dynamic inputs
    for shape_in, static_shape, expected_shape_out in [
        # pyformat: disable
        ([4, 3], None, [4, 3, 1]),
        ([4, 3, 1], None, [4, 3, 1]),
        ([4], [None], [4, 1]),
        ([1], [None], [1]),
        ([4, 3], [None, None], [4, 3, 1]),
        ([4, 1], [None, None], [4, 1]),
        ([4, 1], [None, 1], [4, 1])
        # pyformat: enable
    ]:
      shape_out = self.evaluate(
          sts_util.maybe_expand_trailing_dim(
              tf.placeholder_with_default(
                  input=tf.zeros(shape_in), shape=static_shape))).shape
      self.assertAllEqual(shape_out, expected_shape_out)

if __name__ == "__main__":
  test.main()
