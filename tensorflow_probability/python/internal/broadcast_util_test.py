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
"""Tests for broadcast_util."""

import tensorflow.compat.v2 as tf
from tensorflow_probability.python.internal import broadcast_util as bu
from tensorflow_probability.python.internal import test_util


class BroadcastUtilTest(test_util.TestCase):

  @test_util.numpy_disable_test_missing_functionality('tf.unsorted_segment_sum')
  @test_util.jax_disable_test_missing_functionality('tf.unsorted_segment_sum')
  def test_right_justified_unsorted_segment_sum(self):
    # If the segment indices are range(num_segments), segment sum
    # should be the identity.  Here we check dimension alignment:
    # data.shape == [3, 2]; segment_ids.shape == [2]
    data = [[1, 2], [4, 8], [16, 32]]
    segment_ids = [0, 1]
    expected = [[1, 2], [4, 8], [16, 32]]
    self.assertAllEqual(
        expected,
        bu.right_justified_unsorted_segment_sum(
            data, segment_ids, num_segments=2))

    # Check the same but with nontrivial summation, and with a
    # too-large max_segments.  The latter determines the innermost
    # dimension.
    data = [[1, 2], [4, 8], [16, 32]]
    segment_ids = [1, 1]
    expected = [[0, 3, 0, 0], [0, 12, 0, 0], [0, 48, 0, 0]]
    self.assertAllEqual(
        expected,
        bu.right_justified_unsorted_segment_sum(
            data, segment_ids, num_segments=4))

    # If the segment_ids have the same shape as the data,
    # we expect a vector of size num_segments as output
    data = [[1, 2], [4, 8], [16, 32]]
    segment_ids = [[1, 1], [0, 1], [0, 0]]
    expected = [52, 11, 0, 0]
    self.assertAllEqual(
        expected,
        bu.right_justified_unsorted_segment_sum(
            data, segment_ids, num_segments=4))

    # Same as the previous example but with a batch dimension for the data
    data = [[[1, 2], [4, 8], [16, 32]],
            [[64, 128], [256, 512], [1024, 2048]]]
    segment_ids = [[1, 1], [0, 1], [0, 0]]
    expected = [[52, 11, 0, 0], [64 * 52, 64 * 11, 0, 0]]
    self.assertAllEqual(
        expected,
        bu.right_justified_unsorted_segment_sum(
            data, segment_ids, num_segments=4))


if __name__ == '__main__':
  tf.test.main()
