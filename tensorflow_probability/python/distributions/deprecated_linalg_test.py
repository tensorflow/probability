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
"""Tests for deprecated_linalg functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v2 as tf

from tensorflow_probability.python.distributions.deprecated_linalg import tridiag
from tensorflow_probability.python.internal import test_case
from tensorflow.python.framework import test_util  # pylint: disable=g-direct-tensorflow-import


@test_util.run_all_in_graph_and_eager_modes
class TridiagTest(test_case.TestCase):

  def testWorksCorrectlyNoBatches(self):
    self.assertAllEqual(
        [[4., 8., 0., 0.],
         [1., 5., 9., 0.],
         [0., 2., 6., 10.],
         [0., 0., 3, 7.]],
        self.evaluate(tridiag(
            [1., 2., 3.],
            [4., 5., 6., 7.],
            [8., 9., 10.])))

  def testWorksCorrectlyBatches(self):
    self.assertAllClose(
        [[[4., 8., 0., 0.],
          [1., 5., 9., 0.],
          [0., 2., 6., 10.],
          [0., 0., 3, 7.]],
         [[0.7, 0.1, 0.0, 0.0],
          [0.8, 0.6, 0.2, 0.0],
          [0.0, 0.9, 0.5, 0.3],
          [0.0, 0.0, 1.0, 0.4]]],
        self.evaluate(tridiag(
            [[1., 2., 3.],
             [0.8, 0.9, 1.]],
            [[4., 5., 6., 7.],
             [0.7, 0.6, 0.5, 0.4]],
            [[8., 9., 10.],
             [0.1, 0.2, 0.3]])),
        rtol=1e-5, atol=0.)

  def testHandlesNone(self):
    self.assertAllClose(
        [[[4., 0., 0., 0.],
          [0., 5., 0., 0.],
          [0., 0., 6., 0.],
          [0., 0., 0, 7.]],
         [[0.7, 0.0, 0.0, 0.0],
          [0.0, 0.6, 0.0, 0.0],
          [0.0, 0.0, 0.5, 0.0],
          [0.0, 0.0, 0.0, 0.4]]],
        self.evaluate(tridiag(
            diag=[[4., 5., 6., 7.],
                  [0.7, 0.6, 0.5, 0.4]])),
        rtol=1e-5, atol=0.)


if __name__ == '__main__':
  tf.test.main()
