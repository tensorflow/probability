# Copyright 2018 The TensorFlow Probability Authors.
#
# Licensed under the Apache License, Version 2.0 (the 'License');
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
"""Tests for the SeedStream class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as onp  # Avoid JAX rewrite.

import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp

from tensorflow_probability.python.internal import test_util


def _ensure_tuple(value):
  """JAX/non-JAX compatibility.

  Helps with compatibilty between the scalar `int` seeds from vanilla
  `SeedStream` and the `(2,)`-shaped `DeviceArray` from JAX `SeedStream`.

  Args:
    value: `int` or `DeviceArray` to be converted.

  Returns:
    `tuple` of one or two scalars.
  """
  return tuple(onp.reshape(value, [-1]))


@test_util.test_all_tf_execution_regimes
class SeedStreamTest(test_util.TestCase):

  def assertAllUnique(self, items):
    self.assertEqual(len(items), len(set(items)))

  def testNonRepetition(self):
    # The probability of repetitions in a short stream from a correct
    # PRNG is negligible; this test catches bugs that prevent state
    # updates.
    strm = tfp.util.SeedStream(seed=4, salt='salt')
    output = [_ensure_tuple(strm()) for _ in range(50)]
    self.assertEqual(sorted(output), sorted(list(set(output))))

  def testReproducibility(self):
    strm1 = tfp.util.SeedStream(seed=4, salt='salt')
    strm2 = tfp.util.SeedStream(seed=4, salt='salt')
    strm3 = tfp.util.SeedStream(seed=4, salt='salt')
    outputs = [_ensure_tuple(strm1()) for _ in range(50)]
    self.assertEqual(outputs, [_ensure_tuple(strm2()) for _ in range(50)])
    self.assertEqual(outputs, [_ensure_tuple(strm3()) for _ in range(50)])

  def testSeededDistinctness(self):
    strm1 = tfp.util.SeedStream(seed=4, salt='salt')
    strm2 = tfp.util.SeedStream(seed=5, salt='salt')
    self.assertAllUnique(
        [_ensure_tuple(strm1()) for _ in range(50)] +
        [_ensure_tuple(strm2()) for _ in range(50)])

  def testSaltedDistinctness(self):
    strm1 = tfp.util.SeedStream(seed=4, salt='salt')
    strm2 = tfp.util.SeedStream(seed=4, salt='another salt')
    self.assertAllUnique(
        [_ensure_tuple(strm1()) for _ in range(50)] +
        [_ensure_tuple(strm2()) for _ in range(50)])

  def testNestingRobustness(self):
    # SeedStreams started from generated seeds should not collide with
    # the master or with each other, even if the salts are the same.
    strm1 = tfp.util.SeedStream(seed=4, salt='salt')
    strm2 = tfp.util.SeedStream(strm1(), salt='salt')
    strm3 = tfp.util.SeedStream(strm1(), salt='salt')
    outputs = [_ensure_tuple(strm1()) for _ in range(50)]
    self.assertAllUnique(
        outputs + [_ensure_tuple(strm2()) for _ in range(50)] +
        [_ensure_tuple(strm3()) for _ in range(50)])

  def testInitFromOtherSeedStream(self):
    strm1 = tfp.util.SeedStream(seed=4, salt='salt')
    strm2 = tfp.util.SeedStream(strm1, salt='salt')
    strm3 = tfp.util.SeedStream(strm1, salt='another salt')
    out1 = [_ensure_tuple(strm1()) for _ in range(50)]
    out2 = [_ensure_tuple(strm2()) for _ in range(50)]
    out3 = [_ensure_tuple(strm3()) for _ in range(50)]
    self.assertAllEqual(out1, out2)
    self.assertAllUnique(out1 + out3)


if __name__ == '__main__':
  tf.test.main()
