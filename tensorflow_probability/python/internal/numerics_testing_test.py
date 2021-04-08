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
"""Testing the numerics testing utilities."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
import tensorflow.compat.v2 as tf
from tensorflow_probability.python.internal import numerics_testing as nt
from tensorflow_probability.python.internal import test_util


@test_util.numpy_disable_gradient_test
class NumericsTestingTest(test_util.TestCase):

  def testCatastrophicCancellation(self):
    # Check that we can detect catastrophic cancellations

    def bad_identity(x):
      return (1. + x) - 1.

    # The relative error in this implementation of the identity
    # function is terrible.
    small = tf.constant(1e-6, dtype=tf.float32)
    self.assertGreater(self.evaluate(nt.relative_error_at(
        bad_identity, small)), 0.04)

    # But the function itself is well-conditioned
    self.assertAllEqual(self.evaluate(nt.inputwise_condition_numbers(
        bad_identity, small)), [1.])

    # So the error due to poor conditioning is small
    self.assertLess(self.evaluate(nt.error_due_to_ill_conditioning(
        bad_identity, small)), 1e-13)

    # And we have a lot of excess wrong bits
    self.assertGreater(self.evaluate(nt.excess_wrong_bits(
        bad_identity, small)), 19)

  def testLog1p(self):
    # Check that we can tell that log1p is a good idea

    def bad_log1p(x):
      # Extra indirection to fool Grappler in TF1
      return tf.math.log(((1. + x) - 1.) + 1.)

    small = tf.constant(1e-7, dtype=tf.float32)
    self.assertGreater(self.evaluate(nt.excess_wrong_bits(
        bad_log1p, small)), 21)

    self.assertLessEqual(self.evaluate(nt.excess_wrong_bits(
        tf.math.log1p, small)), 0)

if __name__ == "__main__":
  tf.test.main()
