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
"""Shift Tests."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports

from absl.testing import parameterized
import tensorflow.compat.v2 as tf
from tensorflow_probability.python import bijectors as tfb
from tensorflow_probability.python.internal import test_util


@test_util.test_all_tf_execution_regimes
class ShiftTest(test_util.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      dict(testcase_name='static', is_static=True),
      dict(testcase_name='dynamic', is_static=False),
  )
  def testNoBatch(self, is_static):
    shift = tfb.Shift([1., -1.])
    x = self.maybe_static([1., 1.], is_static)
    self.assertAllClose([2., 0.], shift.forward(x))
    self.assertAllClose([0., 2.], shift.inverse(x))
    self.assertAllClose(0.,
                        shift.inverse_log_det_jacobian(x, event_ndims=1))

  @parameterized.named_parameters(
      dict(testcase_name='static', is_static=True),
      dict(testcase_name='dynamic', is_static=False),
  )
  def testBatch(self, is_static):
    shift = tfb.Shift([[2., -.5], [1., -3.]])
    x = self.maybe_static([1., 1.], is_static)

    self.assertAllClose([[3., .5], [2., -2.]], shift.forward(x))
    self.assertAllClose([[-1., 1.5], [0., 4.]], shift.inverse(x))
    self.assertAllClose(0., shift.inverse_log_det_jacobian(x, event_ndims=1))


if __name__ == '__main__':
  tf.test.main()
