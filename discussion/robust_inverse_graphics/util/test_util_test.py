# Copyright 2024 The TensorFlow Probability Authors.
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
"""Tests for test_util."""

from flax import struct
import numpy as np
from discussion.robust_inverse_graphics.util import test_util


@struct.dataclass
class TestStruct:
  a: int
  b: int


class TestUtilTest(test_util.TestCase):

  def testAssertAllEqual(self):
    self.assertAllEqual(np.arange(3), np.arange(3))
    with self.assertRaisesRegex(AssertionError, 'message'):
      self.assertAllEqual(np.arange(3), np.arange(4), msg='message')

  def testAssertAllClose(self):
    self.assertAllClose(np.arange(3), np.arange(3))
    self.assertAllClose({'a': np.arange(3)}, {'a': np.arange(3)})
    self.assertAllClose(np.linspace(0., 1.), np.linspace(0., 1.))
    self.assertAllClose(
        np.linspace(0., 1.), np.linspace(0., 1.) + 0.1, atol=0.11)
    self.assertAllClose(
        np.linspace(1., 2.), 1 * 1.1 * np.linspace(1., 2.), rtol=0.11)
    with self.assertRaisesRegex(AssertionError, 'not close'):
      self.assertAllClose(np.arange(3), np.arange(4), msg='message')
    self.assertAllClose(TestStruct(1, 2), TestStruct(1, 2))
    with self.assertRaisesRegex(AssertionError, 'not close'):
      self.assertAllClose(TestStruct(1, 2), TestStruct(1, 3))


if __name__ == '__main__':
  test_util.main()
