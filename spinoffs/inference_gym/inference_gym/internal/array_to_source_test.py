# Lint as: python3
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
#
"""Tests for array_to_source."""

from tensorflow_probability.python.internal import test_util
from inference_gym.internal import array_to_source


class ArrayToSourceTest(test_util.TestCase):

  def testEncodeArray(self):
    array_str = array_to_source.array_to_source('array', [1, 2, 3])
    expected_str = """array = np.array([
    1,
    2,
    3,
]).reshape((3,))
"""
    self.assertEqual(expected_str, array_str)

  def testEncodeScalar(self):
    array_str = array_to_source.array_to_source('array', 4)
    expected_str = """array = np.array([
    4,
]).reshape(())
"""
    self.assertEqual(expected_str, array_str)


if __name__ == '__main__':
  test_util.main()
