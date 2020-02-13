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
"""Tests generating test combinations."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import OrderedDict

# Dependency imports
import tensorflow.compat.v2 as tf
from tensorflow_probability.python.internal import test_combinations
from tensorflow_probability.python.internal import test_util


class TestingCombinationsTest(test_util.TestCase):

  def test_combine(self):
    self.assertEqual([{
        "a": 1,
        "b": 2
    }, {
        "a": 1,
        "b": 3
    }, {
        "a": 2,
        "b": 2
    }, {
        "a": 2,
        "b": 3
    }], test_combinations.combine(a=[1, 2], b=[2, 3]))

  def test_arguments_sorted(self):
    self.assertEqual([
        OrderedDict([("aa", 1), ("ab", 2)]),
        OrderedDict([("aa", 1), ("ab", 3)]),
        OrderedDict([("aa", 2), ("ab", 2)]),
        OrderedDict([("aa", 2), ("ab", 3)])
    ], test_combinations.combine(ab=[2, 3], aa=[1, 2]))

  def test_combine_single_parameter(self):
    self.assertEqual([{
        "a": 1,
        "b": 2
    }, {
        "a": 2,
        "b": 2
    }], test_combinations.combine(a=[1, 2], b=2))

  def test_add(self):
    self.assertEqual(
        [{
            "a": 1
        }, {
            "a": 2
        }, {
            "b": 2
        }, {
            "b": 3
        }],
        (test_combinations.combine(a=[1, 2]) +
         test_combinations.combine(b=[2, 3])))


@test_combinations.generate(
    test_combinations.combine(a=[1, 0], b=[2, 3], c=[1]))
class CombineTheTestSuite(test_util.TestCase):

  def test_add_things(self, a, b, c):
    self.assertLessEqual(3, a + b + c)
    self.assertLessEqual(a + b + c, 5)

  def test_add_things_one_more(self, a, b, c):
    self.assertLessEqual(3, a + b + c)
    self.assertLessEqual(a + b + c, 5)

  def not_a_test(self, a=0, b=0, c=0):
    del a, b, c
    self.fail()

  def _test_but_private(self, a=0, b=0, c=0):
    del a, b, c
    self.fail()

  # Check that nothing funny happens to a non-callable that starts with "_test".
  test_member = 0


if __name__ == "__main__":
  tf.test.main()
