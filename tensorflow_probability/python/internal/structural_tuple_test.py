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
"""Tests for structural_tuple."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports

import tensorflow.compat.v2 as tf

from tensorflow_probability.python.internal import nest_util
from tensorflow_probability.python.internal import structural_tuple
from tensorflow_probability.python.internal import test_util


class StructTupleTest(test_util.TestCase):

  def testCacheWorks(self):
    t1 = structural_tuple.structtuple(['a', 'b', 'c'])
    t2 = structural_tuple.structtuple(['a', 'b', 'c'])
    t3 = structural_tuple.structtuple(['a', 'b'])
    self.assertIs(t1, t2)
    self.assertIsNot(t1, t3)

  def testCacheWithUnderscores(self):
    t1 = structural_tuple.structtuple(['a_b', 'b'])
    t2 = structural_tuple.structtuple(['a', 'b_c'])
    self.assertIsNot(t1, t2)

  def testValidNamedTuple(self):
    t = structural_tuple.structtuple(['a', 'b', 'c'])
    inst = t(a=1, b=2, c=3)
    a, b, c = inst
    self.assertAllEqualNested((1, 2, 3), (a, b, c))
    self.assertAllEqualNested(
        t(2, 3, 4), tf.nest.map_structure(lambda x: x + 1, inst))

  def testArgsExpansion(self):

    def foo(a, b):
      return a + b

    t = structural_tuple.structtuple(['c', 'd'])

    self.assertEqual(3, nest_util.call_fn(foo, t(1, 2)))


if __name__ == '__main__':
  tf.test.main()
