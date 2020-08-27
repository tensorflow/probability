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

  def testSlicing(self):
    t = structural_tuple.structtuple(['a', 'b', 'c'])
    inst = t(a=1, b=2, c=3)

    abc = inst[:]
    self.assertAllEqual((1, 2, 3), tuple(abc))
    self.assertAllEqual(('a', 'b', 'c'), abc._fields)

    ab = inst[:2]
    self.assertAllEqual((1, 2), tuple(ab))
    self.assertAllEqual(('a', 'b'), ab._fields)

    ac = inst[::2]
    self.assertAllEqual((1, 3), tuple(ac))
    self.assertAllEqual(('a', 'c'), ac._fields)

    ab2 = abc[:2]
    self.assertAllEqual((1, 2), tuple(ab2))
    self.assertAllEqual(('a', 'b'), ab2._fields)

  def testConcatenation(self):
    t1 = structural_tuple.structtuple(['a', 'b'])
    t2 = structural_tuple.structtuple(['c', 'd'])
    ab = t1(a=1, b=2)
    cd = t2(c=3, d=4)

    abcd = ab + cd
    self.assertAllEqual((1, 2, 3, 4), tuple(abcd))
    self.assertAllEqual(('a', 'b', 'c', 'd'), abcd._fields)

    cdab = cd + ab
    self.assertAllEqual((3, 4, 1, 2), tuple(cdab))
    self.assertAllEqual(('c', 'd', 'a', 'b'), cdab._fields)

    ab_tuple = ab + (3,)
    self.assertAllEqual((1, 2, 3), ab_tuple)

    tuple_ab = (3,) + ab
    self.assertAllEqual((3, 1, 2), tuple_ab)

  def testArgsExpansion(self):

    def foo(a, b):
      return a + b

    t = structural_tuple.structtuple(['c', 'd'])

    self.assertEqual(3, nest_util.call_fn(foo, t(1, 2)))


if __name__ == '__main__':
  tf.test.main()
