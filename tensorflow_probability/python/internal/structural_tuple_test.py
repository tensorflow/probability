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

# Dependency imports

import tensorflow.compat.v2 as tf

from tensorflow_probability.python.internal import nest_util
from tensorflow_probability.python.internal import structural_tuple
from tensorflow_probability.python.internal import test_util
from tensorflow.python.util import nest  # pylint: disable=g-direct-tensorflow-import

JAX_MODE = False


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
    inst2 = inst._replace(a=2)
    self.assertAllEqualNested(t(2, 2, 3), inst2)
    self.assertTrue(nest._is_namedtuple(inst))

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

  def testMoreThan255Fields(self):
    num_fields = 1000
    t = structural_tuple.structtuple(
        ['field{}'.format(n) for n in range(num_fields)])
    self.assertLen(t._fields, num_fields)

  def testMake(self):
    t = structural_tuple.structtuple(['a', 'b'])
    ab = t._make((1, 2))
    self.assertEqual(1, ab.a)
    self.assertEqual(2, ab.b)
    ab = t._make((1,))
    self.assertEqual(1, ab.a)
    self.assertIs(None, ab.b)

  def testMakeTooManyValues(self):
    t = structural_tuple.structtuple(['a', 'b'])
    with self.assertRaisesRegex(TypeError,
                                'Expected 2 arguments or fewer, got 3'):
      t._make([1, 2, 3])

  def testNonStrField(self):
    with self.assertRaisesRegex(
        TypeError, 'Field names must be strings: 1 has type <class \'int\'>'):
      structural_tuple.structtuple([1])

  def testInvalidIdentifierField(self):
    with self.assertRaisesRegex(ValueError,
                                'Field names must be valid identifiers: 0'):
      structural_tuple.structtuple(['0'])

  def testKeywordField(self):
    with self.assertRaisesRegex(ValueError,
                                'Field names cannot be a keyword: def'):
      structural_tuple.structtuple(['def'])

  def testUnderscoreField(self):
    with self.assertRaisesRegex(
        ValueError, 'Field names cannot start with an underscore: _a'):
      structural_tuple.structtuple(['_a'])

  def testDuplicateField(self):
    with self.assertRaisesRegex(ValueError,
                                'Encountered duplicate field name: a'):
      structural_tuple.structtuple(['a', 'a'])

  def testDuplicateConstructorArg(self):
    t = structural_tuple.structtuple(['a'])
    with self.assertRaisesRegex(TypeError,
                                'Got multiple values for argument a'):
      t(1, a=2)

  def testUnexpectedConstructorArg(self):
    t = structural_tuple.structtuple(['a'])
    with self.assertRaisesRegex(TypeError,
                                'Got an unexpected keyword argument b'):
      t(b=2)

  def testMissingAttribute(self):
    t = structural_tuple.structtuple(['a'])
    a = t()
    with self.assertRaisesRegex(AttributeError,
                                'StructTuple has no attribute b'):
      _ = a.b

  def testReplaceUnknownFields(self):
    t = structural_tuple.structtuple(['a'])
    a = t()
    with self.assertRaisesRegex(
        ValueError, r'Got unexpected field names: \[\'b\', \'c\'\]'):
      a._replace(b=1, c=2)


if JAX_MODE:
  import jax  # pylint: disable=g-import-not-at-top

  class StructTupleJAXTest(test_util.TestCase):

    def testTreeUtilIntegration(self):
      t = structural_tuple.structtuple(['a', 'b', 'c'])
      inst = t(a=1, b=2, c=3)
      mapped = jax.tree_util.tree_map(lambda x: x + 1, inst)
      self.assertIsInstance(mapped, type(inst))
      self.assertAllEqualNested(t(2, 3, 4), mapped)


if __name__ == '__main__':
  test_util.main()
