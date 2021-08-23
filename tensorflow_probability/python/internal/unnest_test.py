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
"""Tests for nested attribute access utilities."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

from tensorflow_probability.python.internal import test_util
from tensorflow_probability.python.internal import unnest


JAX_MODE = False
NUMPY_MODE = False


class FakeResults(
    collections.namedtuple(
        'FakeResults',
        ['unique_core'])):
  __slots__ = ()


class FakeNestingResults(
    collections.namedtuple(
        'FakeNestingResults',
        ['unique_nesting',
         'inner_results'])):
  __slots__ = ()


class FakeAtypicalNestingResults(
    collections.namedtuple(
        'FakeAtypicalNestingResults',
        ['unique_atypical_nesting',
         'atypical_inner_results'])):
  __nested_attrs__ = 'atypical_inner_results'
  __slots__ = ()


def _build_deeply_nested(a, b, c, d, e):
  return FakeNestingResults(
      unique_nesting=a,
      inner_results=FakeAtypicalNestingResults(
          unique_atypical_nesting=b,
          atypical_inner_results=FakeAtypicalNestingResults(
              unique_atypical_nesting=c,
              atypical_inner_results=FakeNestingResults(
                  unique_nesting=d,
                  inner_results=FakeResults(e)))))


SINGLETON = object()


@test_util.test_all_tf_execution_regimes
class TestNestedAccessors(test_util.TestCase):

  def test_flat(self):
    results = FakeResults(0)
    self.assertTrue(unnest.has_nested(results, 'unique_core'))
    self.assertFalse(unnest.has_nested(results, 'foo'))
    self.assertEqual(results.unique_core,
                     unnest.get_innermost(results, 'unique_core'))
    self.assertEqual(results.unique_core,
                     unnest.get_outermost(results, 'unique_core'))
    self.assertRaises(
        AttributeError, lambda: unnest.get_innermost(results, 'foo'))
    self.assertRaises(
        AttributeError, lambda: unnest.get_outermost(results, 'foo'))
    self.assertIs(unnest.get_innermost(results, 'foo', SINGLETON), SINGLETON)
    self.assertIs(unnest.get_outermost(results, 'foo', SINGLETON), SINGLETON)

  def test_nesting(self):
    results = FakeNestingResults(unique_nesting=0, inner_results=FakeResults(1))
    self.assertTrue(unnest.has_nested(results, 'unique_nesting'))
    self.assertTrue(unnest.has_nested(results, 'unique_core'))
    self.assertFalse(hasattr(results, 'unique_core'))
    self.assertFalse(unnest.has_nested(results, 'foo'))
    self.assertEqual(results.unique_nesting,
                     unnest.get_innermost(results, 'unique_nesting'))
    self.assertEqual(results.unique_nesting,
                     unnest.get_outermost(results, 'unique_nesting'))
    self.assertEqual(results.inner_results.unique_core,
                     unnest.get_innermost(results, 'unique_core'))
    self.assertEqual(results.inner_results.unique_core,
                     unnest.get_outermost(results, 'unique_core'))
    self.assertRaises(
        AttributeError, lambda: unnest.get_innermost(results, 'foo'))
    self.assertRaises(
        AttributeError, lambda: unnest.get_outermost(results, 'foo'))
    self.assertIs(unnest.get_innermost(results, 'foo', SINGLETON), SINGLETON)
    self.assertIs(unnest.get_outermost(results, 'foo', SINGLETON), SINGLETON)

  def test_atypical_nesting(self):
    results = FakeAtypicalNestingResults(
        unique_atypical_nesting=0,
        atypical_inner_results=FakeResults(1))
    self.assertTrue(unnest.has_nested(results, 'unique_atypical_nesting'))
    self.assertTrue(unnest.has_nested(results, 'unique_core'))
    self.assertFalse(hasattr(results, 'unique_core'))
    self.assertFalse(unnest.has_nested(results, 'foo'))
    self.assertEqual(
        results.unique_atypical_nesting,
        unnest.get_innermost(results, 'unique_atypical_nesting'))
    self.assertEqual(
        results.unique_atypical_nesting,
        unnest.get_outermost(results, 'unique_atypical_nesting'))
    self.assertEqual(
        results.atypical_inner_results.unique_core,
        unnest.get_innermost(results, 'unique_core'))
    self.assertEqual(
        results.atypical_inner_results.unique_core,
        unnest.get_outermost(results, 'unique_core'))
    self.assertRaises(
        AttributeError, lambda: unnest.get_innermost(results, 'foo'))
    self.assertRaises(
        AttributeError, lambda: unnest.get_outermost(results, 'foo'))
    self.assertIs(unnest.get_innermost(results, 'foo', SINGLETON), SINGLETON)
    self.assertIs(unnest.get_outermost(results, 'foo', SINGLETON), SINGLETON)

  def test_deeply_nested(self):
    results = _build_deeply_nested(0, 1, 2, 3, 4)
    self.assertTrue(unnest.has_nested(results, 'unique_nesting'))
    self.assertTrue(unnest.has_nested(results, 'unique_atypical_nesting'))
    self.assertTrue(unnest.has_nested(results, 'unique_core'))
    self.assertFalse(hasattr(self, 'unique_core'))
    self.assertFalse(unnest.has_nested(results, 'foo'))
    self.assertEqual(unnest.get_innermost(results, 'unique_nesting'), 3)
    self.assertEqual(unnest.get_outermost(results, 'unique_nesting'), 0)
    self.assertEqual(
        unnest.get_innermost(results, 'unique_atypical_nesting'), 2)
    self.assertEqual(
        unnest.get_outermost(results, 'unique_atypical_nesting'), 1)
    self.assertEqual(unnest.get_innermost(results, 'unique_core'), 4)
    self.assertEqual(unnest.get_outermost(results, 'unique_core'), 4)
    self.assertRaises(
        AttributeError, lambda: unnest.get_innermost(results, 'foo'))
    self.assertRaises(
        AttributeError, lambda: unnest.get_outermost(results, 'foo'))
    self.assertIs(unnest.get_innermost(results, 'foo', SINGLETON), SINGLETON)
    self.assertIs(unnest.get_outermost(results, 'foo', SINGLETON), SINGLETON)

  def test_flat_replace(self):
    results = FakeResults(0)
    self.assertEqual(
        unnest.replace_innermost(results, unique_core=1).unique_core, 1)
    self.assertEqual(unnest.replace_innermost(
        results, return_unused=True, unique_core=1, foo=2),
                     (FakeResults(1), {'foo': 2}))
    self.assertEqual(
        unnest.replace_outermost(results, unique_core=2).unique_core, 2)
    self.assertEqual(unnest.replace_outermost(
        results, return_unused=True, unique_core=2, foo=3),
                     (FakeResults(2), {'foo': 3}))
    self.assertRaises(ValueError, lambda: unnest.replace_innermost(   # pylint: disable=g-long-lambda
        results, unique_core=1, foo=1))
    self.assertRaises(ValueError, lambda: unnest.replace_outermost(   # pylint: disable=g-long-lambda
        results, unique_core=1, foo=1))

  def test_nesting_replace(self):
    def build(a, b):
      return FakeNestingResults(unique_nesting=a, inner_results=FakeResults(b))
    results = build(0, 1)
    self.assertEqual(unnest.replace_innermost(results, unique_nesting=2),
                     build(2, 1))
    self.assertEqual(
        unnest.replace_innermost(results, unique_core=2), build(0, 2))
    self.assertEqual(
        unnest.replace_innermost(results, unique_nesting=2, unique_core=3),
        build(2, 3))
    self.assertEqual(unnest.replace_innermost(
        results, return_unused=True, unique_nesting=2, unique_core=3, foo=4),
                     (build(2, 3), {'foo': 4}))
    self.assertEqual(unnest.replace_outermost(results, unique_nesting=2),
                     build(2, 1))
    self.assertEqual(
        unnest.replace_outermost(results, unique_core=2), build(0, 2))
    self.assertEqual(
        unnest.replace_outermost(results, unique_nesting=2, unique_core=3),
        build(2, 3))
    self.assertEqual(unnest.replace_outermost(
        results, return_unused=True, unique_nesting=2, unique_core=3, foo=4),
                     (build(2, 3), {'foo': 4}))
    self.assertRaises(ValueError, lambda: unnest.replace_innermost(   # pylint: disable=g-long-lambda
        results, unique_core=1, foo=1))
    self.assertRaises(ValueError, lambda: unnest.replace_outermost(   # pylint: disable=g-long-lambda
        results, unique_core=1, foo=1))

  def test_atypical_nesting_replace(self):
    def build(a, b):
      return FakeAtypicalNestingResults(
          unique_atypical_nesting=a, atypical_inner_results=FakeResults(b))
    results = build(0, 1)
    self.assertEqual(unnest.replace_innermost(
        results, unique_atypical_nesting=2),
                     build(2, 1))
    self.assertEqual(
        unnest.replace_innermost(results, unique_core=2), build(0, 2))
    self.assertEqual(unnest.replace_innermost(
        results, unique_atypical_nesting=2, unique_core=3),
                     build(2, 3))
    self.assertEqual(unnest.replace_innermost(
        results, return_unused=True,
        unique_atypical_nesting=2, unique_core=3, foo=4),
                     (build(2, 3), {'foo': 4}))
    self.assertEqual(unnest.replace_outermost(
        results, unique_atypical_nesting=2),
                     build(2, 1))
    self.assertEqual(
        unnest.replace_outermost(results, unique_core=2), build(0, 2))
    self.assertEqual(unnest.replace_outermost(
        results, unique_atypical_nesting=2, unique_core=3),
                     build(2, 3))
    self.assertEqual(unnest.replace_outermost(
        results, return_unused=True,
        unique_atypical_nesting=2, unique_core=3, foo=4),
                     (build(2, 3), {'foo': 4}))
    self.assertRaises(ValueError, lambda: unnest.replace_innermost(   # pylint: disable=g-long-lambda
        results, unique_core=1, foo=1))
    self.assertRaises(ValueError, lambda: unnest.replace_outermost(   # pylint: disable=g-long-lambda
        results, unique_core=1, foo=1))

  def test_deeply_nested_replace(self):
    results = _build_deeply_nested(0, 1, 2, 3, 4)
    self.assertTrue(unnest.replace_innermost(results, unique_nesting=5),
                    _build_deeply_nested(0, 1, 2, 5, 4))
    self.assertTrue(unnest.replace_outermost(results, unique_nesting=5),
                    _build_deeply_nested(5, 1, 2, 3, 4))
    self.assertTrue(unnest.replace_innermost(
        results, unique_atypical_nesting=5),
                    _build_deeply_nested(0, 1, 5, 3, 4))
    self.assertTrue(unnest.replace_outermost(
        results, unique_atypical_nesting=5),
                    _build_deeply_nested(0, 5, 2, 3, 4))
    self.assertTrue(unnest.replace_innermost(results, unique_core=5),
                    _build_deeply_nested(0, 1, 2, 3, 5))
    self.assertTrue(unnest.replace_outermost(results, unique_core=5),
                    _build_deeply_nested(0, 1, 2, 3, 5))
    self.assertTrue(unnest.replace_innermost(
        results, unique_nesting=5, unique_atypical_nesting=6, unique_core=7),
                    _build_deeply_nested(0, 1, 6, 5, 7))
    self.assertTrue(unnest.replace_innermost(
        results, return_unused=True, unique_nesting=5,
        unique_atypical_nesting=6, unique_core=7, foo=8),
                    (_build_deeply_nested(0, 1, 6, 5, 7), {'foo': 8}))
    self.assertTrue(unnest.replace_outermost(
        results, unique_nesting=5, unique_atypical_nesting=6, unique_core=7),
                    _build_deeply_nested(5, 6, 2, 3, 7))
    self.assertTrue(unnest.replace_outermost(
        results, return_unused=True, unique_nesting=5,
        unique_atypical_nesting=6, unique_core=7, foo=8),
                    (_build_deeply_nested(5, 6, 2, 3, 7), {'foo': 8}))
    self.assertRaises(ValueError, lambda: unnest.replace_innermost(   # pylint: disable=g-long-lambda
        results, unique_core=1, foo=1))
    self.assertRaises(ValueError, lambda: unnest.replace_outermost(   # pylint: disable=g-long-lambda
        results, unique_core=1, foo=1))

  def test_get_nested_obj(self):
    class Nest:
      __nested_attrs__ = ('inner1', 'inner2')

      def __init__(self, inner1, inner2, inner3):
        self.inner1 = inner1
        self.inner2 = inner2
        self.inner3 = inner3

    nest = Nest('x', 'y', 'z')

    # Search stops if `nested_key` is found.
    self.assertEqual(unnest.get_nested_objs(
        nest, nested_key='__nested_attrs__',
        fallback_attrs=('inner1', 'inner2', 'inner3')),
                     [('inner1', 'x'), ('inner2', 'y')])

    # If `nested_key` is not found, search in all of `fallback_attrs`.
    self.assertEqual(unnest.get_nested_objs(
        nest, nested_key='does_not_exist',
        fallback_attrs=('inner1', 'inner2', 'inner3')),
                     [('inner1', 'x'), ('inner2', 'y'), ('inner3', 'z')])

    # If nothing is found, empty list is returned.
    self.assertEqual(unnest.get_nested_objs(
        nest, nested_key='does_not_exist',
        fallback_attrs=('does_not_exist')),
                     [])

    # `getattr(obj, nested_key)` and `fallback_attrs` can be either strings or
    # collections of strings.
    nest2 = Nest('x', 'y', 'z')
    nest2.__nested_attrs__ = 'inner3'
    self.assertEqual(unnest.get_nested_objs(
        nest2, nested_key='__nested_attrs__',
        fallback_attrs=('inner1', 'inner2')),
                     [('inner3', 'z')])
    self.assertEqual(unnest.get_nested_objs(
        nest2, nested_key='does_not_exist', fallback_attrs='inner2'),
                     [('inner2', 'y')])


@test_util.test_all_tf_execution_regimes
class UnnestingWrapperTests(test_util.TestCase):

  def test_inner(self):
    results = _build_deeply_nested(0, 1, 2, 3, 4)
    wrap = unnest.UnnestingWrapper(results)
    self.assertEqual(wrap.unique_nesting, 3)
    self.assertEqual(wrap.unique_atypical_nesting, 2)
    self.assertEqual(wrap.unique_core, 4)
    self.assertRaises(AttributeError, lambda: wrap.foo)

  def test_outer(self):
    results = _build_deeply_nested(0, 1, 2, 3, 4)
    wrap = unnest.UnnestingWrapper(results, innermost=False)
    self.assertEqual(wrap.unique_nesting, 0)
    self.assertEqual(wrap.unique_atypical_nesting, 1)
    self.assertEqual(wrap.unique_core, 4)
    self.assertRaises(AttributeError, lambda: wrap.foo)


if __name__ == '__main__':
  test_util.main()
