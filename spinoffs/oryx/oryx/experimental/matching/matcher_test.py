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
"""Tests for tensorflow_probability.spinoffs.oryx.experimental.matching.matcher."""

from absl.testing import absltest
from oryx.experimental.matching import matcher
from oryx.internal import test_util


class MatcherTest(test_util.TestCase):

  def test_default_matcher_correctly_matches_equal_values(self):
    self.assertDictEqual(matcher.match(1., 1.), {})
    self.assertDictEqual(matcher.match(1., 1), {})
    self.assertDictEqual(matcher.match('hello', 'hello'), {})

  def test_default_matcher_errors_on_nonequal_values(self):
    with self.assertRaises(matcher.MatchError):
      matcher.match(0., 1.)
    with self.assertRaises(matcher.MatchError):
      matcher.match('a', 'b')

  def test_not_pattern_correctly_matches_nonequal_values(self):
    self.assertDictEqual(matcher.match(matcher.Not(0.), 1.), {})
    self.assertDictEqual(matcher.match(matcher.Not('a'), 1.), {})

  def test_is_match_returns_the_correct_bool_values_according_to_match(self):
    self.assertTrue(matcher.is_match(matcher.Not(0.), 1.))
    self.assertFalse(matcher.is_match(matcher.Not('a'), 'a'))

  def test_is_match_errors_with_star_pattern(self):
    with self.assertRaises(ValueError):
      matcher.is_match(matcher.Star(1), 1.)

  def test_match_errors_with_star_pattern(self):
    with self.assertRaises(ValueError):
      matcher.match(matcher.Star(1), 1.)

  def test_match_all_errors_with_star_pattern(self):
    with self.assertRaises(ValueError):
      list(matcher.match_all(matcher.Star(1), 1.))

  def test_tuple_patterns_match_equal_tuples(self):
    self.assertDictEqual(matcher.match((1, 2, 3), (1, 2, 3)), {})
    self.assertDictEqual(matcher.match(((1, 2), 2, 3), ((1, 2), 2, 3)), {})

  def test_tuple_patterns_error_on_nonequal_tuples(self):
    with self.assertRaises(matcher.MatchError):
      matcher.match((1, 2, 4), (1, 2, 3))

  def test_dict_patterns_match_equal_dicts(self):
    self.assertDictEqual(matcher.match(dict(a=1, b=2), dict(a=1, b=2)), {})
    self.assertDictEqual(matcher.match(dict(a=1, b=2), dict(b=2, a=1)), {})

  def test_dict_patterns_error_on_nonequal_dicts(self):
    with self.assertRaises(matcher.MatchError):
      matcher.match(dict(a=1, b=2), dict(a=2, b=2))
    with self.assertRaises(matcher.MatchError):
      matcher.match(dict(a=1, b=2), dict(a=1, b=2, c=3))

  def test_var_pattern_matches_any_expression(self):
    self.assertDictEqual(matcher.match(matcher.Var('x'), 1.), {'x': 1.})
    self.assertDictEqual(
        matcher.match(matcher.Var('x'), 'hello'), {'x': 'hello'})

  def test_var_pattern_matches_when_bound_value_matches(self):
    x = matcher.Var('x')
    self.assertDictEqual(matcher.match((x, x), (1, 1)), dict(x=1))

  def test_var_correctly_applies_restrictions_when_matching(self):
    is_positive = lambda a: a > 0
    is_even = lambda a: a % 2 == 0
    x = matcher.Var('x', restrictions=[is_positive, is_even])
    self.assertDictEqual(matcher.match(x, 2.), dict(x=2))
    with self.assertRaises(matcher.MatchError):
      matcher.match(x, -2)
    with self.assertRaises(matcher.MatchError):
      matcher.match(x, 3)

  def test_var_pattern_errors_when_bound_value_doesnt_match(self):
    x = matcher.Var('x')
    with self.assertRaises(matcher.MatchError):
      matcher.match((x, x), (1, 2))

  def test_dot_pattern_matches_without_creating_binding(self):
    self.assertDictEqual(matcher.match(matcher.Dot, 1), {})
    self.assertDictEqual(matcher.match((matcher.Dot, matcher.Dot), (1, 2)), {})

  def test_choice_correctly_matches_multiple_options(self):
    pattern = matcher.Choice(1, 2)
    self.assertDictEqual(matcher.match(pattern, 1), {})
    self.assertDictEqual(matcher.match(pattern, 2), {})
    with self.assertRaises(matcher.MatchError):
      matcher.match(pattern, 3)

  def test_choice_will_backtrack_and_try_other_options(self):
    x, y = matcher.Var('x'), matcher.Var('y')
    self.assertDictEqual(
        matcher.match((matcher.Choice(x, y), x, y), (3, 2, 3)),
        dict(x=2, y=3))

  def test_choice_with_name_binds_value(self):
    pattern = matcher.Choice((1, 2), (2, 1), name='x')
    self.assertDictEqual(matcher.match(pattern, (1, 2)), {'x': (1, 2)})
    self.assertDictEqual(matcher.match(pattern, (2, 1)), {'x': (2, 1)})

  def test_star_match_correctly_matches_sequence_of_patterns(self):
    pattern = (matcher.Star(1),)
    self.assertDictEqual(matcher.match(pattern, ()), {})
    self.assertDictEqual(matcher.match(pattern, (1,)), {})
    self.assertDictEqual(matcher.match(pattern, (1, 1)), {})
    self.assertDictEqual(matcher.match(pattern, (1, 1, 1)), {})

    with self.assertRaises(matcher.MatchError):
      matcher.match(pattern, (1, 2))

  def test_star_match_produces_multiple_matches(self):
    pattern = (matcher.Star(1), matcher.Star(1))
    self.assertLen(list(matcher.match_all(pattern, ())), 1)
    self.assertLen(list(matcher.match_all(pattern, (1,))), 2)
    self.assertLen(list(matcher.match_all(pattern, (1, 1))), 3)

    pattern = (matcher.Star(1), matcher.Star(1), matcher.Star(1))
    self.assertLen(list(matcher.match_all(pattern, ())), 1)
    self.assertLen(list(matcher.match_all(pattern, (1,))), 3)
    self.assertLen(list(matcher.match_all(pattern, (1, 1))), 6)
    self.assertLen(list(matcher.match_all(pattern, (1, 1, 1))), 10)

  def test_star_nongreedily_matches_by_default(self):
    pattern = (matcher.Star(1, name='x'), matcher.Star(1, name='y'))
    # x will be the smallest possible match
    self.assertDictEqual(matcher.match(pattern, (1, 1)), dict(x=(), y=(1, 1)))

  def test_star_greedily_matches_when_flag_is_set(self):
    pattern = (matcher.Star(1, name='x', greedy=True),
               matcher.Star(1, name='y', greedy=True))
    # x will be the largset possible match
    self.assertDictEqual(matcher.match(pattern, (1, 1)), dict(x=(1, 1), y=()))

    pattern = (matcher.Star(1, name='x', greedy=True),
               matcher.Star(1, name='y', greedy=False))
    self.assertDictEqual(matcher.match(pattern, (1, 1)), dict(x=(1, 1), y=()))

  def test_star_match_binds_name_to_environment(self):
    pattern = (matcher.Star(matcher.Var('x')),)
    self.assertDictEqual(matcher.match(pattern, ()), {})
    self.assertDictEqual(matcher.match(pattern, (1,)), dict(x=1))
    self.assertDictEqual(matcher.match(pattern, (1, 1)), dict(x=1))

    with self.assertRaises(matcher.MatchError):
      matcher.match(pattern, (1, 2))

  def test_star_can_nest_to_match_nested_patterns(self):
    pattern = (matcher.Star((matcher.Star(1),)),)
    self.assertDictEqual(matcher.match(pattern, ()), {})
    self.assertDictEqual(matcher.match(pattern, ((),)), {})
    self.assertDictEqual(matcher.match(pattern, ((1,), (1, 1, 1))), {})

    with self.assertRaises(matcher.MatchError):
      matcher.match(pattern, (1,))

  def test_star_with_accumulate_collects_values(self):
    pattern = (matcher.Star((matcher.Var('x'), matcher.Var('y')),
                            accumulate=['y']),)
    self.assertDictEqual(
        matcher.match(pattern, ((1, 2), (1, 3))), dict(x=1, y=(2, 3)))

    with self.assertRaises(matcher.MatchError):
      matcher.match(pattern, ((1, 2), (2, 3)))

  def test_star_with_name_binds_result(self):
    pattern = (matcher.Star(1, name='x'),)
    self.assertDictEqual(matcher.match(pattern, (1, 1)), dict(x=(1, 1)))
    pattern = (matcher.Star(matcher.Var('y'), name='x'),)
    self.assertDictEqual(matcher.match(pattern, (1, 1)), dict(y=1, x=(1, 1)))

  def test_plus_errors_on_empty_tuple(self):
    pattern = (matcher.Plus(1),)
    with self.assertRaises(matcher.MatchError):
      matcher.match(pattern, ())

  def test_star_with_plus_matches_nonempty_tuple(self):
    pattern = (matcher.Plus(1),)
    self.assertDictEqual(matcher.match(pattern, (1, 1)), {})

  def test_segment_matches_tuple_slices(self):
    pattern = (matcher.Segment('a'),)
    self.assertDictEqual(matcher.match(pattern, (1, 2, 3)), {'a': (1, 2, 3)})

    pattern = (matcher.Segment('a'), 2, 3)
    self.assertDictEqual(matcher.match(pattern, (1, 2, 3)), {'a': (1,)})

    pattern = (matcher.Segment('a'), 2, 3)
    with self.assertRaises(matcher.MatchError):
      matcher.match(pattern, (1, 2))

  def test_segment_matches_multiple_tuple_slices(self):
    pattern = (matcher.Segment('a'), matcher.Segment('b'))
    self.assertLen(list(matcher.match_all(pattern, ())), 1)
    self.assertLen(list(matcher.match_all(pattern, (1,))), 2)
    self.assertLen(list(matcher.match_all(pattern, (1, 1))), 3)

  def test_segment_must_be_the_same_when_given_same_name(self):
    pattern = (matcher.Segment('a'), matcher.Segment('a'))
    self.assertDictEqual(matcher.match(pattern, ()), {'a': ()})
    self.assertDictEqual(matcher.match(pattern, (1, 1)), {'a': (1,)})
    with self.assertRaises(matcher.MatchError):
      matcher.match(pattern, (1, 1, 1))

    pattern = (matcher.Segment('x'), matcher.Segment('y'), matcher.Segment('x'))
    matches = list(matcher.match_all(pattern, (1,) * 10))
    self.assertLen(matches, 6)
    for i in range(len(matches)):
      self.assertDictEqual(matches[i], dict(x=(1,) * i, y=(1,) * (10 - 2 * i)))


if __name__ == '__main__':
  absltest.main()
