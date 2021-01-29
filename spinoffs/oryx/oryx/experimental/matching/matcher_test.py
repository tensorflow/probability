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


if __name__ == '__main__':
  absltest.main()
