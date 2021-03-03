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
"""Tests for tensorflow_probability.spinoffs.oryx.experimental.matching.rules."""

from absl.testing import absltest
from oryx.experimental.matching import matcher
from oryx.experimental.matching import rules
from oryx.internal import test_util

is_number = lambda x: isinstance(x, (int, float))
is_positive = lambda x: x > 0
is_tuple = lambda x: isinstance(x, tuple)

Number = lambda name: matcher.Var(name, restrictions=[is_number])
Positive = lambda name: matcher.Var(name, restrictions=[is_number, is_positive])
Tuple = lambda name: matcher.Var(name, restrictions=[is_tuple])


class RulesTest(test_util.TestCase):

  def test_rule_should_rewrite_matching_expression(self):
    one_rule = rules.make_rule(2., lambda: 1.)
    self.assertEqual(one_rule(2.), 1.)

  def test_rule_doesnt_rewrite_nonmatching_expression(self):
    one_rule = rules.make_rule(2., lambda: 1.)
    self.assertEqual(one_rule(0.), 0.)

  def test_rule_should_pass_bindings_into_rewrite(self):
    add_one = rules.make_rule(matcher.Var('a'), lambda a: a + 1)
    self.assertEqual(add_one(1.), 2.)

  def test_rule_with_restrictions_should_not_rewrite_if_no_match(self):
    add_one = rules.make_rule(Number('a'), lambda a: a + 1)
    self.assertEqual(add_one(1.), 2.)
    self.assertEqual(add_one(()), ())

  def test_rule_list_should_apply_first_rule_that_matches(self):
    rule = rules.rule_list(
        rules.make_rule(Positive('a'), lambda a: a + 1.),
        rules.make_rule(Number('a'), lambda a: a + 2.))
    self.assertEqual(rule(-1.), 1.)
    self.assertEqual(rule(2.), 3)

  def test_rule_list_should_not_rewrite_expression_if_no_rules_match(self):
    rule = rules.rule_list(
        rules.make_rule(Positive('a'), lambda a: a + 2.),
        rules.make_rule(Positive('a'), lambda a: a + 3.))
    self.assertEqual(rule(-1.), -1.)

  def test_in_order_should_apply_rules_even_if_multiple_match(self):
    rule = rules.in_order(
        rules.make_rule(Positive('a'), lambda a: a + 2.),
        rules.make_rule(Positive('a'), lambda a: a + 3.))
    self.assertEqual(rule(1.), 6.)

  def test_iterated_should_apply_rule_until_expression_no_longer_matches(self):
    rule = rules.iterated(
        rules.make_rule(Positive('a'), lambda a: a - 1.))
    self.assertEqual(rule(1.), 0.)
    self.assertEqual(rule(10.), 0.)
    self.assertEqual(rule(-1.), -1.)

  def test_rewrite_subexpressions_should_not_rewrite_primitive_types(self):
    rule = rules.rewrite_subexpressions(
        rules.make_rule(Positive('a'), lambda a: a + 1.))
    self.assertEqual(rule(1.), 1.)
    self.assertEqual(rule(-1.), -1.)

  def test_rewrite_subexpressions_should_rewrite_tuple_elements(self):
    rule = rules.rewrite_subexpressions(
        rules.make_rule(Positive('a'), lambda a: a + 1.))
    self.assertEqual(rule((-1., 0., 1.)), (-1., 0., 2.))

  def test_rewrite_subexpressions_should_not_recursively_rewrite_elements(self):
    rule = rules.rewrite_subexpressions(
        rules.make_rule(Positive('a'), lambda a: a + 1.))
    self.assertEqual(rule(((-1.,), 1., (1., 1.))), ((-1.,), 2., (1., 1.)))

  def test_bottom_up_should_recursively_rewrite_elements(self):
    rule = rules.bottom_up(
        rules.make_rule(Positive('a'), lambda a: a + 1.))
    self.assertEqual(rule(((-1.,), 1., (1., 1.))), ((-1.,), 2., (2., 2.)))

  def test_bottom_up_should_rewrite_from_children_to_root(self):
    rule = rules.bottom_up(rules.in_order(
        rules.make_rule(Positive('a'), lambda a: a + 1.),
        rules.make_rule(Tuple('t'), lambda t: sum(t))  # pylint: disable=unnecessary-lambda
        ))
    self.assertEqual(rule((1., 2., 3.)), 9.)

  def test_top_down_should_recursively_rewrite_elements(self):
    rule = rules.top_down(
        rules.make_rule(Positive('a'), lambda a: a + 1.))
    self.assertEqual(rule(((-1.,), 1., (1., 1.))), ((-1.,), 2., (2., 2.)))

  def test_top_down_should_rewrite_from_root_to_children(self):
    rule = rules.top_down(rules.in_order(
        rules.make_rule(Tuple('t'), lambda t: sum(t)),  # pylint: disable=unnecessary-lambda
        rules.make_rule(Positive('a'), lambda a: a + 1.),
        ))
    self.assertEqual(rule((1., 2., 3.)), 7.)

  def test_term_rewriter_should_recursively_rewrite_until_convergence(self):
    rule = rules.term_rewriter(
        rules.make_rule(Positive('a'), lambda a: a - 1.))
    self.assertEqual(rule(1.), 0.)
    self.assertEqual(rule((1., 2., 3.)), (0., 0., 0.))
    self.assertEqual(rule(((1., 2.), 3.)), ((0., 0.), 0.))

if __name__ == '__main__':
  absltest.main()
