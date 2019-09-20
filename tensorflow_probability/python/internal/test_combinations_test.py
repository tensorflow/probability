# Copyright 2018 The TensorFlow Probability Authors.
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
"""Tests for tensorflow_probability.python.internal.test_combinations."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import tensorflow.compat.v2 as tf
from tensorflow_probability.python.internal import test_case
from tensorflow_probability.python.internal import test_combinations

from tensorflow.python.eager import context  # pylint: disable=g-direct-tensorflow-import
from tensorflow.python.eager import def_function  # pylint: disable=g-direct-tensorflow-import
from tensorflow.python.framework import combinations as tf_combinations  # pylint: disable=g-direct-tensorflow-import
from tensorflow.python.framework import test_combinations as tf_test_combinations  # pylint: disable=g-direct-tensorflow-import


#
# The are following are "pretend" test case classes, which are actually examined
# in unit tests below, to verify (and document) the generated test names.
#
@test_combinations.test_all_tf_execution_regimes
class PretendTestCaseClass(parameterized.TestCase):

  def test_something(self):
    self.skipTest('Fake test')


@test_combinations.test_all_tf_execution_regimes
class PretendParameterizedTestCaseClass(parameterized.TestCase):

  @parameterized.named_parameters([dict(testcase_name='p123', p='123')])
  def test_something(self, p):
    del p  # avoid unused
    self.skipTest('Fake test')


@test_combinations.test_graph_and_eager_modes
class PretendTestCaseClassGraphAndEagerOnly(parameterized.TestCase):

  def test_something(self):
    self.skipTest('Fake test')


class TestCombinationsTest(test_case.TestCase, parameterized.TestCase):

  #
  # These tests check that the generated names are as expected.
  #
  def test_generated_test_case_names(self):
    expected_test_names = [
        'test_something_test_mode_eager_tffunction_disabled',
        'test_something_test_mode_eager_tffunction_enabled',
        'test_something_test_mode_graph_tffunction_enabled',
    ]

    for expected_test_name in expected_test_names:
      self.assertIn(expected_test_name, dir(PretendTestCaseClass))

  def test_generated_parameterized_test_case_names(self):
    expected_test_names = [
        'test_something_p123_test_mode_eager_tffunction_disabled',
        'test_something_p123_test_mode_eager_tffunction_enabled',
        'test_something_p123_test_mode_graph_tffunction_enabled',
    ]

    for expected_test_name in expected_test_names:
      self.assertIn(expected_test_name, dir(PretendParameterizedTestCaseClass))

  def test_generated_graph_and_eager_test_case_names(self):
    expected_test_names = [
        'test_something_test_mode_eager',
        'test_something_test_mode_eager',
        'test_something_test_mode_graph',
    ]

    for expected_test_name in expected_test_names:
      self.assertIn(expected_test_name,
                    dir(PretendTestCaseClassGraphAndEagerOnly))

  #
  # These tests ensure that the test generators do what they say on the tin.
  #
  @tf_test_combinations.generate(
      tf_test_combinations.combine(mode='graph'),
      test_combinations=[tf_combinations.EagerGraphCombination()])
  def test_graph_mode_combination(self):
    self.assertFalse(context.executing_eagerly())

  @tf_test_combinations.generate(
      tf_test_combinations.combine(mode='eager'),
      test_combinations=[tf_combinations.EagerGraphCombination()])
  def test_eager_mode_combination(self):
    self.assertTrue(context.executing_eagerly())

  @tf_test_combinations.generate(
      tf_test_combinations.combine(tf_function='enabled'),
      test_combinations=[
          test_combinations.ExecuteFunctionsEagerlyCombination()])
  def test_tf_function_enabled_mode_combination(self):
    self.assertFalse(def_function.RUN_FUNCTIONS_EAGERLY)

  @tf_test_combinations.generate(
      tf_test_combinations.combine(tf_function='disabled'),
      test_combinations=[
          test_combinations.ExecuteFunctionsEagerlyCombination()])
  def test_tf_function_disabled_mode_combination(self):
    self.assertTrue(def_function.RUN_FUNCTIONS_EAGERLY)


if __name__ == '__main__':
  tf.test.main()
