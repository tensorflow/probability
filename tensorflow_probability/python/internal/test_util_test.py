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
"""Tests for utilities for testing distributions and/or bijectors."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

# Dependency imports
from absl import flags
from absl.testing import flagsaver
from absl.testing import parameterized
import numpy as np
import tensorflow.compat.v1 as tf1
import tensorflow.compat.v2 as tf

from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import test_combinations
from tensorflow_probability.python.internal import test_util
from tensorflow_probability.python.util.seed_stream import SeedStream
from tensorflow.python.eager import context  # pylint: disable=g-direct-tensorflow-import


FLAGS = flags.FLAGS


JAX_MODE = False


def _maybe_jax(x):
  if JAX_MODE:
    from jax import random as jaxrand  # pylint: disable=g-import-not-at-top
    x = jaxrand.PRNGKey(x)

  return x


@test_util.test_all_tf_execution_regimes
class SeedSettingTest(test_util.TestCase):

  def testTypeCorrectness(self):
    assert isinstance(test_util.test_seed_stream(), SeedStream)
    assert isinstance(
        test_util.test_seed_stream(hardcoded_seed=7), SeedStream)
    assert isinstance(test_util.test_seed_stream(salt='foo'), SeedStream)

  def testSameness(self):
    with flagsaver.flagsaver(vary_seed=False):
      self.assertAllEqual(test_util.test_seed(), test_util.test_seed())
      self.assertAllEqual(test_util.test_seed_stream()(),
                          test_util.test_seed_stream()())
      with flagsaver.flagsaver(fixed_seed=None):
        x = 47
        expected = _maybe_jax(x)
        self.assertAllEqual(expected, test_util.test_seed(hardcoded_seed=x))

  def testVariation(self):
    with flagsaver.flagsaver(vary_seed=True, fixed_seed=None):
      self.assertFalse(
          np.all(test_util.test_seed() == test_util.test_seed()))
      self.assertFalse(
          np.all(test_util.test_seed_stream()() ==
                 test_util.test_seed_stream()()))
      x = 47
      expect_not = _maybe_jax(x)
      self.assertFalse(
          np.all(expect_not == test_util.test_seed(hardcoded_seed=x)))

  def testFixing(self):
    expected = _maybe_jax(58)
    with flagsaver.flagsaver(fixed_seed=58):
      self.assertAllEqual(expected, test_util.test_seed())
      self.assertAllEqual(expected, test_util.test_seed(hardcoded_seed=47))


class _TestCaseTest(object):

  def setUp(self):  # pylint: disable=g-missing-super-call
    np.random.seed(932)

  def test_assert_all_finite_input_finite(self):
    minval = tf.constant(dtype_util.min(self.dtype), dtype=self.dtype)
    maxval = tf.constant(dtype_util.max(self.dtype), dtype=self.dtype)

    # This tests if the minimum value for the dtype is detected as finite.
    self.assertAllFinite(minval)

    # This tests if the maximum value for the dtype is detected as finite.
    self.assertAllFinite(maxval)

    # This tests if a rank 3 `Tensor` with entries in the range
    # [0.4*minval, 0.4*maxval] is detected as finite.
    # The choice of range helps to avoid overflows or underflows
    # in tf.linspace calculations.
    num_elem = 1000
    shape = (10, 10, 10)
    a = tf.reshape(tf.linspace(0.4*minval, 0.4*maxval, num_elem), shape)
    self.assertAllFinite(a)

  def test_assert_all_finite_input_nan(self):
    # This tests if np.nan is detected as non-finite.
    num_elem = 1000
    shape = (10, 10, 10)
    a = np.linspace(0., 1., num_elem)
    a[50] = np.nan
    a = tf.reshape(tf.convert_to_tensor(value=a, dtype=self.dtype), shape)
    with self.assertRaisesRegexp(AssertionError, 'Arrays are not equal'):
      self.assertAllFinite(a)

  def test_assert_all_finite_input_inf(self):
    # This tests if np.inf is detected as non-finite.
    num_elem = 1000
    shape = (10, 10, 10)
    a = np.linspace(0., 1., num_elem)
    a[100] = np.inf
    a = tf.reshape(tf.convert_to_tensor(value=a, dtype=self.dtype), shape)
    with self.assertRaisesRegexp(AssertionError, 'Arrays are not equal'):
      self.assertAllFinite(a)

  def test_assert_all_finite_input_py_literal(self):
    # This tests if finite Python literals are detected as finite.
    for a in [1, 3., -7.1e-12]:
      self.assertAllFinite(a)

    b = [0, 1.45e17, 0x2a]
    self.assertAllFinite(b)

    c = (1, 2., 3, 4)
    self.assertAllFinite(c)

  def test_assert_all_nan_input_all_nan(self):
    a = tf.convert_to_tensor(
        value=np.full((10, 10, 10), np.nan), dtype=self.dtype)
    self.assertAllNan(a)

  def test_assert_all_nan_input_some_nan(self):
    a = np.random.rand(10, 10, 10)
    a[1, :, :] = np.nan
    a = tf.convert_to_tensor(value=a, dtype=self.dtype)
    with self.assertRaisesRegexp(AssertionError, 'Arrays are not equal'):
      self.assertAllNan(a)

  def test_assert_all_nan_input_numpy_rand(self):
    a = np.random.rand(10, 10, 10).astype(dtype_util.as_numpy_dtype(self.dtype))
    with self.assertRaisesRegexp(AssertionError, 'Arrays are not equal'):
      self.assertAllNan(a)

  def test_assert_all_nan_input_inf(self):
    a = tf.convert_to_tensor(
        value=np.full((10, 10, 10), np.inf), dtype=self.dtype)
    with self.assertRaisesRegexp(AssertionError, 'Arrays are not equal'):
      self.assertAllNan(a)

  def test_assert_all_nan_input_placeholder_with_default(self):
    all_nan = np.full((10, 10, 10),
                      np.nan).astype(dtype_util.as_numpy_dtype(self.dtype))
    a = tf1.placeholder_with_default(all_nan, shape=all_nan.shape)
    self.assertAllNan(a)

  def test_assert_all_are_not_none(self):
    no_nones = [1, 2, 3]
    self.assertAllNotNone(no_nones)

    has_nones = [1, 2, None]
    with self.assertRaisesRegexp(
        AssertionError,
        r'Expected no entry to be `None` but found `None` in positions \[2\]'):
      self.assertAllNotNone(has_nones)

  def test_assert_nested_all_ok(self):
    self.assertAllAssertsNested(self.assertEqual, [4, {'a': 3}], [4, {'a': 3}])

  def test_assert_nested_mismatched_structure(self):
    expected_msg = (
        "test:\n\nThe two structures don't have the same sequence type. " +
        "Input structure has type <(class|type) 'list'>, while shallow "
        "structure has type <(class|type) 'dict'>.")

    with self.assertRaisesRegexp(AssertionError, expected_msg):
      self.assertAllAssertsNested(self.assertEqual, {'a': 3}, [3], msg='test')

  def test_assert_nested_mismatched_elements(self):
    expected_msg = (r"""test:

Structure 0:
A\(a=3, b=4\)

Structure 1:
A\(a=3, b=5\)

Exceptions:

Path: b
Exception: AssertionError
4 != 5""")

    namedtuple = collections.namedtuple('A', 'a, b')

    with self.assertRaisesRegexp(AssertionError, expected_msg):
      self.assertAllAssertsNested(
          self.assertEqual,
          namedtuple(a=3, b=4),
          namedtuple(a=3, b=5),
          msg='test')

  def test_assert_nested_shallow(self):
    with self.assertRaises(AssertionError):
      self.assertAllAssertsNested(lambda *args: True, [1, 2], 5)
    self.assertAllAssertsNested(lambda *args: True, [1, 2], 5, shallow=1)

  def test_assert_nested_check_types(self):
    with self.assertRaises(AssertionError):
      self.assertAllAssertsNested(lambda *args: True, [1, 2], (1, 2))
    self.assertAllAssertsNested(
        lambda *args: True, [1, 2], (1, 2), check_types=False)


@test_util.test_all_tf_execution_regimes
class TestCaseTestFloat32(_TestCaseTest, test_util.TestCase):
  dtype = tf.float32


@test_util.test_all_tf_execution_regimes
class TestCaseTestFloat64(_TestCaseTest, test_util.TestCase):
  dtype = tf.float64


#
# The are following are 'pretend' test case classes, which are actually examined
# in unit tests below, to verify (and document) the generated test names.
#
@test_util.test_all_tf_execution_regimes
class PretendTestCaseClass(test_util.TestCase):

  def test_snake_case_name(self):
    self.skipTest('Fake test')

  def testCamelCaseName(self):
    self.skipTest('Fake test')


@test_util.test_all_tf_execution_regimes
class PretendParameterizedTestCaseClass(test_util.TestCase):

  @parameterized.named_parameters([dict(testcase_name='p123', p='123')])
  def test_snake_case_name(self, p):
    del p  # avoid unused
    self.skipTest('Fake test')

  @parameterized.named_parameters([dict(testcase_name='p123', p='123')])
  def testCamelCaseName(self, p):
    del p  # avoid unused
    self.skipTest('Fake test')


@test_util.test_graph_and_eager_modes
class PretendTestCaseClassGraphAndEagerOnly(test_util.TestCase):

  def test_snake_case_name(self):
    self.skipTest('Fake test')

  def testCamelCaseName(self):
    self.skipTest('Fake test')


class TestCombinationsTest(test_util.TestCase):

  #
  # These tests check that the generated names are as expected.
  #
  def test_generated_test_case_names(self):
    expected_test_names = [
        'test_snake_case_name_eager_no_tf_function',
        'test_snake_case_name_eager',
        'test_snake_case_name_graph',
        'testCamelCaseName_eager_no_tf_function',
        'testCamelCaseName_eager',
        'testCamelCaseName_graph',
    ]

    for expected_test_name in expected_test_names:
      self.assertIn(expected_test_name, dir(PretendTestCaseClass))

  def test_generated_parameterized_test_case_names(self):
    expected_test_names = [
        'test_snake_case_name_p123_eager_no_tf_function',
        'test_snake_case_name_p123_eager',
        'test_snake_case_name_p123_graph',
        'testCamelCaseNamep123_eager_no_tf_function',
        'testCamelCaseNamep123_eager',
        'testCamelCaseNamep123_graph',
    ]

    for expected_test_name in expected_test_names:
      self.assertIn(expected_test_name, dir(PretendParameterizedTestCaseClass))

  def test_generated_graph_and_eager_test_case_names(self):
    expected_test_names = [
        'test_snake_case_name_eager',
        'test_snake_case_name_graph',
        'testCamelCaseName_eager',
        'testCamelCaseName_graph',
    ]

    for expected_test_name in expected_test_names:
      self.assertIn(expected_test_name,
                    dir(PretendTestCaseClassGraphAndEagerOnly))

  #
  # These tests ensure that the test generators do what they say on the tin.
  #
  @test_combinations.generate(
      test_combinations.combine(mode='graph'),
      test_combinations=[test_util.EagerGraphCombination()])
  def test_graph_mode_combination(self):
    self.assertFalse(context.executing_eagerly())

  @test_combinations.generate(
      test_combinations.combine(mode='eager'),
      test_combinations=[test_util.EagerGraphCombination()])
  def test_eager_mode_combination(self):
    self.assertTrue(context.executing_eagerly())

  @test_combinations.generate(
      test_combinations.combine(tf_function=''),
      test_combinations=[
          test_util.ExecuteFunctionsEagerlyCombination()])
  def test_tf_function_enabled_mode_combination(self):
    self.assertFalse(tf.config.experimental_functions_run_eagerly())

  @test_combinations.generate(
      test_combinations.combine(tf_function='no_tf_function'),
      test_combinations=[
          test_util.ExecuteFunctionsEagerlyCombination()])
  def test_tf_function_disabled_mode_combination(self):
    self.assertTrue(tf.config.experimental_functions_run_eagerly())


if __name__ == '__main__':
  tf.test.main()
