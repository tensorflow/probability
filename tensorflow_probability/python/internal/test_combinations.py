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
"""Decorators for testing TFP code under combinations of TF features."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import contextlib
import tensorflow.compat.v2 as tf

from tensorflow.python.eager import def_function  # pylint: disable=g-direct-tensorflow-import
from tensorflow.python.framework import combinations  # pylint: disable=g-direct-tensorflow-import
from tensorflow.python.framework import test_combinations  # pylint: disable=g-direct-tensorflow-import

__all__ = [
    'test_all_tf_execution_regimes',
    'test_graph_and_eager_modes',
]


@contextlib.contextmanager
def _tf_function_mode_context(tf_function_mode):
  """Context manager controlling `tf.function` behavior (enabled/disabled).

  Before activating, the previously set mode is stored. Then the mode is changed
  to the given `tf_function_mode` and control yielded back to the caller. Upon
  exiting the context, the mode is returned to its original state.

  Args:
    tf_function_mode: a Python `str`, either 'disabled' or 'enabled'. If
    'enabled', `@tf.function`-decorated code behaves as usual (ie, a background
    graph is created). If 'disabled', `@tf.function`-decorated code will behave
    as if it had not been `@tf.function`-decorated. Since users will be able to
    do this (e.g., to debug library code that has been
    `@tf.function`-decorated), we need to ensure our tests cover the behavior
    when this is the case.

  Yields:
    None
  """
  if tf_function_mode not in ['enabled', 'disabled']:
    raise ValueError(
        'Only allowable values for tf_function_mode_context are `enabled` and '
        '`disabled`; but got `{}`'.format(tf_function_mode))
  original_mode = def_function.RUN_FUNCTIONS_EAGERLY
  try:
    tf.config.experimental_run_functions_eagerly(tf_function_mode == 'disabled')
    yield
  finally:
    tf.config.experimental_run_functions_eagerly(original_mode)


class ExecuteFunctionsEagerlyCombination(test_combinations.TestCombination):
  """A `TestCombinationi` for enabling/disabling `tf.function` execution modes.

  For more on `TestCombination`, check out
  'tensorflow/python/framework/test_combinations.py' in the TensorFlow code
  base.

  This `TestCombination` supports two values for the `tf_function`
  combination argument: 'disabled' and 'enabled'. The mode switching is
  performed using `tf.experimental_run_functions_eagerly(mode)`.
  """

  def context_managers(self, kwargs):
    mode = kwargs.pop('tf_function', 'enabled')
    return [_tf_function_mode_context(mode)]

  def parameter_modifiers(self):
    return [test_combinations.OptionalParameter('tf_function')]


def test_all_tf_execution_regimes(test_class_or_method=None):
  """Decorator for generating a collection of tests in various contexts.

  Must be applied to subclasses of `parameterized.TestCase` (from
  `absl/testing`), or a method of such a subclass.

  When applied to a test method, this decorator results in the replacement of
  that method with a collection of new test methods, each executed under a
  different set of context managers that control some aspect of the execution
  model. This decorator generates three test scenario combinations:

    1. Eager mode with `tf.function` decorations enabled
    2. Eager mode with `tf.function` decorations disabled
    3. Graph mode (eveything)

  When applied to a test class, all the methods in the class are affected.

  Args:
    test_class_or_method: the `TestCase` class or method to decorate.

  Returns:
    decorator: A generated TF `test_combinations` decorator, or if
    `test_class_or_method` is not `None`, the generated decorator applied to
    that function.
  """
  decorator = test_combinations.generate(
      (test_combinations.combine(mode='graph',
                                 tf_function='enabled') +
       test_combinations.combine(mode='eager',
                                 tf_function=['enabled', 'disabled'])),
      test_combinations=[
          combinations.EagerGraphCombination(),
          ExecuteFunctionsEagerlyCombination(),
      ])

  if test_class_or_method:
    return decorator(test_class_or_method)
  return decorator


def test_graph_and_eager_modes(test_class_or_method=None):
  """Decorator for generating graph and eager mode tests from a single test.

  Must be applied to subclasses of `parameterized.TestCase` (from
  absl/testing), or a method of such a subclass.

  When applied to a test method, this decorator results in the replacement of
  that method with a two new test methods, one executed in graph mode and the
  other in eager mode.

  When applied to a test class, all the methods in the class are affected.

  Args:
    test_class_or_method: the `TestCase` class or method to decorate.

  Returns:
    decorator: A generated TF `test_combinations` decorator, or if
    `test_class_or_method` is not `None`, the generated decorator applied to
    that function.
  """
  decorator = test_combinations.generate(
      test_combinations.combine(mode=['graph', 'eager']),
      test_combinations=[combinations.EagerGraphCombination()])

  if test_class_or_method:
    return decorator(test_class_or_method)
  return decorator
