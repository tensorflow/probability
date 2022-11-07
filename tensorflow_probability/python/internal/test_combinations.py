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
"""Facilities for creating multiple test combinations.

Here is a simple example for testing various optimizers in Eager and Graph:

class AdditionExample(test.TestCase, parameterized.TestCase):
  @combinations.generate(
     combinations.combine(mode=['graph', 'eager'],
                          optimizer=[AdamOptimizer(),
                                     GradientDescentOptimizer()]))
  def testOptimizer(self, optimizer):
    ... f(optimizer)...

This will run `testOptimizer` 4 times with the specified optimizers: 2 in
Eager and 2 in Graph mode.
The test is going to accept the same parameters as the ones used in `combine()`.
The parameters need to match by name between the `combine()` call and the test
signature.  It is necessary to accept all parameters. See `OptionalParameter`
for a way to implement optional parameters.

`combine()` function is available for creating a cross product of various
options.

The execution of generated tests can be customized in a number of ways:
-  The test can be skipped if it is not running in the correct environment.
-  The arguments that are passed to the test can be additionaly transformed.
-  The test can be run with specific Python context managers.
These behaviors can customized by providing instances of `TestCombination` to
`generate()`.
"""

from collections import OrderedDict
import contextlib
import types
import unittest

# Dependency imports
from absl.testing import parameterized
import six

from tensorflow.python.util import tf_inspect  # pylint: disable=g-direct-tensorflow-import


class TestCombination(object):
  """Customize the behavior of `generate()` and the tests that it executes.

  Here is sequence of steps for executing a test combination:
    1. The test combination is evaluated for whether it should be executed in
       the given environment by calling `should_execute_combination`.
    2. If the test combination is going to be executed, then the arguments for
       all combined parameters are validated.  Some arguments can be handled in
       a special way.  This is achieved by implementing that logic in
       `ParameterModifier` instances that returned from `parameter_modifiers`.
    3. Before executing the test, `context_managers` are installed
       around it.
  """

  def should_execute_combination(self, kwargs):
    """Indicates whether the combination of test arguments should be executed.

    If the environment doesn't satisfy the dependencies of the test
    combination, then it can be skipped.

    Args:
      kwargs:  Arguments that are passed to the test combination.

    Returns:
      A tuple boolean and an optional string.  The boolean False indicates
    that the test should be skipped.  The string would indicate a textual
    description of the reason.  If the test is going to be executed, then
    this method returns `None` instead of the string.
    """
    del kwargs
    return (True, None)

  def parameter_modifiers(self):
    """Returns `ParameterModifier` instances that customize the arguments."""
    return []

  def context_managers(self, kwargs):
    """Return context managers for running the test combination.

    The test combination will run under all context managers that all
    `TestCombination` instances return.

    Args:
      kwargs:  Arguments and their values that are passed to the test
        combination.

    Returns:
      A list of instantiated context managers.
    """
    del kwargs
    return []


class ParameterModifier(object):
  """Customizes the behavior of a particular parameter."""

  DO_NOT_PASS_TO_THE_TEST = object()

  def __init__(self, parameter_name=None):
    """Construct a parameter modifier that may be specific to a parameter.

    Args:
      parameter_name:  A `ParameterModifier` instance may operate on a class of
        parameters or on a parameter with a particular name.  Only
        `ParameterModifier` instances that are of a unique type or were
        initialized with a unique `parameter_name` will be executed.
        See `__eq__` and `__hash__`.
    """
    object.__init__(self)
    self._parameter_name = parameter_name

  def modified_arguments(self, kwargs, requested_parameters):
    """Replace user-provided arguments before they are passed to a test.

    This makes it possible to adjust user-provided arguments before passing
    them to the test method.

    Args:
      kwargs:  The combined arguments for the test.
      requested_parameters: The set of parameters that are defined in the
        signature of the test method.

    Returns:
      A dictionary with updates to `kwargs`.  Keys with values set to
      `ParameterModifier.DO_NOT_PASS_TO_THE_TEST` are going to be deleted and
      not passed to the test.
    """
    del kwargs, requested_parameters
    return {}

  def __eq__(self, other):
    """Compare `ParameterModifier` by type and `parameter_name`."""
    if self is other:
      return True
    elif type(self) is type(other):
      return self._parameter_name == other._parameter_name
    else:
      return False

  def __ne__(self, other):
    return not self.__eq__(other)

  def __hash__(self):
    """Compare `ParameterModifier` by type or `parameter_name`."""
    if self._parameter_name:
      return hash(self._parameter_name)
    else:
      return id(self.__class__)


class OptionalParameter(ParameterModifier):
  """A parameter that is optional in `combine()` and in the test signature."""

  def modified_arguments(self, kwargs, requested_parameters):
    if self._parameter_name in requested_parameters:
      return {}
    else:
      return {self._parameter_name: ParameterModifier.DO_NOT_PASS_TO_THE_TEST}


def generate(combinations, test_combinations=()):
  """A decorator for generating combinations of a test method or a test class.

  Parameters of the test method must match by name to get the corresponding
  value of the combination.  Tests must accept all parameters that are passed
  other than the ones that are `OptionalParameter`.

  Args:
    combinations: a list of dictionaries created using combine().
    test_combinations: a tuple of `TestCombination` instances that customize
      the execution of generated tests.

  Returns:
    a decorator that will cause the test method or the test class to be run
    under the specified conditions.

  Raises:
    ValueError: if any parameters were not accepted by the test method
  """
  def decorator(test_method_or_class):
    """The decorator to be returned."""

    # Generate good test names that can be used with --test_filter.
    named_combinations = []
    for combination in combinations:
      # We use OrderedDicts in `combine()` to ensure stable order of keys in
      # each dictionary.
      assert isinstance(combination, OrderedDict)
      generated_testcase_name = '_'.join([
          str(value) for (_, value) in combination.items() if str(value)
      ])
      named_combinations.append(
          OrderedDict(
              list(combination.items()) +
              [('testcase_name', '_{}'.format(generated_testcase_name))]))

    if isinstance(test_method_or_class, type):
      class_object = test_method_or_class
      class_object._test_method_ids = test_method_ids = {}
      for name, test_method in six.iteritems(class_object.__dict__.copy()):
        if (name.startswith(unittest.TestLoader.testMethodPrefix) and
            isinstance(test_method, types.FunctionType)):
          delattr(class_object, name)
          methods = {}
          parameterized._update_class_dict_for_param_test_case(
              class_object.__name__, methods, test_method_ids, name,
              parameterized._ParameterizedTestIter(
                  _augment_with_special_arguments(
                      test_method, test_combinations=test_combinations),
                  named_combinations, parameterized._NAMED, name))
          for method_name, method in six.iteritems(methods):
            setattr(class_object, method_name, method)

      return class_object
    else:
      test_method = _augment_with_special_arguments(
          test_method_or_class, test_combinations=test_combinations)
      return parameterized.named_parameters(*named_combinations)(test_method)

  return decorator


def _augment_with_special_arguments(test_method, test_combinations):
  def decorated(self, **kwargs):
    """A wrapped test method that can treat some arguments in a special way."""
    original_kwargs = kwargs.copy()

    # Skip combinations that are going to be executed in a different testing
    # environment.
    reasons_to_skip = []
    for combination in test_combinations:
      should_execute, reason = combination.should_execute_combination(
          original_kwargs.copy())
      if not should_execute:
        reasons_to_skip.append(' - ' + reason)

    if reasons_to_skip:
      self.skipTest('\n'.join(reasons_to_skip))

    customized_parameters = []
    for combination in test_combinations:
      customized_parameters.extend(combination.parameter_modifiers())
    customized_parameters = set(customized_parameters)

    # The function for running the test under the total set of
    # `context_managers`:
    def execute_test_method():
      requested_parameters = tf_inspect.getfullargspec(test_method).args
      for customized_parameter in customized_parameters:
        for argument, value in customized_parameter.modified_arguments(
            original_kwargs.copy(), requested_parameters).items():
          if value is ParameterModifier.DO_NOT_PASS_TO_THE_TEST:
            kwargs.pop(argument, None)
          else:
            kwargs[argument] = value

      omitted_arguments = set(requested_parameters).difference(
          set(list(kwargs.keys()) + ['self']))
      if omitted_arguments:
        raise ValueError('The test requires parameters whose arguments '
                         'were not passed: {} .'.format(omitted_arguments))
      missing_arguments = set(list(kwargs.keys()) + ['self']).difference(
          set(requested_parameters))
      if missing_arguments:
        raise ValueError('The test does not take parameters that were passed '
                         ': {} .'.format(missing_arguments))

      kwargs_to_pass = {}
      for parameter in requested_parameters:
        if parameter == 'self':
          kwargs_to_pass[parameter] = self
        else:
          kwargs_to_pass[parameter] = kwargs[parameter]
      with self.cached_session():
        test_method(**kwargs_to_pass)

    # Install `context_managers` before running the test:
    context_managers = []
    for combination in test_combinations:
      for manager in combination.context_managers(
          original_kwargs.copy()):
        context_managers.append(manager)

    with contextlib.ExitStack() as context_stack:
      for manager in context_managers:
        context_stack.enter_context(manager)
      execute_test_method()

  return decorated


def combine(**kwargs):
  """Generate list of combinations based on keyword arguments.

  Args:
    **kwargs: keyword arguments of form `option=[possibilities, ...]`
         or `option=the_only_possibility`.

  Returns:
    a list of dictionaries for each combination. Keys in the dictionaries are
    the keyword argument names.  Each key has one value - one of the
    corresponding keyword argument values.
  """
  if not kwargs:
    return [OrderedDict()]

  sort_by_key = lambda k: k[0]
  kwargs = OrderedDict(sorted(kwargs.items(), key=sort_by_key))
  first = list(kwargs.items())[0]

  rest = dict(list(kwargs.items())[1:])
  rest_combined = combine(**rest)

  key = first[0]
  values = first[1]
  if not isinstance(values, list):
    values = [values]

  return [
      OrderedDict(sorted(list(combined.items()) + [(key, v)], key=sort_by_key))
      for v in values
      for combined in rest_combined
  ]
