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
"""Utilities for testing TFP code."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import contextlib
import functools
import os
import unittest

from absl import flags
from absl import logging
from absl.testing import parameterized
import numpy as np
import six
import tensorflow.compat.v1 as tf1
import tensorflow.compat.v2 as tf
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import test_combinations
from tensorflow_probability.python.internal.backend.numpy import ops
from tensorflow_probability.python.util.seed_stream import SeedStream
from tensorflow.python.eager import context  # pylint: disable=g-direct-tensorflow-import
from tensorflow.python.ops import gradient_checker_v2  # pylint: disable=g-direct-tensorflow-import
from tensorflow.python.util import nest  # pylint: disable=g-direct-tensorflow-import


__all__ = [
    'substrate_disable_stateful_random_test',
    'numpy_disable_gradient_test',
    'numpy_disable_variable_test',
    'jax_disable_variable_test',
    'jax_disable_test_missing_functionality',
    'disable_test_for_backend',
    'test_all_tf_execution_regimes',
    'test_graph_and_eager_modes',
    'test_graph_mode_only',
    'test_seed',
    'test_seed_stream',
    'floats_near',
    'DiscreteScalarDistributionTestHelpers',
    'TestCase',
    'VectorDistributionTestHelpers',
]


JAX_MODE = False
NUMPY_MODE = False

# Flags for controlling test_teed behavior.
flags.DEFINE_bool('vary_seed', False,
                  ('Whether to vary the PRNG seed unpredictably.  '
                   'With --runs_per_test=N, produces N iid runs.'),
                  allow_override=True)

flags.DEFINE_string('fixed_seed', None,
                    ('PRNG seed to initialize every test with.  '
                     'Takes precedence over --vary-seed when both appear.'),
                    allow_override=True,
                    allow_override_cpp=False,
                    allow_hide_cpp=True)


class TestCase(tf.test.TestCase, parameterized.TestCase):
  """Class to provide TensorFlow Probability specific test features."""

  def maybe_static(self, x, is_static):
    """If `not is_static`, return placeholder_with_default with unknown shape.

    Args:
      x: A `Tensor`
      is_static: a Python `bool`; if True, x is returned unchanged. If False, x
        is wrapped with a tf1.placeholder_with_default with fully dynamic shape.

    Returns:
      maybe_static_x: `x`, possibly wrapped with in a
      `placeholder_with_default` of unknown shape.
    """
    if is_static:
      return x
    else:
      return tf1.placeholder_with_default(x, shape=None)

  def assertAllAssertsNested(self, assert_fn, *structure, **kwargs):
    """Run `assert_fn` on `structure` and report which elements errored.

    This function will run `assert_fn` on each element of `structure` as
    `assert_fn(structure[0], structure[1], ...)`, collecting any exceptions
    raised in the process. Afterward, it will report which elements of
    `structure` triggered an assertion, as well as the assertions themselves.

    Args:
      assert_fn: A callable that accepts as many arguments as there are
        structures.
      *structure: A list of nested structures.
      **kwargs: Valid keyword args are:

        * `shallow`: If not None, uses this as the shared tree prefix of
          `structure` for the purpose of being able to use `structure` which
          only share that tree prefix (e.g. `[1, 2]` and `[[1], 2]` share the
          `[., .]` tree prefix).
        * `msg`: Used as the message when a failure happened. Default:
          `"AllAssertsNested failed"`.
        * `check_types`: If `True`, types of sequences are checked as well,
          including the keys of dictionaries. If `False`, for example a list and
          a tuple of objects may be equivalent. Default: `False`.

    Raises:
      AssertionError: If the structures are mismatched, or at `assert_fn` raised
        an exception at least once.
    """
    shallow = kwargs.pop('shallow', None)
    if shallow is None:
      shallow = structure[0]
    msg = kwargs.pop('msg', 'AllAssertsNested failed')

    def _one_part(*structure):
      try:
        assert_fn(*structure)
      except Exception as part_e:  # pylint: disable=broad-except
        return part_e

    try:
      maybe_exceptions = nest.map_structure_up_to(shallow, _one_part,
                                                  *structure, **kwargs)
      overall_exception = None
      exceptions_with_paths = [
          (p, e)
          for p, e in nest.flatten_with_joined_string_paths(maybe_exceptions)
          if e is not None
      ]
    except Exception as e:  # pylint: disable=broad-except
      overall_exception = e
      exceptions_with_paths = []

    final_msg = '{}:\n\n'.format(msg)
    if overall_exception:
      final_msg += str(overall_exception)
      raise AssertionError(final_msg)
    if exceptions_with_paths:
      for i, one_structure in enumerate(structure):
        final_msg += 'Structure {}:\n{}\n\n'.format(i, one_structure)
      final_msg += 'Exceptions:\n\n'
      for p, exception in exceptions_with_paths:
        final_msg += 'Path: {}\nException: {}\n{}\n\n'.format(
            p,
            type(exception).__name__, exception)
      # Drop the final two newlines.
      raise AssertionError(final_msg[:-2])

  def assertAllEqualNested(self, a, b, check_types=False, shallow=None):
    """Assert that analogous entries in two nested structures are equivalent.

    Args:
      a: A nested structure.
      b: A nested structure.
      check_types: If `True`, types of sequences are checked as well, including
        the keys of dictionaries. If `False`, for example a list and a tuple of
        objects may be equivalent.
      shallow: If not None, uses this as the shared tree prefix of `a` and `b`
        for the purpose of being able to use `a` and `b` which only share that
        tree prefix (e.g. `[1, 2]` and `[[1], 2]` share the `[., .]` tree
        prefix).
    """
    self.assertAllAssertsNested(
        self.assertAllEqual,
        a,
        b,
        check_types=check_types,
        msg='AllEqualNested failed',
        shallow=shallow)

  def assertAllCloseNested(
      self, a, b, rtol=1e-06, atol=1e-06, check_types=False):
    """Assert that analogous entries in two nested structures have near values.

    Args:
      a: A nested structure.
      b: A nested structure.
      rtol: scalar relative tolerance.
        Default value: `1e-6`.
      atol: scalar absolute tolerance.
        Default value: `1e-6`.
      check_types: If `True`, types of sequences are checked as well, including
        the keys of dictionaries. If `False`, for example a list and a tuple of
        objects may be equivalent.
    """
    self.assertAllAssertsNested(
        lambda x, y: self.assertAllClose(x, y, rtol=rtol, atol=atol),
        a,
        b,
        check_types=check_types,
        msg='AllCloseNested failed')

  def assertAllTrue(self, a):
    """Assert that all entries in a boolean `Tensor` are True."""
    a_ = self._GetNdArray(a)
    all_true = np.ones_like(a_, dtype=np.bool_)
    self.assertAllEqual(all_true, a_)

  def assertAllFalse(self, a):
    """Assert that all entries in a boolean `Tensor` are False."""
    a_ = self._GetNdArray(a)
    all_false = np.zeros_like(a_, dtype=np.bool_)
    self.assertAllEqual(all_false, a_)

  def assertAllFinite(self, a):
    """Assert that all entries in a `Tensor` are finite.

    Args:
      a: A `Tensor` whose entries are checked for finiteness.
    """
    is_finite = np.isfinite(self._GetNdArray(a))
    all_true = np.ones_like(is_finite, dtype=np.bool_)
    self.assertAllEqual(all_true, is_finite)

  def assertAllPositiveInf(self, a):
    """Assert that all entries in a `Tensor` are equal to positive infinity.

    Args:
      a: A `Tensor` whose entries must be verified as positive infinity.
    """
    is_positive_inf = np.isposinf(self._GetNdArray(a))
    all_true = np.ones_like(is_positive_inf, dtype=np.bool_)
    self.assertAllEqual(all_true, is_positive_inf)

  def assertAllNegativeInf(self, a):
    """Assert that all entries in a `Tensor` are negative infinity.

    Args:
      a: A `Tensor` whose entries must be verified as negative infinity.
    """
    is_negative_inf = np.isneginf(self._GetNdArray(a))
    all_true = np.ones_like(is_negative_inf, dtype=np.bool_)
    self.assertAllEqual(all_true, is_negative_inf)

  def assertNotAllZero(self, a):
    """Assert that all entries in a `Tensor` are nonzero.

    Args:
      a: A `Tensor` whose entries must be verified as nonzero.
    """
    self.assertNotAllEqual(a, tf.nest.map_structure(tf.zeros_like, a))

  def assertAllNotNan(self, a):
    """Assert that every entry in a `Tensor` is not NaN.

    Args:
      a: A `Tensor` whose entries must be verified as not NaN.
    """
    is_not_nan = ~np.isnan(self._GetNdArray(a))
    all_true = np.ones_like(is_not_nan, dtype=np.bool_)
    self.assertAllEqual(all_true, is_not_nan)

  def assertAllNan(self, a):
    """Assert that every entry in a `Tensor` is NaN.

    Args:
      a: A `Tensor` whose entries must be verified as NaN.
    """
    is_nan = np.isnan(self._GetNdArray(a))
    all_true = np.ones_like(is_nan, dtype=np.bool_)
    self.assertAllEqual(all_true, is_nan)

  def assertAllNotNone(self, a):
    """Assert that no entry in a collection is None.

    Args:
      a: A Python iterable collection, whose entries must be verified as not
      being `None`.
    """
    each_not_none = [x is not None for x in a]
    if all(each_not_none):
      return

    msg = (
        'Expected no entry to be `None` but found `None` in positions {}'
        .format([i for i, x in enumerate(each_not_none) if not x]))
    raise AssertionError(msg)

  def assertAllIs(self, a, b):
    """Assert that each element of `a` `is` `b`.

    Args:
      a: A Python iterable collection, whose entries must be elementwise `is b`.
      b: A Python iterable collection, whose entries must be elementwise `is a`.
    """
    if len(a) != len(b):
      raise AssertionError(
          'Arguments `a` and `b` must have the same number of elements '
          'but found len(a)={} and len(b)={}.'.format(len(a), len(b)))
    each_is = [a is b for a, b in zip(a, b)]
    if all(each_is):
      return
    msg = (
        'For each element expected `a is b` but found `not is` in positions {}'
        .format([i for i, x in enumerate(each_is) if not x]))
    raise AssertionError(msg)

  def evaluate_dict(self, dictionary):
    """Invokes `self.evaluate` on the `Tensor`s in `dictionary`.

    Reconstructs the results as a dictionary with the same keys and values.
    Leaves non-`Tensor` values alone (lest `self.evaluate` fail on them).

    This can be useful to debug Hypothesis examples, with
    `hp.note(self.evaluate_dict(dist.parameters()))`.  The standard
    `self.evaluate` can fail if the `parameters` dictionary contains
    non-`Tensor` values (which it typically does).

    Args:
      dictionary: Dictionary to traverse.

    Returns:
      result: Dictionary with the same keys, but with `Tensor` values
        replaced by the results of `self.evaluate`.
    """
    python_values = {}
    tensor_values = {}
    for k, v in dictionary.items():
      if tf.is_tensor(v):
        tensor_values[k] = v
      else:
        python_values[k] = v
    return dict(self.evaluate(tensor_values), **python_values)

  def compute_max_gradient_error(self, f, args, delta=1e-3):
    """Wrapper around TF's gradient_checker_v2.

    `gradient_checker_v2` depends on there being a default session, but our test
    setup, using test_combinations, doesn't run the test function under a global
    `self.test_session()` context. Thus, when running
    `gradient_checker_v2.compute_gradient`, we need to ensure we're in a
    `self.test_session()` context when not in eager mode. This function bundles
    up the relevant logic, and ultimately returns the max error across autodiff
    and finite difference gradient calculations.

    Args:
      f: callable function whose gradient to compute.
      args: Python `list` of independent variables with respect to which to
      compute gradients.
      delta: floating point value to use for finite difference calculation.

    Returns:
      err: the maximum error between all components of the numeric and
      autodiff'ed gradients.
    """
    if JAX_MODE:
      return _compute_max_gradient_error_jax(f, args, delta)
    def _compute_error():
      return gradient_checker_v2.max_error(
          *gradient_checker_v2.compute_gradient(f, x=args, delta=delta))
    if tf.executing_eagerly():
      return _compute_error()
    else:
      # Make sure there's a global default session in graph mode.
      with self.test_session():
        return _compute_error()

  def skip_if_no_xla(self):
    try:
      tf.function(lambda: tf.constant(0), jit_compile=True)()
    except (tf.errors.UnimplementedError, NotImplementedError) as e:
      if 'Could not find compiler' in str(e):
        self.skipTest('XLA not available')

  def make_input(self, number):
    """Create inputs with varied dtypes and static or dynamic shape.

    Helper to run tests with different dtypes and both statically defined and
    not statically defined shapes. This helper wraps the inputs to a
    distribution, typically a python literal or numpy array. It will then
    attach shapes and dtypes as specified by the test class's `dtype` and
    `use_static_shape` attributes.

    Note: `tf.Variables` are used to represent inputs which don't have
    statically defined shapes, as well as fp64 inputs with statically defined
    shapes. For fp32 with statically defined shapes, inputs are wrapped with
    `tf.convert_to_tensor`.

    Args:
      number: Input value(s), typically python literal, list, or numpy array.

    Returns:
      output: Tensor or Variable with new dtype and shape.
    """
    try:
      num_shape = number.shape
    except AttributeError:
      num_shape = None
    target_shape = num_shape if self.use_static_shape else None
    if self.dtype in [np.float64, tf.float64] or not self.use_static_shape:
      output = tf.Variable(number, dtype=self.dtype, shape=target_shape,)
      self.evaluate(output.initializer)
    elif self.dtype in [np.float32, tf.float32] and self.use_static_shape:
      output = tf.convert_to_tensor(number, dtype=self.dtype)
    else:
      raise TypeError('Only float32 and float64 supported, got',
                      str(self.dtype))
    return output

if JAX_MODE:
  from jax import jacrev  # pylint: disable=g-import-not-at-top
  from jax import vmap  # pylint: disable=g-import-not-at-top

  def _compute_max_gradient_error_jax(f, xs, scale=1e-3):
    f_jac = jacrev(f, argnums=range(len(xs)))
    theoretical_jacobian = f_jac(*xs)
    numerical_jacobian = [_compute_numerical_jacobian_jax(f, xs, i, scale)
                          for i in range(len(xs))]
    return np.max(np.array(
        [np.max(np.abs(a - b))
         for (a, b) in zip(theoretical_jacobian, numerical_jacobian)]))

  def _compute_numerical_jacobian_jax(f, xs, i, scale=1e-3):
    dtype_i = xs[i].dtype
    shape_i = xs[i].shape
    size_i = np.product(shape_i, dtype=np.int32)
    def grad_i(d):
      return (f(*(xs[:i] + [xs[i] + d * scale] + xs[i+1:]))
              - f(*(xs[:i] + [xs[i] - d * scale] + xs[i+1:]))) / (2. * scale)
    ret = vmap(grad_i)(
        np.eye(size_i, dtype=dtype_i).reshape((size_i,) + shape_i))
    ret = np.moveaxis(ret, 0, -1)
    return np.reshape(ret, ret.shape[:-1] + shape_i)


@contextlib.contextmanager
def _tf_function_mode_context(tf_function_mode):
  """Context manager controlling `tf.function` behavior (enabled/disabled).

  Before activating, the previously set mode is stored. Then the mode is changed
  to the given `tf_function_mode` and control yielded back to the caller. Upon
  exiting the context, the mode is returned to its original state.

  Args:
    tf_function_mode: a Python `str`, either 'no_tf_function' or ''.
    If '', `@tf.function`-decorated code behaves as usual (ie, a
    background graph is created). If 'no_tf_function', `@tf.function`-decorated
    code will behave as if it had not been `@tf.function`-decorated. Since users
    will be able to do this (e.g., to debug library code that has been
    `@tf.function`-decorated), we need to ensure our tests cover the behavior
    when this is the case.

  Yields:
    None
  """
  if tf_function_mode not in ['', 'no_tf_function']:
    raise ValueError(
        'Only allowable values for tf_function_mode_context are "" '
        'and "no_tf_function"; but got "{}"'.format(tf_function_mode))
  original_mode = tf.config.functions_run_eagerly()
  try:
    tf.config.run_functions_eagerly(tf_function_mode == 'no_tf_function')
    yield
  finally:
    tf.config.run_functions_eagerly(original_mode)


class EagerGraphCombination(test_combinations.TestCombination):
  """Run the test in Graph or Eager mode.  Graph is the default.

  The optional `mode` parameter controls the test's execution mode.  Its
  accepted values are "graph" or "eager" literals.
  """

  def context_managers(self, kwargs):
    # TODO(isaprykin): Switch the default to eager.
    mode = kwargs.pop('mode', 'graph')
    if mode == 'eager':
      return [context.eager_mode()]
    elif mode == 'graph':
      return [tf1.Graph().as_default(), context.graph_mode()]
    else:
      raise ValueError(
          '`mode` must be "eager" or "graph". Got: "{}"'.format(mode))

  def parameter_modifiers(self):
    return [test_combinations.OptionalParameter('mode')]


class ExecuteFunctionsEagerlyCombination(test_combinations.TestCombination):
  """A `TestCombinationi` for enabling/disabling `tf.function` execution modes.

  For more on `TestCombination`, check out
  'tensorflow/python/framework/test_combinations.py' in the TensorFlow code
  base.

  This `TestCombination` supports two values for the `tf_function` combination
  argument: 'no_tf_function' and ''. The mode switching is performed using
  `tf.experimental_run_functions_eagerly(mode)`.
  """

  def context_managers(self, kwargs):
    mode = kwargs.pop('tf_function', '')
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
                                 tf_function='') +
       test_combinations.combine(
           mode='eager', tf_function=['', 'no_tf_function'])),
      test_combinations=[
          EagerGraphCombination(),
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
      test_combinations=[EagerGraphCombination()])

  if test_class_or_method:
    return decorator(test_class_or_method)
  return decorator


def test_graph_mode_only(test_class_or_method=None):
  """Decorator for ensuring tests run in graph mode.

  Must be applied to subclasses of `parameterized.TestCase` (from
  absl/testing), or a method of such a subclass.

  When applied to a test method, this decorator results in the replacement of
  that method with one new test method, executed in graph mode.

  When applied to a test class, all the methods in the class are affected.

  Args:
    test_class_or_method: the `TestCase` class or method to decorate.

  Returns:
    decorator: A generated TF `test_combinations` decorator, or if
    `test_class_or_method` is not `None`, the generated decorator applied to
    that function.
  Raises:
    SkipTest: Raised when not running in the TF backend.
  """
  if JAX_MODE or NUMPY_MODE:
    raise unittest.SkipTest('Ignoring TF Graph Mode tests in non-TF backends.')

  decorator = test_combinations.generate(
      test_combinations.combine(mode=['graph']),
      test_combinations=[EagerGraphCombination()])

  if test_class_or_method:
    return decorator(test_class_or_method)
  return decorator


def is_numpy_not_jax_mode():
  return NUMPY_MODE and not JAX_MODE


def numpy_disable_gradient_test(test_fn_or_reason, reason=None):
  """Disable a gradient-using test when using the numpy backend."""

  if not callable(test_fn_or_reason):
    if reason is not None:
      raise ValueError('Unexpected test_fn: {}'.format(test_fn_or_reason))
    return functools.partial(numpy_disable_gradient_test,
                             reason=test_fn_or_reason)

  if not NUMPY_MODE:
    return test_fn_or_reason

  def new_test(self, *args, **kwargs):  # pylint: disable=unused-argument
    self.skipTest('gradient-using test disabled for numpy{}'.format(
        ': {}'.format(reason) if reason else ''))

  return new_test


def numpy_disable_variable_test(test_fn):
  """Disable a Variable-using test when using the numpy backend."""

  if not NUMPY_MODE:
    return test_fn

  def new_test(self, *args, **kwargs):
    self.skipTest('tf.Variable-using test disabled for numpy')
    return test_fn(self, *args, **kwargs)

  return new_test


def jax_disable_variable_test(test_fn):
  """Disable a Variable-using test when using the JAX backend."""

  if not JAX_MODE:
    return test_fn

  def new_test(self, *args, **kwargs):
    self.skipTest('tf.Variable-using test disabled for JAX')
    return test_fn(self, *args, **kwargs)

  return new_test


def substrate_disable_stateful_random_test(test_fn):
  """Disable a test of stateful randomness."""

  def new_test(self, *args, **kwargs):
    if not hasattr(tf.random, 'uniform'):
      self.skipTest('Test uses stateful random sampling')
    return test_fn(self, *args, **kwargs)

  return new_test


def numpy_disable_test_missing_functionality(issue_link):
  """Disable a test for unimplemented numpy functionality."""

  def f(test_fn_or_class):
    """Decorator."""
    if JAX_MODE:
      return test_fn_or_class
    if tf.Variable != ops.NumpyVariable:
      return test_fn_or_class

    reason = 'Test disabled for Numpy missing functionality: {}'.format(
        issue_link)

    if isinstance(test_fn_or_class, type):
      return unittest.skip(reason)(test_fn_or_class)

    def new_test(self, *args, **kwargs):
      self.skipTest(reason)
      return test_fn_or_class(self, *args, **kwargs)

    return new_test

  return f


def jax_disable_test_missing_functionality(issue_link):
  """Disable a test for unimplemented JAX functionality."""

  def f(test_fn_or_class):
    if not JAX_MODE:
      return test_fn_or_class

    reason = 'Test disabled for JAX missing functionality: {}'.format(
        issue_link)

    if isinstance(test_fn_or_class, type):
      return unittest.skip(reason)(test_fn_or_class)

    def new_test(self, *args, **kwargs):
      self.skipTest(reason)
      return test_fn_or_class(self, *args, **kwargs)

    return new_test

  return f


def tf_tape_safety_test(test_fn):
  """Only run a test of TF2 tape safety against the TF backend."""

  def new_test(self, *args, **kwargs):
    if JAX_MODE or (tf.Variable == ops.NumpyVariable):
      self.skipTest('Tape-safety tests are only run against TensorFlow.')
    return test_fn(self, *args, **kwargs)

  return new_test


def disable_test_for_backend(disable_numpy=False,
                             disable_jax=False,
                             reason=None):
  """Disable a test for backends with a specified reason."""
  if not (disable_numpy or disable_jax):
    raise ValueError('One of `disable_numpy` or `disable_jax` must be true.')
  if not reason:
    raise ValueError('`reason` must be specified.')

  def decor(test_fn_or_class):
    if not ((disable_numpy and is_numpy_not_jax_mode()) or
            (disable_jax and JAX_MODE)):
      return test_fn_or_class
    return unittest.skip(reason)(test_fn_or_class)

  return decor


def test_seed(hardcoded_seed=None,
              set_eager_seed=True,
              sampler_type='stateful'):
  """Returns a command-line-controllable PRNG seed for unit tests.

  If your test will pass a seed to more than one operation, consider using
  `test_seed_stream` instead.

  When seeding unit-test PRNGs, we want:

  - The seed to be fixed to an arbitrary value most of the time, so the test
    doesn't flake even if its failure probability is noticeable.

  - To switch to different seeds per run when using --runs_per_test to measure
    the test's failure probability.

  - To set the seed to a specific value when reproducing a low-probability event
    (e.g., debugging a crash that only some seeds trigger).

  To those ends, this function returns 17, but respects the command line flags
  `--fixed_seed=<seed>` and `--vary_seed` (Boolean, default False).
  `--vary_seed` uses system entropy to produce unpredictable seeds.
  `--fixed_seed` takes precedence over `--vary_seed` when both are present.

  Note that TensorFlow graph mode operations tend to read seed state from two
  sources: a "graph-level seed" and an "op-level seed".  test_util.TestCase will
  set the former to a fixed value per test, but in general it may be necessary
  to explicitly set both to ensure reproducibility.

  Args:
    hardcoded_seed: Optional Python value.  The seed to use instead of 17 if
      both the `--vary_seed` and `--fixed_seed` flags are unset.  This should
      usually be unnecessary, since a test should pass with any seed.
    set_eager_seed: Python bool.  If true (default), invoke `tf.random.set_seed`
      in Eager mode to get more reproducibility.  Should become unnecessary
      once b/68017812 is resolved.
    sampler_type: 'stateful' or 'stateless'. 'stateless' means we return a seed
      pair.

  Returns:
    seed: 17, unless otherwise specified by arguments or command line flags.
  """
  if flags.FLAGS.fixed_seed is not None:
    answer = int(flags.FLAGS.fixed_seed)
  elif flags.FLAGS.vary_seed:
    entropy = os.urandom(64)
    # Why does Python make it so hard to just grab a bunch of bytes from
    # /dev/urandom and get them interpreted as an integer?  Oh, well.
    if six.PY2:
      answer = int(entropy.encode('hex'), 16)
    else:
      answer = int.from_bytes(entropy, 'big')
    logging.warning('Using seed %s', answer)
  elif hardcoded_seed is not None:
    answer = hardcoded_seed
    if JAX_MODE and np.shape(answer) == (2,):
      # Workaround for test_seed(hardcoded_seed=test_seed()), which can happen
      # e.g. with the run_test_sample_consistent_log_prob methods above.
      answer = answer[-1]
  else:
    answer = 17
  if sampler_type == 'stateless' or JAX_MODE:
    answer = tf.constant([0, answer % (2**32 - 1)], dtype=tf.uint32)
    if not JAX_MODE:
      answer = tf.bitcast(answer, tf.int32)
  # TODO(b/68017812): Remove this clause once eager correctly supports seeding.
  elif tf.executing_eagerly() and set_eager_seed:
    tf.random.set_seed(answer)
  return answer


def test_seed_stream(salt='Salt of the Earth', hardcoded_seed=None):
  """Returns a command-line-controllable SeedStream PRNG for unit tests.

  When seeding unit-test PRNGs, we want:

  - The seed to be fixed to an arbitrary value most of the time, so the test
    doesn't flake even if its failure probability is noticeable.

  - To switch to different seeds per run when using --runs_per_test to measure
    the test's failure probability.

  - To set the seed to a specific value when reproducing a low-probability event
    (e.g., debugging a crash that only some seeds trigger).

  To those ends, this function returns a `SeedStream` seeded with `test_seed`
  (which see).  The latter respects the command line flags `--fixed_seed=<seed>`
  and `--vary_seed` (Boolean, default False).  `--vary_seed` uses system entropy
  to produce unpredictable seeds.  `--fixed_seed` takes precedence over
  `--vary_seed` when both are present.

  Note that TensorFlow graph mode operations tend to read seed state from two
  sources: a "graph-level seed" and an "op-level seed".  test_util.TestCase will
  set the former to a fixed value per test, but in general it may be necessary
  to explicitly set both to ensure reproducibility.

  Args:
    salt: Optional string wherewith to salt the returned SeedStream.  Setting
      this guarantees independent random numbers across tests.
    hardcoded_seed: Optional Python value.  The seed to use if both the
      `--vary_seed` and `--fixed_seed` flags are unset.  This should usually be
      unnecessary, since a test should pass with any seed.

  Returns:
    strm: A SeedStream instance seeded with 17, unless otherwise specified by
      arguments or command line flags.
  """
  return SeedStream(test_seed(hardcoded_seed), salt=salt)


def test_np_rng(hardcoded_seed=None):
  """Returns a command-line-controllable Numpy PRNG for unit tests.

  When seeding unit-test PRNGs, we want:

  - The seed to be fixed to an arbitrary value most of the time, so the test
    doesn't flake even if its failure probability is noticeable.

  - To switch to different seeds per run when using --runs_per_test to measure
    the test's failure probability.

  - To set the seed to a specific value when reproducing a low-probability event
    (e.g., debugging a crash that only some seeds trigger).

  To those ends, this function returns a `np.random.RandomState` seeded with
  `test_seed` (which see).  The latter respects the command line flags
  `--fixed_seed=<seed>` and `--vary_seed` (Boolean, default False).
  `--vary_seed` uses system entropy to produce unpredictable seeds.
  `--fixed_seed` takes precedence over `--vary_seed` when both are present.

  Args:
    hardcoded_seed: Optional Python value.  The seed to use if both the
      `--vary_seed` and `--fixed_seed` flags are unset.  This should usually be
      unnecessary, since a test should pass with any seed.

  Returns:
    rng: A `np.random.RandomState` instance seeded with 17, unless otherwise
      specified by arguments or command line flags.
  """
  raw_seed = test_seed(hardcoded_seed=hardcoded_seed)
  # Jax backend doesn't have the random module; but it shouldn't be needed,
  # because this helper should only be used to generate test data.
  return np.random.RandomState(seed=raw_seed % 2**32)


def floats_near(target, how_many, dtype=np.float32):
  """Returns all the floats nearest the given target.

  This is useful for brute-force testing for situations where round-off errors
  may violate software invariants (e.g., interpolation result falling outside
  the interval being interpolated into).

  This implementation may itself have numerical infelicities, so may contain
  gaps and duplicates, but should be pretty good for non-zero (and non-denormal)
  targets.

  Args:
    target: Float near which to produce candidates.
    how_many: How many candidates to produce.
    dtype: The floating point type of outputs to emit.  The returned values
      are supposed to densely cover the space of floats representable in this
      dtype near the target.

  Returns:
    floats: A 1-D numpy array of `how_many` floats of the requested type,
      densely covering the space of representable floats near `target`.
  """
  eps = np.finfo(dtype).eps
  offset = eps * how_many / 2
  return np.linspace(target * (1. - offset), target * (1. + offset),
                     how_many, dtype=dtype)


class DiscreteScalarDistributionTestHelpers(object):
  """DiscreteScalarDistributionTestHelpers."""

  def run_test_sample_consistent_log_prob(
      self, sess_run_fn, dist,
      num_samples=int(1e5), num_threshold=int(1e3), seed=None,
      batch_size=None,
      rtol=1e-2, atol=0.):
    """Tests that sample/log_prob are consistent with each other.

    "Consistency" means that `sample` and `log_prob` correspond to the same
    distribution.

    Note: this test only verifies a necessary condition for consistency--it does
    does not verify sufficiency hence does not prove `sample`, `log_prob` truly
    are consistent.

    Args:
      sess_run_fn: Python `callable` taking `list`-like of `Tensor`s and
        returning a list of results after running one "step" of TensorFlow
        computation, typically set to `sess.run`.
      dist: Distribution instance or object which implements `sample`,
        `log_prob`, `event_shape_tensor` and `batch_shape_tensor`.
      num_samples: Python `int` scalar indicating the number of Monte-Carlo
        samples to draw from `dist`.
      num_threshold: Python `int` scalar indicating the number of samples a
        bucket must contain before being compared to the probability.
        Default value: 1e3; must be at least 1.
        Warning, set too high will cause test to falsely pass but setting too
        low will cause the test to falsely fail.
      seed: Python `int` indicating the seed to use when sampling from `dist`.
        In general it is not recommended to use `None` during a test as this
        increases the likelihood of spurious test failure.
      batch_size: Hint for unpacking result of samples. Default: `None` means
        batch_size is inferred.
      rtol: Python `float`-type indicating the admissible relative error between
        analytical and sample statistics.
      atol: Python `float`-type indicating the admissible absolute error between
        analytical and sample statistics.

    Raises:
      ValueError: if `num_threshold < 1`.
    """
    if num_threshold < 1:
      raise ValueError('num_threshold({}) must be at least 1.'.format(
          num_threshold))
    # Histogram only supports vectors so we call it once per batch coordinate.
    y = dist.sample(num_samples, seed=test_seed_stream(hardcoded_seed=seed))
    y = tf.reshape(y, shape=[num_samples, -1])
    if batch_size is None:
      batch_size = tf.reduce_prod(dist.batch_shape_tensor())
    batch_dims = tf.shape(dist.batch_shape_tensor())[0]
    edges_expanded_shape = 1 + tf.pad(tensor=[-2], paddings=[[0, batch_dims]])
    for b, x in enumerate(tf.unstack(y, num=batch_size, axis=1)):
      counts, edges = self.histogram(x)
      edges = tf.reshape(edges, edges_expanded_shape)
      probs = tf.exp(dist.log_prob(edges))
      probs = tf.reshape(probs, shape=[-1, batch_size])[:, b]

      [counts_, probs_] = sess_run_fn([counts, probs])
      valid = counts_ > num_threshold
      probs_ = probs_[valid]
      counts_ = counts_[valid]
      self.assertAllClose(probs_, counts_ / num_samples,
                          rtol=rtol, atol=atol)

  def run_test_sample_consistent_mean_variance(
      self, sess_run_fn, dist,
      num_samples=int(1e5), seed=None,
      rtol=1e-2, atol=0.):
    """Tests that sample/mean/variance are consistent with each other.

    "Consistency" means that `sample`, `mean`, `variance`, etc all correspond
    to the same distribution.

    Args:
      sess_run_fn: Python `callable` taking `list`-like of `Tensor`s and
        returning a list of results after running one "step" of TensorFlow
        computation, typically set to `sess.run`.
      dist: Distribution instance or object which implements `sample`,
        `log_prob`, `event_shape_tensor` and `batch_shape_tensor`.
      num_samples: Python `int` scalar indicating the number of Monte-Carlo
        samples to draw from `dist`.
      seed: Python `int` indicating the seed to use when sampling from `dist`.
        In general it is not recommended to use `None` during a test as this
        increases the likelihood of spurious test failure.
      rtol: Python `float`-type indicating the admissible relative error between
        analytical and sample statistics.
      atol: Python `float`-type indicating the admissible absolute error between
        analytical and sample statistics.
    """
    x = tf.cast(dist.sample(num_samples,
                            seed=test_seed_stream(hardcoded_seed=seed)),
                dtype=tf.float32)
    sample_mean = tf.reduce_mean(x, axis=0)
    sample_variance = tf.reduce_mean(tf.square(x - sample_mean), axis=0)
    sample_stddev = tf.sqrt(sample_variance)

    [
        sample_mean_,
        sample_variance_,
        sample_stddev_,
        mean_,
        variance_,
        stddev_
    ] = sess_run_fn([
        sample_mean,
        sample_variance,
        sample_stddev,
        dist.mean(),
        dist.variance(),
        dist.stddev(),
    ])

    self.assertAllClose(mean_, sample_mean_, rtol=rtol, atol=atol)
    self.assertAllClose(variance_, sample_variance_, rtol=rtol, atol=atol)
    self.assertAllClose(stddev_, sample_stddev_, rtol=rtol, atol=atol)

  def histogram(self, x, value_range=None, nbins=None, name=None):
    """Return histogram of values.

    Given the tensor `values`, this operation returns a rank 1 histogram
    counting the number of entries in `values` that fell into every bin. The
    bins are equal width and determined by the arguments `value_range` and
    `nbins`.

    Args:
      x: 1D numeric `Tensor` of items to count.
      value_range:  Shape [2] `Tensor`. `new_values <= value_range[0]` will be
        mapped to `hist[0]`, `values >= value_range[1]` will be mapped to
        `hist[-1]`. Must be same dtype as `x`.
      nbins:  Scalar `int32 Tensor`.  Number of histogram bins.
      name: Python `str` name prefixed to Ops created by this class.

    Returns:
      counts: 1D `Tensor` of counts, i.e.,
        `counts[i] = sum{ edges[i-1] <= values[j] < edges[i] : j }`.
      edges: 1D `Tensor` characterizing intervals used for counting.
    """
    with tf.name_scope(name or 'histogram'):
      x = tf.convert_to_tensor(value=x, name='x')
      if value_range is None:
        value_range = [
            tf.reduce_min(x), 1 + tf.reduce_max(x)
        ]
      value_range = tf.convert_to_tensor(value=value_range, name='value_range')
      lo = value_range[0]
      hi = value_range[1]
      if nbins is None:
        nbins = tf.cast(hi - lo, dtype=tf.int32)
      delta = (hi - lo) / tf.cast(
          nbins, dtype=dtype_util.base_dtype(value_range.dtype))
      edges = tf.range(
          start=lo, limit=hi, delta=delta, dtype=dtype_util.base_dtype(x.dtype))
      counts = tf.histogram_fixed_width(x, value_range=value_range, nbins=nbins)
      return counts, edges


class VectorDistributionTestHelpers(object):
  """VectorDistributionTestHelpers helps test vector-event distributions."""

  def run_test_sample_consistent_log_prob(
      self,
      sess_run_fn,
      dist,
      num_samples=int(1e5),
      radius=1.,
      center=0.,
      seed=None,
      rtol=1e-2,
      atol=0.):
    """Tests that sample/log_prob are mutually consistent.

    "Consistency" means that `sample` and `log_prob` correspond to the same
    distribution.

    The idea of this test is to compute the Monte-Carlo estimate of the volume
    enclosed by a hypersphere, i.e., the volume of an `n`-ball. While we could
    choose an arbitrary function to integrate, the hypersphere's volume is nice
    because it is intuitive, has an easy analytical expression, and works for
    `dimensions > 1`.

    Technical Details:

    Observe that:

    ```none
    int_{R**d} dx [x in Ball(radius=r, center=c)]
    = E_{p(X)}[ [X in Ball(r, c)] / p(X) ]
    = lim_{m->infty} m**-1 sum_j^m [x[j] in Ball(r, c)] / p(x[j]),
        where x[j] ~iid p(X)
    ```

    Thus, for fixed `m`, the above is approximately true when `sample` and
    `log_prob` are mutually consistent.

    Furthermore, the above calculation has the analytical result:
    `pi**(d/2) r**d / Gamma(1 + d/2)`.

    Note: this test only verifies a necessary condition for consistency--it does
    does not verify sufficiency hence does not prove `sample`, `log_prob` truly
    are consistent. For this reason we recommend testing several different
    hyperspheres (assuming the hypersphere is supported by the distribution).
    Furthermore, we gain additional trust in this test when also tested `sample`
    against the first, second moments
    (`run_test_sample_consistent_mean_covariance`); it is probably unlikely that
    a "best-effort" implementation of `log_prob` would incorrectly pass both
    tests and for different hyperspheres.

    For a discussion on the analytical result (second-line) see:
      https://en.wikipedia.org/wiki/Volume_of_an_n-ball.

    For a discussion of importance sampling (fourth-line) see:
      https://en.wikipedia.org/wiki/Importance_sampling.

    Args:
      sess_run_fn: Python `callable` taking `list`-like of `Tensor`s and
        returning a list of results after running one "step" of TensorFlow
        computation, typically set to `sess.run`.
      dist: Distribution instance or object which implements `sample`,
        `log_prob`, `event_shape_tensor` and `batch_shape_tensor`. The
        distribution must have non-zero probability of sampling every point
        enclosed by the hypersphere.
      num_samples: Python `int` scalar indicating the number of Monte-Carlo
        samples to draw from `dist`.
      radius: Python `float`-type indicating the radius of the `n`-ball which
        we're computing the volume.
      center: Python floating-type vector (or scalar) indicating the center of
        the `n`-ball which we're computing the volume. When scalar, the value is
        broadcast to all event dims.
      seed: Python `int` indicating the seed to use when sampling from `dist`.
        In general it is not recommended to use `None` during a test as this
        increases the likelihood of spurious test failure.
      rtol: Python `float`-type indicating the admissible relative error between
        actual- and approximate-volumes.
      atol: Python `float`-type indicating the admissible absolute error between
        actual- and approximate-volumes. In general this should be zero since
        a typical radius implies a non-zero volume.
    """

    def actual_hypersphere_volume(dims, radius):
      # https://en.wikipedia.org/wiki/Volume_of_an_n-ball
      # Using tf.lgamma because we'd have to otherwise use SciPy which is not
      # a required dependency of core.
      dims = tf.cast(dims, dtype=radius.dtype)
      return tf.exp((dims / 2.) * np.log(np.pi) -
                    tf.math.lgamma(1. + dims / 2.) + dims * tf.math.log(radius))

    def monte_carlo_hypersphere_volume(dist, num_samples, radius, center):
      # https://en.wikipedia.org/wiki/Importance_sampling
      x = dist.sample(num_samples, seed=test_seed_stream(hardcoded_seed=seed))
      x = tf.identity(x)  # Invalidate bijector cacheing.
      inverse_log_prob = tf.exp(-dist.log_prob(x))
      importance_weights = tf1.where(
          tf.norm(tensor=x - center, axis=-1) <= radius, inverse_log_prob,
          tf.zeros_like(inverse_log_prob))
      return tf.reduce_mean(importance_weights, axis=0)

    # Build graph.
    with tf.name_scope('run_test_sample_consistent_log_prob'):
      radius = tf.convert_to_tensor(radius, dist.dtype)
      center = tf.convert_to_tensor(center, dist.dtype)
      batch_shape = dist.batch_shape_tensor()
      actual_volume = actual_hypersphere_volume(
          dims=dist.event_shape_tensor()[0],
          radius=radius)
      sample_volume = monte_carlo_hypersphere_volume(
          dist,
          num_samples=num_samples,
          radius=radius,
          center=center)
      init_op = tf1.global_variables_initializer()

    # Execute graph.
    sess_run_fn(init_op)
    [batch_shape_, actual_volume_, sample_volume_] = sess_run_fn([
        batch_shape, actual_volume, sample_volume])

    # Check results.
    self.assertAllClose(np.tile(actual_volume_, reps=batch_shape_),
                        sample_volume_,
                        rtol=rtol, atol=atol)

  def run_test_sample_consistent_mean_covariance(
      self,
      sess_run_fn,
      dist,
      num_samples=int(1e5),
      seed=None,
      rtol=1e-2,
      atol=0.1,
      cov_rtol=None,
      cov_atol=None):
    """Tests that sample/mean/covariance are consistent with each other.

    "Consistency" means that `sample`, `mean`, `covariance`, etc all correspond
    to the same distribution.

    Args:
      sess_run_fn: Python `callable` taking `list`-like of `Tensor`s and
        returning a list of results after running one "step" of TensorFlow
        computation, typically set to `sess.run`.
      dist: Distribution instance or object which implements `sample`,
        `log_prob`, `event_shape_tensor` and `batch_shape_tensor`.
      num_samples: Python `int` scalar indicating the number of Monte-Carlo
        samples to draw from `dist`.
      seed: Python `int` indicating the seed to use when sampling from `dist`.
        In general it is not recommended to use `None` during a test as this
        increases the likelihood of spurious test failure.
      rtol: Python `float`-type indicating the admissible relative error between
        analytical and sample statistics.
      atol: Python `float`-type indicating the admissible absolute error between
        analytical and sample statistics.
      cov_rtol: Python `float`-type indicating the admissible relative error
        between analytical and sample covariance. Default: rtol.
      cov_atol: Python `float`-type indicating the admissible absolute error
        between analytical and sample covariance. Default: atol.
    """

    x = dist.sample(num_samples, seed=test_seed_stream(hardcoded_seed=seed))
    sample_mean = tf.reduce_mean(x, axis=0)
    sample_covariance = tf.reduce_mean(
        _vec_outer_square(x - sample_mean), axis=0)
    sample_variance = tf.linalg.diag_part(sample_covariance)
    sample_stddev = tf.sqrt(sample_variance)

    [
        sample_mean_,
        sample_covariance_,
        sample_variance_,
        sample_stddev_,
        mean_,
        covariance_,
        variance_,
        stddev_
    ] = sess_run_fn([
        sample_mean,
        sample_covariance,
        sample_variance,
        sample_stddev,
        dist.mean(),
        dist.covariance(),
        dist.variance(),
        dist.stddev(),
    ])

    self.assertAllClose(mean_, sample_mean_, rtol=rtol, atol=atol)
    self.assertAllClose(covariance_, sample_covariance_,
                        rtol=cov_rtol or rtol,
                        atol=cov_atol or atol)
    self.assertAllClose(variance_, sample_variance_, rtol=rtol, atol=atol)
    self.assertAllClose(stddev_, sample_stddev_, rtol=rtol, atol=atol)


def _vec_outer_square(x, name=None):
  """Computes the outer-product of a vector, i.e., x.T x."""
  with tf.name_scope(name or 'vec_osquare'):
    return x[..., :, tf.newaxis] * x[..., tf.newaxis, :]
