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

import contextlib
import functools
import math
import os
import random
import re
import sys
import unittest

from absl import flags
from absl import logging
from absl.testing import parameterized
import numpy as np

import six
import tensorflow.compat.v1 as tf1
import tensorflow.compat.v2 as tf
from tensorflow_probability.python.bijectors import bijector
from tensorflow_probability.python.internal import distribution_util
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import empirical_statistical_testing
from tensorflow_probability.python.internal import samplers
from tensorflow_probability.python.internal import test_combinations
from tensorflow_probability.python.util.deferred_tensor import DeferredTensor
from tensorflow_probability.python.util.deferred_tensor import TransformedVariable
from tensorflow_probability.python.util.seed_stream import SeedStream
from tensorflow.python.util import nest  # pylint: disable=g-direct-tensorflow-import
from absl.testing import absltest


__all__ = [
    'substrate_disable_stateful_random_test',
    'numpy_disable_gradient_test',
    'numpy_disable_variable_test',
    'jax_disable_variable_test',
    'jax_disable_test_missing_functionality',
    'disable_test_for_backend',
    'main',
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
TF_MODE = not (JAX_MODE or NUMPY_MODE)

flags.DEFINE_string('test_regex', '',
                    ('If set, only run test cases for which this regex '
                     'matches "<TestCase>.<test_method>".'),
                    allow_override=True)

# Flags for controlling test_teed behavior.
flags.DEFINE_bool('vary_seed', False,
                  ('Whether to vary the PRNG seed unpredictably.  '
                   'With --runs_per_test=N, produces N iid runs.'),
                  allow_override=True)

flags.DEFINE_string('fixed_seed', None,
                    ('PRNG seed to initialize every test with.  '
                     'Takes precedence over --vary_seed when both appear.'),
                    allow_override=True,
                    allow_override_cpp=False,
                    allow_hide_cpp=True)

flags.DEFINE_enum('analyze_calibration', 'none',
                  ['none', 'brief', 'full'],
                  ('If set, auto-fails assertAllMeansClose and prints '
                   'a report of how failure-prone the test is.'),
                  allow_override=True)

FLAGS = flags.FLAGS


_TEST_BASE_CLASSES = (parameterized.TestCase,)
if TF_MODE:
  _TEST_BASE_CLASSES = _TEST_BASE_CLASSES + (tf.test.TestCase,)


_DEFAULT_SEED = 87654321


class TestCase(*_TEST_BASE_CLASSES):
  """Class to provide TensorFlow Probability specific test features."""

  def setUp(self):
    if TF_MODE:
      super(TestCase, self).setUp()
    else:
      # Fix the numpy and math random seeds.
      np.random.seed(_DEFAULT_SEED)
      random.seed(_DEFAULT_SEED)

  def tearDown(self):
    if TF_MODE:
      super(TestCase, self).tearDown()

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

  @contextlib.contextmanager
  def cached_session(self):
    if TF_MODE:
      with super(TestCase, self).cached_session() as sess:
        yield sess
    else:
      with contextlib.nullcontext():
        yield

  @contextlib.contextmanager
  def session(self):
    if TF_MODE:
      with super(TestCase, self).session() as sess:
        yield sess
    else:
      with contextlib.nullcontext():
        yield

  def evaluate(self, x):
    if TF_MODE:
      return super(TestCase, self).evaluate(x)

    if JAX_MODE:
      import jax  # pylint: disable=g-import-not-at-top

    def _evaluate(x):
      if x is None:
        return x
      # TODO(b/223267515): Improve handling of JAX typed PRNG keys.
      if (
          JAX_MODE
          and hasattr(x, 'dtype')
          and jax.dtypes.issubdtype(x.dtype, jax.dtypes.prng_key)
      ):
        return x
      return np.array(x)
    return tf.nest.map_structure(_evaluate, x, expand_composites=True)

  def _GetNdArray(self, a):
    if TF_MODE:
      return super(TestCase, self)._GetNdArray(a)
    return np.array(a)

  def _evaluateTensors(self, a, b):
    if JAX_MODE:
      import jax  # pylint: disable=g-import-not-at-top
      # HACK: In assertions (like self.assertAllClose), convert typed PRNG keys
      # to raw arrays so they can be compared with our existing machinery.
      if hasattr(a, 'dtype') and jax.dtypes.issubdtype(
          a.dtype, jax.dtypes.prng_key
      ):
        a = jax.random.key_data(a)
      if hasattr(b, 'dtype') and jax.dtypes.issubdtype(
          b.dtype, jax.dtypes.prng_key
      ):
        b = jax.random.key_data(b)
    if tf.is_tensor(a) and tf.is_tensor(b):
      (a, b) = self.evaluate([a, b])
    elif tf.is_tensor(a) and not tf.is_tensor(b):
      a = self.evaluate(a)
    elif not tf.is_tensor(a) and tf.is_tensor(b):
      b = self.evaluate(b)
    return a, b

  def assertDTypeEqual(self, target, expected_dtype):
    """Assert ndarray data type is equal to expected.

    Args:
      target: The numpy `ndarray`, or anything that can be converted into a
        numpy `ndarray` (including Tensor).
      expected_dtype: Expected data type.
    """
    target = self._GetNdArray(target)
    if not isinstance(target, list):
      arrays = [target]
    for arr in arrays:
      self.assertEqual(arr.dtype, expected_dtype)

  # pylint: disable=g-doc-return-or-yield
  @contextlib.contextmanager
  def assertRaisesWithPredicateMatch(self, exception_type,
                                     expected_err_re_or_predicate):
    """Returns a context manager to enclose code expected to raise an exception.

    If the exception is an OpError, the op stack is also included in the message
    predicate search.

    Args:
      exception_type: The expected type of exception that should be raised.
      expected_err_re_or_predicate: If this is callable, it should be a function
        of one argument that inspects the passed-in exception and returns True
        (success) or False (please fail the test). Otherwise, the error message
        is expected to match this regular expression partially.

    Returns:
      A context manager to surround code that is expected to raise an
      exception.
    """
    if callable(expected_err_re_or_predicate):
      predicate = expected_err_re_or_predicate
    else:

      def predicate(e):
        err_str = e.message if isinstance(e, tf.errors.OpError) else str(e)
        op = e.op if isinstance(e, tf.errors.OpError) else None
        while op is not None:
          err_str += '\nCaused by: ' + op.name
          op = op._original_op  # pylint: disable=protected-access
        logging.info('Searching within error strings: "%s" within "%s"',
                     expected_err_re_or_predicate, err_str)
        return re.search(expected_err_re_or_predicate, err_str)

    try:
      yield
      self.fail(exception_type.__name__ + ' not raised')
    except Exception as e:  # pylint: disable=broad-except
      if not isinstance(e, exception_type) or not predicate(e):
        raise AssertionError('Exception of type %s: %s' %
                             (str(type(e)), str(e))) from e

  @contextlib.contextmanager
  def assertRaisesOpError(self, msg):
    if TF_MODE:
      with super(TestCase, self).assertRaisesOpError(msg):
        yield
    else:
      try:
        yield
        self.fail('No exception raised. Expected exception similar to '
                  'tf.errors.OpError with message: %s' % msg)
      except Exception as e:  # pylint: disable=broad-except
        err_str = str(e)
        if re.search(msg, err_str):
          return
        logging.error('Expected exception to match `%s`!', msg)
        raise

  def assertNear(self, f1, f2, err, msg=None):
    """Asserts that two floats are near each other.

    Checks that |f1 - f2| < err and asserts a test failure
    if not.

    Args:
      f1: A float value.
      f2: A float value.
      err: A float value.
      msg: An optional string message to append to the failure message.
    """
    # f1 == f2 is needed here as we might have: f1, f2 = inf, inf
    self.assertTrue(
        f1 == f2 or math.fabs(f1 - f2) <= err, '%f != %f +/- %f%s' %
        (f1, f2, err, ' (%s)' % msg if msg is not None else ''))

  def assertArrayNear(self, farray1, farray2, err, msg=None):
    """Asserts that two float arrays are near each other.

    Checks that for all elements of farray1 and farray2
    |f1 - f2| < err.  Asserts a test failure if not.

    Args:
      farray1: a list of float values.
      farray2: a list of float values.
      err: a float value.
      msg: Optional message to report on failure.
    """
    self.assertEqual(len(farray1), len(farray2), msg=msg)
    for f1, f2 in zip(farray1, farray2):
      self.assertNear(float(f1), float(f2), err, msg=msg)

  def assertNotAllEqual(self, a, b, msg=None):
    """Asserts that two numpy arrays or Tensors do not have the same values.

    Args:
      a: the expected numpy ndarray or anything can be converted to one.
      b: the actual numpy ndarray or anything can be converted to one.
      msg: Optional message to report on failure.
    """
    try:
      self.assertAllEqual(a, b)
    except AssertionError:
      return
    msg = msg or ''
    raise AssertionError('The two values are equal at all elements. %s' % msg)

  def assertNotAllClose(self, a, b, rtol=1e-6, atol=1e-6, msg=None):
    """Assert that two numpy arrays, or Tensors, do not have near values.

    Args:
      a: The expected numpy `ndarray`, or anything that can be converted into a
        numpy `ndarray` (including Tensor), or any arbitrarily nested of
        structure of these.
      b: The actual numpy `ndarray`, or anything that can be converted into a
        numpy `ndarray` (including Tensor), or any arbitrarily nested of
        structure of these.
      rtol: relative tolerance.
      atol: absolute tolerance.
      msg: Optional message to report on failure.

    Raises:
      AssertionError: If `a` and `b` are unexpectedly close at all elements.
    """
    try:
      self.assertAllClose(a, b, rtol=rtol, atol=atol, msg=msg)
    except AssertionError:
      return
    msg = msg or ''
    raise AssertionError('The two values are close at all elements. %s' % msg)

  def assertEqual(self, first, second, msg=None):
    if isinstance(first, list) and isinstance(second, tuple):
      first = tuple(first)
    if isinstance(first, tuple) and isinstance(second, list):
      second = tuple(second)
    return super(TestCase, self).assertEqual(first, second, msg)

  def assertAllCloseAccordingToType(self,
                                    a,
                                    b,
                                    rtol=1e-6,
                                    atol=1e-6,
                                    float_rtol=1e-6,
                                    float_atol=1e-6,
                                    half_rtol=1e-3,
                                    half_atol=1e-3,
                                    bfloat16_rtol=1e-2,
                                    bfloat16_atol=1e-2,
                                    msg=None):
    """Like assertAllClose, but also suitable for comparing fp16 arrays.

    In particular, the tolerance is reduced to 1e-3 if at least
    one of the arguments is of type float16.

    Args:
      a: the expected numpy ndarray or anything can be converted to one.
      b: the actual numpy ndarray or anything can be converted to one.
      rtol: relative tolerance.
      atol: absolute tolerance.
      float_rtol: relative tolerance for float32.
      float_atol: absolute tolerance for float32.
      half_rtol: relative tolerance for float16.
      half_atol: absolute tolerance for float16.
      bfloat16_rtol: relative tolerance for bfloat16.
      bfloat16_atol: absolute tolerance for bfloat16.
      msg: Optional message to report on failure.
    """
    (a, b) = self._evaluateTensors(a, b)
    a = self._GetNdArray(a)
    b = self._GetNdArray(b)
    # types with lower tol are put later to overwrite previous ones.
    if (a.dtype == np.float32 or b.dtype == np.float32 or
        a.dtype == np.complex64 or b.dtype == np.complex64):
      rtol = max(rtol, float_rtol)
      atol = max(atol, float_atol)
    if a.dtype == np.float16 or b.dtype == np.float16:
      rtol = max(rtol, half_rtol)
      atol = max(atol, half_atol)
    if not NUMPY_MODE:
      if a.dtype == tf.bfloat16 or b.dtype == tf.bfloat16:
        rtol = max(rtol, bfloat16_rtol)
        atol = max(atol, bfloat16_atol)

    self.assertAllClose(a, b, rtol=rtol, atol=atol, msg=msg)

  def assertAllEqual(self, a, b, msg=None):
    """Asserts that two numpy arrays or Tensors have the same values.

    Args:
      a: the expected numpy ndarray or anything can be converted to one.
      b: the actual numpy ndarray or anything can be converted to one.
      msg: Optional message to report on failure.
    """
    msg = msg if msg else ''
    (a, b) = self._evaluateTensors(a, b)
    a = self._GetNdArray(a)
    b = self._GetNdArray(b)
    # Arbitrary bounds so that we don't print giant tensors.
    if (b.ndim <= 3 or b.size < 500):
      self.assertEqual(
          a.shape, b.shape, 'Shape mismatch: expected %s, got %s.'
          ' Contents: %r. \n%s.' % (a.shape, b.shape, b, msg))
    else:
      self.assertEqual(
          a.shape, b.shape, 'Shape mismatch: expected %s, got %s.'
          ' %s' % (a.shape, b.shape, msg))

    same = (a == b)

    dtype_list = [np.float16, np.float32, np.float64]
    if not NUMPY_MODE:
      dtype_list += [tf.bfloat16]

    if a.dtype in dtype_list:
      same = np.logical_or(same, np.logical_and(np.isnan(a), np.isnan(b)))
    msgs = [msg]
    if not np.all(same):
      # Adds more details to np.testing.assert_array_equal.
      diff = np.logical_not(same)
      if a.ndim:
        x = a[np.where(diff)]
        y = b[np.where(diff)]
        msgs.append('not equal where = {}'.format(np.where(diff)))
      else:
        # np.where is broken for scalars
        x, y = a, b
      msgs.append('not equal lhs = %r' % x)
      msgs.append('not equal rhs = %r' % y)

      if (a.dtype.kind != b.dtype.kind and
          {a.dtype.kind, b.dtype.kind}.issubset({'U', 'S', 'O'})):
        a_list = []
        b_list = []
        # OK to flatten `a` and `b` because they are guaranteed to have the
        # same shape.
        for out_list, flat_arr in [(a_list, a.flat), (b_list, b.flat)]:
          for item in flat_arr:
            if isinstance(item, str):
              out_list.append(item.encode('utf-8'))
            else:
              out_list.append(item)
        a = np.array(a_list)
        b = np.array(b_list)

      np.testing.assert_array_equal(a, b, err_msg='\n'.join(msgs))

  def assertAllGreater(self, a, comparison_target):
    """Assert element values are all greater than a target value.

    Args:
      a: The numpy `ndarray`, or anything that can be converted into a numpy
        `ndarray` (including Tensor).
      comparison_target: The target value of comparison.
    """
    (a, comparison_target) = self._evaluateTensors(a, comparison_target)
    a = self._GetNdArray(a)
    self.assertGreater(np.min(a), comparison_target)

  def assertAllLess(self, a, comparison_target):
    """Assert element values are all less than a target value.

    Args:
      a: The numpy `ndarray`, or anything that can be converted into a numpy
        `ndarray` (including Tensor).
      comparison_target: The target value of comparison.
    """
    (a, comparison_target) = self._evaluateTensors(a, comparison_target)
    a = self._GetNdArray(a)
    self.assertLess(np.max(a), comparison_target)

  def assertAllGreaterEqual(self, a, comparison_target):
    """Assert element values are all greater than or equal to a target value.

    Args:
      a: The numpy `ndarray`, or anything that can be converted into a numpy
        `ndarray` (including Tensor).
      comparison_target: The target value of comparison.
    """
    (a, comparison_target) = self._evaluateTensors(a, comparison_target)
    a = self._GetNdArray(a)
    self.assertGreaterEqual(np.min(a), comparison_target)

  def assertAllLessEqual(self, a, comparison_target):
    """Assert element values are all less than or equal to a target value.

    Args:
      a: The numpy `ndarray`, or anything that can be converted into a numpy
        `ndarray` (including Tensor).
      comparison_target: The target value of comparison.
    """
    (a, comparison_target) = self._evaluateTensors(a, comparison_target)
    a = self._GetNdArray(a)
    self.assertLessEqual(np.max(a), comparison_target)

  def assertShapeEqual(self, input_a, input_b, msg=None):
    if TF_MODE:
      super(TestCase, self).assertShapeEqual(input_a, input_b, msg=msg)
    else:
      self.assertTupleEqual(input_a.shape, input_b.shape, msg=msg)

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

  def assertAllInRange(self,
                       target,
                       lower_bound,
                       upper_bound,
                       open_lower_bound=False,
                       open_upper_bound=False):
    """Assert that elements in a Tensor are all in a given range.

    Args:
      target: The numpy `ndarray`, or anything that can be converted into a
        numpy `ndarray` (including Tensor).
      lower_bound: lower bound of the range
      upper_bound: upper bound of the range
      open_lower_bound: (`bool`) whether the lower bound is open (i.e., > rather
        than the default >=)
      open_upper_bound: (`bool`) whether the upper bound is open (i.e., < rather
        than the default <=)

    Raises:
      AssertionError:
        if the value tensor does not have an ordered numeric type (float* or
          int*), or
        if there are nan values, or
        if any of the elements do not fall in the specified range.
    """
    target = self._GetNdArray(target)
    if not (np.issubdtype(target.dtype, np.floating) or
            np.issubdtype(target.dtype, np.integer)):
      raise AssertionError(
          'The value of %s does not have an ordered numeric type, instead it '
          'has type: %s' % (target, target.dtype))

    nan_subscripts = np.where(np.atleast_1d(np.isnan(target)))
    if np.size(nan_subscripts):
      raise AssertionError(
          '%d of the %d element(s) are NaN. '
          'Subscripts(s) and value(s) of the NaN element(s):\n' %
          (len(nan_subscripts[0]), np.size(target)) +
          '\n'.join(self._format_subscripts(nan_subscripts, target)))

    range_str = (('(' if open_lower_bound else '[') + str(lower_bound) + ', ' +
                 str(upper_bound) + (')' if open_upper_bound else ']'))

    violations = (
        np.less_equal(target, lower_bound) if open_lower_bound else np.less(
            target, lower_bound))
    violations = np.logical_or(
        violations,
        np.greater_equal(target, upper_bound)
        if open_upper_bound else np.greater(target, upper_bound))
    violation_subscripts = np.where(np.atleast_1d(violations))
    if np.size(violation_subscripts):
      raise AssertionError(
          '%d of the %d element(s) are outside the range %s. ' %
          (len(violation_subscripts[0]), np.size(target), range_str) +
          'Subscript(s) and value(s) of the offending elements:\n' +
          '\n'.join(self._format_subscripts(violation_subscripts, target)))

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

  def assertAllClose(self, a, b, rtol=1e-06, atol=1e-06, msg=None, path=None):
    path = [] if path is None else path
    path_str = ''
    msg = msg if msg else ''
    if isinstance(a, (list, tuple)):
      # Try to directly compare a, b as ndarrays; if not work, then traverse
      # through the sequence, which is more expensive.
      try:
        (a, b) = self._evaluateTensors(a, b)
        a_as_ndarray = self._GetNdArray(a)
        b_as_ndarray = self._GetNdArray(b)
        self.assertAllClose(
            a_as_ndarray,
            b_as_ndarray,
            rtol=rtol,
            atol=atol,
            msg='Mismatched value: a%s is different from b%s. %s' %
            (path_str, path_str, msg))
      except (ValueError, TypeError, NotImplementedError) as e:
        if len(a) != len(b):
          raise ValueError(
              'Mismatched length: a%s has %d items, but b%s has %d items. %s' %
              (path_str, len(a), path_str, len(b), msg)) from e
        for idx, (a_ele, b_ele) in enumerate(zip(a, b)):
          path.append(str(idx))
          self.assertAllClose(
              a_ele, b_ele, rtol=rtol, atol=atol, path=path, msg=msg)
          del path[-1]
    else:
      (a, b) = self._evaluateTensors(a, b)
      a = self._GetNdArray(a)
      b = self._GetNdArray(b)
      # When the array rank is small, print its contents. Numpy array printing
      # is implemented using inefficient recursion so prints can cause tests to
      # time out.
      if a.shape != b.shape and (b.ndim <= 3 or b.size < 500):
        shape_mismatch_msg = (
            'Shape mismatch: expected %s, got %s with contents '
            '%s.') % (a.shape, b.shape, b)
      else:
        shape_mismatch_msg = 'Shape mismatch: expected %s, got %s.' % (a.shape,
                                                                       b.shape)
      self.assertEqual(a.shape, b.shape, shape_mismatch_msg)

      msgs = [msg]
      a_dtype = a.dtype
      if not np.allclose(a, b, rtol=rtol, atol=atol):
        # Adds more details to np.testing.assert_allclose.
        #
        # NOTE: numpy.allclose (and numpy.testing.assert_allclose)
        # checks whether two arrays are element-wise equal within a
        # tolerance. The relative difference (rtol * abs(b)) and the
        # absolute difference atol are added together to compare against
        # the absolute difference between a and b.  Here, we want to
        # tell user which elements violate such conditions.
        cond = np.logical_or(
            np.abs(a - b) > atol + rtol * np.abs(b),
            np.isnan(a) != np.isnan(b))
        if a.ndim:
          x = a[np.where(cond)]
          y = b[np.where(cond)]
          msgs.append('not close where = {}'.format(np.where(cond)))
        else:
          # np.where is broken for scalars
          x, y = a, b
        msgs.append('not close lhs = {}'.format(x))
        msgs.append('not close rhs = {}'.format(y))
        msgs.append('not close dif = {}'.format(np.abs(x - y)))
        msgs.append('not close tol = {}'.format(atol + rtol * np.abs(y)))
        msgs.append('dtype = {}, shape = {}'.format(a_dtype, a.shape))
        np.testing.assert_allclose(
            a, b, rtol=rtol, atol=atol, err_msg='\n'.join(msgs), equal_nan=True)

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

  def assertAllMeansClose(
      self, to_reduce, expected, axis, atol=1e-6, rtol=1e-6, msg=None):
    """Assert means of `to_reduce` along `axis` as `expected`, with diagnostics.

    Operationally, this is equivalent to

    ```
    means = tf.reduce_mean(to_reduce, axis)
    assertAllClose(means, expected, atol, rtol, msg)
    ```

    except that by intercepting samples before the reduction is
    carried out, `assertAllMeansClose` can diagnose the statistical
    significance of failures.

    Specifically, `to_reduce` is assumed to be sampled IID along
    `axis`.  Based on this, it's possible to estimate the probability
    of `assertAllMeansClose` failing as the upstream PRNG seed is
    varied, and suggest parameter changes to control that probability.
    To assess a particular test statistically, run it with

    ```
    --test_arg=--vary_seed --test_arg=--analyze_calibration=brief
    ```

    or

    ```
    --test_arg=--vary_seed --test_arg=--analyze_calibration=full
    ```

    To avoid bias in the reported diagnostics, either value of
    `--analyze_calibration` force-fails the assertion; diagnostics are
    reported independently of whether the current sample's mean is
    close to `expected` or not.

    Caveats:

    - `--vary_seed` is important to prevent bias: if
      `--analyze_calibration` is not passed, `assertAllMeansClose`
      only fails if the mean of `to_reduce` is far from `expected`.  A
      seed that is brought to your attention by this happening is by
      construction unlucky, and diagnostics reported from it (e.g., by
      passing `--analyze_calibration` but not `--vary_seed`) will be
      overly pessimistic.

    - The report produced by `assertAllMeansClose` only assesses
      significance; i.e., assuming the test and the code under test
      are correct, how should the parameters of the test be set to
      control accidental failures.  Sometimes, a bug will manifest as
      absurd suggestions for making the test pass---it's up to the
      user to notice this happening.

    - The report makes assumptions it does not test:

      - that the elements of `to_reduce` actually are IID along `axis`;

      - that there are enough of them that the empirical distribution
        observed by one call to `assertAllMeansClose` is a good
        approximation to the true generating distribution; and

      - in the case of the Gaussian extrapolation, that there are
        enough samples that the distribution on means is approximately
        Gaussian.

    - The suggestions in the report are extrapolations based on a
      random sample.  They may vary across runs and are not guaranteed
      to be accurate.  In particular, if increasing the number of
      samples a test draws, it's reasonable to rerun the diagnostics,
      because they now have more information to work with.

    Args:
      to_reduce: Tensor of samples, presumed IID along `axis`.
        Other dimensions are taken to be batch dimensions.
      expected: Tensor of expected mean values.  Must broadcast
        with the reduction of `to_reduce` along `axis`.
      axis: Python `int` giving the reduction axis.
      atol: Tensor of absolute tolerances for the means.  Must
        broadcast with the reduction of `to_reduce` along `axis`.
      rtol: Tensor of relative tolerances for the means.  Must
        broadcast with the reduction of `to_reduce` along `axis`.
      msg: Optional string to insert into the failure message,
        if any.

    """
    mean = tf.reduce_mean(to_reduce, axis=axis)
    if FLAGS.analyze_calibration == 'none':
      msg = (msg or '') + '\nTo assess statistically, run with'
      msg += '\n  --test_arg=--vary_seed --test_arg=--analyze_calibration=brief'
      msg += '\nor'
      msg += '\n  --test_arg=--vary_seed --test_arg=--analyze_calibration=full'
      self.assertAllClose(mean, expected, atol=atol, rtol=rtol, msg=msg)
    else:
      to_reduce = self._GetNdArray(to_reduce)
      expected = self._GetNdArray(expected)
      if msg is None:
        msg = ''
      else:
        msg += '\n'
      if FLAGS.analyze_calibration == 'brief':
        msg += empirical_statistical_testing.brief_report(
            to_reduce, expected, axis, atol, rtol)
        msg += ('\nFor more information, run with '
                '--test_arg=--analyze_calibration=full.')
      else:
        msg += empirical_statistical_testing.full_report(
            to_reduce, expected, axis, atol, rtol)
      if not FLAGS.vary_seed and FLAGS.fixed_seed is None:
        msg += '\nWARNING: Above report may be biased as --vary_seed='
        msg += 'False and --fixed_seed is not set.  '
        msg += 'See docstring of `assertAllMeansClose`.'
      raise AssertionError(msg)

  def assertConvertVariablesToTensorsWorks(self, obj):
    """Checks that Variables are correctly converted to Tensors inside CTs."""
    self.assertIsInstance(obj, tf.__internal__.CompositeTensor)
    tensor_obj = obj._convert_variables_to_tensors()  # pylint: disable=protected-access
    self.assertIs(type(obj), type(tensor_obj))
    self.assertEmpty(tensor_obj.variables)
    self._check_tensors_equal_variables(obj, tensor_obj)

  def _check_tensors_equal_variables(self, obj, tensor_obj):
    """Checks that Variables in `obj` have equivalent Tensors in `tensor_obj."""
    if isinstance(obj, (tf.Variable, DeferredTensor)):
      self.assertAllClose(tf.convert_to_tensor(obj),
                          tf.convert_to_tensor(tensor_obj))
      if isinstance(obj, TransformedVariable):
        self.assertIsInstance(tensor_obj, DeferredTensor)
        self.assertNotIsInstance(tensor_obj, TransformedVariable)
      if isinstance(obj, DeferredTensor):
        if isinstance(obj._transform_fn, bijector.Bijector):  # pylint: disable=protected-access
          self._check_tensors_equal_variables(
              obj._transform_fn, tensor_obj._transform_fn)  # pylint: disable=protected-access
      else:
        self.assertIsInstance(tensor_obj, tf.Tensor)
    elif isinstance(obj, tf.__internal__.CompositeTensor):
      params = getattr(obj, 'parameters', {})
      tensor_params = getattr(tensor_obj, 'parameters', {})
      self.assertAllEqual(params.keys(), tensor_params.keys())
      self._check_tensors_equal_variables(params, tensor_params)
    elif tf.__internal__.nest.is_mapping(obj):
      for k, v in obj.items():
        self._check_tensors_equal_variables(v, tensor_obj[k])
    elif tf.nest.is_nested(obj):
      for x, y in zip(obj, tensor_obj):
        self._check_tensors_equal_variables(x, y)
    else:
      # We only check Tensor, CompositeTensor, and nested structure parameters.
      pass

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
    """Computes difference between autodiff and numerical gradient.

    Args:
      f: callable function whose gradient to compute.
      args: Python `list` of independent variables with respect to which to
      compute gradients.
      delta: floating point value to use for finite difference calculation.

    Returns:
      err: the maximum error between all components of the numeric and
      autodiff'ed gradients.
    """
    return _compute_max_gradient_error(f, args, delta, eval_fn=self.evaluate)

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


def _compute_max_gradient_error(f, xs, scale=1e-3, eval_fn=None):
  """Compute the max difference between autodiff and numerical jacobian."""
  xs = tf.nest.map_structure(tf.convert_to_tensor, xs)
  if JAX_MODE:
    import jax  # pylint: disable=g-import-not-at-top
    f_jac = jax.jacrev(f, argnums=range(len(xs)))
    theoretical_jacobian = f_jac(*xs)
  else:
    def _theoretical_jacobian(xs):
      # The reason we don't use tfp.math.value_and_jacobian is we don't know
      # which dimensions are batch dimensions.
      with tf.GradientTape() as tape:
        tape.watch(xs)
        ys = f(*xs)
      theoretical_jacobian = tape.jacobian(ys, xs)
      return theoretical_jacobian
    theoretical_jacobian = _theoretical_jacobian(xs)

  numerical_jacobian = [_compute_numerical_jacobian(f, xs, i, scale)
                        for i in range(len(xs))]
  theoretical_jacobian, numerical_jacobian = eval_fn([
      theoretical_jacobian, numerical_jacobian])
  return np.max(np.array(
      [np.max(np.abs(a - b))
       for (a, b) in zip(theoretical_jacobian, numerical_jacobian)]))


def _compute_numerical_jacobian(f, xs, i, scale=1e-3):
  """Compute the numerical jacobian of `f`."""
  dtype_i = xs[i].dtype
  shape_i = xs[i].shape
  size_i = np.prod(shape_i, dtype=np.int32)
  def grad_i(d):
    return (f(*(xs[:i] + [xs[i] + d * scale] + xs[i+1:]))
            - f(*(xs[:i] + [xs[i] - d * scale] + xs[i+1:]))) / (2. * scale)

  if JAX_MODE:
    import jax  # pylint: disable=g-import-not-at-top

    ret = jax.vmap(grad_i)(
        np.eye(size_i, dtype=dtype_i).reshape((size_i,) + shape_i))
  else:
    @tf.function
    def _numerical_jacobians():
      """Computes numerical jacobian."""
      # The reason we don't use `vectorized_map` is the underlying function
      # might not be vmap-able.
      return tf.map_fn(
          grad_i,
          tf.reshape(tf.eye(size_i, dtype=dtype_i), (size_i,) + shape_i),
          parallel_iterations=3)
    ret = _numerical_jacobians()

  ret = distribution_util.rotate_transpose(ret, -1)
  return tf.reshape(ret, ret.shape[:-1] + shape_i)


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
    from tensorflow.python.eager import context  # pylint: disable=g-direct-tensorflow-import,g-import-not-at-top
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
  if not TF_MODE:
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
    if not NUMPY_MODE:
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
    if not TF_MODE:
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
    sampler_type: 'stateful', 'stateless' or 'integer'. 'stateless'
      returns a seed suitable to pass to stateful PRNGs. 'integer' is returns a
      seed suitable for PRNGs which expect single-integer seeds (e.g. numpy).

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
    if JAX_MODE and not isinstance(answer, int):
      # Workaround for test_seed(hardcoded_seed=test_seed()), which can happen
      # e.g. with the run_test_sample_consistent_log_prob methods above.
      answer = samplers.get_integer_seed(answer)
  else:
    answer = 17
  if sampler_type == 'stateless' or JAX_MODE:
    answer = answer % (2**32 - 1)
    if JAX_MODE:
      import jax  # pylint: disable=g-import-not-at-top
      answer = jax.random.key(answer)
    else:
      answer = tf.constant([0, answer], dtype=tf.uint32)
      answer = tf.bitcast(answer, tf.int32)
  # TODO(b/68017812): Remove this clause once eager correctly supports seeding.
  elif tf.executing_eagerly() and set_eager_seed:
    tf.random.set_seed(answer)
  if sampler_type == 'integer':
    answer = samplers.get_integer_seed(answer)
  return answer


def clone_seed(seed):
  """Clone a seed: this is useful for JAX's experimental key reuse checking."""
  if JAX_MODE:
    import jax  # pylint: disable=g-import-not-at-top
    if hasattr(jax.random, 'clone'):
      # jax v0.4.26 or later
      return jax.random.clone(seed)
    else:
      # older jax versions
      return jax.random.wrap_key_data(
          jax.random.key_data(seed), impl=jax.random.key_impl(seed)
      )
  return seed


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
  raw_seed = test_seed(hardcoded_seed=hardcoded_seed, sampler_type='integer')
  return np.random.RandomState(seed=raw_seed)


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
      importance_weights = tf.where(
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


class _TestLoader(absltest.TestLoader):
  """A custom TestLoader that allows for Regex filtering test cases."""

  def getTestCaseNames(self, testCaseClass):  # pylint:disable=invalid-name
    names = super().getTestCaseNames(testCaseClass)
    if FLAGS.test_regex:
      pattern = re.compile(FLAGS.test_regex)
      names = [
          name for name in names
          if pattern.search(f'{testCaseClass.__name__}.{name}')
      ]
    return names


def main(jax_mode=JAX_MODE, jax_enable_x64=True):
  """Test main function that injects a custom loader."""
  if jax_mode and jax_enable_x64:
    from jax import config  # pylint: disable=g-import-not-at-top
    config.update('jax_enable_x64', True)

  # This logic is borrowed from TensorFlow.
  run_benchmarks = any(
      arg.startswith('--benchmarks=') or arg.startswith('-benchmarks=')
      for arg in sys.argv)

  if run_benchmarks:
    # TensorFlow will use its benchmarks runner in this case, which is separate
    # from the regular unittests and wouldn't be able to use the loaders anyway
    # (and it already supports regexes).
    tf.test.main()
  else:
    if TF_MODE:
      from tensorflow.python.framework import test_util  # pylint: disable=g-direct-tensorflow-import,g-import-not-at-top
      test_util.InstallStackTraceHandler()
    absltest.main(testLoader=_TestLoader())
